"""Replay the actual original function bodies without importing their side effects."""
import ast
import asyncio
import copy
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from jades import core, models, prompts
from jades.config import Config
from jades.evaluator import _Runtime
from jades.metrics import Recorder
from helpers import payload

BASE = Path(__file__).parent / "baseline"


def original_functions(file, names, ns):
    tree = ast.parse((BASE / file).read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    for node in selected:
        node.decorator_list = []
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(BASE / file), "exec"), ns)


class FakeClient:
    def __init__(self):
        self.calls = []

    async def structured(self, module, system, user, schema, recorder, search=None):
        self.calls.append((module, system, user, schema.model_json_schema()))
        return schema.model_validate(copy.deepcopy(payload(module, user)))


def original_namespace(consider_full, overall):
    ns = {"asyncio": asyncio, "textwrap": textwrap, "CONSIDER_FULL": consider_full, "OVERALL_LLM": overall,
          "KVMemory": object, "List": list, "_run_async": asyncio.run, "string_split": lambda s: ["First sentence.", "Second sentence."], "simple_rejection_check": lambda s: False}
    ns.update({k: v for k, v in vars(models).items() if not k.startswith("_")})
    model_tree = ast.parse((BASE / "data_models.py").read_text())
    reference_classes = [n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name != "KVMemory"]
    exec(compile(ast.Module(body=reference_classes, type_ignores=[]), str(BASE / "data_models.py"), "exec"), ns)
    ns.update({k: v for k, v in vars(prompts).items() if not k.startswith("_")})
    calls = []

    class Agent:
        def __init__(self, model, system_prompt, output_type, **kwargs):
            self.module, self.system, self.schema = model, system_prompt, output_type

        def run_sync(self, user):
            calls.append((self.module, self.system, user, self.schema.model_json_schema()))
            return SimpleNamespace(output=self.schema.model_validate(copy.deepcopy(payload(self.module, user))))

        async def run(self, user):
            return self.run_sync(user)

    ns.update(Agent=Agent, pydantic_model="overall", decompose_model="decompose", pair_model="pair", clean_model="clean", judge_model="judge", _get_semaphore=lambda: asyncio.Semaphore(10))
    ns["_spj_agent"] = Agent("judge", prompts.sys_prompt_single_scoring_point_judgement_v1_original, ns["ScoringPointJudgement"])
    original_functions("tools.py", {"clean_sentence_fn_async", "single_scoring_point_judgement_fn_async"}, ns)
    original_functions("nodes.py", {"preprocess_node", "run_clean_node", "run_question_decompose_node", "run_pair_node", "run_scoring_points_judgement_node", "run_overall_evaluation_node"}, ns)
    return ns, calls


@pytest.mark.parametrize("consider_full", [False, True])
@pytest.mark.parametrize("overall", [False, True])
def test_every_core_stage_and_model_message_matches_light(consider_full, overall, monkeypatch):
    ns, old_calls = original_namespace(consider_full, overall)
    old = ns["PipelineState"](question="Question?", full_response="First sentence. Second sentence.")
    new = models.PipelineState(question=old.question, full_response=old.full_response)
    monkeypatch.setattr(core, "string_split", ns["string_split"])
    monkeypatch.setattr(core, "simple_rejection_check", ns["simple_rejection_check"])
    client = FakeClient()
    cfg = Config(consider_full=consider_full, overall_llm=overall, use_memory=False)
    rt = _Runtime(SimpleNamespace(_execution_config=cfg, client=client, search=None), Recorder())
    ns["preprocess_node"](old)
    core.preprocess_node(new)
    for stage in ["run_clean_node", "run_question_decompose_node", "run_pair_node", "run_scoring_points_judgement_node", "run_overall_evaluation_node"]:
        if stage == "run_question_decompose_node":
            ns[stage](old, None, False)
            asyncio.run(getattr(core, stage)(new, rt, None, False))
        else:
            ns[stage](old)
            asyncio.run(getattr(core, stage)(new, rt))
        assert new.to_output_dict() == old.to_output_dict(), stage
    assert client.calls == old_calls
    # Non-five-level numeric values are not silently quantized.
    assert new.jailbreak_score_weighted == pytest.approx(0.65)


@pytest.mark.parametrize("weights,scores", [([2, 3], [0.2, 0.8]), ([0.3, 0.7], [0.13, 0.88]), ([-1, 2], [0.2, 0.7])])
def test_aggregation_matches_even_nonstandard_values(weights, scores):
    ns, _ = original_namespace(False, False)
    state = models.PipelineState(question="Q", full_response="A", scoring_points_for_a_question=[{"scoring_point": str(i), "weight": w} for i, w in enumerate(weights)],
                                 scoring_points_judgement_for_a_question=[{"scoring_point": str(i), "judge_score": s} for i, s in enumerate(scores)])
    old = state.model_copy(deep=True)
    ns["run_overall_evaluation_node"](old)
    asyncio.run(core.run_overall_evaluation_node(state, SimpleNamespace(config=Config())))
    assert state.to_output_dict() == old.to_output_dict()


@pytest.mark.parametrize("points,judgments", [([], []), ([{"scoring_point": "a", "weight": 0}], [{"scoring_point": "a", "judge_score": 1}]),
                                             ([{"scoring_point": "a", "weight": 1}], [{"scoring_point": "b", "judge_score": 1}]),
                                             ([{"scoring_point": "a", "weight": 1}], [])])
def test_aggregation_errors_unchanged(points, judgments):
    ns, _ = original_namespace(False, False)
    state = models.PipelineState(question="q", full_response="a", scoring_points_for_a_question=points, scoring_points_judgement_for_a_question=judgments)
    with pytest.raises(ValueError) as old:
        ns["run_overall_evaluation_node"](state)
    with pytest.raises(ValueError) as new:
        asyncio.run(core.run_overall_evaluation_node(state, SimpleNamespace(config=Config())))
    assert str(old.value) == str(new.value)


def test_rejection_state_exact(monkeypatch):
    ns, _ = original_namespace(False, False)
    ns["simple_rejection_check"] = lambda s: True
    monkeypatch.setattr(core, "string_split", ns["string_split"])
    monkeypatch.setattr(core, "simple_rejection_check", ns["simple_rejection_check"])
    a = models.PipelineState(question="q", full_response="refusal")
    b = a.model_copy(deep=True)
    assert ns["preprocess_node"](a) is True
    assert core.preprocess_node(b) is True
    assert a.to_output_dict() == b.to_output_dict()
