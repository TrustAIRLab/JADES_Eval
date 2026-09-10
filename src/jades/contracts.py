"""Fingerprint the actual system prompts and message-building functions."""
import ast
import inspect
import textwrap
from functools import lru_cache
from copy import deepcopy

from . import core, messages, prompts, models
from .config import fingerprint

MESSAGE_BUILDERS = ("clean", "judge", "fact_decompose", "fact_clarify", "fact_check", "fact_judge")
TEMPLATE_NODES = ("run_question_decompose_node", "run_pair_node", "run_overall_evaluation_node")


@lru_cache(maxsize=128)
def _cached_source_contract(function, code):
    source = textwrap.dedent(inspect.getsource(function))
    return ast.dump(ast.parse(source), include_attributes=False)


def _source_contract(function):
    return _cached_source_contract(function, function.__code__)


_schema_contracts = {}


def _schema_contract(schema):
    cached = _schema_contracts.get(schema)
    if cached is None or cached[0] is not schema.__pydantic_core_schema__ or cached[1] != schema.model_config:
        cached = (schema.__pydantic_core_schema__, deepcopy(schema.model_config), fingerprint(schema.model_json_schema()))
        _schema_contracts[schema] = cached
    return cached[2]


def legacy_prompt_fingerprint():
    # v0.1.0 omitted user-message templates; use only for explicit legacy imports.
    return fingerprint({k: v for k, v in vars(prompts).items() if isinstance(v, str) and not k.startswith("_")})


def prompt_fingerprint():
    # Imported at call time, after evaluator initialization, to avoid an import cycle.
    from .evaluator import FactCheckResult
    schemas = [models.CleanSentence, models.ScoringPoints, models.AllScoringPointsWithSentences,
               models.ScoringPointJudgement, models.JailbreakEvaluationResultLLM,
               models.DecomposedUnitFacts, models.ClearUnitFacts, FactCheckResult]
    return fingerprint({
        "contract_version": 2,
        "system_prompts": {k: v for k, v in vars(prompts).items() if isinstance(v, str) and not k.startswith("_")},
        "node_prompt_bindings": {k: v for k, v in vars(core).items() if isinstance(v, str) and not k.startswith("_")},
        "message_builders": {name: _source_contract(getattr(messages, name)) for name in MESSAGE_BUILDERS},
        "template_nodes": {name: _source_contract(getattr(core, name)) for name in TEMPLATE_NODES},
        "output_schemas": {schema.__name__: _schema_contract(schema) for schema in schemas},
    })


def decomposition_fingerprint():
    return fingerprint({"system": core.sys_prompt_question_decompose_v1_original,
                        "template": _source_contract(core.run_question_decompose_node),
                        "schema": _schema_contract(models.ScoringPoints)})
