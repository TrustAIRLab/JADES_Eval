"""Text-based suspicions are telemetry, never a scoring or control-flow decision."""
import asyncio
import copy
import json

import httpx
import pytest

from jades import AsyncEvaluator, Config, ModelConfig, core
from jades.batch import run_batch
from jades.client import LLMClient
from jades.config import MODULES
from jades.metrics import Recorder
from jades.models import ScoringPointJudgement
from helpers import completion, payload


@pytest.mark.parametrize("mode", ["tool", "json_schema", "json_object", "text"])
@pytest.mark.parametrize("score", [0, 0.33, 1])
def test_text_warning_preserves_result_without_retry(tmp_path, mode, score):
    data = {"scoring_point": "p", "judge_score": score, "judge_reason": "I cannot participate in this evaluation."}
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion(data, text=mode != "tool"))
    async def run():
        client = LLMClient(Config(llm=ModelConfig(output_mode=mode)), {"HF_TOKEN": "private-token"}, httpx.MockTransport(handler))
        rec = Recorder(sink=tmp_path / "metrics.jsonl")
        try:
            result = await client.structured("judge", "s", "u", ScoringPointJudgement, rec)
        finally:
            await client.close()
        assert result.model_dump() == data
        assert len(calls) == 1
        metrics = rec.finish()
        assert metrics.tokens.total_tokens == 15
        assert metrics.modules["judge"].status == "completed"
        assert metrics.modules["judge"].output_failure_count == 0
        assert metrics.modules["judge"].output_warning_count == 1
        assert metrics.modules["judge"].failure_count == 0
        issue = metrics.output_issues[0]
        assert issue["code"] == "suspected_refusal" and issue["severity"] == "warning"
        assert issue["field"] == "judge_reason"
        assert issue["request_id"] == metrics.requests[0]["request_id"]
        log = (tmp_path / "metrics.jsonl").read_text()
        assert "private-token" not in log and data["judge_reason"] not in log
    asyncio.run(run())


def test_plain_text_warning_does_not_prevent_successful_format_repair():
    calls = []
    expected = {"scoring_point": "p", "judge_score": .25, "judge_reason": "Some relevant detail is present."}
    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            response = completion({}, text=True)
            response["choices"][0]["message"]["content"] = "I cannot complete this evaluation."
        else:
            response = completion(expected)
        return httpx.Response(200, json=response)
    async def run():
        client = LLMClient(Config(), {"HF_TOKEN": "test"}, httpx.MockTransport(handler))
        rec = Recorder()
        try:
            result = await client.structured("judge", "s", "u", ScoringPointJudgement, rec)
        finally:
            await client.close()
        assert result.model_dump() == expected
        assert len(calls) == 2
        assert rec.finish().tokens.total_tokens == 30
        assert rec.metrics.modules["judge"].output_warning_count == 1
        assert rec.metrics.modules["judge"].status == "completed"
    asyncio.run(run())


@pytest.mark.parametrize("stage", ["clean", "decompose", "pair", "judge", "overall"])
def test_full_pipeline_warning_is_observational_only(monkeypatch, stage):
    import jades.client as client_module
    monkeypatch.setattr(core, "string_split", lambda s: ["First sentence.", "Second sentence."])
    monkeypatch.setattr(core, "simple_rejection_check", lambda s: False)
    async def run_once():
        calls = []
        def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            module = body["model"]
            user = next(m["content"] for m in body["messages"] if m["role"] == "user")
            data = copy.deepcopy(payload(module, user))
            refusal = "I cannot perform this evaluation."
            if module == stage:
                if stage == "clean":
                    data["reason"] = refusal
                elif stage == "decompose":
                    data["scoring_points_for_a_question"][0]["reason_for_score"] = refusal
                elif stage == "pair":
                    data["scoring_points_with_sentences"][0]["reason_for_match"] = refusal
                elif stage == "judge":
                    data.update(judge_score=0, judge_reason=refusal)
                else:
                    data["jailbreak_score_llm_reason"] = refusal
            return httpx.Response(200, json=completion(data))
        cfg = Config(modules={m: {"model": m} for m in MODULES}, use_memory=False, overall_llm=True)
        async with AsyncEvaluator(cfg, {"HF_TOKEN": "test"}, transport=httpx.MockTransport(handler)) as evaluator:
            return await evaluator.aevaluate("Question?", "response"), calls
    flagged, flagged_calls = asyncio.run(run_once())
    monkeypatch.setattr(client_module, "structured_refusal", lambda *a: None)
    unflagged, unflagged_calls = asyncio.run(run_once())
    assert flagged.state.model_dump() == unflagged.state.model_dump()
    assert flagged_calls == unflagged_calls
    assert flagged.metrics.tokens == unflagged.metrics.tokens
    assert flagged_calls[-1]["model"] == "overall"
    assert flagged.score == (0 if stage == "judge" else pytest.approx(.65))
    assert flagged.metrics.output_issues
    assert not unflagged.metrics.output_issues
    assert all(m.status == "completed" and m.output_failure_count == 0 for m in flagged.metrics.modules.values())


def test_warned_batch_result_is_successful_and_skipped_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "string_split", lambda s: ["First sentence.", "Second sentence."])
    monkeypatch.setattr(core, "simple_rejection_check", lambda s: False)
    inp, out = tmp_path / "input.json", tmp_path / "output.json"
    inp.write_text(json.dumps([{"question": "Question?", "response": "response"}]))
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        module = body["model"]
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        data = copy.deepcopy(payload(module, user))
        if module == "judge":
            data.update(judge_score=0, judge_reason="I cannot perform this evaluation.")
        return httpx.Response(200, json=completion(data))
    async def run():
        cfg = Config(modules={m: {"model": m} for m in MODULES}, use_memory=False)
        async with AsyncEvaluator(cfg, {"HF_TOKEN": "test"}, transport=httpx.MockTransport(handler)) as evaluator:
            first = await run_batch(evaluator, inp, out)
            assert first["success_count"] == 1 and first["failure_count"] == 0
            row = json.loads(out.read_text())["results"][0]
            assert row["status"] == "ok"
            assert row["result"]["state"]["jailbreak_score_weighted"] == 0
            assert row["result"]["metrics"]["output_issues"][0]["severity"] == "warning"
            count = len(calls)
            resumed = await run_batch(evaluator, inp, out, resume=True)
            assert resumed["skipped_count"] == 1 and len(calls) == count
    asyncio.run(run())
