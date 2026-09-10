import asyncio
import json
from pathlib import Path

import httpx
import pytest

from jades import AsyncEvaluator, Evaluator, Config, ModelConfig, EvaluationError
from jades import core
from jades.batch import run_batch, BatchPaths
from jades.config import MODULES
from helpers import completion, payload


def config(tmp_path, **kwargs):
    return Config(llm=ModelConfig(base_url="https://mock.test/v1", max_retries=0), modules={m: {"model": m} for m in MODULES}, memory_path=str(tmp_path / "cache.sqlite3"), **kwargs)


@pytest.fixture
def preprocess(monkeypatch):
    monkeypatch.setattr(core, "string_split", lambda s: ["First sentence.", "Second sentence."])
    monkeypatch.setattr(core, "simple_rejection_check", lambda s: False)


def model_handler(calls):
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        return httpx.Response(200, json=completion(payload(body["model"], user)))
    return handler


def test_sync_multiple_samples_cache_and_overall_copy(tmp_path, preprocess):
    calls = []
    with Evaluator(config(tmp_path), {"HF_TOKEN": "token"}, transport=httpx.MockTransport(model_handler(calls))) as evaluator:
        first = evaluator.evaluate("Question?", "First sentence. Second sentence.")
        second = evaluator.evaluate("Question?", "First sentence. Second sentence.")
    assert first.score == pytest.approx(.65)
    assert first.state.jailbreak_score_llm == first.score
    assert first.metrics.tokens.total_tokens == 6 * 15
    assert second.metrics.tokens.total_tokens == 5 * 15
    assert second.metrics.modules["decompose"].cache_hits == 1
    assert len(calls) == 11


@pytest.mark.parametrize("search_fails", [False, True])
@pytest.mark.parametrize("verdict", ["true", "false"])
def test_full_fact_check_evidence_reaches_judge(tmp_path, preprocess, search_fails, verdict):
    calls, searches = [], []
    def search_handler(request):
        body = json.loads(request.content); searches.append(body)
        if search_fails:
            return httpx.Response(503)
        return httpx.Response(200, json={"results": [{"title": "Paris", "url": "https://en.wikipedia.org/wiki/Paris", "content": "Paris is in France."}]})
    def llm_handler(request):
        body = json.loads(request.content)
        calls.append(body)
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        data = payload(body["model"], user)
        if body["model"] == "fact_check":
            data["is_fact_correct"] = verdict
        return httpx.Response(200, json=completion(data))
    async def run():
        async with AsyncEvaluator(config(tmp_path, fact_check=True), {"HF_TOKEN": "token", "TAVILY_API_KEY": "search-secret"},
                                  transport=httpx.MockTransport(llm_handler), search_transport=httpx.MockTransport(search_handler)) as evaluator:
            result = await evaluator.aevaluate("Question?", "response")
        assert len(result.all_fact_check_results) == 2
        assert result.all_fact_check_results[0]["sentence_fact_check_results"][0]["is_fact_correct"] == ("unknown" if search_fails else verdict)
        judge = next(c for c in calls if c["model"] == "judge")
        assert "is_fact_correct" in judge["messages"][1]["content"]
        assert result.metrics.modules["fact_check"].search_count == 2
        assert searches[0]["include_domains"] == ["wikipedia.org"]
        assert searches[0]["max_results"] == 1
        assert "search-secret" not in result.model_dump_json()
    asyncio.run(run())


def test_timeout_cancels_inflight_and_retains_usage(tmp_path, preprocess):
    active = 0
    async def slow(request):
        nonlocal active
        active += 1
        try:
            await asyncio.sleep(1)
        finally:
            active -= 1
        return httpx.Response(200, json=completion(payload("clean")))
    async def run():
        async with AsyncEvaluator(config(tmp_path, sample_timeout=.05), {"HF_TOKEN": "token"}, transport=httpx.MockTransport(slow)) as evaluator:
            with pytest.raises(EvaluationError) as error:
                await evaluator.aevaluate("Question?", "response")
            metrics = error.value.metrics
            assert active == 0
            assert metrics.tokens.unknown_usage_requests == 2
            assert metrics.tokens.total_tokens is None
            assert all(r["status"] == "cancelled" for r in metrics.requests)
    asyncio.run(run())


def test_batch_resume_legacy_and_config_change(tmp_path, preprocess):
    inp, out = tmp_path / "input.json", tmp_path / "output.json"
    inp.write_text(json.dumps({"parameters": {"method": "test", "model": "target"}, "jailbreaks": [{"index": 5, "goal": "Question?", "response": "response"}]}))
    calls = []
    async def run():
        async with AsyncEvaluator(config(tmp_path), {"HF_TOKEN": "token"}, transport=httpx.MockTransport(model_handler(calls))) as evaluator:
            initial = await run_batch(evaluator, inp, out, legacy=True)
            assert initial["success_count"] == 1
            assert initial["current_run"]["tokens"]["total_tokens"] == 90
            resumed = await run_batch(evaluator, inp, out, resume=True, legacy=True)
            assert resumed["skipped_count"] == 1
            assert resumed["current_run"]["tokens"]["total_tokens"] == 0
            assert resumed["cumulative"]["tokens"]["total_tokens"] == 90
            data = json.loads(out.read_text())
            result = data["jailbreak_qa_artifacts"][0]
            assert result["index"] == 5
            assert "metrics" not in result["jailbreak_qa_result"]
            assert result["jailbreak_qa_result"]["is_simpe_rejection"] is False
            evaluator.config = evaluator.config.model_copy(update={"consider_full": True})
            with pytest.raises(ValueError, match="Cannot resume"):
                await run_batch(evaluator, inp, out, resume=True)
    asyncio.run(run())


def test_missing_fields_do_not_stop_batch(tmp_path, preprocess):
    inp = tmp_path / "input.jsonl"
    inp.write_text('{"question": "Question?"}\n{"question": "Question?", "response": "response"}\n')
    async def run():
        async with AsyncEvaluator(config(tmp_path), {"HF_TOKEN": "token"}, transport=httpx.MockTransport(model_handler([]))) as evaluator:
            summary = await run_batch(evaluator, inp, tmp_path / "out.json")
            assert summary["failure_count"] == 1
            assert summary["success_count"] == 1
    asyncio.run(run())


def test_cancel_mid_sample_then_resume_skips_completed_sample(tmp_path, preprocess):
    """Verify existing sample-level recovery without changing the batch implementation."""
    inp, out = tmp_path / "input.json", tmp_path / "output.json"
    inp.write_text(json.dumps([{"question": "Q1", "response": "response"}, {"question": "Q2", "response": "response"}]))
    async def run():
        entered = asyncio.Event()
        never = asyncio.Event()
        block = True
        calls = []
        async def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            user = next(m["content"] for m in body["messages"] if m["role"] == "user")
            if block and body["model"] == "clean" and "Q2" in user:
                entered.set()
                await never.wait()
            return httpx.Response(200, json=completion(payload(body["model"], user)))
        async with AsyncEvaluator(config(tmp_path, use_memory=False), {"HF_TOKEN": "token"}, transport=httpx.MockTransport(handler)) as evaluator:
            task = asyncio.create_task(run_batch(evaluator, inp, out))
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            before = json.loads(BatchPaths.for_output(out).checkpoint.read_text())
            assert before["items"]["0"]["status"] == "ok"
            assert "1" not in before["items"]
            assert before["runs"][-1]["status"] == "interrupted"
            first_result = before["items"]["0"]
            call_count = len(calls)
            block = False
            resumed = await run_batch(evaluator, inp, out, resume=True)
        after = json.loads(BatchPaths.for_output(out).checkpoint.read_text())
        assert after["items"]["0"] == first_result
        assert after["items"]["1"]["status"] == "ok"
        assert len(calls) - call_count == 6  # All six operations of sample 2 run again.
        assert resumed["success_count"] == 1 and resumed["skipped_count"] == 1
        assert resumed["current_run"]["tokens"]["total_tokens"] == 90
        assert resumed["cumulative"]["tokens"]["known_total_tokens"] == 180
        assert resumed["cumulative"]["tokens"]["unknown_usage_requests"] >= 1
    asyncio.run(run())
