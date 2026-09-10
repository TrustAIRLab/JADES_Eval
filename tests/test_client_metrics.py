import asyncio
import json
import time

import httpx
import pytest

from jades.client import LLMClient, ModelOutputError, LLMCallError
from jades.config import Config, ModelConfig
from jades.metrics import Recorder, token_summary, summarize_requests
from jades.models import ScoringPointJudgement
from helpers import completion

VALID = {"scoring_point": "p", "judge_score": 0.25, "judge_reason": "test"}


def configuration(**llm):
    return Config(llm=ModelConfig(base_url="https://mock.test/v1", model="judge", **llm), max_concurrency=2)


def test_invalid_output_repair_usage_and_secret_redaction(tmp_path):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=completion({"invalid": True} if len(calls) == 1 else VALID))
    async def run():
        client = LLMClient(configuration(), {"HF_TOKEN": "DO_NOT_EXPOSE"}, httpx.MockTransport(handler))
        rec = Recorder(sink=tmp_path / "metrics.jsonl")
        try:
            result = await client.structured("judge", "system", "private sample text", ScoringPointJudgement, rec)
            assert result.judge_score == .25
        finally:
            await client.close()
        metrics = rec.finish()
        assert metrics.tokens.total_tokens == 30
        assert metrics.tokens.cached_input_tokens == 4
        assert metrics.tokens.reasoning_tokens == 6
        assert [r["purpose"] for r in metrics.requests] == ["evaluation", "output_repair"]
        assert calls[0]["messages"] == [{"role": "system", "content": "system"}, {"role": "user", "content": "private sample text"}]
        text = (tmp_path / "metrics.jsonl").read_text()
        assert "DO_NOT_EXPOSE" not in text
        assert "private sample text" not in text
    asyncio.run(run())


def test_429_retry_records_every_attempt_and_unknown_usage():
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(429, json={"error": {"message": "secret-body", "type": "rate_limit_error"}})
        return httpx.Response(200, json=completion(VALID))
    async def run():
        client = LLMClient(configuration(max_retries=1), {"HF_TOKEN": "token"}, httpx.MockTransport(handler))
        rec = Recorder()
        try:
            await client.structured("judge", "s", "u", ScoringPointJudgement, rec)
        finally:
            await client.close()
        metrics = rec.finish()
        assert count == 2
        assert metrics.tokens.total_tokens is None
        assert metrics.tokens.known_total_tokens == 15
        assert metrics.tokens.unknown_usage_requests == 1
        assert metrics.modules["judge"].retry_wait_seconds_sum >= .9
        assert "secret-body" not in metrics.model_dump_json()
    asyncio.run(run())


def test_auth_error_no_retry_no_raw_body():
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(401, json={"error": {"message": "token-is-secret"}})
    async def run():
        client = LLMClient(configuration(), {"HF_TOKEN": "token-is-secret"}, httpx.MockTransport(handler))
        rec = Recorder()
        try:
            with pytest.raises(LLMCallError) as e:
                await client.structured("judge", "s", "u", ScoringPointJudgement, rec)
            assert "token-is-secret" not in str(e.value)
        finally:
            await client.close()
        assert count == 1
        assert rec.finish().tokens.unknown_usage_requests == 1
    asyncio.run(run())


def test_concurrency_wall_time_and_request_sums():
    active = peak = 0
    async def handler(request):
        nonlocal active, peak
        active += 1; peak = max(peak, active)
        await asyncio.sleep(.04)
        active -= 1
        return httpx.Response(200, json=completion(VALID))
    async def run():
        client = LLMClient(configuration(), {"HF_TOKEN": "token"}, httpx.MockTransport(handler))
        # Precreate the client so first-import initialization does not distort this concurrency test.
        client._client(client.config.for_module("judge"))
        rec = Recorder()
        try:
            with rec.module("judge"):
                await asyncio.gather(*(client.structured("judge", "s", "u", ScoringPointJudgement, rec) for _ in range(6)))
        finally:
            await client.close()
        m = rec.finish()
        assert peak == 2
        assert m.tokens.total_tokens == 90
        assert m.modules["judge"].request_time_seconds_sum > m.modules["judge"].wall_time_seconds
        assert m.modules["judge"].queue_wait_seconds_sum > .08
    asyncio.run(run())


def test_usage_without_provider_fields_and_no_double_count():
    a = {"request_id": "a", "module": "judge", "model": "m", "kind": "llm", "api_time_seconds": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "completion_tokens_details": {"reasoning_tokens": 3}}}
    assert token_summary([a]).total_tokens == 15
    assert summarize_requests([a, a])["tokens"]["known_total_tokens"] == 15
    b = {**a, "request_id": "b", "usage": None}
    assert token_summary([a, b]).total_tokens is None
    assert token_summary([a, b]).known_total_tokens == 15
    assert token_summary([]).total_tokens == 0
    assert token_summary([{**b, "sent": False}]).usage_complete
    assert token_summary([{**b, "kind": "search"}]).total_tokens == 0


def test_recorder_clock_and_failed_module():
    now = [0.0]
    rec = Recorder(clock=lambda: now[0])
    with pytest.raises(RuntimeError):
        with rec.module("clean"):
            now[0] = 5
            raise RuntimeError("failure")
    rec.resource("load", 2)
    now[0] = 8
    metrics = rec.finish()
    assert metrics.modules["clean"].wall_time_seconds == 5
    assert metrics.modules["clean"].status == "failed"
    assert metrics.wall_time_seconds == 8
    assert metrics.resource_time_seconds == 2
    assert metrics.evaluation_time_seconds == 6


def test_interrupted_start_remains_unknown_on_resume():
    from jades.batch import observed_requests
    start = {"event": "request_started", "kind": "llm", "request_id": "r1", "module": "judge", "model": "m", "sent": True, "api_time_seconds": 0, "queue_wait_seconds": 0}
    missing = observed_requests([start])
    assert missing[0]["status"] == "interrupted"
    assert token_summary(missing).unknown_usage_requests == 1
    end = {**start, "event": "request", "status": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
    assert token_summary(observed_requests([start, end])).total_tokens == 3


def test_search_tool_round_trip_usage():
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        body = json.loads(request.content)
        if count == 1:
            response = completion(VALID)
            response["choices"][0]["message"]["tool_calls"][0]["function"] = {"name": "web_search", "arguments": '{"query": "verify"}'}
            return httpx.Response(200, json=response)
        assert any(m.get("role") == "tool" and "evidence" in m.get("content", "") for m in body["messages"])
        return httpx.Response(200, json=completion(VALID))
    async def run():
        client = LLMClient(configuration(), {"HF_TOKEN": "token"}, httpx.MockTransport(handler))
        rec = Recorder()
        async def search(query):
            assert query == "verify"
            return "retrieved evidence"
        try:
            await client.structured("judge", "s", "u", ScoringPointJudgement, rec, search)
        finally:
            await client.close()
        assert rec.finish().tokens.total_tokens == 30
        assert rec.metrics.requests[-1]["purpose"] == "after_search"
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["json_schema", "json_object", "text"])
def test_explicit_output_protocols(mode):
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion(VALID, text=True, usage=False))
    async def run():
        client = LLMClient(configuration(output_mode=mode), {"HF_TOKEN": "token"}, httpx.MockTransport(handler))
        rec = Recorder()
        try:
            result = await client.structured("judge", "s", "u", ScoringPointJudgement, rec)
        finally:
            await client.close()
        assert result.judge_score == .25
        assert rec.finish().tokens.total_tokens is None
        assert "tools" not in seen[0]
        if mode != "text":
            assert seen[0]["response_format"]["type"] == mode
    asyncio.run(run())
