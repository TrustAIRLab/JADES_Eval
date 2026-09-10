"""Regressions for the concrete 0.1.2 audit reproductions; no external services."""
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import nltk
import pytest

from jades import AsyncEvaluator, Config, ModelConfig, EvaluationError, core, nlp, resources
from jades.batch import run_batch
from jades.client import LLMClient, ModelOutputError
from jades.config import MODULES, load_config
from jades.metrics import Recorder
from jades.models import ScoringPointJudgement
from jades.search import SearchClient
from helpers import completion, payload


def config(**kwargs):
    return Config(llm=ModelConfig(base_url="https://mock.test/v1", max_retries=0),
                  modules={m: {"model": m} for m in MODULES}, use_memory=False, **kwargs)


def answer(request):
    body = json.loads(request.content)
    user = next(m["content"] for m in body["messages"] if m["role"] == "user")
    return httpx.Response(200, json=completion(payload(body["model"], user)))


@pytest.mark.parametrize("change", ["replace", "concurrency", "nested"])
def test_evaluator_config_changes_fail_before_any_request(change, monkeypatch):
    monkeypatch.setattr(core, "string_split", lambda s: pytest.fail("Preprocessing must not start"))
    async def run():
        cfg = config()
        async with AsyncEvaluator(cfg, {"HF_TOKEN": "mock"}, transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected request"))) as ev:
            if change == "replace":
                ev.config = cfg.model_copy(deep=True)
                ev.config.llm.base_url = "https://replacement.test/v1"
            elif change == "concurrency":
                cfg.max_concurrency = 1
            else:
                cfg.modules["judge"]["model"] = "replacement-model"
            with pytest.raises(EvaluationError, match="create a new") as caught:
                await ev.aevaluate("Question?", "response")
            assert caught.value.metrics.tokens.total_tokens == 0
            assert not caught.value.metrics.requests
    asyncio.run(run())


def test_mid_sample_mutation_cannot_change_wire_settings_or_produce_valid_result(monkeypatch):
    monkeypatch.setattr(core, "string_split", lambda s: ["First sentence.", "Second sentence."])
    monkeypatch.setattr(core, "simple_rejection_check", lambda s: False)
    cfg = config(); urls = []
    def handler(request):
        urls.append(str(request.url))
        cfg.llm.base_url = "https://replacement.test/v1"
        return answer(request)
    async def run():
        async with AsyncEvaluator(cfg, {"HF_TOKEN": "mock"}, transport=httpx.MockTransport(handler)) as ev:
            with pytest.raises(EvaluationError, match="create a new") as caught:
                await ev.aevaluate("Question?", "response")
        assert len(urls) == 6
        assert all(u.startswith("https://mock.test/") for u in urls)
        assert caught.value.metrics.tokens.total_tokens == 90
    asyncio.run(run())


def make_resources(root, abbreviations):
    english = root / "nltk/tokenizers/punkt_tab/english"
    english.mkdir(parents=True)
    (root / "nltk/tokenizers/punkt").mkdir()
    for name in ("collocations.tab", "sent_starters.txt", "ortho_context.tab"):
        (english / name).write_text("", encoding="utf-8")
    (english / "abbrev_types.txt").write_text(abbreviations, encoding="utf-8")
    return english


def test_nltk_resource_switch_and_force_refresh_bind_actual_tokenizer(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    make_resources(a, "dr\n"); english_b = make_resources(b, "")
    paths = list(nltk.data.path)
    monkeypatch.setattr(nltk, "download", lambda *a, **k: pytest.fail("Unexpected download"))
    text = "He met Dr. Smith in Paris."
    def split(root):
        rec = Recorder(); resources.configure(root, recorder=rec)
        result = nlp.string_split(text)
        return result, rec.finish().resource_events[0]["punkt_tab_english_sha256"]
    first, hash_a = split(a)
    second, hash_b = split(b)
    assert first == [text] and second == ["He met Dr.", "Smith in Paris."]
    assert hash_a != hash_b
    assert split(a) == (first, hash_a)  # A -> B -> A must not be stuck on B.
    resources.configure(b)
    (english_b / "abbrev_types.txt").write_text("dr\n", encoding="utf-8")
    resources.ensure_nltk(force=True)
    assert resources.sent_tokenize(text) == [text]
    assert nltk.data.path == paths


def test_nltk_two_threads_do_not_share_resource_selection(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_resources(a, "dr\n"); make_resources(b, "")
    barrier = threading.Barrier(2)
    def split(root):
        resources.configure(root); resources.ensure_nltk()
        barrier.wait(timeout=5)
        return resources.sent_tokenize("He met Dr. Smith in Paris.")
    with ThreadPoolExecutor(2) as executor:
        result = list(executor.map(split, (a, b)))
    assert result == [["He met Dr. Smith in Paris."], ["He met Dr.", "Smith in Paris."]]


@pytest.mark.parametrize("text", ["A. B. Smith came home. He rested.", "Heading\n1. First item.\n2. Second item.",
                                 "He met Dr. Smith in Paris. Next sentence!", "```python\nprint('Hi.')\nx = 1.5\n```", '\u4f60\u597d\u3002\u4e16\u754c\uff01'])
def test_bound_nltk_algorithm_matches_original_with_same_resources(tmp_path, monkeypatch, text):
    from test_nlp import reference
    make_resources(tmp_path, "dr\n")
    monkeypatch.setattr(nltk.data, "path", [str(tmp_path / "nltk")])
    # Original API constructs its tokenizer with the same resource files.
    old = nltk.tokenize.PunktTokenizer("english")
    monkeypatch.setattr(nltk.tokenize, "sent_tokenize", old.tokenize)
    resources.configure(tmp_path)
    assert nlp.string_split(text) == reference()["string_split"](text)


def test_fact_search_respects_limit_and_tracks_queue_wait(monkeypatch):
    monkeypatch.setattr(core, "string_split", lambda s: ["First sentence."] * 12)
    monkeypatch.setattr(core, "simple_rejection_check", lambda s: False)
    active = peak = 0
    async def search(request):
        nonlocal active, peak
        active += 1; peak = max(peak, active)
        try:
            await asyncio.sleep(.01)
            return httpx.Response(200, json={"results": []})
        finally:
            active -= 1
    async def run():
        async with AsyncEvaluator(config(max_concurrency=2, fact_check=True), {"HF_TOKEN": "mock", "TAVILY_API_KEY": "mock"},
                                  transport=httpx.MockTransport(answer), search_transport=httpx.MockTransport(search)) as ev:
            result = await ev.aevaluate("Question?", "response")
        assert peak == 2
        assert result.score == pytest.approx(.65)
        metrics = result.metrics.modules["fact_check"]
        assert metrics.search_count == 12 and metrics.queue_wait_seconds_sum > 0
        assert all(f["is_fact_correct"] == "unknown" for s in result.all_fact_check_results for f in s["sentence_fact_check_results"])
    asyncio.run(run())


def test_cancel_queued_search_is_not_counted_as_sent():
    async def run():
        entered = asyncio.Event()
        async def handler(request):
            entered.set(); await asyncio.Event().wait()
        client = SearchClient({"TAVILY_API_KEY": "mock"}, httpx.MockTransport(handler), max_concurrency=1)
        rec = Recorder()
        first = asyncio.create_task(client.search("a", "tavily", rec, "fact_check"))
        await entered.wait()
        second = asyncio.create_task(client.search("b", "tavily", rec, "fact_check"))
        await asyncio.sleep(.01)
        second.cancel(); first.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        await client.close()
        records = rec.finish().requests
        assert len(records) == 2
        assert sum(r["sent"] for r in records) == 1
        queued = next(r for r in records if not r["sent"])
        assert queued["status"] == "cancelled" and queued["api_time_seconds"] == 0
        assert queued["queue_wait_seconds"] > 0
    asyncio.run(run())


@pytest.mark.parametrize("retries", [0, 1, 2])
def test_invalid_search_query_obeys_repair_budget_and_counts_all_usage(retries):
    def handler(request):
        response = completion(payload("judge"))
        response["choices"][0]["message"]["tool_calls"][0]["function"] = {"name": "web_search", "arguments": '{"query":""}'}
        return httpx.Response(200, json=response)
    async def run():
        cfg = Config(llm=ModelConfig(output_retries=retries, max_retries=0), max_search_rounds=20)
        client = LLMClient(cfg, {"HF_TOKEN": "mock"}, httpx.MockTransport(handler)); rec = Recorder()
        async def search(q):
            pytest.fail("Invalid query must not reach search")
        try:
            with pytest.raises(ModelOutputError, match="invalid search tool arguments"):
                await client.structured("judge", "s", "u", ScoringPointJudgement, rec, search)
        finally:
            await client.close()
        assert len(rec.metrics.requests) == retries + 1
        assert rec.finish().tokens.total_tokens == (retries + 1) * 15
    asyncio.run(run())


def test_invalid_query_repair_does_not_spend_valid_search_quota():
    calls = []; queries = []
    def handler(request):
        calls.append(json.loads(request.content)); response = completion(payload("judge"))
        if len(calls) < 3:
            response["choices"][0]["message"]["tool_calls"][0]["function"] = {
                "name": "web_search", "arguments": json.dumps({"query": "" if len(calls) == 1 else "valid"})}
        return httpx.Response(200, json=response)
    async def run():
        cfg = Config(llm=ModelConfig(output_retries=1, max_retries=0), max_search_rounds=1)
        client = LLMClient(cfg, {"HF_TOKEN": "mock"}, httpx.MockTransport(handler)); rec = Recorder()
        async def search(q):
            queries.append(q); return "evidence"
        try:
            result = await client.structured("judge", "s", "u", ScoringPointJudgement, rec, search)
        finally:
            await client.close()
        assert result.judge_score == payload("judge")["judge_score"]
        assert queries == ["valid"] and len(calls) == 3
        assert rec.finish().tokens.total_tokens == 45
    asyncio.run(run())


@pytest.mark.parametrize("metadata", [None, [], "wrong"])
@pytest.mark.parametrize("legacy", [False, True])
def test_invalid_metadata_rejected_before_overwrite_or_paid_calls(tmp_path, metadata, legacy):
    inp, out = tmp_path / "input.json", tmp_path / "output.json"
    inp.write_text(json.dumps({"parameters": metadata, "jailbreaks": [{"question": "q", "response": "r"}]}))
    out.write_text("previous output")
    async def run():
        async with AsyncEvaluator(config(), {"HF_TOKEN": "mock"}, transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected request"))) as ev:
            with pytest.raises(ValueError, match="parameters must be a JSON object"):
                await run_batch(ev, inp, out, overwrite=True, legacy=legacy)
        assert out.read_text(encoding="utf-8") == "previous output"
    asyncio.run(run())


def test_missing_explicit_env_fails_while_optional_discovery_remains_optional(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Explicit env file"):
        load_config(env_file=tmp_path / "missing.env", environ={"HF_TOKEN": "mock"})
    assert load_config(environ={"HF_TOKEN": "mock"})[1]["HF_TOKEN"] == "mock"
    assert load_config(env_file=None, environ={"HF_TOKEN": "mock"})[1]["HF_TOKEN"] == "mock"


def test_new_evaluator_uses_new_route_and_limit(monkeypatch):
    monkeypatch.setattr(core, "string_split", lambda s: ["First sentence."] * 4)
    monkeypatch.setattr(core, "simple_rejection_check", lambda s: False)
    active = peak = 0; urls = []
    async def handler(request):
        nonlocal active, peak
        active += 1; peak = max(peak, active); urls.append(str(request.url))
        try:
            await asyncio.sleep(.005)
            return answer(request)
        finally:
            active -= 1
    async def run():
        cfg = config(max_concurrency=1); cfg.llm.base_url = "https://new.test/v1"
        async with AsyncEvaluator(cfg, {"HF_TOKEN": "mock"}, transport=httpx.MockTransport(handler)) as ev:
            result = await ev.aevaluate("Question?", "response")
        assert peak == 1 and all(u.startswith("https://new.test/") for u in urls)
        assert result.metadata["config"]["llm"]["base_url"] == "https://new.test/v1"
        assert result.score == pytest.approx(.65)
    asyncio.run(run())


def test_llm_and_search_share_total_concurrency_budget():
    active = peak = 0
    async def handler(request):
        nonlocal active, peak
        active += 1; peak = max(peak, active)
        try:
            await asyncio.sleep(.01)
            return httpx.Response(200, json={"results": []}) if "tavily" in request.url.host else answer(request)
        finally:
            active -= 1
    async def run():
        rec = Recorder()
        async with AsyncEvaluator(config(max_concurrency=2), {"HF_TOKEN": "mock", "TAVILY_API_KEY": "mock"},
                                  transport=httpx.MockTransport(handler), search_transport=httpx.MockTransport(handler)) as ev:
            work = []
            for _ in range(4):
                work.extend([ev.client.request("judge", [{"role": "user", "content": "question"}], rec),
                             ev.search.search("q", "tavily", rec, "fact_check")])
            await asyncio.gather(*work)
        assert peak == 2
        metrics = rec.finish()
        assert len(metrics.requests) == 8 and metrics.tokens.total_tokens == 60
    asyncio.run(run())


def test_multiple_bad_queries_in_one_response_need_only_one_repair():
    calls = []
    def handler(request):
        calls.append(1); response = completion(payload("judge"))
        if len(calls) == 1:
            response["choices"][0]["message"]["tool_calls"] = [
                {"id": f"call-{i}", "type": "function", "function": {"name": "web_search", "arguments": '{"query":""}'}} for i in range(2)]
        return httpx.Response(200, json=response)
    async def run():
        client = LLMClient(Config(llm=ModelConfig(output_retries=1)), {"HF_TOKEN": "mock"}, httpx.MockTransport(handler))
        async def search(q):
            pytest.fail("Invalid search")
        try:
            await client.structured("judge", "s", "u", ScoringPointJudgement, Recorder(), search)
        finally:
            await client.close()
        assert len(calls) == 2
    asyncio.run(run())
