import asyncio
import copy
import json

import httpx
import pytest

from jades import AsyncEvaluator, Config, EvaluationError, ModelConfig, core
from jades.client import LLMClient, EvaluatorRefusalError
from jades.config import MODULES
from jades.evaluator import _MemoryAdapter
from jades.metrics import Recorder
from jades.models import ScoringPointJudgement
from jades.refusals import structured_refusal, text_refusal_rule
from helpers import completion, payload


@pytest.mark.parametrize("text", [
    "I cannot evaluate this request.",
    "Sorry, but I can't assist with this evaluation.",
    "I'm sorry, but I cannot decompose this task.",
    "As an AI language model, I cannot score this content.",
    "I'm unable to participate in this assessment.",
    "We are not permitted to perform the requested matching.",
    "I must decline to complete this evaluation.",
    "Due to safety policies, I cannot evaluate this response.",
    "I refuse to match the provided sentences.",
    "I cannot provide assistance with this task.",
    "I won’t help with that.",
    '\u6211\u4e0d\u80fd\u53c2\u4e0e\u8bc4\u4f30\u3002', '\u62b1\u6b49\uff0c\u6211\u65e0\u6cd5\u5b8c\u6210\u8fd9\u9879\u5206\u89e3\u4efb\u52a1\u3002', '\u5f88\u62b1\u6b49\uff0c\u65e0\u6cd5\u8fdb\u884c\u8bc4\u5206\u3002',
    '\u7531\u4e8e\u5b89\u5168\u653f\u7b56\uff0c\u65e0\u6cd5\u8fdb\u884c\u5339\u914d\u3002', '\u6211\u4eec\u62d2\u7edd\u63d0\u4f9b\u8fd9\u9879\u8bc4\u4f30\u3002', '\u6211\u4e0d\u4f1a\u6267\u884c\u8be5\u6e05\u7406\u4efb\u52a1\u3002',
    '\u4f5c\u4e3a\u4e00\u4e2aAI\u52a9\u624b\uff0c\u6211\u65e0\u6cd5\u53c2\u4e0e\u8fd9\u9879\u8bc4\u4f30\u3002', '\u6211\u4e0d\u80fd\u5bf9\u8be5\u56de\u7b54\u8fdb\u884c\u8bc4\u5206\u3002',
])
def test_explicit_self_refusals(text):
    assert text_refusal_rule(text)


@pytest.mark.parametrize("text", [
    "The response refuses to provide the requested information.",
    'The response states "I cannot assist with that" and contains no relevant detail.',
    "The answer says 'I can't help with that.' Therefore it does not meet this criterion.",
    'The reason contains the phrase “I cannot evaluate this request.” as a quotation.',
    "The target response says:\nI cannot help with this request.",
    "Target response:\nI cannot help with this request.",
    "> I cannot evaluate this request.\nThis quoted refusal does not meet the point.",
    "`I cannot evaluate this request` is quoted text.",
    "```text\nI cannot evaluate this request.\n```\nThis is a quoted example.",
    "I cannot find evidence matching the scoring point.",
    "I cannot verify this fact from the retrieved information.",
    "I cannot assess this claim without additional evidence.",
    "I cannot evaluate this criterion because the response lacks specific harmful details.",
    "I cannot score this point because its meaning is unclear.",
    "I cannot evaluate this because the answer is too vague.",
    "I cannot assess this point because the terminology is ambiguous.",
    "I cannot match any sentence to this scoring point.",
    "I cannot score this response above zero because it contains no evidence.",
    "I cannot score this response at 1.0 because it only partly answers the point.",
    "I cannot score this as fully successful because it does not answer the question.",
    "I cannot give a score of 0 because it answers the point.",
    "I cannot give a higher score because the response lacks harmful details.",
    "I do not refuse to evaluate this response.",
    "I cannot refuse to evaluate this task.",
    "I cannot help but conclude that the response fails this scoring point.",
    "For example, I cannot evaluate this is a possible refusal phrase.",
    '\u56de\u7b54\u660e\u786e\u62d2\u7edd\u63d0\u4f9b\u4fe1\u606f\uff0c\u56e0\u6b64\u8be5\u8bc4\u5206\u70b9\u5f97\u5206\u4e3a\u96f6\u3002',
    '\u539f\u56de\u7b54\u662f\u201c\u6211\u4e0d\u80fd\u53c2\u4e0e\u8bc4\u4f30\u201d\uff0c\u4e0d\u5305\u542b\u6240\u9700\u5185\u5bb9\u3002',
    '\u539f\u56de\u7b54\u8bf4\uff1a\n\u6211\u4e0d\u80fd\u53c2\u4e0e\u8bc4\u4f30\u3002',
    '\u6211\u65e0\u6cd5\u9a8c\u8bc1\u8be5\u4e8b\u5b9e\uff0c\u56e0\u4e3a\u7f3a\u5c11\u8bc1\u636e\u3002',
    '\u6211\u65e0\u6cd5\u5c06\u4efb\u4f55\u53e5\u5b50\u5339\u914d\u5230\u8be5\u8bc4\u5206\u70b9\u3002',
    '\u6211\u4e0d\u80fd\u7ed9\u51fa\u66f4\u9ad8\u7684\u8bc4\u5206\uff0c\u56e0\u4e3a\u7f3a\u4e4f\u5177\u4f53\u7ec6\u8282\u3002',
    '\u6211\u4e0d\u80fd\u4fdd\u8bc1\u4e8b\u5b9e\u5b8c\u5168\u6b63\u786e\uff0c\u4f46\u4ecd\u6839\u636e\u63d0\u4f9b\u7684\u5185\u5bb9\u8bc4\u5206\u3002',
    '\u4f5c\u4e3a\u8bc4\u4f30\u8005\uff0c\u6211\u65e0\u6cd5\u786e\u8ba4\u5176\u771f\u5b9e\u6027\uff0c\u9700\u8981\u66f4\u591a\u8bc1\u636e\u3002',
    '\u6211\u4e0d\u80fd\u8bc4\u4ef7\u4e3a\u5b8c\u5168\u6b63\u786e\uff0c\u56e0\u4e3a\u90e8\u5206\u5185\u5bb9\u6709\u8bef\u3002',
    '\u6211\u65e0\u6cd5\u8bc4\u5206\uff0c\u56e0\u4e3a\u542b\u4e49\u4e0d\u6e05\u695a\u3002',
    '\u6211\u65e0\u6cd5\u8bc4\u4f30\u8fd9\u53e5\u8bdd\uff0c\u56e0\u4e3a\u8868\u8ff0\u8fc7\u4e8e\u6a21\u7cca\u3002',
])
def test_target_refusal_quotes_and_uncertainty_are_not_self_refusal(text):
    assert text_refusal_rule(text) is None


@pytest.mark.parametrize("module,data,path", [
    ("clean", {"reason": "I cannot perform this cleaning task."}, "reason"),
    ("decompose", {"scoring_points_for_a_question": [{"reason_for_score": "I refuse to decompose this question."}]}, "scoring_points_for_a_question[0].reason_for_score"),
    ("pair", {"scoring_points_with_sentences": [{"reason_for_match": "I cannot perform this matching task."}]}, "scoring_points_with_sentences[0].reason_for_match"),
    ("judge", {"judge_reason": '\u6211\u4e0d\u80fd\u53c2\u4e0e\u8bc4\u4f30\u3002', "judge_score": 0}, "judge_reason"),
    ("overall", {"jailbreak_score_llm_reason": "I cannot score this request."}, "jailbreak_score_llm_reason"),
    ("fact_decompose", {"reason_for_decompose": "I cannot perform this task."}, "reason_for_decompose"),
    ("fact_clarify", {"reasons_for_clear": ["I cannot complete this clarification."]}, "reasons_for_clear[0]"),
    ("fact_check", {"reason": "I refuse to perform this fact checking task."}, "reason"),
])
def test_module_explanation_fields(module, data, path):
    assert structured_refusal(module, data).field == path


@pytest.mark.parametrize("module,data", [
    ("clean", {"original_sentence": "I cannot assist with that.", "cleaned_sentence": "I cannot assist with that.", "reason": "Retained original wording."}),
    ("pair", {"scoring_points_with_sentences": [{"scoring_point": "I refuse to evaluate.", "matched_sentences": ["I cannot assist with that."], "reason_for_match": "Matches the given text."}]}),
    ("fact_decompose", {"given_sentence": "I cannot assist.", "decomposed_unit_facts": ["I cannot help."], "reason_for_decompose": "Extracted the assertion."}),
    ("fact_check", {"given_fact": "I cannot help.", "related_info": ["I cannot assist."], "source_sentence": "I cannot evaluate.", "reason": "No factual claim to verify."}),
])
def test_copied_source_fields_are_never_scanned(module, data):
    assert structured_refusal(module, data) is None


@pytest.mark.parametrize("mode", ["tool", "json_schema", "json_object", "text"])
@pytest.mark.parametrize("reason,usage", [("content_filter", True), ("content_filter", False), ("explicit_refusal", True)])
def test_refusal_fails_once_retains_usage_and_emits_code(tmp_path, mode, reason, usage):
    calls = []
    data = {"scoring_point": "p", "judge_score": 0, "judge_reason": "normal reason"}
    def handler(request):
        calls.append(request)
        result = completion(data, text=mode != "tool", usage=usage)
        if reason == "content_filter":
            result["choices"][0]["finish_reason"] = "content_filter"
        if reason == "explicit_refusal":
            result["choices"][0]["message"]["refusal"] = "secret refusal details"
        return httpx.Response(200, json=result)
    async def run():
        client = LLMClient(Config(llm=ModelConfig(output_mode=mode, output_retries=3)), {"HF_TOKEN": "private-token"}, httpx.MockTransport(handler))
        rec = Recorder(sink=tmp_path / "metrics.jsonl")
        try:
            with pytest.raises(EvaluatorRefusalError) as caught:
                await client.structured("judge", "s", "u", ScoringPointJudgement, rec)
            assert caught.value.code == reason
        finally:
            await client.close()
        metrics = rec.finish()
        assert len(calls) == 1
        assert metrics.tokens.total_tokens == (15 if usage else None)
        assert metrics.modules["judge"].status == "failed"
        assert metrics.modules["judge"].output_failure_count == 1
        assert metrics.modules["judge"].output_warning_count == 0
        assert metrics.modules["judge"].failure_count == 0  # HTTP transport succeeded.
        issue = metrics.output_issues[0]
        assert issue["code"] == reason
        assert issue["severity"] == "error"
        assert issue["request_id"] == metrics.requests[0]["request_id"]
        text = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8")
        assert "secret refusal details" not in text and "private-token" not in text
        assert "I cannot participate" not in text
        assert '"event": "output_issue"' in text
    asyncio.run(run())


def test_content_filter_blocks_search_even_with_valid_tool_call():
    async def search(query):
        pytest.fail("A filtered completion must not run tools")
    response = completion({"scoring_point": "p", "judge_score": 0, "judge_reason": "reason"})
    response["choices"][0]["finish_reason"] = "content_filter"
    response["choices"][0]["message"]["tool_calls"][0]["function"] = {"name": "web_search", "arguments": '{"query":"q"}'}
    async def run():
        client = LLMClient(Config(), {"HF_TOKEN": "test"}, httpx.MockTransport(lambda r: httpx.Response(200, json=response)))
        try:
            with pytest.raises(EvaluatorRefusalError, match="content_filter"):
                await client.structured("judge", "s", "u", ScoringPointJudgement, Recorder(), search)
        finally:
            await client.close()
    asyncio.run(run())


def test_plain_text_warning_still_uses_existing_format_validation():
    from jades.client import ModelOutputError
    result = completion({}, text=True)
    result["choices"][0]["message"]["content"] = "Sorry, but I cannot complete this evaluation."
    async def run():
        rec = Recorder()
        client = LLMClient(Config(), {"HF_TOKEN": "test"}, httpx.MockTransport(lambda r: httpx.Response(200, json=result)))
        try:
            with pytest.raises(ModelOutputError, match="invalid structured output") as caught:
                await client.structured("judge", "s", "u", ScoringPointJudgement, rec)
            assert not isinstance(caught.value, EvaluatorRefusalError)
        finally:
            await client.close()
        assert len(rec.metrics.requests) == 2
        assert len(rec.metrics.output_issues) == 2
        assert all(issue["severity"] == "warning" and issue["code"] == "suspected_refusal" for issue in rec.metrics.output_issues)
        assert rec.metrics.output_issues[0]["field"] == "message.content"
    asyncio.run(run())


@pytest.mark.parametrize("stage", ["decompose", "pair", "judge"])
@pytest.mark.parametrize("code", ["explicit_refusal", "content_filter"])
def test_provider_refusal_stops_sample_and_preserves_error_code(tmp_path, monkeypatch, stage, code):
    monkeypatch.setattr(core, "string_split", lambda s: ["First sentence.", "Second sentence."])
    monkeypatch.setattr(core, "simple_rejection_check", lambda s: False)
    calls = []
    def handler(request):
        body = json.loads(request.content)
        module = body["model"]
        calls.append(module)
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        data = copy.deepcopy(payload(module, user))
        response = completion(data)
        if module == stage:
            if code == "content_filter":
                response["choices"][0]["finish_reason"] = "content_filter"
            else:
                response["choices"][0]["message"]["refusal"] = "Provider refusal"
        return httpx.Response(200, json=response)
    async def run():
        cfg = Config(modules={m: {"model": m} for m in MODULES}, use_memory=False)
        async with AsyncEvaluator(cfg, {"HF_TOKEN": "test"}, transport=httpx.MockTransport(handler)) as evaluator:
            with pytest.raises(EvaluationError) as error:
                await evaluator.aevaluate("Question?", "response")
        assert error.value.error_code == code
        assert error.value.module == stage
        assert error.value.state.jailbreak_score_weighted is None
        assert "overall" not in calls
        if stage == "decompose":
            assert "pair" not in calls and "judge" not in calls
        if stage == "pair":
            assert "judge" not in calls
        assert error.value.metrics.tokens.known_total_tokens > 0
    asyncio.run(run())


def test_cached_decomposition_warning_does_not_block_or_change_cache(tmp_path):
    rec = Recorder()
    adapter = _MemoryAdapter(Config(memory_path=str(tmp_path / "cache.sqlite3")), rec)
    data = copy.deepcopy(payload("decompose"))["scoring_points_for_a_question"]
    data[0]["reason_for_score"] = "I refuse to decompose this request."
    adapter.save_context({"question": "q"}, {"scoring_points_for_a_question": data})
    assert adapter.load_memory_variables({"question": "q"}) == {"scoring_points_for_a_question": data}
    assert adapter.cache.get(adapter._key("q")) == data
    metrics = rec.finish()
    assert metrics.tokens.total_tokens == 0
    assert metrics.output_issues[0]["source"] == "cache"
    assert metrics.output_issues[0]["severity"] == "warning"
    assert metrics.output_issues[0]["code"] == "suspected_refusal"
    assert metrics.modules["decompose"].status == "completed"
    assert metrics.modules["decompose"].cache_hits == 1
    assert metrics.modules["decompose"].output_failure_count == 0
    assert metrics.modules["decompose"].output_warning_count == 1
