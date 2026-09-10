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
    "我不能参与评估。", "抱歉，我无法完成这项分解任务。", "很抱歉，无法进行评分。",
    "由于安全政策，无法进行匹配。", "我们拒绝提供这项评估。", "我不会执行该清理任务。",
    "作为一个AI助手，我无法参与这项评估。", "我不能对该回答进行评分。",
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
    "回答明确拒绝提供信息，因此该评分点得分为零。",
    '原回答是“我不能参与评估”，不包含所需内容。',
    "原回答说：\n我不能参与评估。",
    "我无法验证该事实，因为缺少证据。",
    "我无法将任何句子匹配到该评分点。",
    "我不能给出更高的评分，因为缺乏具体细节。",
    "我不能保证事实完全正确，但仍根据提供的内容评分。",
    "作为评估者，我无法确认其真实性，需要更多证据。",
    "我不能评价为完全正确，因为部分内容有误。",
    "我无法评分，因为含义不清楚。",
    "我无法评估这句话，因为表述过于模糊。",
])
def test_target_refusal_quotes_and_uncertainty_are_not_self_refusal(text):
    assert text_refusal_rule(text) is None


@pytest.mark.parametrize("module,data,path", [
    ("clean", {"reason": "I cannot perform this cleaning task."}, "reason"),
    ("decompose", {"scoring_points_for_a_question": [{"reason_for_score": "I refuse to decompose this question."}]}, "scoring_points_for_a_question[0].reason_for_score"),
    ("pair", {"scoring_points_with_sentences": [{"reason_for_match": "I cannot perform this matching task."}]}, "scoring_points_with_sentences[0].reason_for_match"),
    ("judge", {"judge_reason": "我不能参与评估。", "judge_score": 0}, "judge_reason"),
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
        text = (tmp_path / "metrics.jsonl").read_text()
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
