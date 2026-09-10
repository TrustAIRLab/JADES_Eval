"""End-to-end regression for refusal phrases inside target response fields."""
import asyncio
import json

import httpx
import pytest

from jades.client import LLMClient, ModelOutputError
from jades.config import Config, ModelConfig
from jades.metrics import Recorder
from jades.models import CleanSentence, AllScoringPointsWithSentences
from jades.refusals import plain_refusal_rule
from helpers import completion


@pytest.mark.parametrize("mode", ["tool", "json_schema", "json_object", "text"])
@pytest.mark.parametrize("source", [
    '"First. I cannot participate in this evaluation."',
    'He wrote "First sentence. I cannot perform this evaluation." in the response.',
    'Start. "First. I cannot participate in this evaluation." Then continue.',
    '\u4ed6\u8bf4\u201c\u7b2c\u4e00\u53e5\u3002\u6211\u4e0d\u80fd\u53c2\u4e0e\u8bc4\u4f30\u3002\u201d\u7136\u540e\u7ee7\u7eed\u56de\u7b54\u3002',
    'I cannot participate in this evaluation.',
])
def test_source_refusal_survives_all_output_protocols(mode, source):
    data = {"original_sentence": source, "cleaned_sentence": source, "remove_whole": False, "reason": "Preserved the original sentence."}
    calls = []
    def handler(request):
        calls.append(request)
        response = completion(data, text=mode != "tool")
        # Some providers supply both a function result and explanatory content.
        if mode == "tool":
            response["choices"][0]["message"]["content"] = source
        return httpx.Response(200, json=response)
    async def run():
        client = LLMClient(Config(llm=ModelConfig(output_mode=mode)), {"HF_TOKEN": "test"}, httpx.MockTransport(handler))
        rec = Recorder()
        try:
            result = await client.structured("clean", "s", "u", CleanSentence, rec)
        finally:
            await client.close()
        assert result.model_dump() == data
        assert len(calls) == 1
        assert rec.finish().output_issues == []
        assert rec.metrics.tokens.total_tokens == 15
    asyncio.run(run())


def test_matched_sentence_with_escaped_refusal_is_not_evaluator_refusal():
    data = {"scoring_points_with_sentences": [{"scoring_point": "Point A", "matched_sentences": ['"First. I cannot perform this evaluation."'], "reason_for_match": "The source sentence matches the supplied point."}]}
    async def run():
        client = LLMClient(Config(llm=ModelConfig(output_mode="json_object")), {"HF_TOKEN": "test"}, httpx.MockTransport(lambda r: httpx.Response(200, json=completion(data, text=True))))
        rec = Recorder()
        try:
            result = await client.structured("pair", "s", "u", AllScoringPointsWithSentences, rec)
        finally:
            await client.close()
        assert result.model_dump() == data
        assert rec.finish().output_issues == []
    asyncio.run(run())


@pytest.mark.parametrize("raw", [
    '{"original_sentence": "I cannot perform this evaluation."}',
    '{"original_sentence": "First. I cannot perform this evaluation.',
    '```json\n{"original_sentence":"I cannot evaluate this."}\n```',
])
def test_invalid_structured_source_is_a_format_error_not_refusal(raw):
    assert plain_refusal_rule(raw) is None
    def handler(request):
        response = completion({}, text=True)
        response["choices"][0]["message"]["content"] = raw
        return httpx.Response(200, json=response)
    async def run():
        client = LLMClient(Config(llm=ModelConfig(output_mode="json_object", output_retries=0)), {"HF_TOKEN": "test"}, httpx.MockTransport(handler))
        rec = Recorder()
        try:
            with pytest.raises(ModelOutputError, match="invalid structured output"):
                await client.structured("clean", "s", "u", CleanSentence, rec)
        finally:
            await client.close()
        assert rec.finish().output_issues == []
    asyncio.run(run())
