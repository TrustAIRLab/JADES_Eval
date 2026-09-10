import asyncio
import json

import httpx
import pytest

from jades.client import LLMClient
from jades.config import Config, ModelConfig, load_config, UntrustedConnectionConfig
from jades.metrics import Recorder
from jades.models import ScoringPointJudgement
from helpers import completion


@pytest.mark.parametrize("key", ["model", "messages", "tools", "tool_choice", "response_format", "stream", "timeout", "n", "functions", "function_call", "parallel_tool_calls", "extra_headers", "extra_query", "temperature", "max_tokens"])
def test_extra_body_cannot_override_protected_fields(key):
    with pytest.raises(ValueError, match="extra_body"):
        ModelConfig(parameters={"extra_body": {key: "override"}})


@pytest.mark.parametrize("key", ["timeout", "n", "functions", "function_call", "stream_options", "parallel_tool_calls"])
def test_top_level_transport_protocol_parameters_rejected(key):
    with pytest.raises(ValueError):
        ModelConfig(parameters={key: 1})


def test_vendor_extension_preserved_and_response_model_recorded():
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        data = completion({"scoring_point": "p", "judge_score": .3, "judge_reason": "evidence"})
        data["model"] = "actual-service-model"
        return httpx.Response(200, json=data)
    async def run():
        cfg = Config(llm=ModelConfig(model="configured-alias", parameters={"extra_body": {"thinking": {"type": "disabled"}}}))
        client = LLMClient(cfg, {"HF_TOKEN": "synthetic"}, httpx.MockTransport(handler))
        rec = Recorder()
        try:
            result = await client.structured("judge", "system", "user", ScoringPointJudgement, rec)
        finally:
            await client.close()
        assert result.judge_score == .3
        assert seen[0]["model"] == "configured-alias"
        assert seen[0]["thinking"] == {"type": "disabled"}
        assert seen[0]["messages"][0]["content"] == "system"
        request = rec.finish().requests[0]
        assert request["model"] == "configured-alias"
        assert request["response_model"] == "actual-service-model"
    asyncio.run(run())


def test_low_level_kwargs_cannot_bypass_extra_body_validation():
    async def run():
        client = LLMClient(Config(), {"HF_TOKEN": "synthetic"}, httpx.MockTransport(lambda r: pytest.fail("Must not send request")))
        try:
            with pytest.raises(ValueError, match="extra_body"):
                await client.request("judge", [], Recorder(), extra_body={"model": "shadow"})
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["toml", "env"])
def test_implicit_file_cannot_redirect_environment_secret(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    if kind == "toml":
        (tmp_path / "jades.toml").write_text('[llm]\nbase_url="https://untrusted.invalid/v1"\napi_key_env="UNRELATED_SECRET"\n')
    else:
        (tmp_path / ".env").write_text('JADES_BASE_URL=https://untrusted.invalid/v1\nJADES_API_KEY_ENV=UNRELATED_SECRET\n')
    with pytest.raises(UntrustedConnectionConfig) as caught:
        load_config(environ={"UNRELATED_SECRET": "SYNTHETIC_SECRET"})
    assert "SYNTHETIC_SECRET" not in str(caught.value)


def test_explicit_trusted_files_and_process_overrides_still_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "jades.toml").write_text('[modules.judge]\nbase_url="https://judge.invalid/v1"\napi_key_env="JUDGE_TOKEN"\n')
    (tmp_path / ".env").write_text('HF_TOKEN=hf-test\nJUDGE_TOKEN=judge-test\nUNREFERENCED_SECRET=not-returned\n')
    cfg, env = load_config(config_path="jades.toml", environ={})
    assert cfg.for_module("judge").api_key_env == "JUDGE_TOKEN"
    assert set(env) == {"HF_TOKEN", "JUDGE_TOKEN"}
    cfg, env = load_config(environ={"JADES_JUDGE_BASE_URL": "https://judge.invalid/v1", "JADES_JUDGE_API_KEY_ENV": "JUDGE_TOKEN"})
    assert cfg.for_module("judge").base_url == "https://judge.invalid/v1"


def test_default_dotenv_hf_token_still_automatic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('HF_TOKEN=hf-test\nJADES_MODEL=another-hf-model\n')
    cfg, env = load_config(environ={})
    assert env["HF_TOKEN"] == "hf-test"
    assert cfg.llm.model == "another-hf-model"


def test_explicit_env_file_does_not_trust_unrelated_implicit_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('HF_TOKEN=hf-test\n')
    (tmp_path / "jades.toml").write_text('[llm]\nbase_url="https://untrusted.invalid/v1"\n')
    with pytest.raises(UntrustedConnectionConfig):
        load_config(env_file=".env", environ={})


def test_malformed_provider_usage_is_unknown_not_a_batch_crash(tmp_path):
    data = completion({"scoring_point": "p", "judge_score": .3, "judge_reason": "evidence"})
    data["usage"].update(prompt_tokens="bad", completion_tokens=float("nan"), total_tokens=float("inf"))
    data["usage"]["prompt_tokens_details"] = "invalid"
    async def run():
        client = LLMClient(Config(), {"HF_TOKEN": "test"}, httpx.MockTransport(lambda r: httpx.Response(200, content=json.dumps(data), headers={"Content-Type": "application/json"})))
        rec = Recorder(sink=tmp_path / "metrics.jsonl")
        try:
            result = await client.structured("judge", "s", "u", ScoringPointJudgement, rec)
        finally:
            await client.close()
        assert result.judge_score == .3
        metrics = rec.finish()
        assert metrics.tokens.total_tokens is None and not metrics.tokens.usage_complete
        assert metrics.tokens.unknown_usage_requests == 1
        json.dumps(metrics.model_dump(), allow_nan=False)
    asyncio.run(run())
