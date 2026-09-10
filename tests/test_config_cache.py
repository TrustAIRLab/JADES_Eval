import concurrent.futures
from pathlib import Path

import pytest

from jades.config import Config, ModelConfig, load_config
from jades.cache import DecompositionCache


def test_configuration_priority_and_no_environment_mutation(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=secret\nJADES_MODEL=dotenv\nJADES_JUDGE_MODEL=dotenv-judge\n")
    toml = tmp_path / "jades.toml"
    toml.write_text('[llm]\nmodel="toml"\n[modules.judge]\nbase_url="http://judge.test/v1"\napi_key_env="JUDGE_TOKEN"\n')
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config, credentials = load_config(toml, env, {"llm": {"model": "explicit"}}, {"JADES_MODEL": "process", "JADES_JUDGE_MODEL": "process-judge", "JUDGE_TOKEN": "private"})
    assert config.llm.model == "explicit"
    assert config.for_module("judge").model == "process-judge"
    assert config.for_module("judge").base_url == "http://judge.test/v1"
    assert credentials["HF_TOKEN"] == "secret"
    assert "secret" not in config.model_dump_json()
    import os
    assert "HF_TOKEN" not in os.environ


def test_module_override_does_not_change_others():
    c = Config(modules={"judge": {"model": "judge", "api_key_env": "JUDGE"}})
    assert c.for_module("clean").model == c.llm.model
    assert c.for_module("judge").api_key_env == "JUDGE"


@pytest.mark.parametrize("url", ["https://secret:password@example.com", "https://example.com?api_key=secret", "file:///tmp/test"])
def test_secret_url_rejected(url):
    with pytest.raises(ValueError):
        ModelConfig(base_url=url)


def test_cache_keys_and_concurrent_transactions(tmp_path):
    cache = DecompositionCache(str(tmp_path / "cache.sqlite3"))
    cfg = ModelConfig()
    assert cache.key(" q ", cfg, "p") == cache.key("q", cfg, "p")
    assert cache.key("q", cfg, "p") != cache.key("q", cfg.model_copy(update={"model": "other"}), "p")
    assert cache.key("q", cfg, "p") != cache.key("q", cfg, "other")
    # Initialize schema once then exercise independent SQLite connections.
    cache.get("missing")
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        list(pool.map(lambda i: cache.put(str(i), [{"weight": i}]), range(25)))
    assert all(cache.get(str(i)) == [{"weight": i}] for i in range(25))
