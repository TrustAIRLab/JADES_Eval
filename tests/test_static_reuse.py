"""Static reuse must preserve wire requests and invalidate changed definitions."""
import asyncio
import copy
import json
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from pydantic import BaseModel

from jades.client import LLMClient
from jades.config import Config, ModelConfig
from jades.metrics import Recorder
from jades.models import ScoringPointJudgement
from helpers import completion

VALID = {"scoring_point": "p", "judge_score": .25, "judge_reason": "test"}


def test_config_reuse_invalidation_and_caller_isolation(monkeypatch):
    config = Config(llm=ModelConfig(parameters={"extra_body": {"value": [1]}}))
    initial_fingerprint = config.fingerprint()
    original = ModelConfig.model_validate
    validations = []
    def validate(cls, value, **kwargs):
        validations.append(copy.deepcopy(value))
        return original(value, **kwargs)
    monkeypatch.setattr(ModelConfig, "model_validate", classmethod(validate))
    first = config.for_module("judge")
    for _ in range(20):
        result = config.for_module("judge")
        assert result is not first
        assert result == first
    assert len(validations) == 1
    assert config.fingerprint() == initial_fingerprint
    assert "_module_cache" not in config.model_dump_json()
    first.model = "caller-only"
    first.parameters["extra_body"]["value"].append(99)
    assert config.for_module("judge").parameters == {"extra_body": {"value": [1]}}
    config.llm.parameters["extra_body"]["value"].append(2)
    assert config.for_module("judge").parameters == {"extra_body": {"value": [1, 2]}}
    assert len(validations) == 2
    config.modules["judge"] = {"model": "changed", "parameters": {"extra_body": {"other": 3}}}
    assert config.for_module("judge").model == "changed"
    assert len(validations) == 3
    config.modules["judge"]["parameters"]["extra_body"]["other"] = 4
    assert config.for_module("judge").parameters["extra_body"]["other"] == 4
    assert len(validations) == 4
    assert config.for_module("clean").model == config.llm.model


def test_schema_built_once_across_concurrent_requests(monkeypatch):
    builds = []
    original = ScoringPointJudgement.model_json_schema
    def schema(cls, **kwargs):
        builds.append(1)
        return original(**kwargs)
    monkeypatch.setattr(ScoringPointJudgement, "model_json_schema", classmethod(schema))
    async def run():
        client = LLMClient(Config(), {"HF_TOKEN": "test"}, httpx.MockTransport(lambda req: httpx.Response(200, json=completion(VALID))))
        rec = Recorder()
        try:
            await asyncio.gather(*(client.structured("judge", "s", "u", ScoringPointJudgement, rec) for _ in range(20)))
            assert len(builds) == 1
            assert rec.finish().tokens.total_tokens == 300
        finally:
            await client.close()
    asyncio.run(run())


def test_rebuilt_or_reconfigured_schema_invalidates_cache():
    class Output(BaseModel):
        value: int
    client = LLMClient(Config(), {})
    first = client._output_definition(Output)
    assert client._output_definition(Output) is first
    Output.model_config["title"] = "Changed"
    second = client._output_definition(Output)
    assert second is not first
    assert second["schema"]["title"] == "Changed"
    Output.model_rebuild(force=True)
    assert client._output_definition(Output) is not second


def old_client():
    mod = ModuleType("jades._reference_client")
    mod.__package__ = "jades"
    path = Path(__file__).parent / "baseline/client_0_1_0.py"
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), mod.__dict__)
    return mod.LLMClient


@pytest.mark.parametrize("mode", ["tool", "json_schema", "json_object", "text"])
@pytest.mark.parametrize("repair", [False, True])
def test_cached_and_original_http_requests_are_byte_identical(mode, repair):
    async def replay(client_class):
        bodies = []
        def handler(request):
            bodies.append(request.content)
            data = {"invalid": True} if repair and len(bodies) == 1 else VALID
            return httpx.Response(200, json=completion(data, text=mode != "tool"))
        client = client_class(Config(llm=ModelConfig(output_mode=mode)), {"HF_TOKEN": "test"}, httpx.MockTransport(handler))
        rec = Recorder()
        try:
            outputs = [await client.structured("judge", "unchanged system", "unchanged user", ScoringPointJudgement, rec) for _ in range(2)]
        finally:
            await client.close()
        return bodies, [o.model_dump() for o in outputs], rec.finish().tokens
    old = asyncio.run(replay(old_client()))
    new = asyncio.run(replay(LLMClient))
    assert old == new


@pytest.mark.parametrize("mode", ["tool", "json_schema"])
def test_request_mutation_does_not_poison_schema_template(mode):
    from openai.types.chat import ChatCompletion
    seen = []
    class MutatingClient(LLMClient):
        async def request(self, module, messages, recorder, purpose="evaluation", **kwargs):
            seen.append(copy.deepcopy(kwargs))
            definition = kwargs["tools"][0]["function"]["parameters"] if mode == "tool" else kwargs["response_format"]["json_schema"]["schema"]
            definition["properties"].clear()
            return ChatCompletion.model_validate(completion(VALID, text=mode != "tool"))
    async def run():
        client = MutatingClient(Config(llm=ModelConfig(output_mode=mode)), {})
        for _ in range(2):
            await client.structured("judge", "s", "u", ScoringPointJudgement, Recorder())
    asyncio.run(run())
    assert seen[0] == seen[1]


def test_resource_checks_cached_and_paths_versions_force_invalidate(tmp_path, monkeypatch):
    import nltk
    from jades import resources
    from concurrent.futures import ThreadPoolExecutor
    english = tmp_path / "english"
    english.mkdir()
    (english / "params.txt").write_text("test tokenizer parameters")
    paths, reads = [], []
    original_read = Path.read_bytes
    def read(path):
        if path.parent == english:
            reads.append(str(path))
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(nltk.data, "path", [])
    monkeypatch.setattr(nltk.data, "find", lambda name, **kwargs: paths.append(name) or str(english))
    monkeypatch.setattr(nltk, "download", lambda *a, **k: pytest.fail("Unexpected download"))
    monkeypatch.setattr(resources, "_nltk_checks", {})
    def check():
        rec = Recorder()
        resources.configure(tmp_path / "root", "rev-a", rec)
        resources.ensure_nltk()
        return rec.finish().resource_events[0]
    with ThreadPoolExecutor(4) as pool:
        events = list(pool.map(lambda _: check(), range(8)))
    assert len(paths) == 3 and len(reads) == 1
    assert sum(not e["reused"] for e in events) == 1
    assert len({e["punkt_tab_english_sha256"] for e in events}) == 1
    resources.configure(tmp_path / "root", "rev-b")
    resources.ensure_nltk()
    assert len(reads) == 2
    resources.configure(tmp_path / "other-root", "rev-b")
    resources.ensure_nltk()
    assert len(reads) == 3
    monkeypatch.setattr(nltk, "__version__", "another-version")
    resources.ensure_nltk()
    assert len(reads) == 4
    nltk.data.path.append("another-search-path")
    resources.ensure_nltk()
    assert len(reads) == 5
    resources.ensure_nltk(force=True)
    assert len(reads) == 6


def test_failed_resource_check_is_not_cached(tmp_path, monkeypatch):
    import nltk
    from jades import resources
    monkeypatch.setattr(nltk.data, "path", [])
    monkeypatch.setattr(resources, "_nltk_checks", {})
    def missing(name, **kwargs):
        raise LookupError(name)
    attempts = []
    monkeypatch.setattr(nltk.data, "find", missing)
    monkeypatch.setattr(nltk, "download", lambda *a, **k: attempts.append(1) or False)
    resources.configure(tmp_path)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="NLTK resource unavailable"):
            resources.ensure_nltk()
    assert len(attempts) == 2
    assert resources._nltk_checks == {}


def test_semantic_model_metadata_is_reused(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from jades import resources
    lookups = []
    monkeypatch.setattr(resources, "_models", {})
    monkeypatch.setattr(resources.importlib.metadata, "version", lambda name: lookups.append(name) or "fixed-version")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **k: str(tmp_path / k["revision"])))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda path: object()))
    rec = Recorder()
    resources.configure(tmp_path, "revision-one", rec)
    first = resources.get_semantic_checker()
    for _ in range(5):
        assert resources.get_semantic_checker() is first
    assert lookups == ["sentence-transformers", "torch"]
    events = rec.finish().resource_events
    assert not events[0]["reused"]
    assert all(e["reused"] for e in events[1:])
    resources.configure(tmp_path, "revision-two")
    assert resources.get_semantic_checker() is not first
    assert len(lookups) == 4
