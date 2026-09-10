"""Fingerprint display is opt-in and cannot affect inference or recovery."""
import asyncio
import copy
import json

import httpx
import pytest

from jades import AsyncEvaluator, Config, core, messages
from jades.batch import BatchPaths, run_batch
from jades.cli import main
from jades.config import MODULES
from helpers import completion, payload


@pytest.fixture
def nlp(monkeypatch):
    monkeypatch.setattr(core, "string_split", lambda s: ["First sentence.", "Second sentence."])
    monkeypatch.setattr(core, "simple_rejection_check", lambda s: False)


def settings():
    return Config(use_memory=False, modules={m: {"model": m} for m in MODULES})


def responder(calls):
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        return httpx.Response(200, json=completion(copy.deepcopy(payload(body["model"], user))))
    return handler


def test_python_default_serialization_hides_fingerprints_and_export_is_explicit(nlp):
    calls = []
    async def run():
        async with AsyncEvaluator(settings(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(responder(calls))) as evaluator:
            hidden = await evaluator.aevaluate("q", "r")
            assert "prompt_fingerprint" not in hidden.metadata
            assert "config_fingerprint" not in hidden.model_dump_json()
            assert "prompt_fingerprint" not in repr(hidden)
            count = len(calls)
            shown = hidden.to_dict(include_fingerprints=True)
            assert len(shown["metadata"]["prompt_fingerprint"]) == 64
            assert len(calls) == count
            assert "prompt_fingerprint" not in hidden.to_dict()["metadata"]
            assert "prompt_fingerprint" not in hidden.metadata
            explicit = await evaluator.aevaluate("q", "r", include_fingerprints=True)
            assert explicit.metadata["prompt_fingerprint"] == shown["metadata"]["prompt_fingerprint"]
            assert explicit.state == hidden.state
            assert explicit.metrics.tokens == hidden.metrics.tokens
            assert calls[:count] == calls[count:]
            assert "prompt_fingerprint" not in explicit.to_dict()["metadata"]
    asyncio.run(run())


def test_batch_display_toggle_on_resume_preserves_internal_contract(tmp_path, nlp):
    inp, out = tmp_path / "input.json", tmp_path / "result.json"
    inp.write_text(json.dumps([{"question": "q", "response": "r"}]))
    calls = []
    async def run():
        async with AsyncEvaluator(settings(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(responder(calls))) as evaluator:
            hidden_summary = await run_batch(evaluator, inp, out)
            hidden_output = json.loads(out.read_text())
            paths = BatchPaths.for_output(out)
            checkpoint = json.loads(paths.checkpoint.read_text())
            signature = checkpoint["signature"]
            original_row = copy.deepcopy(checkpoint["items"]["0"])
            for key in ("signature", "prompt_fingerprint", "config_fingerprint", "compatibility_warnings"):
                assert key not in hidden_summary
            assert "signature" not in hidden_output
            assert "prompt_fingerprint" not in hidden_output["results"][0]["result"]["metadata"]
            assert len(original_row["result"]["metadata"]["prompt_fingerprint"]) == 64
            count = len(calls)
            shown_summary = await run_batch(evaluator, inp, out, resume=True, include_fingerprints=True)
            shown_output = json.loads(out.read_text())
            assert shown_summary["signature"] == shown_output["signature"] == signature
            assert shown_output["results"][0]["result"]["metadata"]["prompt_fingerprint"]
            assert len(calls) == count and shown_summary["skipped_count"] == 1
            await run_batch(evaluator, inp, out, resume=True)
            again = json.loads(paths.checkpoint.read_text())
            assert again["signature"] == signature
            assert again["items"]["0"] == original_row
            assert "signature" not in json.loads(out.read_text())
            assert len(calls) == count
    asyncio.run(run())


@pytest.mark.parametrize("include", [False, True])
def test_compatibility_warning_display_is_opt_in(tmp_path, nlp, include):
    inp, out = tmp_path / "input.json", tmp_path / "result.json"
    inp.write_text(json.dumps([{"question": "q", "response": "r"}]))
    async def run():
        async with AsyncEvaluator(settings(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(responder([]))) as evaluator:
            await run_batch(evaluator, inp, out)
            path = BatchPaths.for_output(out).checkpoint
            checkpoint = json.loads(path.read_text())
            checkpoint["compatibility_warnings"] = ["legacy_prompt_fingerprint_incomplete"]
            path.write_text(json.dumps(checkpoint))
            summary = await run_batch(evaluator, inp, out, resume=True, include_fingerprints=include)
            output = json.loads(out.read_text())
            if include:
                assert summary["compatibility_warnings"] == ["legacy_prompt_fingerprint_incomplete"]
                assert output["compatibility_warnings"] == summary["compatibility_warnings"]
            else:
                assert "compatibility_warnings" not in summary
                assert "compatibility_warnings" not in output
            assert json.loads(path.read_text())["compatibility_warnings"] == ["legacy_prompt_fingerprint_incomplete"]
    asyncio.run(run())


def test_hidden_fingerprints_still_reject_template_changes(tmp_path, nlp, monkeypatch):
    inp, out = tmp_path / "input.json", tmp_path / "result.json"
    inp.write_text(json.dumps([{"question": "q", "response": "r"}]))
    async def run():
        async with AsyncEvaluator(settings(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(responder([]))) as evaluator:
            await run_batch(evaluator, inp, out)
            old = messages.clean
            monkeypatch.setattr(messages, "clean", lambda *a: old(*a) + "changed")
            with pytest.raises(ValueError, match="Cannot resume"):
                await run_batch(evaluator, inp, out, resume=True)
    asyncio.run(run())


def test_accounting_warnings_are_not_hidden(tmp_path, nlp):
    inp, out = tmp_path / "input.json", tmp_path / "result.json"
    inp.write_text(json.dumps([{"question": "q", "response": "r"}]))
    async def run():
        async with AsyncEvaluator(settings(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(responder([]))) as evaluator:
            await run_batch(evaluator, inp, out)
            BatchPaths.for_output(out).metrics.unlink()
            summary = await run_batch(evaluator, inp, out, resume=True)
            assert "metrics_history_missing" in summary["accounting_warnings"]
            assert not summary["cumulative"]["tokens"]["usage_complete"]
    asyncio.run(run())


def test_cli_default_and_explicit_display(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # Empty input exercises actual CLI serialization without model requests.
    inp, out = tmp_path / "input.json", tmp_path / "result.json"
    inp.write_text("[]")
    monkeypatch.delenv("JADES_BASE_URL", raising=False)
    monkeypatch.delenv("JADES_API_KEY_ENV", raising=False)
    args = ["evaluate", "--input", str(inp), "--output", str(out)]
    assert main(args) == 0
    assert "fingerprint" not in capsys.readouterr().out.lower()
    assert "signature" not in json.loads(out.read_text())
    assert main(args + ["--resume", "--include-fingerprints"]) == 0
    text = capsys.readouterr().out
    assert "Prompt fingerprint:" in text and "Batch signature:" in text


def test_legacy_export_shape_stays_unchanged_when_details_requested(tmp_path, nlp):
    inp, out = tmp_path / "input.json", tmp_path / "legacy.json"
    inp.write_text(json.dumps([{"question": "q", "response": "r"}]))
    async def run():
        async with AsyncEvaluator(settings(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(responder([]))) as evaluator:
            summary = await run_batch(evaluator, inp, out, legacy=True, include_fingerprints=True)
        assert "signature" in summary
        exported = json.loads(out.read_text())
        assert set(exported) == {"meta_data", "jailbreak_qa_artifacts"}
        assert "metadata" not in exported["jailbreak_qa_artifacts"][0]["jailbreak_qa_result"]
    asyncio.run(run())


def test_projection_does_not_mutate_user_content_or_checkpoint_data():
    from jades.presentation import project_result
    data = {"metadata": {"prompt_fingerprint": "hidden", "config_fingerprint": "hidden-too", "version": "v"},
            "state": {"full_response": "signature and prompt_fingerprint are user text"}}
    original = copy.deepcopy(data)
    projected = project_result(data)
    assert data == original
    assert projected["state"] == original["state"]
    assert projected["metadata"] == {"version": "v"}
