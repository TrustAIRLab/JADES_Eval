import asyncio
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from jades import AsyncEvaluator, Config, ModelConfig, core, messages
from jades.batch import BatchPaths, run_batch, read_events, atomic_json, read_input
from jades.config import MODULES, fingerprint
from jades.contracts import prompt_fingerprint, legacy_prompt_fingerprint, decomposition_fingerprint
from jades.locking import output_lock, OutputInUseError
from helpers import completion, payload


@pytest.fixture
def nlp(monkeypatch):
    monkeypatch.setattr(core, "string_split", lambda s: ["First sentence.", "Second sentence."])
    monkeypatch.setattr(core, "simple_rejection_check", lambda s: False)


def config(**kwargs):
    return Config(llm=ModelConfig(max_retries=0), modules={m: {"model": m} for m in MODULES}, use_memory=False, **kwargs)


def handler(request):
    body = json.loads(request.content)
    user = next(m["content"] for m in body["messages"] if m["role"] == "user")
    return httpx.Response(200, json=completion(copy.deepcopy(payload(body["model"], user))))


def input_file(tmp_path, questions=("q",)):
    path = tmp_path / "input.json"
    path.write_text(json.dumps([{"question": q, "response": "response"} for q in questions]), encoding="utf-8")
    return path


def test_two_runners_cannot_share_output_and_cancel_releases_lock(tmp_path, nlp):
    inp, out = input_file(tmp_path), tmp_path / "result.json"
    async def run():
        entered = asyncio.Event()
        hold = asyncio.Event()
        async def blocked(request):
            entered.set()
            await hold.wait()
        async with AsyncEvaluator(config(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(blocked)) as first:
            task = asyncio.create_task(run_batch(first, inp, out))
            await asyncio.wait_for(entered.wait(), 5)
            checkpoint = BatchPaths.for_output(out).checkpoint
            saved = checkpoint.read_bytes()
            async with AsyncEvaluator(config(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(lambda r: pytest.fail("Second writer must not call API"))) as second:
                with pytest.raises(OutputInUseError):
                    await run_batch(second, inp, out, resume=True)
            assert checkpoint.read_bytes() == saved
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        async with AsyncEvaluator(config(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(handler)) as next_runner:
            result = await run_batch(next_runner, inp, out, resume=True)
        assert result["success_count"] == 1
    asyncio.run(run())


def test_lock_is_cross_process_and_released_after_process_death(tmp_path):
    out = tmp_path / "result.json"
    script = "from jades.locking import output_lock\nimport sys\nwith output_lock(sys.argv[1]):\n print('locked',flush=True)\n sys.stdin.read(1)\n"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    process = subprocess.Popen([sys.executable, "-u", "-c", script, str(out)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(OutputInUseError):
            with output_lock(out):
                pytest.fail("lock must exclude another process")
        process.kill()
        process.wait(timeout=5)
        with output_lock(out):
            pass
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


@pytest.mark.parametrize("stage,bad", [("judge", float("nan")), ("judge", "NaN"), ("judge", float("inf")), ("decompose", float("inf")), ("overall", float("nan"))])
def test_nonfinite_sample_does_not_abort_batch(tmp_path, nlp, stage, bad):
    inp, out = input_file(tmp_path, ("bad", "good")), tmp_path / "result.json"
    is_bad = False
    def response(request):
        nonlocal is_bad
        body = json.loads(request.content); module = body["model"]
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        if module == "clean":
            is_bad = "§GIVEN QUESTION§: bad" in user
        data = copy.deepcopy(payload(module, user))
        if is_bad and module == stage:
            if stage == "decompose":
                data["scoring_points_for_a_question"][0]["weight"] = bad
            elif stage == "overall":
                data["jailbreak_score_llm"] = bad
            else:
                data["judge_score"] = bad
        return httpx.Response(200, json=completion(data))
    async def run():
        async with AsyncEvaluator(config(overall_llm=True), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(response)) as evaluator:
            summary = await run_batch(evaluator, inp, out)
        assert summary["success_count"] == 1 and summary["failure_count"] == 1
        rows = json.loads(out.read_text(encoding="utf-8"))["results"]
        assert rows[0]["status"] == "error" and rows[1]["status"] == "ok"
        assert summary["current_run"]["tokens"]["known_total_tokens"] > 0
        assert len(json.loads(BatchPaths.for_output(out).checkpoint.read_text(encoding="utf-8"))["items"]) == 2
    asyncio.run(run())


def test_finite_operands_that_overflow_are_isolated(tmp_path, nlp):
    inp, out = input_file(tmp_path), tmp_path / "result.json"
    def response(request):
        body = json.loads(request.content); module = body["model"]
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        data = copy.deepcopy(payload(module, user))
        if module == "decompose":
            for point in data["scoring_points_for_a_question"]:
                point["weight"] = 1e308
        if module == "judge":
            data["judge_score"] = 1e308
        return httpx.Response(200, json=completion(data))
    async def run():
        async with AsyncEvaluator(config(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(response)) as evaluator:
            summary = await run_batch(evaluator, inp, out)
        assert summary["failure_count"] == 1
        assert "Non-finite" in json.loads(out.read_text(encoding="utf-8"))["results"][0]["error"]
    asyncio.run(run())


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_resume_recovers_known_usage_without_claiming_complete_history(tmp_path, nlp, damage):
    inp, out = input_file(tmp_path), tmp_path / "result.json"
    async def run():
        async with AsyncEvaluator(config(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(handler)) as evaluator:
            first = await run_batch(evaluator, inp, out)
            paths = BatchPaths.for_output(out)
            if damage == "missing":
                paths.metrics.unlink()
            else:
                paths.metrics.write_text('not json\n{}\n', encoding="utf-8")
            resumed = await run_batch(evaluator, inp, out, resume=True)
            assert first["cumulative"]["tokens"]["total_tokens"] == 90
            assert resumed["skipped_count"] == 1
            assert resumed["cumulative"]["tokens"]["known_total_tokens"] == 90
            assert resumed["cumulative"]["tokens"]["total_tokens"] is None
            assert not resumed["cumulative"]["tokens"]["usage_complete"]
            assert resumed["current_run"]["tokens"]["total_tokens"] == 0
            assert resumed["accounting_warnings"]
            again = await run_batch(evaluator, inp, out, resume=True)
            assert again["cumulative"]["tokens"]["known_total_tokens"] == 90
    asyncio.run(run())


def test_sidecar_names_do_not_collide(tmp_path, nlp):
    inp = input_file(tmp_path)
    a, b = tmp_path / "report.json", tmp_path / "report.jsonl"
    assert not set(BatchPaths.for_output(a).files()).intersection(BatchPaths.for_output(b).files())
    async def run():
        async with AsyncEvaluator(config(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(handler)) as evaluator:
            await run_batch(evaluator, inp, a)
            original = {p: p.read_bytes() for p in BatchPaths.for_output(a).files()}
            await run_batch(evaluator, inp, b, overwrite=True)
            assert all(p.read_bytes() == data for p, data in original.items())
            run_id = json.loads(a.read_text(encoding="utf-8"))["results"][0]["run_id"]
            await run_batch(evaluator, inp, a, resume=True)
            assert json.loads(a.read_text(encoding="utf-8"))["results"][0]["run_id"] == run_id
    asyncio.run(run())


def test_overwrite_missing_credentials_preserves_existing_files(tmp_path, nlp):
    inp, out = input_file(tmp_path), tmp_path / "report.json"
    async def run():
        async with AsyncEvaluator(config(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(handler)) as evaluator:
            await run_batch(evaluator, inp, out)
        original = {p: p.read_bytes() for p in BatchPaths.for_output(out).files()}
        async with AsyncEvaluator(config(), {}, transport=httpx.MockTransport(lambda r: pytest.fail("No network expected"))) as evaluator:
            with pytest.raises(ValueError, match="Missing credential"):
                await run_batch(evaluator, inp, out, overwrite=True)
            assert all(p.read_bytes() == data for p, data in original.items())
            # A completed resume is read-only with respect to the model service.
            summary = await run_batch(evaluator, inp, out, resume=True)
            assert summary["skipped_count"] == 1
    asyncio.run(run())


def test_missing_log_preserves_previous_failed_attempt_usage(tmp_path, nlp):
    inp, out = input_file(tmp_path), tmp_path / "report.json"
    fail = True
    def response(request):
        if fail and json.loads(request.content)["model"] == "judge":
            return httpx.Response(401, json={"error": {"message": "synthetic failure"}})
        return handler(request)
    async def run():
        nonlocal fail
        async with AsyncEvaluator(config(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(response)) as evaluator:
            failed = await run_batch(evaluator, inp, out)
            assert failed["failure_count"] == 1
            fail = False
            recovered = await run_batch(evaluator, inp, out, resume=True)
            known = recovered["cumulative"]["tokens"]["known_total_tokens"]
            assert known > recovered["current_run"]["tokens"]["known_total_tokens"]
            BatchPaths.for_output(out).metrics.unlink()
            skipped = await run_batch(evaluator, inp, out, resume=True)
            assert skipped["cumulative"]["tokens"]["known_total_tokens"] == known
            assert skipped["cumulative"]["tokens"]["unknown_usage_requests"] >= 1
            assert skipped["skipped_count"] == 1
    asyncio.run(run())


def test_prompt_changes_block_resume(tmp_path, nlp, monkeypatch):
    inp, out = input_file(tmp_path), tmp_path / "report.json"
    async def run():
        async with AsyncEvaluator(config(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(handler)) as evaluator:
            await run_batch(evaluator, inp, out)
            before = prompt_fingerprint()
            old = messages.clean
            monkeypatch.setattr(messages, "clean", lambda *a: old(*a) + " Changed template.")
            assert prompt_fingerprint() != before
            with pytest.raises(ValueError, match="Cannot resume"):
                await run_batch(evaluator, inp, out, resume=True)
    asyncio.run(run())


def test_core_pair_and_decomposition_templates_are_fingerprinted(monkeypatch):
    before = prompt_fingerprint()
    original = core.run_pair_node
    async def changed_pair(state, runtime):
        return await original(state, runtime)
    monkeypatch.setattr(core, "run_pair_node", changed_pair)
    assert prompt_fingerprint() != before
    old = decomposition_fingerprint()
    monkeypatch.setattr(core, "sys_prompt_question_decompose_v1_original", "changed")
    assert decomposition_fingerprint() != old


def test_output_schema_change_changes_prompt_contract(monkeypatch):
    from jades.models import ScoringPointJudgement
    before = prompt_fingerprint()
    monkeypatch.setitem(ScoringPointJudgement.model_config, "title", "Different wire schema title")
    assert prompt_fingerprint() != before


def test_partial_utf8_log_tail_is_repaired_and_reported(tmp_path):
    p = tmp_path / "events.jsonl"
    complete = '{"event":"test","text":"\u4e2d\u6587"}\n'.encode()
    p.write_bytes(complete + b'{"text":"\xe4\xb8')
    issues = []
    events = read_events(p, issues=issues)
    assert events == [{"event": "test", "text": '\u4e2d\u6587'}]
    assert p.read_bytes() == complete
    assert issues == ["metrics_partial_tail"]


def test_output_cannot_overwrite_input_even_with_force(tmp_path, nlp):
    inp = input_file(tmp_path)
    saved = inp.read_bytes()
    async def run():
        async with AsyncEvaluator(config(), {"HF_TOKEN": "test"}, transport=httpx.MockTransport(handler)) as evaluator:
            with pytest.raises(ValueError, match="overwrite the input"):
                await run_batch(evaluator, inp, inp, overwrite=True)
    asyncio.run(run())
    assert inp.read_bytes() == saved


@pytest.mark.parametrize("missing_metrics", [False, True])
def test_explicit_legacy_import_preserves_originals_and_marks_uncertainty(tmp_path, nlp, missing_metrics):
    inp, first = input_file(tmp_path, ('\u4e2d\u6587 q',)), tmp_path / "generated.json"
    old = tmp_path / "old.checkpoint.json"
    target = tmp_path / "imported.json"
    async def run():
        cfg = config()
        async with AsyncEvaluator(cfg, {"HF_TOKEN": "test"}, transport=httpx.MockTransport(handler)) as evaluator:
            await run_batch(evaluator, inp, first)
            paths = BatchPaths.for_output(first)
            data = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
            for key in ("schema_version", "output_name", "requests", "accounting_incomplete", "accounting_warnings"):
                data.pop(key, None)
            items, _ = read_input(inp)
            old_config = cfg.model_dump()
            old_config["sample_timeout"] = int(old_config["sample_timeout"])
            old_config["llm"]["timeout"] = int(old_config["llm"]["timeout"])
            for row in data["items"].values():
                row["result"]["metadata"]["config"] = old_config
            data["signature"] = fingerprint({"input": items, "config": old_config, "prompts": legacy_prompt_fingerprint(), "response_field": "response", "version": "0.1.0"})
            atomic_json(old, data)
            if not missing_metrics:
                old.with_name("old.metrics.jsonl").write_bytes(paths.metrics.read_bytes())
            original = old.read_bytes()
            imported = await run_batch(evaluator, inp, target, resume=True, legacy_checkpoint=old, include_fingerprints=True)
            assert old.read_bytes() == original
            assert imported["skipped_count"] == 1
            assert imported["cumulative"]["tokens"]["known_total_tokens"] == 90
            assert imported["compatibility_warnings"] == ["legacy_prompt_fingerprint_incomplete"]
            assert imported["cumulative"]["tokens"]["usage_complete"] is (not missing_metrics)
            current = json.loads(BatchPaths.for_output(target).checkpoint.read_text(encoding="utf-8"))
            assert current["schema_version"] == 2
            assert current["output_name"] == target.name
            await run_batch(evaluator, inp, target, resume=True)
            with pytest.raises(FileExistsError, match="fresh output"):
                await run_batch(evaluator, inp, first, resume=True, legacy_checkpoint=old)
    asyncio.run(run())
