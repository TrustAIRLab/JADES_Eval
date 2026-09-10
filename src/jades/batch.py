"""Sample checkpoints with exclusive writers, explicit migration and durable usage history."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .config import fingerprint, MODULES, Config
from .contracts import legacy_prompt_fingerprint
from .evaluator import EvaluationError, prompt_fingerprint
from .locking import output_lock
from .metrics import summarize_requests, utc_now
from .presentation import project_result, compatibility_fields


@dataclass(frozen=True)
class BatchPaths:
    output: Path
    metrics: Path
    checkpoint: Path
    summary: Path

    @classmethod
    def for_output(cls, output):
        output = Path(output)
        return cls(output, Path(str(output) + ".metrics.jsonl"),
                   Path(str(output) + ".checkpoint.json"), Path(str(output) + ".summary.json"))

    def files(self):
        return (self.output, self.metrics, self.checkpoint, self.summary)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_input(path, response_field="response"):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("parameters", {}) if isinstance(data, dict) else {}
    if not isinstance(metadata, dict):
        raise ValueError("Input parameters must be a JSON object")
    items = data.get("jailbreaks") if isinstance(data, dict) and "jailbreaks" in data else ([data] if isinstance(data, dict) else data)
    if not isinstance(items, list):
        raise ValueError("Expected a list, single question/response object, or JailbreakBench document")
    normalized = []
    for i, row in enumerate(items):
        if not isinstance(row, dict):
            normalized.append({"index": i, "input_error": "Item must be an object"})
            continue
        question = row.get("goal") if "goal" in row else row.get("question")
        response = row.get(response_field)
        item = {"index": row.get("index"), "question": question, "response": response}
        if not isinstance(question, str) or not isinstance(response, str):
            item["input_error"] = "Missing or non-string question/response field"
        normalized.append(item)
    return normalized, metadata


def _reject_constant(value):
    raise ValueError("Non-finite value in persisted JSON")


def read_events(path, *, issues=None):
    """Strict reader, with recoverable final-line truncation under the output lock."""
    if not Path(path).exists():
        return []
    lines = Path(path).read_bytes().splitlines(keepends=True)
    events = []
    offset = 0
    for i, raw in enumerate(lines):
        if not raw.strip():
            offset += len(raw)
            continue
        try:
            event = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
            if not isinstance(event, dict):
                raise ValueError("Metrics event must be an object")
            events.append(event)
        except (ValueError, UnicodeError):
            if i != len(lines) - 1:
                raise ValueError("Corrupt metrics file: invalid interior JSONL record") from None
            # Preserve all complete bytes; do not let a partial UTF-8 tail block recovery.
            with Path(path).open("r+b") as f:
                f.truncate(offset)
            if issues is not None:
                issues.append("metrics_partial_tail")
        offset += len(raw)
    return events


def observed_requests(events):
    starts = {e["request_id"]: {**e, "event": "request", "status": "interrupted", "usage": None,
                               "duration_complete": False} for e in events if e.get("event") == "request_started"}
    starts.update({e["request_id"]: e for e in events if e.get("event") == "request"})
    return list(starts.values())


def _validate_record(record):
    for key in ("request_id", "module", "model", "kind", "status"):
        if not isinstance(record.get(key), str):
            raise ValueError("Invalid request record in accounting history")
    if record["kind"] not in ("llm", "search"):
        raise ValueError("Invalid request kind in accounting history")
    for key in ("api_time_seconds", "queue_wait_seconds"):
        value = record.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid request timing in accounting history")


def _load_history(path, *, required=False):
    issues = []
    if not path.exists():
        return [], ["metrics_history_missing"] if required else []
    try:
        events = read_events(path, issues=issues)
        for record in observed_requests(events):
            _validate_record(record)
        return events, issues
    except (OSError, ValueError, KeyError, TypeError):
        # Keep damaged logs for investigation; recover known requests from checkpoint.
        return [], ["metrics_history_unreadable"]


def _metrics_records(metrics):
    if not isinstance(metrics, dict):
        return []
    return [{**r, "run_id": metrics.get("run_id"), "sample_id": metrics.get("sample_id")}
            for r in metrics.get("requests", [])]


def _merge_records(ledger, records):
    for record in records:
        _validate_record(record)
        identifier = record["request_id"]
        old = ledger.get(identifier)
        if old and old.get("duration_complete", True) and not record.get("duration_complete", True):
            continue  # An orphan start must not overwrite a saved completion.
        ledger[identifier] = record


def _checkpoint_ledger(checkpoint):
    ledger = dict(checkpoint.get("requests", {}))
    for identifier, record in ledger.items():
        _validate_record(record)
        if identifier != record["request_id"]:
            raise ValueError("Invalid checkpoint accounting identifier")
    for row in checkpoint.get("items", {}).values():
        metrics = row.get("result", {}).get("metrics") or row.get("metrics")
        _merge_records(ledger, _metrics_records(metrics))
    return ledger


def _record_history_warnings(checkpoint, warnings):
    existing = checkpoint.setdefault("accounting_warnings", [])
    for warning in warnings:
        if warning not in existing:
            existing.append(warning)
    if warnings:
        checkpoint["accounting_incomplete"] = True


def _accounting_summary(records, incomplete=False):
    result = summarize_requests(records)
    result["history_complete"] = not incomplete
    if incomplete:
        # Lost history may contain calls with unknown identities/counts. Keep known
        # subtotals, but never label them as the complete input/output/total usage.
        groups = [result] + list(result["by_module"].values()) + list(result["by_model"].values())
        for group in groups:
            group["tokens"].update(input_tokens=None, output_tokens=None, total_tokens=None, usage_complete=False)
    return result


def _read_checkpoint(path):
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(data, dict) or not isinstance(data.get("items"), dict) or not isinstance(data.get("runs"), list):
        raise ValueError("Invalid checkpoint structure")
    return data


def _protect_paths(input_path, paths):
    if paths.output.name.endswith((".checkpoint.json", ".metrics.jsonl", ".summary.json", ".lock")):
        raise ValueError("Output must not itself use a reserved sidecar filename")
    for path in paths.files():
        if path.is_symlink():
            raise ValueError("Output and sidecar paths must not be symbolic links")
        if path.resolve() == Path(input_path).resolve() or (path.exists() and os.path.samefile(path, input_path)):
            raise ValueError("Output and sidecar paths must not overwrite the input file")


def _legacy_signature_matches(checkpoint, items, config, response_field):
    recorded_configs = [row.get("result", {}).get("metadata", {}).get("config")
                        for row in checkpoint["items"].values()]
    recorded_configs = [value for value in recorded_configs if isinstance(value, dict)]
    recorded_configs = list({fingerprint(value): value for value in recorded_configs}.values())
    if recorded_configs:
        # Verify typed settings against the requested configuration, but reproduce
        # the old hash with its exact original JSON number representations.
        if any(Config.model_validate(value).fingerprint() != config.fingerprint() for value in recorded_configs):
            return False
        candidates = recorded_configs
    else:
        # Empty/failed-only v0.1.0 checkpoints have no config snapshot. Older
        # Pydantic defaults could encode these float fields as integers. Only
        # account for that known representation ambiguity, not model parameters.
        candidates = [config.model_dump()]
        for path in [("sample_timeout",), ("llm", "timeout")]:
            variants = []
            for candidate in candidates:
                changed = deepcopy(candidate)
                parent = changed
                for key in path[:-1]:
                    parent = parent[key]
                value = parent[path[-1]]
                if isinstance(value, float) and value.is_integer():
                    parent[path[-1]] = int(value)
                    variants.append(changed)
            candidates.extend(variants)
    return any(checkpoint.get("signature") == fingerprint({
        "input": items, "config": value, "prompts": legacy_prompt_fingerprint(),
        "response_field": response_field, "version": "0.1.0",
    }) for value in candidates)


def _import_legacy(source, paths, items, config, response_field, signature):
    source = Path(source)
    if not source.name.endswith(".checkpoint.json"):
        raise ValueError("Legacy checkpoint must end with .checkpoint.json")
    if any(p.exists() for p in paths.files()):
        raise FileExistsError("Legacy import requires a fresh output path; original files are preserved")
    checkpoint = _read_checkpoint(source)
    if checkpoint.get("schema_version", 1) != 1:
        raise ValueError("--legacy-checkpoint accepts only v0.1.0 checkpoints")
    if not _legacy_signature_matches(checkpoint, items, config, response_field):
        raise ValueError("Cannot import legacy checkpoint: input, configuration or recorded prompts changed")
    expected_signature = checkpoint["signature"]
    # The explicit source argument acknowledges that old fingerprints omitted some
    # templates. Do not silently claim the historical templates were verified.
    checkpoint.update(schema_version=2, output_name=paths.output.name, signature=signature,
                      legacy_import={"checkpoint": str(source.resolve()), "signature": expected_signature},
                      compatibility_warnings=["legacy_prompt_fingerprint_incomplete"])
    checkpoint["requests"] = _checkpoint_ledger(checkpoint)
    json.dumps(checkpoint, ensure_ascii=False, allow_nan=False).encode("utf-8")
    # Establish a recoverable checkpoint before copying a potentially large log.
    # If copying is interrupted, ordinary --resume can recover known ledger data.
    atomic_json(paths.checkpoint, checkpoint)
    prefix = source.name[:-len(".checkpoint.json")]
    source_metrics = source.with_name(prefix + ".metrics.jsonl")
    if source_metrics.exists():
        paths.metrics.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=paths.metrics.name + ".", suffix=".tmp", dir=paths.metrics.parent)
        try:
            with os.fdopen(fd, "wb") as destination, source_metrics.open("rb") as original:
                shutil.copyfileobj(original, destination)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, paths.metrics)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return checkpoint


async def run_batch(evaluator, input_path, output_path, *, response_field="response", resume=False,
                    legacy=False, overwrite=False, limit=None, legacy_checkpoint=None, include_fingerprints=False):
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if legacy_checkpoint and not resume:
        raise ValueError("--legacy-checkpoint requires --resume")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    for module in MODULES:
        evaluator.config.for_module(module)
    output = Path(output_path)
    output = output.parent.resolve() / output.name
    paths = BatchPaths.for_output(output)
    _protect_paths(input_path, paths)
    # Held over all reads, evaluation, writes and final accounting. OS locks are
    # released on crash; the persistent lock inode is deliberately never deleted.
    with output_lock(output):
        if legacy_checkpoint:
            with output_lock(Path(legacy_checkpoint).resolve()):
                return await _run_locked(evaluator, input_path, paths, response_field=response_field,
                                         resume=resume, legacy=legacy, overwrite=overwrite, limit=limit,
                                         legacy_checkpoint=legacy_checkpoint, include_fingerprints=include_fingerprints)
        return await _run_locked(evaluator, input_path, paths, response_field=response_field,
                                 resume=resume, legacy=legacy, overwrite=overwrite, limit=limit,
                                 include_fingerprints=include_fingerprints)


async def _run_locked(evaluator, input_path, paths, *, response_field, resume, legacy, overwrite, limit,
                      legacy_checkpoint=None, include_fingerprints=False):
    start = time.perf_counter()
    run_id = uuid.uuid4().hex
    out, metrics_path, checkpoint_path, summary_path = paths.files()
    items, source_meta = read_input(input_path, response_field)
    # Validate global metadata before any overwrite or paid evaluation occurs.
    json.dumps(source_meta, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if limit is not None:
        items = items[:limit]
    contract = {"input": items, "config": evaluator.config.model_dump(), "prompts": prompt_fingerprint(),
                "response_field": response_field, "contract_version": 2}
    signature = fingerprint(contract)
    checkpoint = {"schema_version": 2, "signature": signature, "output_name": out.name, "items": {}, "runs": [], "requests": {}}
    replace_existing = False
    if legacy_checkpoint:
        checkpoint = _import_legacy(legacy_checkpoint, paths, items, evaluator.config, response_field, signature)
    elif resume:
        if not checkpoint_path.exists():
            old = out.with_suffix(".checkpoint.json")
            if old.exists():
                raise ValueError(f"Legacy checkpoint found: {old}. Use --resume --legacy-checkpoint with a fresh output path; old template provenance is incomplete")
            raise FileNotFoundError("Resume requires a checkpoint")
        checkpoint = _read_checkpoint(checkpoint_path)
        if checkpoint.get("schema_version") != 2:
            raise ValueError("Legacy checkpoint requires explicit --legacy-checkpoint import")
        if checkpoint["signature"] != signature:
            raise ValueError("Cannot resume: input, model configuration or prompts changed")
        if os.path.normcase(checkpoint.get("output_name", "")) != os.path.normcase(out.name):
            raise ValueError("Checkpoint belongs to a different output filename")
    elif any(p.exists() for p in paths.files()):
        if not overwrite:
            raise FileExistsError("Output exists; use --resume or --overwrite")
        replace_existing = True

    will_evaluate = any("input_error" not in item and checkpoint["items"].get(str(i), {}).get("status") != "ok"
                        for i, item in enumerate(items))
    validate = getattr(evaluator, "validate_configuration", None)
    if will_evaluate and validate is not None:
        validate()
    if replace_existing:
        for p in paths.files():
            p.unlink(missing_ok=True)

    prior_runs = bool(checkpoint["runs"])
    ledger = _checkpoint_ledger(checkpoint)
    events, history_issues = _load_history(metrics_path, required=prior_runs)
    _merge_records(ledger, observed_requests(events))
    if any(r.get("status") == "running" for r in checkpoint["runs"]):
        history_issues.append("previous_run_not_finalized")
    _record_history_warnings(checkpoint, history_issues)
    checkpoint["requests"] = ledger
    checkpoint["runs"].append({"run_id": run_id, "started_at": utc_now(), "status": "running"})
    atomic_json(checkpoint_path, checkpoint)
    current_success = current_failures = skipped = 0
    evaluated_samples = 0
    completed = False
    pending_error = None

    def write_outputs():
        rows = [checkpoint["items"][str(i)] for i in range(len(items)) if str(i) in checkpoint["items"]]
        if legacy:
            records = [{"new_index": r["new_index"], "index": r["index"], "jailbreak_qa_result": r.get("legacy_result", {}),
                        **({"error": r["error"]} if "error" in r else {})} for r in rows]
            atomic_json(out, {"meta_data": {"jailbreak_method": source_meta.get("method", "unknown"), "target_model": source_meta.get("model", "unknown"),
                        "jailbreak_qa_agent": "jades", "jailbreak_qa_agent_llm": contract["config"]["llm"]["model"]}, "jailbreak_qa_artifacts": records})
        else:
            public_rows = []
            for row in rows:
                public = {k: v for k, v in row.items() if k != "legacy_result"}
                if "result" in public:
                    public["result"] = project_result(public["result"], include_fingerprints=include_fingerprints)
                public_rows.append(public)
            atomic_json(out, {"schema_version": "1", "source_metadata": source_meta,
                              **({"signature": signature} if include_fingerprints else {}),
                              **compatibility_fields(checkpoint.get("compatibility_warnings", []), include_fingerprints=include_fingerprints),
                              "results": public_rows})

    try:
        for i, item in enumerate(items):
            existing = checkpoint["items"].get(str(i))
            if existing and existing["status"] == "ok":
                skipped += 1
                continue
            row = {"new_index": i, "index": item["index"], "run_id": run_id}
            if "input_error" in item:
                row.update(status="error", error=item["input_error"])
                current_failures += 1
            else:
                result = None
                try:
                    evaluated_samples += 1
                    result = await evaluator.aevaluate(item["question"], item["response"], metrics_path=metrics_path, run_id=run_id, sample_id=str(i))
                    row.update(status="ok", result=result.to_checkpoint_dict(), legacy_result=result.to_legacy_dict())
                    # Persistence-domain errors belong to this sample, not the batch.
                    json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    current_success += 1
                except EvaluationError as exc:
                    row.update(status="error", error=str(exc), metrics=exc.metrics.model_dump())
                    current_failures += 1
                except (ValueError, TypeError) as exc:
                    row = {"new_index": i, "index": item["index"], "run_id": run_id, "status": "error",
                           "error": f"Invalid evaluation result: {type(exc).__name__}"}
                    if result is not None:
                        row["metrics"] = result.metrics.model_dump()
                    current_failures += 1
            _merge_records(ledger, _metrics_records(row.get("result", {}).get("metrics") or row.get("metrics")))
            checkpoint["items"][str(i)] = row
            atomic_json(checkpoint_path, checkpoint)
            write_outputs()
        completed = True
    except BaseException as exc:
        pending_error = exc
        raise
    finally:
        try:
            elapsed = time.perf_counter() - start
            final_events, final_issues = _load_history(metrics_path, required=prior_runs or bool(ledger))
            _merge_records(ledger, observed_requests(final_events))
            _record_history_warnings(checkpoint, final_issues)
            checkpoint["runs"][-1].update(ended_at=utc_now(), wall_time_seconds=elapsed,
                                         status="completed" if completed else "interrupted")
            checkpoint["requests"] = ledger
            atomic_json(checkpoint_path, checkpoint)
            write_outputs()
            requests = list(ledger.values())
            current = [r for r in requests if r.get("run_id") == run_id]
            modules = {}
            for row in checkpoint["items"].values():
                if row.get("run_id") != run_id:
                    continue
                metrics = row.get("result", {}).get("metrics") or row.get("metrics") or {}
                for name, stats in metrics.get("modules", {}).items():
                    modules[name] = modules.get(name, 0) + stats["wall_time_seconds"]
            summary = {"run_id": run_id, "wall_time_seconds": elapsed, "success_count": current_success,
                       "failure_count": current_failures, "skipped_count": skipped, "total_items": len(items),
                       "samples_per_second": (current_success + current_failures) / elapsed if elapsed else 0,
                       "module_wall_time_seconds_sum": modules,
                       "current_run": _accounting_summary(current, bool(final_issues) and (evaluated_samples > 0 or not completed)),
                       "cumulative": _accounting_summary(requests, checkpoint.get("accounting_incomplete", False)),
                       "accounting_warnings": checkpoint.get("accounting_warnings", []),
                       **compatibility_fields(checkpoint.get("compatibility_warnings", []), include_fingerprints=include_fingerprints),
                       **({"signature": signature, "prompt_fingerprint": contract["prompts"],
                           "config_fingerprint": evaluator.config.fingerprint()} if include_fingerprints else {}),
                       "runs": checkpoint["runs"]}
            atomic_json(summary_path, summary)
        except Exception:
            if pending_error is None:
                raise
            # Preserve the original cancellation/persistence error, not a secondary
            # cleanup failure. The last atomic checkpoint remains authoritative.
    return summary
