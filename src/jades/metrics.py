"""Usage is server-reported; missing usage is never inferred to be zero."""
from __future__ import annotations

import json
import time
import uuid
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Tokens(BaseModel):
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    total_tokens: int | None = 0
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    known_input_tokens: int = 0
    known_output_tokens: int = 0
    known_total_tokens: int = 0
    usage_complete: bool = True
    unknown_usage_requests: int = 0


def token_summary(requests: list[dict]) -> Tokens:
    llm = [r for r in requests if r.get("kind") == "llm" and r.get("sent", True)]
    def value(r, key):
        u = r.get("usage") or {}
        return _count(u.get(key)) if isinstance(u, dict) else None
    known = {k: sum(value(r, k) or 0 for r in llm) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    unknown = sum(any(value(r, k) is None for k in known) for r in llm)
    def total(key):
        return None if any(value(r, key) is None for r in llm) else known[key]
    def detail(r, group, key):
        usage = r.get("usage") or {}
        values = usage.get(group) if isinstance(usage, dict) else None
        return _count(values.get(key)) if isinstance(values, dict) else None
    cached = [detail(r, "prompt_tokens_details", "cached_tokens") for r in llm]
    reasoning = [detail(r, "completion_tokens_details", "reasoning_tokens") for r in llm]
    return Tokens(input_tokens=total("prompt_tokens"), output_tokens=total("completion_tokens"), total_tokens=total("total_tokens"),
                  known_input_tokens=known["prompt_tokens"], known_output_tokens=known["completion_tokens"], known_total_tokens=known["total_tokens"],
                  usage_complete=unknown == 0, unknown_usage_requests=unknown,
                  cached_input_tokens=sum(x for x in cached if x is not None) if any(x is not None for x in cached) else None,
                  reasoning_tokens=sum(x for x in reasoning if x is not None) if any(x is not None for x in reasoning) else None)


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def sanitize_usage(value):
    """Retain provider fields, replacing unpersistable non-finite numbers with null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize_usage(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_usage(item) for item in value]
    return value


class ModuleMetrics(BaseModel):
    wall_time_seconds: float = 0
    request_time_seconds_sum: float = 0
    queue_wait_seconds_sum: float = 0
    retry_wait_seconds_sum: float = 0
    request_count: int = 0
    failure_count: int = 0
    output_failure_count: int = 0
    output_warning_count: int = 0
    search_count: int = 0
    cache_hits: int = 0
    status: str = "completed"
    tokens: Tokens = Field(default_factory=Tokens)


class Metrics(BaseModel):
    run_id: str
    sample_id: str
    started_at: str
    ended_at: str | None = None
    wall_time_seconds: float = 0
    resource_time_seconds: float = 0
    evaluation_time_seconds: float = 0
    tokens: Tokens = Field(default_factory=Tokens)
    modules: dict[str, ModuleMetrics] = Field(default_factory=dict)
    requests: list[dict[str, Any]] = Field(default_factory=list)
    resource_events: list[dict[str, Any]] = Field(default_factory=list)
    output_issues: list[dict[str, Any]] = Field(default_factory=list)


class Recorder:
    def __init__(self, run_id: str | None = None, sample_id: str | None = None,
                 sink: str | Path | None = None, clock=time.perf_counter):
        self.clock = clock
        self.start = clock()
        self.sink = Path(sink) if sink else None
        self._response_ids = {}
        self.metrics = Metrics(run_id=run_id or uuid.uuid4().hex, sample_id=sample_id or uuid.uuid4().hex, started_at=utc_now())

    def event(self, event: dict):
        if self.sink:
            self.sink.parent.mkdir(parents=True, exist_ok=True)
            with self.sink.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"run_id": self.metrics.run_id, "sample_id": self.metrics.sample_id, **event}, ensure_ascii=False) + "\n")
                f.flush()

    def request(self, record: dict):
        record = {"request_id": uuid.uuid4().hex, **record}
        self.metrics.requests.append(record)
        self.event({"event": "request", **record})

    def bind_response(self, response, request_id):
        self._response_ids[id(response)] = request_id

    def output_issue(self, module, code, *, response=None, field=None, rule_id=None, source="model", severity="error"):
        issue = {"module": module, "code": code, "severity": severity, "field": field, "rule_id": rule_id, "source": source,
                 "request_id": self._response_ids.get(id(response)) if response is not None else None}
        self.metrics.output_issues.append(issue)
        metrics = self.metrics.modules.setdefault(module, ModuleMetrics())
        if severity == "error":
            metrics.status = "failed"
        self.event({"event": "output_issue", **issue})

    def resource(self, name: str, seconds: float, **metadata):
        event = {"name": name, "wall_time_seconds": seconds, **metadata}
        self.metrics.resource_events.append(event)
        self.event({"event": "resource", **event})

    @contextmanager
    def module(self, name: str):
        m = self.metrics.modules.setdefault(name, ModuleMetrics())
        start = self.clock()
        try:
            yield m
        except BaseException:
            m.status = "failed"
            raise
        finally:
            m.wall_time_seconds += self.clock() - start

    def retry_wait(self, module: str, seconds: float):
        self.metrics.modules.setdefault(module, ModuleMetrics()).retry_wait_seconds_sum += seconds

    def finish(self) -> Metrics:
        self.metrics.wall_time_seconds = self.clock() - self.start
        self.metrics.ended_at = utc_now()
        self.metrics.resource_time_seconds = sum(e["wall_time_seconds"] for e in self.metrics.resource_events)
        self.metrics.evaluation_time_seconds = max(0, self.metrics.wall_time_seconds - self.metrics.resource_time_seconds)
        self.metrics.tokens = token_summary(self.metrics.requests)
        for name in set(self.metrics.modules) | {r["module"] for r in self.metrics.requests}:
            m = self.metrics.modules.setdefault(name, ModuleMetrics())
            req = [r for r in self.metrics.requests if r["module"] == name]
            m.tokens = token_summary(req)
            m.request_count = sum(r["kind"] == "llm" and r.get("sent", True) for r in req)
            m.search_count = sum(r["kind"] == "search" and r.get("sent", True) for r in req)
            m.failure_count = sum(r["status"] != "ok" for r in req)
            m.output_failure_count = sum(issue["module"] == name and issue.get("severity", "error") == "error" for issue in self.metrics.output_issues)
            m.output_warning_count = sum(issue["module"] == name and issue.get("severity") == "warning" for issue in self.metrics.output_issues)
            m.request_time_seconds_sum = sum(r["api_time_seconds"] for r in req)
            m.queue_wait_seconds_sum = sum(r["queue_wait_seconds"] for r in req)
        self.event({"event": "sample_summary", "metrics": self.metrics.model_dump(exclude={"requests"})})
        return self.metrics


def summarize_requests(requests: list[dict]) -> dict:
    # A resumed checkpoint and sidecar may contain the same request.
    unique = {r["request_id"]: r for r in requests}
    rows = list(unique.values())
    def group(field):
        return {str(key): {"tokens": token_summary([r for r in rows if r.get(field) == key]).model_dump(),
                           "request_time_seconds_sum": sum(r["api_time_seconds"] for r in rows if r.get(field) == key),
                           "request_count": sum(r.get("sent", True) for r in rows if r.get(field) == key)}
                for key in sorted({r.get(field, "") for r in rows}, key=str)}
    return {"tokens": token_summary(rows).model_dump(), "by_module": group("module"), "by_model": group("model")}
