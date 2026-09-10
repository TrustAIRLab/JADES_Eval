# Measurement contract

Metrics are collected at the OpenAI SDK transport wrapper, covering async and sync entry points. SDK retries are disabled; JADES records every HTTP attempt, including retries and structured-output repair. Neither metrics nor request identifiers enter model prompts.

- `wall_time_seconds`: elapsed monotonic time, including preprocessing, resources, queueing, retry backoff, retrieval and model calls.
- `resource_time_seconds`: NLP preparation/loading spans. `evaluation_time_seconds` is wall time minus these spans. First-run resource work is visible separately; local imports/load time is included in resource preparation.
- `modules.*.wall_time_seconds`: elapsed time inside that module. `request_time_seconds_sum` adds individual HTTP durations and may exceed wall time under concurrency.
- `queue_wait_seconds_sum`: semaphore waiting. `retry_wait_seconds_sum`: actual backoff waiting.
- LLM `request_count` counts attempts submitted to the client, not logical successful completions. `search_count` is separate. A queued cancelled request has `sent=false` and consumes no known model tokens.
- HTTP `failure_count` remains a transport metric. `output_failure_count` counts provider refusal/content-filter errors; `output_warning_count` counts local text-rule suspicions. `output_issues` includes `severity` (`error` or `warning`), request ID, module, rule and field without raw refusal text. A `suspected_refusal` warning does not mark a module/sample failed or modify its score. HTTP success alone does not imply valid structured output.
- `input_tokens`, `output_tokens`, `total_tokens`: authoritative server usage. If any contributing request is missing a field, that aggregate field is `null`. `known_*` fields still sum reported values. `usage_complete` and `unknown_usage_requests` disclose incompleteness.
- Provider token counts must be nonnegative integers. Malformed counts are unknown; non-finite raw usage numbers are replaced by `null` so accounting itself cannot make a checkpoint unpersistable. Valid provider usage fields are retained.
- Cached-input and reasoning token fields are provider-reported breakdowns. They are not added again to `total_tokens`. The per-request `usage` retains the provider's original usage fields.
- A network failure, cancelled submitted request or server error without usage has unknown usage. A cache hit with no HTTP request has zero new token use. Search HTTP requests do not count as LLM tokens.

The default logs contain model/endpoint identifiers, request IDs, status, timings and usage, not prompts, completions, credentials or headers. Evaluation results separately retain original input and scoring details as in the research pipeline. Treat result files as research data.

CLI summaries report current-run and deduplicated cumulative usage, plus sample throughput and totals grouped by model and module. Request IDs prevent duplicate counting during resume. All sidecars and checkpoints must remain together. Interrupted requests whose outcome was never observed cannot have authoritative usage; no local estimate is presented as billable usage.

Version 0.1.1 also stores deduplicated request records in the checkpoint. Missing/damaged metrics history is reported through `accounting_warnings` and `history_complete=false`; recovered `known_*` totals remain available while complete cumulative totals become `null`. `unknown_usage_requests` only counts known requests with missing usage: it cannot count requests whose records were lost entirely. The history flag covers that additional uncertainty. A skipped-only resume makes no new model calls even if historical accounting is incomplete.

`model` identifies the configured model/route used for the request; `response_model` retains the provider's reported model identifier. A provider alias may legitimately differ. Protected `extra_body` fields cannot override the configured wire model.

Example:

```python
try:
    result = evaluator.evaluate(question, response)
except EvaluationError as error:
    print(error.metrics.tokens.known_total_tokens)
    print(error.metrics.tokens.unknown_usage_requests)
```

No monetary cost is inferred; prices and billing policies vary by provider.
