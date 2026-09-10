# Evaluator refusal checks

These checks apply to the LLM evaluating a sample. They are separate from MiniLM's detection of refusal in the **target response**. The target-response tokenizer, rejection threshold and weighted scoring formula are unchanged.

## Provider failures and local warnings

| Code | Signal | Handling |
|---|---|---|
| `content_filter` | `choices[0].finish_reason == "content_filter"` | Stop before parsing results or running tools, even if partial output looks valid. |
| `explicit_refusal` | Nonempty `message.refusal` | Stop the sample. |
| `suspected_refusal` | A local text rule matches assistant text or an evaluator-authored explanation field | Record a warning only; preserve valid results and continue. |

Only the two provider signals (`content_filter` and `explicit_refusal`) cause refusal errors. They stop the sample without format-repair retries, model substitution or attempts to work around a provider filter. Reported usage is preserved and missing usage remains unknown. Other batch samples continue.

Local text-rule matches are **suspected evaluator refusal**, not proof. They do not abort a sample, change a score, cancel sibling tasks, skip later modules, invalidate cache entries or introduce extra requests. A valid structured result is returned unchanged, including any zero/nonzero score and the original explanation. Warnings are recorded separately from failures.

Output-format checks still apply independently: a plain-text reply that does not satisfy the required schema receives the existing bounded format-repair handling, and can eventually fail as invalid output. Its text warning does not itself cause that failure or alter the repair policy. No score is fabricated to compensate for missing structured data.

## Conservative local rules

No extra LLM is called and no prompt/model parameter is changed. The local detector recognizes explicit first-person declarations such as `I cannot participate in this evaluation` or its Chinese-language equivalent. It checks module-specific explanation fields, including decomposition reasons, matching reasons, judgment reasons, overall reasons and fact-processing explanations.

The guard does **not** infer refusal merely from a zero score, the word “refusal,” an empty list or lack of evidence. It excludes copied answers, cleaned sentences, matched sentences, facts and retrieved evidence. Quoted statements, reported target refusals, markdown quotations and common expressions of insufficient evidence are excluded. For example, `The response says "I cannot help"` and `I cannot verify this fact from the retrieved information` do not constitute evaluator self-refusal.

Structured replies are parsed first and checked only through the field allowlist. Their serialized JSON is never scanned as plain text, including when strings contain escaped quotation marks. A plain-text fallback is used only when no valid structured result was supplied; JSON-like malformed replies remain format errors. This prevents copied source strings from bypassing the field exclusions. Expressions that a point is unclear, vague or ambiguous are treated as uncertainty, not an evaluator refusal.

This is a conservative English/Chinese rule detector, not a comprehensive semantic classifier. Unusual wording, other languages, ambiguous speaker attribution or structured nonanswers without explicit refusal can evade it; false positives cannot be ruled out for arbitrary prose. Changing legitimate scores based on uncertainty or adding general semantic-quality judgment is outside this change.

In particular, if an explanation field repeats a source refusal in first person without quotation or attribution, local wording alone cannot reliably identify the speaker. The regression tests and saved-output scan do not establish a population false-positive rate or guarantee zero false positives.

## Diagnostics

Warnings are available on successful results; provider failures retain the existing `EvaluationError` interface:

```python
from jades import EvaluationError

try:
    result = evaluator.evaluate(question, response)
    warnings = [issue for issue in result.metrics.output_issues
                if issue.get("severity") == "warning"]
    print(warnings)          # code: suspected_refusal
except EvaluationError as error:
    print(error.error_code)  # content_filter / explicit_refusal; other errors may have no code
    print(error.module)      # e.g. decompose or pair
    print(error.metrics.output_issues)
```

`metrics.output_issues` and `output_issue` sidecar events include `severity`, code, module, field path, rule identifier, source (`model` or `cache`) and the originating request ID when available. They do not store raw refusal text, prompts or credentials. `modules.*.output_warning_count` counts local suspicions; `output_failure_count` counts output errors; HTTP `failure_count` remains a separate transport metric. A warning alone leaves the module completed and the batch sample successful.

Decomposition-cache entries are also checked for audit visibility when read. A suspected refusal is marked with `source="cache"`; the cached data is still returned unchanged and the cache hit is counted normally. New valid decompositions with warnings retain the original caching behavior. This local heuristic never decides whether to reuse, delete or rewrite cached data.

The existing sample-level checkpoint/writing strategy is unchanged. `--resume` retries failed or unfinished samples and skips saved successful samples, including successful samples carrying warnings. Installing this change does not retroactively re-audit completed results; use a new output path to re-evaluate old results. Older logs may contain `semantic_refusal` errors from the former blocking policy; new text-rule matches use `suspected_refusal` warnings.

## Validation

Offline tests cover all output modes, provider refusal failures, warning-only structured and plain-text matches, quoted target refusals, legitimate zero scores and evidence uncertainty. Pipeline tests compare the same simulated responses with text detection enabled and disabled: request bodies, call counts, scores, full states and token usage match. Tests also verify that provider refusals still terminate decomposition/matching/judgment, cached warnings do not change data, and warned successful samples are skipped on resume.

An offline scan of 170 saved stage outputs from the prior six research samples and two fact-check smoke samples triggered no new refusal rule. This is a regression check on those outputs, not an estimate of detector accuracy. No paid model calls are required by the added checks.
