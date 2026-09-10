# 0.1.3: fixes from the 0.1.2 P2 audit

This release fixes the six P2 categories. The P3 checkpoint diagnostic improvement remains deferred. The GitHub release workflow publishes this version to PyPI.

## Configuration matches execution

An evaluator uses a validated private configuration snapshot from construction. Replacing its public `config` object or mutating its fields is rejected before evaluation; mutation during an active sample is detected before returning a successful result, with known usage retained. In-flight calls continue to use the original snapshot. Construct a new `Evaluator` or `AsyncEvaluator` to change models, endpoints, credentials or concurrency. Editing a `Config` before construction remains supported. The low-level LLM client also rejects changing its concurrency limit after construction.

## Resource-bound NLTK tokenizers

Each thread resolves the configured resource directory before the normal NLTK resource paths, without modifying `nltk.data.path`. A tokenizer is bound to the resolved English Punkt parameters, their content hash and NLTK version. The same PunktSentenceTokenizer algorithm, paragraph splitting and custom merging rules remain in use; no code-block protection or rule-based fallback was added.

A -> B -> A resource switching and two concurrent evaluator threads are isolated. `ensure_nltk(force=True)` refreshes the resource hash and selects the matching tokenizer. Keep resources immutable during evaluation; same-path semantic-model weight replacement still requires a process restart. Initial tokenizer construction is recorded as a separate resource event.

## Shared concurrency and query repair

LLM and search calls share the instance's `max_concurrency` semaphore. Search queue time is recorded separately from API time. A cancelled queued search is marked unsent and is not counted as a sent search.

Invalid search-tool arguments consume the configured `output_retries` repair budget, once per invalid model response, instead of silently using the valid-search quota. Valid queries alone consume `max_search_rounds`. Every model attempt, including exhausted repairs, retains its usage. With `output_retries=0`, the first invalid query fails after one LLM response. This is an intentional change to an infrastructure error path. Normal valid tool interactions are unchanged.

## Early input/configuration errors

An explicitly selected nonempty `.env` path must be an existing file. Missing automatically discovered `.env` and Python `env_file=None` remain supported; process environment precedence is unchanged.

An input document's `parameters`, when present, must be a JSON object. Null, arrays and strings are rejected before any model call or output overwrite, in both normal and legacy export modes. Omitted `parameters` still defaults to `{}`.

## Compatibility

Business prompts, core nodes, sentence merge rules, semantic rejection template/model/threshold and weighted scoring remain unchanged. Fingerprint display remains opt-in. No new success-path score normalization or short circuit was added. LLM/search scheduling and the documented malformed-query failure path change as described above.

The configuration schema and prompt fingerprint are unchanged, so correctly recorded 0.1.2 completed checkpoints can resume without new model calls. Historical successful scores are not retroactively corrected: if a prior run replaced evaluator configuration or switched differing NLP resources within the same process, re-evaluate it into a fresh output using the intended configuration/resources.
