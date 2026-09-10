# JADES

JADES evaluates a question/response pair using decompositional scoring. This package preserves the core business logic of the `JADES_light` baseline retained in the test fixtures: NLTK splitting, MiniLM rejection detection, cleaning, decomposition, matching, per-point judgment, and weighted aggregation.

## Install

Python 3.10 or newer:

```bash
pip install jades-eval
jades init
cp .env.example .env
```

Set `HF_TOKEN` in `.env`. The default model is **exactly** `deepseek-ai/DeepSeek-V4-Flash-0731:together`, at `https://router.huggingface.co/v1`. The package never silently substitutes a different model. Availability depends on that endpoint and your account.

Source code and release downloads: [GitHub](https://github.com/TrustAIRLab/jades-eval). Package index: [PyPI](https://pypi.org/project/jades-eval/).

Core dependencies include NLTK and sentence-transformers/PyTorch. First use prepares the original NLP assets; it does not fall back to a different algorithm. To prepare them before an offline run:

```bash
jades prepare-resources
```

## Python

```python
from jades import Evaluator

with Evaluator.from_env() as evaluator:
    result = evaluator.evaluate(
        question="What is the capital of France?",
        response="The capital of France is Paris.",
        metrics_path="example.metrics.jsonl",
    )
    print(result.score)
    print(result.metrics.wall_time_seconds)
    print(result.metrics.tokens.total_tokens)
    print(result.metrics.tokens.usage_complete)
    legacy = result.to_legacy_dict()
```

For notebooks and async applications:

```python
from jades import AsyncEvaluator

async def evaluate_pair(question, response):
    async with AsyncEvaluator.from_env() as evaluator:
        return await evaluator.aevaluate(question, response)
```

`evaluate_many()` / `aevaluate_many()` accept dictionaries with `question` and `response`. They preserve order and return `EvaluationError` objects for failed samples. Single-sample failures raise `EvaluationError`, whose `metrics` retain completed usage and whose `state` contains partial progress. Failure is never encoded as a zero score.

Provider content filtering (`content_filter`) and explicit refusal fields (`message.refusal`) stop the sample. English/Chinese text-rule matches only mark **suspected evaluator refusal**: they do not stop the pipeline, change scores or trigger extra calls. Inspect `result.metrics.output_issues` for `code="suspected_refusal", severity="warning"`; provider errors remain available through `EvaluationError.error_code` and `.metrics`. Invalid structured output still follows the original format-check/repair behavior. See [refusal-check scope and limitations](docs/refusal-checks.md).

## CLI and resume

```bash
jades evaluate --input samples.json --output results.json
jades evaluate --input samples.json --output results.json --resume
jades evaluate --input samples.json --output legacy.json --legacy
jades evaluate --input samples.json --output checked.json --fact-check
```

Accepts JailbreakBench `{"parameters": ..., "jailbreaks": [{"goal": ..., "response": ...}]}`, a list/single object with `question/response`, or JSONL. Use `--response-field truncated_response` for that original input field. `--limit N` is useful for smoke tests. Existing outputs require `--resume` or `--overwrite`.

Alongside `results.json`, version 0.1.1 writes `results.json.metrics.jsonl`, `results.json.checkpoint.json`, and `results.json.summary.json`. The full output filename is retained so `.json` and `.jsonl` outputs cannot share sidecars. A process lock rejects simultaneous writers to the same output and is released by the OS on exit. Resume skips successful samples, retries failed samples, and refuses changed inputs/configuration/actual prompt templates. Keep these files together. Legacy **export** still retains `messages`, `next`, and the original `is_simpe_rejection` spelling.

Recovery is **per sample**, not per node or partially completed API request. If interrupted in the middle of a sample, that sample runs again (with any existing decomposition cache); already saved successful samples are skipped. In-flight calls may have consumed tokens even if the client never received their usage. Preserve the checkpoint and metrics sidecar and keep the input, options (including `--limit`) and model configuration unchanged when resuming.

The checkpoint also preserves a request ledger. If a metrics log is missing or damaged, known usage can be recovered from the checkpoint, but cumulative totals are marked incomplete instead of claiming zero consumption. `known_*` totals remain available.

Old 0.1.0 checkpoints omitted some template fingerprints and used colliding sidecar names. Import them explicitly into a **fresh output**, after stopping old runners:

```bash
jades evaluate --input samples.json --output resumed.json --resume \
  --legacy-checkpoint results.checkpoint.json
```

This preserves the original files and records `legacy_prompt_fingerprint_incomplete`: historical template identity cannot be proven from the old checkpoint. Subsequent runs use `--output resumed.json --resume` normally. See [0.1.1 changes and migration](docs/hardening-0.1.1.md).

## Optional fingerprint details

System-generated prompt/config fingerprints, batch signatures and fingerprint-related compatibility details are hidden from public results, summaries and terminal output by default. Enable them explicitly:

```bash
jades evaluate --input samples.json --output results.json --include-fingerprints
# Re-export completed results with details without rerunning successful samples:
jades evaluate --input samples.json --output results.json --resume --include-fingerprints
```

Python: `evaluator.evaluate(question, response, include_fingerprints=True)`, or inspect an existing result with `result.to_dict(include_fingerprints=True)` without calling a model again. The async and `evaluate_many` interfaces support the same option. `to_dict()` defaults to hiding these fields.

Internal checkpoint files always retain the validation information. The display option does not change scoring, cache keys or resume checks. `--legacy` export keeps its historical shape; explicit fingerprint details are available in the companion summary. Accounting warnings and failure messages remain visible. See [display behavior](docs/fingerprint-display.md).

## Models and credentials

`jades.toml` contains non-secret settings; `.env` contains credentials. Each module can independently select a model, endpoint, credential environment-variable name, timeout and generation settings:

```toml
[modules.judge]
model = "your-judge-model"
base_url = "https://your-provider.example/v1"
api_key_env = "JUDGE_TOKEN"
```

```dotenv
HF_TOKEN=your_hugging_face_token
JUDGE_TOKEN=your_judge_token
# Equivalent per-module environment configuration:
JADES_JUDGE_MODEL=your-judge-model
JADES_JUDGE_BASE_URL=https://your-provider.example/v1
JADES_JUDGE_API_KEY_ENV=JUDGE_TOKEN
```

Modules: `clean`, `decompose`, `pair`, `judge`, `overall`, `fact_decompose`, `fact_clarify`, `fact_check`. A module inherits unspecified fields from `[llm]`. Priority: explicit Python/CLI overrides > process environment > `.env` > TOML > defaults. `.env` is read without modifying the process environment or expanding other variables. Explicitly choose files with `--config` / `--env-file`, or the Python `config_path` / `env_file` arguments.

Default HF credentials still load automatically. An implicitly discovered file is **not allowed to redirect `base_url` or select a different `api_key_env`**. For custom routing, explicitly select the trusted source:

```bash
jades evaluate --config jades.toml --input samples.json --output results.json
# If routing is defined through JADES_* entries in .env:
jades evaluate --env-file .env --input samples.json --output results.json
```

Python equivalents are `Evaluator.from_env(config_path="jades.toml")` and `Evaluator.from_env(env_file=".env")`. Process `JADES_*` variables and explicit Python overrides are also trusted. The CLI displays the selected sources and destinations without printing credential values. Explicitly selected configuration remains trusted input; this restriction does not make arbitrary untrusted configuration safe.

Default structured output is a `final_result` function tool, corresponding to light's PydanticAI tool output. Providers must support this protocol. Explicit `output_mode = "json_schema"`, `"json_object"` or `"text"` settings are available for other providers; they change the output protocol and, for text/JSON modes, append schema instructions. There is no automatic fallback. `temperature=0.0` is preserved. Set additional supported arguments under `[llm.parameters]` or `[modules.judge.parameters]`.

Extra parameters cannot override routing, messages, the single-result output protocol or request timeout. Legitimate vendor extensions such as `extra_body.thinking` remain supported, but protected fields inside `extra_body` are rejected. Request records include both configured `model` and provider `response_model`.

## Search and full fact checking

Both are off by default. `use_web_search=true` exposes Brave search during point judgment (`BRAVE_API_KEY`). `fact_check=true` enables sentence-level atomic-fact decomposition, clarification, retrieval and evidence-aware judgment. Its default source matches the extension: Tavily, Wikipedia-only, advanced search, one result, raw content enabled (`TAVILY_API_KEY`).

Set `fact_search_provider="brave"` explicitly to use Brave instead; this changes retrieved evidence. No provider is silently substituted. Evidence is recorded with URLs and excerpts. No retrieved evidence produces `unknown`, not `true` or `false`. Search failure remains visible in metrics.

## Compatibility and measurement

See [behavior contract](docs/compatibility.md), [metrics definitions](docs/metrics.md), and [release instructions](docs/releasing.md). Core prompts and intermediate states are tested against reference snapshots. Fixed model outputs reproduce the original calculations; a new LLM or new search results need not reproduce historical scores.

Cache is enabled, as in the current light configuration. SQLite caches decompositions by question, resolved model configuration, system prompt and the actual decomposition template. Older cache entries are preserved but are not silently reused under the strengthened key; an initial decomposition may be recomputed.

Evaluators fix their configuration at construction. Create a new `Evaluator` / `AsyncEvaluator` to change models, endpoints, credentials, or concurrency; replacing or mutating an existing evaluator's configuration raises a clear error. LLM and search requests share `max_concurrency`. Explicit `--env-file` / `env_file` paths must exist; automatic discovery and `env_file=None` remain optional. Input `parameters`, when present, must be a JSON object.

Static reuse additionally caches resolved module configuration and schema/tool definitions, without caching LLM answers or changing requests. Before constructing an evaluator, nested `Config` changes invalidate its resolved-config cache. NLP resource checks and fingerprints are reused within a process while paths and versions remain fixed; changing the search path or configured revision triggers a fresh check. Keep resource files immutable during a run. If replacing files at the same path in a running Python process, call `jades.resources.ensure_nltk(force=True)` to refresh the check (this also selects a tokenizer bound to the refreshed NLTK content; restart the process for changed semantic model weights); `jades prepare-resources` always performs a fresh check. See [optimization verification](docs/static-optimization.md).

## Development

```bash
pip install -e '.[dev]'
pytest
python -m build
```

The test suite is offline and never loads your credentials or calls external models. Real provider validation is separate and requires your own credentials.

The 0.1.3 infrastructure fixes and compatibility notes are documented in [P2 fixes](docs/p2-fixes-0.1.3.md).
