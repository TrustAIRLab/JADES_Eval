# Static reuse verification

Only the requested two optimizations are implemented. The batch writer and checkpoint/recovery algorithm are unchanged.

## Resolved settings and schema definitions

- Resolved per-module settings are cached privately, outside serialized configuration and fingerprints. Changes to defaults or nested module overrides invalidate the entry. Returned settings have isolated parameter dictionaries so callers cannot corrupt another request's configuration.
- Each client builds a model's JSON Schema, final-result tool definition and JSON schema instruction text once. Schema rebuilds or model-config changes invalidate that entry. Search-tool definitions are also reused. Mutable wire payloads receive copies rather than references to cached templates.
- No LLM answer is cached. Prompts, model parameters, output protocols, request counts and sampling remain unchanged.

## NLP checks

- A successful NLTK resource check and fingerprint are reused by resource directory, effective NLTK search path, NLTK version and configured resource revision. Failed checks are not cached.
- The existing MiniLM instance cache now also retains its package-version metadata instead of reading package metadata on every check.
- Resource metrics still record every call, the original fingerprint/version and whether validation was reused. Download lists describe only downloads made during that call.
- Resources must remain fixed within a run. NLTK tokenizers are bound to the resolved resource path and content hash without changing the global NLTK search path. Same-path NLTK replacement requires an explicit forced check; changed semantic weights still require a process restart. Evaluator configuration is fixed at construction; create a new evaluator to change it. `prepare-resources` forces a resource check. Tokenization and embedding calculations are unchanged.

## Validation

- HTTP request bodies are compared byte-for-byte against the pre-optimization client for all four output modes, both with and without output-repair retries, including repeated calls that exercise warm caches.
- Tests cover nested configuration changes, caller mutation isolation, schema rebuilds, concurrent schema use, resource path/version changes, concurrent initial resource checks and failed resource retries.
- An interrupted two-sample run was resumed: the first saved result was preserved, the unfinished second sample ran again, and its prior partial usage remained in the cumulative metrics.
- The existing original-light node replay and all other tests remain part of the suite.

A local microbenchmark measured warm configuration resolution at approximately 4.67 → 4.09 microseconds per call, and fresh schema generation versus cached tool-copy preparation at approximately 116.54 → 14.51 microseconds per call. These are local preparation measurements, not end-to-end model latency improvements. Token consumption is unchanged. No new paid model calls were needed to verify these changes.

## Resume usage

```bash
jades evaluate --input samples.json --output results.json --resume
```

This resumes at **sample boundaries**. It does not continue a clean/judge node or a partially received model response. Keep the checkpoint and metrics sidecar with the results and use the same input/configuration. An in-flight request can be billed even when no usage was returned; such usage is marked unknown instead of zero. The persistence strategy itself was not modified by this optimization.
