# 0.1.1: audit fixes and compatibility

> Version 0.1.2 changes only the default visibility of fingerprint details. The signatures, fingerprints, and related compatibility records described below remain in internal checkpoints. Public output hides them unless `--include-fingerprints` is enabled. Resume validation and the fixes on this page remain effective.

This release addressed result-integrity issues identified by the Fable 5.1 review and local reproductions. It did not change business prompts, sentence-splitting algorithms, rejection thresholds, or the weighted formula for finite scores. Textual refusal warnings remain nonblocking.

## Fixes

1. **Concurrent output overwrites.** An operating-system process lock covers output reads, evaluation, checkpoint writes, and final accounting. A second writer fails immediately. The OS releases the lock when the process exits. Lock files are not deleted, avoiding replacement of a locked inode. Checkpoints are still rewritten after each sample; no new incremental storage architecture was introduced.
2. **SDK parameter overrides.** Both `parameters` and `extra_body` are checked for protected routing, message, output-protocol, and transport fields. Legitimate provider extensions, such as thinking settings, remain supported. Requests record both the configured model and the provider-reported model to make aliases traceable.
3. **Non-finite values aborting batches.** Output models reject NaN/Inf using the existing repair budget. Numeric overflow during aggregation is also a sample error. Results are checked for finite, UTF-8-encodable JSON before persistence, so invalid results do not enter checkpoints. Errors are not converted to zero scores, and cleanup failures no longer mask the original exception.
4. **Incomplete prompt fingerprints.** Fingerprints include actual user-message builders, relevant core-node templates, bound prompts, and output schemas as well as system prompts. Source is normalized as an AST; installation paths and source line numbers do not define template identity. Decomposition cache keys also include the template and schema. Unchanged definitions reuse their calculated fingerprints.
5. **Sidecar filename collisions.** Sidecars retain the full output filename, so `report.json` and `report.jsonl` have separate checkpoint, metrics, and summary files. Output cannot overwrite the input file or directly use a reserved sidecar filename.
6. **Historical usage appearing as zero after resume.** Checkpoints retain a request ledger, merged with logs by request ID on resume. Missing or damaged logs, partial final records, and unfinished prior runs produce explicit incomplete-history markers. Known usage remains available; missing history is not reported as a complete zero total.
7. **Implicit configuration redirecting credentials.** Default HF_TOKEN loading remains automatic, but implicitly discovered working-directory files cannot change the effective base_url/api_key_env. Custom routing requires an explicitly trusted config/env file, process-level JADES_* variables, or Python overrides. Clients receive only referenced credential variables and search credentials. The CLI shows configuration sources, models, and endpoints without printing credential values.
8. **Recovery file encoding.** Checkpoints explicitly use UTF-8. A truncated UTF-8 log tail can be trimmed to the last complete record while holding the output lock, with history marked incomplete.

## Resume current checkpoints

For output `results.json`, keep these files together:

- `results.json.checkpoint.json`
- `results.json.metrics.jsonl`
- `results.json.summary.json`

```bash
jades evaluate --input samples.json --output results.json --resume
```

Keep inputs, configuration, and templates unchanged. Recovery is still per sample: saved successes are skipped; failed or interrupted samples run again. Samples with textual warnings remain successful and are not rerun merely because of those warnings.

## Import a 0.1.0 checkpoint

Older checkpoints did not retain complete template fingerprints, so historical template identity cannot be established automatically. Explicitly select the old checkpoint and a **new output filename**:

```bash
jades evaluate --input samples.json --output resumed.json --resume \
  --legacy-checkpoint results.checkpoint.json
```

Inputs and configuration must still match the old signature. Original files are neither deleted nor overwritten. Logs are copied to the new output's sidecars, and saved successes are preserved. The imported records retain `legacy_prompt_fingerprint_incomplete` rather than presenting missing historical information as verified.

Continue with `--output resumed.json --resume` afterward. Stop old runners before importing: older versions do not follow the new locking protocol. The lock coordinates cooperating local runners; it is not isolation against a malicious filesystem writer.

## Custom endpoints

```bash
jades evaluate --config jades.toml --input samples.json --output results.json
# If routing is configured through JADES_* variables in .env:
jades evaluate --env-file .env --input samples.json --output results.json
```

No extra option is needed when `.env` contains only credential values and the default HF connection is retained. Explicitly selected configuration is still trusted input; do not mark an untrusted file as trusted. This protection focuses on credential routing and does not make arbitrary model or feature settings in malicious configuration safe.

## Preserved behavior

- Text-rule matches only create warnings; they do not stop evaluation, change scores, or add model calls.
- `message.refusal` and `content_filter` still stop the sample.
- Search failures still produce unknown evidence and error metrics, without a new score penalty.
- `reasoning_content` is not universally removed because some providers require it.
- Changing concurrency, timeouts, or other configuration still prevents ordinary resume, preserving the documented strict configuration-matching policy.

Validation covered process exclusion and lock release after process death, NaN/Inf/overflow isolation, valid provider extensions, implicit/explicit configuration boundaries, missing and damaged logs, changed prompts, filename isolation, truncated UTF-8 tails, and legacy checkpoint import. Regression tests used local mocks without real model or search calls.

At this release, **248 tests passed**. Previously saved results for six research samples were also imported and resumed: all six successes were skipped, **181,637 known historical tokens** were recovered, and no new calls or tokens were recorded. SHA-256 hashes of the original result, checkpoint, and metrics files remained unchanged. The validation retained `legacy_prompt_fingerprint_incomplete`; it did not retroactively mark missing template information as verified.

Original audit reproduction scripts remain historical evidence. Validate corrected behavior with `tests/test_hardening_config.py`, `tests/test_hardening_batch.py`, and the complete regression suite, rather than requiring old defects to reproduce.
