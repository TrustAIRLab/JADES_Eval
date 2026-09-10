# Fingerprint visibility (0.1.2)

Public output hides the following system-generated details by default:

- `prompt_fingerprint` and `config_fingerprint` in individual result metadata.
- `signature` in batch results and summaries.
- Fingerprint-related compatibility details, such as `legacy_prompt_fingerprint_incomplete`.

Scores, explanations, model configuration, time/token metrics, and model/resource versions are still included. Fields with the same names inside user input are not modified.

## CLI

```bash
# Hide fingerprint details by default.
jades evaluate --input samples.json --output results.json

# Include fingerprint details explicitly.
jades evaluate --input samples.json --output results.json --include-fingerprints

# Change the display of saved results without rerunning successful samples.
jades evaluate --input samples.json --output results.json --resume --include-fingerprints
```

When enabled, these details appear in modern result JSON, summary JSON, and terminal output. The `--legacy` result body stays compatible; fingerprint details appear in its summary file.

Resume still handles failed or unfinished samples according to the normal rules. The display option itself makes no model calls and does not invalidate completed samples.

## Python

```python
# Metadata and model_dump omit fingerprints by default.
result = evaluator.evaluate(question, response)
public_data = result.to_dict()

# Serialize existing details without another LLM call.
debug_data = result.to_dict(include_fingerprints=True)

# Alternatively, include them when evaluating.
result = evaluator.evaluate(question, response, include_fingerprints=True)
```

`AsyncEvaluator.aevaluate()`, `evaluate_many()`, and `aevaluate_many()` also accept `include_fingerprints=True`.

## Internal persistence and validation

Checkpoints are internal recovery files. They always retain complete signatures, individual result fingerprints, and compatibility records. Hiding details in public output does not delete or alter the checkpoint's validation data. The display option is not part of Config, cache keys, or the batch signature, so toggling it does not change resume compatibility.

Resume still rejects mismatched inputs, configurations, or templates with an error. Accounting warnings, including incomplete usage history, remain visible regardless of this option.
