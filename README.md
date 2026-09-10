# JADES Eval

JADES Eval measures how well a model response fulfills a given question or task, including requests used in jailbreak evaluations. Provide a **question** and a **response** to receive a score, scoring explanations, elapsed time, and token usage.

Choose one of two ways to use JADES:

| Method | Best for | Start here |
|---|---|---|
| **Python** | Calling JADES from your scripts, notebooks, or evaluation pipelines | [Python walkthrough](#option-1-python-walkthrough) |
| **CLI** | Evaluating JSON/JSONL files, saving results, and resuming interrupted batches | [CLI walkthrough](#option-2-cli-walkthrough) |

[PyPI package](https://pypi.org/project/jades-eval/) · [GitHub Releases](https://github.com/TrustAIRLab/JADES_Eval/releases)

## Before you start: install and configure credentials

Both methods use the same setup. You need **Python 3.10 or newer**. We recommend installing JADES in a dedicated Python virtual environment. On Windows, use PowerShell for the commands below.

### Step 1: create a working directory and install

```bash
mkdir jades-demo
cd jades-demo
python -m pip install --upgrade jades-eval
```

### Step 2: create the configuration files

```bash
jades init
```

This creates two files in your current directory:

- `jades.toml`: model names, service endpoints, and evaluation settings.
- `.env.example`: a credentials template.

### Step 3: add your credentials

Copy the template:

```bash
cp .env.example .env
```

Open `.env` in a text editor and enter your own Hugging Face token:

```dotenv
HF_TOKEN=your_hugging_face_token
```

JADES uses these defaults, so you do not need to edit `jades.toml` yet:

| Setting | Default |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731:together` |
| Endpoint | `https://router.huggingface.co/v1` |
| Credential variable | `HF_TOKEN` |

Run all subsequent commands and scripts from `jades-demo`. Keep credentials in `.env`; do not put them in Python scripts or commit them to a public repository. LLM calls use your account and are billed by your provider.

On first use, JADES downloads and caches the sentence-splitting and semantic-detection resources as needed, so startup may take longer. You can also prepare the resources before running either example:

```bash
jades prepare-resources --config jades.toml --env-file .env
```

## Option 1: Python walkthrough

### Step 1: create `demo.py`

Save this complete example as `demo.py`. The geography question demonstrates the API; replace `question` and `response` with your own evaluation sample when you are ready.

```python
import json
from pathlib import Path

from jades import Evaluator, EvaluationError

question = "What is the capital of France?"
response = "Paris is the capital of France. It is home to the Eiffel Tower."

try:
    with Evaluator.from_env(
        config_path="jades.toml",
        env_file=".env",
    ) as evaluator:
        result = evaluator.evaluate(
            question=question,
            response=response,
            metrics_path="python-demo.metrics.jsonl",
        )
except EvaluationError as error:
    print("Evaluation failed:", error)
    print("Known tokens consumed before failure:", error.metrics.tokens.known_total_tokens)
    raise SystemExit(1)

print("Score:", result.score)
print(f"Elapsed time: {result.metrics.wall_time_seconds:.2f} seconds")
print("Input tokens:", result.metrics.tokens.input_tokens)
print("Output tokens:", result.metrics.tokens.output_tokens)
print("Total tokens:", result.metrics.tokens.total_tokens)

Path("python-result.json").write_text(
    json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("Full results saved to python-result.json")
```

### Step 2: run the script

```bash
python demo.py
```

### Step 3: inspect the results

The terminal prints the score, elapsed time, and token usage. A successful run also creates:

| File | Contents |
|---|---|
| `python-result.json` | Full results, explanations for each scoring point, and usage summaries |
| `python-demo.metrics.jsonl` | Timing, status, and usage for individual requests |

You can also inspect scoring explanations directly in Python:

```python
# Append this after the successful evaluation in demo.py.
for point in result.state.scoring_points_judgement_for_a_question or []:
    print(point["scoring_point"])
    print("Score:", point["judge_score"])
    print("Reason:", point["judge_reason"])
```

Higher scores indicate that the response fulfills more of the evaluated task. Results depend on the sample and evaluator model; this example does not prescribe a fixed score. A token field of `None` means the provider did not supply complete usage information, not that the request consumed zero tokens.

<details>
<summary>Optional: evaluate multiple samples in Python</summary>

Save this as `batch_demo.py`, then run `python batch_demo.py`:

```python
from jades import Evaluator, EvaluationError

samples = [
    {
        "question": "What is the capital of France?",
        "response": "Paris is the capital of France.",
    },
    {
        "question": "Name two planets in the Solar System.",
        "response": "Mars and Jupiter are planets in the Solar System.",
    },
]

with Evaluator.from_env(config_path="jades.toml", env_file=".env") as evaluator:
    results = evaluator.evaluate_many(samples)

for index, result in enumerate(results):
    if isinstance(result, EvaluationError):
        print(index, "Failed:", result)
    else:
        print(index, "Score:", result.score, "Total tokens:", result.metrics.tokens.total_tokens)
```

Results follow the input order. For automatic file output and resumable batches, use the CLI walkthrough below.

</details>

<details>
<summary>Optional: use JADES in a notebook or asynchronous application</summary>

Run this in a notebook cell:

```python
from jades import AsyncEvaluator

async with AsyncEvaluator.from_env(
    config_path="jades.toml",
    env_file=".env",
) as evaluator:
    result = await evaluator.aevaluate(
        question="What is the capital of France?",
        response="Paris is the capital of France.",
    )

print(result.score)
```

</details>

## Option 2: CLI walkthrough

### Step 1: create `samples.json`

Save the following as `samples.json`. Each sample contains a `question` and a `response`:

```json
[
  {
    "question": "What is the capital of France?",
    "response": "Paris is the capital of France. It is home to the Eiffel Tower."
  },
  {
    "question": "Name two planets in the Solar System.",
    "response": "Mars and Jupiter are planets in the Solar System."
  }
]
```

### Step 2: run the evaluation

```bash
jades evaluate --config jades.toml --env-file .env --input samples.json --output results.json
```

The terminal reports the number of successful, failed, and skipped samples, together with the elapsed time and known token usage for this run.

### Step 3: open the output files

The working directory will contain:

| File | Purpose |
|---|---|
| `results.json` | Scores, explanations, and processing status for each sample |
| `results.json.summary.json` | Batch timing, success/failure counts, and current/cumulative usage |
| `results.json.metrics.jsonl` | Individual request records |
| `results.json.checkpoint.json` | Saved progress for resuming the batch |

Open `results.json` in a text editor. Each item in its `results` list corresponds to one sample:

| Field | Meaning |
|---|---|
| `status` | `ok` for success; `error` for failure |
| `result.state.jailbreak_score_weighted` | The sample's score |
| `result.state.scoring_points_judgement_for_a_question` | Scores and explanations for individual scoring points |
| `result.metrics.wall_time_seconds` | The sample's elapsed time |
| `result.metrics.tokens.total_tokens` | Total tokens for this evaluation attempt; `null` means incomplete usage |
| `error` | The explanation for a failed sample; failed entries must not be treated as valid scores |

### Step 4: resume after an interruption

Keep the output files and use the same input, configuration, and output path, adding `--resume`:

```bash
jades evaluate --config jades.toml --env-file .env --input samples.json --output results.json --resume
```

Successfully saved samples are skipped. Failed or interrupted samples are evaluated again. Recovery operates on **whole samples**, not individual model calls within a sample. Requests sent before an interruption may have consumed tokens.

To evaluate every sample again, choose a new output path, such as `--output results-new.json`.

### Step 5: use your own data

Replace the example questions and responses with your own samples. JADES also accepts:

- **JSONL**: one JSON object containing `question` and `response` per line. Use a `.jsonl` filename.
- **JailbreakBench format**: a top-level `jailbreaks` list whose samples contain `goal` and `response`. If present, top-level `parameters` must be a JSON object.
- **Truncated responses**: add `--response-field truncated_response` when that is the response field in your data.

See all CLI options:

```bash
jades evaluate --help
```

## Change models for either method

You can use one model for every module or configure modules individually.

### Use one model for all modules

Open the generated `jades.toml` and **replace its existing `[llm]` section**; do not add a second section with the same name. Replace the example values with your provider's actual settings:

```toml
[llm]
model = "your-model-name"
base_url = "https://your-provider.example/v1"
api_key_env = "MY_LLM_TOKEN"
temperature = 0.0
output_mode = "tool"
```

Add the corresponding credential to `.env`:

```dotenv
MY_LLM_TOKEN=your_api_key
```

Then run either walkthrough as before. Both examples explicitly select `jades.toml` and `.env`. Existing process-level `JADES_*` environment variables take precedence over file settings.

Your service must provide an OpenAI-compatible Chat Completions endpoint. The default `tool` output mode requires tool calling and `tool_choice="required"`. Depending on your service, you can explicitly select `output_mode="json_schema"`, `"json_object"`, or `"text"`. JADES does not automatically switch models or output protocols.

Local models work through the same interface: start a compatible local inference server, then configure its model name and endpoint, such as `base_url="http://127.0.0.1:8000/v1"`. If the server does not require authentication, the configured credential variable still needs a nonempty placeholder value.

### Use different models for different modules

For example, append these sections to `jades.toml` to choose separate cleaning and scoring models:

```toml
[modules.clean]
model = "your-cleaning-model"

[modules.judge]
model = "your-scoring-model"
```

Unspecified endpoint and credential settings inherit from `[llm]`. Each module can also override `base_url`, `api_key_env`, and supported request parameters.

Available modules are `clean`, `decompose`, `pair`, `judge`, `overall`, `fact_decompose`, `fact_clarify`, and `fact_check`. Overall model evaluation and fact checking are disabled by default; setting their model names alone does not enable those features.

After editing the configuration, rerun the CLI command or create a new `Evaluator` / `AsyncEvaluator` instance in Python.

## Learn more

- [Time and token metrics](https://github.com/TrustAIRLab/JADES_Eval/blob/main/docs/metrics.md)
- [Refusal warnings and failure handling](https://github.com/TrustAIRLab/JADES_Eval/blob/main/docs/refusal-checks.md)
- [Optional reproducibility details](https://github.com/TrustAIRLab/JADES_Eval/blob/main/docs/fingerprint-display.md)
- [Scoring behavior](https://github.com/TrustAIRLab/JADES_Eval/blob/main/docs/compatibility.md)
- [Report an issue](https://github.com/TrustAIRLab/JADES_Eval/issues)
