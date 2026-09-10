# JADES Eval

JADES Eval 用于评估模型回答对给定问题或任务的满足程度，适合越狱评估等研究场景。你提供一组 **问题（question）和回答（response）**，JADES 返回评分、评价理由、耗时和 token 用量。

你可以选择两种使用方式：

| 使用方式 | 适合谁 | 从哪里开始 |
|---|---|---|
| **Python** | 希望在自己的脚本、Notebook 或评估流程中调用 | [Python 分步示例](#方式一python-分步示例) |
| **CLI（命令行）** | 已有 JSON/JSONL 数据，希望直接批量评估、保存结果和断点续跑 | [CLI 分步示例](#方式二cli-分步示例) |

[PyPI 安装包](https://pypi.org/project/jades-eval/) · [GitHub Releases](https://github.com/TrustAIRLab/JADES_Eval/releases)

## 开始前：安装并配置凭据

两种使用方式共用以下准备步骤。需要 **Python 3.10 或更新版本**，建议在独立的 Python 虚拟环境中安装。Windows 用户可以使用 PowerShell 执行下列命令。

### 第 1 步：创建工作目录并安装

```bash
mkdir jades-demo
cd jades-demo
python -m pip install --upgrade jades-eval
```

### 第 2 步：生成配置文件

```bash
jades init
```

当前目录会生成：

- `jades.toml`：设置模型、服务地址及评估选项。
- `.env.example`：凭据模板。

### 第 3 步：填写自己的凭据

复制模板：

```bash
cp .env.example .env
```

用文本编辑器打开 `.env`，填写你自己的 Hugging Face token：

```dotenv
HF_TOKEN=替换为你的_Hugging_Face_token
```

默认使用以下模型和服务，不需要额外修改 `jades.toml`：

| 配置 | 默认值 |
|---|---|
| 模型 | `deepseek-ai/DeepSeek-V4-Flash-0731:together` |
| 服务地址 | `https://router.huggingface.co/v1` |
| 凭据变量 | `HF_TOKEN` |

后面的命令和脚本都在 `jades-demo` 目录下运行。凭据保存在 `.env` 中，不要写入 Python 脚本或提交到公开仓库。LLM 调用使用你的账号，产生的用量由服务提供方计费。

首次使用时，JADES 会按需下载并缓存分句和语义检测资源，因此启动可能较慢。也可以在运行示例前预先准备资源：

```bash
jades prepare-resources --config jades.toml --env-file .env
```

## 方式一：Python 分步示例

### 第 1 步：创建 `demo.py`

将下面的完整代码保存为 `demo.py`。这里使用一个简单的地理问答来演示调用方式；实际评估时，把 `question` 和 `response` 换成你的待测样本。

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
    print("评估失败：", error)
    print("失败前已知的 token 消耗：", error.metrics.tokens.known_total_tokens)
    raise SystemExit(1)

print("评分：", result.score)
print(f"耗时：{result.metrics.wall_time_seconds:.2f} 秒")
print("输入 token：", result.metrics.tokens.input_tokens)
print("输出 token：", result.metrics.tokens.output_tokens)
print("总 token：", result.metrics.tokens.total_tokens)

Path("python-result.json").write_text(
    json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("完整结果已保存到 python-result.json")
```

### 第 2 步：运行

```bash
python demo.py
```

### 第 3 步：查看结果

终端会打印本次评分、耗时和 token 用量。运行成功后，还会生成两个文件：

| 文件 | 内容 |
|---|---|
| `python-result.json` | 完整评分、各评分点的理由，以及用量汇总 |
| `python-demo.metrics.jsonl` | 每次请求的耗时、状态和用量记录 |

在 Python 中，也可以直接读取：

```python
# 接在 demo.py 的成功结果之后。
for point in result.state.scoring_points_judgement_for_a_question or []:
    print(point["scoring_point"])
    print("分数：", point["judge_score"])
    print("理由：", point["judge_reason"])
```

分数越高，表示回答对被评估任务的满足程度越高。实际结果取决于样本及评估模型；示例不预设固定分数。token 字段为 `None` 时，表示服务未提供完整用量，不能当作零消耗。

<details>
<summary>可选：在 Python 中一次评估多个样本</summary>

保存为 `batch_demo.py`，然后运行 `python batch_demo.py`：

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
        print(index, "失败：", result)
    else:
        print(index, "评分：", result.score, "总 token：", result.metrics.tokens.total_tokens)
```

返回结果与输入顺序一致。需要自动保存文件和断点续跑时，使用下面的 CLI 方式。

</details>

<details>
<summary>可选：在 Notebook 或异步程序中使用</summary>

在 Notebook 单元格中运行：

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

## 方式二：CLI 分步示例

### 第 1 步：创建输入文件 `samples.json`

将下面的内容保存为 `samples.json`。每条样本都包含 `question` 和 `response`：

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

### 第 2 步：运行评估

```bash
jades evaluate --config jades.toml --env-file .env --input samples.json --output results.json
```

终端会显示成功、失败和跳过的样本数，以及本次运行的耗时和已知 token 用量。

### 第 3 步：打开结果文件

运行后，工作目录中会出现：

| 文件 | 用途 |
|---|---|
| `results.json` | 查看每条样本的评分、理由及处理状态 |
| `results.json.summary.json` | 查看整个批次的耗时、成功/失败数、本次与累计用量 |
| `results.json.metrics.jsonl` | 查看逐次请求记录 |
| `results.json.checkpoint.json` | 保存进度，供断点续跑使用 |

用文本编辑器打开 `results.json`。它的 `results` 列表中，每项对应一条样本：

| 字段 | 含义 |
|---|---|
| `status` | `ok` 表示成功，`error` 表示失败 |
| `result.state.jailbreak_score_weighted` | 该样本的评分 |
| `result.state.scoring_points_judgement_for_a_question` | 各评分点的分数与理由 |
| `result.metrics.wall_time_seconds` | 该样本的耗时 |
| `result.metrics.tokens.total_tokens` | 该样本本次评估的总 token 用量；`null` 表示不完整 |
| `error` | 失败样本的错误说明；失败项不应作为有效评分使用 |

### 第 4 步：中断后继续运行

如果运行中断，保留上述文件，并使用相同的输入、配置和输出路径，追加 `--resume`：

```bash
jades evaluate --config jades.toml --env-file .env --input samples.json --output results.json --resume
```

已成功保存的样本会跳过；失败或中断的样本会重新评估。恢复以**整条样本**为单位，不会从样本中间的某次模型调用继续。中断前已经发出的请求可能产生用量。

如果希望重新评估全部样本，使用新的输出路径即可，例如 `--output results-new.json`。

### 第 5 步：换成自己的数据

把示例中的问题和回答替换为你的数据即可。还支持：

- **JSONL**：每行一个包含 `question`、`response` 的 JSON 对象，输入文件使用 `.jsonl` 后缀。
- **JailbreakBench 格式**：顶层 `jailbreaks` 列表中的样本使用 `goal` 和 `response`。如果提供顶层 `parameters`，它必须是 JSON 对象。
- **截断回答字段**：数据使用 `truncated_response` 时，添加 `--response-field truncated_response`。

更多命令行选项：

```bash
jades evaluate --help
```

## 更换模型：两种方式都适用

你可以让所有模块共用一个模型，也可以为不同模块分别设置模型。

### 所有模块共用一个模型

打开生成的 `jades.toml`，**替换已有的 `[llm]` 段**，不要重复添加同名段。以下值需要换成你服务商提供的实际配置：

```toml
[llm]
model = "你的模型名称"
base_url = "https://your-provider.example/v1"
api_key_env = "MY_LLM_TOKEN"
temperature = 0.0
output_mode = "tool"
```

在 `.env` 中添加对应凭据：

```dotenv
MY_LLM_TOKEN=你的_API_密钥
```

之后仍按上面的 Python 或 CLI 示例运行。示例已显式指定 `jades.toml` 和 `.env`，会加载你选择的配置文件。系统中已设置的 `JADES_*` 环境变量会优先于文件配置。

服务需要提供 OpenAI 兼容的 Chat Completions 接口。默认 `tool` 模式需要支持工具调用和 `tool_choice="required"`；也可按服务支持情况显式设置 `output_mode="json_schema"`、`"json_object"` 或 `"text"`。JADES 不会自动替换模型或输出协议。

本地模型也可以接入：先启动兼容的本地推理服务，再设置它的模型名和地址，例如 `base_url="http://127.0.0.1:8000/v1"`。如果本地服务不要求认证，凭据变量仍需填写一个非空占位值。

### 不同模块使用不同模型

例如，在 `jades.toml` 中追加以下配置，为句子清理和评分分别选择模型：

```toml
[modules.clean]
model = "你的清理模型"

[modules.judge]
model = "你的评分模型"
```

未填写的服务地址和凭据变量会继承 `[llm]` 配置。模块也可以单独设置 `base_url`、`api_key_env` 和调用参数。

可配置模块包括 `clean`、`decompose`、`pair`、`judge`、`overall`、`fact_decompose`、`fact_clarify` 和 `fact_check`。整体模型评价与事实核查默认关闭，仅填写这些模块的模型名不会启用对应功能。

修改配置后，CLI 重新运行即可；Python 用户需要重新创建 `Evaluator` 或 `AsyncEvaluator` 实例。

## 进一步使用

- [耗时和 token 统计说明](docs/metrics.md)
- [拒答提示与失败处理](docs/refusal-checks.md)
- [可选的复现信息输出](docs/fingerprint-display.md)
- [评分行为说明](docs/compatibility.md)
- [问题反馈](https://github.com/TrustAIRLab/JADES_Eval/issues)
