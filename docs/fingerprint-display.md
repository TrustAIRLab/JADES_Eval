# 指纹信息的显示开关（0.1.2）

默认对外输出隐藏系统生成的以下内容：

- 单条结果 metadata 中的 `prompt_fingerprint`、`config_fingerprint`。
- 整批结果及汇总中的 `signature`。
- `legacy_prompt_fingerprint_incomplete` 等指纹相关兼容性详情。

分数、评分理由、模型配置、耗时/token 统计及模型/资源版本等信息照常输出。输入数据自身的同名字段不会被修改。

## CLI

```bash
# 默认隐藏
jades evaluate --input samples.json --output results.json

# 显式显示
jades evaluate --input samples.json --output results.json --include-fingerprints

# 对已有完成结果切换显示；成功样本不会重跑
jades evaluate --input samples.json --output results.json --resume --include-fingerprints
```

显式开启后，现代结果 JSON、汇总 JSON 和终端会显示相关信息。旧格式 `--legacy` 的结果本体保持兼容，指纹详情放在汇总文件中。

恢复任务仍按原规则处理失败/未完成样本；显示开关本身不产生模型调用，也不会让已完成样本失效。

## Python

```python
result = evaluator.evaluate(question, response)  # metadata / model_dump 默认不含指纹
public_data = result.to_dict()
debug_data = result.to_dict(include_fingerprints=True)  # 仅序列化，无额外 LLM 调用

# 或在评估调用中显式开启：
result = evaluator.evaluate(question, response, include_fingerprints=True)
```

`AsyncEvaluator.aevaluate()`、`evaluate_many()` 和 `aevaluate_many()` 同样支持 `include_fingerprints=True`。

## 内部数据与校验

Checkpoint 是内部恢复文件，始终保存完整签名、单条结果的指纹和兼容性记录。默认公开结果只是投影，不删除或修改 checkpoint 的校验依据。显示选项不进入 Config、缓存键或批次签名，所以切换显示不会改变恢复兼容性。

实际输入/配置/模板不匹配时，恢复仍会被拒绝并给出错误说明。用量历史不完整等 accounting warnings 也不会被这个选项隐藏。
