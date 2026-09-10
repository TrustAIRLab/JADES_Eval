# 0.1.1 — 审核问题修复与兼容性

> 0.1.2 仅调整指纹信息的默认显示：下文涉及的签名、指纹及相关兼容性详情仍保存在内部 checkpoint，对外默认隐藏，需 `--include-fingerprints` 才显示。恢复校验及本页修复保持有效。

本次针对 Fable 5.1 审核及本地复现中的结果完整性问题。没有修改业务提示词、分句算法、拒答阈值、有限分数的加权公式，也没有把文本拒绝告警重新改成失败。

## 已修复

1. **同输出并发覆盖**：对输出持有操作系统进程锁，覆盖读取、评估、checkpoint 写入和最后统计全过程。第二个写入者立即报错；进程退出后锁由 OS 释放。锁文件不删除，避免已锁 inode 被替换。仍然按样本重写整份 checkpoint，没有改成新的增量存储架构。
2. **SDK 参数绕过**：同时检查 `parameters` 和 `extra_body` 中受保护的路由、消息、输出协议及传输字段。供应商的合法扩展（如 thinking）保留。记录配置模型和服务返回模型，便于追溯别名。
3. **非有限数值中止整批**：输出模型拒绝 NaN/Inf，沿用既有格式修复次数；汇总产生的数值溢出也作为样本错误。持久化前检查结果是否可编码成有限 UTF-8 JSON，单条非法结果不再进入 checkpoint。不会伪造零分。错误后的清理失败不再遮蔽原始异常。
4. **提示词指纹遗漏**：除系统提示词外，纳入实际用户消息构造函数、相关核心节点模板、节点绑定的提示词和发送给模型的输出 schema。按 AST 归一化源码，不把安装绝对路径或源文件行号作为模板身份。缓存键也纳入问题分解模板和 schema。未改变的定义复用已计算的指纹。
5. **sidecar 冲突**：新命名保留完整输出文件名。`report.json` 与 `report.jsonl` 的 checkpoint/metrics/summary 各自独立。拒绝把输入文件当输出覆盖，也拒绝直接把保留的 sidecar 名作为输出。
6. **恢复后历史用量变零**：checkpoint 保存请求账本，恢复时与日志按 request ID 合并。日志丢失、损坏、部分尾行或未完成的历史运行均有明确的不完整性标记。保留已知用量，不再把缺失历史声明成完整的零。
7. **隐式配置改变凭据去向**：仍自动加载默认 HF_TOKEN，但隐式 cwd 文件不能改变有效 base_url/api_key_env。自定义连接需要显式可信配置文件、显式 env 文件、进程 JADES_* 变量或 Python 覆盖。仅向客户端提供实际引用的凭据变量及搜索凭据。CLI 显示生效来源、模型与端点，不打印凭据值。
8. **恢复读取编码**：checkpoint 明确使用 UTF-8；日志尾部的截断 UTF-8 可在持锁状态下截断到完整记录，并标明历史不完整。

## 当前版本恢复

输出 `results.json` 对应：

- `results.json.checkpoint.json`
- `results.json.metrics.jsonl`
- `results.json.summary.json`

```bash
jades evaluate --input samples.json --output results.json --resume
```

保持输入、配置和模板一致。仍然是样本级恢复：已保存成功样本跳过，失败/中断样本重跑。文本告警样本仍算成功，不会因为告警而重复调用。

## 导入 0.1.0 checkpoint

旧 checkpoint 没有保存完整模板指纹，无法自动证明历史模板一致。需明确指定旧文件并选择**新的输出文件名**：

```bash
jades evaluate --input samples.json --output resumed.json --resume \
  --legacy-checkpoint results.checkpoint.json
```

输入和配置仍须匹配旧签名。旧文件不删除、不覆盖；日志复制到新文件族，已保存的成功结果保留。新结果/summary 会记录 `legacy_prompt_fingerprint_incomplete`，不会把历史缺失的信息冒充为已验证。

以后继续使用 `--output resumed.json --resume`。导入前停止旧版运行进程：旧版程序不遵守新锁协议。新锁保护的是遵守协议的本地运行实例，不是对恶意文件系统写入者的隔离机制。

## 自定义端点

```bash
jades evaluate --config jades.toml --input samples.json --output results.json
# 如果连接路由写在 .env 的 JADES_* 变量里：
jades evaluate --env-file .env --input samples.json --output results.json
```

只有凭据值写在 `.env`、且仍使用默认 HF 连接时，不需要增加参数。显式指定的配置仍是可信输入；不要把不可信配置变成显式可信来源。该保护聚焦凭据路由，并不意味着任意恶意配置中的模型或功能设置都安全。

## 保持原策略的部分

- 文本规则命中只做 warning，不中止、不改分、不新增模型请求。
- `message.refusal` 与 `content_filter` 仍停止该样本。
- 搜索失败仍为 unknown 并保留 error 统计，没有擅自改成新的评分扣分规则。
- 没有一律删除 `reasoning_content`；不同供应商的协议需求仍需保留。
- 修改并发/超时等配置后仍拒绝普通恢复，保留已文档化的严格配置匹配策略。

验证覆盖跨进程互斥与进程死亡后的释放、NaN/Inf/溢出隔离、合法 vendor 参数、隐式/显式配置边界、丢失与损坏日志恢复、提示词变更、文件名隔离、UTF-8 尾部修复和旧 checkpoint 导入。所有回归测试使用本地 mock，不调用真实模型或搜索服务。

本轮 **248 项测试通过**。此外，对之前保存的六条真实研究样本完成旧 checkpoint 导入和再次恢复：跳过六条已成功样本，恢复已知历史用量 **181,637 tokens**，新增调用与 token 为零，原结果/checkpoint/metrics 文件 SHA-256 均未变化。该验证保留 `legacy_prompt_fingerprint_incomplete` 标记，不把旧文件缺失的模板信息补写成已验证。

原审核复现脚本保留为历史证据；修复后的行为应使用 `tests/test_hardening_config.py`、`tests/test_hardening_batch.py` 及完整回归套件验证，而不是要求旧漏洞继续复现。
