# 采样与思考

[English](../../configuration/sampling.md)

逐请求的采样字段、请求继承的默认值，以及思考如何结束。

请求自带采样字段。请求省略的字段先取同名的启动选项（`--temperature`、`--top-p` 等，见[服务器选项](index.md#服务器)；命令行优先于配置文件的 `server` 块），再取模型文件的 `sampling` 默认值（见下文），最后取引擎默认。请求字段永远优先。`/props` 报告当前生效的默认值。超出范围的取值会被拒绝，返回 400 并点名该字段；范围内的取值原样送达采样器。

| 字段 | 类型、取值 | 作用 |
| --- | --- | --- |
| `temperature` | 数字 >= 0 | `0` 为贪心（argmax）；越高越随机。贪心请求在任何其他设置下都保持精确。 |
| `top_p` | (0, 1] 内的数字 | 核采样：保留概率累加到此值的最小 token 集合。`1` 为关闭。 |
| `top_k` | 整数 >= 0 | 只保留最可能的 K 个 token。`0`（或 `-1`）为关闭：词表中的每个 token。 |
| `min_p` | [0, 1] 内的数字 | 采样截断：丢弃概率低于 `min_p` 乘以最可能 token 概率的 token。`0` 为关闭。它作用于模型的 token。 |
| `repetition_penalty` | 数字 > 0 | 对窗口中出现过的 token 的乘法惩罚；`1` 为关闭。 |
| `presence_penalty` | 数字 | 对窗口中出现过的任何 token 的固定惩罚；`0` 为关闭。 |
| `frequency_penalty` | 数字 | 随某 token 出现次数增长的惩罚；`0` 为关闭。 |
| `penalty_last_n` | 整数 0..4096 | 三项惩罚回望的窗口长度（token 数；`0`：无）。只在设置了某个惩罚时生效；省略时为 64。惩罚在批处理窗口中同样适用；流水线化的批处理组会拒绝它们（400）。 |
| `seed` | 整数 | 固定随机抽取，使采样回复可复现。 |
| `relaxed_drafting` | [0, 1] 内的数字，或 `false` | 本请求的 relaxed-drafting 阈值；`false` 或 `0` 强制精确采样。需要投机解码；见[投机解码](index.md#投机解码-mtp)。 |

API 读取的其他请求字段：`max_tokens` / `max_completion_tokens`（Anthropic 的 `max_tokens`；`0` 或 `-1` 表示上下文的剩余部分）、`stop` / `stop_sequences`、`logit_bias`、`reasoning_effort`、`enable_thinking` 和 `reasoning_budget_tokens`（思考 token 的上限；`0` 为无上限）。Anthropic 的 `thinking.budget_tokens` 和 llama.cpp 的 `thinking_budget_tokens` 是同一个预算，多个同时发送时 `reasoning_budget_tokens` 优先。一个都没给的请求取服务器的 [`reasoning-budget-tokens`](index.md#服务器)。

`answer-room` 开启且有效输出上限不小于 1024 token 时，思考被限制在上限减去 `min(max(512, cap // 4), cap // 2)`。8192 token 的上限最多允许 6144 个思考 token；其余的 2048 个包含结束思考的部分和答案。更小的正数预算保持不变。上下文拟合使用缩减后的上限。请求显式给 `0` 的预算会同时关闭服务器默认值和自动预算；更小的输出上限保持原有行为。预留空间能减少空答案，但不能保证答案完整。

### 思考循环

当模型在思考中连续重复同一个 token 256 次时，服务器用模板自己的结束符（`\n</think>\n\n`）结束思考，让模型作答。回复的 `finish_reason` 保持真实：答案完整结束是 `stop`，只有到 `max_tokens` 才是 `length`。服务器记录 `thinking repeated one token N times, ending it`。只监控思考部分；答案中的分隔符、填充、base64 和数字串绝不会被截断。把 [`thinking-loop-tokens`](index.md#服务器) 设为其他次数，或设为 `0` 关闭该保护。

### 模型文件的采样默认值

模型的 GGUF 元数据（模型文件 `meta` 中的 `gguf_kv`）携带 llama.cpp 的 `general.sampling.*` 键。
服务器把它们作为这些请求字段的默认值：

| 键 | 请求字段 | 允许取值 |
| --- | --- | --- |
| `general.sampling.temp` | `temperature` | 数字 >= 0 |
| `general.sampling.top_p` | `top_p` | 0 < 值 <= 1 |
| `general.sampling.top_k` | `top_k` | 整数 >= 0（`0` 或 `-1` = 关闭） |
| `general.sampling.min_p` | `min_p` | 0 <= 值 <= 1 |
| `general.sampling.penalty_repeat` | `repetition_penalty` | 数字 > 0（`1` = 关闭） |
| `general.sampling.penalty_last_n` | `penalty_last_n` | 整数 0..4096 |

浮点键按具有相同 float32 值的最短十进制读取（0.95，而不是 0.949999988）。

非法值会让服务器在启动时停止；其他 `general.sampling.*` 键被忽略。生效的默认值在启动时以 `[server] sampling ...` 打印，由启动选项设定的会标注 `(cli)` 或 `(config)`，来自模型文件的无标注。启动选项会覆盖模型文件中该键的值；其余保持不变。
