# 采样与思考

[English](../../configuration/sampling.md)

请求自己的采样字段优先。请求没有给出的字段取同名服务器选项，然后取模型文件的 `sampling` 默认值，最后取引擎的。
`/props` 报告当前生效的值。超出范围的取值会得到 400，并指出是哪个字段。

| 字段 | 类型、取值 | 作用 |
| --- | --- | --- |
| `temperature` | 数字 >= 0 | `0` 是贪婪。贪婪请求在其他任何设置下都保持精确。 |
| `top_p` | (0, 1] | 保留累积概率达到该值的最小集合。`1` 关闭。 |
| `top_k` | 整数 >= 0 | 保留最可能的 K 个 token。`0` 或 `-1` 关闭。 |
| `min_p` | [0, 1] | 丢掉概率低于最高 token 概率乘以 `min_p` 的 token。 |
| `repetition_penalty` | > 0 | 对窗口内 token 的乘性惩罚。`1` 关闭。 |
| `presence_penalty`、`frequency_penalty` | 数字 | 定额惩罚与按次数计数的惩罚。`0` 关闭。 |
| `penalty_last_n` | 0..4096 | 惩罚的窗口（默认 64）。 |
| `seed` | 整数 | 固定抽样，得到可复现的采样回复。 |
| `relaxed_drafting` | [0, 1] 或 `false` | 该请求的宽松 drafting 阈值。`false` 或 `0` 强制精确采样。 |

同时会读取：`max_tokens` / `max_completion_tokens`（`0` 或 `-1` 表示用到上下文剩余部分）、`stop`、`logit_bias`、
`reasoning_effort`、`enable_thinking`、`reasoning_budget_tokens`（`0` 表示不限思考预算）。Anthropic 的
`thinking.budget_tokens` 是同一个预算。

## 思考

- **预算：** 开启 `answer-room` 且上限至少 1,024 token 时，思考被限制在上限减去 `min(max(512, cap // 4), cap // 2)`。
  8,192 token 的上限允许 6,144 个思考 token，剩下 2,048 个用于收束和答案。预留的空间不保证答案完整。

## 思考循环

- **循环：** 当同一个 token 在思考中连续重复 256 次时，服务器收束思考并让模型作答（`thinking-loop-tokens`）。
  只监视思考部分。`finish_reason` 保持如实。

## 模型文件中的默认值

模型 GGUF 元数据里的 `general.sampling.temp`、`top_p`、`top_k`、`min_p`、`penalty_repeat` 和 `penalty_last_n` 是对应字段的
默认值。取值不合法会让启动停止。生效的默认值会打印在模型那一行（`temp=1.0 top_p=0.95 ...`）。
