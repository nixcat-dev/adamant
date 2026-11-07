# 模型

[English](../../models/index.md)

一个模型就是一个 `.basalt` 文件。服务器从文件里读出架构，并挑选 chat 格式、工具语法和思考开关。多个条目可以共用一个
地址（[路由](../configuration/models.md)）。这个模型支持什么：

| | Qwen3.8-Flash-Next |
| --- | --- |
| 架构 id | `qwen4exp` |
| Pack | 9 个 pack，默认 `Qwen3.8-FN-NVFP4-DQ6_K`，[Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) |
| 层、专家 | 48 层（36 层 GDN，12 层 QSA）；每层 512 个专家，top-10 |
| 上下文 | 默认 8,192；用 `--max-context` 设置 |
| Drafting | pack 内的 MTP drafter，默认开启；`--spec` 默认 8 |
| Chat、工具、思考 | 支持；`<tool_call>` 工具 |
| 推理力度 | `none`、`low`、`medium`、`high`、`max`（`high` = `max`）；数字会被拒绝 |
| 图像 | `--vision dynamic`、GPU 编号或 `cpu` |
| `--kv` | fp16（默认）、int8、q4_0、k8v4 |
| 卡 | 主卡加最多三张专家层卡 |
| 层拆分 | 支持，需要匹配的卡，启动时定价 |
| 并行槽位 | 支持 |
| 有损标志 | `--relaxed-drafting`、`--route-prior`、`--route-tail-skip`（关闭） |
| 空闲睡眠与唤醒 | 支持（Linux） |
| `basalt-bench --quality` | 支持 |
| 指南 | [指南](qwen38-flash-next.md) |
| 精度 | [精度](../precision/qwen38-flash-next.md) |

`src/models/builtin.cpp` 里的 adapter 注册表就是引擎能运行的架构清单；模型自己的代码放在 `src/models/<model>` 下
（[新增模型](../adding-a-model.md)）。
