# Adamant 与 Strata 在同一块 RTX 4090 上

[English](../../benchmarks/strata-comparison.md)

开发性对比，记录于 2026-11-05，在一块功耗限制 450 W 的 RTX 4090（24 GB）、64 GB 系统 RAM 和 Intel Core i7-13700K 上进行。它比较的是部署状态的 Adamant 与 Strata 配置，两者使用同一个 UD-IQ4_XS 源模型家族。这是单机运行，不是通用的性能或质量结论。

## 结果

每个单元格是三次运行的中位数，括号内为观测到的最小值和最大值。每次请求生成 512 个 token，返回 HTTP 200，在 token 上限处结束，并报告缓存提示词 token 为零。各 API 封装之间的提示词长度有几个 token 的差异。Adamant 的 prefill 速率和 decode 速率是服务器日志中的四舍五入值；Strata 的数值来自其引擎状态计数器。墙钟时间由客户端测量，包含请求开销。它不是引擎的 decode 时间，也不是首 token 时间。

| 上下文 | Fixture | Adamant prefill tok/s | Strata prefill tok/s | Adamant decode tok/s | Strata decode tok/s | Adamant 墙钟 s | Strata 墙钟 s |
|---|---|---:|---:|---:|---:|---:|---:|
| 4K | 散文 | 2,927 (2,798–2,930) | 1,740 (1,659–1,746) | 160.1 (153.5–162.8) | 115.9 (110.7–119.0) | 4.6 (4.6–4.7) | 6.8 (6.8–7.0) |
| 4K | 代码 | 2,768 (2,747–2,777) | 1,740 (1,738–1,740) | 165.4 (164.2–167.9) | 132.8 (132.3–141.1) | 4.6 (4.5–4.6) | 6.2 (6.0–6.2) |
| 4K | 结构化 | 2,772 (2,689–2,877) | 1,740 (1,740–1,743) | 145.9 (144.0–152.3) | 108.2 (106.7–113.6) | 5.0 (4.8–5.0) | 7.1 (6.9–7.2) |
| 64K | 散文 | 6,153 (5,902–6,157) | 3,558 (3,413–3,560) | 144.9 (134.4–146.8) | 114.7 (113.3–117.5) | 14.0 (13.9–14.7) | 22.5 (22.4–23.4) |
| 64K | 代码 | 6,149 (6,145–6,152) | 3,557 (3,529–3,560) | 161.2 (160.1–165.4) | 136.5 (121.6–136.9) | 13.6 (13.6–13.7) | 21.8 (21.8–22.5) |
| 64K | 结构化 | 6,155 (6,151–6,156) | 3,556 (3,536–3,558) | 136.0 (135.6–148.6) | 109.1 (109.0–113.5) | 14.2 (13.9–14.2) | 22.8 (22.6–22.9) |

这些结果显示，在这台机器上全部六个受测 fixture/上下文组合中，Adamant 的实测 prefill 和 decode 速率都更高。本对比不测量质量，也不要求输出一致。三次重复只能粗略反映运行间波动；范围以及单独的对照运行见[原始测量数据](../../benchmarks/strata-comparison-data.csv)。

## 方法与配置

两个引擎先后通过同一个本地 llama-swap OpenAI 端点提供服务。GPU 功耗限制保持 450 W。每个引擎先收到一次加载/预热请求（不计入表格），然后是 18 次计时请求：散文、代码和结构化输出 fixture，两种提示词长度，各重复三次。确切的任务模板未保留。Adamant 先跑；Strata 第二。两个引擎没有按请求交替。两次额外的同任务对照在主测项之后运行，单独收录在 CSV 中，不与测量重复次数合并。没有并发的 benchmark 请求。对比结束后服务器已切回 Adamant。

提示词由检入仓库的 `README.md`、`CONTRIBUTING.md`、`docs/precision/*` 和 `docs/running/*` 拼接而成，带有 `Source section:` 标记。4K 语料为 14,683 个字符（SHA-256
`4005a780cd1d48665b4434e77e442e00ee7f13651488a204fea2894ec9d2a4af`）。64K 语料是对应更长拼接文本的前 240,000 个字符（SHA-256
`cbf1d1480883aeadc21c22f72072a7c8237e6dd03f720a7b6372fb9935a6375d`）。用例文本为 `Benchmark case {nonce}. {task}\n\n` 后接语料；
nonce 为 `nonce-20261105-{engine}-{depth}-{fixture}-r{rep}`。各 fixture 的任务：

- 散文：“Summarize the main technical ideas and caveats in this material in a concise, coherent explanation.”
- 代码：“From the material, write a Python function that validates a JSON object with required fields and returns clear errors. Include type hints and a short example.”
- 结构化：“Return a JSON object with keys `summary`, `key_points` (array), and `caveats` (array), based on the material. Output valid JSON only.”

测试框架记录了载荷哈希。有效输入约为 4.1K 和 64.2K token；精确的 API 计数在 CSV 中。生成设置为 temperature 0、top-p 1、seed 1、不启用 thinking，上限 512 个 token。两个引擎在每个计时请求上都报告缓存提示词 token 为零。

| | Adamant | Strata |
|---|---|---|
| 来源 | `4a0d140e0953` (0.2.1) | `ffa80d73133e` (0.2.10；构建版本 0.1.56) |
| 引擎二进制 SHA-256 | `5ef85c26af2b…a9f2defc` | `0c143e5281ba…241e91eb23` |
| 运行时 | 自带 MTP，spec 4，PCIe 平衡关 | DFlash2 draft，spec 最低 0.5，adaptive 每 1，PCIe 平衡开 |
| 共享设置 | UD-IQ4_XS 源家族，BF16 PLE，上下文 262144，int8 KV，自动专家/prefill 放置，GPU 0 | 相同的源家族与共享设置 |

源输入通过比对大小和首尾各 64 MiB 的指纹加以核对。Adamant 运行的是转换后的 `.basalt` 文件；Strata 读取原始的三个 GGUF 分片。共享的 BF16 PLE 数据为 102,400,495,616 字节。两个引擎并未使用相同的活跃张量表示或精度路径：Strata 的一些 BF16 张量由量化源张量展开得到，而 Adamant 使用原生表示。draft 方法与 PCIe 调度也不同。这是部署配置之间的对比，不是受控的 kernel 级对比；不能由此得出精度一致或输出质量的结论。

Strata 的运行在深层 prefill 期间遇到系统可用内存偏低（采样区间内最小可用内存约 237 MiB，swap 使用 503 MiB）。这可能影响计时。加载和预热时间不计入请求计时；一次 Strata 加载/预热约持续 70 秒。单独的 4K 散文对照在 Strata 上测得 104.4 tok/s，在 Adamant 上 160.2 tok/s；64K 代码对照分别测得 123.7 和 169.6 tok/s。这些对照在主测项之后、一次 Adamant 重载之后运行，因此能反映计时波动，但不用于修正中位数。
