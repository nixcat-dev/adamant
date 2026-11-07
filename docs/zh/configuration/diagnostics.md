# 诊断与测量标志

[English](../../configuration/diagnostics.md)

用于测量、调试和测试引擎的标志。全部默认关闭，都不是设置项。
所属程序：**CLI** = `bin/basalt` 一次性模式（`--tokens IDS`），**serve** = `--serve` 引擎，**bench** = `basalt-bench`。

## 一次性生成

| 标志 | 所属程序 | 默认值 | 作用 |
|---|---|---|---|
| `--tokens`、`--tokens-file` | CLI | 必填 | 以 token id 给出的提示词。 |
| `--max-new` | CLI | 16 | 生成的 token 数。 |
| `--greedy` | CLI | 开启 | Argmax 解码。 |
| `--seed` | CLI | | 在该种子下以 `--top-k` 20、`--top-p` 0.95、`--temperature` 1.0 采样。 |
| `--stop-eos` | CLI | 关闭 | 遇到轮次结束 token 时停止。 |
| `--eos-ids` | engine | `248044,248046` | 轮次结束的 token。 |
| `--gpu-plan` | engine | | 以 JSON 打印 `--gpu` 的选择并退出。 |

## 计时与 profile

| 标志 | 所属程序 | 作用 |
|---|---|---|
| `--stats` | CLI | 每 token 的实际耗时、主机各阶段和 CPU 池的细分。 |
| `--no-pool` | CLI | 跳过 CPU 专家池：纯 GPU 下限。 |
| `--no-capture` | CLI | 直接运行各层，不用图捕获。需要 `--no-pool`。 |
| `--stage-timing` | CLI | 按 kernel 数给出各阶段占比。需要 `--no-capture`。 |
| `--graph-only` | CLI | 只给每层的 `pre` 图计时。 |
| `--gpu-only-full` | CLI | 重放每个图和 head，不用池：每 token 的 GPU 下限。不与层拆分同用。 |
| `--gpu-stages` | CLI | 给每层的三个图计时，打印后退出。不与层拆分同用。 |
| `--profile` | serve | 每个请求的 GPU 阶段时间。不用于 A/B 计时。 |
| `--wait-stats` | serve | decode 循环中主机等待的时间。 |
| `--decode-timing` | serve | 每个请求一行，说明 decode 时间花在哪。 |
| `--draft-log FILE` | serve | 每个校验窗口一行（宽度、接受数、时间）。 |

## 转储与投机测试

| 标志 | 所属程序 | 作用 |
|---|---|---|
| `--dump-residual`、`--dump-mixed`、`--dump-final-r` | CLI | 把残差或 head 输入写成 f32，用于二分定位。 |
| `--dump-routing` | CLI | 每层每个位置的路由专家 id 和权重。 |
| `--spec-oracle FILE` | CLI | pack 没有 drafter 时配合 `--spec N`：从预期的续写生成 draft。校验精确性测试。 |
| `--prefill-until P` | CLI | 把前 P 个提示词位置按批处理；其余以 teacher-forced 方式运行。 |
| `--spec-corrupt N` | CLI | 把每第 N 个 draft 换成错误的 token，用于测试拒绝。 |
| `--diag-teacher` | CLI | 配合 `--spec-oracle`：teacher-force 文件里的 token 并统计不同的选择。 |
| `--diag-tier-frozen` | CLI, serve | 固定放置：VRAM 层保持它持有的东西，不做搬运或提升。 |
| `--diag-drafter-gpu DEV` | CLI, serve | drafter 运行在哪张卡上。 |
| `--force-zero-doorbell` | serve | 在装不下每个专家的卡上做调度测试。输出按设计就是错的。 |

## 专家与 n-gram 表

逐层 n-gram（PLE）表留在磁盘上，按需读取它的行。除非 `--ple-gguf` 指向另一个携带该表的 GGUF，否则从模型文件读取；这两个文件
都可以只含该表，也可以与其他 tensor 混在一起。行的类型是文件中实际具有的 `IQ4_NL`、`Q5_0`、`Q8_0` 或 `BF16`：每行分别为
90 B、110 B、170 B 或 320 B。

| 标志 | 所属程序 | 取值、默认值 | 作用 |
| --- | --- | --- | --- |
| `--ple-gguf` | engine | 路径，默认为模型文件 | 从这个 GGUF 而不是模型文件读取该表。 |
| `--ple-io` | engine | `direct`、`ram`；默认 `direct` | `direct`：不带缓冲读取，表不进入 RAM 或文件缓存。`ram`：把文件连同整张表映射进 RAM。 |
| `--ple-row-cache` | engine | 行数，1048576 | 已取行的有界缓存，每行占一行自己的字节数；`0` 关闭。 |
| `--ple-inflight` | engine | 整数，256 | 未完成的读取数。 |
| `--ple-delay-us` | engine | 微秒，0 | 故障注入：每次行读取不会早于这个时间完成。 |

## 质量门与 benchmark

| 标志或键 | 所属程序 | 默认值 | 作用 |
|---|---|---|---|
| `--quality` | bench | 20 个 chunk | 通过 decode 路径运行 PPL/KLD/top-1 质量门。[方法](../precision/quality.md) |
| `--kld-base FILE` | bench | | llama.cpp 的 KL 散度基准文件。 |
| `--quality-t N` | bench | 3 | 质量门中每个校验窗口的 token 数。 |
| `--ppl-base`、`--ppl-out` | serve | | `ppl=` 请求使用的基准文件，以及它们写行的位置。 |
| `ppl=K`、`ppl_t=T` | 请求键 | | 给第 K 个 chunk 打分；窗口宽度 T。 |

请求键写在引擎 stdin 协议的 `GEN` 行上，位于 token id 之前。

| bench 标志 | 默认值 | 作用 |
|---|---|---|
| `-t conv,struct,prose` | | decode fixture，按该顺序运行。 |
| `-p DEPTHS` | 没有 `-t` 或 `-p` 时 `-p 16384` | prefill 深度，例如 `32k`。 |
| `-r N` | 1 | 重复次数；大于 1 时表格显示均值与偏差。 |
| `--seed N` | | 给每个 decode 请求设种子。 |
| `--set "FLAG VALUE"` | | 替换或添加引擎参数（可重复）。 |
| `--engine PATH`、`--fixtures DIR` | 构建出的引擎、`data/bench` | 二进制与 fixture。 |
| `-o md` 或 `json` | md | 输出格式。 |
| `--power-label W` | 从 GPU 读取 | 要打印的功耗限制。 |
| `--logs DIR`、`--no-windows`、`--print-command` | | 日志、窗口统计、打印命令。 |
| `--hardware` | | 测量 CPU、RAM 和每块 GPU 的链路，然后退出。 |
| `--community`、`--submitter NAME`、`--community-file PATH` | | 社区预设及它那一行，行里记录模型（[社区实测](../community.md)）。 |

`basalt-device` 是构建目录里的辅助程序，接受 `--list-devices`（运行时可见的每块 GPU，以及本构建能否运行它）、`--link-rates`
（每块 GPU 一行 `pcie_h2d BUS_ID GB_PER_S`，`--hardware` 会读它）和 `--selftest`；不带标志时它打印主 GPU 和内存规划。

## 实验键（`--exp`）

`--exp NAME=VALUE[,...]` 设置实验开关。它们不承诺稳定性，并随其所属实验一起消失。

- `concur9_logit_hash=1`：在 stderr 上打印每个被接受行的 head logits 哈希。
- `concur5_digest=1`：批处理槽位每个窗口打印一次状态摘要。
- `spec_log=1`：配合 `--draft-log` 时，同时写下每行的目标分布。
- `host_ev=1`：打印每个窗口的校验、提交和 draft 的 GPU 时间，以及空闲间隔。
