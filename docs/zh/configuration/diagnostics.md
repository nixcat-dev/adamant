# 诊断与测量标志

[English](../../configuration/diagnostics.md)

用于测量、调试和测试引擎的标志。它们不是设置：每一项都默认关闭，[配置参考](index.md)里列的是设置。

二进制一栏说明标志由谁接收：

- **CLI**：不带 `--serve` 的 `bin/basalt`，即一次性模式：`bin/basalt --model FILE.basalt --tokens IDS [flags]`
  读取提示词、生成，然后退出。
- **serve**：`bin/basalt --serve`，即服务器的引擎；这些标志放在配置文件的 `engine` 块里，或像任何引擎标志一样放在服务器的命令行上。
- **engine**：两种模式都适用。
- **bench**：`bin/basalt-bench`。传给它的其他每个 `--flag` 都是引擎标志，原样转传。

## 一次性生成

| 标志 | 二进制 | 取值、默认 | 作用 |
| --- | --- | --- | --- |
| `--tokens` | CLI | 逗号分隔的 token id | 提示词，已预分词。必须给出它或 `--tokens-file`。 |
| `--tokens-file` | CLI | 路径 | 从文件读取提示词，id 以逗号或空白分隔。 |
| `--max-new` | CLI | 整数，16 | 要生成的 token 数。 |
| `--greedy` | CLI | 开关，开 | argmax 解码。 |
| `--seed` | CLI | 整数 | 改为采样，使用这个 Philox seed，在 `--top-k`（20）、`--top-p`（0.95）和 `--temperature`（1.0）之下。 |
| `--stop-eos` | CLI | 开关，关 | 遇到轮次结束 token 时停止。`--serve` 总是在那里停止。 |
| `--eos-ids` | engine | id 列表，`248044,248046` | 轮次结束 token；隐含 `--stop-eos`。服务器期待默认值。 |
| `--gpu-plan` | engine | 开关 | 把 `--gpu` 的选择作为一行 JSON 打印后退出；无需模型。服务器会调用它。 |

## 转储

全部为 CLI，全部默认关闭。

| 标志 | 取值 | 写入内容 |
| --- | --- | --- |
| `--dump-residual` | 路径 | 运行结束处的残差（hc x n_embd，f32），用于对输出头做二分排查。 |
| `--dump-mixed` | 路径 | 运行结束处输出头读取的注意力后残差（n_embd，f32）。 |
| `--dump-routing` | 路径 | 每层每个位置的被路由专家 id 与权重。不能与 `--no-pool` 同用。 |
| `--dump-final-r` | 路径 | 每个位置之后，最后一层之后的残差（hc x n_embd，f32；drafter 的输入）。仅 token 路径。 |

## 计时与 profile

| 标志 | 二进制 | 作用 |
| --- | --- | --- |
| `--stats` | CLI | 打印逐 token 墙钟时间、其宿主阶段和专家 pool 的分解。 |
| `--no-pool` | CLI | 跳过 CPU 专家 pool：仅 GPU 的下限，不含由 CPU 计算的专家。 |
| `--no-capture` | CLI | 直接逐层运行，而不是重放图。需要 `--no-pool`。 |
| `--stage-timing` | CLI | 按 kernel 数量给出各阶段占比，不是时间 profile（未捕获的区间包含宿主空隙）。需要 `--no-capture`。 |
| `--graph-only` | CLI | 只重放每层的 `pre` 图并对其计时。不含 `post` 图和输出头，所以它不是 GPU 下限。 |
| `--gpu-only-full` | CLI | 在没有 pool 的情况下重放每层的全部图和输出头：逐 token 的 GPU 下限。不能与层分割同用。 |
| `--gpu-stages` | CLI | 把每层捕获为三个图（mixer、FFN 与 router、post），在捕获之外对其计时，打印后退出。不能与层分割同用。 |
| `--profile` | serve | 逐请求报告 decode 窗口的 GPU 各阶段时间。会关闭 programmatic dependent launch，因此不适合 A/B 计时。 |
| `--wait-stats` | serve | decode 循环中的宿主等待时间，每 200 个窗口报告一次。 |
| `--decode-timing` | serve | 每个请求一行：其 decode 时间花在了哪里。 |
| `--draft-log` | serve | 路径。每个验证窗口一行（宽度、接受数、时间），供研究 draft 策略。 |

## 投机解码测试

| 标志 | 二进制 | 取值 | 作用 |
| --- | --- | --- | --- |
| `--spec-oracle` | CLI | 路径 | 在没有 drafter 的 pack 上配合 `--spec N`：从 token 文件（期望的后续文本）产生草稿，这是验证路径的精确性测试。提示词查找窗口仍然优先。 |
| `--prefill-until` | CLI | token 数，0 | 只对提示词的前 P 个位置做批处理，其余走 token 路径，采用 teacher forcing，以便后面的位置可与 oracle 对照评分。`0`：整个提示词，除最后一个位置。 |
| `--spec-corrupt` | CLI | 整数，0 | 把每第 N 个草稿替换成错误 token，用于测试拒绝。 |

## 调度测试

| 标志 | 二进制 | 作用 |
| --- | --- | --- |
| `--force-zero-doorbell` | serve | 无论驻留情况如何，让每个 decode 窗口（单流与批处理）都跑在全驻留 pack 使用的 zero-doorbell 图上。不在 VRAM 的被路由专家会缺失，所以输出是错的：它测试的是窗口在装不下所有专家的卡上的调度。 |

## n-gram 表的读取

每层的 n-gram（PLE）表留在磁盘上，按需读取其中的行。默认从模型文件读取，除非 `--ple-gguf` 指定了另一个携带该表的 GGUF，两个文件都可以只含该表或与其他张量混在一起。行采用文件中有的格式，`IQ4_NL`、`Q5_0`、`Q8_0` 或 `BF16`：每行 90 B、110 B、170 B 或 320 B。

| 标志 | 二进制 | 取值、默认 | 作用 |
| --- | --- | --- | --- |
| `--ple-gguf` | engine | 路径，默认为模型文件 | 改为从该 GGUF 读取表。 |
| `--ple-io` | engine | `direct`、`ram`；默认 `direct` | `direct`：无缓冲读取，表完全不进入 RAM 或文件缓存。`ram`：映射文件，启动时把整张表锁定在 RAM。 |
| `--ple-row-cache` | engine | 行数，1048576 | 已取行的有界缓存，每行一份字节数；`0` 关闭。 |
| `--ple-inflight` | engine | 整数，256 | 在途读取数。 |
| `--ple-delay-us` | engine | 微秒，0 | 故障注入：每次行读取不早于该时长完成。 |

## 质量关卡

用 `basalt-bench --quality` 运行；[质量的测量方法](../precision/quality.md)给出方法。

| 标志或键 | 二进制 | 取值、默认 | 作用 |
| --- | --- | --- | --- |
| `--quality` | bench | chunk 数，20 | 通过 decode 路径运行 PPL/KLD/top-1 关卡。单独使用时只跑关卡；与 `-p` 或 `-t` 同用时在其之后运行。 |
| `--kld-base` | bench | 路径 | llama.cpp 的 KL 散度基线文件。 |
| `--quality-t` | bench | token 数，3 | 关卡中每个验证窗口的 token 数。 |
| `--ppl-base` | serve | 路径 | `ppl=` 请求评分所对照的基线文件。 |
| `--ppl-out` | serve | 路径 | `ppl=` 请求把逐行结果写到何处。 |
| `ppl=K` | 请求键 | chunk 索引 | 对 `--ppl-base` 的第 K 个 chunk 评分，该 chunk 其余部分经验证窗口 teacher forcing。 |
| `ppl_t=T` | 请求键 | token 数，3 | `ppl=` 请求的窗口宽度。 |
| `ppl_score=0` | 请求键 | 整数，1 | 运行同样固定宽度的 teacher-forcing 窗口，但不复制 logits、不给行评分、也不写入 `--ppl-out`。 |
| `ppl_force=1` | 请求键 | 整数，0 | 使用自然的草稿窗口选择，并对照存储的后续文本评判每个窗口。原样输出该后续文本；语法和 relaxed drafting 不改变重放。 |

请求键放在引擎 stdin 协议的 `GEN` 行上，位于 token id 之前。

## basalt-bench

`basalt-bench --help` 给出完整说明和 fixture 的制作方法；社区预设（`--community`、`--submitter`、`--community-file`）见[构建](../building.md)。

| 标志 | 取值、默认 | 作用 |
| --- | --- | --- |
| `-t` | `conv,struct,prose` 的任意组合 | decode fixture，总是按该顺序运行（draft 状态跨请求延续）。 |
| `-p` | 以 token 计的深度（接受 `32k` 风格写法） | prefill 深度。没有 `-t` 或 `-p` 时，bench 运行 `-p 16384 -t conv,struct,prose`。 |
| `-r` | 整数，1 | 整套的重复次数；大于 1 时表格给出均值和标准差，第一次重复是冷启动。 |
| `--seed` | 整数 >= 1 | 为每个 decode 请求设种子，使采样 fixture 每次运行产生相同文本。 |
| `--model`、`--gpu`、`--spec` | 同引擎 | 传给引擎；`--spec 0` 是朴素 decode 基线。 |
| `--set` | `"FLAG VALUE ..."` | 替换或添加引擎参数；可重复。`--` 之后的参数作用相同。 |
| `--engine` | 路径 | 引擎二进制；默认为与 bench 一起构建的那个。 |
| `--fixtures` | 目录，`data/bench` | 预分词提示词所在之处。 |
| `-o` | `md`、`json`；默认 `md` | markdown 表格，或把完整细节以 JSON 输出到 stdout。 |
| `--power-label` | 瓦特 | 打印此功耗限制，代替从 GPU 读到的那个。 |
| `--logs` | 目录 | 把引擎日志、draft 日志和质量关卡结果保存在那里。 |
| `--no-windows` | 开关 | 不向引擎请求逐窗口日志；没有窗口统计。 |
| `--print-command` | 开关 | 打印引擎命令后退出。 |
| `--hardware` | 开关 | 测量并打印 CPU、RAM 读取速度以及每块 GPU 的 PCIe 链路与拷贝速度，然后退出；无需模型。 |

`basalt-device` 是构建目录里的一个辅助程序，接受 `--list-devices`（运行时看到的每块 GPU 以及本构建能否在其上运行）、`--link-rates`（每块 GPU 一行 `pcie_h2d BUS_ID GB_PER_S`，由 `--hardware` 读取）和 `--selftest`；不带标志时打印主 GPU 和内存计划。

## 实验键（`--exp`）

`--exp NAME=VALUE[,NAME=VALUE...]`（engine）设置实验开关。它们的存在是为了测试某个改动，不承诺稳定性，实验结束即移除。引擎不读取的键会被忽略；`--exp` 可以给多次。配置文件中写作：`"engine": { "exp": "host_ev=1" }`。

| 键 | 默认 | 作用 |
| --- | --- | --- |
| `concur9_logit_hash` | 0 | `1`：在 stderr 打印每个被接受行的输出头 logits 的哈希（单流与批处理槽位）。 |
| `concur5_digest` | 0 | `1`：批处理槽位在每个窗口后于 stdout 打印其状态的 `BDIG` 摘要，且窗口不重叠。 |
| `spec_log` | 0 | `1` 且配合 `--draft-log FILE`：写 `FILE.tk`，每个被验证行的目标分布排在 drafter 的 top 4 旁边。 |
| `host_ev` | 0 | `1`：在 decode 循环的图启动处记录事件；之后每个请求会打印一个窗口的 verify、commit 和 draft 的 GPU 时间，以及下一个窗口前的空闲间隙（中位数与均值）。 |
