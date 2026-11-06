# 路由改动（需显式开启，精度更低）

[English](../../precision/routing.md)

两个 flag 改变一个 decode 窗口计算哪些专家。二者都默认关闭，从不被自动选中，并列入有损 flag 的启动警告
（与 `--relaxed-drafting` 共用，[features.md](features.md#宽松起草)）：

    basalt: warning: lossy flags enabled, these flags will affect the accuracy of the served model
    basalt: warning: --route-prior 0.5 changes expert routing
    basalt: warning: --route-tail-skip 9 drops low-ranked missed experts

Prompt batch 不受影响。两者都需要设备规划的 MoE 尾段（它关闭时这两个也保持关闭，并留下一行启动信息），
并作用于任何 GPU 层持有的专家。两者都关闭时，输出与没有这些代码的构建逐位相同。误差随装进 VRAM 的模型
部分变少而增大，因为未命中的专家更多。

## Route prior

`--route-prior [LAMBDA]`（不带参数 = 0.5，LAMBDA 在 (0, 16] 内）。decode 窗口选择它的 top-10 专家时，仿佛
GPU 持有的每个专家的 router logit 都加了 LAMBDA。混合权重仍是 router 自己的。LAMBDA 越高，路由越偏向已
持有的专家：未命中更少，改变更多。

## Route tail skip

`--route-tail-skip [R]`（不带参数 = 9，R 在 1..9）。在一个 decode 窗口中，当窗口里提到某个 GPU 都没持有的
专家的每一个条目，在其 token 内按 router 权重都排到第 R 名或更靠后（0 起算，共 10 名）时，这个专家既不拷贝
也不计算。它的贡献为零，其余权重不做重新归一化。R 越低丢弃越多。engine 在退出时记录丢弃了多少条目。思路
来自 Strata 的 route tail skip。

## 数值

对照 BF16 参考基准的质量门槛（[quality.md](quality.md#bf16-参考基准)）：100 个 chunk，T 3，`--gpu 0,1`
（400 W 的 RTX 5090 + 180 W 的 RTX 5060 Ti），int8 KV，`--max-context 8192`，两个 flag 一起设置。Decode 是
`--parallel 1` 下各一次冷启动 prose tok/s，thinking 关闭；decode 运行时开着 `--relaxed-drafting`（1.5 / 7 时
0.3，16 / 1 时 0.01），因为门槛不做采样。

| Pack | route prior / tail skip | PPL | KLD 均值 | KLD p99 | 相同 top-1 | Prose tok/s |
|---|---|---|---|---|---|---|
| IQ3_S | off | 3.382 | 0.1499 | 1.65 | 87.54% | 341.1 |
| IQ3_S | 1.5 / 7 | 3.437 | 0.1828 | 2.10 | 86.55% | 314.1 |
| IQ3_S | 16 / 1 | 3.520 | 0.2166 | 2.58 | 85.47% | |
| NVFP4 | off | 3.304 | 0.0614 | 0.71 | 91.81% | 281.6 |
| NVFP4 | 1.5 / 7 | 3.616 | 0.2240 | 2.63 | 85.00% | 289.0 |
| NVFP4 | 16 / 1 | 4.422 | 0.4800 | 5.70 | 78.72% | 329.0 |
| Q8_0 | off | 3.282 | 0.0197 | 0.24 | 95.39% | 112.4 |
| Q8_0 | 1.5 / 7 | 4.141 | 0.3736 | 4.06 | 80.28% | 219.3 |
| Q8_0 | 16 / 1 | 8.873 | 1.2338 | 9.12 | 62.26% | 274.9 |

质量各列来自比 decode 列更晚的构建，读数低于同样 flag 在更早构建上的表现。flag 代码未变；起始专家放置
（`data/profiles` 中学到的 profile）让留在 VRAM 之外的专家更少，所以这些 flag 能转向或跳过的空间也更小
（NVFP4：这些运行中 1.4% 的路由专家在 CPU 层）。

作为参照，这个门槛上每个精确 pack 都列在
[Hugging Face 页面](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt#quality)。速度来自更少的未命中，
所以它随 VRAM 之外专家的比例增长，误差亦然。不带参数的默认值（0.5 / 9）未在此测量。
