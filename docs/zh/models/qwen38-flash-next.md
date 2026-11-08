# Qwen3.8-Flash-Next

[English](../../models/qwen38-flash-next.md)

用与任何模型相同的命令运行：`basalt serve PACK`。它支持什么：[模型](index.md)。舍入与质量：
[精度](../precision/qwen38-flash-next.md)。

## Pack

[Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) 上的 pack，名字都是 `Qwen3.8-FN-` 加后缀：

- `NVFP4-DQ6_K`（默认）与 `NVFP4-DQ8`：NVIDIA 的 NVFP4 专家。`D` 是稠密投影的格式：Q6_K，DQ8 则是 Q8_0。
  DQ8 比 DQ6_K 慢，KLD 更低（[精度](../precision/qwen38-flash-next.md#6-bit-稠密投影)）。
- `UD-IQ1_S`、`UD-IQ1_M`、`UD-IQ2_XS`、`UD-IQ3_XXS`、`UD-IQ4_XS`、`UD-Q4_K_XL`、`UD-Q5_K_XL`、`UD-Q6_K_XL` 和 `Q8_0`：Unsloth 的 GGUF 类型。
- Unsloth 的 `UD-IQ1_S` GGUF 按原样转换（70.4 GB）。其中一层的 gate 行为 IQ2_S、up 行为 IQ2_XXS；每一层都与其他
  UD pack 跑相同的快速 kernel。

大小以卡上为准。布局以及如何自己构建 pack：[模型文件](../model-file.md)。

## 内存

- **上下文：** `--max-context`，默认 8,192。KV 缓存随它增长。`--kv fp16`（默认）、`int8`（VRAM 减半）、`q4_0`
  和 `k8v4` 用精度换 VRAM（[精度](../precision/qwen38-flash-next.md#实测成本)）。
- **专家：** 主卡的缓存持有被路由最多的那部分，其余在系统 RAM，自适应层随会话变化搬动它们
  （[自适应层](../configuration/adaptive.md)）。
- **RAM：** 在 RAM 副本之前，每个槽位的 checkpoint 块被分配，页锁定提示词缓冲的 RAM 被留出：最大块上的 PLE 行、
  一层拆分的交接、以及提示词拆分的权重、交接和 staging。每个缓冲创建时就占住自己的 RAM；没有缓冲占用的部分在补充
  之前先还回去。副本之后在驱动页跟踪之外留出 `--ram-reserve-mib`（专家留在 SSD 时打印
  `RAM floor X GiB of N GiB free`）。
- **卡：** 主卡之后的每张卡都是专家层，最多三张。匹配的卡可以改为拆分各层（[GPU](../configuration/gpus.md)）。
- **弹性 KV：** `--max-context` 超过 32,768 时（且未使用 `--kv-resident`、`--vram-elastic` 或层拆分），每个会话
  在启动时为它的前 32,768 个 cell 提供后端。请求需要更多 cell 时从专家缓存借，不需要时还回。空闲会话的 KV 在 15 秒
  后移到 RAM。启动行 `elastic KV: ...` 给出具体数字。

## 长提示词

- `--prefill auto` 每块最多读 16,384 个 token（NVFP4 为 32,768）。CPU 计算流式提示词专家中的 0.6
  （`--prefill-cpu-share`）。
- 64 个 token 以内的新增部分走 decode 窗口（`--short-read`）。
- 有多张卡时，长提示词可以在它们之上作为流水线读取（[GPU](../configuration/gpus.md#跨卡的长提示词)）。
- 与解码槽位并行的提示词：[运行](../running.md#与解码槽位并行的提示词)。

## Drafting

pack 带 MTP drafter 时开启。`--spec N` 设置窗口（默认 8：最后一个被接受的 token 加最多 7 个 draft）。`--spec 0`
关闭。`--suffix-draft`（默认 3）还会用上下文里更早的一处重复来填满窗口。输出与纯解码完全一致
（[精度](../precision/qwen38-flash-next.md#起草不改变输出)）。结构化输出 draft 得最好，散文最差。

## 并行槽位

`--parallel N`（1 到 8）同时服务 N 个请求（[运行](../running.md)）。

## 有损标志

默认关闭。`--relaxed-drafting`、`--route-prior` 和 `--route-tail-skip` 各自改变输出，并各自在启动时给出一条警告。
它们做什么、代价是什么：[精度](../precision/qwen38-flash-next.md#宽松起草)。

## 限制

- **不支持视频。**
- **双卡空闲睡眠** 可能让第二张卡的专家丢失直到重启（在 8,192 上下文、不开弹性 KV 时出现过，64K 或 262,144 时没有）。
