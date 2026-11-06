# 质量如何测量

[English](../../precision/quality.md)

精度表在 [index.md](index.md)。

`basalt-bench --quality`（`tools/ppl_gate.py`）以 512 token 的 chunk 给 wikitext-2（`wiki.test.raw`）打分。每个
chunk 的前半做 prefill，后半（第 256 到 510 行）经 decode 路径打分，对照一个由
`llama-perplexity --kl-divergence-base` 写出、以 llama.cpp 自己的 `.kld` 格式读取的 llama.cpp 基准文件。基准是
BF16 的原始模型（[下文](#bf16-参考基准)）。它的 256-token prompt 走 MMQ，所以不会用到 NVFP4 prompt 路径的
4-bit 激活——那从 1,024-token 的 chunk 才开始，且已单独测量（[index.md](index.md)）。

打分框架（`include/basalt/program/ppl_gate.hpp`（KLD 函数））从头 logits 的 fp32 log-softmax 在全词表上计算
KL(base ‖ test)；基准按 scale*u16+min 解码，基准 log 概率 ≤ -16 的项被跳过（略微低估）。prompt 半段会与基准
token 校验相等，窗口 token 就是基准 token，而 KLD 随位置是平的，所以两者是对齐的。

## BF16 参考基准

基准是 unsloth 的 Qwen3.8-Flash-Next BF16 GGUF 经 `llama-perplexity`（提交 436f6f89e）产出，100 个 512
chunk，`-b 8192 -ub 8192`，稠密层在 GPU 上、专家从磁盘读取：PPL 3.2749（在被打分的行上是 3.2733），token
与门槛读取的完全相同。每个官方 pack 对照它的 KLD、top-1 与 PPL，用这个门槛（100 个 chunk，T 3，int8 KV，
`--resident-experts`，层级固定，400 W 的 RTX 5090 + 180 W 的 RTX 5060 Ti）列在
[Hugging Face 页面](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt#quality) 上。llama.cpp 自己的
Q8_0 GGUF 经 `llama-perplexity --kl-divergence`（整个 chunk 一次 batch 完成）对照它得 KLD 0.0325、top-1
93.98%；Basalt Q8_0 pack 得 0.0200 与 95.34%。

更早的数字是对照一个 llama.cpp Q8_0 基准测的。那个基准带着 llama.cpp 自己的 Q8_0 舍入（下一节），所以每个
pack 对照它都高出 0.003 到 0.010 KLD，而 Basalt Q8_0 pack 离 BF16（0.0200）比离它（0.0304）更近。这些页面中
的所有数字都是对照 BF16 基准的。

## 为什么 llama.cpp 的 Q8_0 离 BF16 更远

llama.cpp 会量化它 Q8_0 matmul 的激活。在超连接读取中（down 投影 10240 → 320，然后 up）求和存在相消，所以
输入的 Q8_0 量化（每元素 0.2 到 0.4%）在第 0 层的混合流上变成约 1%。Basalt 的 hc 读取使用 fp32 激活，与精确
算术吻合到 1e-5。

在第 0 层，用 numpy 从 llama.cpp 自己的 hc_norm 输出与 GGUF Q8_0 hc 权重计算，对照不做激活量化的 fp64 结果：
Basalt 的混合流相对误差为 0.00000，llama.cpp 的是 1.06%。llama.cpp 对照一个 Q8_0 激活量化的仿真相差 0.009%。
这就是 llama.cpp 跑 Q8_0 GGUF 对照 BF16 得 KLD 0.0325、而 Basalt Q8_0 pack 得 0.0200（上表）的主要原因。
