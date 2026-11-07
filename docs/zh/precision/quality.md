# 质量如何测量

[English](../../precision/quality.md)

Basalt 在同一段文本上把模型对照一个全精度参考打分，走的是生成 token 的 decode 路径。
[Qwen3.8-Flash-Next](qwen38-flash-next.md) 就是这样打分的。

## 门槛

`basalt-bench --quality` 运行门槛。`tools/ppl_gate.py` 对照构建好的 engine 离线运行它（[命令](#复现参考基准)）。

- **文本：** wikitext-2，每个 chunk 512 个 token。每个 chunk 的前 256 个 token 做 prefill。
- **打分的行：** 第 256 到 510 行，经 decode 路径、按 `--quality-t`（默认 3）的验证窗口进行。窗口的 token 就是参考文本，
  所以不使用草稿的输出。
- **参考：** llama.cpp 的 KL 散度基准文件（`--kl-divergence-base`）。
- **KLD** 是在全词表上计算的 KL(base || test)，取自 fp32 log-softmax，以 double 求和。基准 log 概率为 -16 或更低的项
  被跳过，会略微低估。
- **报告：** PPL、平均 KLD（表中另外给出中位数与 p99）以及 top-1 一致率。
- **长度：** 快速检查用 20 个 chunk。改动了数值就需要 100 个。
- **覆盖：** 256-token 的 prefill 半段走 MMQ prompt 路径。门槛既不到达长 chunk 的 prompt 路径，也不到达跨卡切分的
  prompt；两者单独测量（[Flash-Next prefill](qwen38-flash-next.md#prefill)）。
- **确定性：** 层级固定时，同一构建的两次运行逐行一致（见[贪心一致性](#贪心一致性)）。

## BF16 参考基准

每个分数都对照原始模型的 BF16：

| 模型 | 基准 | 基准 PPL |
| --- | --- | --- |
| Qwen3.8-Flash-Next | unsloth 的 BF16 GGUF 经 `llama-perplexity`（llama.cpp 提交 436f6f89e），100 个 512 chunk，`-b 8192 -ub 8192` | 3.2749（在被打分的行上是 3.2733） |

官方 Flash-Next pack 对照这个基准的分数列在其
[Hugging Face 页面](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt#quality) 上。该门槛是 100 个 chunk、T 3、
int8 KV、专家常驻、层级固定。

量化过的基准不是参考。llama.cpp 自己的 Q8_0 GGUF 会带上它自己的舍入，所以每个 pack 对照 Q8_0 基准都比对照 BF16 高
0.003 到 0.010 KLD（见 [Flash-Next](qwen38-flash-next.md#摘要)）。

数值改动：与上一构建并列报告 PPL、平均 KLD 与 top-1。逐位中性的改动：展示贪心一致性。

## 贪心一致性

先固定放置，否则在 CPU 池与 GPU 之间移动的专家会舍入不同：`--adapt-swaps 0 --peer-adapt-swaps 0`、固定的
`--expert-cache N`，以及固定的对等层级。

强制窗口一致性（同样的窗口宽度必须给出同样的行）：

```sh
python3 tools/ppl_gate.py --model PACK.basalt --engine-bin bin/basalt --base flashnext-bf16-base.kld \
    --chunks 20 --t {1|3|7} --out DIR --expert-cache N --kv int8 --resident-experts --peer-exclusive --prefill auto
python3 tools/ppl_gate.py --diff DIR1/rows.tsv DIR7/rows.tsv
```

MTP 开与关，逐 token：每种配置一个 engine 会话，通过 `--serve`：

```sh
bin/basalt --serve --model MODEL.basalt {--spec 0 | --spec 7} --expert-cache N --prefill auto --kv int8 \
    --vram-reserve-mib 700 --resident-experts --peer-exclusive --gpu 0,1 --max-context 8192 \
    --adapt-swaps 0 --peer-adapt-swaps 0 --exp concur9_logit_hash=1
```

`READY` 之后，stdin 上的 `GEN 256 temperature=0 ID,ID,...` 返回 `T ID` 行，并以 `DONE` 结束；`QUIT` 结束会话。
stderr 上的 `LH` 行是哈希行。按位置比较，并按位置匹配行。

## 复现参考基准

Flash-Next 基准（模型放不进 RAM 与 VRAM，所以专家从磁盘流式读取）：

```sh
llama-perplexity -m Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf -f wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base flashnext-bf16-base.kld -c 512 -b 8192 -ub 8192 --chunks 100 -ngl 99 --n-cpu-moe 99 -t 16
```

对照它给 pack 打分：用同一条命令再加上 `-m PACK.gguf --kl-divergence`。
