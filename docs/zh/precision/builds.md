# 跨构建的精度

[English](../../precision/builds.md)

Linux 与 Windows 构建之间的编译器一致性，以及如何复现本目录中的数字。

## 编译器一致性

CPU 专家 kernel 和专家池由所在平台各自的编译器编译，所以两个受支持的平台运行的并不是
同一份目标代码：Windows 上是 MSVC，Linux 上是 GCC。这在精度上的代价是测量出来的，不是
假设的。同一提交的两个构建针对同一段解出的专家 section 运行同样两个程序：`expert_parity`
覆盖 Q2_0 专家 kernel 及其 AVX-512 VNNI 路径，`pool_test` 覆盖调度它们的池：

| 检查项 | MSVC 19.44（Windows） | GCC 13.3（Linux） |
| --- | --- | --- |
| `expert_parity` | 0 失败 | 0 失败 |
| blob 解码为活动缩放因子 | 40/40 非零，\|s\| 在 [0.0179, 0.03564] | 相同 |
| 下一个专家是不同的数据 | 96.9% 的字节不同 | 相同 |
| VNNI 对比标量，INT8 激活 | rel 1.111e-06 | rel 9.412e-07 |
| VNNI 对比标量，FP32 激活 | rel 1.209e-02 | rel 1.139e-02 |
| `pool_test` | 0 失败 | 0 失败 |
| 池输出对比串行运行 | 25600 个中 0 个不同，最差 \|d\| 0.000e+00 | 相同 |

结构检查与池输出完全一致，池的输出逐位相同。两个跨编译器的指标在第三位有效数字上一致，
这正是 FMA 收缩与归约顺序能够移动的范围，而且它们分别比 kernel 断言的 3e-3 容差低三到四个
数量级。FP32 那一行由测试报告，并不做断言：那是 Q8_1 激活约定的代价，而且在两个编译器上
代价相同。

范围：这两个程序覆盖 Q2_0 专家 kernel 与专家池。量化 kernel（`iq_avx512`、`iq_avx2`、
`kq_avx2`、`kq_avx1`、`native_expert`、`ggml_cpu_dots`）同样由两套工具链编译，但这里的
程序没有用真实权重调用过它们，所以这张表不构成对它们的证据。（`iq_avx2_test` 与
`pool_native_test` 会在任意 AVX2 CPU 上用随机 blob 和模型几何跑 AVX2 i-quant 行与池的
原生路径；它们的结果不在本表中。）

### 两个构建的贪心 token 并不相同

在容差内一致并不是逐位一致，token 流能说明这一点。同一个 pack、同一个提交、temperature 0、
专家层级固定（`--expert-cache 4096 --resident-experts`）、自适应换入换出关闭，两个构建的
专家放置完全相同（VRAM 10.2 GiB、RAM 36.7 GiB、5341 个槽位），三条 bench prompt 各生成
300 token：`struct` 逐 token 相同，`conv` 在第 23 个 token 处出现分歧，`prose` 在第 44 个。
连跑两次、同一个构建用同样的参数对照自身，三个 prompt 全部逐 token 相同，所以差异跟着构建
走，而不是跟着运行走。

由此有两点。移植不承诺与 Linux 逐字节一致，这就是该承诺实际含义的测量。而且被测的是主机
编译器，不是 GPU 工具链：Linux 侧随后用 CUDA 13.0（与 Windows 构建相同的 toolkit 版本）
重新构建，仍在同样两个 token 处以相同的 id 出现分歧，所以工具链版本对此没有影响。两个构建
之间剩下的差别是 CPU 上的 GCC 对阵 MSVC——78% 的专家行在 CPU 上计算——上表的数字说明它们
相差多远。

这一切要有意义，自适应换入换出必须**关闭**：开着时，一次构建连自己的 token 都无法在两次
运行之间复现，因为边 decode 边在层级之间交换专家。

质量门槛是判定这种分歧可否接受的检查，它需要一个由 llama.cpp 基于 BF16 模型建立的
KL 散度基准（[quality.md](quality.md#bf16-参考基准)）。取这些数字的机器上没有建立过这样的
基准，因此 Windows 构建尚未跑过该门槛。

两个程序都读取一段扁平的专家 section：512 个 1,382,400 字节的 blob，即为其编写的 Q2_0
布局。pack 的 `mtp/experts` section 携带一段，可原样拷出，所以同样的字节能在任一构建上
运行：

```sh
python3 -c "import sys; sys.path.insert(0,'tools'); from model_file import ModelFile; \
  open('experts.bin','wb').write(ModelFile('PACK.basalt').read('mtp/experts'))"
./expert_parity --file experts.bin
./pool_test --file experts.bin
```

## 复现方法

先构建 llama.cpp，再从 unsloth 的 BF16 GGUF 构建 BF16 基准（100 个 chunk；模型放不进
RAM 加 VRAM，所以专家从磁盘流式读取，大批量时每 16 个 chunk 才读一遍）：

```sh
llama-perplexity -m Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf -f wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base flashnext-bf16-base.kld -c 512 -b 8192 -ub 8192 --chunks 100 -ngl 99 --n-cpu-moe 99 -t 16
```

llama.cpp 对一个 pack 给出的数字：

```sh
llama-perplexity -m PACK.gguf -f wikitext-2-raw/wiki.test.raw --kl-divergence-base flashnext-bf16-base.kld \
    --kl-divergence -c 512 -b 8192 -ub 8192 --chunks 100 -ngl 99 --n-cpu-moe 99 -t 16
```

Basalt 的数字，以及强制窗口一致性检查（分别用 `--t` 1、3、7 运行，再 diff 各行）：

```sh
python3 tools/ppl_gate.py --model PACK.basalt --engine-bin bin/basalt --base flashnext-bf16-base.kld \
    --chunks 20 --t {1|3|7} --out DIR --expert-cache N --kv int8 --resident-experts \
    --peer-exclusive --prefill auto
python3 tools/ppl_gate.py --diff DIR1/rows.tsv DIR7/rows.tsv
```

MTP 开/关一致性：每个配置一个 engine 会话，通过 `--serve` 驱动：

```sh
bin/basalt --serve --model MODEL.basalt {--spec 0 | --spec 7} \
    --expert-cache N --prefill auto --kv int8 --vram-reserve-mib 700 --resident-experts \
    --peer-exclusive --gpu 0,1 --max-context 8192 --adapt-swaps 0 --peer-adapt-swaps 0 \
    --exp concur9_logit_hash=1
```

等待 stdout 出现 `READY` 行，然后在 stdin 每行写一个请求：`GEN 256 temperature=0 ID,ID,...`，
后接 prompt 的 token id。每生成一个 token，stdout 会回一行 `T ID`；`DONE` 结束该请求，
`QUIT` 结束会话。logit hash 以 `LH` 行打到 stderr，每接受一行一条，带其位置与 hash。

对比两个会话的 token id 与逐位置 hash。`--spec 0` 时每个 token 一行；`--spec 7` 的运行可能
在请求末尾之后多打若干行，所以请按位置匹配行。这里使用的固定大小是：IQ3_XXS 为 N = 11582、
M = 8876，IQ3_S 为 N = 9979、M = 7562。
