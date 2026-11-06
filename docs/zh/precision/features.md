# 各特性的精度

[English](../../precision/features.md)

decode 路径的精度表在 [index.md](index.md)；质量方法在 [quality.md](quality.md)。本页覆盖
NVFP4 稠密投影、两阶段验证头、logit bias、投机解码与宽松起草。

## NVIDIA NVFP4 pack 的 6-bit 稠密投影

官方 NVIDIA NVFP4 pack 把它的稠密投影存成 Q6_K，而 Q8_0 变体把它们存成 Q8_0。这是一次
权重格式变化：`convert.py --checkpoint` 从 NVIDIA 的 BF16 对每个投影只量化一次。
`--requant-dense-q6k PACK --checkpoint DIR` 给出同样的字节。不带 `--checkpoint` 时，它从
pack 的 Q8_0 值量化，会舍入两次；官方文件不是这样做出来的。

- **变动范围：** attention q/k/v/o、GDN qkv、z 与 out，以及共享专家的 gate 和 up：252 个
  tensor，2,948 MiB 稠密权重减少 654 MiB。共享专家的 down（640 个值的行不是完整的 256
  值块）、头与 token embedding 保持 Q8_0；router、超连接矩阵、indexer、ssm_alpha/beta 与
  PLE key/value 保持 BF16；专家、PLE 表、drafter 与 vision 不变。
- **来源与舍入：** 每个 tensor 都从 NVIDIA 的 BF16 权重只量化一次，绝不从 Q8_0 副本量化
  （那会舍入两次）。GGUF 列出的 48 个 GDN 值头与 checkpoint 的顺序不同；转换器会重新
  排序。`--requant-dense-q6k --checkpoint` 先把每个 BF16 tensor 与 pack 的 Q8_0 副本比较
  （最大距离为 0.62 个 Q8_0 步长；超过 1.05 即停止，这能抓住 tensor 或头排序错误）；
  `--checkpoint` 搭配 `--sections-from` 按名字读取 BF16 tensor，给出同样的 Q6_K 字节。
- **方法：** 每 256 值块内，以 signed-max 起始网格做每 16 值的最小二乘缩放因子，8-bit
  缩放因子对一个 fp16 超缩放因子取值，数值按存储形式（ggml 的 `block_q6_K` 布局）的缩放
  因子舍入。误差在高分布组上约为 sigma 的 1.9%，转换器的测试以 6-bit 步长为界约束它。
- **为什么更快：** 6-bit 稠密权重在 decode 中走 tensor-core 稠密路径，而 Q8_0 稠密权重退回
  mmvq kernel，而且字节数约少 22%（2,948 MiB 中少 654 MiB）。

速度，RTX 5090 在 400 W、带一张 5060 Ti 对等卡，两个 pack 在同一会话中交替（ABBA，每臂
6 个全新进程），开发测量；6 次均值：

| 测试 | 8-bit 稠密 tok/s | 6-bit 稠密 tok/s | 变化 | 每窗口毫秒 |
|---|---:|---:|---:|---|
| prose | 213.4 | 229.9 | +7.8% | 10.00 到 9.24 |
| conversation | 251.0 | 265.1 | +5.6% | 10.34 到 9.35 |
| structured | 420.4 | 463.5 | +10.2% | 15.88 到 14.39 |
| prefill，8K 与 32K | 3,806 与 7,791 | 3,845 与 7,856 | +1% | |

草稿接受率在运行间波动范围内不变，所以收益来自窗口时间。

质量，针对 BF16 参考、经 decode 路径（[quality.md](quality.md#bf16-参考基准)），100 个
chunk，400 W + 180 W，一个构建：

| 稠密格式 | PPL | KLD 均值 | KLD p99 | 相同 top-1 |
|---|---:|---:|---:|---:|
| Q8_0（NVFP4-DQ8） | 3.2970 | 0.0561 | 0.641 | 92.25% |
| Q6_K（NVFP4-DQ6_K） | 3.3037 | 0.0624 | 0.728 | 91.65% |

按 chunk 配对的 KLD 差为 +0.0059（95% CI +0.0043 到 +0.0074，[isolated-costs.md](isolated-costs.md)），
所以损失超出了运行间噪声：约 10% 的 KLD 和 0.5 个百分点的 top-1。长 prompt 上（12 个 4,096 token
的 chunk，没有参考基准，因此只有 PPL）PPL 从 2.9547 到 2.9816；配对 NLL +0.0091（95% CI
+0.0049 到 +0.0132）。Q6_K 稠密经 dense_tc 与 dense_tc_pair 运行；适用与任何 Q6_K 稠密
tensor 相同的算术（q8_1 激活、int8 tensor-core 点积）。

## 两阶段验证头

Q8_0 头在已有可证明界的情况下于验证窗口内分两阶段运行（每个消费者读的都是精确头的值），
Q6_K 头则经由 2-bit 副本加上近似的 softmax 尾部：[two-stage-head.md](two-stage-head.md)。

## 请求 logit bias

`logit_bias`（Chat Completions）由验证器自己的采样施加，作用于每个窗口的每一行，因此 bias
是推测必须复现的分布的一部分：本就在每个窗口之后运行的主机侧采样（temperature 或惩罚使用
的同一个）带着 bias 运行，草稿只有与那次抽样相同才被接受。草稿头忽略 bias；它只提议 token，
所以它提议的被禁 token 会被拒绝，带 bias 的请求接受的草稿更少。没有 bias 的请求走未改动的
捕获贪心采样。

在 IQ3_S 上按下一节所述固定层级检查：带 bias 的请求（贪心与带 seed、ban、boost、混合 map、
带与不带 presence penalty）在 `--spec 7` 下返回与 `--spec 0` 相同的文本，空 map 的请求与
不带该字段的请求一致。-100 的 bias 永不产出该 token，+100 的 bias 每一步都产出它。

## 请求 grammar

带 `grammar` 键的请求（engine 以 `--grammar on` 启动）在每个验证窗口之后于主机上检查。
token 按顺序喂给 grammar；在它拒绝的第一个 token 处，该行带着 grammar 的 mask 加入请求的
logit bias 重新抽取（同样的 filter、penalty 与逐位置抽样），窗口到此为止，其后的草稿像一次
拒绝一样被丢弃。grammar 接受的窗口不受影响：没有额外的 launch 或同步。`--batch` 下每个槽位
持有自己的 guard，同样的规则按槽位执行：扫描它保留的选取，第一个被拒的行带着该槽位的采样
与位置在 mask 下重新抽取，槽位的段在该行处结束。从独占运行移入槽位的请求先把它已生成的
token 重放过一遍 grammar。没有 grammar 的槽位与请求走未改动的路径。
工具调用 grammar 从调用起始符开始，JSON grammar 从 thinking 块之后的第一个 token 开始。带
grammar 的请求不启用宽松起草。prompt 从不加约束。不带该键的请求，以及 `--grammar off` 的
engine，走未改动的路径。schema grammar 是 llama.cpp 的（整数界、字符串长度、pattern、数组
长度与 format 都会执行）；llama.cpp 无法表达的（`uniqueItems`、数值界、没有 grammar 形式的
pattern）不执行。

## 投机解码不改变输出

独立生成器从最后一个 prompt token 开始验证，包括一个 token prompt 的零号位置。初始窗口
只有一行；后续窗口使用与更长 prompt 相同的验证器与草稿策略。

在普通单流 decode 中，无论窗口多宽，verify 都给出与 plain decode 相同的 token logits
（verify.hpp:3-8）。本节端到端地检验这一点。GDN 层的中段依次以 `dense_tc_pair`、
`dec_gdn_rec` 和 `dense_tc` 运行，其输出不依赖窗口宽度。

默认窗口为 8 个 token（`--spec 8`；显式的 `--spec N` 按给定值使用）。带 drafter 的贪心解码
在任意窗口下都保持精确。采样请求仍从模型自身的分布抽取，但窗口长度决定每个位置使用哪一次
随机抽样，所以带 seed 的采样请求在不同窗口下可能返回不同文本。下表用 `--spec 7` 测量；
精确性论证不依赖宽度。

**方法。** 275 条 prompt：272 条关闭 thinking 的 chat 模板 prompt（152 条 prose、60 条
chat、60 条 code）加上 `data/bench` 的 3 条 decode prompt（conv、struct、prose）。每条 256
个新 token，temperature 0，每个配置一个 engine 会话，prompt 顺序相同。一个 logit-hash
诊断会为每个被接受的行打印其全词表 fp32 头 logits 的 FNV-1a hash。比较 `--spec 0` 与
`--spec 7` 之间每个生成位置的 token id 与 hash。

专家层级固定（`--adapt-swaps 0 --peer-adapt-swaps 0`，专家缓存与对等层固定为自动适配选出
的大小），因此两次运行中相同的专家位于相同的设备。

| Pack | Prompts | 相同的 token | 比较的 logit 行 | 逐位相同的行 | 首次分歧 | 接受的草稿 | Decode tok/s，spec 0 → spec 7 |
|---|---|---|---|---|---|---|---|
| IQ3_XXS | 275 | 275（100%） | 62,271 | 62,271 | 无 | 37,892 / 54,206（69.9%） | 183.9 → 319.8 |
| IQ3_S | 275 | 275（100%） | 60,878 | 60,878 | 无 | 36,650 / 52,312（70.1%） | 145.1 → 239.5 |

吞吐量是这 275 个请求上 400 W 的纯 decode 速率。spec 7 运行的验证窗口宽度为 1 到 7，所以
比较的行包含在每个窗口宽度 T 下计算的 logits，且全部等于 spec 0 运行的 T=1 logits。在更早
的一个构建上，重复运行 IQ3_XXS spec 0 与第一次逐位相同（62,491 行中的 62,491 行），所以
该设置在运行之间是确定性的。

### 强制窗口 KLD

质量门槛是 teacher-forced 的：窗口 token 就是参考文本，强制宽度 `--t`，不使用 drafter 的输出
（`src/program/serve/request_decode.cpp` 的 `ppl != nullptr` 分支）。强制 drafter 实际产生的
窗口宽度会走与真实起草相同的 verify kernel。

IQ3_S，20 个 wikitext-2 chunk，针对 BF16 基准（基准 PPL 2.2883），int8 KV，层级固定，
400 W + 180 W：

| 强制窗口 T | PPL | KLD 均值 | KLD 中位数 | KLD p99 | 相同 top-1 | rows.tsv md5 |
|---|---|---|---|---|---|---|
| 1 | 2.3893 | 0.17480 | 0.01585 | 2.3380 | 88.92% | 60db649beeb84321e53ee96868e72f30 |
| 3 | 2.3893 | 0.17480 | 0.01585 | 2.3380 | 88.92% | 60db649beeb84321e53ee96868e72f30 |
| 7 | 2.3893 | 0.17480 | 0.01585 | 2.3380 | 88.92% | 60db649beeb84321e53ee96868e72f30 |

全部 5,100 行逐行数据（NLL 与 KLD）在 T=1、3、7 之间逐字节相同。窗口宽度不改变 logits，
所以 MTP 开与关的 KLD 相等；0.175 是量化相对基准的差距。

### 适用范围

- 层级固定时一致性是逐位的。开启自适应时，专家在 CPU 池与 GPU 之间移动，二者的舍入不同：
  这是舍入差异，不是质量差异。弹性 KV 从专家缓存借走又归还的槽位，以及对等卡接管这些槽位
  原持专家的情形，同样如此。空闲对话移到 RAM 再移回，字节分毫不差：层级固定时，它接下来的
  轮次写入相同的 token、接受相同的草稿，与一直留在 VRAM 时一样。
- GPU 层（VRAM 与对等卡）共用一套算术；CPU 层用 ggml 的 vec_dot_type 量化（i-quant 的
  gate/up 为每 256 一个 Q8_K，对应 GPU 上每 32 一个 q8_1；native_expert.cpp:48-49、
  74-88）。质量门槛上约 0.8% 的路由专家在 CPU 上计算（IQ3_XXS，默认放置）。划分按放置
  固定，所以 spec 0 与 spec 7 相同。
- prompt 长度为 19 到 395 token、各新增 256 token：这测的是 decode 路径，不是长上下文。

### 分层切分上的重叠窗口

两卡分层切分可以在第二张卡还在验证窗口 K 时，就让第一张卡开始窗口 K+1（见
[GPU](../configuration/gpus.md)）。第一张卡从一份状态快照出发，仿佛每一行都被保留一样提交
K（该提交会改变 recurrent state、conv 历史、indexer 的池化行与 block 位置、PLE 历史），并从
“K 被整体接受、且其最后一次选取就是 drafter 的下一个 token”这一猜测开始运行 K+1。猜测失败时
就恢复快照，并按没有重叠的方式重跑 K+1；猜测成立时，第二张卡就在 K 的提交之后运行 K+1。每一行
都由相同的 kernel 在相同的输入上计算，所以贪心输出就是普通切分的输出：专家缓存固定时，两个构建
为 conv、struct、prose 以及 3-5K token 的 prompt 写出相同的 400 个 token（一个专家在 CPU
层与同一专家在 GPU 上相差 8e-8，会在未固定层级的两次运行之间翻转一个接近平局的 token）。
带评分（`ppl=`）的请求、带 repetition penalty 或 grammar 的请求、宽松起草和 batch 槽位
从不重叠窗口。

## 宽松起草

有损标志集中列在同一条启动警告里；路由相关的在 [routing.md](routing.md)。

`--relaxed-drafting P`（需显式开启；默认关闭）为采样请求放弃上述保证。在精确模式下，草稿
只有等于目标模型在该行的抽样才会保留（`src/program/serve/request_decode.cpp`，验证窗口之后
的接受循环）。在宽松模式下，当目标在请求的最终分布（已应用 top_k、top_p、min_p、temperature
与 penalties）下给出概率不小于 P 时，草稿也会被保留。采样器在抽样结果旁报告那个概率
（`sample_tokens_probe`，src/kernels/cuda/sampler.cu:1718）；logits 与抽样本身不变。

**改变的是什么。** 草稿是草稿头的 argmax，由上下文固定。在草稿 d 满足 p(d) ≥ P 的行上，
每次输出都是 d，而不是从 p 中抽样：该行变成贪心。它与模型分布的总变差距离恰为 1 − p(d)，
在其它每一行为 0。被保留草稿之后的各行以它为条件，正如普通解码那样。保持精确的有：贪心请求
（什么都不抽）、带 `logit_bias` 的请求、JSON `response_format`（服务器发送 `relax=0`）、
质量门槛、`--parallel` batch 槽位、任何起草出的附加 token（id 从 `<|endoftext|>` 起的控制
token、think 与工具标签、turn 结束），以及从 `<tool_call>` 到 `</tool_call>` 的每个 token，
所以工具名及其参数都精确采样。

**精确模式不受影响。** 关闭该标志且层级固定时，IQ3_XXS，三条 bench prompt 贪心，外加 conv
和 prose 在 temperature 0.6 与 1.0、各两个 seed（600 token）：全部 11 条 token 流与加入该
标志之前的构建相同。开启该标志时，三条贪心流也相同。

**每个 token 的分布代价（离线回放）。** 记录了真实的验证窗口（每行：草稿、目标的抽样、目标的
最终概率），来自 bench 的 prose 与 conv prompt（三个 seed）和五条自然 chat prompt，在
temperature 0.6 与 1.0（top_p 0.95、top_k 20）、IQ3_S 与 IQ3_XXS 上，每格约 3,500 个
token。回放按规则应当保留草稿时保留，并对每个生成 token 求宽松行的 1 − p(d) 均值
（TV/token；精确模式为 0）。token 收益是回放出的每窗口额外 token 数（按记录的窗口宽度）；
下面的在线收益更大，因为窗口控制器在保留更多草稿时会拉长窗口。

| 模型 | Prompts | T | 精确接受 | P = 0.5 收益 / TV | P = 0.3 收益 / TV | P = 0.2 收益 / TV |
|---|---|---|---|---|---|---|
| IQ3_XXS | prose | 0.6 | 64.1% | +4.3% / 0.026 | +6.9% / 0.042 | +9.2% / 0.052 |
| IQ3_XXS | natural | 0.6 | 64.5% | +5.3% / 0.030 | +8.4% / 0.050 | +10.7% / 0.063 |
| IQ3_XXS | prose | 1.0 | 61.6% | +5.6% / 0.035 | +11.1% / 0.064 | +13.9% / 0.080 |
| IQ3_XXS | natural | 1.0 | 62.9% | +6.6% / 0.043 | +12.1% / 0.073 | +15.2% / 0.089 |
| IQ3_S | prose | 0.6 | 63.7% | +3.0% / 0.022 | +5.9% / 0.037 | +7.2% / 0.046 |
| IQ3_S | natural | 0.6 | 64.5% | +4.9% / 0.030 | +8.0% / 0.049 | +9.8% / 0.060 |
| IQ3_S | prose | 1.0 | 61.1% | +5.1% / 0.035 | +9.3% / 0.061 | +12.4% / 0.079 |
| IQ3_S | natural | 1.0 | 62.3% | +7.1% / 0.042 | +11.9% / 0.072 | +14.7% / 0.089 |

每次宽松保留都移动 1 − p(d) 的概率质量，所以每单位 TV 换得的 token 在各个阈值下几乎相同：
每 0.01 TV/token 约 +1.5%。Medusa 风格的典型接受（随 p 的熵下降的阈值）也回放过，落在同一条
线上；它并不是更划算的交换，所以该标志就是朴素的概率阈值。

**速度（400 W）。** `basalt-bench`（conv 与 prose 在 temperature 0.6，struct 贪心），带
`--spec 7 --kv int8 --vram-reserve-mib 500 --resident-experts --peer-exclusive`，每个值
三轮、相互交错，去掉最冷的一轮：

| 模型 | 样例 | 精确 | P = 0.5 | P = 0.3 | P = 0.2 |
|---|---|---|---|---|---|
| IQ3_XXS | prose | 348.7 | 366.8 (+5.2%) | 373.8 (+7.2%) | 382.4 (+9.7%) |
| IQ3_XXS | conv | 383.8 | 421.5 (+9.8%) | 431.4 (+12.4%) | 423.3 (+10.3%) |
| IQ3_XXS | struct（贪心） | 660.2 | 658.5 | 659.4 | 658.0 |
| IQ3_S | prose | 323.8 | – | 347.2 (+7.2%) | – |
| IQ3_S | conv | 354.2 | – | 392.9 (+10.9%) | – |

每个模型一个 engine 会话，prose prompt 与五条自然 prompt，两个 seed，四种设置按 prompt
交错（decode tok/s，400 W）：

| 模型 | T | 精确 | P = 0.5 | P = 0.3 | P = 0.2 |
|---|---|---|---|---|---|
| IQ3_XXS | 1.0 | 332.5 | 366.9 (+10.3%) | 385.5 (+15.9%) | 399.1 (+20.0%) |
| IQ3_XXS | 0.6 | 351.2 | 370.1 (+5.4%) | 383.1 (+9.1%) | 380.9 (+8.5%) |
| IQ3_S | 1.0 | 336.3 | 356.8 (+6.1%) | 369.7 (+9.9%) | 392.3 (+16.7%) |
| IQ3_S | 0.6 | 347.0 | 359.2 (+3.5%) | 370.9 (+6.9%) | 382.4 (+10.2%) |

**质量。** teacher-forced 的 KLD 门槛看不到这一改动：它给 logits 打分，而宽松起草改变的是
发出哪个 token，不是 logits。所以证据是上面的回放 TV 加采样任务。IQ3_S 通过 server，同样的
flags，temperature 1.0，top_p 0.95，top_k 20，thinking 关闭，每题一个 seed 使各设置成对，
四种设置在同一个 server 会话内（按请求的 `relaxed_drafting`），400 W：

| 任务 | 精确 | P = 0.5 | P = 0.3 | P = 0.2 |
|---|---|---|---|---|
| gsm8k，300 题（灵活抽取） | 0.947 ± 0.025 | 0.973 (+0.027 ± 0.020) | 0.980 (+0.033 ± 0.022) | 0.977 (+0.030 ± 0.023) |
| MMLU-Pro，每科 20 题（280） | 0.682 ± 0.055 | 0.714 (+0.032 ± 0.040) | 0.725 (+0.043 ± 0.045) | 0.721 (+0.039 ± 0.046) |
| 按规则保留的草稿，占每个生成 token | 0 | 3.0% | 5.0% | 5.6%（含工具请求） |

± 为 95% 区间（方括号内为相对精确模式的配对差）。更早的一次会话里所有臂都在跑精确模式
（按请求的键没传到 engine），四次精确运行的 gsm8k 在 0.960 到 0.970、MMLU-Pro 在 0.696 到
0.714 之间，两次运行之间有 6 到 9 题和 24 到 35 题翻转（自适应专家层级每次运行舍入不同）：
本次会话的精确臂处于该区间低端。在这个规模上测不出损失；宽松各臂持平或高于精确运行，这与
“让有把握的行变贪心”（相当于更低的有效 temperature）的改变相符。代价是多样性，这些任务
测不到；上面的 TV/token 才是直接度量。

工具调用，IQ3_S，同样的采样，8 个工具（如 `replace_in_file` 与 `read_file` 这类名字相近的）
上的 50 个请求，一半开启 thinking：精确 53 次调用、P = 0.2 52 次调用，每个名字都是声明过的
工具，没有泄漏调用文本，所有参数都是合法 JSON。engine 日志显示这些请求中该规则几乎没有保留
草稿（调用本身就是精确的）。

**选择 P。** 单写 `--relaxed-drafting` 即 0.5：代价最小（TV/token 0.03 到 0.04），在
temperature 0.6 约 +5%，在 1.0 约 +6 到 10%。0.3 让收益与代价都翻倍；0.2 是激进的一端。
高于 0.5 的值保留的草稿很少（0.7：在 TV 0.015 下约 +2.5%）。

## prefill 的 CPU 份额

CPU 池与 GPU 对一个流式 prompt 专家的舍入不同（见 [index.md](index.md) 中 CPU 份额一段），
所以 `--prefill-cpu-share` 能挪动数字。NVFP4 pack（DQ6_K），质量门槛 20 个 chunk 对照 BF16
参考，T 3，int8 KV，`--spec 0`，400 W + 180 W，份额 0（全 GPU）对 1（全 CPU）：PPL
2.3046 / 2.3068，KLD 0.07337 / 0.07085，top-1 93.04% / 93.12%。配对 KLD 差为 +0.0025
[-0.0005, +0.0055]，在该门槛的噪声之内，所以两者之间的任何份额都被它约束。`basalt-probe`
可能把不同的份额作为建议打印出来（[probe](../probe.md)）；它从不自行更改。

## 固定文本回放与 MTP 段移植

`ppl_score=0` 跳过主机 logit 拷贝与逐行打分；它是计时诊断，不是质量门槛。`ppl_force=1` 在
保留自然草稿窗口选择的同时发出参考续文。两者都不改变默认的打分或解码。

MTP 稠密段移植会保留 tensor 格式与目标权重、目标 HC8 值、词表和专家。它用现有量化器重新生成
由 MTP 派生的 HC8 条目。不同的 MTP 提议可能改变推测窗口的几何结构，其中既有的对几何敏感的舍入
即使目标 tensor 字节未变也可能改变贪心轨迹。一次简短的贪心比较发现，被测试的 Lamina pack 输出
有差异且没有速度收益；未做 BF16 PPL/KLD 验证，所以该资产未经确认可以作为生产替代品。
