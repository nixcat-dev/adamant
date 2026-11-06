# 各项独立的精度代价

[English](../../precision/isolated-costs.md)

decode 路径的表在 [index.md](index.md)；质量方法在 [quality.md](quality.md)。本页测量
NVFP4 pack 的 decode 路径上每一处低于模型自身精度的舍入，各自单独关闭，以及全部一起关闭。
模型自身的精度就是 NVIDIA 的 checkpoint：按发布形式的 NVFP4 专家、BF16 稠密权重、fp32 激活
与求和。

## 方法

- **门槛：** `tools/ppl_gate.py`（`basalt-bench --quality`），100 个 wikitext-2 chunk，256
  token 的 prompt 半段由 prompt 路径读取，第 256 到 510 行经 decode 以 T 3 对照原始模型的
  llama.cpp BF16 基准打分（[quality.md](quality.md#bf16-参考基准)）。RTX 5090 在 400 W，
  RTX 5060 Ti 以 180 W 作为专家层，`--kv int8 --resident-experts --expert-cache 7000`，层级
  固定。默认就是带上这些 flag 的官方 pack（Q6_K 稠密）。
- **配对差：** 对每个 chunk，该实验分支的平均 KLD（及 top-1 相同率）减去默认分支在同样行上的
  值；表中给出 100 个 chunk 上的均值及基于 t 的 95% 区间。**无损**指区间包含 0，或输出逐位
  相等，或有证明使其精确；**可测量**指区间不含 0。
- **噪声：** 门槛是确定性的（两次默认运行：25,500 行中 25,500 行逐位相等）。只是重排求和的
  改动会让配对均值移动至多约 0.001（下面的重排行），区间半宽约 0.001，所以比这更小的移动
  无法判定。
- **变体：** 每个开关都是一次性的构建或转换器选项，不属于 engine。激活开关把 fp32 值（其
  int8 行即由它们生成）经一个普通 fp32 kernel 喂给投影（权重解码到 fp32，fp32 求和）；它
  同时关掉那些直接写 int8 行的融合 kernel，所以它的对照是同一构建、关掉这些 kernel 但保留
  int8 行。BF16-dense pack 是 `convert.py --checkpoint` 把每个投影都存为 NVIDIA 的 BF16
  而不是 Q6_K。
- **门槛长度：** 被打分的行位于 256 到 510。QSA indexer 的预算是 2,048 token，所以在这个
  长度下每个 cell 都会被选中，indexer 自身的舍入永远到不了输出；KV 格式的长上下文影响也未被
  覆盖。

## 全部关闭对比出厂默认

“全部关闭”指 BF16-dense pack：每个稠密投影、共享专家与头都用 fp32 激活，BF16 超连接矩阵
（没有 8-bit 副本，fp32 求和），fp16 KV，处处不用 fast math、`expf`、除法与开方取精确，
fp32 indexer key。所有分支中路由专家都保持 NVIDIA 的 NVFP4 码。

| 分支 | PPL | KLD 均值 | KLD p99 | 相同 top-1 | 配对 dKLD [95% CI] | 配对 top-1，百分点 [95% CI] |
|---|---:|---:|---:|---:|---|---|
| 默认（Q6_K 稠密，int8 KV） | 3.2975 | 0.06213 | 0.711 | 91.59% | | |
| 全部关闭 | 3.2866 | 0.05308 | 0.631 | 92.55% | -0.0091 [-0.0110, -0.0072] | +0.95 [+0.68, +1.22] |
| 全部关闭，专家激活也算 ¹ | 3.2850 | 0.05382 | 0.633 | 92.30% | -0.0078 [-0.0093, -0.0064] | +0.43 [+0.15, +0.70] |
| Q8_0-dense pack，其它不变 | 3.2863 | 0.05629 | 0.669 | 92.14% | -0.0059 [-0.0074, -0.0043] | +0.55 [+0.28, +0.82] |

¹ 单卡、256 槽位的专家缓存，因此 CPU 层计算 98.8% 的专家行，每个激活保留为一个 int8 值加
一个 int8 残差（每行两次点积，误差不超过 32 值块最大值的 1.6e-5）；在同样放置下与默认
flag 配对比较。

去掉所有可消除的舍入后，KLD 下降约 0.009（15%），top-1 一致率提升约一个百分点。大头是
Q6_K 稠密投影：只换成 Q8_0-dense pack 就让 KLD 下降 0.0059。全部关闭相对 Q8_0-dense pack
再降 -0.0032 [-0.0047, -0.0018]，top-1 再增 +0.40 [+0.18, +0.63] 个百分点，即其余舍入的
合计（BF16 对 Q8_0 稠密为 -0.0014，以及下面各项）。

对照更早的 llama.cpp Q8_0 基准，全部关闭与 Q8_0-dense pack 读数相同（0.0624 对 0.0623）：
那个基准自带舍入，把最后的 0.003 掩盖了。

速度，每行一次冷跑（`basalt-bench` 样例，默认服务 flag，400 W + 180 W，或 400 W 单张
5090）。全部关闭的 decode 用的是一次性 kernel（每行一个 warp 的 fp32 矩阵-向量乘与未融合
路径），所以它的代价是专用 kernel 成本的上界：

| 配置 | prose，2 卡 | struct，2 卡 | prose，单 5090 | struct，单 5090 | prefill 16K，2 卡 |
|---|---:|---:|---:|---:|---:|
| 默认 | 326.7 | 586.8 | 272.7 | 450.5 | 7,760 |
| 全部关闭 ² | 162.9 | 293.3 | 137.4 | 225.0 | 6,017 |
| Q8_0-dense pack | 304.1 | 566.6 | | | |

² 保留 fast math 与 fp16 indexer key；完全全部关闭的构建在两卡 prose 上读得 156.0 tok/s，
而只改精确数学的构建单独就 336.0 对 326.7（各一次运行，在单次波动之内）。全部关闭的时间
去向，两卡 prose：BF16 超连接矩阵走未融合读取 238.0 tok/s，激活开关所需的未融合 kernel
294.6，BF16 稠密加 fp32 激活经一次性 kernel 192.2。

Q8_0-dense pack 是用生产 kernel 达到全部关闭质量的方式：同一会话中比默认慢 7%（prose）和
3%（struct）（ABBA 数字：[6-bit 稠密投影](features.md#nvidia-nvfp4-pack-的-6-bit-稠密投影)）。
两个 NVFP4 pack 都在 Hugging Face 上：`Qwen3.8-FN-NVFP4-DQ6_K.basalt`（上面的默认）与
`Qwen3.8-FN-NVFP4-DQ8.basalt`（`D` 标注稠密格式）。

## 无损与可测量

除注明对照外都与默认配对；dKLD 为负意味着关掉该舍入使 KLD 降低那么多。不改变任何输出位的
条目注明了得出这一结论的方式。

| 条目 | 开关 | 配对 dKLD [95% CI] | top-1，百分点 | 结论 |
|---|---|---|---|---|
| Q6_K 稠密投影（对 BF16） | BF16-dense pack | -0.0072 [-0.0086, -0.0058] | +0.82 | 可测量 |
| Q6_K 稠密投影（对 Q8_0） | Q8_0-dense pack | -0.0059 [-0.0074, -0.0043] | +0.55 | 可测量 |
| Q8_0 稠密（对 BF16） | BF16-dense 对 Q8_0-dense | -0.0014 [-0.0025, -0.0003] | +0.27 | 可测量 |
| int8 超连接副本（hc8），decode | BF16 矩阵，fp32 求和 | -0.0002 [-0.0014, +0.0011] | +0.02 | 无损 |
| decode 与 prompt 路径都算 hc8 | 彻底没有 8-bit 副本 | -0.0014 [-0.0027, -0.0001] | +0.15 | 可测量，处于噪声边缘 |
| 进入 GDN 与 QSA 投影的 int8 激活 | fp32 激活（对照：融合 kernel 关闭） | -0.0004 [-0.0014, +0.0006] | -0.04 | 无损 |
| 进入共享专家的 int8 激活 | fp32 激活（同一对照） | +0.0005 [-0.0004, +0.0014] | -0.02 | 无损 |
| 进入头的 int8 激活 | fp32 激活（同一对照） | -0.00009 [-0.00012, -0.00005] | 0.00 | 可测量（KLD 的 0.1%） |
| 进入三者全部的 int8 激活 | fp32 激活（同一对照） | -0.0010 [-0.0019, -0.0002] | +0.15 | 可测量，处于噪声边缘 |
| 进入路由专家的 int8 激活（W4A8） | 两项拆分（对照：同样放置） | +0.0002 [-0.0008, +0.0012] | -0.15 | 无损 |
| 18 个 kernel 文件中的 fast math，`__expf`、`__fdividef`、`rsqrtf`、近似 `ex2`/`rcp` | 精确数学构建（对照：fp32 indexer key 构建） | -0.0003 [-0.0012, +0.0006] | +0.11 | 无损 |
| 示例配置的 int8 KV 对 fp16 KV | `--kv fp16` | +0.0000 [-0.0010, +0.0010] | +0.15 | 无损 |
| k8v4 KV | `--kv k8v4` | +0.0056 [+0.0038, +0.0073] | -0.20 | 可测量 |
| q4_0 KV | `--kv q4_0` | +0.0093 [+0.0074, +0.0112] | -0.38 | 可测量 |
| prompt 路径（fp16 GEMM、BF16 hc 与 router 输入、fp16 共享专家、MMQ 专家、fp16-MMA attention、BF16/TF32 tensor core 上的分块 GDN） | prompt 半段经 decode teacher-forced | -0.0000 [-0.0010, +0.0009] | +0.13 | 无损 |
| CPU 层对 GPU 层（Q8_0 对 q8_1 激活、kernel 顺序） | 98.8% 的专家行在 CPU，单卡 | -0.0005 [-0.0015, +0.0005] | +0.28 | 无损 |
| dense_tc 对 native_mmvq 归约顺序 | 关掉 dense_tc | +0.0003 [-0.0004, +0.0010] | +0.14 | 无损 |
| 融合 kernel 对未融合 kernel（hc 读取 int8 行、GDN 递推、mid chain、attention int8 输出、拆分查询投影、glue gate/up） | 激活开关的对照 | +0.0000 [-0.0008, +0.0008] | +0.10 | 无损（并非逐位：0.3% 的行相等） |
| 验证窗口宽度 | T 1 对 T 3，20 个 chunk | 0 | 0 | 无损：5,100 行中 5,100 行逐位相等 |
| fp16 舍入的 indexer key | fp32 key | 0 | 0 | 2,048 token 以下无影响：25,500 行中 25,500 行逐位相等 |
| 两阶段 Q8_0 头（六位粗筛） | 无需开关 | | | 有证明即无损：每个消费者读的都是精确的头（[两阶段头](two-stage-head.md#精确的位平面-q8_0-头)）；门槛打分的是精确遍 |
| drafter（Q8_0/Q2_0 权重、fp16 草稿 attention、2-bit 草稿头粗筛） | 无需开关 | | | 对输出无影响：每个 token 由 verify 决定（[features](features.md#投机解码不改变输出)） |

## 清单

decode 路径上所有低于 checkpoint 精度的舍入，以及 prompt 路径把哪些喂给了被打分的行。

| 条目 | 格式 | 来源 | 输出影响 | 已测量 |
|---|---|---|---|---|
| 路由专家 | NVIDIA 的 NVFP4 码与缩放因子，逐位 | tools/convert/checkpoint.py | 无（模型自身） | |
| 稠密投影 | 从 BF16 量化为 Q6_K（共享专家 down 为 Q8_0） | checkpoint.py（CheckpointDense） | 有 | 可测量 |
| 输出头 | 从 BF16 量化为 Q8_0 | checkpoint.py（GLOBAL） | 有 | 未单独测量 ³ |
| token embedding | 从 BF16 量化为 Q8_0 | checkpoint.py（GLOBAL） | 有 | 未单独测量 ³ |
| PLE n-gram 表 | NVIDIA 的 FP8 值重新分块为 Q8_0 | model_sections.py（PleFp8） | 有 | 未单独测量 ⁴ |
| 超连接矩阵 | int8 副本，每 32 个一个 fp16 缩放因子 | tools/convert/hc8.py；hc_fused.cu | 有 | 无损 |
| hc up 乘积 | fp32 输入拆 fp16 高/低，fp16 MMA，fp32 求和 | hc_fused.cu | 有 | 无损（与 hc8 一起） |
| router、ssm_alpha/beta、indexer 投影、PLE key/value、共享专家门控 | 按存储形式 BF16，fp32 激活与求和 | native_bf16.cu、dec_gdn_rec.cu | 无 | |
| norm、conv1d、ssm_dt | F32（norm 权重 BF16 + 1，精确） | checkpoint.py | 无 | |
| ssm_a | f64 中计算 -exp(A_log)，一次舍入到 F32 | checkpoint.py | 低于 BF16 源的精度 | |
| PLE conv1d | 从 BF16 转为 F16（范围内精确；转换器检查） | checkpoint.py | 无 | |
| 稠密投影激活 | q8_1（int8，每 32 个一个 fp16 缩放因子） | native_mmvq.cu、dense_tc.cu、record_gdn.cpp、record_qsa.cpp | 有 | 无损 |
| 共享专家激活 | gate/up 用 q8_1，SwiGLU 后 down 用 q8_1 | shared_expert.cu | 有 | 无损 |
| 头激活 | q8_1 | verify/record_head.cpp、q80_vhead.cu | 有 | 可测量，0.0001 |
| 路由专家激活 | q8_1（GPU），CPU 层每 32 个 Q8_0 | iq_kernels.cu、dec_moe.cuh、native_expert.cpp | 有 | 无损 |
| KV cache | 默认 fp16，示例配置 int8，k8v4，q4_0 | kv_q8.cu、kv_q4.cu | 有 | int8 无损，k8v4 与 q4_0 可测量；fp16 没有 fp32 对照 ⁵ |
| indexer key | fp16 舍入，以 fp32 保存 | native_qsa_indexer.cu、qsa_dec_fused.cu | 仅 2,048 token 以上 | 门槛未触及 |
| indexer 分数（prompt 路径） | 3xTF32 | qsa_select.cu | 仅 2,048 token 以上 | 门槛未触及 |
| fast-math kernel 文件 | 近似除法与开方，denormal 清零 | CMakeLists.txt（`--use_fast_math` 名单） | 有 | 无损 |
| SwiGLU、sigmoid 门控、softmax、norm 中的近似 intrinsic | `__expf`、`__fdividef`、`rsqrtf`、`ex2.approx`、`rcp.approx` | hc_fused.cu、qsa_dec_attn.cu、dec_gdn_rec.cu、shared_expert.cu、iq_kernels.cu、sampler.cu 等 | 有 | 无损（与 fast math 一起） |
| 归约顺序（dense_tc、融合 kernel、CPU 层） | 另一种顺序的 fp32 求和 | dense_tc.cu、mid_chain.cu、pool.cpp | 有 | 无损 |
| prompt 路径 | fp16 GEMM、BF16 hc 与 router 输入、int8 MMQ 专家、fp16-MMA attention、BF16/TF32 分块 GDN | src/prefill | 有，经由 prompt 的 KV 与状态 | 无损 |
| 两阶段 Q8_0 头 | 六位粗筛，界有证明 | q80_vhead.cu | 无 | 有证明 |
| drafter | Q8_0/Q2_0 权重、fp16 草稿 attention、2-bit 头粗筛 | src/core/mtp | 无 | 按构造 |
| GDN mixer 切分 | 跨两卡的 split-K | dec_moe.cuh | 有，仅当对等卡快于主卡 | [gdn-split.md](gdn-split.md)；测试卡上关闭（对等卡更慢） |

³ 头 kernel（位平面、两阶段粗筛、草稿头行）与 embedding gather 都读 Q8_0；没有可切到的
BF16 头或 embedding 路径。⁴ engine 没有 FP8 表路径。对照精确的 FP8 值，Q8_0 行差 5.4e-3
相对 RMS（采样 65,536 行）；FP8 值本身只有三个尾数位。⁵ 没有 fp32 KV；fp16 比 checkpoint
计算所用的 BF16 还多三个尾数位。

## 解读结果

- **稠密投影：** 默认舍入中最大的一项。相对 Q8_0 稠密，Q6_K 花费 0.0059 KLD 与 0.55 个
  百分点的 top-1；相对 BF16 是 0.0072 与 0.82；Q8_0 对 BF16 稠密是 0.0014。Q8_0-dense
  pack 以速度表中的代价找回大部分；其余所有舍入合计（含 BF16 稠密）再贡献 0.003。
- **hc8：** decode 中在噪声内；prompt 路径也算时，0.0014 处于区间边缘。
- **8-bit 激活：** 稠密投影、共享专家与 NVFP4 专家（W4A8 对两项 int8 拆分）都在噪声内。头的
  int8 输入花费 0.0001，之所以能分辨是因为其配对差非常一致；三个稠密消费者合计 0.0010，处于
  区间边缘。
- **KV：** 在这个长度上 int8 与 fp16 的差异在噪声内；k8v4 与 q4_0 分别花 0.006 与 0.009。
- **fast math 与近似 intrinsic、归约顺序、prompt 路径、CPU 对 GPU 层：** 都在噪声内。
