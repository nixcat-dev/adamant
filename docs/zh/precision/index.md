# 精度与质量

[English](../../precision/index.md)

**Adamant：** 本移植当前的执行路径及其验证状态见 [Ada 的执行与验证](ada.md)。下面的页面是
继承自 Basalt 的材料；Adamant 自己的测量结果在 [README](../../../README.zh-CN.md)。

## 继承的 Basalt 参考内容

下面的描述、源码引用与测量结果描述的是 Blackwell 上的上游 Basalt，不是当前的 Adamant 实现。
保留它们是作为参考证据，而非 Adamant 的结果。特别是，下文描述的原生 FP4 prompt 执行与持久化
GDN 中段都不是 Adamant 的路径。

本页说明 Basalt 以什么精度计算、投机解码（MTP）不改变其输出，以及相对同一个参考（BF16 的原始
模型）它的质量与 llama.cpp 相比如何。

测试环境：一张功率限制 400 W 的 RTX 5090，一张 RTX 5060 Ti 作为对等专家层，128 GB 系统 RAM
（操作系统可见 121 GiB）。每节都注明所做的测量属于哪个构建。

## 摘要

- **MTP 不改变输出**（除非设置了 `--relaxed-drafting`：[细节](features.md#宽松起草)）。专家层级
  固定时，两个被测试的 pack 上 275 条贪心 prompt 全部 275 条产生完全相同的 token，并且每个全
  词表 fp32 logit 行（62,271 行与 60,878 行）在 `--spec 0` 与 `--spec 7` 之间逐位相同。
- **有损标志需显式开启。** `--relaxed-drafting`、`--route-prior` 与 `--route-tail-skip` 会改变
  输出；每个都默认关闭，并列入同一条启动警告（[宽松起草](features.md#宽松起草)、[路由](routing.md)）。
- **比 llama.cpp 更接近 BF16。** 在同一套 Q8_0 权重上对照 BF16 参考，llama.cpp 自己的
  `llama-perplexity` 得 KLD 0.0325，Basalt Q8_0 pack 得 0.0200
  （[原因](quality.md#为什么-llamacpp-的-q8_0-离-bf16-更远)）。Q6_K 头（IQ3_S）的两阶段验证头
  增加约 0.0022（[细节](two-stage-head.md#q6_k-头的近似尾部)）。
- **只量化一次，绝不两次。** GGUF pack 按原样保留 GGUF 的 block。NVIDIA NVFP4 pack 逐位保留
  NVIDIA 的 NVFP4 专家 blob，并把每个稠密 tensor 从 NVIDIA 的 BF16 量化一次：投影为 Q6_K
  （[细节](features.md#nvidia-nvfp4-pack-的-6-bit-稠密投影)），头、token embedding 与共享专家
  down 为 Q8_0。它的 n-gram 表是 NVIDIA 的 FP8 值重新分块为 Q8_0（多一次舍入）。GGUF pack 的
  表则是下载自带的 IQ4_NL，除非用 `--ple-gguf` 指向另一个 GGUF：随后由一张独立的 Q8_0 表服务
  gather，因此 engine 读取的行是 8-bit 而非 4.5-bit，而 pack 里的行完全不被读取。Q5_0 或 BF16
  表也从那里读；BF16 行精确扩展为 f32，所以 gather 用的就是 checkpoint 自己的值，根本没有表的
  舍入。decode 还从一个 int8 副本（每 32 个输入一个 fp16 缩放因子）读取超连接矩阵，而 Q6_K 输出
  头在 verify 中经一张 2-bit 副本筛选：靠前的行是精确的 Q6_K logits，远端尾部保留副本的近似
  logits（[细节](two-stage-head.md#q6_k-头的近似尾部)）。Q8_0 头按其自身编码的高六位筛选，且界
  已有证明：每个消费者读的都是精确头的值（[细节](two-stage-head.md#精确的位平面-q8_0-头)）。
- **每次舍入单独测量。** 对照 NVIDIA checkpoint 的精度，默认配置里只有 Q6_K 稠密投影造成可测的
  KLD（约 0.006，以及 0.3 到 0.6 个百分点的 top-1）；其它每个默认舍入（hc8、进入稠密投影的
  8-bit 激活、共享专家与 NVFP4 专家、int8 KV、fast math、prompt 路径）都约 0.001 或更低
  （[细节](isolated-costs.md)）。
- **关键之处用 fp32。** 激活在残差流、norm、router、softmax 与 logits 中是 fp32，进入量化
  matmul 时为 int8（q8_1）。GDN 状态是 fp32。KV 默认 fp16，另有 int8、q4_0 与 k8v4 选项。
- **质量经 decode 路径打分。** Prefill 是另一种计算（fp16/BF16 GEMM、int8 MMQ，NVFP4 专家还有
  FP4 tensor core）；质量门槛评的是 decode 路径。
- **参考是全精度的。** 质量对照一个由 llama.cpp 生成的原始模型 BF16 基准来测量
  （[方法](quality.md#bf16-参考基准)，[所有 pack](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt#quality)）。

## decode 路径精度

来源是 `src/`、`include/basalt/` 与 `tools/convert/` 下的 `文件:行号`。模型几何：n_embd 2560、
4 条超连接流、48 层（36 GDN、12 QSA）、512 个专家、top-10、head_dim 256。

### 权重

| Tensor 类别 | 格式 | 来源 |
|---|---|---|
| 路由专家 | GGUF block 逐字节拷贝，不再量化。gate 与 up 每层共用一种类型；各层之间类型混用（IQ3_XXS pack：gate/up 为 IQ2_XXS、IQ2_XS、IQ3_XXS、IQ3_S、IQ2_S；down 在 30 层为 Q2_0、18 层为 IQ4_NL。IQ3_S pack：gate/up 多为 IQ3_XXS、IQ3_S、IQ2_S，一个 IQ4_XS；down 在 39 层为 IQ4_NL、9 层为 Q2_0） | docs/model-file.md；iq_kernels.cu:531-566；tests/data/native_experts/ |
| 路由专家，NVFP4 格式 | NVFP4（类型 40）：每 16 个权重 8 字节 E2M1 码加一个 UE4M3 缩放因子；每个专家一段 16 字节的 FP32 尾部 {s_gate, s_up, s_down, 0}。每个权重 4.5 bit，全部 48 层 | tools/convert/checkpoint.py；iq_kernels.cu（nvfp4_scale_dev） |
| NVIDIA-checkpoint pack（`--checkpoint`） | 直接从 NVIDIA 的 safetensors 构建。专家是 NVIDIA 的 NVFP4 码与 UE4M3 缩放因子，逐位搬进 Basalt 的 64 值 block；每个专家的尾部是它的三个 `weight_scale_2` 值（不使用 `input_scale` 条目）。稠密 tensor 从 BF16 转换一次：attention、GDN qkv/z/out 与共享专家 gate/up 为 Q6_K（`--dense-q8-0` 时为 Q8_0），共享专家 down、头与 token embedding 为 Q8_0（ggml 的舍入）；router、超连接矩阵、indexer、ssm_alpha/beta 与 PLE key/value 按存储形式保留 BF16，小矩阵一律不再量化；norm 权重 BF16 + 1 以 F32 计算（精确）；ssm_a = -exp(A_log) 在 f64 中计算并一次舍入到 F32；GDN 值头重排为 GGUF 顺序。n-gram 表是 checkpoint 的 FP8 E4M3 码乘其唯一 BF16 缩放因子后重新分块为 Q8_0（32 个一块、fp16 缩放因子 = absmax/127、ggml 的舍入）：距精确 FP8 值至多半个 Q8_0 步长加缩放因子的 fp16 舍入，相比之下其它 pack 用的是 4.5-bit 的 IQ4_NL 表 | tools/convert/checkpoint.py；convert.py（from_checkpoint）；model_sections.py（PleFp8）；test_checkpoint.py；test_convert.py（PleFp8Test） |
| 共享专家 gate/up/down | GGUF block，任何 native_mmvq 类型 | native_mmvq.hpp:129 |
| 共享专家标量门控 | BF16 | model_sections.py:48 |
| QSA q/k/v/o、GDN qkv、z（attn_gate）、ssm_out | GGUF block（dense_tc 接受 Q4_K、Q5_K、Q6_K、IQ4_XS、IQ3_S、Q3_K、Q2_0、Q8_0；否则走 native_mmvq）。dense_tc 上的 Q8_0 与 native_mmvq 并非逐位相等：int8 tensor-core 点积是每 32 值块的精确 int32，然后 fp32 缩放项按固定顺序求和，而 mmvq 在 warp 树中按 8 值四分之一块对 `d_w * d_x * sumi` 舍入。对照 float64 参考两者都约在 1e-7 RMS（dense_tc_parity，真实投影几何，T = 1..8，每列逐位不依赖 T）。Q4_K_XL pack 上的质量门槛（20 个 chunk，decode 路径，400 W）：基准 PPL 2.3272 / KLD 0.0687 / p99 1.054 / top-1 相同率 92.80%，Q8_0 走 dense_tc 时为 2.3269 / 0.0685 / 1.044 / 93.08%；全部差值都在 20-chunk 运行的运行间噪声内。官方 NVIDIA NVFP4 pack 把这些（以及共享专家 gate/up）以 Q6_K 存储，源自 NVIDIA 的 BF16 权重，所以它们走 dense_tc（[细节](features.md#nvidia-nvfp4-pack-的-6-bit-稠密投影)） | dense_tc.cu:1109-1121；dense_tc.hpp:2-12 |
| GDN ssm_alpha、ssm_beta；QSA indexer q/k 投影 | BF16 | model_sections.py:52-53 |
| Router | BF16，不量化 | model_sections.py:48 |
| Norm 权重、ssm_a、ssm_dt、conv1d、ssm_norm | F32 | model_sections.py:55-59 |
| 超连接（hc）矩阵（down、up、inject；每层及输出头各一份） | pack 中按来源的格式存储（GSQ-RCO 与 NVIDIA pack 为 BF16，UD-Q4_K_XL 为 Q8_0）；decode 从 pack 的 `hc8` section 读一个 int8 副本：int8 码、每 32 个连续输入一个 fp16 缩放因子，scale = f16(absmax/127)，码 clamp 到 ±127，由转换器逐位按 kernel 量化器的算法算出 | tools/convert/hc8.py；hc_fused.cu（hc8_quant_kernel）；verify/hc8.cpp |
| Token embedding | 保持 GGUF 类型，驻留在映射的宿主内存中，每个 token 反量化一行为 fp32。最低的 8192 行（id 最小的）在 VRAM 还有一份副本，token 在其中时 gather 读它；同样的字节经过同样的反量化器，所以数值相同 | native_head.hpp:49-53；native_head.cpp（iq_embed_rows 调用）；iq_kernels.cu（iq_embed_rows）；verify/record_inputs.cpp；native_expert_parity --q8-embd-hot |
| 输出头 | 一种 native_mmvq 类型，配有专门的 Q5_K 与 Q6_K kernel；IQ3_S pack 的头是 Q6_K，NVFP4 pack 的是 Q8_0。Q8_0 头以位平面保存（一行的 80 个 fp16 缩放因子，然后每个码加 128 后第 2..5 位、第 6..7 位与第 0..1 位各自独立成平面：每行仍是 2,720 字节）；它的 kernel 把平面读回 Q8_0 码，即 native_mmvq 的那些位。在 verify 中分两阶段运行：每行由缩放因子与第 2..7 位粗筛，界可证明覆盖该 logit，上界够到该列第 K 大 logit 下界的行被精确打分（K：贪心取 1，采样的请求取 top_k 至 64；更长的列表、penalty 或 logit bias 时取 0，所有行精确）；最大的 K 项就是精确头的那几项，其余每一项都是其上界、排在其下，而所有读取整行的消费者先跑精确遍（[两阶段验证头](two-stage-head.md#精确的位平面-q8_0-头)）；若对等卡比 CUDA0 慢，对等卡在自己那份副本上对照 CUDA0 的阈值粗筛末尾若干行并列出打过分的目的项，由 CUDA0 合并：得到同样的最大 K 项（[拆分](two-stage-head.md#较慢对等卡下的拆分)）。Q6_K 头在 verify 中分两阶段运行：每行都由 2-bit 副本（下面草稿头的拟合：四个级别、每 32 个权重一个 fp16 缩放因子，缩放因子乘以 sqrt(\|w\|²/⟨w, ŵ⟩)）打分；距该列第 64 名近似 logit 下界 4 以内的行精确重打分（plain Q6_K kernel 的位），其余各行保留近似 logit。argmax 与采样器的 top-64 列表来自精确行；softmax 的尾部是近似的（见[两阶段验证头](two-stage-head.md#q6_k-头的近似尾部)）；采样更长列表的请求，其窗口的头再精确跑一遍。该副本（248,320 行共 199 MB）在加载时、专家缓存定容之前打包 | native_head.cpp；verify/record_head.cpp、sampling.cpp、head_split.cpp；q80_vhead.cu、q80p.cuh；q6k_head.cu（q6k_vhead） |
| MTP drafter 稠密部分 | fc_embedding、fc_hidden、q/k/v/o、indexer 投影与共享专家投影为 Q8_0；其余为 BF16；一维 tensor 为 F32 | model_sections.py:447-492 |
| MTP drafter 专家 | Q2_0，全部驻留 VRAM | model_sections.py:496-507 |
| MTP 草稿头 | 主头的行（同一类型），可选词表子集：Q6_K 或 Q8_0 头经由子集的行映射从主头（Q8_0 头的位平面）读取，Q5_K 头则收集一份副本（两种方式位相同）；对 Q5_K、Q6_K 或 Q8_0 头，先由 2-bit 副本给前若干行打分（四个级别、每 32 个权重一个 fp16 缩放因子），距最佳近似分 5 以内的行精确重打分（超过 64 行符合时取近似分最好的 64 行，平局取行号小者；保留的行及其修正求和的顺序从不依赖线程时序），所以草稿与精确 argmax 只在罕见情况下不同，且草稿概率是近似的：softmax 求和中其它行的项来自 2-bit 打分。副本的缩放因子（三者同理）每组乘以 sqrt(\|w\|²/⟨w, ŵ⟩)，因为最小二乘副本给接近最佳的行打分偏低、会虚增概率（在 Q5_K parity 测试的合成头上跨 12 个 seed，概率的平均相对误差从 0.028-0.043 降到 0.010-0.020，最坏的一次从上至 1.04 降到 0.45 以下）；IQ3_S pack 的 1500 个真实草稿头输入上概率平均误差 +0.005（用最小二乘缩放因子则 +0.017），选取与精确 argmax 在 99.93% 的情况下一致；NVFP4 pack Q8_0 头的 3,566 个输入（冷 prose 与 conv、两张卡）平均误差 +0.005，选取一致率 99.97% | core/mtp/head.cpp、forward.cpp；head_proxy.cuh、q5k_head.cu、q6k_head.cu、q80_proxy.cu |
| MTP hc 矩阵 | 同一批 int8 副本 | core/mtp/load.cpp:268-282 |
| MTP 草稿长度 | 终止草稿的累计接受阈值是基准值乘一个 1.0 到 1.5 的系数，由 pack 最终层级中每个专家槽位平均由 CPU 服务的专家字节数一次性确定（约 1.1 MB 时 1.0，3.5 MB 及以上时 1.5）；它是 pack 及其放置的函数，与计时无关。被接受的 token 仍只来自 verify，但窗口宽度会随 pack 和构建变化，而 verify kernel 不是跨宽度 batch 不变的，所以宽度变化可能在两个候选相距在舍入以内的任何地方改变逐 token 输出 | draft_cum.hpp；core/mtp/draft.cpp；program/generate/dispatch.cpp |

drafter 只提议 token；每个 token 都由 verify 决定，所以 drafter 的精度影响的是速度，不是输出。

### 各 matmul 的激活

| Matmul | 激活格式 | 来源 |
|---|---|---|
| hc down 与 inject（hc 读取） | fp32，激活不量化；int8 权重精确转换，每组 fp32 fmaf | hc_fused.cu:380-390、512 |
| hc up | fp32 输入拆成 fp16 高、低两部分（约 22-bit 尾数），fp16 tensor-core mma 以 fp32 累加，逐组 fp16 缩放因子在 fp32 中施加 | hc_fused.cu:675-744 |
| Router | fp32 乘 BF16 权重，fp32 FMA | hc_fused.cu:765-838；native_bf16.cu:45-122 |
| GDN qkv、QSA q、GDN ssm_out、QSA o | q8_1：int8 码、每 32 个值一个 fp16 缩放因子，int8 tensor-core 点积（dense_tc）。decode 窗口的 QSA 查询投影分两次 launch 运行：先查询行，再在 attention 旁边跑门控行，每个 16 行 tile 用 dense_tc 的算术：位相同 | native_mmvq.cu:147-160；dense_tc.hpp:2-12；qsa_mid.cu |
| GDN z、QSA k/v | 符合条件时 z 与 qkv 一起走 dense_tc；否则 native_mmvq（对 q8_1 的 dp4a） | dense_tc.cu:1147-1152；verify/record_gdn.cpp、verify/record_qsa.cpp |
| 路由专家 gate/up | fp32 FFN 输入的 q8_1；dp4a，每块内 int32 精确 | hc_fused.cu:67-81；iq_kernels.cu:201-226 |
| 路由专家 down | fp32 SwiGLU，然后每 32 个 q8_1 | iq_kernels.cu:1881-1913 |
| 共享专家 | gate/up 用 q8_1；fp32 SwiGLU，down 用 q8_1；sigmoid 门控 fp32 | shared_expert.cu:104-144、210-234 |
| GDN alpha/beta、indexer 投影 | fp32 激活乘 BF16 权重，fp32 FMA | dec_gdn_rec.cu:88-114；verify/record_gdn.cpp、verify/record_qsa.cpp |
| 输出头 | fp32 最终流的 q8_1，走 mmvq 或 Q5_K/Q6_K kernel；两阶段 Q6_K 头的第一段把同样的 q8_1 码作为 int8 mma 输入（每 32 组精确 int32）、fp32 缩放项；Q8_0 头的粗筛把它们作为 dp4a 输入对码的高六位求和（每 32 组精确 int32）、fp32 项 | verify/record_head.cpp；q6k_head.cu；q80_vhead.cu |
| CPU 专家层（未驻留在 GPU 上的专家） | fp32 输入用 ggml 的 vec_dot_type 量化（i-quant gate/up 每 256 一个 Q8_K；down 用 Q8_0 或 Q8_K），所以与 GPU 层的舍入不同。哪些专家由 GPU 起步取决于 engine 对该模型的 profile（`data/profiles`），或先前运行用 `--expert-profile-save` 写下的文件（若存在）；自适应层从那里搬动它们，所以两次按不同 profile 启动的运行可能在某个 token 的末位上不同。一次休眠与唤醒（`--idle-sleep`）会把每个专家放回到驱动恢复所及范围内它原先占据的槽位（最后几个槽可能留空，因此下一个请求可能相对于不休眠时舍入不同），从磁盘读回的对话就是当时保存的字节（按其最后一个检查点存储；比运行末端短的一条回复，会再次走 prompt 路径读取） | native_expert.cpp:48-49、74-88；program/generate/model.cpp（profile）；program/serve/commands.cpp（SLEEP/WAKE）；conversation_store.cpp |
| CPU 专家层，Q6_K gate/up | fp32 输入的每 256 一个 Q8_K；ggml 的 AVX2 Q6_K x Q8_K 点积（AVX-512 CPU 也走它）：6-bit 码减 32，带符号的每 16 缩放因子，块内 int32 整数求和，在 fp32 中乘 fp16 block scale 与 Q8_K scale。对照按反量化操作数计算的双精度参考检查（最坏误差为绝对项之和的 1.3e-8） | ggml_cpu_dots.c，native_expert_parity --cpu-synthetic |
| CPU 专家层，AVX2 i-quant 行 | IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_XS 的 gate/up 对 Q8_K，IQ4_NL 的 down 对 Q8_0：每个 32 值块只解码一次，每个 token 的 int8 乘积按块在 int32 中求和，然后是 fp32 block scale。CPU 有 AVX-VNNI 时，int16 乘积用一条 `vpdpwssd` 而不是 `madd` + `add` 累加，得到同样的整数（两者都不会饱和），所以每个输出位都不变；那里的 IQ4_NL 把 w + 128 作为 `vpdpbusd` 的无符号操作数，并把四路求和的起点设为 -128 乘激活之和，对 Q8_0 量化器在 scale 非零的块里写出的每个码（-127..127；只有 fp16 scale 为 0 时它才写 -128，而那在两种构建中都会让乘积归零），与绝对值乘施加符号后的激活所得的整数相同。IQ4_NL 行按每次多行运行，每行按 block 顺序走自己的 fp32 链，所以每次的行数不改变任何位。池把一层切成行区间分给各线程（某个专家自己的 gate/up 行一算完就把它的中间值量化一次）：每行都由相同的代码从相同的字节算出，所以输出不依赖切法或工作线程数（逐位）。两条路径都对照按反量化操作数计算的双精度参考检查 | iq_avx2_rows.inc、iq_avx2_vnni.cpp、pool.cpp；iq_avx2_test、pool_native_test |
| CPU 专家层，Q2_0 down 行 | AVX2 用 `maddubs` + `madd`、AVX-VNNI 用 `vpdpbusd`，把 0..3 的码对 int8 激活求和，然后施加 block 与 chunk 缩放因子及预先算好的修正项。两者都对照双精度参考检查 | q2_avx2_rows.inc、q2_avx2_vnni.cpp；iq_avx2_test |
| CPU 专家层，NVFP4 专家 | fp32 输入与 fp32 SwiGLU 输出的每 32 一个 Q8_0；每 16 值子块内整数求和，在 fp32 中乘 UE4M3 子块缩放因子与 Q8_0 缩放因子。CPU 有 AVX-VNNI-INT8 时，256-bit 多 token 行用 `vpdpbssd` 直接在 int32 中累加有符号字节乘积，而不是 `vpsignb` + `vpmaddubsw` + `vpmaddwd`，得到同样的整数（两者都不会饱和），所以每个输出位都不变。专家的 FP32 尾部缩放因子继续在 fp32 中施加：s_gate 与 s_up 在 SwiGLU 之前乘在 gate/up 点积上，s_down 乘在 down 输出上（把 s_down 折进 up 会让 h 靠近 1e-5，那里激活的 block scale 会变成非规格化数）。对照同权重下的双精度专家：因激活舍入相对误差 1.2e-2；用 CPU 自己舍入后的激活则 7e-8 | native_expert.cpp（native_gu_rows、native_down_rows）；nvfp4_avx2_rows.inc、nvfp4_avx2_vnni.cpp、nvfp4_avx512.cpp；ggml_cpu_dots.c（ggml_vec_dot_nvfp4_q8_0）；nvfp4_cpu_rows_test、native_expert_parity --synthetic nvfp4/nvfp4 |
| CPU 未命中经 PCIe 暂存（单卡上的 batch 窗口） | batch 窗口中，CPU 未命中足够多的一层，其末尾若干专家（数量由 CPU 池与链路的每专家成本算出）从 RAM 副本拷入 VRAM 暂存并由 CUDA0 的分组 kernel 计算：用 GPU 层的算术而非 CPU 层，逐位等同于这些专家从 VRAM 得到的行。哪些专家去算取决于该窗口的路由以及 RAM 层把它们放在何处，所以 batch 输出像开启自适应时一样舍入；单独 decode 的请求走单流路径，不做暂存 | dec_moe.cuh（dm_stage_split、dm_stage_kernel）；verify/expert_resources.cpp（dm_init）、verify/batch_execution.cpp（dm_stage_places）、verify/record_moe.cpp；dec_moe_parity |
| GPU 专家，NVFP4（分组 kernel：逐项与 decode-once，以及两张卡上由设备规划的 MoE 尾段；NVFP4 gate/up 走 glue 路径，plan 与层级环都在 gate/up launch 内部而不是单独的 route launch；全部逐位相等） | fp32 输入与 fp32 SwiGLU 输出的 q8_1（与其它 GPU 格式同样的舍入）；E2M1 码经 16 项 int8 表，dp4a 每 16 值子块内 int32 精确求和，在 fp32 中乘 q8_1 缩放因子与 UE4M3 子块缩放因子。s_gate 与 s_up 在 SwiGLU 之前乘在 gate/up 点积上、s_down 乘在 down 输出上，都在 fp32 中，与 CPU 层一致。对照用 kernel 自己的 q8_1 激活的双精度专家：相对误差 5-8e-8（gate、up、down）；对照未舍入输入：1.2e-2；GPU 层对 CPU 层：8e-8（真实专家，层 0-3、20、47） | iq_kernels.cu（vec_dot_nvfp4_q8_1、Split<40>、native_gu_kernel、native_down_kernel 及其 multi 变体）；dec_moe.cuh（dm_gu_row、dm_down_item、dm_gu_kernel、dm_glue_gu1_kernel）；native_expert_parity --experts PACK；iq_multi_parity、native_grouped_parity、dec_moe_parity --nvfp4 |
| GPU 专家，Q8_0 gate/up 配 Q8_0 down（decode-once kernel 与设备规划的 MoE 尾段，与逐项 kernel 逐位相等；无数值改动） | fp32 输入与 fp32 SwiGLU 输出的 q8_1；int8 权重，dp4a 每 32 值块内 int32 精确求和，在 fp32 中乘 fp16 block scale 与 q8_1 缩放因子，每个 kernel 中是同一个表达式 | iq_kernels.cu（Split<8>）；dec_moe.cuh（DM_PAIRS）；iq_multi_parity、dec_moe_parity --kq |
| GPU 专家，Q6_K gate/up 配 Q8_0 down（decode-once 分组/verify kernel、设备规划的 MoE 尾段与对等卡 kernel；与逐项 kernel 逐位相等；对现有格式无数值改动） | fp32 输入与 fp32 SwiGLU 输出的 q8_1；6-bit 码减 32 后对 q8_1 字做 dp4a，每个 16 值子块的 int32 和乘其带符号 int8 缩放因子，32 值的乘积在 fp32 中求和并乘每块的 q8_1 缩放因子，再乘 fp16 block scale。decode-once 与逐项 kernel 共用这套算术。prompt 路径（MMQ）落地后单列一行 | iq_kernels.cu（Split<14>）、dec_moe_parity --kq |
| prompt 路径，NVFP4 专家（MMQ，默认） | fp32 输入与 fp32 SwiGLU 输出的 q8_1，与其他 MMQ 格式相同；E2M1 码经 16 项 int8 表进入 tile，int8 mma 每 16 值子块内 int32 精确求和，在 fp32 中乘 UE4M3 子块缩放因子与 q8_1 缩放因子。尾部在 SwiGLU 一步以 fp32 施加：h = silu(s_gate g) * (s_up u) * s_down，所以 s_down 随 H 进入 down 的乘积（q8_1 保持每 32 个值一个 fp32 缩放因子，所以 H 舍入到相同的码，只差该缩放因子的 fp32 舍入）。**相对它取代的 FP16 回退路径，数值有变化**：误差现在是所有 MMQ 格式共有的 int8 激活舍入。真实几何下的合成专家：gate/up 乘积对该 q8_1 输入的双精度参考相对误差 1.7e-7；层输出对该参考 6.8e-5（H 码在舍入平局处）；对未舍入的双精度参考 1.6e-2（FP16 回退路径为 5.2e-4）；MMQ 对 FP16 回退 1.6e-2 | mmq_load_tiles.cuh（ggml_cuda_mmq_load_tiles_nvfp4）；mmq_vec_dot.cuh（q8_0_16）；moe_mmq.cu（swiglu_kernel、gather 尾部）；prefill/expert_mmq.cpp；prefill_mmq_nvfp4_test |
| prompt 路径，NVFP4 专家（FP4 tensor core，chunk 至少 1,024 token；更短的 chunk 走 MMQ） | **W4A4。相对它取代的 MMQ 路径，数值有变化。** fp32 输入与 fp32 SwiGLU 输出以每 16 个值一个 UE4M3 缩放因子、每个 token 一个 fp32 缩放因子（amax / (6 x 448)）舍入为 E2M1 码；权重 block 原地读取，`mma kind::mxf4nvf4`（block scale，每 k64 四个 scale）以 fp32 累加。尾部缩放因子像 MMQ 一样在 SwiGLU 一步施加。真实几何下的合成专家：gate/up 对双精度参考的相对 rms 9.4e-2（MMQ：1.6e-2）。对照带同样 E2M1/UE4M3 激活舍入的 CPU 参考，kernel 相对 rms 一致到 1.4e-7（gate/up）与 3.7e-7（层输出），且某个 token 的输出逐位不依赖其 tile 与邻居。质量门槛 256 token 的 prompt 半段低于 1,024 token 的下限，所以门槛跑的是 MMQ prompt 专家，没有走到这条路径。长 prompt，NVIDIA-checkpoint pack，400 W，T 3，int8 KV，常驻专家：12 个 4,096 token 的 chunk（每个由 8 个连续的 512-token wikitext chunk 组成，所以每 512 个 token 有一个 BOS），2,049 token 的 prompt 半段走这条路径，每 chunk 2,047 行 teacher-forced（共 24,564 行）：PPL 2.9556，MMQ 为 2.9547（各自标准误 0.037）；按行配对的 NLL 差 +0.0003（+-0.0014，比率 1.0003）。对照同布局的 BF16 基准（llama.cpp 436f6f89e，400 W + 180 W，层级固定），这条路径对 MMQ、按 chunk 配对，95%：2,049-token prompt、24 个 chunk，KLD 0.0455 对 0.0437，+0.0018 [+0.0002, +0.0035]，top-1 -0.19 个百分点 [-0.31, -0.06]，开头被打分的行最大、之后衰减；8,193-token prompt、6 个 chunk，KLD 0.0377 对 0.0369，+0.0008 [-0.0015, +0.0030]，top-1 +0.01 个百分点 [-0.16, +0.19]，6 个 chunk 不足以判定。MMQ 会慢读 17 到 19%（冷启动，400 W + 180 W：16K 7807 对 6456 t/s，64K 10126 对 8201 t/s）。与主卡同速的对等卡用这些 kernel 为至多 5,120 token 的流式 chunk 计算它所持有专家的行，读自己那份激活的副本、按整块 tile 分片：与主卡的行逐位相等，所以没有数值变化；更慢的对等卡不参与流式 chunk（2,560 到 6,143 token 的 chunk 那时按单卡方式读取，所以其文本可能在末位与之前不同）（prefill_fp4_test 对等份额：512 专家、top 10、四种路由含热点专家） | moe_fp4.cu；prefill/expert_fp4.cpp（fp4_flush）；prefill/peer_experts.cpp（对等卡的 FP4 份额）；prefill_fp4_test |
| prompt 路径，NVFP4 专家（FP16 回退，仅 MMQ 没有 tile 之处） | 权重反量化到 fp32（精确），在 fp32 中乘尾部缩放因子，一次舍入到 fp16；fp16 GEMM。低于 fp16 正规范围（约 6e-5）的权重在那里失去相对精度 | iq_kernels.cu（dq_nvfp4）；prefill/expert_fp16.cpp:13-30 |

与 llama.cpp 不同，Basalt 不把 hc 读取的激活量化成 Q8_0。

### 累加、状态、softmax 与 logits

| 项目 | 精度 | 来源 |
|---|---|---|
| int8 点积 | 每 32 值块内精确 int32，然后 fp32 缩放项在 fp32 中求和 | dense_tc.cu:62-68；dense_tc.hpp:2-12 |
| 残差流与 hc buffer | fp32 | verify.hpp:661；hc_fused.hpp:28-40 |
| RMSNorm | fp32，eps 1e-6 | verify/common.hpp（EPS） |
| fp16/BF16 累加器 | 没有；hc up 使用 fp16 mma 输入、fp32 累加 | hc_fused.cu:675-744 |
| KV cache（`--kv`） | fp16、int8、q4_0（别名 q4）、k8v4（int8 K、旋转 Q4_0 V）。engine 默认 fp16；README 中的示例配置用 int8 | program/cli/arguments.cpp:66（`--kv` 解析）；program/generate/configure.cpp:203-205（qsa_set_kv_int8、qsa_set_kv_q4、qsa_set_kv_hybrid）；docs/configuration/ |
| int8 KV 布局 | int8 码、每 64 个值一个 fp16 缩放因子（256 宽的头 4 个，K 与 V 分开），scale = f16(amax/127) | kv_q8.hpp:3-5、21-23；kv_q8.cu:47-53 |
| 弹性 KV（serve） | 一个会话在启动时后备其下限的 cell，每个请求后备 prompt 加一个步长，并在后来的更短请求用不完时按下限的网格归还；KV 剩余显存放专家槽位。K/V 数值不变，但一个请求带着多少个槽位 decode 会改变放置（哪些专家由 GPU、哪些由 CPU 计算），所以在一次更深请求之后的自由生成文本可能与归还更少的构建不同。长 prompt 的 KV 占用的 CUDA0 槽位（至少 32768 个新 cell）的原专家移入对等卡的槽位，接替路由最少的专家（按最近路由、再按 profile 的排名）：对等卡用 GPU 层的算术计算它们，原先由 CPU 层计算；被替换的专家转到 CPU 层 | program/generate/sessions.cpp（kKvElasticFloor）、program/serve/verify.cpp（kKvGrowCells）、program/serve/kv.cpp（kv_fit、kv_release、peer_take）、core/kv_arena.cpp（网格） |
| Attention | K/V 反量化到 fp32；分数 fp32；每 64-cell 块在线 softmax，fp32（`__expf`），fp32 合并；输出门控 sigmoid，然后 q8_1 供 o 投影 | qsa_dec_attn.cu:138、255-427 |
| 仅草稿用的 attention | fp16 mma kernel；只改变草稿，从不影响被接受的 token | qsa_dec_attn.cu:430-431 |
| Indexer | Key 按 fp16 舍入、以 fp32 保存；block 分数 fp32；radix top-k，平局取较低 cell | native_qsa_indexer.hpp:11-16；qsa_select.hpp:13-14 |
| GDN recurrent 状态与 conv 历史 | fp32 | dec_gdn.hpp:19；verify/record_gdn.cpp |
| 至多八行的窗口中的 GDN 中段（mid chain）、异步 commit 收尾器 | qkv + z 成对、递推与输出投影作为一个持久 kernel 运行。在至少 160 个 SM 的设备上，输出和状态与三个独立 kernel 逐位相同。在少于 160 个 SM 的设备上，输出投影采用单 tile kernel 的归约顺序，与该设备未融合 kernel 的顺序不同（只影响那里的数值；此类卡 20-chunk 门槛，PPL 2.3076 到 2.3024，平均 KLD 0.0772 到 0.0769，top-1 93.10 到 92.69 %）；更大的设备保持逐位。低于 160 SM 的普通单流 decode 中，每个窗口宽度都走 mid chain；160 SM 及以上时，五到八行的窗口保留三个独立 kernel、归约顺序相同。带 profile 的运行、batch 窗口和跨两卡 mixer 切分使用独立 kernel，在低于 160 SM 时可能舍入不同。普通单流 decode 中某个 token 的输出不依赖宽度（36 SM 卡上对照 BF16 参考的 20-chunk 门槛、IQ3_S pack、180 W：3、5、8 行窗口给出完全相同的行，与之前 3 行窗口给出的行相同，PPL 2.3687、平均 KLD 0.17189、top-1 89.02 %；5 行的未融合 kernel 给 PPL 2.3844、平均 KLD 0.17164、top-1 89.04 %，且没有任何一行与 3 行窗口相同）。commit 从一个计数器领取它的项目，下一个窗口以一个领走其余部分的收尾 kernel 开始；首层之后的 GDN 层在下一个窗口内部提交，在窗口等待 CPU 层时先行完成；每个项目都照旧计算。它需要一个不带 fast math 编译的编译单元，因此保留在 CMakeLists.txt 的 fast-math 名单之外；`mid_chain_parity`、`dec_commit_parity --finisher` 与 `--lazy` 校验这些位 | mid_chain.hpp；mid_chain.cu；verify_kernels.cu |
| 跨两卡按 key head 切分的 GDN mixer（对等卡快于主卡；启动时决定） | 对等卡更慢时关闭：整个 mixer 在主卡上，逐位不变。开启时：out 投影是两个 fp32 部分和的 split-K 和，对等卡把它的投影跑成 q8_1 行上的 dp4a，且 q8_1 的 block scale 可能在舍入平局处差一个 fp16 步长，所以归约顺序改变末位（[细节](gdn-split.md)） | dec_moe.cuh（dm_mix_rows、dm_mix_rows_m）；verify/gdn_split.cpp（gs_plan） |
| Router softmax 与 top-10 | fp32，平局取较低的专家序号，权重在十个之上重新归一化 | native_router.cu:44-99 |
| Router，n_expert 不是 512（剪枝 pack） | 通用 router kernel 而非融合的 native kernel。最大值在 fp32；逐专家 exp 以 double 计算 (logit - max)；求和用 double，单线程升序；概率按 `(float)(exp * (float)(1/sum))` 舍入到 fp32。按 fp32 概率做十次 block 级 argmax 选出 top-10，平局取较低的专家序号。十个权重在 double 中求和，下限 2^-14，在 double 中相除、存为 fp32。native 路径的 exp、求和与倒数都在 fp32 中、以 warp 顺序求和，所以概率在末位上有差异，在罕见的近乎平局情况下选中的专家或权重可能与它不同 | router_top10.cu:57-174（max 75-88、exp 93、sum 96-100、概率 115、选择 132-163、重新归一化 166-173）；native_router.cu:44-99 |
| 专家合并 | fp32，按 router 排名顺序，再加上共享专家 | native_moe.cu:36-50 |
| 最终 logits | fp32 `[T][n_vocab]` | verify.hpp:671 |
| 贪心采样 | fp32 argmax，平局取最小序号 | sampler.cu:101-167 |
| 抽样选取（top_k、top_p、min_p、temperature、抽签） | top_k <= 32 时一次 launch（`sampler_sort_kernel`，sampler.cu:1056-1144；尾部在 `sampled_tail_pair`，986）：每个 4,096-logit 块用双调网络找出自己的前 k，行的最后一个块合并各块列表并在共享内存中带着列表跑尾部。更宽的 top_k 走一次 launch 的分轮切分（`sampler_fused_kernel`，695）。与两次 launch 的分轮（`sampler_split_part_kernel` 586、`sampler_split_merge_kernel` 669，parity 参考）相比：top-k 集合相同、exp 参数相同、求和顺序与商相同，因此逐位选取与 probe 概率都相同。长度超过 64 的列表（top_k 0、全词表、或超过 64）走宽路径：行的受罚 logits 按选择顺序做 radix 排序（平局取较低的 id，-0 = +0），然后在排序后的列表上跑同一条链、exp 参数相同，其求和在每个 SM 上按 1,024 线程分块（每块内各项按序，各块之和用固定 shuffle 树），每次越界都按 cum >= top_p S 或 u S < cum 判定，因此同一个 seed 能重放同样的抽签。与 64 项以内的窄 kernel 相比（sampler_parity --split，split 3）：选取相同，probe 的概率相差几个 float ulp（求和顺序）。**top_k 0 或超过 64 时数值有变化**：此前这类列表会被截到 64；现在是所请求的列表。top_k 1..64：无变化 | sampler.cu；sampler_parity --split |
| 请求的 `logit_bias` | 仅当请求发送了 bias：在 penalty 与 filter 之前把 fp32 bias 加到 fp32 logits；-100 即 -inf（该 token 不可选），其它值按给定相加。没有 bias 时采样器读同样的 logits、走与之前相同的代码路径 | sampler.cu；logit_bias.hpp |
| KLD 打分（主机） | 从 fp32 logits 以 double 累计 log-sum-exp 与 KLD | ppl_gate.hpp（KLD 函数） |

### prefill

Prefill 与 decode 是两种不同的计算，读写同样的 KV 与状态格式。

| 项目 | Prefill | 来源 |
|---|---|---|
| 稠密投影 | 权重反量化到 fp16，激活舍入到 fp16，cuBLAS 以 fp32 累加；BF16 tensor 走 BF16 x BF16、fp32 计算；没有 q8_1 | gemm.cu:123-197；prefill/projection.cpp:29-40 |
| hc 矩阵 | 激活舍入到 BF16；down/up 用反量化到 BF16 的 int8 副本；inject 用 BF16 tensor。从 33 个 token 起，up 乘积与混合作为一个 kernel 运行（FP32 门控行留在寄存器中）：每个门控值是 k 从零开始的一条 m16n8k16 链，即 cuBLAS 为该乘积运行的链，混合保持 gr_mix_r 的算术，因此输出与分两步时位相同（`prefill_hc_up_mix_test`） | prefill/projection.cpp:16-22（up_mix）；kernels.cu（gr_up_mix_kernel）；gemm.cu（bf16_w、bf16_rows） |
| Router | BF16 激活（decode：fp32） | prefill/routing.cpp:26 |
| 共享专家 | fp16 激活、fp16 SwiGLU | prefill/shared_expert.cpp:10-20 |
| 路由专家 | 融合的 int8 mma kernel、llama.cpp MMQ（q8_1）或反量化到 fp16 的 cuBLAS（仅 IQ1_M）。NVFP4 从 1,024-token chunk 起走 FP4 tensor-core kernel，以下走 MMQ（见上面激活表）。K-quant（Q4_K、Q5_K、Q5_1：UD-Q4_K_XL）在 MMQ 上运行，fp16 激活缩放因子、每 32 的部分和取自舍入后的 int8 值（llama.cpp 对浮点求和），因此乘积恰为反量化权重乘舍入后的激活。Q6_K gate/up（UD-Q5_K_XL、UD-Q6_K_XL）只走 MMQ，没有 FP16 路径：6-bit 码减 32 作为 int8、每 16 个值一个 float scale（d 乘 int8 子缩放因子），对同样的 int8 激活、每 32 个一个 fp32 scale（没有部分和，因为不存在最小值） | moe_fused_iq.hpp:1-12；moe_mmq.hpp:1-6；mmq/quantize.cu:59-69 |
| Attention | int8 KV：q 与 p 舍入到 fp16（q 预先乘 2 的幂），V 缩放因子折进 fp16 B 操作数，一次 fp16 mma、fp32 累加，逐 warp 在线 softmax 最后合并一次。输出在 FP64 参考输出尺度的 1.5e-3 之内（此前的 hi/lo 拆分：1e-6）。fp16 与 Q4_0 KV 保留 q 与 p 的 hi/lo 拆分 | qsa_prompt_attn.hpp:1-22 |
| CPU 上的路由专家（chunk 低于 6,144 token） | VRAM 缓存表之外（借出的槽位算在内）、有宿主副本且至多 8 行的专家走 CPU 池的 ggml 点积，如 decode 的 RAM 层：激活采用权重类型 `vec_dot` 的格式（IQ 类型每 256 值一个 Q8_K，而非每 32 一个 q8_1），H 量化成 down 类型的格式。它们的行不加权写入，由合并像 GPU 那样加权。哪些专家去算，是缓存之外的路由专家中固定的一份、行数少的优先（见下面 CPU 份额一段） | prefill/cpu_experts.cpp:48-78、120 |
| Indexer 分数 | 3xTF32 tensor core | qsa_select.hpp:36-38 |
| GDN | 分块 delta rule，每块 16 个 token：q/k 在 fp32 中做 L2 归一化再舍入到 BF16；v、门控与状态 fp32。K S 在 TF32 tensor core 上，Q S 与状态更新 K^T Delta 在 BF16 tensor core 上（Delta 只为该乘积舍入到 BF16），fp32 累加；块内求解在 fp32。某个 token 的输出取决于 16-token 分块网格落在哪里（相对约 1e-3）。输出投影输入 fp16。相对此前的 kernel：输出相对 RMS 2.9e-3。NVIDIA-checkpoint NVFP4 pack（稠密 Q8_0）上的 20-chunk 质量门槛，同样固定的层级，相对上一构建的变化落在运行噪声之内 | gdn_chunked.cu |

**专家的 CPU 份额。** `tests/cuda/prefill_cpu_share_test.cpp` 在模型几何下对随机原生 blob
运行池与 MMQ 路径，每个专家至多 8 行，对照双精度参考。专家输出的相对 RMS 误差，MMQ / CPU：
普通激活 1.0e-2 / 1.2e-2（全部四种格式对），当 0.5% 的激活大 12 倍时 1.4e-2 到 1.6e-2 /
3.1e-2 到 3.4e-2（CPU 的 Q8_K block 宽 256 个值，MMQ 的宽 32）。某行的输出不依赖有多少
token 共享其专家的工作（逐位，每一对）。质量门槛，IQ3_S，100 个 chunk，对照 BF16 基准，
400 W + 180 W，256-token 的 prompt 半段走这条路径：KLD 0.14920，而全部 prompt 专家都在 GPU
上（`--prefill-cpu-share 0`）时为 0.15003，配对差 -0.0008 [-0.0021, +0.0004]。该份额默认是
缓存之外路由专家的固定 0.6（`--prefill-cpu-share`），不是由计时驱动的值；缓存只通过它的表
起作用，不通过哪次读取借出哪些槽位、哪些被借出的专家保留 RAM 副本（两者都随读取的 chunk 与
K/V 模式而定），所以 CPU 计算哪些专家只取决于路由与层级，prefill 在不同运行之间、不同 K/V
模式之间逐位可复现（NVFP4 pack 与 Q4_K_XL 上六份完全相同的贪心输出）。不同的层级布局（缓存
大小、对等槽位、自适应换入换出）会挪动 CPU 的专家及其舍入，所以一致性比较应在同一配置的会话
之间进行。`--parallel` 下，独自运行的 prompt 读取（至多 16384 token，或没有任何槽位 decode
时的任何读取）取同样的份额；与 decode 槽位并行的更长读取把它流式的每个专家都放到 GPU 上
计算。decode 槽位最近路由到的专家在这种读取占用缓存槽位时留在 VRAM（它们与路由更少的专家
交换位置），所以其间 CPU 计算槽位的哪些专家遵循它们的路由，与任何层级布局一样。

**按 chunk 大小选择的专家路径。** 少于 6,144 token 的 prompt chunk 只暂存其 token 路由到的
专家并在 MMQ 上运行它们，NVFP4 专家从 1,024 个 token 起走 FP4 tensor core（上面 FP4 那一行），
行数少的非常驻专家由 CPU 池承担；6,144 token 及以上的 chunk 让每个非常驻专家经环形缓冲流过
并跑融合 kernel（NVFP4 没有融合 kernel：仍留在 FP4 tensor core 上，或 1,024 token 以下走
MMQ，所以那里的分界只改变哪些专家被暂存）。暂存全部专家的成本与 chunk 大小无关，所以只有当
大多数专家反正都会命中时才划算。分界以下的 chunk 中，FP4 专家像流式遍历那样 16 个一组运行，
与逐个处理逐位相等（`prefill_fp4_test` 检查一组拆成若干连续段的所有拆法）；其 32 个暂存槽位
像其它 buffer 一样从 prompt 的借用额度中出，所以读取期间 GPU 持有的专家、以及随之的 CPU
份额，随环形缓冲的大小变化，正如随任何层级布局变化。这种 chunk 的每一层还会在自己的 FFN 行
上跑下一层的 router，猜测给出超过 8 行的专家在本层收尾时被提前复制到自己的槽位：这只改变
从何处读取专家的字节，从不改变计算的内容。1,024 到 6,143 token 的 chunk（裸 `--prefill auto`
下这个长度的整个 prompt）走 MMQ 路径，而更低的分界会把它们送进融合 kernel（有融合 kernel 的
pack；NVFP4 pack 无论如何跑同样的 FP4 或 MMQ 乘积）；两条路径的舍入按上面的表所述不同。
质量门槛 256-token 的 prompt 半段低于两个分界，所以分界不挪动它的数字。12 个 4,096-token
chunk（2,048-token 的 prompt 半段，只有 PPL）上，PPL 是 2.9581，低分界时为 2.9547，按 token
配对的 NLL 差 +0.0011 +- 0.0011（400 W，单张 5090）。对等卡没有直接访问时，分界保持
2,560（`--peer-exclusive` 时为 4,096）。

**Prompt chunk 大小。** 裸 `--prefill auto` 以每块至多 16384 token 的等大 chunk 读取
prompt，专家 blob 超过 1.5 MB 的 pack（NVFP4）为 32768。一个 chunk 的 GEMM tile 决定其
K/V 的舍入，所以这类 pack 上超过 16384 token 的 prompt 分块不同，其 K/V 末位可能不同；至多
16384 token 的 prompt 和质量门槛 256-token 的 prompt 半段照旧分块，所以门槛的数字不动。
`--parallel` 下，独自运行的读取按单流路径的方式分块；与 decode 槽位并行的读取取其上限借用
额度允许的小 chunk，所以同一 prompt 的 K/V 末位在两种方式之间可能不同。一个与 decode 槽位
并行、又为更短请求让路的读取，稍后在同样的 chunk 里、在槽位旁边、无 CPU 份额地读完其余部分。

**短首块。** 超过 chunk 上限（且至少 32,768 token）的 prompt 把前 16,384 个 token 作为一个
chunk 读取，其余按每块至多上限的等大 chunk 读取，于是第一个 chunk 需要等待的 PLE 行更少。
这只是移动了 chunk 边界（GEMM tile，即 K/V 的舍入），其它不变：不超过上限的 prompt 与质量
门槛 256-token 的 prompt 半段照旧分块（NVIDIA-checkpoint pack 上的 20-chunk 门槛与上一构建
完全相同）。长上下文检查，65,515-token prompt，前 64 个贪心 token 用旧分块的 token
teacher-forced，spec 1，int8 KV，400 W：下一 token 分布相对旧分块的 KLD 均值 0.0032、中位数
0.00013、最大 0.038，64 个位置中 64 个 top-1 相同。64K benchmark prompt 的 16-token 贪心
文本在一个近乎平局的词上不同。

**跨卡切分 prompt。** 配有专家层卡时，prompt 中一段很长的全新部分可以跨卡读取
（[GPU](../configuration/gpus.md)）：层卡读取除最后一个之外每个 chunk 的较深层，各承担一个
连续块。chunk 就是主卡独自读取的那些，所以没有 chunk 边界移动。在层卡上，那些层把超连接
down/up 矩阵读成 pack 的 BF16 tensor（主卡读它的 8-bit 副本并反量化到 BF16），其 cuBLAS
乘积可能跑别的 kernel；路由专家和其余乘积保持主卡的算术。同一型号卡上任意一层的同一阶段给出
同样的位：把一张层卡的层切到两个阶段，得到同样的 token 与检查点。草稿层在主卡上用最后阶段
逐位读回的行构建它的 K/V。启用 KV 流式（`kv-resident`）时，主卡对层卡的 K/V 与其 RAM 副本
之间逐字节来回，所以读取结果不变：65,536-token prompt 在开与不开流式时给出同样的 400 个
贪心 token。定价留在主卡上的部分、单个 chunk 的部分，以及质量门槛 256-token 的 prompt 半段
照旧读取。跨两卡的长上下文检查，NVIDIA NVFP4 pack，int8 KV，spec 1，400 W，把主卡读取的前
64 个贪心 token teacher-forced 进切分读取，全词表 KLD：65,535-token prompt 均值 0.0008、
中位数 0.0003、最大 0.012，top-1 64 中 64；122,879-token prompt 均值 0.0005、中位数 0.0003、
最大 0.003，top-1 64 中 64。`--parallel` 下槽位全空闲时读取不变：122,879-token prompt 之后
的 64 个贪心 token 在 1 个槽位与 4 个槽位下完全相同（同样的专家缓存）。为更短请求让路的
读取在主卡读完它保留 chunk 的较深层后结束，正如主卡读取那样；prompt 其余部分从其断点起按其
自己的 chunk 稍后再读，正如主卡读取让路之后。

**K-quant 专家上 MMQ。** `tests/cuda/prefill_mmq_kq_test.cpp` 在模型几何下对随机零中心专家
运行 MMQ 与它取代的 FP16 反量化 + cuBLAS 路径，对照双精度参考。专家输出的相对 RMS 误差，
MMQ / FP16 路径：普通激活 1.08e-2 / 5e-4（Q4_K/Q5_1、Q4_K/Q8_0、Q5_K/Q8_0），0.5% 的激活
大 12 倍时 1.48e-2 到 1.51e-2 / 5e-4（Q6_K/Q8_0，只有 MMQ：普通 1.17e-2，带离群值
1.55e-2）；用 llama.cpp 的浮点部分和，MMQ 得 1.21e-2 到 1.76e-2 与 1.77e-2 到 2.19e-2。
单独重跑的一行或在另一个 batch 中运行，与它在完整 batch 中的行相比最多差该行最大输出值的
7e-4：MMQ 的 stream-k 拆分与 tile 宽度随 launch 的大小变化，而 H 的 int8 舍入能把一次重排的
求和变成一步 int8。i-quant 的 MMQ 路径同样（6.5e-4，IQ4_XS/IQ4_NL）。UD-Q4_K_XL 上的
100-chunk 质量门槛，256-token 的 prompt 半段走这条路径，相对 FP16 路径的变化落在运行噪声
之内。
