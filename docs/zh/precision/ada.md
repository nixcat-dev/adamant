# Ada 的执行与验证

[English](../../precision/ada.md)

Adamant 的 Ada 实现尚处于开发早期。Linux sm_89 Release 构建已在 CUDA 13.0.88、GCC 14.3.1
和警告视为错误的条件下编译通过。运行时验证尚不完备；能编译并不能确立数值正确性。
[qwen38-flash-next.md](qwen38-flash-next.md) 及与其并列的页面中的上游测量结果不是 Adamant 的结果。

## Ampere（sm_86 和 sm_80）

`include/basalt/cuda_target.hpp` 在 8.9 之外接受 compute capability 8.0 和 8.6；`CMAKE_CUDA_ARCHITECTURES`
接受 `80`、`86` 与 `89` 的任意组合（默认 `89`），其他值为配置错误。`gpu "auto"` 选择每一块可见的 Ada 或
Ampere 卡；显式指定的其他能力的卡会终止启动。任何 kernel 都不使用 FP8 或 FP4 tensor 指令
（NVFP4 走整数 MMQ），因此按设计没有任何 weight pack 或 KV 格式在 Ampere 上被排除。

### sm_80 验证

Linux Release 构建在一块 NVIDIA A100-SXM4-80GB 上测试，强制功耗上限为 400 W，
工具链为 CUDA 13.0.88 和 GCC 13.3.0。一次干净构建用时 707.527 秒。76 项 CPU
测试全部通过，用时 54.89 秒；102 项可运行 GPU 测试全部通过，用时 753.37 秒。
两项 GPU 跳过分别是需要多卡的 peer 测试和需要模型的生成测试；后一项用下载的 pack 重跑后通过。
端到端验证使用 Hugging Face revision `65de5a9e636e8bbe1200cbcd6eb8aa421e6a66c6`
的公开 Basalt v2 `Qwen3.8-FN-UD-IQ3_XXS` pack，配置为 int8 KV、
`resident-experts`、`expert-cache auto`、`prefill auto`、262144 最大上下文以及
MTP `spec=8`。全部 46.84 GiB 专家位于 VRAM，没有 RAM 或 SSD 溢出。标准 decode
fixture 为：`conv` 165.5 tok/s（1057 tokens）、`struct` 274.0 tok/s（7230 tokens）、
`prose` 138.7 tok/s（1200 tokens）。另一次采样运行以 5711.22 tok/s 端到端速度
prefill 65535 tokens，随后以 163.02 tok/s 生成 512 tokens。标准 cold 66617-token
prefill 为 engine 5734.0 tok/s、wall 5717 tok/s。这些数字只验证该 pack 与配置，不代表
所有 quant 或 workload。这证明了 sm_80 的构建、kernel 测试套件、长上下文 prefill 与
生成路径；BF16 质量门槛与多 GPU 路径仍待完成。

### sm_86 状态

sm_86 构建可以编译；sm_86 硬件上仍未完成的有：

- 对照当前与 CPU 参考 kernel 的一致性测试。
- 针对 BF16 参考的 decode 路径 PPL/KLD（`basalt-bench --quality`）。
- `cp.async` copy 路径的 memcheck。
- 依 SM 数量设门槛的快速路径及其在更小卡上的回退。
- 8 GB 级卡的性能与默认调优。

## 执行改动

| 操作 | Adamant 实现 | 需要验证的数值后果 |
|---|---|---|
| MMQ prompt 专家 | 现有整数 MMA tile，不使用 programmatic dependent launch；保留全部已配置格式 | 不打算改动算术或 tile 选择。不假定跨设备相等。 |
| NVFP4 prompt 专家 | 所有 prompt 尺寸都走整数 MMQ，而非原生 FP4 tensor 指令 | 长 prompt 下激活改用 MMQ 的 q8_1 舍入，而不是 E2M1/UE4M3 舍入。这使数值相对上游的原生 FP4 路径发生变化。权重字节保持同一格式。 |
| 稠密 tensor-core 投影与 QSA 中段 | 本地异步 copy 搬运与受支持的整数 MMA 指令 | 必须对照参考实现比较缩放因子的施加与归约顺序，并测试 token 宽度不变性。 |
| 融合的超连接读取 | 用协作式异步 copy 取代 tensor-map 传输；hc8 存储与未融合实现保留 | copy/布局等价与算术不变只是设计意图，而非实测的逐位一致。选择这一读取而非未融合读取可能改变数值。 |
| 融合的 GDN 递推与归一化 | 一个 block 处理四组列，以本地共享内存交换代替分布式共享内存 | 每个 rank 的算术与 norm 部分和的顺序都设计为与参考一致。需要输出/状态一致性与同步性验证。独立的递推/归一化路径保留。 |
| 持久化 GDN 中段 | 不属于 Ada 实现；投影、递推和归一化作为独立 kernel 执行 | 路径选择与归约顺序可能与上游持久化路径不同。请评估质量，不要假定结果相同。 |
| 分阶段搬运的 GPU 专家行 | 用协作式异步 copy 与 block 屏障取代批量 copy 的完成屏障 | 权重字节与行点积算术设计为保持不变。请验证 stage 复用、尾部工作项与对等执行。 |
| 分组专家启动 | 用有序 stream 启动取代依赖启动 | 现有融合的 SwiGLU/q8_1 算术保留；请验证顺序与精确输出。 |
| 标量 CPU 量化器与激活一致性参考 | 主机编译禁用乘加收缩 | Q8_K 在整数舍入前先把乘积舍入到 fp32，与显式舍入的 SIMD 及 GPU kernel 一致。这会使标量量化相对依赖编译器收缩的构建发生变化；请验证精确的平局与极小激活。 |
| 权重预取 | 按缓存行预取取代批量预取 | 不打算改变模型算术。请独立剖析；不作任何速度主张。 |
| 原生 GR RMSNorm 权重预取 | 在支持的行宽下，先于原始输入归约加载 gamma 值，随后在输出遍使用；输入累加循环不变 | `native_gr_norm_parity` 与旧 kernel 逐位一致，并与双精度 CPU 参考比较。450 W 下的开发 A/B：2560 列，基线→预取 2.769→2.493 µs；4096 列，2.615→2.683 µs。结果随宽度变化；不作端到端速度主张。技术做法参考 NInfer Apache-2.0 提交 `9954867a`；未复制任何源码。 |
| 原生 MMQ 专家 gather 与输出 map | 每个原生专家组在其暂存槽位就绪后，由一次分组 gather 处理；identity 行的目标由专家边界推出，而不再依赖每次运行生成的 iota 缓冲 | 权重字节与 MMQ 算术不变。分组/单专家一致性测试通过；IQ4_XS 定向 memcheck 与 racecheck 报告零错误与零危险。一次针对部署包的 memcheck 完成了 511 token 的 prefill 并到达 verify capture，之后超时、未给出 sanitizer 摘要；该项未完成。 |
| 共享专家 gate/up MMVQ 成对启动 | 一次启动用相同的 Q8_1 输入处理两个同格式的 Q8_0 或 Q6_K 投影；每个输出使用现有精确布局的 MMVQ 行与归约代码 | CPU 参考与定向 GPU 一致性测试通过；最终二进制的部署包贪心检查在 3 个请求中共 900 token 通过。成对资格限制在 2–8 列、精确多列模式与共享 stream；其余调用仍走原启动方式。一次匹配的 450 W 开发 A/B 使用部署的 IQ4_XS pack，profile `c31d790a…`，7,461 个常驻槽位，CPU share 0.60，PCIe share 0.55，上下文 262144，int8 KV，spec 4，chunk 32768，seed 1：4096-token prefill 2777.34→2775.68 ms；decode 换算 152.396→153.537 ms，struct 232.277→234.923 ms，prose 152.182→152.530 ms；struct 验证窗口 17.197→17.003 ms，verify 14.096→13.905 ms（58.15→58.81 tokens/s）。这是一次包含若干小幅改动的开发配对；不能据此得出普遍加速。全模型 memcheck 未完成。因没有可用的 BF16 参考运行，BF16 质量对比尚未进行。 |
| decode 的 CPU/PCIe 未命中划分（`ADAMANT_PCIE_BALANCE=1`） | 每层拟合把固定系数 1.85 改为线性池速率系数：从 30 GB/s 时的 1.85 到 12 GB/s 时的 0.8；取整后的 PCIe 份额上限从 0.60 提高到 0.95 | 调度改变哪些专家未命中在 CPU、哪些在 GPU 上运行。专家算术不变，但累加顺序与输出数值可能改变。BF16 质量验证与 Ada 性能测量尚未完成；不作质量或速度主张。 |

### 0.3.1 路径

- **混合 gate/up 层**（`experts.layout` v7：UD-IQ1_S 第 18 层是 IQ2_S gate 配 IQ2_XXS up）。每一半都由同一行代码按其
  自身格式解码；decode glue 与持久化对等 kernel 有混合构建，MMQ prompt 槽位按两种类型的整块补齐，CPU 层让每一半走
  其类型的行 kernel。设计为与统一 kernel 逐位一致；`mixed_gu_parity` 在 sm_89 上逐位通过。
- **对等 prompt 共享**（`prefill::PeerShare`）。第二张卡用与主卡相同的 MMQ 格式和相同的 q8_1 激活，从暂存的分组
  buffer 计算其专家的 prompt 行，再把行送回。设计为与主卡的行逐位一致；上游 `peer_share_parity` 检查 gather 的字节。
  `peer_share_parity` 与 `peer_experts_parity` 需要两张 GPU，单卡上跳过；未在 Ada 上测量。不携带 FP4 流式 chunk 回退，
  因此达到或超过 `stream_all_min()` 的 chunk 仍留在主卡。
- **glue 路径格式**（Q5_K/Q6_K gate/up 行、IQ4_XS gate/up 行、(IQ3_S|IQ4_XS)/Q8_0 对）。行点积与累加顺序相同，只融合
  启动；`dec_moe_parity --kq` 与 `native_grouped_parity --glue` 在 sm_89 上逐位通过。
- **批量 K/V 流式**（`kv_stream` 的分段 resolve，`--kv-resident` 配 `--batch`）。相同的行经分段游标读取；
  `kv_stream_parity` 在 sm_89 上通过。

在部署的 IQ4_XS pack 上，64 个贪心 token（prose ids、int8 KV、expert cache 700、swaps 0、单张 450 W 的 RTX 4090）在
合并前的构建与本构建之间逐位相同。

## prompt 显存与对等执行

整数 MMQ 会为所有路由行设置中间 buffer 的大小。因此，与原生 FP4 实现相比，大的 NVFP4
prompt chunk 可能需要多得多的暂存内存；自动拟合可能因此选择更小的 chunk。请用真实分配尺寸
和长 prompt 来验证，而不是从短样例外推。只有仅路由的 prompt chunk 会与对等卡共享；达到或超过
`stream_all_min()` 的 chunk 留在主卡上。对等独占流式与常规对等放置都需要在第二张卡上分别覆盖。

prompt 切分保留了它基于权重元数据、独立于已移除的执行 API 的 NVFP4 pack 资格检查。其继承
的成本模型与 chunk 阈值没有针对 Ada 上的整数 MMQ 标定过；切分的正确性与性能收益都尚未
确立。

## 所需证据

[贡献要求](../../../CONTRIBUTING.zh-CN.md)适用，尤其是每个新增或改动的 kernel 都需要的一致性
测试。现有 kernel 与 prompt 路径的一致性测试必须对照当前参考 kernel 和独立 CPU 参考通过，
包括非有限输出拒绝、模型几何、窗口宽度和 token 位置。

改动的 copy 与同步路径必须做内存、竞争、初始化和同步检查。资源与图捕获行为必须在目标 GPU
上验证，而不是由一次通过的主机测试推断。

移植完成要求针对 BF16 参考做 decode 路径的 PPL/KLD 与 top-1 一致率
（[方法](quality.md#bf16-参考基准)），外加 prefill/长上下文以及 MTP 开/关检查。报告结果时
请给出确切的模型与样例标识、固定的专家放置、参考的出处、配置，以及施加的 GPU 功率上限。
不要把短时质量样例当作对所有 prompt 路径的覆盖。

门槛尚未完成。kernel 一致性与内存、竞争、初始化、同步检查已在 sm_89 上通过；端到端质量
结果与性能主张尚未确立。目前已取得的发展测量连同其功率上限与配置列在
[README](../../../README.zh-CN.md#ada-上的实测)。
