# 为 Adamant 贡献代码

[English](CONTRIBUTING.md)

Adamant 的目标是在 NVIDIA Ada GPU 上运行 Qwen3.8-Flash-Next。Ada 支持仍在开发中；继承来的 Blackwell 实现不能证明 Ada 上的正确性或性能。下面的[一致性、质量与基准测试规则](#每项变更必须包含的内容)规定了一项变更需要什么，[Ada 平台的执行与验证](docs/zh/precision/ada.md)记录了尚待测量的路径。

## 适用范围

- **只支持 Ada（sm_89）上的 CUDA。** 不支持 ROCm/HIP、AMD、Intel/SYCL/oneAPI、Apple/Metal，也没有仅 CPU 的主路径（驻留在 RAM 中的 CPU 专家和 `--vision cpu` 仍受支持）。不做多架构兼容层。不支持的指令必须换成经过验证的 Ada 实现，而不是藏在可启动的空 kernel 或 trap stub 后面。除非明确批准了范围变更，否则保持模型格式覆盖面。
- **Linux 和 Windows x64 都受支持；在 Linux 上测试。** CI 在每个 pull request 上编译并对 Windows 做 CPU 测试。Windows 使用与 Linux 相同的 GPU 角色（主卡加专家层）运行，不作性能声明，Windows 的限制见 `docs/platforms.md`。变更不得移除平台接缝（`src/platform/`、CMake `if(WIN32)`）。仅影响 Windows 的问题打 `windows` 标签。官方基准数字以维护者的 Linux 运行为准。
- **引擎代码与平台无关。** 在 `src/platform/` 和 `include/basalt/platform/` 之外，不得出现 `_WIN32`、`_MSC_VER`、`__linux__`、`<windows.h>`、`<sys/mman.h>`、`<unistd.h>` 或裸 OS 调用（mmap、madvise、pread、pthread/affinity、fork/exec/pipe、dlopen、读 /proc 或 /sys）。要调用平台层；缺少接缝时就在平台层里补上，Linux 和 Windows 各一个实现体，Linux 实现体就是你本来会写的那段代码，并在 `tests/platform/` 里加一个在两个平台上都运行的测试。路径一律 UTF-8，并经过平台的文件调用。`tools/check_platform_boundary.sh` 会强制执行这条规则。
- **引擎日志行带有自己的级别。** 写到 stderr 的一行要经过 `include/basalt/log.hpp`（`basalt::log::info`、`warn` 或 `error`），输出格式为 `basalt <area>: [warning: |error: ]<text>`；服务器和终端只从这个标记读取级别，绝不从文字内容猜。warning 表示回退或功能丧失，error 表示失败或拒绝，其余都是 info。引擎代码在任何其他地方提及 stderr 时，`tools/check_stderr_logging.sh` 会报错。
- **不向仓库添加 `.bat` 或 PowerShell 脚本。** 在 shell 里调用工具链自带脚本（比如 `vcvarsall.bat`）的成文构建配方不算脚本。
- **只保留一条快路径。** 不做逐用户的调优标志，也不在运行时选择替代 kernel。胜出的变更取代旧路径；没胜出的就不合并。A/B 与诊断标志默认关闭，也不作为设置写进文档。面向用户的设置仅限于模型、上下文、多 GPU 和服务选项。
- **不设环境变量开关。** 引擎读取的一切都是命令行标志或配置项；诊断项（profiling、计时、日志）也是标志，默认关闭。

## 每项变更必须包含的内容

1. **一致性测试（parity test）**：每个新增或改动的 kernel 都要有：随机输入加上真实模型几何，对照 (a) 它所替换的当前 Basalt kernel 和 (b) 该运算的纯 CPU 参考实现（先反量化，再以双精度累加）比较。报告相对两者的最大绝对误差和相对误差。要求按位保持不变的 kernel，必须与当前 kernel 的冻结输出哈希一致。一个 token 的输出不得取决于 batch/窗口宽度或该 token 在其中的位置；测试必须检查这一点。
2. **一份 profile。** 给出 kernel 的 microbench 行：每次调用的 µs 和达到的 GB/s（或 TFLOP/s），相对于它必须搬运的字节数或 FLOPs。给出 `basalt-bench` 的端到端数字：每个 decode 窗口的毫秒数和 passes/s，而不只是 tok/s——draft 的运气会让 tok/s 波动 ±3-5%。
3. **一次质量检查：**
   - 输出按位一致：给出相对上一个构建的贪心（greedy）一致性检查。要固定专家层（pin），因为 GPU 模块大小的变化会改变自动确定的专家层规模，从而改变舍入。
   - 改变了数值行为（新的归约顺序、精度、量化）：通过 decode 路径运行对照 BF16 参考的困惑度/KLD 关卡（[docs/precision/quality.md](docs/zh/precision/quality.md)）。把 PPL、平均 KLD 和 top-1 一致率与上一个构建的数字并列报告。大的改动需要完整的 100-chunk 运行，而不是只用 20-chunk 的快速关卡。
4. **基准测试规范：**
   - 每个性能数字都要注明强制生效的 GPU 功耗限制。A/B 运行之间要保持一致；不要把上游的 400 W 约定悄悄套到另一台机器上。
   - A/B 比较要在同一个会话内交替运行两个构建。不同会话里取的数字会漂移，不具可比性。
   - 保持 `basalt-bench` 定义的预热和 fixture 顺序。

## 代码风格

- kernel 写成 `__device__` 阶段函数，把 block/warp 索引作为参数传入，外面包一层薄薄的 `__global__` 包装，以便日后融合成持久化 kernel。
- 注释要简短、无状态：说明代码做什么、为什么。注释里不写硬件名、实测数字、日期或历史。那些属于 commit message 或 PR。
- 不要使用 TMA、分布式共享内存、thread-block cluster、programmatic dependent launch 或原生 FP4 指令：目标硬件不支持。
- 保留改编过的第三方代码的许可证头（ggml/llama.cpp 是 MIT）。

## 提交与 pull request

- commit 要小而内聚，message 风格为 `area: what it does`。每个 commit 都必须能构建。
- 绝不 amend 或 force-push 共享分支；要加新 commit。
- 仓库里不放笔记、草稿文件或生成的报告。
- 文档：单个文件不超过 300 行；更长的主题拆成一个文件夹里的多个文件；文档保持简短、无状态、不重复。
- 文档是双语文档：每个英文文档都有一个简体中文对应文件（`docs/` 镜像到 `docs/zh/`，另有 `README.zh-CN.md`、`CONTRIBUTING.zh-CN.md`），在顶部互相链接。一对文件要在同一个 commit 里一起改。完整规则见 `AGENTS.md`。
- PR 描述要列出：改了什么、一致性测试结果、microbench 和 `basalt-bench` 的前后对比（注明功耗限制），以及质量检查结果。

## 其他贡献方式

- 不能贡献代码？把你硬件上的 benchmark 运行加进社区表格（细节见[社区实测](docs/zh/community.md)）：
  1. 运行 `basalt-bench --community --model PACK --submitter YourName`。它会把这一行加进 `docs/community/benchmarks.json`（[构建](docs/zh/building.md)）。
  2. 检查这一行。手动修改后，用 `python3 tools/check_benchmarks.py --fix docs/community/benchmarks.json` 格式化并校验。
  3. 提交它。
  4. 开一个 pull request；`pages` 工作流会运行同样的检查。
  5. 合并后，这一行就会出现在实测页面上。
- 有空闲 GPU 时间的 Ada（sm_89）机器能帮忙验证这个移植；多卡系统还能覆盖对等专家执行。
- 几小时的机器远程访问也能加快开发。
- 或者通过 GitHub Sponsors（仓库的 Sponsor 按钮）赞助项目：这笔钱用来支付 agentic 开发的 token，以及在不同硬件上租用时间做测试。

## 许可证

Basalt 采用 MIT 许可证，基于 Strata（MIT，Copyright (c) 2026 Niko1221 and the Strata contributors）。贡献代码即表示你同意你的贡献按相同条款授权。见 `LICENSE` 和 `THIRD_PARTY_LICENSES`。
