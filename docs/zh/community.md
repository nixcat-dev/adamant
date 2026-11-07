# 社区实测

[English](../community.md)

由用户提交的 benchmark 运行，全部用同一预设采集以便各行可比，存放在 `docs/community/benchmarks.json`。该文件里现有的行是 Basalt 的 Blackwell 运行，随 fork 一并继承；Adamant 还没有自己的公开发布页面。

## 添加你的运行

1. 停掉所有正在运行的 Basalt 服务器（GPU 必须空闲），然后运行 `basalt-bench --community --model PACK`。它会启动自己的引擎（三次，每次测量一个），在每次启动期间通过 stderr 显示加载进度，把它的行加进 `docs/community/benchmarks.json` 并打印出来。它使用每一块可见的 Ada GPU：第二块卡成为对等卡，`CUDA_VISIBLE_DEVICES=0` 则得到单卡行。
2. `--submitter NAME` 是必填项：你想显示的名字（不含空格的一个名字，最多 40 个字符；不做核实）。`notes`（最多 200 个字符）是可选的，需手动添加。
3. 检查 `docs/community/benchmarks.json` 中的这一行，提交它并开一个 pull request。如果 bench 无法写入文件，它会打印这一行并说明粘贴到哪里。

这一行记录 `model`（`Qwen3.8-Flash-Next`，即模型的 adapter 为 pack 的架构报告的名字，而不是文件名）和 `kv`，即这次运行使用的 KV 缓存格式：预设会要求 `--kv int8`，模型没有这个选项时就用模型自己的格式。

你需要足够的空闲 RAM，保证没有任何专家从 SSD 读取；bench 会检查这一点，否则拒绝打印结果行。

可选字段，由 `--community` 在引擎启动前采集（单独打印这些项请用 `basalt-bench --hardware`）：

- `cpu`：处理器及其物理核心数，例如 "AMD Ryzen 9 9950X, 16 cores"。
- `ram_gbps`：宿主 RAM 读取速度，GB/s，每个物理核心一个线程流式读取一个 2 GiB 缓冲区，多轮取最佳。
- 每块 GPU 的 `pcie`：驱动报告的链路，例如 "5.0 x16"：最大世代（当前世代在空闲时会降档）和当前宽度（最大宽度取决于卡而不是插槽，因此两个卡共享一个插槽时读作 x8）。
- 每块 GPU 的 `h2d_gbps`：pinned 内存的 host-to-device 拷贝速度，GB/s，多次 256 MiB 拷贝取最佳。

bench 按统一格式写这个文件。手动修改后，运行 `python3 tools/check_benchmarks.py --fix docs/community/benchmarks.json`；它会按统一格式重写文件并做校验。CI 运行同一个检查器但不带 `--fix`，任何其他格式都会导致失败。纯 benchmark 的 pull request 会跳过完整构建。
