# 平台

[English](../platforms.md)

Adamant 原生运行于 Linux x64 与 Windows x64，硬件为 NVIDIA Ada Lovelace（sm_89）与 Ampere（sm_86）。Linux 是 engine 测试与跑
基准的平台。CI 两个平台都构建，并各跑 CPU 测试。GPU 测试与质量门槛在带卡的机器上运行。sm_86 构建可以编译；Ampere 上的一致性、质量与性能仍待测量。

## 系统要求

| | Linux | Windows |
| --- | --- | --- |
| 操作系统 | x86-64，AVX2（有 AVX-512 时也会使用） | Windows 10 1903+ x64 |
| 编译器、构建 | GCC 13+、CMake 3.24+ | MSVC 19.44（VS 2022）、CMake 3.24+、Ninja |
| CUDA、Python | 13、3.10+ | 13、3.10+ |
| GPU | NVIDIA Ada Lovelace（sm_89）或 Ampere（sm_86）；可选第二张 GPU 作为对等专家层（`gpu 0,1`） | NVIDIA Ada Lovelace（sm_89）或 Ampere（sm_86）；可选第二张 GPU 作为对等专家层（`gpu 0,1`） |

## 发布归档

`tools/package_release.py` 把已构建的目录树打包成 `basalt-vX.Y.Z-linux-x64.tar.gz` 或 `basalt-vX.Y.Z-windows-x64.zip`：
`bin/`（`basalt`、`basalt-bench`、`basalt-probe`、`basalt-vision`）、`serve/`、`tools/`（服务器导入的部分）、`data/`、`docs/`、
`requirements.txt` 和许可证。归档里不含 CUDA 库，也不含 Python 环境。需要：

- R580 分支或更新的 NVIDIA 驱动。归档用 CUDA 13.0.2 构建，其 Linux 最低版本是 580.95.05；NVIDIA 自 13.0 起不再给出
  Windows 最低版本，所以在 Windows 上请用较新的驱动。
- 库路径上的 CUDA 13 运行时库（cudart、cuBLAS）（Windows 上在 `PATH` 中）。
- Python 3.10+，并在归档顶层目录的 `.venv` 中装好依赖；`bin/basalt serve` 在它存在时会使用它：

```sh
cd basalt-vX.Y.Z-linux-x64                   # tar xzf the archive first; on Windows unzip it
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Windows 上：`py -3 -m venv .venv`，然后 `.venv\Scripts\pip install -r requirements.txt`。之后从模型下载那一步起照 README 的
快速开始做：`bin/basalt serve --config basalt_config.json`。请在顶层目录运行 `basalt-bench`（它读取 `data/bench`）。

## 平台代码

与操作系统相关的代码只存在于 `src/platform/`、`include/basalt/platform/` 以及 CMake 的 `if(WIN32)` 块中。其他任何
地方都不得出现 `_WIN32`、`_MSC_VER`、`__linux__`、`<windows.h>`、`<sys/mman.h>`、`<unistd.h>` 或裸 OS 调用
（mmap、madvise、pread、pthread/affinity、fork/exec/pipe、dlopen、读 /proc 或 /sys）。`tools/check_platform_boundary.sh`
会强制执行这一点。每个接缝都有 Linux 与 Windows 两份实现，Linux 实现就是你本来会写的那段代码，并在 `tests/platform/`
中有一个两边都运行的测试。路径一律 UTF-8，并经过平台的文件调用。

## Windows 限制

- WDDM 下 pinned memory 受驱动上限约束：整个 arena 无法注册时，pinned 切片被限制在 `RAM/2 - 8 GiB`。
- 大页需要"锁定内存页"特权（Administrators）。没有该特权时 engine 使用 4 KiB 页，并在启动时说明。
- 没有共享 arena 文件，也没有 cgroup 配额。`--ple-io direct` 经由文件缓存读取。
- `basalt serve` 把 Python 服务器作为子进程运行。
- 连接了显示器的 GPU 可能触发驱动的超时恢复（TDR）。请调高
  `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` 下的 `TdrDelay`。
- 同时驱动桌面的卡，其空闲 VRAM 会在不同次启动间浮动，贪心输出可能因运行而异。要可复现的输出，请固定
  `expert-cache` 并调高 `vram-reserve-mib`。
- Windows 上没有测量过质量，也没有面向它的官方基准数字。请把 Windows 问题报为打了 `windows` 标签的 issue。

## 在 Windows 上构建

在 Developer 命令提示符中运行（或先调用 `vcvarsall.bat x64`）：

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build -L cpu -j 8 --output-on-failure
```

若测试创建 fixture 失败，把 `TEMP` 与 `TMP` 指向构建目录树内。CUDA 13.2+ 的头文件需要 `/Zc:preprocessor`；CMakeLists
会给 nvcc 的主机编译器传这个 flag。

## 两个构建下的数值

CPU 专家 kernel 在 Windows 上由 MSVC 编译，在 Linux 上由 GCC 编译。同一个 commit 下，`expert_parity` 与
`pool_test` 在两个平台上都是 0 失败，pool 输出与串行运行逐位一致。

贪心 token 在两个构建之间会有差异：固定专家层、关闭自适应换入换出时，`struct` 在 300 多个 token 上一致，
`conv` 在第 23 个 token、`prose` 在第 44 个 token 分叉。同一个构建对自身运行时是一致的，因此差异来自主机
编译器的 CPU 代码（占专家行的 78%）。Windows 构建的质量尚未运行（[精度](precision/quality.md)）。
