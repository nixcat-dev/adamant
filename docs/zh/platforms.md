# 平台

[English](../platforms.md)

Adamant 原生运行于 Linux x64 与 Windows x64，硬件为 NVIDIA Ada Lovelace（sm_89）。Linux 是 engine 测试与跑
基准的平台。CI 两个平台都构建，并各跑 CPU 测试。GPU 测试与质量门槛在带卡的机器上运行。

## 系统要求

| | Linux | Windows |
| --- | --- | --- |
| 操作系统 | x86-64，AVX2（有 AVX-512 时也会使用） | Windows 10 1903 或更新，x64 |
| 编译器 | GCC 13 或更新 | Visual Studio 2022（MSVC 19.44） |
| CUDA | 13 | 13 |
| 构建 | CMake 3.24+ | CMake 3.24+ 与 Ninja |
| Python | 3.10+ | 3.10+ |
| GPU | NVIDIA Ada Lovelace（sm_89）；可选第二张 GPU 作为对等专家层（`gpu 0,1`） | NVIDIA Ada Lovelace（sm_89）；可选第二张 GPU 作为对等专家层（`gpu 0,1`） |

## 平台代码

与操作系统相关的代码位于 `src/platform/`（声明在 `include/basalt/platform/`）以及 CMake 的 `WIN32` 块中。它们
之外的 engine 代码不点名任何操作系统或编译器。`tools/check_platform_boundary.sh` 强制执行这一点。每个接缝在一
个声明背后都有 Linux 与 Windows 两份实现，并在 `tests/platform/` 中有一个两边都运行的测试。

| 接缝 | 头文件 | 封装内容 |
| --- | --- | --- |
| 文件 I/O | `file.hpp` | UTF-8 路径、打开、按位置读取、不带缓冲的文件描述符 |
| 不带缓冲的读取 | `direct_file.hpp` | 异步的不带缓冲文件读取（n-gram 表） |
| 文件视图 | `mapped_file.hpp` | 映射文件、预取与释放提示、加锁 |
| 内存 | `memory.hpp` | RAM 探测、页面加锁、对齐与大页分配 |
| CPU | `cpu.hpp` | 拓扑 |
| 进程 | `process.hpp` | 启动、等待、终止、管道、信号 |
| 系统 | `system.hpp`、`dynlib.hpp` | 进程内存、磁盘查询、后台全速运行、NVML 加载器 |
| 编译器 | `compiler.hpp`、`format_check.hpp` | CPUID、位扫描、spin 提示、fp16 舍入、printf 格式检查 |
| Win32 | `win.hpp` | 唯一引入 `windows.h` 的地方 |

接收路径的程序带有 manifest（`src/platform/basalt.exe.manifest`），把进程代码页设为 UTF-8。超过普通 Win32
长度限制的路径以其 `\\?\` 形式打开，因此系统长路径策略无论开关都能工作。

## Windows 限制

- WDDM 下 pinned memory 受驱动上限约束。整个 arena 无法注册时，pinned 切片被限制在 GPU 的共享内存额度内
  （`RAM/2 - 8 GiB`），启动说明会写明被压缩到了哪里。
- 大页需要"锁定内存页"特权，默认只授予 Administrators；锁定常驻专家副本需要"增加进程工作集"，该项授予
  Users，因此无需提权即可工作。engine 启用其令牌持有的那一项。没有大页时退回 4 KiB 页，并在启动时说明。
- 共享 arena 文件与 cgroup 配额不存在。engine 拒绝前者，并报称没有容器限制。
- n-gram 表（`--ple-io direct`）经由文件缓存读取。文件被映射或经由缓存读取时，NTFS 对它的无缓冲读取一次只
  完成一个，而模型文件两者皆是。读取带进来的表页属于缓存内存，程序需要时 Windows 会让出。
- `basalt serve` 把 Python 服务器作为子进程运行，并以其退出码退出；Linux 上是替换进程。
- 连接了显示器的 GPU 受驱动的超时恢复（TDR）约束。若长 kernel 被重置，请调高
  `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` 下的 `TdrDelay`。
- 同时驱动桌面的卡，其空闲 VRAM 会在不同次启动间浮动几个 MiB，自动定容的专家缓存因此可能落在不同的槽位数
  上，贪心输出可能因运行而异。要可复现的输出（`tools/greedy_identity.py`），请固定 `expert-cache` 并调高
  `vram-reserve-mib`。
- Windows 上没有测量过质量。没有它的困惑度或 KL 散度记录；编译器一致性检查见
  [precision/builds.md](precision/builds.md)。没有面向 Windows 的门面基准数字。

Windows 依靠社区支持。请把 Windows 问题报为打了 `windows` 标签的 issue。

## 在 Windows 上构建

在 Developer 命令提示符中运行（或先调用 `vcvarsall.bat x64`），让 `cl.exe` 在 `PATH` 上：

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build -L cpu -j 8 --output-on-failure
```

若测试创建样例失败，把 `TEMP` 与 `TMP` 指向构建目录树内的一个目录。`cpu_quota_test` 与 `pool_affinity_test`
触及仅 Linux 的接口，不构建。

CUDA 13.2 及更新版自带的 CCCL 头文件在 cl.exe 的传统预处理器下会 `#error`（C1189）；CMakeLists 给 nvcc 的主机
编译器传 `/Zc:preprocessor`，已在 CUDA 13.4 与 MSVC 19.44 上验证。在本 CMakeLists 之外编译这些头文件的任何东西
都需要同一个 flag。
