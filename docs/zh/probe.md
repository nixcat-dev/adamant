# basalt-probe

[English](../probe.md)

`bin/basalt-probe` 把引擎作为子进程启动（`--serve`，与 `basalt-bench` 相同），向它索取本机的速率，在三个标准 fixture 上测量冷启动 decode 速度，并打印一份带建议值的简短报告。它是手动的：不保存任何东西（`--expert-profile-save` 文件只被比较，从不写入）。它是一个独立的程序，不是 `basalt` 的子命令。

```
basalt-probe --model PACK.basalt -- ENGINE FLAGS...
```

| 选项 | 含义 |
|---|---|
| `--model FILE` | 模型文件 |
| `--engine BIN`、`--fixtures DIR`、`--logs DIR` | 与 `basalt-bench` 相同 |
| `--gpu A,B` 或 `auto` | 选卡，与服务器相同；默认 GPU 0 |
| `-- ARGS` | 引擎 flag，与 `basalt-bench` 相同；报告会打印它用到的 flag。`--expert-profile-save FILE` 只被比较，从不写入 |

## 它做什么

1. 启动引擎，用 `PROBE rates` 读取速率：GPU 拷贝与链路速率、主机读取速率，以及每个工作线程数下的 CPU 专家耗时。
2. 各跑一次 `conv`、`struct` 与 `prose`，贪心，各 400 token。这些是**冷**速度：加载后的第一个请求就计数。
3. 测量它的建议值（见下）。

token 预算也是"冷"的一部分：tier 会持续适应，更长的请求会越读越快。

## 建议值

只有当某个取值比内置默认值好出超过实测噪声时才打印：

- **`--pool-workers N`**：与实测最快 CPU 专家耗时相差 2% 以内的最少工作线程数。
- **`--adapt-every`、`--adapt-swaps`、`--adapt-decay`**：从起始放置开始、以先 prose 后 struct 的切换来尝试（`PROBE adapt`）。把 `64 / 256 / 0.7`、`1 / 160` 以及一个小轮次（swaps 12，即默认节奏）与引擎自身的设置对比；引擎自身跑两次以确定噪声带。
- **`--prefill-cpu-share F`**：在份额 0、1 与平衡点下的 8192-token chunk。至少快 8% 才打印。
- **`--expert-profile-save FILE`**：当保存的 profile 快得超出噪声带时建议。

## 引擎命令：`PROBE rates`

在 `--serve` 模式下，在请求之间向 stdin 发送 `PROBE rates`，大约半秒后返回 `PROBE begin rates v1`、若干 `PROBE <section> key=value` 行（section 有 `gpu`、`host`、`pack`、`link`、`gpu_mem`、`host_mem`、`cpu_expert`、`peer`、`copy`）以及 `PROBE end ms=`。它不改变引擎状态。`PROBE set workers=K share=F` 设定池的工作线程数与 prompt CPU 份额；`PROBE adapt [every=N] [swaps=N] [decay=F] [profile=FILE]` 重置 tier 并为下一个请求设定自适应设置。
