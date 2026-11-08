# basalt-probe

[English](../probe.md)

`bin/basalt-probe` 把 engine 作为子进程启动（`--serve`，与 `basalt-bench` 相同），向它索取本机速率，在三个标准
样例上测量冷启动 decode 速度，并打印一份带建议值的简短报告。它完全手动：不打印也不保存任何东西
（`--expert-profile-save` 文件只被比较，从不写入）。它是一个独立程序，不是 `basalt` 的子命令。

```
basalt-probe --model PACK.basalt -- ENGINE FLAGS...
```

| 选项 | 含义 |
|---|---|
| `--model FILE` | 模型文件 |
| `--engine BIN`、`--fixtures DIR`、`--logs DIR` | 与 `basalt-bench` 相同 |
| `--gpu A,B,..` 或 `--gpu auto` | 选卡，与服务器相同（[GPU](configuration/gpus.md)）；默认只有 GPU 0 |
| `-- ARGS` | engine flag，与 `basalt-bench` 相同（`--kv`、`--spec`、`--expert-cache`……）。除了以下两项外不添加任何东西：未给出 `--max-context` 时补一个能装下样例的值；把 `--adapt-swaps` 取到不小于它要尝试的最大换入换出次数（RAM 副本的空位在启动时定容；probe 在加载后立刻设定给出的或内置的次数，所以其余数字都按它来）；`--expert-profile-save FILE` 会被移出并比较 FILE。报告打印产出这些数字的 flag |

## 它做什么

1. 启动 engine，等待 READY。
2. 发送 `PROBE rates`（[running](running.md)），把速率打印成报告的第一行：每张 GPU 的显存拷贝速率、第一张 GPU
   的主机到设备链路速率，以及 engine 当前工作线程数下的主机读取速率。
3. 读取各层的放置：主 GPU 的专家槽位来自 engine 的 INFO 行，对等卡的来自其日志。
4. 按此顺序各跑一次 `conv`、`struct`、`prose`，贪心，各 400 个 token。这是**冷启动**速度：加载后的第一个请求
   就计数，不丢弃、不重复。
5. 测量它要建议的各项取值（见下），自适应层的最后测。

token 预算也是"冷"的含义之一：engine 边 decode 边持续调整它的专家层，所以更长的请求会越读越快。速度是在
engine 的上下文长度下测的。样例是短 prompt，所以测量是在样例深度上、上下文已分配但 KV 并未填满到该深度。

## 建议值

对于那些"按其自己的测量在这台机器上胜过 engine 内置默认值"的设置，probe 会打印 `suggested:` 行，每条给出
flag 与取值，附一个简短的实测数字。细节写到 stderr。任何东西都不会被应用或保存；等于默认值的建议不打印。

- **`--pool-workers N`**：CPU 专家的每专家耗时（每次调用 8 个专家）在 1、2、4……以及每个池工作线程数下测量
  （`PROBE rates`）。取的是收益递减点：与最快的实测线程数相差 2% 以内、线程数最少的那一档。只有当它低于池的
  线程数时才打印，形式为 `(N us/expert, best M us)`：选取档的 CPU 耗时与最快档的耗时。按选取档 decode 与默认
  值打平；主机读取速率本身更早饱和，故不采用。
- **`--adapt-every`、`--adapt-swaps`、`--adapt-decay`**：`PROBE adapt`（[running](running.md)）把专家层放回加载
  后的状态，并设定其中一项设置；然后贪心运行 `prose`（1200 token）与 `struct`（400 token）——一次冷启动与一次
  话题切换——以 token 除以 decode 时间打分。设置组合：engine 自己的（最先跑、最后再跑一次：两次之差就是噪声
  带，至少 3%）、`64 / 256 / 0.7`、`1 / 160` 以及一个小轮次（swaps 12，默认节奏）。只有当某个组合超出噪声带地胜过两次 engine 设置的均值时才打印，
  给出有差异的 flag 与 `(adapt test +N%)`。带 `--expert-profile-save FILE` 时，engine 自己的设置再多跑一次、从
  FILE 放置；当这样快得超出噪声带时建议 `--expert-profile-save FILE`，慢时建议 `no --expert-profile-save`。报告
  的 `adapt test` 行给出每个速率与噪声带。
- **`--prefill-cpu-share F`**：engine 在份额 0（全 GPU）与 1（全 CPU）下 prefill 8192-token 的 chunk
  （`PROBE set share=F`，随后一个单 token 请求）；平衡点是 `t0 / (t0 + t1)`，舍入到 0.05。之后按平衡点与按默认
  值（ABBA）各跑一个 chunk，前者必须至少快 8%，因为一次真实 chunk 带着约 5% 的噪声；否则什么都不打印。该行
  显示实测收益，如 `(prefill +9%)`。份额改变 CPU 舍入哪些专家，所以它按机器固定，运行期绝不随计时变动。

### 它覆盖哪些

probe 只在"本机上的快速测量能定出取值、且实测运行表明这个选取确有作用"的地方给建议。其余默认值依赖机器的
设置，以及它不碰的原因：

| 设置 | 覆盖 | 原因 | 增加的秒数 |
|---|---|---|---|
| `pool-workers` | 是 | 每个工作线程数下的 CPU 专家耗时，来自 `PROBE rates` | 0（属于速率测量的一部分） |
| `prefill-cpu-share` | 是 | 在份额 0、1、平衡点与默认值下对 8192-token chunk 计时 | 这些 chunk，几秒钟 |
| `adapt-every`、`adapt-swaps`、`adapt-decay` | 是 | 在已加载的 engine 里从起始放置测每个设置（`PROBE adapt`），无需重启；一次重置后的得分与全新 engine 头几个请求在噪声内一致（[自适应层](configuration/adaptive.md)） | 35（两卡）到 46（单卡）：5 次 7 到 9 s 的运行，每次重置不到 1 s |
| `expert-profile-save` | 是 | 保存的文件对照 engine 的 profile，同样设置、同样序列 | 其中一次运行 |
| `peer-adapt-swaps` | 否 | 在所有测过的工作负载上，更少的对等换入换出没有损失；它跟随 `adapt-swaps` | 0 |
| `vram-reserve-mib`、`peer-reserve-mib` | 否 | 更低的预留多出约 1% 的槽位，在运行噪声之内；它们防的是换页，不足时启动会说明 | 0 |
| `spec` | 否 | 启动时固定（没有逐请求取值），spec 7 对 8 差 1-2%，在单次运行的噪声之内 | 0 |
| `prefill`、`expert-cache` | 否 | 启动时按卡能容纳与借出的量定容 | 0 |
| `peer-exclusive`、`resident-budget-gib`、`pool-affinity` | 否 | 启动时按 RAM 与 CPU 拓扑选定（`[server] auto`） | 0 |
| `suffix-draft`、`mtp-max-t`、`mtp-window`、`short-read` | 否 | 是模型与负载的属性，不是机器的 | 0 |
| `kv`、`max-context` | 否 | 是内存与精度上的取舍，不是速度 | 0 |

探测期间，RAM 副本的空位按四乘 256 个专家保留，而不是四乘给定的次数：在 NVFP4 pack 上比内置的 96 多用约
1.6 GiB RAM。

engine 的 `PROBE set workers=K share=F` 在请求之间为随后的请求设定池的活跃工作线程数与份额；probe 使用它，
用完 engine 就停掉。

## 报告

```
basalt-probe
  5090 1541 GB/s | 5060 Ti 388 GB/s | PCIe 28.9 GB/s | RAM 58 GB/s
  experts 5090 9553 / 5060 Ti 5711 / RAM rest | KV int8 | ctx 262k
  decode cold  prose 194 conv 255 struct 223 tok/s (measured)
  adapt test  1/96/0.85 284 286 | 64/256/0.70 203 | 1/160/0.85 285 | saved profile 286 tok/s (band 3.0%)
  suggested: --pool-workers 8 (52 us/expert, best 51 us)
  suggested: --prefill-cpu-share 0.45 (prefill +9%)
  config: --gpu 0,1 --kv int8 ...
  wall  97.3 s

  Measured on this machine. Hand-tuned settings can still beat these.
```

`wall` 行是整次运行的墙钟时间，包括 engine 的启动。（上面的数字只用于展示版式。）计时与 engine 进度写到
stderr。
