# 运行服务器

[English](../running.md)

设置：[配置](configuration/index.md)。诊断标志：`bin/basalt --help`。多个模型：[模型与路由](configuration/models.md)。

## 启动

页面立刻就绪，模型在它后面加载。加载权重时顶部显示 `Loading` 和进度条上的百分比；在选择器里选一个模型就是加载它。
`/health` 和 `/metrics` 显示加载状态。启动时加载失败会结束服务器。加载期间请求会等待，失败则得到 503。引擎从可执行文件自身的位置找到安装根目录，因此 `data/` 必须在 `bin/` 旁边（或传入 `--expert-profile`）。

## 并发请求：`--parallel N`

`--parallel N`（1 到 8）在批处理槽位里同时服务 N 个请求。更多请求按到达顺序等待。`--max-context` 是共享的：
每个槽位持有 `max-context / N` 个 token（`--parallel 4 --max-context 262144` 得到四个 65,536 的槽位）。比槽位更长的请求
会以 400 拒绝，并指出长度。`--slot-full-context` 让每个槽位都持有完整的 `--max-context`，代价是 N 倍的 KV 缓存。

每个槽位保留两个 checkpoint（系统提示词末尾、最后一轮），因此共用长系统提示词的请求只读一次。

### 与解码槽位并行的提示词

16,384 个 token 以内的提示词一次读完，槽位等待几秒。更长的提示词在自己的线程上与它们并行读取，槽位继续解码。期间到达的
短请求排在当前块的末尾先执行。

## 控制台日志

每行都以 `[server]` 开头，值是 `key=value`。启动时打印一份图例、每个模型一行，以及 `ready`、`loaded`、`kv`、`slots`。
读取提示词时，槽位打印 `pp=<已读>/<总数>` 和速率。请求生成时每 2 秒出现一行 `live`。每个请求以一行 `done` 结束：

```
[server] s1 done  pp=96/89c 0.1s  tg=1720 119.6t/s acc=65% dl=2.9  ttft=0.1s  end=stop  hit=89.7%  pcie=39.3%
```

- `pp=<提示词 token>/<缓存>c`：速率只统计新读的 token。
- `acc`：被接受的 draft 除以提供的 draft。`dl`：每个窗口的 token 数。请求没有 draft 时两者都不出现。
- `end`：`stop`、`length`、`cancelled`、`disconnect` 或 `error`。
- `hit`：专家缓存命中率。`pcie`：路由专家中经 PCIe 读取的比例。

引擎自己的 `basalt ...` 行写入 `~/.cache/basalt/<id>/engine.log`，每个模型文件一个目录（`<id>` 是从文件中取出的 16 位
十六进制数）；引擎启动失败时 `[server] engine log: PATH` 会指出它。该目录也存放会话缓存。不会从模型文件解包任何
内容：服务器从文件 section 读取分词器，引擎与 `basalt-vision` 就地读取其中的 vision projector。

## 统计页面

服务器地址提供一个页面，每秒从 `/metrics`、`/slots` 和 `/api/logs` 更新：GPU 负载、功耗、VRAM 和温度、RAM 与磁盘、
带 60 秒图表的 decode 与 prefill 速率、槽位、最新请求，以及控制台行。

## 磁盘上的会话：`--conversation-cache-disk-mib`

`0` 表示关闭。给定预算后，至少 8,192 个 token 的会话在离开 RAM、离开会话或退出时写入模型缓存目录下的 `conversations`。
只存系统提示词和最后三轮的状态。文件与创建时的模型和设置绑定，并按哈希校验；坏文件会被删除。目录受预算限制
（最久未用的先删），并且每次写入总是给卷留下 4 GiB 空闲。重启后，以已存会话开头的提示词会把它读回来。

## 睡眠：`--idle-sleep`

在没有请求达到给定秒数后，引擎释放它的 VRAM（驱动把 GPU 状态移到主机内存）。进程和它在 RAM 里的专家副本保留。下一个请求
几秒内唤醒它。仅 Linux。当 GPU 状态放不进空闲 RAM（还要留 8 GiB）时，服务器保持清醒。`--idle-unload` 改为结束进程。

## 内存下限

Basalt 会拒绝让机器没有内存可用的设置，并且绝不会缩小你要求的量：

- `--vram-reserve-mib` 低于 256 时在 Windows 和 WSL2 上被拒绝（Linux 上 128）：空闲 VRAM 更少会让驱动把 GPU 内存换页。
- `--ram-reserve-mib`（默认 2048）是留给系统其余部分的空闲 RAM。引擎先分配自己的 host 缓冲，或者在之后页锁定时占住它们的
  RAM；专家层随后取得空闲 RAM 减去保留量、再减去驱动对其 pin 住页面的跟踪（启动日志里的 `RAM floor ...`）。当一部分专家仍留在文件里时，
  专家层还会给读取这些专家所经过的操作系统文件缓存留一个下限：一个提示词 chunk 的文件读取量，加上这些专家中两层的份额，
  至少 4 GiB。在很小的保留量旁边闲置的程序可能被换出。
- 服务期间 RAM 守卫守住保留量：低于它时，把会话停放到磁盘或丢弃。请求最多等 30 秒让 RAM 回来。只有在 RAM 危急时（空闲低于
  1 GiB，或在内存压力下低于保留量）运行中的请求才会以错误停止。`--ram-reserve-mib 0` 关闭守卫。
- SSD 上的专家给出一条警告，所有模型相同：`N experts on SSD (X GiB) · free X GiB RAM`。控制台重复它，统计页面把它的
  Experts 块变为琥珀色。
- 磁盘层保持比保留量高 1 GiB 的下限。
- 会让空闲低于该下限的 `--resident-budget-gib` 会被拒绝，并给出最大的安全值。

## 视觉：`--vision`

`--vision off | dynamic | cpu | <GPU 编号>`（配置：`server` 中的 `"vision"`）。编码器是 Basalt 自己的，所有模式给出的嵌入
逐位相同。所有模式都就地读取模型文件 `vision/mmproj` section 中的 projector。文本仍可能因模式而不同：sidecar 的 VRAM
开销会缩小专家缓存，而哪些专家在 GPU 或 CPU 上运行会改变 logits 的最后几位，所以贪心答案可能在某些 token 后分叉
（`dynamic` 以不同的缓存大小也是如此）。

| 模式 | 编码器运行在 | VRAM 开销 |
| --- | --- | --- |
| `off` | 不运行 | 无 |
| GPU 编号 | 那张卡上的 sidecar `bin/basalt-vision` | 约 2.2 GiB 的专家缓存 |
| `dynamic` | 引擎内，主 GPU 上 | 静息时不占（0.85 GiB pinned RAM） |
| `cpu` | CPU 上的 sidecar | 无 |

`dynamic` 为每张图借用缓存的最后几个槽位，之后再填回（约 70 ms），所以之后缓存完全一致。请求可以命名哪些图像：
[图像来源](configuration/image-sources.md)。图像不能放在系统提示词里。
