# 运行服务器

[English](../running.md)

[配置](configuration/index.md) 中各项设置背后的详细背景说明：并发请求、KV cache、内存下限与图像输入。诊断
flag：`bin/basalt --help`。

## 启动与加载

服务器与页面先起，模型在其后加载。加载期间（启动时、首个 `--lazy` 请求、`POST /load`、空闲卸载之后）页眉显示
`Loading weights`，带百分比与每个设备的 GiB；`--idle-sleep` 唤醒时显示 `Waking up`。`/metrics` 的 `live.state`
为 `loading`，`live.load.kind` 为 `weights` 或 `wake`；`/v1/status` 与 `/health` 带 `loading`。请求会等待，随后
得到服务；加载失败会按 API 自身的错误格式返回 503。启动时的加载只试一次，失败即结束服务器；重新加载最多试
3 次。

## 并发请求：`--parallel N`

`--parallel N`（1..8，也可写配置文件 `server` 段中的 `"parallel"`）在 engine 的 batch 槽位中同时服务 N 个请求；
默认的 1 就是单流路径。更多请求按到达顺序等待（只有为恢复持有其对话的槽位时，后来的请求才可能先行，且至多
几次）。槽位在其请求结束、被取消或客户端离开时释放。`/metrics` 报告 `live.slots`（`total`、`active`、
`queue_depth`）。与 llama.cpp 相同，`--max-context` 是共享的：每个槽位持有 `--max-context / N` 个 token
（`--parallel 4 --max-context 262144` 得到四个 65536 的槽位；要四个 262144 就配 `--max-context 1048576`）。
单个请求独占运行时走的独占路径长度相同，所以能装进槽位的请求永远不会超长，装不下的会以 400 拒绝并注明该
长度。`/slots`、`/props` 与 `/v1/models` 报告每个槽位的上下文。`--slot-full-context`（也可写配置文件 `server`
段中的 `"slot-full-context"`）改为让每个槽位都拥有完整的 `--max-context`，代价是 N 倍的 KV cache；启动行会说
明这一点。各槽位的 KV cache 与同等长度的一个独占会话都取自专家缓存。每个槽位保留两个对话检查点——其系统
提示词的末尾与最近一轮（各自之前至少要有 `--prompt-cache-root` 个 token），新请求只读取它与任一槽位检查点
共享的最长开头之后的内容，无论槽位在 decode 还是空闲：共享同一段长系统提示词的请求只读取它一次。

所有 chat、messages 与 responses 请求都必须在 `model` 中指名所服务的模型：`/v1/models` 列出的别名（或模型名），
或模型文件名。未知或缺失的名字按 API 自身的错误格式返回 404，回复总是带上所服务的名字。

`--max-context` 高于 32768 时 KV cache 是弹性的（`--kv-resident`、`--vram-elastic` 或分层切分时不是；对等卡
没问题）：每个会话在启动时后备它的前 32768 个 cell（K/V 池、indexer 的 block key、RoPE 表与草稿层的 K/V 都
一样），这部分之外的 VRAM 放专家缓存末尾的那些槽位，直到某个请求需要它。一个请求在读取 prompt 之前后备它的
prompt 加 16384 个 cell（decode 越过这条线后每越过一次再加 16384），办法是收回那些槽位；它们的专家变成 CPU
未命中（长 prompt 的会移到对等卡）。下一个需要更少空间的请求把 cell 还回，缓存从 RAM 把槽位重新填满，于是短
请求 decode 时拥有 32k 上下文才会有的缓存。`--parallel` 下，份额超过 4096 token 的槽位在启动时后备 4096 个
cell，并按 4096 增长。空闲对话的 KV 一直保留，直到它旁边的请求运行了 15 s；此后当某个请求需要 VRAM、或
engine 空闲时，它移到 RAM（在 `--conversation-cache-mib` 之外、RAM 下限之上；否则只移到当前持有它的磁盘层）；
它的下一轮再读回来。启动行与每次搬动都会说 `elastic KV: ...`（移出时说 `slot N's idle conversation ...`）。
缓存绝不会低于一个 prompt chunk 所需的槽位数：若某个请求的 KV 即使在其他会话与空闲对话都尽力归还之后仍会把
缓存压到那里，该请求就以这个错误被拒绝；一条回复增长越过该线就以它结束（`--parallel` 槽位在此结束，如同到达
长度上限）；engine 继续运行。

其他槽位在 decode 时，一个至多 16384 token 的新 prompt 会被一次读完：各槽位为它让路（至多几秒），并按单流
路径的方式读取：同样的 chunk 与流式环、CPU 池承担其专家的一份。因此它的第一个 token 几乎与空闲 engine 上一
样快。更长的 prompt 则与它们并行读取：prompt 路径在自己的线程上运行，各槽位的 decode 窗口继续（它们的 kernel
拿更高的 stream 优先级），只有少数触及草稿层或输出的步骤在两个窗口之间运行，所以一个长 prompt 不会阻塞其他
请求。期间到来的、短得多的请求会在长 prompt 当前 chunk 结束时先行；长 prompt 的其余部分在它之后、与槽位并行、
按同样的 chunk 读取。它的借用额度与 KV 将要占用的缓存槽位，在真正被占用前只保住 decode 各槽位路由最少的那些
专家：槽位最近路由到的专家与其同层中路由更少的专家交换位置，因此读取 prompt 期间各槽位的专家仍留在 VRAM。
没有任何其他槽位在 decode 时到来的 prompt 按单流路径读取。

单卡上，batch 窗口中 CPU 未命中足够多的一层，会把其中一部分经 PCIe 送出：CUDA0 把它们从 RAM 副本拷进 VRAM
暂存并计算，与此同时 CPU 池计算其余的，两者同时读 RAM。启动行
`batch windows: a layer with N+ CPU misses sends a share of them over PCIe to CUDA0` 给出阈值。单独 decode 的
请求走单流路径，不受影响。

启动时服务器为每个设备打印一行 `[server] kv`，列出放在那里的全部会话状态，例如
`12.5GiB cuda2 solo=262144 slots=2x262144 draft=0.5GiB`。启用弹性 KV 时（`--max-context` 高于 32768 的服务器的
默认，或 `--parallel` 下槽位高于 4096 token 时）则写作 `0.6/4.4GiB cuda0 elastic`：前者是启动时后备的内存，后者
是缓存在 `--max-context` 下能长到的量。由于这部分内存取自专家缓存，当它让缓存装不下整套专家、或不足无槽位时
的一半时，engine 会发出警告（`warning: the N batch slots' sessions ... take X GiB of CUDA0 ...`），并给出补救
办法：更小的 `--max-context`、不开 `--slot-full-context`、更少的 `--parallel` 槽位。若某个槽位的会话完全放不下，
engine 会点名是哪个槽位（`warning: --batch N: slot K's session does not fit`），并带着放得下的那些槽位启动；若
放得下的不足两个，则 `--batch is off`，服务器一次只服务一个请求。

## 控制台日志

每一行服务器日志都以 `[server]` 开头，后接补齐到固定列的标签；取值为 `key=value`。速率单位 `t/s`，大小
`GiB`/`MiB`，时间 `s`。启动时打印一个区块，以图例开头：

```
[server] legend   pp=prompt read  tg=generation  c=cached  acc=drafts accepted  dl=tokens per window  hit=expert cache  pcie=routed experts over PCIe
[server] engine   ctx=262144 kv=int8 vram-reserve=500MiB pool=8 batch=4 (from config)
[server] sampling temp=1.0 top_p=0.95 top_k=20 min_p=0.0 pres=0.0
[server] vision   mode=dynamic gpu=0 (RTX 5090) load=per-image
[server] auto     peer-excl=off resident=64.0GB pool-affinity=auto
[server] ready    http://127.0.0.1:8080/v1  openai+anthropic  stats http://127.0.0.1:8080/
[server] loaded   68.4GiB 37.5s  cuda0=26.6GiB ram=41.2GiB cuda1=0.6GiB cuda1-copy=14.2GiB
[server] kv       2.6/16.3GiB cuda0 elastic slots=4
[server] slots    par=4 ctx/slot=262144 (slot-full-context), others queue
[server] weights  loaded, ctx=262144 images=on
```

`sampling` 会给命令行上设定的值标 `(cli)`、给配置文件中的标 `(config)`；`auto` 是 engine 自己选定的设置（见
[配置](configuration/index.md)）。之后出现的警告或错误写作 `[server] warning: text` 或
`[server] error: text`。

读取 prompt 期间，槽位大约每 5 s 打印一次 `pp=<read>/<total>`，附读取速率与已用时间：

```
[server] s2 read  pp=24576/66836 9150t/s 3s
```

请求生成期间，每 2 s 一行 `live`，覆盖所有正在生成的槽位：`tg=<tokens>/<max> <rate>`。槽位从 1 开始编号，与
统计页面一致。`total` 是各槽位速率之和。接受率不实时显示；`done` 行带有 engine 的精确数字。

```
[server] live | s1 tg=454/2048 113.7t/s | s2 tg=436/2048 119.0t/s | total 233t/s
```

每个请求以一条带槽位标记的 `done` 行结束：

```
[server] s1 done  pp=96/89c 0.1s  tg=1720 119.6t/s acc=65% dl=2.9  ttft=0.1s  end=stop  hit=89.7%  pcie=39.3%
[server] s2 done  pp=66836/0c 7.3s 9204t/s  tg=512 281.0t/s acc=58% dl=2.7  ttft=7.3s  end=length
```

- `pp=<prompt tokens>/<cached>c <read time>`，新读取达到至少 512 个 token 时附带读取速率；该速率只计新读取的
  token。
- `tg=<tokens> <rate>`。`acc` 是该请求被接受草稿数除以提议草稿数，`dl` 是每窗口平均 token 数；请求没有起草时
  两者都省略。
- `ttft` 是到第一个 token 的时间；`end` 是结束原因（`stop`、`length`、`cancelled`、`disconnect`、`error`）。
- `hit` 是专家缓存命中率，`pcie` 是经 PCIe 读取的路由专家份额；engine 会上报这两项时才会出现。

## 统计页面

服务器自身的地址（上面 `[server] ready` 行里的 `stats`）提供一页，每秒读取一次 `/metrics`、`/slots` 与
`/api/logs`：

- GPUs：engine 使用的每张卡，带 VRAM 以及负载与功耗的 60 s 图（单卡时加温度；三卡及以上是一张表，配平均负载
  与总功耗的图）。System：CPU、RAM、磁盘读取、驱动与 CUDA。Experts：上一个请求的路由查找在哪里得到服务
  （VRAM、对等卡、CPU）。
- Speed：decode 与 prefill 速率及 60 s 图（`--parallel` 时每个槽位一条线）、最近一次 prompt 及其前缀缓存命中，
  以及服务器启动以来最快的一次完整请求（decode 至少 256 个输出 token，prefill 至少 4096 个新读取的 prompt
  token）。
- Slots（`--parallel` 时）：每个忙碌槽位的速率、输出量与其上限之比、上下文、到第一个 token 的时间、前缀命中
  与每窗口 token 数；Accept 是该槽位最近完成的请求。Requests：最新完成的请求。Queue：等待槽位或等待单流的
  请求。
- Logs：控制台日志行。

`/metrics` 在 `history` 里携带图表背后的历史（`tok_s`、`prefill_tok_s_mean`、`slot_tok_s` 每槽位一条序列、
`gpus` 每卡带 `util`、`power` 与 `temp`），并在 `best` 里给出最快的请求。

## 对话落盘：`--conversation-cache-disk-mib`

`--conversation-cache-disk-mib N`（engine flag，也可写配置文件 `engine` 段中的 `"conversation-cache-disk-mib"`）
在 RAM 缓存之下、至多 N MiB 的空间内把对话放到磁盘上。0（默认）为关闭。它写入模型缓存目录内的 `conversations`
文件夹（`~/.cache/basalt/<hash>/`，engine 日志所在处）；`--conversation-cache-dir DIR` 只是另指一个文件夹，
单独不起作用。在 Linux 上，engine 为该层创建的文件夹与写入其中的文件只归其用户（0700、0600）；在 Windows 上
它们继承父文件夹的访问控制列表。该层开启时服务器在启动时打印 `[disk] conversations=DIR disk-mib=N`。一个至少
8192 个 token 的对话，若离开了它的会话（另一个对话占了它的槽位）、被 RAM 缓存逐出、或在 engine 结束时（停止、
空闲卸载）仍被 engine 持有，就会被写入 DIR：一个文件存放它的 token 与运行状态，K/V cache（每 4096 个 cell 一个
文件）与检查点状态则用共享文件，每个都以它覆盖的 token 与其字节来命名，于是同一个对话再写入、或另一个开头
相同的对话，只写入新出现的块与状态。只有系统提示词末尾与最近 3 个轮次末尾的状态会写入（其余检查点留在 RAM）；
在某个未存储的检查点处截断时，从其之前最近的一个已存储状态继续。最后一个检查点（其最后一条 prompt 的末尾）
落后不超过 4096 个 token 的对话，按该检查点存储、不带自己的运行状态：下一轮从那里继续，原样发回的回复会被
再次读入。后续某条 prompt 若以一个已存储的对话开头（它的 token，或它的某个检查点），就把它按那个点截断读回、
从那里继续，重启之后也一样：要等的是磁盘读取 K/V cache。一个文件绑定于模型文件、存储对话格式以及会改变状态
的设置（KV 类型、rope、草稿层、control vectors）；它的每个部分都带哈希，校验不过的文件会被删除，绝不会被恢复。
一秒没有输入（没有任何槽位在 decode）之后，engine 会把每个比其存储副本多出 4096 个 token 的对话写盘，所以崩溃
或被杀最多丢失这么多。预算给该文件夹封顶，最久未用的对话先走（共享文件随最后一个使用它的对话一起删除），并且
每次写入总在卷上留出 4 GiB。对存放模型文件的那块盘写入可能拖慢它的下一次加载：把缓存目录放到另一块盘可以避免。

## 休眠：`--idle-sleep`

`--idle-sleep SECONDS`（服务器选项，也可写配置文件 `server` 段中的 `"idle-sleep"`）在没有请求那么久之后让模型
休眠。engine 先交还其专家缓存的 VRAM，包括对等卡的层级（每个专家都有 RAM 副本，它原先占据的槽位会被记住；
独占的对等层保持不动），并撤销 RAM 副本在驱动上的注册；随后驱动把它剩余的 GPU 状态（权重、KV cache、buffer）
移到主机内存并释放 VRAM。进程、它的专家 RAM 副本、它持有的对话以及自适应层学到的放置都保留。下一个请求把它
唤醒：驱动把 GPU 状态放回同样的地址，engine 重新注册 RAM 副本，缓存从已注册的部分开始回填，于是该请求跑起来
和休眠前一样。服务器先检查 GPU 状态能否装进空闲 RAM 并留出 8 GiB；否则模型保持唤醒
（`[server] warning: the model stays awake ...`）。仅 Linux：其他平台的驱动不提供这样的挂起。`--idle-unload`
仍然结束进程，那也会释放 RAM；两者同时配置时，给 `idle-unload` 更长的时间。

## 内存下限

Basalt 会拒绝让机器内存见底的选择，也从不缩减你要求的量：

- Windows 与 WSL2 上启动时即拒绝低于 256 的 `--vram-reserve-mib`：全部加载后若剩余 VRAM 更少，之后的分配可能
  让驱动把 GPU 内存换页，请求就会停滞。Linux 上驱动改为让这种分配失败，下限是 128。
- 当部分专家要从 SSD 读取（GPU 与 RAM 副本都装不下）时，RAM 副本会为操作系统的文件缓存留出一个下限：RAM
  余量（2 GiB 或 RAM 的 5%，取较大者），加上 4 GiB 与一个 prompt chunk 从文件读取量中的较大者（借出的 GPU
  槽位最多 3.5 GiB，再加上 SSD 上两层的专家）。启动行 `RAM floor ...` 给出具体数字。每个专家都有容身处时，
  只保留那份余量；为借出槽位与自适应换入换出保留的副本不计入其中：它们用的是剩下的 RAM。
- 服务器的 RAM 副本还要为启动后服务所需的量留出空间：对话检查点（每个都是模型完整的运行状态，每个槽位
  `--prompt-cache` 个再加一个）、`--conversation-cache-mib`、一张图的 buffer 以及请求自身的 buffer。启动行
  `RAM ... free, guard ...` 给出具体数字。
- 启用自适应层时，RAM 副本还为离开 GPU 的专家保留空位，这样换话题时能快速交换。启动行
  `free places for the adaptive swaps` 显示数量（服务器上是 `--adapt-swaps` 的四倍）；RAM 紧张时保留得更少，
  一轮换入换出的也更少。
- 显式给出的 `--resident-budget-gib` 若留下的空闲低于该下限，会在启动时被拒绝；错误信息会说明剩余 RAM 与
  安全的最大值（'max'）。
- 服务期间，RAM 守卫对照"余量加一个 prompt chunk 的文件读取量（至多 3.5 GiB；没有专家从 SSD 读取时只算
  余量）"来监视 MemAvailable。低于该线时，engine 已停放的对话离开 RAM（下一个请求要接续的那个除外）：磁盘层
  开着时它们进磁盘（engine 尚无副本的那些），否则被丢弃。engine 还会释放它为复用保留的检查点块，并把空闲堆
  归还给操作系统；新到的请求在启动前最多等 30 s 让 RAM 回来；只给当前对话追加至多 256 个 token 的请求（服务器
  在预算处关闭 thinking）不等。低于守卫线后不再停放任何新内容，磁盘层既不写仍在会话中的对话、也不读回（它
  自己的下限比守卫线高 1 GiB）。已在运行的请求继续；只有 RAM 危急时才以错误停止：空闲低于 1 GiB，或低于守卫
  线且内核内存压力（`/proc/pressure/memory`，avg10）显示任务大部分时间处于停滞（some >= 60% 或 full >= 40%）。
  没有压力读数时（Windows，或没有 PSI 的内核），RAM 持续低于守卫线 10 s 即为危急；更短暂的下探不算。错误
  （`RAM low (... free < ... guard) ...: request stopped` 或 `RAM critical (... free < ... hard floor): request
  stopped`，同样返回给客户端）只有在 RAM 是被 engine 自身的增长吃掉时才点名 `--resident-budget-gib`，否则说明
  其他程序占用了多少。模型休眠期间守卫不做判断（驱动把 GPU 状态放在 RAM 里直到唤醒）；唤醒时重新取一个读数。
  Windows 不把经由映射专家文件读入的页算作可用，所以进入守卫线 1 GiB 之内时，engine 先归还它们再重新读取。

## 视觉输入：`--vision`

图像输入是一个服务器选项，`--vision off | dynamic | cpu | <GPU number>`（配置：`server` 段中的 `"vision": 1`）。
默认为 `off`。编码器是 Basalt 自己的，各模式共用同一批 kernel，所以所有模式给出逐位相同的 embeddings。

一个请求可以怎样指明图像取决于它来自哪里：[图像来源](configuration/image-sources.md)。工具结果里的图像在三种
API 中都留在 prompt 中的原位置。系统提示词不能包含图像：这样的请求会被拒绝。

| 模式 | 编码器运行在哪里 | VRAM 代价 | 新图延迟（IQ3_S，400 W，较早的构建） |
| --- | --- | --- | --- |
| `off` | 无图像输入 | 无 | |
| 一个 GPU 编号 | 该卡上的 sidecar `bin/basalt-vision`（按 `gpu` 的编号），常驻加载 | 约占该卡专家缓存的 2.2 GiB | 1k 个图像 token 时 TTFT 0.77 s，3k 时 1.4 s，4k 时 1.8 s |
| `dynamic` | engine 内部，主 GPU 上 | 静止时无（0.85 GiB 的 pinned RAM） | 0.77 s、1.45 s、1.8 s |
| `cpu` | CPU 上的 sidecar | 无 | 1k 约 6 s，3k 约 40 s，4k 约 70 s |

`dynamic` 每张图借用专家缓存末尾的槽位（任何专家有 RAM 副本的槽位都行），从 pinned 副本上传权重（约 30 ms），
在旁路 stream 上编码，再从 RAM 把这些槽位填回（约 70 ms），因此编码之后缓存原样如初。其他 `--parallel` 槽位在
编码期间暂停。当长对话的弹性 KV 占着那块空间时，各会话先归还它们并未持有的 KV cell（空闲槽位优先），并随自身
增长再取回。只有在即使这样也无法腾出空间时才拒绝图像。任何模式下，见过的图像都由编码器缓存在几十毫秒内给出。
sidecar 的卡是指名而非从 `gpu` 推出的：任何卡都行，无论它是否运行 engine，启动行会点名它。

```json
{ "server": { "gpu": "0,1", "vision": 1 } }
```

指定 GPU 编号时使用 sidecar，主构建也会一并产出（`tools/vision` 也能单独构建）。服务器在 `bin/basalt` 旁边寻找
它，只要求指名的卡存在：sm_89 规则是 engine 的，不是 sidecar 的。sidecar 在配置时针对一张卡构建
（`-DCMAKE_CUDA_ARCHITECTURES=<cc>`）；在别的卡上它会在启动时以 CUDA 自身的错误失败，服务器会把错误显示出来。
非目标卡上的 sidecar 由社区测试。`dynamic` 不需要 sidecar。

图像编码器是 Basalt 自己的，为这个模型的 vision tower 而写（不用 llama.cpp 或 ggml）：GPU 上是 bf16 tensor-core
乘法与 fp32 累加的 flash attention，CPU 上是 AVX-512/AVX2。它可以即插即用地替换 llama.cpp 的 mtmd 编码器，在
所有图像尺寸上都更快，并且更贴近 fp32 参考：mtmd 的 fp16 attention 在大图上会漂移。单张图上限 64 兆像素；超过
上限的图像在解码之前就按其文件头被拒绝。

## `PROBE rates`（engine 命令）

在 `--serve` 下，请求之间在 stdin 输入一行 `PROBE rates`，会在已加载的 engine 内部用约半秒测量本机速率，并在
stdout 打印一个区块：`PROBE begin rates v1`、若干 `PROBE <section> key=value ...` 行、`PROBE end ms=<wall>`。
它不改变任何 engine 状态（层级、缓存、KV、RNG、CPU 池的活跃工作线程数），也不保存任何东西。各 section：

| section | 测量内容 |
|---|---|
| `gpu`、`host`、`pack` | 设备、链路代际与宽度、功率上限、RAM、CPU 池、专家格式 |
| `link` | 每 GPU 的 pinned 主机到设备与设备到主机 GB/s，六次 64 MiB 突发取最优 |
| `gpu_mem` | 每 GPU 的设备间拷贝 GB/s（读与写的字节都计） |
| `host_mem` | CPU 池从驻留 RAM 的专家读取的 GB/s，按活跃工作线程数 |
| `cpu_expert` | 经池调用冷专家，每次 1、2、3、8 个，覆盖每个测过的线程数与当前线程数 |
| `peer` | 对等往返按条目数（1 到 8）及其线性拟合：基础值与每条目微秒数 |
| `copy` | 向每个 GPU 做一次专家大小的拷贝，以及这种拷贝运行期间的对等往返 |

跑不了的步骤打印 `n/a` 并附原因（没有对等卡、没有驻留 RAM 的专家、非原生的专家格式）。batch 期间该命令回答
`ERR PROBE: ...`。

`PROBE set workers=K share=F`（任一键皆可）为随后的请求设定 CPU 池的活跃工作线程数与 prompt 路径的 CPU 份额，
并回答 `PROBE end workers=<count>`。它是给 `basalt-probe` 用的；设置一直有效，直到 engine 停止。

`PROBE adapt [every=N] [swaps=N] [decay=F] [profile=FILE]` 把专家层恢复到 `READY` 时的状态；带 `profile=` 时把
那份已保存 profile 的排名放进同样的槽位，只复制专家不同的槽位。它清零路由计数、轮次计数器与草稿策略，丢弃
会话缓存的 prompt，并为随后的请求设定自适应层的设置，于是下一个请求如同加载后的第一个请求那样运行。它回答
`PROBE end adapt every= swaps= decay= moved= peer_moved= kept= ms=`（每张卡复制的槽位数、保持原样的槽位数、
墙钟时间）。它要求自适应层开启且只有一张主卡（没有分层切分、至多一张对等卡、不能 `--peer-exclusive`）；换入
换出次数高于启动时的 `--adapt-swaps` 时，它各轮找到的 RAM 空位会更少。
