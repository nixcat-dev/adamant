# 配置参考

[English](../../configuration/index.md)

全部设置：服务器选项、引擎标志和采样字段。对速度与精度的影响见[运行](../running.md)和[精度](../precision/index.md)。

- [设置如何给出](#设置如何给出) · [自动设置](#自动设置) · [服务器](#服务器) · [模型与 GPU](#模型与-gpu)
- [上下文与 KV](#上下文与-kv-缓存) · [专家缓存](#专家缓存与内存) · [提示词](#提示词处理) · [drafting](#drafting)
- [槽位](#并行槽位) · [会话缓存](#会话缓存) · [视觉](#视觉) · [控制向量](#控制向量)
- 页面：[模型与路由](models.md)、[采样](sampling.md)、[自适应层](adaptive.md)、[示例](examples.md)、[诊断](diagnostics.md)

## 设置如何给出

```sh
bin/basalt serve --config basalt_config.json [options]
bin/basalt serve MODEL.basalt [--python PY] [options]
```

- **服务器选项**（Python 服务器）：模型、地址、端口、槽位、视觉、日志。命令行上写作 `--name value`。
- **引擎标志**（`bin/basalt --help`）：上下文、KV、专家缓存、prefill、speculation。像任何标志一样给出；服务器按顺序传递，并询问引擎接受哪些。

配置文件是一个 JSON 对象。`models` 列出模型；`engine` 对所有接受它的模型生效；`server` 是共享的：

```json
{ "version": 2, "models": [{ "model": "model.basalt", "kv": "int8" }],
  "engine": { "resident-experts": true }, "server": { "port": 8080 } }
```

- 设置写在配置文件的哪里：`server` 指 `server` 块，`entry` 指一个模型条目或顶层 `engine`，`engine`（引擎标志）同样这两处，
  `model entry` 只指模型条目。
- 键名是去掉短横线的标志名。开关写 `true`。可重复的选项（`allowed-host`、`cors-origin`、`image-url-host`、`image-dir`）取一个列表。
  `model` 路径相对于配置文件。
- 标注"仅 Flash-Next"的行写在其他模型的条目或命令行上会让启动停止；写在顶层 `engine` 里时对该模型略过，并给出一条启动行
  （见[启动时的检查](models.md#启动时的检查)）。
- 优先级：命令行、配置文件、默认值；`--model` 和 `--alias` 不与 `--config` 同用（条目自己命名模型）。块写错或名字未知会让启动停止。
  `api-key` 不能写进文件。
- 默认值是不需要了解机器的最快设置。会改变输出或内存的键（8,192 上下文、fp16 KV）留给你自己设置。

## 自动设置

未给出时，引擎自行选择 `peer-exclusive`、`resident`（`resident-budget-gib`）、`pool-affinity` 和 `pool-workers`。给出的值总是优先。
它会打印一行 `[server] auto`：

| 设置 | 规则 |
| --- | --- |
| `peer-exclusive` | 仅 Flash-Next。只在有对等卡、完整专家集放不进空闲 RAM 而去掉对等卡那份能放下时开启 |
| `resident` | 仅 Flash-Next。安全下限之上的空闲 RAM，上限是 GPU 不持有的专家 |
| `pool-affinity` | `auto` |
| `pool-workers` | 物理核心数减去主机线程；混合 CPU 上是 P 核减一加 E 核的一半 |

## 服务器

命令行 `--name`；"位置"列说明哪个配置块接受它。

| 设置 | 位置 | 类型、取值 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| `model` | 模型条目 | 路径 | 必填 | `.basalt` 文件：权重、分词器、chat 模板、drafter、vision projector 和默认采样。 |
| `alias` | 模型条目 | 字符串 | 文件中的名字 | API 报告的模型名。 |
| `host` | server | 地址 | `127.0.0.1` | 监听地址。`0.0.0.0` 对外服务；绝不要不带 `api-key` 这样用。 |
| `port` | server | 整数 | `8080` | TCP 端口。 |
| `api-key` | 命令行 | 字符串 | 无 | 要求 `/v1/*` 带上这个 key（`Authorization: Bearer` 或 `x-api-key`）。仅命令行。 |
| `allowed-host` | server | 字符串，可重复 | 无 | 也回应这个 `Host` 名；`"*"` 关闭该检查。 |
| `cors-origin` | server | 字符串，可重复 | 无 | 允许来自该来源的网页调用 `/v1/*`。 |
| `fit-max-tokens` | 条目 | 开关 | 关闭 | 把 `max_tokens` 压到剩余上下文内，而不是回 400。 |
| `open` | server | 开关 | 关闭 | 就绪后打开网页。 |
| `lazy` | server | 开关 | 关闭 | 以未加载状态启动；第一个请求时加载。会丢掉 sidecar 编码器。 |
| `idle-unload` | 条目 | 秒 | 关闭 | 空闲这么久后卸载。为其他程序让出 VRAM。 |
| `idle-sleep` | 条目 | 秒 | 关闭 | 空闲这么久后睡眠：VRAM 释放、RAM 保留，几秒内唤醒。仅 Linux。[运行](../running.md) |
| `min-free-vram-mib` | 条目 | 整数 | 关闭 | 只有空闲 VRAM 达到这个量才加载未加载的模型；否则 503。 |
| `grammar` | 条目 | 开关（`--no-grammar`） | 开启 | 在引擎内把工具调用和 JSON 格式（`json_object`、`json_schema`）约束到 schema。关闭时其他 JSON 格式以 400 拒绝。 |
| `max-payload-mib` | server | MiB | `256` | 服务器读取的最大请求体；更大的得到 413。 |
| `debug` | server | 开关 | 关闭 | 每个请求一行日志，并输出模型的原始文本。 |
| `request-lines` | server | 开关 | 关闭 | 在 stdout 回显逐请求摘要。 |
| `temperature`、`top-p`、`top-k`、`min-p`、`presence-penalty`、`frequency-penalty`、`repetition-penalty`、`penalty-last-n`、`seed` | 条目 | 数字（`top-k`、`penalty-last-n`、`seed` 为整数） | 模型文件的 `sampling`，否则引擎的 | 请求未给出该字段时的默认值。见[采样](sampling.md)。 |
| `reasoning-budget-tokens` | 条目 | 整数 | `0` | 请求未指定时用于思考的预算。请求自己的值优先。 |
| `thinking-loop-tokens` | 条目 | 整数 | `256` | 同一个 token 连续重复这么多次时结束思考；`0` 关闭。 |
| `answer-room` | 条目 | 开关（`--no-answer-room`） | 开启 | 为收束思考和给出答案预留空间（上限 1,024 token 或更多）。 |
| `config` | 命令行 | 路径 | 无 | 配置文件。仅命令行。 |

## 模型与 GPU

| 设置 | 位置 | 类型、取值 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| `gpu` | 条目、`basalt`、`basalt-bench`、`basalt-probe` | `"0"`、`"0,1"`、`"auto"` | 仅 GPU 0 | 按 `nvidia-smi` 编号的卡。第一张是主卡。`auto` 取每一块 Ada（sm_89）卡，并把最快的一张作为主卡。[GPU](gpus.md) |
| `peer-reserve-mib` | engine | MiB | 600 | 第二张卡在其专家和提示词缓冲之外保留的空闲 VRAM。启动信息提示对等卡缓冲放不下时调高它。[GPU](gpus.md#两张-16-gb-卡) |

## 上下文与 KV 缓存

| 设置 | 位置 | 类型、取值 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| `max-context` | engine | token | 8192 | 最长上下文。由 `parallel` 个槽位共享。KV 缓存随它增长。 |
| `kv` | engine | `fp16`、`int8`、`q4_0`、`k8v4` | `fp16` | 仅 Flash-Next。KV 存储。`int8` 让 VRAM 减半。`q4_0` 是旋转后的 4-bit；`k8v4` 让 K 用 int8、V 用 4-bit（每个 cell 816 字节，`int8` 是 1,056）。不与 `kv-resident` 同用。 |
| `kv-resident` | engine | cell（至少 20,480）或 `0` | 0 | 仅 Flash-Next。在 VRAM 中保留 N 个 cell 的 KV，其余放在 pinned RAM。 |
| `rope-scaling` | engine | `none`、`linear`、`yarn` | 模型的，否则 `none` | 仅 Flash-Next。超出训练长度扩展上下文。运行期间固定。 |
| `rope-scale` | engine | 数字 | 模型的，否则 1 | 仅 Flash-Next。`linear` 与 `yarn` 的倍数。 |
| `rope-freq-base`、`rope-freq-scale` | engine | 数字 | 0（用模型自己的） | 仅 Flash-Next。原始 RoPE 值。除非了解含义，保持 0。 |
| `yarn-orig-ctx`、`yarn-ext-factor`、`yarn-attn-factor`、`yarn-beta-fast`、`yarn-beta-slow` | engine | 数字 | 0（262144）、auto、1、32、1 | 仅 Flash-Next。与 llama.cpp 相同的 YaRN 参数。 |

## 专家缓存与内存

模型的专家存放在 VRAM（专家缓存）、RAM 或文件中。这些设置决定放在哪里。

| 设置 | 位置 | 类型、取值 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| `vram-reserve-mib` | engine | MiB，至少 256（Linux 上 128） | 700（卡上槽位太少时降到 300） | 主卡上为图、scratch 和 head 保留的空闲 VRAM。会提高到 kernel 需要的下限（Linux 上 `--parallel 4` 时……）。 |
| `resident-experts` | engine | 开关 | 有 profile 时开启 | 仅 Flash-Next。把 GPU 不持有的专家放进页锁定 RAM。RAM 不够时退回文件映射。 |
| `resident-budget-gib` | engine | GiB | auto | 最多 N GiB 的专家放进 RAM，最热的优先。 |
| `mmap-experts` | engine | 开关 | 关闭 | 仅 Flash-Next。用文件映射代替 RAM 副本。 |
| `shared-expert-arena` | engine | 文件路径 | 关闭 | 仅 Flash-Next。让多个进程共用一份常驻 arena 文件（放在 `/dev/shm`）。 |
| `expert-profile` | engine | 文件 | `data/profiles/<arch>-<layers>x<experts>.prof` | 哪些专家起步就驻留 VRAM。 |
| `expert-profile-save` | engine | 文件 | 关闭 | 仅 Flash-Next。退出时和定期保存学到的放置；启动时据此放置。需要自适应层。 |
| `expert-profile-save-every` | engine | 分钟 | 10（`0` 表示只在退出时） | 保存周期。 |
| `resident-cpu-experts` | engine | 开关 | 关闭（`resident-experts` 隐含开启） | 仅 Flash-Next。配合 `mmap-experts`：把常驻 RAM 的专家放在普通 RAM 中。 |
| `vram-elastic` | engine | 开关 | 关闭 | 仅 Flash-Next。把专家缓存分成段，让 `POST /v1/vram` 能在请求之间出借 VRAM。 |
| `vram-segment-mib` | engine | MiB，至少 64 | 512 | 仅 Flash-Next。`vram-elastic` 的段大小。 |
| `expert-cache` | engine | `auto` 或数量 | `auto` | 保留在 VRAM 中的专家块数。`auto` 取权重、会话和保留量之后能装下的部分。 |
| `ram-reserve-mib` | engine | MiB | 2048 | 留给系统其余部分的空闲 RAM。专家层取引擎自身主机缓冲之上的部分，服务的 RAM 守卫守住它。`0` 不保留任何东西并关闭守卫。[内存下限](../running.md#内存下限) |
| `adapt-every` | engine | 窗口数 | 服务器上 1，层拆分时 4（`0` 关闭） | 仅 Flash-Next。每 N 个窗口做一次自适应轮。[自适应层](adaptive.md) |
| `adapt-swaps` | engine | 整数 | 96（`0` 关闭） | 仅 Flash-Next。一轮最多换入主 GPU 的专家数。[自适应层](adaptive.md) |
| `peer-adapt-swaps` | engine | 整数 | `adapt-swaps` | 仅 Flash-Next。一轮最多换入对等卡的专家数。 |
| `adapt-decay` | engine | 数字 | 服务器上 0.85，层拆分时 0.7 | 仅 Flash-Next。每轮之后对路由计数施加的系数。[自适应层](adaptive.md) |

## 提示词处理

提示词按批分块读取。这些设置决定块大小，以及流式专家如何共享。

| 设置 | 位置 | 类型、取值 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| `prefill` | engine | `auto`、`auto:N`（N = 8192、16384 或 32768），或 token 数 | `auto` | 仅 Flash-Next。提示词块大小。 |
| `prefill-cpu-share` | engine | 0 到 1 | 0.6 | 仅 Flash-Next。流式提示词专家中由 CPU 计算的比例。固定值，所以输出逐次运行一致。 |
| `short-read` | engine | token | 64（`0` 关闭） | 仅 Flash-Next。至多 N 个 token 的新增部分走 decode 窗口读取。 |
| `peer-prefill-rows` | engine | 每层行数 | 块 × top-k 的一半 | 仅 Flash-Next。对等卡计算的每层提示词行数。`0` 让提示词留在主卡。 |
| `read-beside` | engine | 开关 | 关闭 | 配合 `batch`：短到可以独占 GPU 的提示词读取（16,384 token 以下）与解码槽位同时进行，而不是挡住它们，因此新提示词读取期间槽位仍持续出 token。没有槽位在解码时走独占路径，所以单流的行为与以前完全一致。与槽位并行时，该读取只出借有 RAM 副本的缓存槽位（有 RAM 副本时没有别的上限），并且因为与窗口共享 GPU 仍比独占读取慢；这样一次读取中的会话 checkpoint 仍会暂停窗口。 |
| `read-decode-share` | engine | 数字 0..1 | 0.5 | 配合 `batch`：与槽位窗口交错进行的提示词块读取中，窗口在下一块之前获得该块时间的比例。`0` 表示把这样的读取整块跑完（窗口随后等待它）。 |
| `read-live-chunk` | engine | 整数 token，256 的倍数 | 1024 | 配合 `batch`：槽位解码时提示词读取每次取的块，因此长提示词每次只挡住窗口一小块。受本次运行自身的块大小限制。 |

## Drafting

除非打开 `relaxed-drafting`，输出与纯解码完全一致。

| 设置 | 位置 | 类型、取值 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| `spec` | engine | 整数 | 8（MTP），否则 0 | 至多 N 个 token 的校验窗口。`0` 或 `1` 关闭。 |
| `suffix-draft` | engine | token | 3（`0` 表示只做 MTP） | 仅 Flash-Next。提示词查重：用至少 N 个 token 的更早重复填充窗口。 |
| `mtp-max-t` | engine | 整数 | 0（= `spec`） | 仅 Flash-Next。限制 MTP 自身的窗口。 |
| `mtp-window` | engine | cell | 32768（`0` 表示全部） | 仅 Flash-Next。drafter 只看最后 N 个 cell。 |
| `relaxed-drafting` | engine（`[P]`） | `true` 或 0 到 1 | 关闭 | 仅 Flash-Next。**改变输出。** 采样时保留模型给出概率不低于 P 的 draft。贪婪、`logit_bias`、JSON 和工具调用保持精确。[精度](../precision/qwen38-flash-next.md#宽松起草) |
| `route-tail-skip` | engine（`[R]`） | `true` 或 1 到 9 | 关闭 | 仅 Flash-Next。**改变输出。** 若某专家在每个窗口条目里排在 R 或更后且没有 GPU 持有它，就跳过它。需要设备端 MoE tail。[精度](../precision/qwen38-flash-next.md#有损路由标志) |
| `route-prior` | engine（`[LAMBDA]`） | `true` 或 0 到 16 | 关闭 | 仅 Flash-Next。**改变输出。** 给已持有的专家的路由 logit 加上 LAMBDA。需要设备端 MoE tail。[精度](../precision/qwen38-flash-next.md#有损路由标志) |

## 并行槽位

| 设置 | 位置 | 类型、取值 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| `parallel` | 条目 | 1..8 | 1 | 同时服务 N 个请求。设置引擎的 `batch`。 |
| `slot-full-context` | 条目 | 开关 | 关闭 | 给每个槽位完整的 `max-context`，代价是 N 倍的 KV。 |

## 会话缓存

| 设置 | 位置 | 类型、取值 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| `prompt-cache` | engine | 整数 | 6 | 保留的会话 checkpoint 数，每个约占 118 MB RAM。`0` 表示每次都从头读提示词。 |
| `prompt-cache-every` | engine | token | 16384（`0` 关闭） | 仅 Flash-Next。每新读 N 个 token 也做一次 checkpoint（只占 RAM）。 |
| `prompt-cache-root` | engine | token | 2048（`0` 表示从不） | 系统提示词达到这个长度时在它末尾做 checkpoint。 |
| `turn-token` | engine | token id | 248045 | 开启一轮对话的 token；checkpoint 落在最后一个它的位置。 |
| `tail-role-token` | engine | token id | -1（关闭） | 以此 token 开头的尾随短轮会把 checkpoint 留在它之前。 |
| `conversation-cache-mib` | engine | MiB | 0（关闭） | 停放会话的 RAM 预算。 |
| `conversation-cache-slots` | engine | 整数 | 4 | 最多停放的会话数。 |
| `conversation-cache-min-free-mib` | engine | MiB | 2560 | 停放时保留的物理 RAM。 |
| `conversation-cache-min-tokens` | engine | token | 0 | 只停放 token 数不少于这个值的会话。 |
| `conversation-cache-disk-mib` | engine | MiB | 0（关闭） | 停放会话的磁盘预算。[运行](../running.md) |
| `conversation-cache-dir` | engine | 目录 | 模型缓存目录 | 磁盘层使用另一个目录。 |

## 视觉

| 设置 | 位置 | 类型、取值 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| `vision` | 条目 | `off`、`dynamic`、`cpu`、GPU 编号 | `off` | 图像输入。给 GPU 编号会在那张卡上跑 sidecar（占用其专家缓存约 2.2 GiB）。`cpu` 很慢。需要带 vision projector 的模型。每张图 32 MiB、64 百万像素。[运行](../running.md) |
| `image-url-host` | server | `HOST` 或 `HOST:PORT`，可重复 | 无 | 允许其他机器使用该主机上的 `http(s)` 图像 URL。[图像来源](image-sources.md) |
| `image-dir` | server | 目录，可重复 | 无 | 允许其他机器使用该目录下的图像文件。[图像来源](image-sources.md) |
| `image-remote-only` | server | 开关 | 关闭 | 把本机请求也当作远程请求。[图像来源](image-sources.md) |

诊断标志：[诊断](diagnostics.md)。

## 控制向量

加到残差流上的 GGUF 控制向量。请求用 `experimental_speed_projection` 切换它。

| 设置 | 位置 | 类型、取值 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| `control-vector` | engine | 文件 | 关闭 | 仅 Flash-Next。缩放为 1 的向量。 |
| `control-vector-scaled` | engine | `FILE:SCALE[,...]` | 关闭 | 仅 Flash-Next。带各自缩放的向量。 |
| `control-vector-layer-range` | engine | `A B` | 1 到最后一层 | 仅 Flash-Next。生效的层，含两端。 |
| `cvec-mode` | engine | `add`、`project` | `add` | 仅 Flash-Next。`add`：h += s v。`project`：h -= s (h.v) v。 |
| `cvec-dir` | engine | `per-layer`、`single:L` | `per-layer` | 仅 Flash-Next。每层用自己的方向，或处处用第 L 层的方向。 |
