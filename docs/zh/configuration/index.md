# 配置参考

[English](../../configuration/index.md)

用户可以传给 Basalt 的每一项设置：服务器选项、引擎标志、逐请求的采样字段，以及模型文件的采样默认值。某项设置对速度、内存和输出精度的影响见[运行](../running.md)和[精度](../precision/index.md)。

- [设置的提供方式](#设置的提供方式)
- [服务器](#服务器)
- [模型与 GPU](#模型与-gpu)
- [上下文与 KV cache](#上下文与-kv-cache)
- [专家缓存与内存](#专家缓存与内存)
- [提示词处理](#提示词处理)
- [投机解码 (MTP)](#投机解码-mtp)
- [并行槽位](#并行槽位)
- [会话缓存](#会话缓存)
- [视觉](#视觉)
- [控制向量](#控制向量)
- [采样与思考](sampling.md)
- [自适应层](adaptive.md)
- [示例配置](examples.md)
- [诊断、测量与 `--exp` 标志](diagnostics.md)

## 设置的提供方式

```sh
bin/basalt serve --config basalt_config.json [options]
bin/basalt serve MODEL.basalt [--python PY] [options]
```

`basalt serve` 在引擎二进制之上运行 Python 服务器（`serve/`）。把 `MODEL.basalt` 作为第一个参数等同于 `--model MODEL.basalt`；`--python` 选择解释器（默认：源码树里的 `.venv/bin/python`，否则 `python3`；Windows 上是 `.venv/Scripts/python.exe`，否则 `python`）。其余参数都交给服务器。

设置分两类：

- **服务器选项**属于 Python 服务器：模型路径、地址、端口、槽位、视觉、日志。在命令行上是 `--name value` 形式。
- **引擎标志**属于引擎二进制（`bin/basalt --help`）：上下文、KV 格式、专家缓存、prefill、投机解码。服务器负责转传。把它们像其他标志一样放在命令行上（`--kv int8`）。服务器按原类型、按原顺序把它们传给引擎，不解释其值，所以接受多个值的标志也可以用（`--control-vector-layer-range 4 44`）。服务器会询问引擎二进制接受哪些标志，遇到不认识的标志名就在启动时停止。`vision` 开启时，服务器自己加上 `--serve`，并加上 `--vision`（`vision dynamic` 时还会带 `--vision-mmproj`）。

配置文件是一个带两个可选块的 JSON 对象：

```json
{
  "server": { "model": "model.basalt", "port": 8080 },
  "engine": { "kv": "int8", "resident-experts": true, "control-vector-layer-range": [4, 44] }
}
```

- 键是去掉前导短横线的标志名（`_` 与 `-` 通用）。值为字符串或数字；开关型设置用 `true`，`false` 表示不开启。可重复的选项（`allowed-host`、`cors-origin`、`image-url-host`、`image-dir`）接受列表。
- `model` 和 `tokenizer` 路径相对于配置文件。
- **优先级：** 命令行 > 配置文件 > 内置默认值。引擎先收到配置文件的标志，再收到命令行的标志，重复给出的标志取最后一次的值。
- `server` 里出现引擎标志、或 `engine` 里出现服务器选项，都会让服务器在启动时停止；不认识的名称同样如此。
- `api-key` 不能在配置文件中设置；设置了它的文件会被拒绝。
- 内置默认值选择无需了解机器情况的最快设置（带 drafter 时投机解码 8、expert cache auto、prefill auto、RAM 装得下则专家常驻、服务器上 adaptive 每窗口一轮），并保留会改变输出或内存用量的设置（4096 上下文、fp16 KV）。这些要由配置文件来设。启动行 `[server] engine ...` 会列出来自文件或命令行的内容。

下表中的"默认值"指该项设置处处缺席时生效的值。

## 自动设置

有四项设置在留空时由引擎在启动时选定：`peer-exclusive`、`resident`（`resident-budget-gib`）、`pool-affinity` 和 `pool-workers`。配置文件或命令行给出的值永远优先，且绝不会被改动。

| 设置 | 规则 |
| --- | --- |
| `peer-exclusive` | 仅在有对等卡且带 `--serve` 时、且完整专家集放不下空闲 RAM 而扣掉对等卡份额后放得下时才开启；否则关闭。 |
| `resident` | 安全下限之上的空闲 RAM，上限为 GPU 未容纳的那部分专家。 |
| `pool-affinity` | `auto`。 |
| `pool-workers` | 物理核心数减宿主线程；在 `auto` 下的混合架构 CPU 上，为 P 核数减一加 E 核数的一半，选了 `p-cores` 则为 P 核数减一；并适配容器的 CPU 配额。 |

引擎打印一行，只点名选定的设置（按上表顺序），服务器每次启动转发一次：

```
[server] auto     peer-excl=off resident=51.2GB pool-affinity=auto pool=15
```

全部设置都已给出时不打印任何行。

## 服务器

服务器选项。命令行用 `--name`，配置文件放 `server` 块里的 `"name"`。

| 设置 | 类型、取值 | 默认值 | 作用与何时修改 |
| --- | --- | --- | --- |
| `model` | 路径 | 无（必填） | `.basalt` 模型文件。权重、分词器、chat 模板、drafter、vision projector 和默认采样全部来自它。 |
| `alias` | 字符串 | 模型文件中存储的名称 | API 报告的模型名。当客户端期待特定名称时修改它。 |
| `host` | 地址 | `127.0.0.1` | 监听的地址。`127.0.0.1` 只服务本机；`0.0.0.0` 同时服务网络。没有 `api-key` 时绝不监听 `127.0.0.1` 以外的地址。 |
| `port` | 整数 | `8080` | TCP 端口。 |
| `api-key` | 字符串 | 无 | 要求 `/v1/*` 携带此 key（`Authorization: Bearer ...` 或 `x-api-key`）。仅限命令行。只要 `host` 不是 `127.0.0.1` 就应设置它。 |
| `allowed-host` | 字符串，可重复 | 无 | 也响应这个 `Host` 名称；`"*"` 关闭该检查。当客户端通过地址以外的名称访问服务器时需要。 |
| `cors-origin` | 字符串，可重复 | 无 | 允许来自该 origin 的网页调用 `/v1/*`。 |
| `fit-max-tokens` | 开关 | 关闭 | 把请求的 `max_tokens` 收敛到剩余上下文，而不是用 400 拒绝。 |
| `open` | 开关 | 关闭 | 模型就绪后在浏览器中打开本地网页。 |
| `lazy` | 开关 | 关闭 | 以未加载状态启动纯文本 API，在第一个请求时加载模型。会关闭视觉。 |
| `idle-unload` | 秒 | 关闭 | 无请求超过此时长后卸载模型；下一个请求再加载。为其他程序释放 VRAM。 |
| `idle-sleep` | 秒 | 关闭 | 无请求超过此时长后让模型休眠：释放其 VRAM，进程保留 RAM 副本、会话和学到的专家放置，下一个请求几秒内唤醒。仅 Linux（由驱动挂起引擎的 GPU 状态）；当该状态放不进空闲 RAM 时保持唤醒。[运行](../running.md) |
| `min-free-vram-mib` | 整数 | 关闭 | 只有当空闲 VRAM 达到此值时才加载未加载的模型，否则回答 503。与 `lazy` 和 `idle-unload` 配合使用。 |
| `grammar` | 开关（`--no-grammar` 或 `grammar: false` 关闭） | 开启 | 在引擎内让每个工具调用和 JSON 响应格式（`json_object`、`json_schema`）都受其 schema 约束，因而无法产生格式错误的调用或对象（类似 llama.cpp 的 Qwen3-Coder 解析器：一次调用的必填参数可以任意顺序出现、各出现一次，最多六个必填参数可以乱序、其余按声明顺序，之后是任意顺序且可重复的可选参数）；模型的 JSON 边生成边原样返回，事后不做校验。与 tools 一起发送的 JSON 格式（`tool_choice` auto）会把回答限定为工具调用或 JSON 值；强制调用则限定为该调用。带 `strict: true` 的请求即使本项关闭也仍启用语法；本项关闭时，其他任何 JSON `response_format`（或 Responses 的 `text.format`、Anthropic 的 `output_config.format`）都会以 HTTP 400 拒绝。需要带 `--grammar` 的引擎；引擎有该标志时服务器就会开启其语法支持。批处理槽位下同样有效：槽位第一个被语法拒绝的取值会在掩码下重新采样（单流规则），其窗口到此为止。 |
| `debug` | 开关 | 关闭 | 每个请求一条日志，并在每次回复后输出模型的原始文本。 |
| `request-lines` | 开关 | 关闭 | 在 stdout 回显引擎的逐请求摘要，供只能看到该输出的守护程序使用。 |
| `temperature`、`top-p`、`top-k`、`min-p`、`presence-penalty`、`frequency-penalty`、`repetition-penalty`、`penalty-last-n`、`seed` | 数字（`top-k`、`penalty-last-n`、`seed`：整数） | 模型文件的 `sampling` 值，否则为引擎默认 | 请求省略该字段时的采样默认值，优先于模型文件的设置。含义与允许取值同[模型文件的采样键](sampling.md#模型文件的采样默认值)。由服务器逐请求应用；引擎不会看到它们。 |
| `reasoning-budget-tokens` | 整数 | `0`（无） | 未指定预算的请求的思考预算（见[采样](sampling.md)）。达到预算时结束思考并开始作答。请求自带的字段优先，请求里给 `0` 表示关闭预算。正数预算还受 `answer-room` 限制。 |
| `thinking-loop-tokens` | 整数 | `256` | 当模型在思考中连续重复同一个 token 达到此次数时，结束思考并让模型作答；`0` 关闭。见[思考循环](sampling.md#思考循环)。 |
| `answer-room` | 开关（`--no-answer-room` 或 `answer-room: false` 关闭） | 开启 | 当有效输出上限不小于 1024 token 时，为结束思考和作答预留空间。作用于缺失的预算并收敛正数预算；请求显式给 `0` 表示不参与。 |
| `tokenizer` | 路径 | 源码树中的 `pack/full/tokenizer`（不存在则用字节分词器） | 分词器目录。仅 mock 引擎使用。 |
| `mock-engine`、`script` | 开关、字符串（可重复） | 关闭、一句简短问候 | 测试用：用脚本引擎替代模型运行，供客户端和测试使用（无需 `model`）；`script` 是它的回答，给出多次时各请求依次得到它们。 |
| `config` | 路径 | 无 | 配置文件。仅限命令行。 |

## 模型与 GPU

| 设置 | 位置 | 类型、取值 | 默认值 | 作用与何时修改 |
| --- | --- | --- | --- | --- |
| `gpu` | 服务器（`basalt`、`basalt-bench`、`basalt-probe` 也支持） | GPU 编号，如 `"0"` 或 `"0,1"`，或 `"auto"` | 仅 GPU 0（一条警告会点名其他可见卡） | 要使用的卡，编号方式与 `nvidia-smi` 一致：精确使用这些卡，列出的第一块是主卡。`auto` 收下所有 Ada 卡并选最快的一块作主卡。引擎给其余卡分派角色（第二块卡成为对等专家层）。见 [GPU](gpus.md)。 |
| `peer-reserve-mib` | 引擎 | 整数 MiB | 600 | 第二块卡在容纳其专家和自身确定的提示词共享缓冲之后留出的空闲 VRAM。当启动打印 `prefill peer: the peer's buffers do not fit` 时调高它。见 [GPU](gpus.md#两张-16-gb-显卡)。 |

## 上下文与 KV cache

| 设置 | 位置 | 类型、取值 | 默认值 | 作用与何时修改 |
| --- | --- | --- | --- | --- |
| `max-context` | 引擎 | 整数 token | 4096 | 引擎允许的最长上下文。长提示词要调高它；KV cache 随之增长。启用 `parallel` 时，这一额度由各槽位共享。 |
| `kv` | 引擎 | `fp16`、`int8`、`q4_0`、`k8v4` | `fp16` | KV 存储。`int8` 让 VRAM 减半。`q4_0` 是 Hadamard 旋转后的 4-bit，在 `int8` 基础上再减半，精度略低。`k8v4` 的 K 用 int8、V 用旋转 4-bit（每 cell 816 B，`int8` 为 1056）；不能与 `kv-resident` 同用。长上下文用 `int8`；VRAM 吃紧再往下调。见[精度](../precision/index.md)。 |
| `kv-resident` | 引擎 | 整数 cell，最小 20480，或 `0` | 0（KV 全部在 VRAM） | 每个注意力层的 KV 只在 VRAM 保留这么多 cell，完整 K/V 放在锁页 RAM；腾出的 VRAM 用于专家槽位。能放进 N 个 cell 的上下文不走流式。适合小卡上的超长上下文。 |
| `rope-scaling` | 引擎 | `none`、`linear`、`yarn` | 模型文件的值，否则 `none` | 把上下文扩展到训练长度以外。整个运行期间固定：缓存中的 K 是 RoPE 之后的存储。 |
| `rope-scale` | 引擎 | 数字 | 模型文件的值，否则 1 | `linear` 与 `yarn` 的扩展倍数。 |
| `rope-freq-base`、`rope-freq-scale` | 引擎 | 数字 | 0（用模型自带的） | 原始频率基数与角度收缩。除非明确知道要什么值，否则保持 0。 |
| `yarn-orig-ctx`、`yarn-ext-factor`、`yarn-attn-factor`、`yarn-beta-fast`、`yarn-beta-slow` | 引擎 | 数字 | 0 (262144)、auto、1、32、1 | YaRN 自身参数，同 llama.cpp。 |

`max-context` 高于 32768 的服务器带有弹性 KV cache：每个会话在启动时支持其前 32768 个 cell，只有请求需要时才从专家缓存借用更多。启用 `parallel` 时，份额高于 4096 token 的每个槽位从 4096 个 cell 起做同样的事、按 4096 增长，并且当旁边的请求已运行 15 s 后，空闲会话的 KV 移入 RAM。`kv-resident`、`vram-elastic` 或层分割时该机制关闭。见[运行](../running.md)。

## 专家缓存与内存

模型的专家存在于 VRAM（专家缓存）、RAM 或文件中。以下设置决定其位置。

| 设置 | 位置 | 类型、取值 | 默认值 | 作用与何时修改 |
| --- | --- | --- | --- | --- |
| `vram-reserve-mib` | 引擎 | 整数 MiB，至少 256（Linux 上 128） | 700（在否则会分到过少槽位的卡上自动下调，最低到 300；你给出的值会被保留，除非按此处所述上调） | 当 `expert-cache auto` 决定缓存大小时，主卡上留给图、暂存区和输出头的空闲 VRAM。`expert-cache auto` 定完缓存后加载的服务器 kernel 和窗口图（96 MiB，`--parallel` 时每槽位再加 24 MiB）要装入其中；预留小到装不下它们又要保持下限时会被上调：Linux 上 `--parallel 4` 时，低于 320 MiB（128 + 96 + 4 x 24）的任何值都变成 320，`expert cache auto` 启动行会同时给出两者（`320 MiB reserved (150 asked, ...)`）。低于下限会被拒绝。如果机器开始对 GPU 内存换页，或其他程序需要 VRAM，就调高它。 |
| `resident-experts` | 引擎 | 开关 | 引擎带有该模型的专家 profile 时开启 | 把 GPU 未容纳的专家保存在页锁定 RAM 中，从而没有任何专家会被再次从文件读取。为操作系统留出 RAM 下限；RAM 装不下时退化为普通文件映射。`mmap-experts` 会关闭它。 |
| `resident-budget-gib` | 引擎 | 数字 GiB | 未设置时 auto | 类似 `resident-experts`，但 RAM 中最多放 N GiB 的专家，热的优先。若会留下低于 RAM 下限的空间则拒绝；错误信息会给出最大的安全值。 |
| `mmap-experts` | 引擎 | 开关 | 关闭 | 放弃默认的常驻 RAM 副本，退回文件映射。速度因此取决于 OS 页缓存。 |
| `shared-expert-arena` | 引擎 | 文件路径 | 关闭 | 用一个共享文件支撑常驻 arena，让多个进程共用一份副本。放在 `/dev/shm` 上。 |
| `expert-profile` | 引擎 | 文件路径 | 引擎自带的（其构建来源源码树中的 `data/profiles/<model>.prof`） | 启动时放置所用的专家 profile，格式同 `data/profiles`：哪些专家优先进 VRAM。设置它可以让某个二进制的放置不再依赖其构建目录，或用于尝试另一个 profile。文件缺失或为空会阻止启动。 |
| `expert-profile-save` | 引擎 | 文件路径 | 关闭 | 在退出和定期把自适应层学到的内容（VRAM 中的专家与路由计数）写入该文件。启动时发现该文件就按它放置，而不是按引擎的 profile，于是第一个请求就跑在上一次运行学到的放置上：工作负载延续时更快，工作负载变化时更慢（[自适应层](adaptive.md)）。需要开启自适应层。 |
| `expert-profile-save-every` | 引擎 | 分钟 | 10（`0` = 仅退出时） | profile 的保存频率。 |
| `resident-cpu-experts` | 引擎 | 开关 | 关闭（`resident-experts` 隐含开启） | 配合 `mmap-experts` 和专家 profile：把 GPU 未容纳的专家保存在普通 RAM 中，从而没有任何专家会被再次从文件读取。`resident-experts` 就是它，外加在可能时把副本页锁定、放不下时以文件映射回退。 |
| `vram-elastic` | 引擎 | 开关 | 关闭 | 把专家缓存保持在分段中，使 `POST /v1/vram` 能在请求之间把 VRAM 借给其他程序。与 `resident-experts` 配合可用：收缩会先把交出的专家复制回 RAM，因此没有任何专家要从文件读取；若下限之上的 RAM 装不下它们则拒绝（不做任何改动）；扩张会释放这些副本。 |
| `vram-segment-mib` | 引擎 | 整数 MiB，至少 64 | 512 | `vram-elastic` 专家缓存的分段大小：VRAM 借出与收回的步长。低于 64 时带警告关闭 `vram-elastic`。 |
| `expert-cache` | 引擎 | `auto`，或专家 blob 数量的整数 | `auto`（有专家 profile 时） | 在 VRAM 中保留这么多专家 blob，并在 GPU 上计算它们的行。`auto` 取权重、会话和 `vram-reserve-mib` 之后剩余 VRAM 装得下的最大数量。 |
| `adapt-every` | 引擎 | 整数窗口 | 服务器 1，层分割时 4（`0` = 关闭） | 每 N 个 decode 窗口一轮自适应。[自适应层](adaptive.md)。 |
| `adapt-swaps` | 引擎 | 整数 | 96（`0` = 关闭） | 一轮自适应最多换入主卡的专家数；大积压之后的服务器一轮会换入四倍数量，且 RAM 副本为其保留四倍 N 个空位。[自适应层](adaptive.md)。 |
| `peer-adapt-swaps` | 引擎 | 整数 | `adapt-swaps` 的值 | 一轮最多换入对等卡的专家数。仅在有对等卡时有意义。 |
| `adapt-decay` | 引擎 | 数字 | 服务器 0.85，层分割时 0.7 | 每轮之后路由计数所乘的系数；越低遗忘越快。[自适应层](adaptive.md)。 |

### 自适应层

引擎把会话路由最多的专家移入 VRAM，轮次在 decode 之旁进行。上面四项设置的作用、取舍和实测数字：[自适应层](adaptive.md)。

启动行 `RAM floor ...` 和 `RAM ... free, guard ...` 给出内存下限背后的数字。见[运行](../running.md)。

## 提示词处理

提示词按批式分块读取。以下设置决定分块大小以及流式专家的共享方式。

| 设置 | 位置 | 类型、取值 | 默认值 | 作用与何时修改 |
| --- | --- | --- | --- | --- |
| `prefill` | 引擎 | `auto`、`auto:N`（N = 8192、16384 或 32768），或以 token 计的分块大小 | `auto` | 按每次这么多 token 读取提示词。`auto` 取专家缓存能借出的最大分块，不超过上限；`auto:N` 设置该上限。 |
| `prefill-cpu-share` | 引擎 | 0 到 1 的数字 | 0.6 | 流式提示词专家中由 CPU 计算的比例；`0` 表示全部由 GPU 流式处理。固定值，所以每次运行输出相同。较弱的 CPU 可能适合更低的值。 |
| `short-read` | 引擎 | 整数 token | 64（`0` = 关闭） | 一段不超过 N token 的新提示词（一条聊天消息、一条短工具结果）走 decode 窗口读取，而不是批式提示词路径。 |
| `peer-prefill-rows` | 引擎 | 整数，每层行数 | 分块 x top-k 的一半 | 对等卡每层计算的提示词行数。`0` 让提示词路径留在主卡。 |

## 投机解码 (MTP)

drafter 提出 token，模型验证它们。除非开启 `relaxed-drafting`，输出与朴素 decode 完全一致。

| 设置 | 位置 | 类型、取值 | 默认值 | 作用与何时修改 |
| --- | --- | --- | --- | --- |
| `spec` | 引擎 | 整数 | 有 drafter 时 8，否则 0 | 验证至多 N 个 token 的窗口：最后一个被接受的 token 加至多 N-1 个草稿。`0` 或 `1` 关闭投机解码。 |
| `suffix-draft` | 引擎 | 整数 token | 3（`0` = 仅 MTP） | 提示词查找：当文本重复上下文此前至少 N 个 token 的片段时，窗口可以用其后紧跟的内容填充，draft 策略认为这样才划算；这样的窗口可以比 `spec` 多至多 2 个 token，上限 8。 |
| `mtp-max-t` | 引擎 | 整数 | 0（= `spec`） | 限制 MTP 自身的窗口；更长的窗口此后只来自提示词查找。 |
| `mtp-window` | 引擎 | 整数 cell | 32768（`0` = 全部） | drafter 只注意最后 N 个 cell。启用 `parallel` 时，每个槽位的草稿 K/V 只保留这些 cell 和它们之前的 16384 个（回退过的一轮），而不是槽位的整个上下文，省出的 VRAM 用于专家槽位。 |
| `relaxed-drafting` | 引擎（`[P]`） | `true`，或 (0, 1] 内的数字 | 关闭 | 可选，且会改变输出。采样时，模型给出概率不低于 P 的草稿会被保留，即便模型自己的抽样本会是另一个 token。更快，但不再是模型精确的采样分布；P 越低越快也越偏离精确。单独的 `true` 或不带值时为 0.5。贪心请求、`logit_bias`、JSON `response_format`、控制 token 和工具调用保持精确。仅限单流。数字：[精度](../precision/features.md)。 |
| `route-tail-skip` | 引擎（`[R]`） | `true`，或 1 到 9 的整数 | 关闭 | 可选，且**改变输出：精度低于模型。**在 decode 窗口中，某个专家任何 GPU 都没有时，若指名它的每一项在其 token 内按路由权重排序都位于 R 或更低（10 中从 0 开始计），则既不复制也不计算它；它贡献零，其他专家保持各自的权重。R 越低丢弃越多。单独的 `true` 或不带值时为 9。需要 device-planned MoE tail，否则保持关闭并在启动时说明。会列入启动时的有损标志警告。数字：[精度](../precision/routing.md)。 |
| `route-prior` | 引擎（`[LAMBDA]`） | `true`，或 (0, 16] 内的数字 | 关闭 | 可选，且**改变输出：精度低于模型。**decode 窗口选择十个专家时，就好像每个 GPU 已容纳的专家的路由 logit 都加了 LAMBDA；权重仍是路由自己的。值越高越倾向于路由到已容纳的专家。单独的 `true` 或不带值时为 0.5。需要 device-planned MoE tail，否则保持关闭并在启动时说明。会列入启动时的有损标志警告。数字：[精度](../precision/routing.md)。 |

## 并行槽位

| 设置 | 位置 | 类型、取值 | 默认值 | 作用与何时修改 |
| --- | --- | --- | --- | --- |
| `parallel` | 服务器 | 整数 1..8 | 1 | 以批处理槽位同时服务 N 个请求；更多请求排队等待。`max-context` 由槽位共享：每个槽位容纳 `max-context / N` 个 token。各槽位的 KV cache 从专家缓存中划出。设置引擎的 `batch`。 |
| `slot-full-context` | 服务器 | 开关 | 关闭 | 启用 `parallel` 时，给每个槽位完整的 `max-context`，而不是均分份额。代价是 N 倍的 KV cache。 |
| `batch` / `slots` | 引擎 | 整数 2..8 | 关闭 | `parallel` 所设置的引擎标志。请用 `parallel`；两者都设置是错误。 |

## 会话缓存

| 设置 | 位置 | 类型、取值 | 默认值 | 作用与何时修改 |
| --- | --- | --- | --- | --- |
| `prompt-cache` | 引擎 | 整数 | 6 | 请求之间保留的会话检查点数量，每个约 118 MB RAM。`0` 表示每个提示词从头读取。更多的检查点让更多聊天可以免于重读而恢复。 |
| `prompt-cache-every` | 引擎 | 整数 token | 16384（`0` = 关闭） | 此外每 N 个新读取的提示词 token 打一次检查点（仅 RAM：磁盘层不存储它们）。 |
| `prompt-cache-root` | 引擎 | 整数 token | 2048（`0` = 从不） | 当提示词首个轮次边界（系统提示词的末尾）至少这么长时为其打检查点。 |
| `turn-token` | 引擎 | token id | 248045（`<\|im_start\|>`） | 开启一个聊天轮的 token；提示词中最后一个该 token 的位置就是会话检查点存放之处。 |
| `tail-role-token` | 引擎 | token id | -1（关闭） | 当倒数第二个轮次以该 role token 开头（一条短的尾部轮次，例如 reasoning-effort 备注）时，检查点放在那个轮次之前，从而保存的是会话本身而非备注。 |
| `conversation-cache-mib` | 引擎 | 整数 MiB | 0（关闭） | 停放会话的 RAM 预算，让各次独立聊天免于重读即可回来。 |
| `conversation-cache-slots` | 引擎 | 整数 | 4 | 最多停放这么多会话。 |
| `conversation-cache-min-free-mib` | 引擎 | 整数 MiB | 2560 | 停放会话时保持空闲的物理 RAM 下限。 |
| `conversation-cache-min-tokens` | 引擎 | 整数 token | 0（每一个都停放） | 只停放至少这么多 token 的会话，适用于 RAM 层、磁盘，以及空闲会话离开 VRAM 时。低于最小值时，会话既不存储也不驱逐：agent 流量的一次性辅助请求不再从有复用收益的长会话手中抢走槽位。 |
| `conversation-cache-disk-mib` | 引擎 | 整数 MiB | 0（关闭） | 把会话停放在磁盘上，至多这么多（最久未用的先被替换），当某个提示词以它们开头时读回，重启后同样如此。大于 0 即开启；文件夹是模型缓存目录下的 `conversations`。[运行](../running.md) |
| `conversation-cache-dir` | 引擎 | 目录 | 模型缓存目录 | 磁盘层的另一个文件夹。预算为 0 时无作用。 |

## 视觉

| 设置 | 位置 | 类型、取值 | 默认值 | 作用与何时修改 |
| --- | --- | --- | --- | --- |
| `vision` | 服务器 | `off`、`dynamic`、`cpu`、一个 GPU 编号 | `off` | 图像输入；布尔值会被拒绝。GPU 编号按 `gpu` 的编号方式：编码器 sidecar 常驻该卡（约占其专家缓存 2.2 GiB），与引擎用哪些卡无关，也不需该卡承担引擎角色。`dynamic`：在引擎内部，仅在读取图像期间借用专家缓存槽位。`cpu`：sidecar 跑在 CPU 上，不占 VRAM，慢。延迟：[运行](../running.md)。GPU 编号需要 `basalt-vision` 二进制与 `bin/basalt` 放在一起。需要一个带 vision projector 的模型文件。一张图在解码前限制为最大 32 MiB、64 megapixels。 |
| `image-url-host` | 服务器 | `HOST` 或 `HOST:PORT`，可重复 | 无 | 允许来自其他机器的请求引用恰好这个 host 上的 `http(s)` 图像 URL。来自本机的请求不需要该标志。[图像来源](image-sources.md) |
| `image-dir` | 服务器 | 目录，可重复 | 无 | 允许来自其他机器的请求引用该目录下的图像文件。来自本机的请求不需要该标志。[图像来源](image-sources.md) |
| `image-remote-only` | 服务器 | 开关 | 关闭 | 把来自本机的请求也当作远程处理。当本机上有东西把别人的请求转发给服务器时设置。[图像来源](image-sources.md) |

诊断标志：[诊断](diagnostics.md)

## 控制向量

控制向量（llama.cpp 格式的 GGUF）被加到残差流上；配合服务器时，请求用 `experimental_speed_projection`（`true` 或 `false`）切换它。

| 设置 | 位置 | 类型、取值 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| `control-vector` | 引擎 | 文件路径 | 关闭 | scale 为 1 的控制向量。 |
| `control-vector-scaled` | 引擎 | `FILE:SCALE[,...]` | 关闭 | 各带 scale 的控制向量。 |
| `control-vector-layer-range` | 引擎 | 两个层号，`A B` | 1 到最后一层 | 其生效的层范围，含端点。 |
| `cvec-mode` | 引擎 | `add`、`project` | `add` | `add`：h += s v。`project`：h -= s (h.v) v，其中 v 是单位向量。 |
| `cvec-dir` | 引擎 | `per-layer`、`single:L` | `per-layer` | 每层用各自的方向，或到处使用第 L 层的方向（`project`）。 |

## 采样

请求字段、请求继承的默认值、思考预算和循环保护见[采样](sampling.md)。
