# .basalt 模型文件

[English](../model-file.md)

一个模型就是一个文件。它装着 engine、服务器与 vision sidecar 要加载的一切：权重、tokenizer 与聊天模板、MTP
drafter、8-bit 超连接副本、vision projector，以及带默认采样的 GGUF 元数据。它不带任何属于 engine 策略的东西：
没有 flag（那些来自 `--config` 或命令行），也没有专家 profile（那是 engine 自己的，[下文](#专家-profile)）。

布局图在 [README](../../README.zh-CN.md) 中。

- **布局：** 一个 64 KiB 头部（magic `BASALTMF`、版本 2、一张至多 511 个命名 section 的表，每项带其偏移、大小
  与 SHA-256），其后是各 section，每个对齐到 2 MiB，以便内存映射或以 `O_DIRECT` 读取。
- **section：** `meta`（JSON：源 GGUF 的元数据及其 `general.sampling.*` 默认值、每个 tensor 的类型、形状与偏移、
  来源与许可证）、`tokenizer/*`、`dense` 与 `dense.index`、`experts` 与 `experts.layout`（路由专家，按层排布）、
  `gguf`（所有非专家的源 tensor）、`mtp/*`（drafter）、`hc8`（decode 所读的超连接矩阵的 int8 副本，由转换器算好，
  启动时直接照原样读取）以及 `vision/*`。权威描述见
  [include/basalt/core/model_format.hpp](../../include/basalt/core/model_format.hpp)。
- **版本：** engine 只读版本 2，其他一律拒绝。版本 1 的文件用
  [tools/migrations/basalt_v1_to_v2.py](../../tools/migrations/README.md) 就地变成版本 2，无需重新下载。改了什么：
  [文件格式变更日志](basalt-file-changelog.md)。
- **照原样发布的权重：** 从 GGUF 构建的 pack 逐字节拷贝每个 tensor，除了 `--compat-bf16`：用于那些把 engine 以
  BF16 读取的小投影存成其他类型的 GGUF（转换器会转换它们并记录在 `meta` 中）。NVIDIA NVFP4 pack 逐位保留
  NVIDIA 的 NVFP4 专家，并把它的稠密 tensor 从 BF16 量化一次（[下文](#nvfp4-pack)）。

从源 GGUF 分片构建一个（给出分片 1；其余的就在它旁边）：

```sh
python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \
    [--mtp-source DIR [--mtp-dense FILE]] [--draft-vocab FILE] [--compat-bf16] [--name NAME]
```

`python3 tools/convert/convert.py --help` 说明所有输入。

从分片下载里截出的单个文件保留着那次下载的 `split.count` 与 `split.tensors.count`，它们描述的是来源模型而非
这个文件。当它声明的每个 tensor 都在文件之内时，它按完整模型打开；否则以被截断为由拒绝；分片模型的各个分片仍
会按这些键校验。

## Packs

你自己从 GGUF 转出来的 pack 没有官方文件那份调校过的 drafter，会比官方 `.basalt` 文件明显更慢。精度如何处理、
质量如何测量：[精度](precision/index.md)。

## NVFP4 pack

官方 NVFP4 文件是 NVIDIA 的 NVFP4 checkpoint
（[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)，修订 `fc694b54`），
把稠密投影存成 6-bit Q6_K 而不是 8-bit Q8_0，从而在 decode 中走 tensor-core 稠密路径。逐 tensor 的明细与其质量
在 Hugging Face 页面上。

Basalt 的转换器直接读 NVIDIA 的 safetensors（`convert.py --checkpoint`）。它把每个专家的 NVFP4 码与缩放因子一个
位都不改地搬进 Basalt 的 block，把每个稠密投影从 BF16 量化一次，把 norm 权重的 "+1" 折进 F32，把 GDN 值头排成
engine 读取的顺序，并从 NVIDIA 的 `config.json`、tokenizer 文件与模型卡片写出元数据和 tokenizer。
`meta.checkpoint` 记录每一步。

两种变体都可以自己从 NVIDIA 的 checkpoint 构建。基础 pack（版本 2）只提供 drafter 与 vision projector，逐字节
拷贝；`hc8` 副本由 NVIDIA 的权重算出。官方文件的则取自 UD-Q4_K_XL。基础的词表必须是 NVIDIA 的：

```sh
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --revision fc694b54 --local-dir nvidia

# 6-bit dense (the official file)
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia \
    --out Qwen3.8-FN-NVIDIA-NVFP4.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"

# 8-bit dense variant
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia --dense-q8-0 \
    --out Qwen3.8-FN-NVIDIA-NVFP4-Q8.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"
```

`--mtp-dense FILE` 用一个调校过的层替换从基础 pack 拷贝的 drafter 稠密部分。该文件是官方 IQ3_S、IQ3_XXS 或
Q2_0 pack 中 `mtp/dense` section 的原始字节（例如 `tools/model_file.py` 的
`ModelFile(path).read("mtp/dense")`）；其大小必须与基础的该 section 一致，转换器把它的 SHA-256 连同调校方式
`self_distilled` 记录在 `meta.mtp.dense` 中。把它加进任一命令：

```sh
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia \
    --mtp-dense tuned-mtp-dense.bin \
    --out Qwen3.8-FN-NVIDIA-NVFP4.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"
```

输出约 122 GiB（8-bit 变体 123 GiB）。各层上 NVFP4 如何计算：[精度](precision/index.md)。

## 专家 profile

哪些专家在 VRAM 中起步，取决于 engine 对该模型的专家 profile，即源码树中的 `data/profiles/<key>.prof`，其中
`<key>` 是由模型元数据得出的 `<general.architecture>-<block_count>x<expert_count>`（`qwen4exp-48x512`）。该模型
的每个 pack 都用它；没有该模型的文件时，专家按首次使用放置。启动行 `profile PATH: N ranked pairs` 点名所用的
文件。

一个 profile 就是一个服务器学到的东西：用一个有代表性的 prompt 组合配 `--expert-profile-save FILE` 跑一个
服务器，然后停掉它（QUIT 会保存文件）。对已有文件使用 `--expert-profile-save` 时也会从它起步，这就等于在替换
`data/profiles` 中那份之前先试用一个 profile。
