# `.basalt` 模型文件

[English](../model-file.md)

一个模型就是一个文件。它装着引擎、服务器与 vision sidecar 要加载的一切：权重、tokenizer 与聊天模板、MTP drafter、8-bit 超连接副本、vision projector，以及带默认采样设置的 GGUF 元数据。它不带任何属于引擎策略的东西：没有 flag（那些来自 `--config` 或命令行），也没有专家 profile（那是引擎自己的）。

- **布局：** 一个 64 KiB 的头部（magic `BASALTMF`、版本 2、至多 511 个带偏移、大小与 SHA-256 的命名 section），其后是按 2 MiB 对齐的各 section。权威描述：[model_format.hpp](../../include/basalt/core/model_format.hpp)。
- **版本：** 引擎只读版本 2。版本 1 的文件可用 [tools/migrations/basalt_v1_to_v2.py](../../tools/migrations/README.md) 转换，无需重新下载。改了什么：[文件格式变更日志](basalt-file-changelog.md)。

## 从 GGUF 构建 pack

```sh
python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \
    [--mtp-source DIR [--mtp-dense FILE]] [--draft-vocab FILE] [--compat-bf16] [--name NAME]
```

`--help` 说明每个输入。`--compat-bf16` 用于转换以其他类型存储 BF16 投影的 GGUF。这样构建、又没有官方调校 drafter 的 pack 会比官方文件慢。

从分片下载中截出的单个文件会保留那次下载的 `split.count` 与 `split.tensors.count`，它们描述的是来源模型而非这个文件。当它声明的每个 tensor 都落在文件之内时，它按完整模型打开；否则以被截断为由拒绝；分片模型的各个分片仍会按这些键校验。

## Packs

- `dense`：F32 与 BF16 tensor。`gguf`：量化的非专家 tensor。`experts`：每层的路由专家。
- `engram`：仅 Engram 表（144 字节的 IQ4_NL 行），按 2 MiB 对齐。
- `vision/mmproj`：只在用 `--mmproj` 构建时才有；引擎与 `basalt-vision --model FILE.basalt` 都就地读取它。

## Flash-Next NVFP4 pack

官方文件是 NVIDIA 的 checkpoint（[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)，修订 `fc694b54`），稠密投影为 Q6_K。转换器直接读取 safetensors：专家逐位搬移，稠密 tensor 从 BF16 量化一次。

```sh
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --revision fc694b54 --local-dir nvidia

# 6-bit dense (the official file)
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia \
    --out Qwen3.8-FN-NVIDIA-NVFP4.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"

# 8-bit dense variant
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia --dense-q8-0 \
    --out Qwen3.8-FN-NVIDIA-NVFP4-Q8.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"
```

`--mtp-dense FILE`（官方 NVFP4、UD-IQ1_M、UD-IQ2_XS 或 UD-IQ3_XXS pack 的 `mtp/dense` 字节）会换入一个调校过的 drafter 层；转换器把它记录在 `meta.mtp.dense` 中。输出约 122 GiB（8-bit 变体为 123 GiB）。基础词表必须是 NVIDIA 的。
