# `.basalt` model file

[简体中文](zh/model-file.md)

A model is one file. It holds everything the engine, the server and the vision sidecar load: the weights, the
tokenizer and chat template, the MTP drafter, the 8-bit hyper-connection copies, the vision projector, and the GGUF
metadata with the default sampling. It carries nothing that is engine policy: no flags (those come from `--config` or
the command line) and no expert profile (the engine's own).

- **Layout:** a 64 KiB header (magic `BASALTMF`, version 2, up to 511 named sections with offset, size and SHA-256), then sections aligned
  to 2 MiB. Authoritative: [model_format.hpp](../include/basalt/core/model_format.hpp).
- **Versions:** the engine reads version 2 only. Version 1 files convert with
  [tools/migrations/basalt_v1_to_v2.py](../tools/migrations/README.md), without a download. What changed:
  [file format changes](basalt-file-changelog.md).

## Build a pack from a GGUF

```sh
python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \
    [--mtp-source DIR [--mtp-dense FILE]] [--draft-vocab FILE] [--compat-bf16] [--name NAME]
```

`--help` describes every input. `--compat-bf16` converts GGUF projections stored as BF16 in another type. A pack built this way
without the official tuned drafter is slower than the official file.

A single file cut out of a split download keeps that download's `split.count` and `split.tensors.count`, which describe
the model it came from rather than the file. It opens as a whole model when every tensor it declares lies inside it,
and is refused as cut short otherwise; the shards of a split model are still checked against those keys.

## Packs

- `dense`: F32 and BF16 tensors. `gguf`: quantized non-expert tensors. `experts`: routed experts per layer.
- `engram`: the Engram table alone (IQ4_NL rows of 144 bytes), 2 MiB aligned.
- `vision/mmproj`: present only when built with `--mmproj`; the engine and `basalt-vision --model FILE.basalt` read it in place.

## Flash-Next NVFP4 pack

The official file is NVIDIA's checkpoint ([nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4),
revision `fc694b54`) with dense projections as Q6_K. The converter reads the safetensors directly: experts move bit for bit, dense
tensors are quantized once from BF16.

```sh
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --revision fc694b54 --local-dir nvidia

# 6-bit dense (the official file)
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia \
    --out Qwen3.8-FN-NVIDIA-NVFP4.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"

# 8-bit dense variant
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia --dense-q8-0 \
    --out Qwen3.8-FN-NVIDIA-NVFP4-Q8.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"
```

`--mtp-dense FILE` (the `mtp/dense` bytes of an official NVFP4, UD-IQ1_M, UD-IQ2_XS or UD-IQ3_XXS pack) swaps in a tuned drafter layer; the converter
records it in `meta.mtp.dense`. Output is about 122 GiB (123 GiB for the 8-bit variant). The base's vocabulary must be NVIDIA's.

## FP4-Local checkpoints

`--checkpoint` also reads Model Optimizer's FP4-Local layout, as in
[nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local): the shards sit in `tensors/`,
the routed experts' scales are CSF stream pairs (`weight_scale.nvfp4_csf_fixed`, `weight_scale.nvfp4_csf_exceptions`), the
projections are MXFP8 and the n-gram table is NVFP4. The commands above take that folder as `--checkpoint`.

- **Experts:** each CSF pair decodes to the UE4M3 scales bit for bit; codes and scales then move as in the NVFP4 pack.
- **Projections:** MXFP8 values (E4M3 codes, one E8M0 scale per 32) widen to f32 exactly, then take the BF16 path: Q6_K or
  Q8_0, or BF16 where the engine reads BF16.
- **N-gram table:** E2M1 code × UE4M3 block scale × `weight_scale_2`, stored as Q8_0 (one rounding).
- **Drafter and vision:** copied from the base pack, as for the NVFP4 pack.
