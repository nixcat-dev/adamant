# Models

A model is one `.basalt` file. The server reads the architecture from the file and picks the chat format, tool syntax and thinking
switch. Several entries can share one address ([router](../configuration/models.md)). What the model supports:

| | Qwen3.8-Flash-Next |
| --- | --- |
| Architecture id | `qwen4exp` |
| Packs | 11 packs, default `Qwen3.8-FN-NVFP4-DQ6_K`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) |
| Layers, experts | 48 (36 GDN, 12 QSA); 512 per layer, top-10 |
| Context | default 8,192; `--max-context` sets it |
| Drafting | MTP drafter in the pack, on; `--spec` default 8 |
| Chat, tools, thinking | yes; `<tool_call>` tools |
| Reasoning effort | `none`, `low`, `medium`, `high`, `max` (`high` = `max`); a number is refused |
| Images | `--vision dynamic`, a GPU number or `cpu` |
| `--kv` | fp16 (default), int8, q4_0, k8v4 |
| Cards | main card plus up to three expert-tier cards |
| Layer split | yes, matched cards, priced at start |
| Parallel slots | yes; each slot drafts |
| Lossy flags | `--relaxed-drafting`, `--route-prior`, `--route-tail-skip` (off) |
| Idle sleep, wake | yes (Linux) |
| `basalt-bench --quality` | yes |
| Guide | [guide](qwen38-flash-next.md) |
| Precision | [precision](../precision/qwen38-flash-next.md) |

The adapter registry in `src/models/builtin.cpp` is the list of architectures the engine can run; a model's own code lives in
`src/models/<model>` ([adding a model](../adding-a-model.md)).
