# 新增模型

[English](../adding-a-model.md)

一个模型位于 `src/models/<arch>/` 和 `include/basalt/models/<arch>/`。共享代码绝不提它的名字
（`tools/check_model_layering.sh` 会检查这一点）。

## 模型要写什么

- 一个 adapter（`include/basalt/model/adapter.hpp`），在 `src/models/builtin.cpp` 注册：校验文件、描述文件、加载文件，
  命名它的 token、pre-tokenizer 和 tool-call 形式。
- 一个 `ModelSession`，带 `decode_window`、`prefill_chunk` 和它支持的状态调用。
- `serve/models/` 里的一个方言：提示词渲染、输出解析、停止文本。

路由随后只需要一条配置条目。服务器里没有任何地方点名某个模型。

## Fixture、标志与加载

- benchmark 的 token id 放在 `data/bench/<arch>`。`--quality` 需要一个 BF16 参考。
- 标志：`parse_flag` 与 `finish_flags`。模型没有的标志就是正常的未知参数报错。token id 标志默认取 pack 的 token。
- 通过 `core::LoadProgress` 报告加载进度。RAM 层用 `core/ram_load`。
- 对话：`save_cells` 与 `restore_cells` 给 RAM 层和磁盘层所需的东西。
- 有自己快路径的模型从 `serve::Runtime` 和 `serve::Request` 派生；见 `src/models/qwen4exp/`。
- kernel：共享 kernel 留在 `src/kernels/`。每个新 kernel 都要有一致性测试和一条 microbench 行。
