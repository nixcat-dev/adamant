# 构建与测试

[English](../building.md)

克隆、要求和构建命令见 [README](../../README.zh-CN.md)；Windows 见[平台](platforms.md)。本页是其余内容。

## 仓库里有什么

- `src/`、`include/`：引擎（`basalt`、`basalt-bench`、`basalt-probe`、`basalt-vision`）。
- `serve/`：带 OpenAI 与 Anthropic API 以及统计页面的 Python 服务器。
- `tools/`：转换器（`tools/convert/convert.py`）、`.basalt` 读写器（`model_file.py`）、质量关卡（`ppl_gate.py`）、贪心一致性检查和分词器。
- `tests/`：一致性与单元测试。`data/`：专家 profile（`data/profiles`）和基准 fixture（`data/bench`）。

构建不拉取任何东西。Basalt 用到的 ggml 与 llama.cpp 部分（量化格式、CPU 点积、MMQ）位于 `src/` 和 `include/basalt/ggml/`，以 `THIRD_PARTY_LICENSES` 中的 MIT 许可证发布。

`-DWERROR=ON` 会把项目警告变成错误（CI 在 Linux 上开启）。

## 测试

用 `-DBUILD_TESTING=ON` 配置、构建，然后运行 `ctest --test-dir build`。

- kernel 一致性测试需要 GPU。`expert_parity` 和 `pool_test` 使用合成的 fixture；`--file experts.bin` 检查真实的二进制块。
- 服务器测试：先 `pip install -r requirements-test.txt`，再运行 `python -m unittest discover -s serve/tests -t .`。
- 分词器一致性：用 `-DBASALT_TOKENIZER_DIR=<pack 的 tokenizer/ 目录>` 注册 `tokenizer_parity`。

用全部标志运行已构建引擎的那个测试需要 `bin/basalt`。两个分词器测试读取 pack 的词表，没有 pack 时跳过；同样的检查会在同一文件里用合成词表运行。

引擎自己完成请求文本的分词（`GENT`）。`tokenizer_test` 在合成词表上检查 C++ 分词器；`tokenizer_parity` 在一个语料库上把它产生的 id 与 `tools/tokenizer.py` 比较（需要 python3 和 `regex` 包；`-DBASALT_TOKENIZER_BIG` 给语料库加上 `--big FILE...`）。`tokenizer_bench DIR FILE...` 对一次冷编码和一次热编码计时。

`basalt-bench` 从 `data/bench` 读取预分词的提示词（Qwen3.8-Flash-Next 分词器的 token id：三个 decode 提示词和一个 prefill 填充文本）；请在仓库根目录运行它，或传 `--fixtures DIR`。`basalt-bench --help` 会说明这些文件的制作方法。

`basalt-bench --community` 为[社区实测](community.md)里的公开表格运行固定预设 `community-1`：三个 decode fixture、一次 16384 token 的 prefill 和一次 65536 token 的 prefill，各跑一次，int8 KV cache，其余全部使用引擎自身默认值（包括 drafting）。decode fixture 在同一个新引擎进程里运行，每个 prefill 各用一个进程，没有预热也没有重复，所以每个数字都是冷启动的第一次运行。它会打印常规表格，把它的行加进构建来源那个 checkout 的 `docs/community/benchmarks.json`（`--community-file PATH` 可另选文件），并打印出来。检查该行、提交并开一个 pull request：

```sh
bin/basalt-bench --community --model Qwen3.8-FN-NVFP4-DQ6_K.basalt --submitter YourName
```

`prefill_16k_tps` 和 `prefill_64k_tps` 是这两个提示词的引擎每秒 token 数；`decode_tps` 是 conv、struct、prose 的生成 token 数除以其 decode 时间之和（按 token 加权）。该行记录引擎 commit、引擎用到的 GPU（主卡在前）及其来自 NVML 的 VRAM 和强制功耗限制、系统 RAM、操作系统（`Linux`、`Windows` 或 `WSL2`）、pack 的量化标签。每个专家都必须位于 VRAM、对等 GPU 或 RAM 中：若引擎会从 SSD 读取任何专家，bench 会在测量前停止，说明会溢出多少 GiB、大约还需要多少空闲 RAM，以非零退出码结束且不打印行。必须有 NVML；没有它、或不是 git 构建时，运行会带着提示停止。只接受 `--model`、`--submitter`、`--fixtures`（仓库的 `data/bench`）、`--logs` 和 `--print-command`：任何其他 bench 选项或引擎标志都会改变负载，会被拒绝。运行会使用每一块可见的 Ada 或 Ampere GPU（`--gpu auto`；`--gpu` 会被拒绝）；用 `CUDA_VISIBLE_DEVICES` 选卡，例如单 GPU 行用 `CUDA_VISIBLE_DEVICES=0`。负载一变，预设的版本号就会变。
