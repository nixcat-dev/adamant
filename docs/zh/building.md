# 构建与测试

[English](../building.md)

克隆、要求和构建命令在 [README](../../README.zh-CN.md) 里；Windows 见[平台](platforms.md)。本页是其余内容。

## 仓库里有什么

- `src/`、`include/`：C++/CUDA 引擎（`basalt`、`basalt-bench`、`basalt-probe` 以及图像编码器 `basalt-vision`）。
- `serve/`：一个 Python 服务器，带 OpenAI 和 Anthropic 兼容 API 以及统计页面。
- `tools/convert/`：模型文件转换器（`convert.py`）、它的 section 构建器和 drafter 获取器（`mtp_fetch.py`）。
- `tools/`：.basalt 读写器（`model_file.py`）、PPL/KLD 质量关卡（`ppl_gate.py`）、贪心一致性检查和分词器。
- `tests/`：一致性（parity）与单元测试。
- `data/`：引擎运行时读取的文件，其中包括专家 profiles（`data/profiles`，决定哪些专家初始驻留 VRAM）。自定义安装或容器镜像会把 `data/` 与二进制、`serve/` 和 `tools/` 一起打包。

## 警告

`-DWERROR=ON` 会把项目自身 C++ 和 CUDA 源码中的警告变成错误（CI 在 Linux 上开启）；默认关闭，这样新版编译器新增的警告就不会搞坏用户的构建。

## 测试

测试默认关闭。用额外选项 `-DBUILD_TESTING=ON` 配置 README 中的构建，构建后运行 `ctest --test-dir build`。

- kernel 一致性测试需要 GPU；文件格式、缓存和内存单元测试不需要。`expert_parity` 和 `pool_test`
  在合成的专家 fixture 上运行；传 `--file experts.bin`（pack 的 experts section，不随包发布）可以检查真实的二进制块。
  `platform_memory_test` 在锁定内存上限（`ulimit -l`）允许时锁定 256 MiB，否则锁定更小的区域，
  并会说明；只有当上限低到什么都锁不住时才会跳过锁定用例。
- 服务器测试：`pip install -r requirements-test.txt`（服务器的依赖）之后运行
  `python -m unittest discover -s serve/tests -t .`。
- 分词器一致性：用 `-DBASALT_TOKENIZER_DIR=<pack 的 tokenizer/ 目录>` 注册 `tokenizer_parity`。

用全部标志运行已构建引擎的那个测试需要 `bin/basalt`。两个分词器测试读取 pack 的词表，没有
pack 时跳过；同样的检查会在同一文件里用合成词表运行。

引擎自己完成请求文本的分词（`GENT`）。`tokenizer_test` 在合成词表上检查 C++ 分词器；`tokenizer_parity`
在一个语料库上把它产生的 id 与 `tools/tokenizer.py` 比较，当 cmake 拿到
`-DBASALT_TOKENIZER_DIR=<pack 提取出的 tokenizer/ 目录>` 时才会注册（需要 python3 和 `regex` 包；`-DBASALT_TOKENIZER_BIG`
给语料库加上 `--big FILE...`）。`tokenizer_bench DIR FILE...` 对一次冷编码和一次热编码计时。

`basalt-bench` 从 `data/bench` 读取预分词的提示词（Qwen3.8-Flash-Next 分词器的 token id：
三个 decode 提示词和一个 prefill 填充文本）；请在仓库根目录运行它，或传 `--fixtures DIR`。`basalt-bench
--help` 会说明这些文件的制作方法。

构建不依赖仓库之外的任何东西，也不拉取任何东西。Basalt 用到的 ggml 和 llama.cpp 的部分（量化格式、
CPU 点积、提示词路径的 MMQ kernel）已提取到 `src/` 和 `include/basalt/ggml/`，其 MIT 许可证在
`THIRD_PARTY_LICENSES` 中。

`basalt-bench --community` 为[社区实测](community.md)里的公开表格运行固定预设 `community-1`：
三个 decode fixture、一次 16384 token 的 prefill 和一次 65536 token 的 prefill，各跑一次，int8 KV cache，
其余全部使用引擎自身默认值（包括 drafting）。decode fixture 在同一个新引擎进程里运行，每个 prefill 各用
一个进程，没有预热也没有重复，所以每个数字都是冷启动的第一次运行。它会打印常规表格，把它的行加进构建来源那个
checkout 的 `docs/community/benchmarks.json`（`--community-file PATH` 可另选文件），并把它打印出来。检查这一行、提交它并开一个 pull request：

```sh
bin/basalt-bench --community --model Qwen3.8-FN-NVFP4-DQ6_K.basalt --submitter YourName
```

`prefill_16k_tps` 和 `prefill_64k_tps` 是引擎在这两个提示词上的每秒 token 数；`decode_tps` 是 conv、struct、prose 的生成 token 除以其 decode 时间之和（token 加权）。该行
记录引擎 commit、引擎用到的 GPU（主卡在前）及其来自 NVML 的 VRAM 和强制功耗限制、系统
RAM、操作系统（`Linux`、`Windows` 或 `WSL2`）、pack 的量化标签。每个专家都必须位于 VRAM、对等 GPU 或 RAM 中：若引擎会从 SSD 读取任何专家，bench 会在测量前停止，说明会有多少 GiB 溢出、大约还需要多少空闲 RAM，返回非零退出码并且不打印行。必须有 NVML；
没有它，或者不是 git 构建，运行会停止并给出提示。只接受 `--model`、`--submitter`、`--fixtures`（仓库的
`data/bench`）、`--logs` 和 `--print-command`：任何其他 bench 选项或引擎标志都会改变负载，会被拒绝。该运行使用每一块可见的 Ada 或 Ampere GPU（`--gpu auto`；拒绝 `--gpu`）；用
`CUDA_VISIBLE_DEVICES` 选卡，例如单卡行用 `CUDA_VISIBLE_DEVICES=0`。预设的版本会随负载的变化而变化。
