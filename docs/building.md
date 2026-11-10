# Building and testing

[简体中文](zh/building.md)

Clone, requirements and the build commands are in the [README](../README.md); Windows: [platforms](platforms.md). This page has the rest.

## What is in the repository

- `src/`, `include/`: the engine (`basalt`, `basalt-bench`, `basalt-probe`, `basalt-vision`).
- `serve/`: the Python server with the OpenAI and Anthropic API and the stats page.
- `tools/`: the converter (`tools/convert/convert.py`), the `.basalt` reader (`model_file.py`), the quality gate (`ppl_gate.py`), the
  greedy-identity check and the tokenizer.
- `tests/`: parity and unit tests. `data/`: expert profiles (`data/profiles`) and benchmark fixtures (`data/bench`).

The build fetches nothing. The ggml and llama.cpp parts Basalt uses (quantization formats, CPU dot products, MMQ) are in `src/` and
`include/basalt/ggml/`, under the MIT licence in `THIRD_PARTY_LICENSES`.

`-DWERROR=ON` makes project warnings errors (CI sets it on Linux).

## Tests

Configure with `-DBUILD_TESTING=ON`, build, and run `ctest --test-dir build`.

- Kernel parity tests need the GPU. `expert_parity` and `pool_test` use a synthetic fixture; `--file experts.bin` checks real blobs.
- Server tests: `python -m unittest discover -s serve/tests -t .` after `pip install -r requirements-test.txt`.
- Tokenizer parity: register `tokenizer_parity` with `-DBASALT_TOKENIZER_DIR=<a pack's tokenizer/ directory>`.

The test that runs the built engine with every flag needs `bin/basalt`. Two tokenizer tests read the pack's vocabulary and skip without a pack; the same checks run on a synthetic vocabulary in the same file.

The engine tokenizes request text itself (`GENT`). `tokenizer_test` checks the C++ tokenizer on a synthetic vocabulary; `tokenizer_parity` compares its ids with `tools/tokenizer.py` over a corpus (with python3 and the `regex` package; `-DBASALT_TOKENIZER_BIG` adds `--big FILE...` to the corpus). `tokenizer_bench DIR FILE...` times a cold and a warm encode.

`basalt-bench` reads its pre-tokenized prompts from `data/bench` (token ids for the Qwen3.8-Flash-Next tokenizer: the three decode prompts and a prefill filler); run it from the repository root or pass `--fixtures DIR`. `basalt-bench --help` says how the files are made.

`basalt-bench --community` runs the fixed preset `community-1` for the public table in [community runs](community.md): the three decode fixtures, a 16384-token prefill and a 65536-token prefill, each once, with int8 KV cache and the engine's own defaults otherwise (drafting included). The decode fixtures run in one fresh engine process and each prefill in its own, with no warm-up pass and no repetition, so every number is the cold first run. It prints the usual table, adds its row to `docs/community/benchmarks.json` of the checkout the build came from (`--community-file PATH` picks another file) and prints it. Review the row, commit it and open a pull request:

```sh
bin/basalt-bench --community --model Qwen3.8-FN-NVFP4-DQ6_K.basalt --submitter YourName
```

`prefill_16k_tps` and `prefill_64k_tps` are the engine tokens per second of those two prompts; `decode_tps` is the generated tokens of conv, struct and prose over their summed decode time (token-weighted). The row records the engine commit, the GPUs the engine used (main first) with VRAM and enforced power limit from NVML, system RAM, the OS (`Linux`, `Windows` or `WSL2`), the pack's quant label. Every expert must sit in VRAM, the peer GPU or RAM: if the engine would read any from the SSD, the bench stops before measuring, says how many GiB would spill and roughly how much more free RAM is needed, exits non-zero and prints no row. NVML is required; without it, or without a git build, the run stops with a message. Only `--model`, `--submitter`, `--fixtures` (the repository's `data/bench`), `--logs` and `--print-command` are accepted: any other bench option or engine flag changes the workload and is refused. The run uses every visible Ada or Ampere GPU (`--gpu auto`; `--gpu` is refused); pick the cards with `CUDA_VISIBLE_DEVICES`, for example `CUDA_VISIBLE_DEVICES=0` for a single-GPU row. The preset's version changes whenever the workload does.
