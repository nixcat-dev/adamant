# Contributing to Adamant

Adamant targets Qwen3.8-Flash-Next on NVIDIA Ada GPUs. Ada support is under development; the inherited Blackwell implementation is not evidence of Ada correctness or performance. The [parity, quality and benchmark rules](#what-every-change-must-include) below define what a change needs, and [Ada execution and validation](docs/precision/ada.md) records the paths still to be measured.

## Scope

- **CUDA on Ada (sm_89) only.** No ROCm/HIP, AMD, Intel/SYCL/oneAPI, Apple/Metal, or CPU-only main path (CPU experts held in RAM and `--vision cpu` remain supported). No multi-architecture compatibility layer. Unsupported instructions must be replaced with validated Ada implementations, not hidden behind launchable empty kernels or trap stubs. Preserve model-format coverage unless a scope change is explicitly approved.
- **Linux and Windows x64 are both supported; Linux is where we test.** CI compiles and CPU-tests Windows on every pull request. Windows runs with the same GPU roles as Linux (main card plus expert tiers), with no performance claims, and the Windows limitations are in `docs/platforms.md`. Changes must not remove the platform seams (`src/platform/`, CMake `if(WIN32)`). Windows-only issues get a `windows` label. Headline benchmark numbers stay with the maintainer's Linux runs.
- **Engine code is platform-neutral.** Outside `src/platform/` and `include/basalt/platform/` there is no `_WIN32`, `_MSC_VER`, `__linux__`, `<windows.h>`, `<sys/mman.h>`, `<unistd.h>` or raw OS call (mmap, madvise, pread, pthread/affinity, fork/exec/pipe, dlopen, /proc or /sys reads). Call the platform layer; when a seam is missing, add it there with a Linux body and a Windows body, the Linux body being the code you would have written anyway, and a test in `tests/platform/` that runs on both. Paths are UTF-8 and go through the platform file calls. `tools/check_platform_boundary.sh` enforces this.
- **Engine log lines carry their level.** A line to stderr goes through `include/basalt/log.hpp` (`basalt::log::info`, `warn` or `error`), which writes `basalt <area>: [warning: |error: ]<text>`; the server and the terminal read the level from that marker and never from the words. A warning is a fallback or a lost function, an error is a failure or a refusal, everything else is info. `tools/check_stderr_logging.sh` fails when engine code names stderr anywhere else.
- **No `.bat` or PowerShell scripts** are added to the tree. A documented build recipe that calls the toolchain's own scripts (`vcvarsall.bat`, say) from a shell is not one.
- **One fast path.** No per-user tuning flags or alternative kernels selected at run time. A change that wins replaces the old path; a change that doesn't win isn't merged. A/B and diagnostic flags are off by default and are not documented as settings. User-facing settings are limited to model, context, multi-GPU and serving options.
- **No environment-variable knobs.** Everything the engine reads is a command-line flag or a config entry; diagnostics (profiling, timing, logs) are flags too, off by default.

## What every change must include

1. **A parity test** for every new or changed kernel: random inputs plus the real model geometry, compared against (a) the current Basalt kernel it replaces and (b) the plain CPU reference for that operation (dequantize, then accumulate in double precision). Report the max absolute and relative error against both. A kernel meant to be bitwise-neutral must match the frozen output hashes of the current kernels. The output for a token must not depend on the batch/window width or the token's position in it; the test must check that.
2. **A profile.** Give a microbench line for the kernel: µs per call and achieved GB/s (or TFLOP/s) against the bytes or FLOPs it must move. Give an end-to-end number from `basalt-bench`: ms per decode window and passes/s, not only tok/s, which drafting luck moves by ±3-5%.
3. **A quality check:**
   - Bitwise-identical output: show the greedy-identity check against the previous build. Pin the expert tiers, since changes in GPU module size move the auto-sized tiers and change rounding.
   - Changed numerics (new reduction order, precision, quantization): run the perplexity/KLD gate against the BF16 reference ([docs/precision/quality.md](docs/precision/quality.md#the-bf16-reference)) through the decode path. Report PPL, mean KLD and top-1 agreement next to the previous build's. Large changes need the full 100-chunk run, not just the 20-chunk quick gate.
4. **Benchmark hygiene:**
   - Every performance number states the enforced GPU power limit. Match it between A/B runs; do not silently apply the upstream 400 W convention to a different machine.
   - A/B comparisons alternate the two builds in the same session. Numbers taken in different sessions drift and aren't comparable.
   - Keep warm-up and fixture order as `basalt-bench` defines them.

## Code style

- Kernels are written as `__device__` stage functions that take their block/warp index as an argument, with thin `__global__` wrappers, so they can be fused into persistent kernels later.
- Comments are brief and stateless: what the code does and why. No hardware names, measured numbers, dates or history in comments. Those belong in the commit message or the PR.
- Do not use TMA, distributed shared memory, thread-block clusters, programmatic dependent launch, or native FP4 instructions: the target does not support them.
- Keep adapted third-party code's license headers intact (ggml/llama.cpp is MIT).

## Commits and pull requests

- Small, logical commits with the message style `area: what it does`. Each commit must build.
- Never amend or force-push shared branches; add new commits.
- No notes, scratch files or generated reports in the repo.
- Docs: no file over 300 lines; split a longer topic into a folder of files; keep docs short, stateless and free of duplication.
- Docs are bilingual: every English doc file has a Simplified Chinese counterpart (`docs/` mirrored under `docs/zh/`, `README.zh-CN.md`, `CONTRIBUTING.zh-CN.md`), cross-linked at the top. Change both files of a pair in the same commit. See `AGENTS.md` for the full rule.
- A PR description lists: what changed, the parity results, the microbench and `basalt-bench` before/after (with the power limit), and the quality-check result.

## Other ways to help

- Cannot contribute code? Add a benchmark run from your hardware to the community table (details in [community runs](docs/community.md)):
  1. Run `basalt-bench --community --model PACK --submitter YourName`. It adds the row to `docs/community/benchmarks.json` ([building](docs/building.md)).
  2. Review the row. After a hand edit, format and check it with `python3 tools/check_benchmarks.py --fix docs/community/benchmarks.json`.
  3. Commit it.
  4. Open a pull request; the `pages` workflow runs the same check.
  5. Once merged, the row appears on the runs page.
- Ada (sm_89) machines with spare GPU time help validate the port; multi-card systems also cover peer expert execution.
- A few hours of remote access to a machine speeds up development.
- Or sponsor the project through GitHub Sponsors (the repository's Sponsor button): it pays for tokens for agentic development and for time on different hardware to test on.

## License

Basalt is MIT licensed and is based on Strata (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors). By contributing you agree your contribution is licensed under the same terms. See `LICENSE` and `THIRD_PARTY_LICENSES`.
