# AGENTS.md

Adamant is a local inference engine for the Qwen3.8-Flash-Next mixture-of-experts model, built for NVIDIA Ada Lovelace (sm_89) on Linux and Windows x64; Ampere (sm_86) is accepted in scope. It runs on an Ada GPU, optionally with a second GPU as a peer expert tier, and system RAM holding the rest. It is a fork of Basalt (MIT), which is a fork of Strata (MIT): the C++/CUDA engine is in `src/` and `include/`, each model's own code under `src/models/<model>`, and the Python server with an OpenAI- and Anthropic-compatible API is in `serve/`.

## Working rules

`CONTRIBUTING.md` sets the project scope (CUDA on Ada and Ampere only) and what every change must include (parity test, profile, quality check). It applies to agents too.

- **Commits:** small and logical, message style `area: what it does`. Never amend and never force-push; always make new commits.
- **Platform code:** engine code never names an OS or a compiler. No `_WIN32`, `_MSC_VER`, `__linux__`, `<windows.h>`, `<sys/mman.h>`, `<unistd.h>` or raw OS calls (mmap, pread, pthread, fork/exec, dlopen, /proc, /sys) outside `src/platform/` and `include/basalt/platform/`. A missing seam is added there with a Linux and a Windows body; the Linux body is the code you would have written anyway. `tools/check_platform_boundary.sh` enforces it.
- **Log lines:** every engine stderr line goes through `include/basalt/log.hpp` with its real level (`info`, `warn`, `error`); never `fprintf(stderr, ...)`. Do not change the text of lines the server parses. `tools/check_stderr_logging.sh` enforces it. Engine and server log lines are terse facts (what was decided, its value, at most the flag to change it), not prose explanations.
- **Code comments:** brief and stateless. Say what the code does and why. No hardware names, measured numbers, dates or history in comments.
- **Don't push or publish anywhere** (GitHub, HF, any remote) without the maintainer's approval.
- **Don't write tools nobody asked for.**
- **No shims for removed things.** A removed or renamed flag, option, field or code path is simply gone: no alias, no fallback, no "was retired" message or list. The normal unknown-flag error covers it.
- **Linux and Windows are both supported; Linux is where we test, and CI compiles and CPU-tests Windows.** Changes must not remove the platform seams (`src/platform/`, CMake `if(WIN32)`); see `CONTRIBUTING.md` and `docs/platforms.md`.
- **Headline benchmark numbers come from the maintainer's own runs.** Agent benches are development A/Bs only. Label every number with its GPU power limit.
- **Quality gate:** PPL/KLD against the BF16 reference (`docs/precision/quality.md#the-bf16-reference`), through the decode path (`basalt-bench --quality`). Byte identity with old output is not required, but say whenever numerics change. Bitwise-neutral refactors need only the greedy-identity check.
- **Keep `docs/precision/` current.** Any engine change that can alter precision (weight formats, activation or accumulation types, KV/state formats, reduction order, decode or MTP paths) updates the page that covers it in the same change, with fresh numbers where it quotes them. The Ada port's open measurements are in `docs/precision/ada.md`.
- **Docs:** no file over 300 lines; split a longer topic into a folder of files; keep docs short, stateless and free of duplication.
- **Docs are bilingual (English + Simplified Chinese).** Every English doc file has a Chinese counterpart: `docs/**.md` → mirrored under `docs/zh/`, `README.md` → `README.zh-CN.md`, `CONTRIBUTING.md` → `CONTRIBUTING.zh-CN.md`; each pair cross-links at the top. When you change an English doc, change its Chinese counterpart in the same commit: same structure, same facts, translated prose (Simplified Chinese), code blocks and commands untouched, links repointed to the `docs/zh/` mirror. `AGENTS.md` has no Chinese counterpart.
- **The engine's own kernels are the reference.** Parity tests compare a changed kernel against the current one and against a plain CPU reference; Strata's original kernels are not kept as oracles. Every new kernel gets a parity test (random inputs + real geometry) and a microbench line.
- **Ada target:** `include/basalt/cuda_target.hpp` is the only place the compute capability is decided, including Ampere (sm_86) targets. Blackwell-only paths (tensor maps, distributed shared memory, programmatic dependent launch, native FP4) are not carried; see `docs/precision/ada.md`.
