# Ideas — untracked working notes

Status: **HAVE** (we already do it) / **PARTIAL** / **MISSING**. Sources cloned under `reference/`
(gitignored): Infernix = same model (Qwen4Exp/Flash-Next), sm_120a, 5090 32 GB, no peer GPU;
ninfer-4090 = sm_89, different (27B) model, no offload at all.

## 1. Rate-fitted split between CPU-compute and PCIe copy — MISSING
We have the CPU expert engine (`src/kernels/cpu/expert.cpp`, `iq_avx2`/`iq_avx512`/`kq_avx2`;
`tiers.cpp:296` "evicted now: the CPU computes it meanwhile"). What they add is choosing *per call,
per miss* which is cheaper: copy it or compute it. Their split is fitted from measured link and CPU
rates (`reference/Infernix/src/runtime/expert_offload/placement.h:30-41`, `256/(1+1.46*C*t)`, reference
link 27.5 GB/s, one switch at 1,536 context tokens for one-column decode). Preconditions: spare
physical cores (they reserve 4 logical CPUs for the engine thread and CUDA runtime, `placement.h:51-78`
— holding the process to 16 logical CPUs cost them 61-87 vs 189-217 tok/s).
Why it can matter for us: every miss costs host DRAM bandwidth either way; decode is latency-bound on
a few misses, prefill is link-bound. With a peer GPU we would have a *three*-way split, not two.
Risk: none to numerics if both routes agree (see item 2); needs a measured cost model, not a guess.

## 2. Placement/order-invariant exact arithmetic — PARTIAL
They guarantee every route (VRAM frame, pinned host, staged, landed, CPU) returns identical bits:
exact int32 per-16-block products, int64 row sums, one rounding at the boundary
(`reference/Infernix/src/ops/common/canonical_math.h:3-9`, `cpu/expert_team.h:5-20`). Activations are
re-encoded per column as 21-bit fixed point, so it is *not* bit-identical to a BF16-activation
reference (their README overstates this; `canonical_math.h:341-376`).
Ours: CPU and GPU are made to agree per format by explicit rounding rules (`docs/precision/ada.md:20`),
but we already document a placement/reduction-order-dependent numerics change
(`docs/precision/index.md:116`, the GDN mid chain on devices under 160 SMs). So: numerically close,
no invariance guarantee. This is the property that makes cache and tier policy testing trustworthy,
because placement stops being a variable.

## 3. Confidence-gated verify extent + column mask — PARTIAL
Their `MoeColumnMask`: a masked column routes nowhere — no job, no miss, no prefetch — and every live
column's bits equal the unmasked call (`reference/Infernix/.../offloaded_sparse_moe.h:52-70`); the
extent is chosen per round from learned per-bucket acceptance (`mtp_confidence_policy.h:8-27`).
Ours: the extent half exists (`src/spec/draft_policy.cpp`, adaptive window by measured cost and match
rate). We have no column mask, so padded or unverified columns still cost expert traffic — relevant
because our verify shape is fixed for graph capture.

## 4. Copy-ahead routing + layer-major prefill walk — CHECK BEFORE BUILDING
Guess layer l+1's routed experts from layer l's MoE input and start their DMA early: correct for 99%
of experts copied, +7.7-15% prefill (`reference/Infernix/src/runtime/expert_offload/expert_stream.h:26-31,
118-128`; starting rule `ahead_rows = max(columns/128, 8)`). Layer-major prefill streams each layer's
non-resident experts once per span instead of once per chunk. Open question: whether our prefill
already streams per span (`src/prefill/expert_stream*`) before spending anything here.

## 5. One-line A/B, no code: `--adapt-decay 0.85`
Ours is 0.7 (`include/basalt/program/options.hpp:191`). Their tuned value on our policy shape is 0.85.
Their other two numbers we already match: promotion floor `once_bar = 1.0f`
(`src/program/serve/state.hpp:283`) and swap margin 1.5 (`src/program/serve/tiers.cpp:209`).

## 6. Offline route-trace replay lab — MISSING
`reference/Infernix/tools/expert_cache_replay/` (`collect_traces.py`, `replay.py`, `route_predictor.py`
modelled operation-for-operation against the C++ policy, same addition order). Lets a cache/tier
policy change be A/B'd on captured routing instead of GPU hours. Host-side only, nothing at runtime.
This is the cheapest way to make item 5 and the tier policy measurable rather than anecdotal.

## 7. Hadamard-rotated int8 K — MISSING
We rotate V before Q4 (`src/kernels/cuda/kv_q4.cu:1`, Walsh-Hadamard) but quantize int8 K unrotated
(`kv_q8.cu`). They rotate K and Q with a normalized H256 as five shuffles plus a butterfly
(`reference/ninfer-4090/src/ops/kv_cache/hadamard_d256.cuh:10-65`) and argue it is what makes int8
keys safe. Changes numerics (quality-direction), needs its own gate.

## Not worth it
Their kernels (sm_120a: FP4 tensor cores, TMA, PDL, `cp.async.bulk`), their 96 MB L2 assumptions,
their constants fitted to a 5090 + Gen5 x8, and anything from ninfer-4090's MoE path (untested on a
4090, and their GEMM is behind ours). No raw benchmark data is committed in either repo, so every
published number in them is an external claim, not evidence.
