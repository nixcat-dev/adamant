# Ada execution and validation

[简体中文](../zh/precision/ada.md)

Adamant's Ada implementation is in early development. The Linux sm_89 Release build
passes with CUDA 13.0.88, GCC 14.3.1, and warnings as errors. Runtime validation
is incomplete; compilation does not establish numerical correctness. Upstream
measurements in [qwen38-flash-next.md](qwen38-flash-next.md) and the pages beside it are not Adamant results.

## Ampere (sm_86)

`include/basalt/cuda_target.hpp` accepts compute capability 8.6 alongside 8.9;
`CMAKE_CUDA_ARCHITECTURES` takes `86`, `89` or `86;89` (default `89`), anything else is a
configure error. `gpu "auto"` selects every visible Ada or Ampere card; an explicit card of
another capability stops the start. No kernel uses FP8 or FP4 tensor instructions (NVFP4 runs
as integer MMQ), so no weight pack or KV format is excluded on 8.6 by design.

Ampere is not yet validated. The sm_86 build compiles; still open on Ampere hardware:

- Kernel parity tests against the current and CPU reference kernels.
- Decode-path PPL/KLD against the BF16 reference (`basalt-bench --quality`).
- Memcheck of the `cp.async` copy paths.
- Fast paths gated on SM count and their fallback on smaller cards.
- Performance and default tuning for 8 GB-class cards.

## Execution changes

| Operation | Adamant implementation | Numerical consequence requiring validation |
|---|---|---|
| MMQ prompt experts | Existing integer MMA tiles, without programmatic dependent launch; all configured formats retained | No arithmetic or tile-selection changes intended. Cross-device equality is not assumed. |
| NVFP4 prompt experts | Integer MMQ at all prompt sizes, not native FP4 tensor instructions | Activations use MMQ's q8_1 rounding instead of E2M1/UE4M3 rounding for long prompts. This changes numerics relative to upstream's native FP4 path. Weight bytes remain in the same format. |
| Dense tensor-core projections and QSA middle | Local async-copy staging and supported integer MMA instructions | Must compare scale application and reduction order to the reference and test token-width invariance. |
| Fused hyper-connection read | Cooperative async copies replace tensor-map transfers; hc8 storage and the unfused implementation remain | Copy/layout equivalence and unchanged arithmetic are design intentions, not measured bitwise identity. Selecting this read instead of the unfused read can change numerics. |
| Fused GDN recurrence and norm | Four column groups in one block, with local shared-memory exchange instead of distributed shared memory | Each rank's arithmetic and the order of norm partials are intended to match the reference. Requires output/state parity and synchronization validation. The separate recurrence/norm path remains. |
| Persistent GDN middle | Not part of the Ada implementation; projections, recurrence, and norm execute as separate kernels | Path selection and reduction order can differ from the upstream persistent path. Evaluate quality rather than assuming identity. |
| Staged GPU expert rows | Cooperative async copies and block barriers replace bulk-copy completion barriers | Weight bytes and row-dot arithmetic are intended to remain unchanged. Validate stage reuse, tail work items, and peer execution. |
| Grouped expert launches | Ordered stream launches replace dependent launches | The existing fused SwiGLU/q8_1 arithmetic remains; validate ordering and exact output. |
| Scalar CPU quantizers and activation parity references | Host compilation disables multiply-add contraction | Q8_K rounds its product to fp32 before integer rounding, matching explicit-rounding SIMD and GPU kernels. This can change scalar quantization compared with a compiler-contracted build; validate exact ties and tiny activations. |
| Weight prefetch | Per-cache-line prefetch replaces bulk prefetch | No model arithmetic changes intended. Profile independently; no speed claim is made. |
| Native GR RMSNorm weight prefetch | Loads gamma values before the original input reduction for supported row widths, then uses them in the output pass; the input accumulation loop is unchanged | `native_gr_norm_parity` matches the previous kernel bitwise and compares against a double-precision CPU reference. Development A/B at 450 W: 2560 columns, baseline→prefetch 2.769→2.493 µs; 4096 columns, 2.615→2.683 µs. Results vary by width; no end-to-end speed claim. Technique follows NInfer Apache-2.0 commit `9954867a`; no source was copied. |
| Native MMQ expert gather and output map | One grouped gather handles each native expert group after waits on its staged slots; identity row destinations are derived from expert bounds instead of a per-run iota buffer | Weight bytes and MMQ arithmetic are unchanged. Grouped/per-expert parity passed; IQ4_XS focused memcheck and racecheck reported zero errors and hazards. A deployed-pack memcheck completed a 511-token prefill and reached verify capture, then timed out without a sanitizer summary; it is incomplete. |
| Shared expert gate/up MMVQ pair | One launch handles two same-format Q8_0 or Q6_K projections over the same Q8_1 inputs; each output uses the existing exact-layout MMVQ row and reduction code | CPU reference and focused GPU parity pass; final-binary deployed-pack greedy check passed 900 tokens across 3 requests. Pair eligibility is restricted to 2–8 columns, exact multi-column mode, and a shared stream; other calls keep the prior launches. One matched development A/B at 450 W used the deployed IQ4_XS pack, profile `c31d790a…`, 7,461 resident slots, CPU share 0.60, PCIe share 0.55, context 262144, int8 KV, spec 4, chunk 32768, seed 1: 4096-token prefill 2777.34→2775.68 ms; decode conversion 152.396→153.537 ms, struct 232.277→234.923 ms, prose 152.182→152.530 ms; struct verify window 17.197→17.003 ms and verify 14.096→13.905 ms (58.15→58.81 tokens/s). This is one development pair with mixed small changes; it does not establish a general speedup. Full-model memcheck is incomplete. BF16 quality comparison is pending because a usable BF16 reference run was unavailable. |
| Decode CPU/PCIe miss split (`ADAMANT_PCIE_BALANCE=1`) | The per-layer fit changes the fixed 1.85 factor to a linear pool-rate factor from 1.85 at 30 GB/s to 0.8 at 12 GB/s; the rounded PCIe share cap rises from 0.60 to 0.95 | Scheduling changes which expert misses run on CPU versus GPU. Expert arithmetic is unchanged, but accumulation order and output numerics may change. BF16 quality validation and Ada performance measurements are pending; no quality or speed claim is made. |
| Single-request staged CPU misses (`--exp solo_stage=1`) | A layer's distinct CPU misses past a threshold are partly copied over PCIe from their RAM copies into staging and run on the GPU in single-request windows of two or more rows, not only in batch windows; layers of mixed expert sizes are accepted, every staging slot holding the largest blob and each copy moving its own layer's bytes. The split is planned from `stage_cpu_gbps` and `stage_link_gbps` (defaults 45 and 18). Off by default | Staged experts run the GPU grouped-expert path instead of the CPU pool, so accumulation order and output numerics change and greedy output diverges. Development A/B on an RTX 3090 at 350 W, IQ3_XXS, int8 KV, spec 8, two passes per depth: ms per generated token 6.91/7.38→6.67/7.23 at 16K, 9.09/7.71→8.38/7.81 at 135K, 10.20/9.96→9.55/9.99 at 223K, within run-to-run variance; tiers not pinned. Parity, PPL/KLD and window-width identity are pending; no quality or speed claim is made. |

### 0.3.1 paths

- **Mixed gate/up layers** (`experts.layout` v7: UD-IQ1_S layer 18 is IQ2_S gate with IQ2_XXS up). Each half is decoded in its
  own format by the same row code; the decode glue and the persistent peer kernel have mixed builds, the MMQ prompt slot is
  padded to whole blocks of both types, and the CPU tier runs each half through its type's row kernel. Intended bitwise-equal
  to the uniform kernels; `mixed_gu_parity` passes bitwise on sm_89.
- **Peer prompt share** (`prefill::PeerShare`). The second card computes its experts' prompt rows from staged group buffers
  with the same MMQ formats and the same q8_1 activations the main card writes, then returns the rows. Intended bitwise-equal
  to the main card's rows; upstream's `peer_share_parity` checks the gathered bytes. `peer_share_parity` and
  `peer_experts_parity` need two GPUs and skip on one; not measured on Ada. The FP4 streamed-chunk fallback is not
  carried, so chunks at or above `stream_all_min()` stay on the primary.
- **Glue-path formats** (Q5_K/Q6_K gate/up rows, IQ4_XS gate/up rows, (IQ3_S|IQ4_XS)/Q8_0 pairs). Same row dots and
  accumulation order, fused launches only; `dec_moe_parity --kq` and `native_grouped_parity --glue` pass bitwise on sm_89.
- **Batch K/V streaming** (`kv_stream`'s segment resolve, `--kv-resident` with `--batch`). Same rows through a segment
  cursor; `kv_stream_parity` passes on sm_89.

On the deployed IQ4_XS pack, 64 greedy tokens (prose ids, int8 KV, expert cache 700, swaps 0, one RTX 4090 at 450 W) are
identical between the pre-merge build and this one.

## Prompt memory and peer execution

Integer MMQ sizes intermediate buffers for all routed rows. Large NVFP4 prompt
chunks can therefore require substantially more scratch memory than the native
FP4 implementation; automatic fitting may select a smaller chunk. Validate actual
allocation sizes and long prompts rather than extrapolating from short fixtures.
Only routed-only prompt chunks are shared with the peer; chunks at or above
`stream_all_min()` stay on the primary. Both peer-exclusive streaming and ordinary
peer placement require separate coverage on a second card.

Prompt splitting retains its existing NVFP4 pack-eligibility check using weight
metadata, independently of the removed execution API. Its inherited cost model
and chunk thresholds have not been calibrated for integer MMQ on Ada; neither
split correctness nor its performance benefit has been established.

## Required evidence

The [contribution requirements](../../CONTRIBUTING.md) apply, in particular the
parity test every new or changed kernel needs. Existing kernel and prompt-path
parity tests must pass against the current reference kernels and independent CPU
references, including non-finite output rejection, model geometry, window widths,
and token positions.

Memory, race, initialization, and synchronization checks are required for changed
copy and synchronization paths. Resource and graph-capture behavior must be
verified on the target GPU, not inferred from a successful host test.

The completed port requires decode-path PPL/KLD and top-1 agreement against the
BF16 reference ([method](quality.md#the-bf16-reference)), plus prefill/long-context and
MTP on/off checks. Report the exact
model and fixture identities, pinned expert placement, reference provenance,
configuration, and enforced GPU power limit with results. Do not treat the
short quality fixture as coverage of every prompt path.

The gate is not complete. Kernel parity and memory, race, initialization and
synchronization checks pass on sm_89; end-to-end quality results and performance
claims have not been established. The development measurements taken so far, with
their power limit and configuration, are in the [README](../../README.md#measured-on-ada).
