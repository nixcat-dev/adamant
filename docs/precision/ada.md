# Ada execution and validation

[简体中文](../zh/precision/ada.md)

Adamant's Ada implementation is in early development. The Linux sm_89 Release build
passes with CUDA 13.0.88, GCC 14.3.1, and warnings as errors. Runtime validation
is incomplete; compilation does not establish numerical correctness. Upstream
measurements in [qwen38-flash-next.md](qwen38-flash-next.md) and the pages beside it are not Adamant results.

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

## Prompt memory and peer execution

Integer MMQ sizes intermediate buffers for all routed rows. Large NVFP4 prompt
chunks can therefore require substantially more scratch memory than the native
FP4 implementation; automatic fitting may select a smaller chunk. Validate actual
allocation sizes and long prompts rather than extrapolating from short fixtures.
Streamed expert chunks execute on the primary GPU; peer computation remains for
routed-only chunks. Both peer-exclusive streaming and ordinary peer placement
require separate coverage on a second card.

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
