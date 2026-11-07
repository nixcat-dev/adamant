// tests/kernels/mmvq_multi_parity.cpp - the multi_exact contract of native_mmvq, tested bitwise.
//
//     build/mmvq_multi_parity
//
// WHY IT EXISTS.  `include/basalt/kernels/native_mmvq.hpp` says of `native_mmvq_set_multi_exact`:
//
//     "true (default): the ncols == 1 layout, every column bitwise equal to a single-column call"
//
// and `include/basalt/models/qwen4exp/decode/verify.hpp` counts "multi-column MMVQ in exact mode" among the reasons the verify window
// produces what greedy decode would, bit for bit.  Nothing tested that sentence: `iq_parity` calls `native_mmvq`
// with two columns and compares against a float64 reference at 2e-2, which cannot see a difference between one
// column and T columns.
//
// WHAT IT COMPARES.  The SAME Q8_1 bytes: one T-column call against T single-column calls, each reading column j
// from the same buffer (the header: "its Q8_1 blocks at j * n_in / 32").  The quantization is shared, so the only
// thing under test is how the columns are laid out.
//
// THE DATA.  The weights are random bytes, except each block's fp16 scales (`d`, and `dmin` where the format has
// one), which are rewritten as normal fp16 values in +-[2^-10, 2^-5): random bytes there would put inf and NaN in
// the outputs, and on the GPU every NaN comes out as the same canonical bit pattern, so a NaN output compares equal
// whatever operations produced it and proves nothing.  With finite scales and finite normal activations (the
// precondition the header puts on the input) every output is finite, and a non-finite one fails the test.
//
// THE NEGATIVE CONTROL.  The same comparison with `multi_exact` off must find differences, or the test has no power
// and its pass means nothing.  Where it has power depends on T: in `native_mmvq.cu` the generic layout uses
// `NW = NCOLS <= 4 ? 4 : 2` warps, and with `WARPS = 4` and `BPI = F::BPI * NW / WARPS` it coincides with the
// exact layout at T <= 4.  So the control is ASSERTED on T > 4 for every case, and at T <= 4 its silence is
// reported as that coincidence rather than skipped.  The control needs a row long enough for the two layouts to
// group blocks differently: IQ4_XS takes 16 blocks per iteration with 4 warps and 8 with 2, so at n_in = 2048 (8
// blocks of 256) each thread holds at most one block in both layouts and they coincide; its case uses n_in = 4096.
//
// THE TENSOR-CORE PATH (MMVQ_MMA).  Where the device runs it, IQ4_XS calls with T > 1 in exact mode take
// native_iq4_xs_mma_kernel.  The last section turns it on with native_mmvq_set_mma and compares, on the same Q8_1
// bytes, its T-column call against the exact dp4a multi-column kernel (the setter off) and against T single-column
// calls (the ncols == 1 kernel, which the tensor-core path never replaces), bitwise, over row tails, rows of 1, 10,
// 16, 24 and 48 blocks, both block sizes the launcher picks, and weights / activations off their 8- and 16-byte
// alignment; the launch counter must move on every tensor-core call, or the check checked nothing.  With
// MMVQ_MMA=1 in the environment the sweeps above take it for their IQ4_XS case as well.
//
//     build/mmvq_multi_parity [--bench] [--pair-bench --power-limit-w W]
//     --bench: the existing dp4a and tensor-core kernels timed at the model's shapes
//     --pair-bench: paired and separate gate/up MMVQ calls at N=2560, M=640, T=8; W labels the enforced limit
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/artifact/dequant.hpp"
#include "basalt/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Case {
    const char* name;
    int type;          // stable GGML type id, from the list in native_mmvq.hpp
    int n_in;
    int n_out;
    int block_elems;   // elements per weight block
    int block_bytes;
    int scale_at[2];   // byte offsets of the block's fp16 scales in native_mmvq.cu's block structs; -1 = none
};

// the types a head or a projection can have in this model
const Case CASES[] = {
    {"Q4_0", 2, 2048, 512, 32, 18, {0, -1}},          // Q40Block: half d
    {"Q8_0", 8, 2048, 512, 32, 34, {0, -1}},          // Q80Block: half d
    {"Q4_K", 12, 2048, 512, 256, 144, {0, 2}},        // Q4KBlock: half2 dm
    {"Q5_K", 13, 2048, 512, 256, 176, {0, 2}},        // Q5KBlock: half2 dm
    {"Q6_K", 14, 2048, 512, 256, 210, {208, -1}},     // Q6KBlock: half d after ql, qh, scales
    {"IQ4_XS", 23, 4096, 512, 256, 136, {0, -1}},     // IQ4XSBlock: half d; n_in 4096, see THE NEGATIVE CONTROL
    {"Q5_K wide", 13, 4096, 2048, 256, 176, {0, 2}},
};

// a normal fp16 in +-[2^-10, 2^-5): exponent field 5..9 (bias 15), any mantissa, either sign
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

// T = 4 is the last width where the two layouts coincide, T = 5 the first where they differ
const int WIDTHS[] = {1, 2, 3, 4, 5, 6, 8};
constexpr int COINCIDE_MAX_T = 4;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// Returns the number of outputs that differ bit for bit; counts the non-finite outputs and the differences on
// outputs that are finite in both paths.
long long compare(const Case& c, int T, cudaStream_t s, long long& nonfinite, long long& diff_finite, bool& ran) {
    ran = false;
    if (!basalt::kernels::native_mmvq_supported(c.type)) return -1;

    const std::size_t wbytes = basalt::kernels::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    const std::size_t qcol = basalt::kernels::native_q8_1_bytes(c.n_in, 1);   // Q8_1 bytes of ONE column
    const std::size_t qall = basalt::kernels::native_q8_1_bytes(c.n_in, T);
    if (qall != qcol * (std::size_t) T) {
        std::printf("%-12s T=%d: the Q8_1 bytes of T columns (%zu) are not T times those of one (%zu): the test "
                    "assumes contiguous columns and they are not\n", c.name, T, qall, qcol);
        return -1;
    }

    std::mt19937 rng(1234u + (unsigned) c.type * 97u + (unsigned) T);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t n_blocks = (std::size_t) c.n_out * (std::size_t) (c.n_in / c.block_elems);
    if (n_blocks * (std::size_t) c.block_bytes != wbytes) {
        std::printf("%-12s T=%d: %zu blocks of %d bytes are not the %zu weight bytes native_mmvq expects\n", c.name,
                    T, n_blocks, c.block_bytes, wbytes);
        return -1;
    }
    for (std::size_t k = 0; k < n_blocks; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    std::vector<float> x((std::size_t) T * c.n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);

    void* dw = nullptr;
    void* xq = nullptr;
    float* dx = nullptr;
    float* dmulti = nullptr;
    float* dsingle = nullptr;
    const std::size_t ybytes = (std::size_t) T * c.n_out * 4;
    if (!ck(cudaMalloc(&dw, wbytes), "malloc w") || !ck(cudaMalloc(&xq, qall), "malloc xq") ||
        !ck(cudaMalloc(&dx, x.size() * 4), "malloc x") || !ck(cudaMalloc(&dmulti, ybytes), "malloc multi") ||
        !ck(cudaMalloc(&dsingle, ybytes), "malloc single"))
        return -1;
    if (!ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "copy w") ||
        !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x"))
        return -1;

    // ONE quantization, shared by both paths: the quantization is not under test
    basalt::kernels::quantize_q8_1_rows(dx, T, c.n_in, xq, s);
    if (!ck(cudaStreamSynchronize(s), "quantize")) return -1;

    try {
        basalt::kernels::native_mmvq(c.type, dw, xq, dmulti, c.n_in, c.n_out, T, s);
        for (int j = 0; j < T; ++j)
            basalt::kernels::native_mmvq(c.type, dw, (const uint8_t*) xq + (std::size_t) j * qcol,
                                         dsingle + (std::size_t) j * c.n_out, c.n_in, c.n_out, 1, s);
    } catch (const std::exception& e) {
        std::printf("%-12s T=%d: %s\n", c.name, T, e.what());
        return -1;
    }
    if (!ck(cudaStreamSynchronize(s), "mmvq")) return -1;

    std::vector<uint32_t> a((std::size_t) T * c.n_out), b(a.size());
    if (!ck(cudaMemcpy(a.data(), dmulti, ybytes, cudaMemcpyDeviceToHost), "read multi") ||
        !ck(cudaMemcpy(b.data(), dsingle, ybytes, cudaMemcpyDeviceToHost), "read single"))
        return -1;

    long long diff = 0;
    nonfinite = 0;
    diff_finite = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float fa, fb;
        std::memcpy(&fa, &a[i], 4);
        std::memcpy(&fb, &b[i], 4);
        const bool both_finite = std::isfinite(fa) && std::isfinite(fb);
        if (!both_finite) ++nonfinite;
        if (a[i] != b[i]) {
            ++diff;
            if (both_finite) ++diff_finite;
        }
    }
    cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(dmulti); cudaFree(dsingle);
    ran = true;
    return diff;
}

struct Totals {
    long long checked = 0, diff = 0, diff_finite = 0, nonfinite = 0;   // all widths
    long long diff_finite_wide = 0;                                     // T > COINCIDE_MAX_T only
    int powerless_cases = 0;   // cases where the control found no difference at any T > COINCIDE_MAX_T
};

int sweep(bool exact, cudaStream_t s, Totals& t) {
    basalt::kernels::native_mmvq_set_multi_exact(exact);
    if (basalt::kernels::native_mmvq_multi_exact() != exact) {
        std::printf("native_mmvq_multi_exact() does not reflect what was set\n");
        return 1;
    }
    std::printf("\n=== multi_exact %s ===\n", exact ? "ON (the default)" : "off (negative control)");
    std::printf("%-12s %3s %10s %10s %14s %10s\n", "type", "T", "outputs", "bits diff", "finite diff", "nonfinite");
    int bad = 0;
    for (const Case& c : CASES) {
        long long case_wide = 0;
        for (int T : WIDTHS) {
            long long nf = 0, dfin = 0;
            bool ran = false;
            const long long diff = compare(c, T, s, nf, dfin, ran);
            if (!ran) { ++bad; continue; }
            const long long n = (long long) T * c.n_out;
            t.checked += n;
            t.diff += diff;
            t.diff_finite += dfin;
            t.nonfinite += nf;
            if (T > COINCIDE_MAX_T) { t.diff_finite_wide += dfin; case_wide += dfin; }
            const char* verdict;
            if (exact) verdict = diff == 0 ? "ok" : "FAIL";
            else if (dfin > 0) verdict = "control sees it";
            else verdict = T <= COINCIDE_MAX_T ? "layouts coincide here (NW = 4)" : "no power at this T";
            std::printf("%-12s %3d %10lld %10lld %14lld %10lld  %s\n", c.name, T, n, diff, dfin, nf, verdict);
            if (exact && diff != 0) ++bad;
        }
        if (!exact && case_wide == 0) ++t.powerless_cases;
    }
    return bad;
}

struct PairQ81Block {
    uint16_t d;
    uint16_t sum;
    int8_t qs[32];
};
static_assert(sizeof(PairQ81Block) == 36);

uint16_t pair_sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t)(((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

bool pair_case(int type, int n_in, int n_out, int ncols, cudaStream_t stream) {
    namespace k = basalt::kernels;
    const int block_elems = type == 8 ? 32 : 256;
    const int block_bytes = type == 8 ? 34 : 210;
    const int scale_offset = type == 8 ? 0 : 208;
    const std::size_t weight_bytes = k::native_mmvq_weight_bytes(type, n_in, n_out);
    const std::size_t qcol_bytes = k::native_q8_1_bytes(n_in, 1);
    const std::size_t qbytes = k::native_q8_1_bytes(n_in, ncols);
    const std::size_t output_count = std::size_t(ncols) * n_out;
    std::mt19937 rng(0x73a1u + (unsigned)type * 31u + (unsigned)n_out);
    std::vector<uint8_t> wa(weight_bytes), wb(weight_bytes);
    for (uint8_t& value : wa) value = (uint8_t)rng();
    for (uint8_t& value : wb) value = (uint8_t)rng();
    const std::size_t blocks = std::size_t(n_out) * (n_in / block_elems);
    for (std::size_t i = 0; i < blocks; ++i) {
        const uint16_t ha = pair_sane_half(rng), hb = pair_sane_half(rng);
        std::memcpy(wa.data() + i * block_bytes + scale_offset, &ha, sizeof(ha));
        std::memcpy(wb.data() + i * block_bytes + scale_offset, &hb, sizeof(hb));
    }
    std::normal_distribution<float> normal(0.0f, 0.7f);
    std::vector<float> x(std::size_t(ncols) * n_in);
    for (float& value : x) value = normal(rng);
    if (ncols > 1)
        std::copy_n(x.begin(), n_in, x.begin() + std::size_t(ncols - 1) * n_in);

    void *dwa = nullptr, *dwb = nullptr, *dxq = nullptr;
    float* dx = nullptr;
    float *d_base_a = nullptr, *d_base_b = nullptr, *d_pair_a = nullptr, *d_pair_b = nullptr;
    float *d_single_a = nullptr, *d_single_b = nullptr;
    auto alloc = [&](void** pointer, std::size_t bytes, const char* label) {
        return ck(cudaMalloc(pointer, bytes), label);
    };
    const std::size_t output_bytes = (output_count + 2) * sizeof(float);
    bool ok = alloc(&dwa, weight_bytes, "pair weights A") && alloc(&dwb, weight_bytes, "pair weights B") &&
              alloc((void**)&dx, x.size() * sizeof(float), "pair input") && alloc(&dxq, qbytes, "pair Q8_1") &&
              alloc((void**)&d_base_a, output_bytes, "pair baseline A") &&
              alloc((void**)&d_base_b, output_bytes, "pair baseline B") &&
              alloc((void**)&d_pair_a, output_bytes, "pair result A") &&
              alloc((void**)&d_pair_b, output_bytes, "pair result B") &&
              alloc((void**)&d_single_a, output_bytes, "pair single A") &&
              alloc((void**)&d_single_b, output_bytes, "pair single B");
    if (!ok) return false;
    ok = ck(cudaMemcpy(dwa, wa.data(), weight_bytes, cudaMemcpyHostToDevice), "copy pair weights A") &&
         ck(cudaMemcpy(dwb, wb.data(), weight_bytes, cudaMemcpyHostToDevice), "copy pair weights B") &&
         ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy pair input") &&
         ck(cudaMemset(d_base_a, 0xa5, output_bytes), "fill pair baseline A") &&
         ck(cudaMemset(d_base_b, 0xa5, output_bytes), "fill pair baseline B") &&
         ck(cudaMemset(d_pair_a, 0xa5, output_bytes), "fill pair output A") &&
         ck(cudaMemset(d_pair_b, 0xa5, output_bytes), "fill pair output B") &&
         ck(cudaMemset(d_single_a, 0xa5, output_bytes), "fill pair single A") &&
         ck(cudaMemset(d_single_b, 0xa5, output_bytes), "fill pair single B");
    if (!ok) return false;
    k::native_quantize_q8_1(dx, dxq, n_in, ncols, stream);
    k::native_mmvq(type, dwa, dxq, d_base_a + 1, n_in, n_out, ncols, stream);
    k::native_mmvq(type, dwb, dxq, d_base_b + 1, n_in, n_out, ncols, stream);
    bool launched = false;
    try {
        launched = k::native_mmvq_pair(type, dwa, dwb, dxq, d_pair_a + 1, d_pair_b + 1,
                                       n_in, n_out, ncols, stream);
    } catch (const std::exception& error) {
        std::printf("MMVQ pair type=%d N=%d M=%d T=%d: %s\n", type, n_in, n_out, ncols, error.what());
        return false;
    }
    if (ncols == 1 ? launched : !launched) {
        std::printf("MMVQ pair type=%d N=%d M=%d T=%d: wrong eligibility result\n", type, n_in, n_out, ncols);
        return false;
    }
    for (int token = 0; token < ncols; ++token) {
        const auto* q = static_cast<const uint8_t*>(dxq) + std::size_t(token) * qcol_bytes;
        k::native_mmvq(type, dwa, q, d_single_a + 1 + std::size_t(token) * n_out, n_in, n_out, 1, stream);
        k::native_mmvq(type, dwb, q, d_single_b + 1 + std::size_t(token) * n_out, n_in, n_out, 1, stream);
    }
    ok = ck(cudaStreamSynchronize(stream), "pair sync");
    std::vector<uint8_t> host_q(qbytes);
    ok = ok && ck(cudaMemcpy(host_q.data(), dxq, qbytes, cudaMemcpyDeviceToHost), "read pair Q8_1");
    std::vector<float> base_a(output_count + 2), base_b(output_count + 2), pair_a(output_count + 2),
                       pair_b(output_count + 2), single_a(output_count + 2), single_b(output_count + 2);
    ok = ok && ck(cudaMemcpy(base_a.data(), d_base_a, output_bytes, cudaMemcpyDeviceToHost), "read pair baseline A") &&
         ck(cudaMemcpy(base_b.data(), d_base_b, output_bytes, cudaMemcpyDeviceToHost), "read pair baseline B") &&
         ck(cudaMemcpy(pair_a.data(), d_pair_a, output_bytes, cudaMemcpyDeviceToHost), "read pair A") &&
         ck(cudaMemcpy(pair_b.data(), d_pair_b, output_bytes, cudaMemcpyDeviceToHost), "read pair B") &&
         ck(cudaMemcpy(single_a.data(), d_single_a, output_bytes, cudaMemcpyDeviceToHost), "read pair single A") &&
         ck(cudaMemcpy(single_b.data(), d_single_b, output_bytes, cudaMemcpyDeviceToHost), "read pair single B");
    if (!ok) return false;
    const uint32_t guard = 0xa5a5a5a5u;
    bool parity = true;
    double max_abs_cpu = 0.0, max_rel_cpu = 0.0;
    static int frozen_type = 0;
    static int frozen_n_in = 0;
    static int frozen_n_out = 0;
    static std::vector<uint8_t> frozen_q;
    static std::vector<float> frozen_a, frozen_b;
    const bool track_positions = n_in == 2560 && n_out == 640;
    bool same_row_position = true;
    if (track_positions && ncols == 1) {
        frozen_type = type;
        frozen_n_in = n_in;
        frozen_n_out = n_out;
        frozen_q.assign(host_q.begin(), host_q.begin() + qcol_bytes);
        frozen_a.assign(single_a.begin() + 1, single_a.begin() + 1 + n_out);
        frozen_b.assign(single_b.begin() + 1, single_b.begin() + 1 + n_out);
    }
    if (track_positions && ncols > 1) {
        const bool have_frozen = frozen_type == type && frozen_n_in == n_in && frozen_n_out == n_out &&
                                 frozen_q.size() == qcol_bytes && frozen_a.size() == std::size_t(n_out) &&
                                 frozen_b.size() == std::size_t(n_out);
        same_row_position = have_frozen;
        if (have_frozen) {
            const std::size_t last_q = std::size_t(ncols - 1) * qcol_bytes;
            same_row_position = same_row_position &&
                                std::memcmp(host_q.data(), frozen_q.data(), qcol_bytes) == 0 &&
                                std::memcmp(host_q.data() + last_q, frozen_q.data(), qcol_bytes) == 0;
            for (int row = 0; row < n_out; ++row) {
                for (int token : {0, ncols - 1}) {
                    const std::size_t at = std::size_t(token) * n_out + row + 1;
                    same_row_position = same_row_position &&
                                        std::memcmp(&single_a[at], &frozen_a[row], sizeof(float)) == 0 &&
                                        std::memcmp(&single_b[at], &frozen_b[row], sizeof(float)) == 0 &&
                                        std::memcmp(&pair_a[at], &frozen_a[row], sizeof(float)) == 0 &&
                                        std::memcmp(&pair_b[at], &frozen_b[row], sizeof(float)) == 0;
                }
            }
        }
    }
    parity = parity && same_row_position;
    for (const auto* values : {&base_a, &base_b, &pair_a, &pair_b, &single_a, &single_b}) {
        uint32_t first = 0, last = 0;
        std::memcpy(&first, values->data(), sizeof(first));
        std::memcpy(&last, values->data() + output_count + 1, sizeof(last));
        parity = parity && first == guard && last == guard;
    }
    std::vector<float> weight_row(n_in);
    std::vector<double> input_row(n_in);
    for (int matrix = 0; matrix < 2; ++matrix) {
        const auto& packed = matrix == 0 ? wa : wb;
        const auto& pair_result = matrix == 0 ? pair_a : pair_b;
        const auto& previous = matrix == 0 ? base_a : base_b;
        const auto& one_col = matrix == 0 ? single_a : single_b;
        const auto& candidate = ncols == 1 ? previous : pair_result;
        if (ncols == 1) {
            for (std::size_t i = 1; i <= output_count; ++i)
                parity = parity && std::memcmp(&pair_result[i], &guard, sizeof(guard)) == 0;
        }
        for (int row = 0; row < n_out; ++row) {
            const uint8_t* weight = packed.data() + std::size_t(row) * (n_in / block_elems) * block_bytes;
            for (int block = 0; block < n_in / block_elems; ++block) {
                const uint8_t* packed_block = weight + std::size_t(block) * block_bytes;
                float* decoded = weight_row.data() + std::size_t(block) * block_elems;
                if (type == 8) basalt::dequantize_q8_0(packed_block, decoded);
                else basalt::dequantize_q6_K(packed_block, decoded);
            }
            for (int token = 0; token < ncols; ++token) {
                const auto* q = reinterpret_cast<const PairQ81Block*>(
                    host_q.data() + std::size_t(token) * qcol_bytes);
                for (int block = 0; block < n_in / 32; ++block) {
                    const float scale = k::f32_from_f16(q[block].d);
                    for (int i = 0; i < 32; ++i)
                        input_row[std::size_t(block) * 32 + i] = double(scale) * q[block].qs[i];
                }
                double reference = 0.0;
                for (int i = 0; i < n_in; ++i) reference += double(weight_row[i]) * input_row[i];
                const std::size_t at = std::size_t(token) * n_out + row + 1;
                const float value = candidate[at];
                const double error = std::abs(double(value) - reference);
                max_abs_cpu = std::max(max_abs_cpu, error);
                max_rel_cpu = std::max(max_rel_cpu, error / std::max(std::abs(reference), 1.0e-30));
                parity = parity && std::isfinite(value) &&
                         std::memcmp(&candidate[at], &previous[at], sizeof(float)) == 0 &&
                         std::memcmp(&candidate[at], &one_col[at], sizeof(float)) == 0 &&
                         (ncols == 1 || std::memcmp(&pair_result[at], &previous[at], sizeof(float)) == 0) &&
                         error <= 5.0e-4 * std::max(1.0, std::abs(reference));
            }
        }
    }
    std::printf("native_mmvq_pair: type=%s N=%d M=%d T=%d %s same_row_position=%s "
                "previous_max_abs=0 previous_max_rel=0 "
                "cpu_max_abs=%.3e cpu_max_rel=%.3e\n",
                type == 8 ? "Q8_0" : "Q6_K", n_in, n_out, ncols, parity ? "PASS" : "FAIL",
                track_positions && ncols > 1 ? (same_row_position ? "PASS" : "FAIL") : "n/a",
                max_abs_cpu, max_rel_cpu);
    cudaFree(dwa); cudaFree(dwb); cudaFree(dx); cudaFree(dxq); cudaFree(d_base_a); cudaFree(d_base_b);
    cudaFree(d_pair_a); cudaFree(d_pair_b); cudaFree(d_single_a); cudaFree(d_single_b);
    return parity;
}

int pair_section(cudaStream_t stream) {
    int failures = 0;
    for (int type : {8, 14}) {
        for (int ncols = 1; ncols <= 8; ++ncols)
            failures += !pair_case(type, 2560, 640, ncols, stream);
        for (int n_out : {1, 3, 639, 641}) {
            for (int ncols : {2, 5, 8})
                failures += !pair_case(type, 256, n_out, ncols, stream);
        }
        for (int n_out : {639, 641}) {
            for (int ncols : {2, 5, 8})
                failures += !pair_case(type, 2560, n_out, ncols, stream);
        }
    }
    basalt::kernels::native_mmvq_set_multi_exact(false);
    bool unexpected_launch = false;
    float *ya = nullptr, *yb = nullptr;
    void *wa = nullptr, *wb = nullptr, *xq = nullptr;
    const int n_in = 2560, n_out = 640, ncols = 2;
    bool allocated = ck(cudaMalloc(&wa, basalt::kernels::native_mmvq_weight_bytes(8, n_in, n_out)), "pair control weights A") &&
                     ck(cudaMalloc(&wb, basalt::kernels::native_mmvq_weight_bytes(8, n_in, n_out)), "pair control weights B") &&
                     ck(cudaMalloc(&xq, basalt::kernels::native_q8_1_bytes(n_in, ncols)), "pair control input") &&
                     ck(cudaMalloc((void**)&ya, std::size_t(ncols) * n_out * sizeof(float)), "pair control output A") &&
                     ck(cudaMalloc((void**)&yb, std::size_t(ncols) * n_out * sizeof(float)), "pair control output B");
    if (!allocated) {
        cudaFree(wa); cudaFree(wb); cudaFree(xq); cudaFree(ya); cudaFree(yb);
        basalt::kernels::native_mmvq_set_multi_exact(true);
        return failures + 1;
    }
    cudaMemset(ya, 0xa5, std::size_t(ncols) * n_out * sizeof(float));
    cudaMemset(yb, 0xa5, std::size_t(ncols) * n_out * sizeof(float));
    try {
        unexpected_launch = basalt::kernels::native_mmvq_pair(8, wa, wb, xq, ya, yb, n_in, n_out, ncols, stream);
    } catch (...) { unexpected_launch = true; }
    std::vector<uint32_t> untouched(std::size_t(ncols) * n_out * 2);
    ck(cudaMemcpy(untouched.data(), ya, std::size_t(ncols) * n_out * sizeof(float), cudaMemcpyDeviceToHost),
       "pair control read A");
    ck(cudaMemcpy(untouched.data() + std::size_t(ncols) * n_out, yb,
                  std::size_t(ncols) * n_out * sizeof(float), cudaMemcpyDeviceToHost), "pair control read B");
    const bool untouched_ok = std::all_of(untouched.begin(), untouched.end(), [](uint32_t bits) {
        return bits == 0xa5a5a5a5u;
    });
    failures += unexpected_launch || !untouched_ok;
    std::printf("native_mmvq_pair negative control: %s\n", (!unexpected_launch && untouched_ok) ? "PASS" : "FAIL");
    cudaFree(wa); cudaFree(wb); cudaFree(xq); cudaFree(ya); cudaFree(yb);
    basalt::kernels::native_mmvq_set_multi_exact(true);
    return failures;
}

// ---- the tensor-core path
struct MmaCase {
    int n_in, n_out;
};
// row tails (n_out % 16, % 32), 1 / 10 / 16 / 24 / 48 blocks per row, and n_out large enough for 32-row blocks
const MmaCase MMA_CASES[] = {
    {256, 40}, {2560, 1000}, {4096, 520}, {6144, 2560}, {12288, 300}, {512, 8200}, {2560, 10240},
};
constexpr int IQ4XS = 23, IQ4XS_BYTES = 136;

struct Buffers {
    uint8_t* w = nullptr;   // the weights at w + w_off
    uint8_t* xq = nullptr;  // the Q8_1 bytes at xq + x_off
    float* x = nullptr;
    float* y_mma = nullptr;
    float* y_old = nullptr;
    float* y_one = nullptr;
    Buffers() = default;
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;
    ~Buffers() {
        cudaFree(w); cudaFree(xq); cudaFree(x); cudaFree(y_mma); cudaFree(y_old); cudaFree(y_one);
    }
};

// random IQ4_XS weights with sane fp16 scales and normal activations, quantized once; offsets in bytes (multiples of 4)
bool fill(Buffers& b, int n_in, int n_out, int T, int w_off, int x_off, unsigned seed, cudaStream_t s) {
    const std::size_t wbytes = basalt::kernels::native_mmvq_weight_bytes(IQ4XS, n_in, n_out);
    const std::size_t qall = basalt::kernels::native_q8_1_bytes(n_in, T);
    const std::size_t ybytes = (std::size_t) T * n_out * 4;
    std::mt19937 rng(seed);
    std::vector<uint8_t> w(wbytes);
    for (auto& v : w) v = (uint8_t) (rng() & 0xff);
    for (std::size_t k = 0; k < wbytes / IQ4XS_BYTES; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(&w[k * IQ4XS_BYTES], &h, 2);
    }
    std::vector<float> x((std::size_t) T * n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    if (!ck(cudaMalloc(&b.w, wbytes + 16), "malloc w") || !ck(cudaMalloc(&b.xq, qall + 16), "malloc xq") ||
        !ck(cudaMalloc(&b.x, x.size() * 4), "malloc x") || !ck(cudaMalloc(&b.y_mma, ybytes), "malloc y") ||
        !ck(cudaMalloc(&b.y_old, ybytes), "malloc y") || !ck(cudaMalloc(&b.y_one, ybytes), "malloc y"))
        return false;
    if (!ck(cudaMemcpy(b.w + w_off, w.data(), wbytes, cudaMemcpyHostToDevice), "copy w") ||
        !ck(cudaMemcpy(b.x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x"))
        return false;
    // the outputs start as three different byte patterns: an output a path does not write cannot compare equal
    if (!ck(cudaMemset(b.y_mma, 0xff, ybytes), "memset") || !ck(cudaMemset(b.y_old, 0xfe, ybytes), "memset") ||
        !ck(cudaMemset(b.y_one, 0xfd, ybytes), "memset"))
        return false;
    basalt::kernels::quantize_q8_1_rows(b.x, T, n_in, b.xq + x_off, s);
    return ck(cudaStreamSynchronize(s), "quantize");
}

long long count_diff(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, long long& nonfinite) {
    long long diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float fa, fb;
        std::memcpy(&fa, &a[i], 4);
        std::memcpy(&fb, &b[i], 4);
        if (!std::isfinite(fa) || !std::isfinite(fb)) ++nonfinite;
        diff += a[i] != b[i];
    }
    return diff;
}

// One case: the tensor-core call against the dp4a multi-column call and T single-column calls.  False on a
// difference, a non-finite output, a call that did not take the tensor-core kernel, or an error.
bool compare_mma(const MmaCase& c, int T, int w_off, int x_off, cudaStream_t s) {
    namespace k = basalt::kernels;
    Buffers b;
    if (!fill(b, c.n_in, c.n_out, T, w_off, x_off, 777u + (unsigned) c.n_in * 3u + (unsigned) c.n_out + (unsigned) T,
              s))
        return false;
    const uint8_t* w = b.w + w_off;
    const uint8_t* xq = b.xq + x_off;
    const std::size_t qcol = k::native_q8_1_bytes(c.n_in, 1);
    long launches = 0;
    try {
        k::native_mmvq_set_mma(true);
        const long before = k::native_mmvq_mma_launches();
        k::native_mmvq(IQ4XS, w, xq, b.y_mma, c.n_in, c.n_out, T, s);
        launches = k::native_mmvq_mma_launches() - before;
        k::native_mmvq_set_mma(false);
        k::native_mmvq(IQ4XS, w, xq, b.y_old, c.n_in, c.n_out, T, s);
        for (int j = 0; j < T; ++j)
            k::native_mmvq(IQ4XS, w, xq + (std::size_t) j * qcol, b.y_one + (std::size_t) j * c.n_out, c.n_in,
                           c.n_out, 1, s);
    } catch (const std::exception& e) {
        std::printf("IQ4_XS %5d x %5d T=%d: %s\n", c.n_in, c.n_out, T, e.what());
        return false;
    }
    if (!ck(cudaStreamSynchronize(s), "mmvq")) return false;
    const std::size_t n = (std::size_t) T * c.n_out;
    std::vector<uint32_t> ym(n), yo(n), y1(n);
    if (!ck(cudaMemcpy(ym.data(), b.y_mma, n * 4, cudaMemcpyDeviceToHost), "read") ||
        !ck(cudaMemcpy(yo.data(), b.y_old, n * 4, cudaMemcpyDeviceToHost), "read") ||
        !ck(cudaMemcpy(y1.data(), b.y_one, n * 4, cudaMemcpyDeviceToHost), "read"))
        return false;
    long long nonfinite = 0;
    const long long d_old = count_diff(ym, yo, nonfinite);
    const long long d_one = count_diff(ym, y1, nonfinite);
    const bool ok = launches == 1 && d_old == 0 && d_one == 0 && nonfinite == 0;
    std::printf("IQ4_XS %5d x %5d T=%d%s: %7zu outputs, %lld differ from the dp4a multi-column call, %lld from the "
                "single-column calls, %lld non-finite%s  %s\n", c.n_in, c.n_out, T,
                w_off || x_off ? " (off-aligned)" : "", n, d_old, d_one, nonfinite,
                launches == 1 ? "" : ", the tensor-core kernel did not run", ok ? "ok" : "FAIL");
    return ok;
}

int mma_section(cudaStream_t s) {
    namespace k = basalt::kernels;
    std::printf("\n=== tensor-core IQ4_XS (MMVQ_MMA) ===\n");
    if (!k::native_mmvq_mma_available()) {
        std::printf("not available on this device or build (needs int8 tensor cores): skipped\n");
        return 0;
    }
    const bool env = k::native_mmvq_mma();
    int bad = 0;
    for (const MmaCase& c : MMA_CASES) {
        for (int T = 2; T <= 8; ++T) bad += !compare_mma(c, T, 0, 0, s);
        for (int T : {3, 8}) bad += !compare_mma(c, T, 4, 4, s);   // the 4-byte copies
    }
    k::native_mmvq_set_mma(env);
    return bad;
}

void bench(cudaStream_t s) {
    namespace k = basalt::kernels;
    if (!k::native_mmvq_mma_available()) {
        std::printf("\n--bench: the tensor-core kernel is not available here\n");
        return;
    }
    const bool env = k::native_mmvq_mma();
    const MmaCase shapes[] = {{2560, 10240}, {6144, 2560}, {2560, 12288}};
    std::printf("\n=== --bench: IQ4_XS us per call, dp4a exact / tensor-core ===\n");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (const MmaCase& c : shapes) {
        std::printf("%5d -> %5d:", c.n_in, c.n_out);
        for (int T = 2; T <= 8; ++T) {
            Buffers b;
            if (!fill(b, c.n_in, c.n_out, T, 0, 0, 99u, s)) break;
            float us[2] = {0, 0};
            for (int mma = 0; mma < 2; ++mma) {
                k::native_mmvq_set_mma(mma == 1);
                constexpr int N = 200;
                for (int r = 0; r < 20; ++r) k::native_mmvq(IQ4XS, b.w, b.xq, b.y_old, c.n_in, c.n_out, T, s);
                cudaEventRecord(e0, s);
                for (int r = 0; r < N; ++r) k::native_mmvq(IQ4XS, b.w, b.xq, b.y_old, c.n_in, c.n_out, T, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                us[mma] = 1000.0f * ms / N;
            }
            std::printf("  T%d %.1f/%.1f", T, us[0], us[1]);
        }
        std::printf("\n");
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    k::native_mmvq_set_mma(env);
}


// ---- the one-column Q8_0 kernel at the draft chain's projection and head shapes, and short rows (the small-K
// layout): bitwise equal to column 0 of a two-column call and to the cache-streaming variant
// (native_q8_0_mmvq_stream), and within float rounding of a float64 reference on the same bytes
struct Q80Geom {
    int n_in, n_out;
};
const Q80Geom Q80_GEOMS[] = {{2560, 12288}, {2560, 40525}, {6144, 2560}, {2560, 512}, {2560, 2048}, {512, 640},
                            {992, 7}, {1024, 130}, {2560, 1}};

double half_to_double(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const double v = e == 0 ? std::ldexp((double) m, -24) : std::ldexp((double) (m + 1024), e - 25);
    return (h >> 15) ? -v : v;
}

bool compare_q80(const Q80Geom& g, cudaStream_t s) {
    namespace k = basalt::kernels;
    constexpr int Q8_0 = 8, BYTES = 34;
    const std::size_t wbytes = k::native_mmvq_weight_bytes(Q8_0, g.n_in, g.n_out);
    const std::size_t qcol = k::native_q8_1_bytes(g.n_in, 1);
    std::mt19937 rng(4242u + (unsigned) g.n_in + (unsigned) g.n_out);
    std::vector<uint8_t> w(wbytes);
    for (auto& v : w) v = (uint8_t) (rng() & 0xff);
    for (std::size_t b = 0; b < wbytes / BYTES; ++b) {
        const uint16_t h = sane_half(rng);
        std::memcpy(&w[b * BYTES], &h, 2);
    }
    std::vector<float> x((std::size_t) 2 * g.n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < g.n_in; ++i) x[i] = x[g.n_in + i] = nd(rng);
    uint8_t *dw = nullptr, *dq = nullptr;
    float *dx = nullptr, *y1 = nullptr, *y2 = nullptr, *y3 = nullptr;
    if (!ck(cudaMalloc(&dw, wbytes), "malloc") || !ck(cudaMalloc(&dq, 2 * qcol), "malloc") ||
        !ck(cudaMalloc(&dx, x.size() * 4), "malloc") || !ck(cudaMalloc(&y1, (std::size_t) g.n_out * 4), "malloc") ||
        !ck(cudaMalloc(&y2, (std::size_t) 2 * g.n_out * 4), "malloc") || !ck(cudaMalloc(&y3, (std::size_t) g.n_out * 4), "malloc"))
        return false;
    cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(y1, 0xff, (std::size_t) g.n_out * 4);
    cudaMemset(y2, 0xfe, (std::size_t) 2 * g.n_out * 4);
    cudaMemset(y3, 0xfd, (std::size_t) g.n_out * 4);
    k::quantize_q8_1_rows(dx, 2, g.n_in, dq, s);
    k::native_mmvq(Q8_0, dw, dq, y1, g.n_in, g.n_out, 1, s);
    k::native_mmvq(Q8_0, dw, dq, y2, g.n_in, g.n_out, 2, s);
    k::native_q8_0_mmvq_stream(dw, dq, y3, g.n_in, g.n_out, s);
    std::vector<uint8_t> xq(2 * qcol);
    std::vector<float> a(g.n_out), b(2 * (std::size_t) g.n_out), c(g.n_out);
    const bool ok_copy = ck(cudaStreamSynchronize(s), "mmvq") &&
                         ck(cudaMemcpy(xq.data(), dq, 2 * qcol, cudaMemcpyDeviceToHost), "read") &&
                         ck(cudaMemcpy(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost), "read") &&
                         ck(cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost), "read") &&
                         ck(cudaMemcpy(c.data(), y3, c.size() * 4, cudaMemcpyDeviceToHost), "read");
    cudaFree(dw); cudaFree(dq); cudaFree(dx); cudaFree(y1); cudaFree(y2); cudaFree(y3);
    if (!ok_copy) return false;
    long long diff = 0, diff_stream = 0;
    double worst = 0.0, scale = 0.0;
    const int nb = g.n_in / 32;
    for (int r = 0; r < g.n_out; ++r) {
        double ref = 0.0;
        for (int blk = 0; blk < nb; ++blk) {
            const uint8_t* wb = &w[((std::size_t) r * nb + blk) * BYTES];
            const uint8_t* xb = &xq[(std::size_t) blk * 36];
            uint16_t hw, hx;
            std::memcpy(&hw, wb, 2);
            std::memcpy(&hx, xb, 2);
            long long dot = 0;
            for (int i = 0; i < 32; ++i) dot += (long long) (int8_t) wb[2 + i] * (int8_t) xb[4 + i];
            ref += half_to_double(hw) * half_to_double(hx) * (double) dot;
        }
        uint32_t ua, ub, uc;
        std::memcpy(&ua, &a[r], 4);
        std::memcpy(&ub, &b[r], 4);
        std::memcpy(&uc, &c[r], 4);
        diff += ua != ub;
        diff_stream += ua != uc;
        worst = std::max(worst, std::fabs((double) a[r] - ref));
        scale = std::max(scale, std::fabs(ref));
    }
    const bool ok = diff == 0 && diff_stream == 0 && worst <= 1e-4 * (scale + 1e-30);
    std::printf("Q8_0 T=1 %5d x %5d: %lld differ from the two-column call, %lld from the streamed variant, max abs err "
                "%.3e of %.3e  %s\n", g.n_in, g.n_out, diff, diff_stream, worst, scale, ok ? "ok" : "FAIL");
    return ok;
}

int q80_section(cudaStream_t s) {
    std::printf("\n=== Q8_0 one column ===\n");
    int bad = 0;
    for (const Q80Geom& g : Q80_GEOMS) bad += !compare_q80(g, s);
    return bad;
}

// The one-column Q8_0 kernel's time per call and its weight bytes' rate, at the shapes above (warm L2 excluded by
// cycling through enough distinct weights).
void bench_q80(cudaStream_t s) {
    namespace k = basalt::kernels;
    std::printf("\n=== --bench: Q8_0 one column, us per call (weights cycled past L2) ===\n");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (const Q80Geom& g : Q80_GEOMS) {
        const std::size_t wbytes = k::native_mmvq_weight_bytes(8, g.n_in, g.n_out);
        const std::size_t stride = (wbytes + 255) / 256 * 256;   // every copy aligned
        const int copies = std::max<int>(2, (int) ((256u << 20) / stride));
        uint8_t* dw = nullptr;
        uint8_t* dq = nullptr;
        float *dx = nullptr, *dy = nullptr;
        if (!ck(cudaMalloc(&dw, stride * copies), "malloc") || !ck(cudaMalloc(&dq, k::native_q8_1_bytes(g.n_in, 1)), "malloc") ||
            !ck(cudaMalloc(&dx, (std::size_t) g.n_in * 4), "malloc") || !ck(cudaMalloc(&dy, (std::size_t) g.n_out * 4), "malloc"))
            return;
        cudaMemset(dw, 0x11, stride * copies);
        cudaMemset(dx, 0, (std::size_t) g.n_in * 4);
        k::quantize_q8_1_rows(dx, 1, g.n_in, dq, s);
        for (int r = 0; r < copies; ++r) k::native_mmvq(8, dw + stride * r, dq, dy, g.n_in, g.n_out, 1, s);
        const int N = copies * 4;
        double us[2] = {0, 0};
        for (int v = 0; v < 2; ++v) {   // plain, streamed
            cudaEventRecord(e0, s);
            for (int r = 0; r < N; ++r) {
                if (v == 0) k::native_mmvq(8, dw + stride * (r % copies), dq, dy, g.n_in, g.n_out, 1, s);
                else k::native_q8_0_mmvq_stream(dw + stride * (r % copies), dq, dy, g.n_in, g.n_out, s);
            }
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            us[v] = 1000.0 * ms / N;
        }
        std::printf("%5d -> %5d: %7.2f us  %6.0f GB/s, streamed %7.2f us (%.1f MB)\n", g.n_in, g.n_out, us[0],
                    wbytes / us[0] / 1e3, us[1], wbytes / 1e6);
        cudaFree(dw); cudaFree(dq); cudaFree(dx); cudaFree(dy);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

void bench_pair(cudaStream_t stream, int power_limit_w) {
    namespace k = basalt::kernels;
    constexpr int n_in = 2560, n_out = 640, ncols = 8, repeats = 100, rounds = 8;
    const std::size_t qbytes = k::native_q8_1_bytes(n_in, ncols);
    std::vector<float> input(std::size_t(n_in) * ncols, 0.25f);
    void *xq = nullptr, *dx = nullptr;
    if (!ck(cudaMalloc(&xq, qbytes), "pair bench Q8_1") ||
        !ck(cudaMalloc(&dx, input.size() * sizeof(float)), "pair bench input")) {
        cudaFree(xq); cudaFree(dx); return;
    }
    if (!ck(cudaMemcpy(dx, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
            "pair bench copy input")) { cudaFree(xq); cudaFree(dx); return; }
    k::native_quantize_q8_1((const float*)dx, xq, n_in, ncols, stream);
    cudaStreamSynchronize(stream);
    cudaEvent_t begin, end;
    if (!ck(cudaEventCreate(&begin), "pair bench event begin") || !ck(cudaEventCreate(&end), "pair bench event end")) {
        cudaFree(xq); cudaFree(dx); return;
    }
    for (int type : {8, 14}) {
        const std::size_t weight_bytes = k::native_mmvq_weight_bytes(type, n_in, n_out);
        std::vector<uint8_t> weights(weight_bytes, 0x5a);
        void *wa = nullptr, *wb = nullptr;
        float *ya = nullptr, *yb = nullptr;
        const bool allocated = ck(cudaMalloc(&wa, weight_bytes), "pair bench weights A") &&
                               ck(cudaMalloc(&wb, weight_bytes), "pair bench weights B") &&
                               ck(cudaMalloc((void**)&ya, std::size_t(ncols) * n_out * sizeof(float)), "pair bench output A") &&
                               ck(cudaMalloc((void**)&yb, std::size_t(ncols) * n_out * sizeof(float)), "pair bench output B");
        if (!allocated) { cudaFree(wa); cudaFree(wb); cudaFree(ya); cudaFree(yb); continue; }
        const bool copied = ck(cudaMemcpy(wa, weights.data(), weight_bytes, cudaMemcpyHostToDevice), "pair bench copy A") &&
                            ck(cudaMemcpy(wb, weights.data(), weight_bytes, cudaMemcpyHostToDevice), "pair bench copy B");
        if (!copied) { cudaFree(wa); cudaFree(wb); cudaFree(ya); cudaFree(yb); continue; }
        for (int i = 0; i < 20; ++i) {
            k::native_mmvq(type, wa, xq, ya, n_in, n_out, ncols, stream);
            k::native_mmvq(type, wb, xq, yb, n_in, n_out, ncols, stream);
            k::native_mmvq_pair(type, wa, wb, xq, ya, yb, n_in, n_out, ncols, stream);
        }
        double separate_us = 0.0, pair_us = 0.0;
        for (int order = 0; order < rounds; ++order) {
            for (int variant = 0; variant < 2; ++variant) {
                const bool paired = ((order + variant) & 1) != 0;
                cudaEventRecord(begin, stream);
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    if (paired) k::native_mmvq_pair(type, wa, wb, xq, ya, yb, n_in, n_out, ncols, stream);
                    else {
                        k::native_mmvq(type, wa, xq, ya, n_in, n_out, ncols, stream);
                        k::native_mmvq(type, wb, xq, yb, n_in, n_out, ncols, stream);
                    }
                }
                cudaEventRecord(end, stream);
                cudaEventSynchronize(end);
                float milliseconds = 0.0f;
                cudaEventElapsedTime(&milliseconds, begin, end);
                (paired ? pair_us : separate_us) += 1000.0 * milliseconds / repeats;
            }
        }
        separate_us /= rounds;
        pair_us /= rounds;
        const double bytes_gb = 2.0 * weight_bytes;
        std::printf("native_mmvq_pair --bench: %s N=%d M=%d T=%d power_limit_w=%d separate=%.2f us (%.1f GB/s) "
                    "pair=%.2f us (%.1f GB/s)\n",
                    type == 8 ? "Q8_0" : "Q6_K", n_in, n_out, ncols, power_limit_w, separate_us,
                    bytes_gb / separate_us / 1e3, pair_us, bytes_gb / pair_us / 1e3);
        cudaFree(wa); cudaFree(wb); cudaFree(ya); cudaFree(yb);
    }
    cudaEventDestroy(begin); cudaEventDestroy(end); cudaFree(xq); cudaFree(dx);
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    bool do_pair_bench = false;
    int power_limit_w = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        do_bench |= arg == "--bench";
        do_pair_bench |= arg == "--pair-bench";
        if (arg == "--power-limit-w" && i + 1 < argc) power_limit_w = std::atoi(argv[++i]);
    }
    if (do_pair_bench && power_limit_w <= 0) {
        std::printf("--pair-bench requires --power-limit-w W (the externally enforced GPU limit)\n");
        return 2;
    }
    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream create")) return 1;

    Totals on, off;
    const int bad = sweep(true, s, on);
    const int bad_off = sweep(false, s, off);
    basalt::kernels::native_mmvq_set_multi_exact(true);   // back to the default
    const int bad_pair = pair_section(s);
    const int bad_mma = mma_section(s);
    const int bad_q80 = q80_section(s);
    if (do_bench) { bench(s); bench_q80(s); }
    if (do_pair_bench) bench_pair(s, power_limit_w);

    std::printf("\nmulti_exact on:   %lld outputs compared, %lld differ bitwise, %lld non-finite\n",
                on.checked, on.diff, on.nonfinite);
    std::printf("multi_exact off:  %lld outputs compared, %lld differ bitwise (%lld on finite outputs, %lld at T > %d)\n",
                off.checked, off.diff, off.diff_finite, off.diff_finite_wide, COINCIDE_MAX_T);

    if (on.nonfinite != 0 || off.nonfinite != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %lld non-finite outputs from finite scales and activations: the "
                    "bit comparison is only meaningful on finite outputs.\n", on.nonfinite + off.nonfinite);
        return 1;
    }
    if (bad_q80 != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %d one-column Q8_0 case(s) differ from the two-column call or the "
                    "float64 reference.\n", bad_q80);
        return 1;
    }
    if (bad_pair != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %d paired Q8_0/Q6_K case(s) differ from the existing kernels or CPU reference.\n",
                    bad_pair);
        return 1;
    }
    if (bad_mma != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %d tensor-core IQ4_XS case(s) differ from the dp4a kernels, have "
                    "non-finite outputs, did not take the tensor-core kernel or did not run.\n", bad_mma);
        return 1;
    }
    if (bad != 0 || bad_off != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %d case(s) with multi_exact on differ from single-column calls "
                    "or did not run, %d case(s) did not run with it off. If they differ, the sentence in "
                    "native_mmvq.hpp does not hold, and neither does the bit-exactness verify.hpp claims.\n",
                    bad, bad_off);
        return 1;
    }
    // THE NEGATIVE CONTROL, asserted where it can see something: past T = 4 the generic layout is a different one,
    // so if it still finds no difference the comparison cannot tell the layouts apart and the pass proves nothing.
    if (off.powerless_cases != 0) {
        std::printf("\nmmvq_multi_parity: FAILED (no power). In %d case(s) no output differs with multi_exact off at "
                    "any T > %d, where the generic layout is not the exact one: for those cases this comparison "
                    "cannot tell the two layouts apart, so their pass above proves nothing.\n", off.powerless_cases,
                    COINCIDE_MAX_T);
        return 1;
    }
    std::printf("\nmmvq_multi_parity: ok. With multi_exact on every column is bitwise equal to a single-column "
                "call; with it off the control finds differences at T > %d in every case, so the comparison has "
                "power.\n",
                COINCIDE_MAX_T);
    return 0;
}
