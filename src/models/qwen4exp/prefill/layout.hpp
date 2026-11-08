#pragma once

#include "common.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/models/qwen4exp/decode/layout.hpp"
#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/models/qwen4exp/kernels/kv_stream.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q4.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_decode_attn.hpp"
#include <vector>

namespace basalt::prefill::detail {

// Either cudaMalloc (owned, freed with the object) or a bump allocation from a borrowed region; with no base and
// no region it only counts, which is how `bytes_needed` sizes the region.
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    bool count_only = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T* take(size_t n, bool& ok) {
        const uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) { used += bytes; return nullptr; }
        if (base != nullptr) {
            if (used + bytes > cap) { ok = false; return nullptr; }
            T* p = (T*) (base + used);
            used += bytes;
            return p;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
        owned->push_back(p);
        used += bytes;
        return (T*) p;
    }
};

constexpr int64_t GEMM_SCRATCH = 32ll << 20;        // FP16 elements for the largest dequantized dense weight
constexpr size_t GEMM_WS = 32u << 20;               // cuBLAS workspace

// Which layers' experts go through MMQ (both weight types covered; the Q2_0 pack always - its blob
// is converted to GGUF Q2_0 blocks on the gather), whether any layer keeps the FP16 path (IQ1_M), and the largest
// gate/up and down matrices a group buffer slot holds.
constexpr int MMQ_GROUP = 16;                  // experts per MMQ launch (the gather is per expert, as blobs arrive)
static_assert(STAGE >= 2 * MMQ_GROUP, "the staging ring holds two groups");
// MMQ reads up to one 256-value tile past a matrix's last row when the row length is not a multiple of it (the down
// product: 640 values).  Those bytes meet zero activations, which is harmless only if they decode to finite numbers -
// llama.cpp zero-pads after every tensor, and so does a group buffer: this many zeroed bytes follow its last expert.
constexpr size_t MMQ_TAIL = 4096;
// A group buffer's down part ends in MMQ_GROUP 16-byte slots after its zeroed tail: each NVFP4 expert's FP32 scale
// tail {s_gate, s_up, s_down, 0}, gathered with it and applied in mmq::swiglu.
constexpr size_t MMQ_SCALES = (size_t) MMQ_GROUP * 16;
struct MmqPlan {
    bool any = false, fallback = true;
    std::vector<char> layer;                   // per layer: MMQ
    size_t gu_max = 0, d_max = 0;
};

// A peer card slower than the main one takes the rows of routed-only chunks up to this size, no more.
constexpr int64_t kPeerSlowRows = 2559;
constexpr size_t al256(size_t v) { return (v + 255) & ~(size_t) 255; }

const MmqPlan& mmq_plan();
float* grp_scales(uint8_t* grp_d);
void take_stage(Alloc& o, const core::SessionState& ss, const basalt::kernels::QsaShapes& s,
                basalt::kernels::KvHostPools& st, bool& ok);
basalt::kernels::QsaAttnPools pools_of(const basalt::kernels::KvHostPools& h, const int32_t* table);
uint64_t gdn_qkv_bytes(size_t T);
uint64_t gdn_set_bytes(size_t T);
uint64_t hc_set_bytes(size_t T);
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const basalt::kernels::QsaShapes& s);
uint64_t moe_set_bytes(size_t T, int64_t n_expert, bool fused);
bool fused_layout(size_t T, bool src);

}  // namespace basalt::prefill::detail
