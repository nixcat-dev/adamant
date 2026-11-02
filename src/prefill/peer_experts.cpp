#include "experts.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/moe_fused.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/log.hpp"
#include <algorithm>

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::Experts::Group::peer_experts() {
    peer_now = use_mmq && !order_peer.empty();
    if (peer_now) {
        PeerPrefill& P = *ppc;
        cudaEventRecord(P.ev_in, m.cs);
        int prevd = 0;
        cudaGetDevice(&prevd);
        cudaSetDevice(P.dev);
        const cudaStream_t ps = P.s;
        cudaStreamWaitEvent(ps, P.ev_in, 0);
        if (P.out_pending) { cudaStreamWaitEvent(ps, P.ev_done, 0); P.out_pending = false; }
        cudaMemcpyPeerAsync(P.mixed, P.dev, m.mixed, prevd, (size_t) T * N * 4, ps);
        cudaMemcpyAsync(P.src, src_h + rows_local, (size_t) rows_peer * 4, cudaMemcpyHostToDevice, ps);
        const size_t n = order_peer.size();
        {
            // groups of up to MMQ_GROUP experts and at most G rows, each computed from its own rows
            P.groups.clear();
            {
                size_t g0 = 0;
                int64_t gr = 0;
                for (size_t j = 0; j < n; ++j) {
                    const int64_t c = m.cnt[(size_t) order_peer[j]];
                    if (j > g0 && (j - g0 == (size_t) MMQ_GROUP || gr + c > P.G)) { P.groups.push_back({g0, j}); g0 = j; gr = 0; }
                    gr += c;
                }
                if (n > g0) P.groups.push_back({g0, n});
            }
            const size_t ng = P.groups.size();
            P.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
            for (size_t j = 0; j < n; ++j) P.bounds_host[j] = m.off[(size_t) order_peer[j]] - (int32_t) rows_local;
            P.bounds_host[n] = (int32_t) rows_peer;
            for (size_t g2 = 0; g2 < ng; ++g2)
                for (size_t i = 0; i <= MMQ_GROUP; ++i)
                    P.bounds_host[n + 1 + g2 * (MMQ_GROUP + 1) + i] =
                        P.bounds_host[std::min(P.groups[g2].second, P.groups[g2].first + i)] -
                        P.bounds_host[P.groups[g2].first];
            cudaMemcpyAsync(P.bounds, P.bounds_host.data(), P.bounds_host.size() * 4, cudaMemcpyHostToDevice, ps);
            const auto& f = lay.fmt[(size_t) l];
            for (size_t g2 = 0; g2 < ng; ++g2) {
                const size_t j0 = P.groups[g2].first, j1 = P.groups[g2].second;
                const int ngx = (int) (j1 - j0);
                int64_t maxr = 0;
                for (size_t j = j0; j < j1; ++j) {
                    const int32_t e = order_peer[j];
                    maxr = std::max<int64_t>(maxr, m.cnt[(size_t) e]);
                    const uint8_t* bd = P.peer->slot_ptr(l, e);
                    const size_t q = j - j0;
                    if (lay.native)
                        mmq::gather_native(bd, bd + f.up_off, mmq_gub / 2, bd + f.down_off, mmq_db,
                                           P.grp_gu + q * mmq_gub, P.grp_d + q * mmq_db, ps,
                                           f.tail ? bd + f.bytes - f.tail : nullptr,
                                           grp_scales(P.grp_d) + 4 * q);
                    else
                        mmq::gather_q2(bd, P.grp_gu + q * mmq_gub, P.grp_d + q * mmq_db, ps);
                }
                const int64_t r0 = P.bounds_host[j0], nr = P.bounds_host[j1] - r0;
                if (nr <= 0) continue;
                const int32_t* rel = P.bounds + n + 1 + g2 * (MMQ_GROUP + 1);
                cudaMemsetAsync(P.grp_gu + (size_t) ngx * mmq_gub, 0, MMQ_TAIL, ps);
                cudaMemsetAsync(P.grp_d + (size_t) ngx * mmq_db, 0, MMQ_TAIL, ps);
                mmq::quantize(P.mixed, P.src + r0, P.Xq_g, mmq_gt, N, N, nr, ps);
                mmq::Product gu;
                gu.w = P.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                gu.n = ngx; gu.xq = P.Xq_g; gu.bounds = rel; gu.ids = P.ident;
                gu.total_rows = nr; gu.max_rows = maxr; gu.dst = P.GU_g; gu.ld_dst = 1280;
                P.run_ctx->run(gu, ps);
                mmq::swiglu(P.GU_g, P.H_g, nr, 640, !lay.native, ps,
                            lay.native && f.tail ? grp_scales(P.grp_d) : nullptr, rel, ngx);
                mmq::quantize(P.H_g, nullptr, P.Hq_g, mmq_dt, 640, 640, nr, ps);
                const int b = (int) (g2 & 1);
                if (P.dm_live[b]) cudaStreamWaitEvent(ps, P.ev_dm[b], 0);   // its last rows have left
                mmq::Product dn;
                dn.w = P.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                dn.n = ngx; dn.xq = P.Hq_g; dn.bounds = rel;
                dn.ids = P.ident; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = P.Dm_b[b];
                dn.ld_dst = N;
                P.run_ctx->run(dn, ps);
                cudaEventRecord(P.ev_grp[g2 % PeerPrefill::kGrpEv], ps);
                cudaStreamWaitEvent(P.s_out, P.ev_grp[g2 % PeerPrefill::kGrpEv], 0);
                cudaMemcpyPeerAsync(m.Dm + (size_t) (rows_local + r0) * N, prevd, P.Dm_b[b], P.dev,
                                    (size_t) nr * N * 4, P.s_out);
                cudaEventRecord(P.ev_dm[b], P.s_out);
                P.dm_live[b] = true;
            }
        }
        cudaEventRecord(P.ev_done, P.s_out);
        P.out_pending = true;
        cudaSetDevice(prevd);
    }
    return true;
}

}  // namespace basalt::prefill
