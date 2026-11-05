// include/basalt/program/pcie_balance.hpp - the PCIe share, refined from what the CPU pool actually costs.
//
// A decode window's missed experts are split: the GPU reads a share of them over the link and computes them on its
// SMs, the CPU pool computes the rest.  The startup probe prices the link alone, which is only half the question:
// when the pool is fast the GPU's copy adds nothing, and when it is slow the copy is what saves the window.  This
// fits the pool's real cost per expert from the windows that have already run and recomputes the share from it.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace basalt::program {

/// Pool cost factor used to choose the GPU share of missed experts.
inline double pcie_pool_factor(double pool_gbps) {
    if (pool_gbps >= 30.0) return 1.85;
    if (pool_gbps <= 12.0) return 0.8;
    return 0.8 + (pool_gbps - 12.0) * (1.05 / 18.0);
}

/// PCIe share rounded to 0.05 and bounded for the auto-balancer.
inline double pcie_share_for(double link_gbps, double pool_gbps) {
    const double raw = link_gbps / (link_gbps + pcie_pool_factor(pool_gbps) * pool_gbps);
    return std::min(0.95, std::max(0.10, std::round(raw * 20.0) / 20.0));
}

struct PcieBalance {
    bool on = false;
    double link_gbps = 0.0, blob_bytes = 0.0;
    double mark_ms = 0.0;
    int64_t mark_jobs = 0;
    double acc_ms = 0.0;
    int64_t acc_jobs = 0;
    int acc_win = 0, updates = 0;

    void mark(double ms_run, int64_t jobs) {
        mark_ms = ms_run;
        mark_jobs = jobs;
    }

    /// After a decode window: the new share, or -1 while there is nothing to change.  `tc_us`/`rate` get the reading.
    double window(double ms_run, int64_t jobs, double cur, double* tc_us, double* rate) {
        if (!on) return -1.0;
        acc_ms += ms_run - mark_ms;
        acc_jobs += jobs - mark_jobs;
        mark(ms_run, jobs);
        ++acc_win;
        if (acc_win < (updates == 0 ? 48 : 96)) return -1.0;   // the first fit over 48 windows, later ones over 96
        if (acc_jobs < 300) {          // too few missed experts to read the pool's speed (everything hits): look again
            if (acc_win >= 2000) on = false;
            return -1.0;
        }
        const double tc = acc_ms * 1000.0 / (double) acc_jobs;   // microseconds an expert on the pool
        acc_ms = 0.0;
        acc_jobs = 0;
        acc_win = 0;
        if (++updates >= 3) on = false;                          // three fits, then it stops
        if (!(tc > 1.0) || !(blob_bytes > 0.0) || !(link_gbps > 0.0)) return -1.0;
        const double r = blob_bytes / (tc * 1e-6) / 1e9;         // the pool's bytes a second
        const double f = pcie_share_for(link_gbps, r);
        *tc_us = tc;
        *rate = r;
        return std::fabs(f - cur) < 0.025 ? -1.0 : f;
    }
};

}  // namespace basalt::program
