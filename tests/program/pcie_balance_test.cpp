// tests/program/pcie_balance_test.cpp - pool-rate share rule and decode-window fitting.
#include "basalt/program/pcie_balance.hpp"

#include <cmath>
#include <cstdio>

using namespace basalt::program;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

int main() {
    CHECK(near(pcie_pool_factor(0.0), 0.8));
    CHECK(near(pcie_pool_factor(12.0), 0.8));
    CHECK(near(pcie_pool_factor(21.0), 1.325));
    CHECK(near(pcie_pool_factor(30.0), 1.85));
    CHECK(near(pcie_pool_factor(65.0), 1.85));

    CHECK(near(pcie_share_for(56.1, 61.2), 0.35));
    CHECK(near(pcie_share_for(56.3, 10.3), 0.85));
    CHECK(near(pcie_share_for(50.0, 1.0), 0.95));
    CHECK(near(pcie_share_for(1.0, 100.0), 0.10));
    for (double r = 1.0; r <= 100.0; r += 1.0) {
        const double share = pcie_share_for(50.0, r);
        CHECK(share >= 0.10 && share <= 0.95);
        CHECK(near(std::round(share * 20.0) / 20.0, share));
    }

    PcieBalance b;
    b.on = true;
    b.link_gbps = 30.0;
    b.blob_bytes = 1.0e6;
    b.mark(0.0, 0);
    double tc_us = 0.0, rate = 0.0;
    for (int w = 1; w <= 48; ++w) {
        const double share = b.window((double) w, (int64_t) w * 10, 0.6, &tc_us, &rate);
        if (w < 48) CHECK(share < 0.0);
        else CHECK(near(share, 0.8));
    }
    CHECK(b.updates == 1 && b.acc_win == 0);
    CHECK(near(tc_us, 100.0));
    CHECK(near(rate, 10.0));

    PcieBalance off;
    off.mark(0.0, 0);
    CHECK(off.window(1.0, 10, 0.6, &tc_us, &rate) < 0.0);
    CHECK(off.acc_win == 0 && off.acc_jobs == 0);

    std::printf(fails ? "pcie_balance_test: FAILED\n" : "pcie_balance_test: ok\n");
    return fails ? 1 : 0;
}
