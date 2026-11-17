#pragma once
// CUDA peer-to-peer (NVLink / PCIe P2P) enablement helper for multi-GPU tiers.

#include "basalt/core/diag.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/vmm.hpp"

#include <cuda_runtime.h>

namespace basalt::core {

/// Enable bidirectional P2P access between `dev0` and `dev1` (gated by --exp peer_p2p: 0=off, 1=native atomics, 2=any).
inline bool enable_peer_access(int dev0, int dev1) {
    const double mode = exp_flag("peer_p2p", 1);
    if (mode == 0.0 || dev0 < 0 || dev1 < 0 || dev0 == dev1) return false;
    int c01 = 0, c10 = 0;
    if (cudaDeviceCanAccessPeer(&c01, dev0, dev1) != cudaSuccess ||
        cudaDeviceCanAccessPeer(&c10, dev1, dev0) != cudaSuccess || !c01 || !c10) {
        (void) cudaGetLastError();
        return false;
    }
    // By default require native P2P atomics so PCIe root-complex cards keep host-pinned staging unless forced (mode 2).
    if (mode < 2.0) {
        int a01 = 0, a10 = 0;
        if (cudaDeviceGetP2PAttribute(&a01, cudaDevP2PAttrNativeAtomicSupported, dev0, dev1) != cudaSuccess ||
            cudaDeviceGetP2PAttribute(&a10, cudaDevP2PAttrNativeAtomicSupported, dev1, dev0) != cudaSuccess ||
            !a01 || !a10) {
            (void) cudaGetLastError();
            return false;
        }
    }
    auto enable_one = [](int src, int dst) -> bool {
        OnDevice on(src);
        const cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
        if (e == cudaSuccess) return true;
        if (e == cudaErrorPeerAccessAlreadyEnabled) {
            (void) cudaGetLastError();
            return true;
        }
        (void) cudaGetLastError();
        return false;
    };
    if (!enable_one(dev0, dev1)) return false;
    if (!enable_one(dev1, dev0)) {
        OnDevice on(dev0);
        (void) cudaDeviceDisablePeerAccess(dev1);
        (void) cudaGetLastError();
        return false;
    }
    // Virtual memory allocations mapped before or after P2P setup need explicit peer access via cuMemSetAccess.
    vmm::allow_peer(dev0, dev1);
    vmm::allow_peer(dev1, dev0);
    return true;
}

}  // namespace basalt::core
