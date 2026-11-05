// Native GR RMSNorm prefetch parity and microbenchmark.

#include "basalt/kernels/native_gr_norm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-5f;

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        std::exit(2);
    }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}

// Frozen prefetch-free kernel used as the bitwise baseline.
template<int BlockSize>
__global__ void baseline_norm(const float* __restrict__ input, const float* __restrict__ gamma,
                              float* __restrict__ output,
                              int n_cols, float epsilon) {
    const int tid = threadIdx.x;
    const std::size_t offset = std::size_t(blockIdx.x) * n_cols;
    input += offset;
    gamma += offset;
    output += offset;
    float partial = 0.0f;
    for (int col = tid; col < n_cols; col += BlockSize) {
        const float value = input[col];
        partial += value * value;
    }
    __shared__ float sums[32];
    partial = warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = 0.0f;
    if (lane < BlockSize / 32) partial = sums[lane];
    partial = warp_sum(partial);
    const float scale = rsqrtf(partial / n_cols + epsilon);
    for (int col = tid; col < n_cols; col += BlockSize)
        output[col] = scale * input[col] * gamma[col];
}

void run_baseline(const float* x, const float* gamma, float* y, int cols, int rows, cudaStream_t stream) {
    if (cols < 1024)
        baseline_norm<256><<<unsigned(rows), 256, 0, stream>>>(x, gamma, y, cols, kEpsilon);
    else
        baseline_norm<1024><<<unsigned(rows), 1024, 0, stream>>>(x, gamma, y, cols, kEpsilon);
    check(cudaGetLastError(), "baseline launch");
}

struct DeviceBuffers {
    float* x = nullptr;
    float* gamma = nullptr;
    float* old_y = nullptr;
    float* new_y = nullptr;
    float* row_y = nullptr;
    ~DeviceBuffers() {
        cudaFree(x);
        cudaFree(gamma);
        cudaFree(old_y);
        cudaFree(new_y);
        cudaFree(row_y);
    }
};

bool run_case(int cols, int rows, int pattern, bool bench, int power_limit_w, cudaStream_t stream) {
    const std::size_t count = std::size_t(cols) * rows;
    std::vector<float> x(count), gamma(count), old_y(count), new_y(count);
    std::mt19937 rng(0x91a2u + cols * 17u + pattern);
    std::uniform_real_distribution<float> values(-0.5f, 0.5f);
    for (std::size_t i = 0; i < count; ++i) {
        x[i] = pattern == 0 ? values(rng) : (pattern == 1 ? 0.0f : ((i & 1u) ? 1.0e-20f : -1.0e-20f));
        gamma[i] = values(rng);
    }

    DeviceBuffers d;
    check(cudaMalloc(&d.x, count * sizeof(float)), "alloc x");
    check(cudaMalloc(&d.gamma, count * sizeof(float)), "alloc gamma");
    check(cudaMalloc(&d.old_y, count * sizeof(float)), "alloc old output");
    check(cudaMalloc(&d.new_y, count * sizeof(float)), "alloc new output");
    check(cudaMalloc(&d.row_y, std::size_t(cols) * sizeof(float)), "alloc row output");
    check(cudaMemcpyAsync(d.x, x.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream), "copy x");
    check(cudaMemcpyAsync(d.gamma, gamma.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream), "copy gamma");
    run_baseline(d.x, d.gamma, d.old_y, cols, rows, stream);
    basalt::kernels::native_gr_rms_norm_weighted(d.x, d.gamma, d.new_y, cols, rows, kEpsilon, stream);
    check(cudaMemcpyAsync(old_y.data(), d.old_y, count * sizeof(float), cudaMemcpyDeviceToHost, stream), "read old output");
    check(cudaMemcpyAsync(new_y.data(), d.new_y, count * sizeof(float), cudaMemcpyDeviceToHost, stream), "read new output");
    check(cudaStreamSynchronize(stream), "parity sync");

    bool ok = true;
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(new_y[i]) || std::memcmp(&old_y[i], &new_y[i], sizeof(float)) != 0) ok = false;
    }

    for (int row = 0; row < rows; ++row) {
        double sum_sq = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double value = x[std::size_t(row) * cols + col];
            sum_sq += value * value;
        }
        const double scale = 1.0 / std::sqrt(sum_sq / cols + double(kEpsilon));
        for (int col = 0; col < cols; ++col) {
            const std::size_t i = std::size_t(row) * cols + col;
            const double reference = scale * double(x[i]) * double(gamma[i]);
            const double error = std::abs(double(new_y[i]) - reference);
            max_abs = std::max(max_abs, error);
            max_rel = std::max(max_rel, error / std::max(std::abs(reference), 1.0e-30));
            if (error > 2.0e-5 * std::max(1.0, std::abs(reference))) ok = false;
        }

        basalt::kernels::native_gr_rms_norm_weighted(d.x + std::size_t(row) * cols,
            d.gamma + std::size_t(row) * cols, d.row_y, cols, 1, kEpsilon, stream);
        std::vector<float> one_row(cols);
        check(cudaMemcpyAsync(one_row.data(), d.row_y, std::size_t(cols) * sizeof(float), cudaMemcpyDeviceToHost, stream),
              "read single row");
        check(cudaStreamSynchronize(stream), "row sync");
        for (int col = 0; col < cols; ++col)
            if (std::memcmp(&one_row[col], &new_y[std::size_t(row) * cols + col], sizeof(float)) != 0) ok = false;
    }
    std::printf("native_gr_norm_parity: cols=%d rows=%d pattern=%d %s cpu_max_abs=%.3e cpu_max_rel=%.3e\n",
                cols, rows, pattern, ok ? "PASS" : "FAIL", max_abs, max_rel);

    if (bench && pattern == 0) {
        constexpr int iterations_per_sample = 10;
        constexpr int samples = 20;
        cudaEvent_t start{}, stop{};
        check(cudaEventCreate(&start), "event start");
        check(cudaEventCreate(&stop), "event stop");
        auto sample = [&](bool baseline) {
            check(cudaEventRecord(start, stream), "record start");
            for (int i = 0; i < iterations_per_sample; ++i) {
                if (baseline) run_baseline(d.x, d.gamma, d.old_y, cols, rows, stream);
                else basalt::kernels::native_gr_rms_norm_weighted(d.x, d.gamma, d.new_y, cols, rows, kEpsilon, stream);
            }
            check(cudaEventRecord(stop, stream), "record stop");
            check(cudaEventSynchronize(stop), "bench sync");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
            return double(ms) * 1000.0 / iterations_per_sample;
        };
        double old_us = 0.0;
        double new_us = 0.0;
        for (int i = 0; i < samples; ++i) {
            if ((i & 1) == 0) {
                old_us += sample(true);
                new_us += sample(false);
            } else {
                new_us += sample(false);
                old_us += sample(true);
            }
        }
        old_us /= samples;
        new_us /= samples;
        const double bytes = double(count) * sizeof(float) * 4.0;
        std::printf("native_gr_norm_bench: cols=%d rows=%d power_limit=%dW baseline=%.3f us %.2f GB/s prefetch=%.3f us %.2f GB/s\n",
                    cols, rows, power_limit_w, old_us, bytes / old_us / 1000.0,
                    new_us, bytes / new_us / 1000.0);
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
    }
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int power_limit_w = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else if (std::strcmp(argv[i], "--power-limit-w") == 0 && i + 1 < argc) power_limit_w = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: native_gr_norm_parity [--bench --power-limit-w W]\n");
            return 2;
        }
    }
    if (bench && power_limit_w <= 0) {
        std::fprintf(stderr, "--bench requires the enforced --power-limit-w value\n");
        return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("native_gr_norm_parity: no CUDA device, skipped\n");
        return 77;
    }
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream), "stream create");
    bool ok = true;
    for (int cols : {1, 31, 255, 256, 257, 1023, 1024, 1025, 2560, 4095, 4096, 4097, 5000, 5120, 5121})
        for (int pattern = 0; pattern < 3; ++pattern)
            ok = run_case(cols, 3, pattern, bench, power_limit_w, stream) && ok;
    check(cudaStreamDestroy(stream), "stream destroy");
    std::printf("native_gr_norm_parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
