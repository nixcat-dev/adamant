// Real-artifact BF16 PLE rows: the table's row decode against an independent widening, and the reader's own
// per-row and 16-row paths against the same oracle.  bf16 is the top half of an f32, so this is exact: the
// expected maximum error is zero, not a tolerance.
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/models/qwen4exp/kernels/ngram.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using basalt::kernels::PLE_HEAD_DIM;
using basalt::kernels::PLE_N_HEADS;

// The oracle, written out here rather than borrowed from the engine: assemble the two bytes little-endian, put
// them in the top half of a float, and read it back.  Nothing about the table is assumed.
float oracle(const uint8_t* row, int j) {
    const uint32_t bits = (uint32_t) row[2 * j] | ((uint32_t) row[2 * j + 1] << 8);
    const uint32_t wide = bits << 16;
    float f;
    std::memcpy(&f, &wide, sizeof f);
    return f;
}

double row_error(const uint8_t* row, const float* got) {
    double worst = 0.0;
    for (int j = 0; j < PLE_HEAD_DIM; ++j) worst = std::max(worst, (double) std::fabs(got[j] - oracle(row, j)));
    return worst;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: ple_bf16_parity <gguf-containing-ple>\n");
        return 2;
    }
    basalt::GgufFile gguf(argv[1]);
    const auto* tensor = gguf.find("per_layer_token_embd.weight");
    if (!tensor || std::strcmp(tensor->type_name(), "BF16") != 0 || tensor->shape.size() != 2 ||
        tensor->shape[0] != (uint64_t) PLE_HEAD_DIM) {
        std::fprintf(stderr, "expected a BF16 PLE [%d, N]\n", PLE_HEAD_DIM);
        return 2;
    }
    const size_t row_bytes = basalt::kernels::PLE_ROW_BYTES_BF16;
    basalt::kernels::PleIoOptions options;
    options.mode = basalt::kernels::PleIo::Mmap;
    basalt::kernels::PleTable table;
    std::string err;
    if (!table.open(argv[1], err, options)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (std::strcmp(table.format(), "BF16") != 0) {
        std::fprintf(stderr, "the table reports format %s, not BF16\n", table.format());
        return 1;
    }
    const uint64_t rows = table.rows();
    if (rows != tensor->shape[1]) {
        std::fprintf(stderr, "table rows %llu != the tensor's %llu\n", (unsigned long long) rows,
                     (unsigned long long) tensor->shape[1]);
        return 1;
    }
    const uint8_t* bytes = gguf.tensor_data(*tensor);

    // Real geometry over the whole row range: the first and last rows are the ones an off-by-one loses, and the
    // fixed probes include a head-vocabulary boundary.  The rest are random rows of the real table.
    std::vector<uint32_t> probes = {0, 1, 12345, 20000003, (uint32_t) (rows - 1)};
    std::mt19937 gen(20261105);
    std::uniform_int_distribution<uint32_t> pick(0, (uint32_t) (rows - 1));
    for (int i = 0; i < 11; ++i) probes.push_back(pick(gen));

    double max_abs = 0.0;
    for (uint32_t row : probes) {
        float got[PLE_HEAD_DIM];
        table.read_row(row, got);
        max_abs = std::max(max_abs, row_error(bytes + (size_t) row * row_bytes, got));
    }

    // One token's window through the split issue/collect path the decode uses.
    uint32_t window[PLE_N_HEADS];
    for (int i = 0; i < PLE_N_HEADS; ++i) window[i] = probes[(size_t) i % probes.size()];
    float batch[PLE_N_HEADS * PLE_HEAD_DIM];
    if (!table.issue(window) || !table.collect(batch, err)) {
        std::fprintf(stderr, "collect: %s\n", err.c_str());
        return 1;
    }
    for (int h = 0; h < PLE_N_HEADS; ++h)
        max_abs = std::max(max_abs, row_error(bytes + (size_t) window[h] * row_bytes, batch + h * PLE_HEAD_DIM));

    std::printf("BF16 PLE: %zu rows, %zu B a row, %zu probes, max_abs %.3e %s\n", (size_t) rows, row_bytes,
                probes.size(), max_abs, max_abs == 0.0 ? "PASS" : "FAIL");
    return max_abs == 0.0 ? 0 : 1;
}
