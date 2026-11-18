// src/program/serve/token_logprobs.hpp - a request's logprobs=K: the LP line that follows each token it is sent.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace basalt::program::serve {

/// The most alternatives `logprobs=K` takes; INFO reports it as logprobs=N.
constexpr int kLogprobsMax = 20;

/// `LP <logprob> [<id>:<logprob> ...]` for `token`, picked from the head row `logits` (n_vocab raw logits: before
/// temperature, penalties, bias and grammar): its log-softmax value, then the `k` likeliest tokens, best first, a tie
/// to the lower id.  The softmax runs in double.
inline std::string lp_line(const float* logits, int64_t n_vocab, int32_t token, int k) {
    thread_local std::vector<int32_t> ord;
    ord.resize((size_t) n_vocab);
    for (int64_t i = 0; i < n_vocab; ++i) ord[(size_t) i] = (int32_t) i;
    const int64_t top = std::min<int64_t>(std::max(k, 1), n_vocab);   // ord[0] is the largest logit
    std::partial_sort(ord.begin(), ord.begin() + top, ord.end(), [logits](int32_t a, int32_t b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    });
    const double mx = logits[ord[0]];
    double sum = 0.0;
    for (int64_t i = 0; i < n_vocab; ++i) sum += std::exp((double) logits[i] - mx);
    const double lse = mx + std::log(sum);
    char buf[48];
    std::snprintf(buf, sizeof buf, "LP %.6f", (double) logits[token] - lse);
    std::string line = buf;
    for (int64_t j = 0; j < std::min<int64_t>(k, n_vocab); ++j) {
        const int32_t id = ord[(size_t) j];
        std::snprintf(buf, sizeof buf, " %d:%.6f", (int) id, (double) logits[id] - lse);
        line += buf;
    }
    return line;
}

}  // namespace basalt::program::serve
