// tests/program/token_logprobs_test.cpp - the LP line of a logprobs request (token_logprobs.hpp) against a reference
// in long double: rows of the real vocabulary size, ties, no alternatives, more alternatives than tokens, and logits far
// from zero.  The line prints six decimals, so every value must be within 1e-6 of the reference.
#include "token_logprobs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace basalt::program::serve;

static int g_fail = 0;
#define CHECK(c)                                                                         \
    do {                                                                                 \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

namespace {

uint64_t g_rng = 0;
uint64_t next_u64() {   // splitmix64: the same sequence on every compiler
    uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
float uniform(float lo, float hi) { return lo + (hi - lo) * (float) ((next_u64() >> 40) * (1.0 / 16777216.0)); }

struct Parsed {
    bool ok = false;
    double lp = 0;
    std::vector<std::pair<int, double>> top;
};

Parsed parse(const std::string& line) {
    Parsed p;
    std::istringstream in(line);
    std::string head, item;
    if (!(in >> head >> p.lp) || head != "LP") return p;
    while (in >> item) {
        const size_t c = item.find(':');
        if (c == std::string::npos) return p;
        p.top.emplace_back(std::atoi(item.c_str()), std::strtod(item.c_str() + c + 1, nullptr));
    }
    p.ok = true;
    return p;
}

// the line `lp_line` should print, in long double: the token's log-softmax value and the k best ids, a tie to the
// lower id
void check_row(const std::vector<float>& lg, int32_t token, int k) {
    const int64_t V = (int64_t) lg.size();
    long double mx = lg[0];
    for (float v : lg) mx = std::max(mx, (long double) v);
    long double sum = 0;
    for (float v : lg) sum += std::exp((long double) v - mx);
    const long double lse = mx + std::log(sum);
    std::vector<int32_t> ord((size_t) V);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int32_t a, int32_t b) { return lg[(size_t) a] > lg[(size_t) b]; });
    const Parsed p = parse(lp_line(lg.data(), V, token, k));
    CHECK(p.ok);
    CHECK(std::fabs(p.lp - (double) ((long double) lg[(size_t) token] - lse)) <= 1e-6);
    CHECK((int64_t) p.top.size() == std::min<int64_t>(k, V));
    for (size_t j = 0; j < p.top.size() && j < ord.size(); ++j) {
        CHECK(p.top[j].first == ord[j]);
        CHECK(std::fabs(p.top[j].second - (double) ((long double) lg[(size_t) ord[j]] - lse)) <= 1e-6);
    }
}

void fixed_cases() {
    const std::vector<float> ties = {1, 3, 3, 2, 3, 0, -1, 3};
    check_row(ties, 2, 5);   // the four 3s by id, then the 2
    const Parsed t = parse(lp_line(ties.data(), 8, 2, 5));
    CHECK(t.ok && t.top.size() == 5 && t.top[0].first == 1 && t.top[1].first == 2 && t.top[2].first == 4 &&
          t.top[3].first == 7 && t.top[4].first == 3);
    check_row(ties, 6, 0);   // no alternatives: the token's own value only
    CHECK(lp_line(ties.data(), 8, 6, 0).find(':') == std::string::npos);
    const std::vector<float> few = {0.5f, -2.0f, 4.0f, 4.0f, 1.0f};
    check_row(few, 0, kLogprobsMax);   // more alternatives asked than tokens: every token, once
    const std::vector<float> one = {7.0f};
    check_row(one, 0, 3);   // a single token has log-probability 0
    CHECK(parse(lp_line(one.data(), 1, 0, 3)).lp == 0.0);
    std::vector<float> far(1000);
    for (float& v : far) v = uniform(990.0f, 1000.0f);   // exp(v) alone would overflow
    check_row(far, 17, 10);
    for (float& v : far) v = uniform(-1000.0f, -990.0f);   // ... or underflow
    check_row(far, 17, 10);
}

void vocabulary_rows() {
    std::vector<float> lg(248320);
    for (int row = 0; row < 4; ++row) {
        for (float& v : lg) v = uniform(-12.0f, 12.0f);
        for (int s = 0; s < 5; ++s) lg[next_u64() % lg.size()] += uniform(5.0f, 30.0f);   // a few clear leaders
        const int32_t best = (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        check_row(lg, best, kLogprobsMax);
        check_row(lg, (int32_t) (next_u64() % lg.size()), kLogprobsMax);   // a sampled token from the tail
        check_row(lg, best, 0);
        // a greedy pick is its row's top alternative: the same id and the same printed value
        const std::string line = lp_line(lg.data(), (int64_t) lg.size(), best, 1);
        const size_t c = line.find(':');
        CHECK(c != std::string::npos && line.substr(3, line.find(' ', 3) - 3) == line.substr(c + 1) &&
              std::atoi(line.c_str() + line.find(' ', 3) + 1) == best);
    }
}

}  // namespace

int main() {
    fixed_cases();
    vocabulary_rows();
    if (g_fail != 0) {
        std::printf("token_logprobs_test: %d failures\n", g_fail);
        return 1;
    }
    std::printf("token_logprobs_test: ok\n");
    return 0;
}
