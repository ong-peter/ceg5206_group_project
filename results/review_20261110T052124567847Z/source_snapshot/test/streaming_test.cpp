#include "attention.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool success, const char* message) {
    if (!success) throw std::runtime_error(message);
}

template<class Exception, class Function>
void rejects(Function call) {
    bool rejected = false;
    try { call(); } catch (const Exception&) { rejected = true; }
    require(rejected, "Invalid configuration was not rejected");
}

void contract(std::size_t n, std::size_t d, AttentionConfig config) {
    const std::size_t count = attention_streaming_scratch(n, d, config);
    const std::size_t r = std::min(n, config.br), c = std::min(n, config.bc);
    const std::size_t workers = std::min(config.threads, 1 + (n - 1) / r);
    require(count == workers * (r * c + r * d + 2 * r + c * d + (workers > 1 ? 32 : 0)),
            "Scratch accounting does not match the documented allocation");
    constexpr float guard = 12345.0f;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    // The exact-size allocation also exposes any overrun directly to ASan.
    std::vector<float> scratch(count, poison), guarded(count + 2, guard);
    std::vector<float> q(n * d), k(n * d), v(n * d), out(n * d + 2, guard);
    std::vector<float> scores(n * n), expected(n * d), serial(n * d);
    for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t i = 0; i < n * d; ++i) {
            q[i] = std::sin(static_cast<float>(i + pass * 7));
            k[i] = std::cos(static_cast<float>(i + pass * 11));
            v[i] = std::sin(static_cast<float>(i + pass * 13) * 0.3f);
        }
        const auto old_q = q, old_k = k, old_v = v;
        attention_baseline(q.data(), k.data(), v.data(), expected.data(), scores.data(), n, d);
        std::fill(out.begin() + 1, out.end() - 1, poison);
        attention_streaming(q.data(), k.data(), v.data(), out.data() + 1, scratch.data(), n, d, config);
        require(q == old_q && k == old_k && v == old_v, "Kernel modified its inputs");
        require(out.front() == guard && out.back() == guard, "Output boundary was overwritten");
        for (std::size_t i = 0; i < expected.size(); ++i)
            require(std::isfinite(out[i + 1]) &&
                    std::abs(out[i + 1] - expected[i]) <= 1e-4f + 1e-3f * std::abs(expected[i]),
                    "Streaming differs from baseline");
        const auto first = out;
        attention_streaming(q.data(), k.data(), v.data(), out.data() + 1, guarded.data() + 1, n, d, config);
        require(first == out, "Output depends on old scratch contents");
        require(guarded.front() == guard && guarded.back() == guard, "Scratch boundary was overwritten");
        attention_streaming(q.data(), k.data(), v.data(), out.data() + 1, scratch.data(), n, d, config);
        require(first == out, "Repeated calls differ");
        AttentionConfig single = config;
        single.threads = 1;
        std::vector<float> single_scratch(attention_streaming_scratch(n, d, single), poison);
        attention_streaming(q.data(), k.data(), v.data(), serial.data(), single_scratch.data(), n, d, single);
        require(std::equal(serial.begin(), serial.end(), out.begin() + 1),
                "Thread count changes the per-row recurrence");
    }
}
} // namespace

int main() {
    try {
        std::size_t checked = 0;
        std::vector<std::size_t> threads{1};
#ifdef _OPENMP
        threads = {1, 2, 4};
#else
        rejects<std::invalid_argument>([] { attention_streaming_scratch(1, 1, {1, 1, 2}); });
#endif
        for (const auto shape : {std::pair<std::size_t, std::size_t>{1, 1}, {17, 7}, {127, 33}, {513, 65}})
            for (std::size_t br : {1u, 16u, 32u, 64u, 1024u})
                for (std::size_t bc : {1u, 7u, 8u, 9u, 16u, 31u, 32u, 33u, 64u, 1024u})
                    for (const auto t : threads) {
                        contract(shape.first, shape.second, {br, bc, t});
                        ++checked;
                    }
        const auto huge = std::numeric_limits<std::size_t>::max();
        contract(17, 7, {huge, huge, 1}); // Oversized tiles are safely clamped.
        for (const auto config : {AttentionConfig{0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, huge}})
            rejects<std::invalid_argument>([&] { attention_streaming_scratch(1, 1, config); });
        rejects<std::invalid_argument>([] { attention_streaming_scratch(0, 1, {}); });
        rejects<std::invalid_argument>([] { attention_streaming_scratch(1, 0, {}); });
        rejects<std::invalid_argument>([] {
            attention_streaming(nullptr, nullptr, nullptr, nullptr, nullptr, 1, 1, {0, 1, 1});
        });
        rejects<std::overflow_error>([&] { attention_streaming_scratch(huge, 2, {}); });
        rejects<std::overflow_error>([&] { attention_streaming_scratch(huge / 8, 1, {huge, huge, 1}); });
        std::cout << "Streaming API: " << checked + 1
                  << " shape/tile/thread configurations passed; input preservation, scratch bounds, "
                     "buffer reuse, thread invariance and rejection checks passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
