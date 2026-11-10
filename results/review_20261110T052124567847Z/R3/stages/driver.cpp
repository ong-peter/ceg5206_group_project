
#include "attention.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
#if defined(PROFILE_BLOCKED)
extern double profile_stage_ms[3];
#elif defined(PROFILE_STREAMING)
extern double profile_scores_ms, profile_update_ms;
#else
double profile_stage_ms[3];
#endif
template<class T> std::vector<T> read(const char* path, std::size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(T)))
        throw std::runtime_error("Incorrect profile input size");
    file.seekg(0);
    std::vector<T> result(count);
    file.read(reinterpret_cast<char*>(result.data()), count * sizeof(T));
    return result;
}
int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::invalid_argument("profile N d br bc repeats inputs reference");
        const std::size_t n = std::stoull(argv[1]), d = std::stoull(argv[2]);
        const AttentionConfig config{std::stoull(argv[3]), std::stoull(argv[4]), 1};
        const int repeats = std::stoi(argv[5]);
        const auto data = read<float>(argv[6], 3 * n * d);
        const auto expected = read<double>(argv[7], n * d);
        const float *q = data.data(), *k = q + n * d, *v = k + n * d;
        std::vector<float> out(n * d);
#if defined(PROFILE_BLOCKED)
        std::vector<float> scratch(attention_blocked_scratch(n, d, config));
        const auto call = [&] { attention_blocked(q, k, v, out.data(), scratch.data(), n, d, config); };
#elif defined(PROFILE_STREAMING)
        std::vector<float> scratch(attention_streaming_scratch(n, d, config));
        const auto call = [&] { attention_streaming(q, k, v, out.data(), scratch.data(), n, d, config); };
#else
        std::vector<float> scratch(n * n);
        const auto call = [&] {
            auto start = Clock::now();
            attention_scores(q, k, scratch.data(), n, d);
            profile_stage_ms[0] += ms_since(start);
            start = Clock::now();
            attention_softmax(scratch.data(), n);
            profile_stage_ms[1] += ms_since(start);
            start = Clock::now();
            attention_values(scratch.data(), v, out.data(), n, d);
            profile_stage_ms[2] += ms_since(start);
        };
#endif
        for (int i = 0; i < 5; ++i) call();
#if defined(PROFILE_STREAMING)
        profile_scores_ms = profile_update_ms = 0;
#else
        std::fill(profile_stage_ms, profile_stage_ms + 3, 0.0);
#endif
        std::vector<double> totals;
        for (int i = 0; i < repeats; ++i) {
            const auto start = Clock::now();
            call();
            totals.push_back(ms_since(start));
        }
        for (std::size_t i = 0; i < n * d; ++i)
            if (!std::isfinite(out[i]) || std::abs(out[i] - expected[i]) > 1e-4 + 1e-3 * std::abs(expected[i]))
                throw std::runtime_error("Profile correctness failed");
        double mean = 0;
        for (double t : totals) mean += t / repeats;
        std::sort(totals.begin(), totals.end());
        std::cout << std::setprecision(10);
#if defined(PROFILE_STREAMING)
        std::cout << profile_scores_ms / repeats << ',' << profile_update_ms / repeats;
#else
        std::cout << profile_stage_ms[0] / repeats << ',' << profile_stage_ms[1] / repeats << ','
                  << profile_stage_ms[2] / repeats;
#endif
        std::cout << ',' << mean << ',' << totals[totals.size() / 2] << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
