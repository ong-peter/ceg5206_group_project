#include "attention.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

std::size_t number(const char* text) {
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected an unsigned integer");
    const auto parsed = std::stoull(value);
    if (parsed > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Integer is too large");
    return static_cast<std::size_t>(parsed);
}

std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("Matrix size is too large");
    return a * b;
}

double percentile(const std::vector<double>& sorted, double p) {
    const double position = p * static_cast<double>(sorted.size() - 1);
    const auto low = static_cast<std::size_t>(position);
    const auto high = std::min(low + 1, sorted.size() - 1);
    return sorted[low] + (position - low) * (sorted[high] - sorted[low]);
}

int main(int argc, char** argv) {
    try {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
        if (argc != 5 && argc != 6)
            throw std::invalid_argument("Usage: attention N d input.bin output.bin [repeats]");
        const std::size_t n = number(argv[1]);
        const std::size_t d = number(argv[2]);
        const std::size_t repeats = argc == 6 ? number(argv[5]) : 0;
        if (!n || !d)
            throw std::invalid_argument("N and d must be positive");
        const std::size_t nd = product(n, d);
        const std::size_t input_count = product(3, nd);
        const std::size_t input_bytes = product(sizeof(float), input_count);
        const std::size_t scratch_count = product(n, n);
        const std::size_t scratch_bytes = product(sizeof(float), scratch_count);
        const std::size_t io_bytes = product(16, nd);
        if (input_bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::overflow_error("Input file is too large");
        std::ifstream input(argv[3], std::ios::binary | std::ios::ate);
        if (!input || input.tellg() != static_cast<std::streamoff>(input_bytes))
            throw std::runtime_error("Input must contain exactly Q, K and V as native FP32");
        input.seekg(0);
        std::vector<float> data(input_count);
        input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(input_bytes));
        if (!input || !std::all_of(data.begin(), data.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("Input must contain finite FP32 values");
        std::vector<float> out(nd), scores(scratch_count);
        const float* q = data.data();
        const float* k = q + nd;
        const float* v = k + nd;
        const auto call = [&] { attention_baseline(q, k, v, out.data(), scores.data(), n, d); };
        call();
        if (!std::all_of(out.begin(), out.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("Non-finite output; input magnitude is unsupported");
        if (repeats) {
            for (int i = 0; i < 5; ++i)
                call();
            const auto time_batch = [&](std::size_t batch) {
                const auto start = Clock::now();
                for (std::size_t i = 0; i < batch; ++i)
                    call();
                return std::chrono::duration<double>(Clock::now() - start).count();
            };
            std::size_t batch = 1;
            while (time_batch(batch) < 0.01 && batch < 1024)
                batch *= 2;
            std::vector<double> times;
            times.reserve(repeats);
            for (std::size_t i = 0; i < repeats; ++i)
                times.push_back(1000.0 * time_batch(batch) / batch);
            auto sorted = times;
            std::sort(sorted.begin(), sorted.end());
            std::cout << "n,d,median_ms,q1_ms,q3_ms,batch,warmups,repeats,scratch_bytes,io_bytes,checksum,samples_ms\n"
                      << std::setprecision(10) << n << ',' << d << ','
                      << percentile(sorted, 0.5) << ',' << percentile(sorted, 0.25) << ','
                      << percentile(sorted, 0.75) << ',' << batch << ",5," << repeats << ','
                      << scratch_bytes << ',' << io_bytes << ','
                      << std::accumulate(out.begin(), out.end(), 0.0) << ',';
            for (std::size_t i = 0; i < times.size(); ++i)
                std::cout << (i ? ";" : "") << times[i];
            std::cout << '\n';
        }
        if (!std::all_of(out.begin(), out.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("Non-finite output; input magnitude is unsupported");
        std::ofstream output(argv[4], std::ios::binary);
        output.write(reinterpret_cast<const char*>(out.data()),
                     static_cast<std::streamsize>(nd * sizeof(float)));
        if (!output)
            throw std::runtime_error("Could not write output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
