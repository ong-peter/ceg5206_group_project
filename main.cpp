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

// Binary files: native-endian FP32, no headers; input Q then K then V, output O.
// Buffers occupy 16*n*d bytes plus scratch. Repeats defaults to zero.
int main(int argc, char** argv) {
    try {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
        if (argc == 2 && std::string(argv[1]) == "--list") {
            std::cout << "baseline\n";
#ifdef WITH_BLOCKED
            std::cout << "blocked\n";
#endif
#ifdef WITH_STREAMING
            std::cout << "streaming\n";
#endif
            return 0;
        }
        if (argc < 5)
            throw std::invalid_argument("Usage: attention N d input.bin output.bin [repeats] "
                                        "[--algorithm NAME] [--br B] [--bc B] [--threads T]");
        const std::size_t n = number(argv[1]);
        const std::size_t d = number(argv[2]);
        std::size_t repeats = 0;
        std::string algorithm = "baseline";
        AttentionConfig config;
        int argument = 5;
        if (argument < argc && std::string(argv[argument]).rfind("--", 0) != 0)
            repeats = number(argv[argument++]);
        while (argument < argc) {
            const std::string option = argv[argument++];
            if (argument == argc)
                throw std::invalid_argument("Missing option value");
            const char* value = argv[argument++];
            if (option == "--algorithm") algorithm = value;
            else if (option == "--br") config.br = number(value);
            else if (option == "--bc") config.bc = number(value);
            else if (option == "--threads") config.threads = number(value);
            else throw std::invalid_argument("Unknown option: " + option);
        }
        if (!n || !d)
            throw std::invalid_argument("N and d must be positive");
        if (!config.br || !config.bc || !config.threads ||
            config.threads > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Tile sizes and thread count must be positive; threads must fit int");
        const std::size_t nd = product(n, d);
        const std::size_t input_count = product(3, nd);
        const std::size_t input_bytes = product(sizeof(float), input_count);
        const std::size_t io_bytes = product(16, nd);
        AttentionFunction kernel = nullptr;
        std::size_t scratch_count = 0;
        if (algorithm == "baseline") {
            if (config.threads != 1)
                throw std::invalid_argument("Baseline supports one thread only");
            kernel = attention_baseline;
            scratch_count = product(n, n);
        }
#ifdef WITH_BLOCKED
        else if (algorithm == "blocked") {
            kernel = attention_blocked;
            scratch_count = attention_blocked_scratch(n, d, config);
        }
#endif
#ifdef WITH_STREAMING
        else if (algorithm == "streaming") {
            kernel = attention_streaming;
            scratch_count = attention_streaming_scratch(n, d, config);
        }
#endif
        else {
            throw std::invalid_argument("Algorithm unavailable: " + algorithm + "; use --list");
        }
        const std::size_t scratch_bytes = product(sizeof(float), scratch_count);
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
        std::vector<float> out(nd), scratch(scratch_count);
        const float* q = data.data();
        const float* k = q + nd;
        const float* v = k + nd;
        const auto call = [&] { kernel(q, k, v, out.data(), scratch.data(), n, d, config); };
        std::fill(out.begin(), out.end(), std::numeric_limits<float>::quiet_NaN());
        std::fill(scratch.begin(), scratch.end(), std::numeric_limits<float>::quiet_NaN());
        call();
        if (!std::all_of(out.begin(), out.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("Non-finite output; input magnitude is unsupported");
        {
            const auto first = out;
            call();
            if (first != out)
                throw std::runtime_error("Repeated calls with reused buffers produced different outputs");
        }
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
            std::cout << "algorithm,n,d,br,bc,threads,median_ms,q1_ms,q3_ms,batch,warmups,repeats,scratch_bytes,io_bytes,checksum,samples_ms\n"
                      << std::setprecision(10) << algorithm << ',' << n << ',' << d << ','
                      << (algorithm == "baseline" ? 0 : config.br) << ','
                      << (algorithm == "baseline" ? 0 : config.bc) << ',' << config.threads << ','
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
