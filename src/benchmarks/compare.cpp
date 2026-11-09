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

#define DECLARE_VARIANT(name) \
    std::size_t name##_scratch(std::size_t, std::size_t, const AttentionConfig&); \
    void name(const float*, const float*, const float*, float*, float*, \
              std::size_t, std::size_t, const AttentionConfig&)
DECLARE_VARIANT(attention_original);
DECLARE_VARIANT(attention_rescale_only);
DECLARE_VARIANT(attention_packed_only);

using Clock = std::chrono::steady_clock;

template<class T>
std::vector<T> read(const char* path, std::size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(T)))
        throw std::runtime_error("Incorrect benchmark input size");
    file.seekg(0);
    std::vector<T> result(count);
    file.read(reinterpret_cast<char*>(result.data()), count * sizeof(T));
    if (!file) throw std::runtime_error("Could not read benchmark input");
    return result;
}

double percentile(std::vector<double> samples, double p) {
    std::sort(samples.begin(), samples.end());
    const double index = p * (samples.size() - 1);
    const auto low = static_cast<std::size_t>(index);
    const auto high = std::min(low + 1, samples.size() - 1);
    return samples[low] + (index - low) * (samples[high] - samples[low]);
}

struct Variant {
    std::string name;
    AttentionFunction function;
    std::vector<float> out, scratch;
    std::vector<double> samples;
    std::size_t batch = 1;
    double max_error = 0, relative_error = 0;
    Variant(const char* label, AttentionFunction fn, std::size_t nd, std::size_t count)
        : name(label), function(fn), out(nd), scratch(count) {}
};

int main(int argc, char** argv) {
    try {
        if (argc != 8)
            throw std::invalid_argument("Usage: compare N d input-fp32.bin reference-fp64.bin Br Bc repeats");
        const std::size_t n = std::stoull(argv[1]), d = std::stoull(argv[2]);
        const AttentionConfig config{std::stoull(argv[5]), std::stoull(argv[6]), 1};
        const std::size_t repeats = std::stoull(argv[7]);
        if (!n || n > 8192 || !d || d > 1024 || !config.br || !config.bc || !repeats || repeats > 1000)
            throw std::invalid_argument("Benchmark dimensions/repeats outside supported bounds");
        auto data = read<float>(argv[3], 3 * n * d);
        const auto expected = read<double>(argv[4], n * d);
        const auto pristine = data;
        const float* q = data.data();
        const float* k = q + n * d;
        const float* v = k + n * d;
        std::vector<Variant> variants;
        variants.emplace_back("baseline", attention_baseline, n*d, n*n);
        variants.emplace_back("original", attention_original, n*d, attention_original_scratch(n,d,config));
        variants.emplace_back("rescale_only", attention_rescale_only, n*d, attention_rescale_only_scratch(n,d,config));
        variants.emplace_back("packed_only", attention_packed_only, n*d, attention_packed_only_scratch(n,d,config));
        variants.emplace_back("optimized", attention_streaming, n*d, attention_streaming_scratch(n,d,config));
        const auto call = [&](Variant& x) {
            x.function(q, k, v, x.out.data(), x.scratch.data(), n, d, config);
        };
        const auto batch_time = [&](Variant& x) {
            const auto start = Clock::now();
            for (std::size_t b = 0; b < x.batch; ++b) call(x);
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        };
        const auto validate = [&](Variant& x) {
            double squared_error = 0, squared_norm = 0;
            for (std::size_t i = 0; i < n*d; ++i) {
                const double error = std::abs(static_cast<double>(x.out[i]) - expected[i]);
                if (!std::isfinite(x.out[i]) || !std::isfinite(expected[i]) ||
                    error > 1e-4 + 1e-3 * std::abs(expected[i]))
                    throw std::runtime_error("Independent FP64 validation failed: " + x.name);
                x.max_error = std::max(x.max_error, error);
                squared_error += error * error;
                squared_norm += expected[i] * expected[i];
            }
            x.relative_error = std::sqrt(squared_error) / std::max(std::sqrt(squared_norm), 1e-12);
            if (data != pristine) throw std::runtime_error("Input was modified: " + x.name);
        };
        for (auto& x : variants) {
            std::fill(x.out.begin(), x.out.end(), std::numeric_limits<float>::quiet_NaN());
            std::fill(x.scratch.begin(), x.scratch.end(), std::numeric_limits<float>::quiet_NaN());
            call(x);
            validate(x);
            for (int warmup = 0; warmup < 5; ++warmup) call(x);
            while (batch_time(x) < 10.0 && x.batch < 1024) x.batch *= 2;
            x.samples.reserve(repeats);
        }
        // Rotate and reverse order between rounds. Each sample includes the
        // complete call (including K packing), with allocations/I/O excluded.
        for (std::size_t round = 0; round < repeats; ++round) {
            for (std::size_t step = 0; step < variants.size(); ++step) {
                const auto index = (round + (round % 2 ? variants.size() - 1 - step : step)) % variants.size();
                auto& x = variants[index];
                x.samples.push_back(batch_time(x) / x.batch);
            }
        }
        std::cout << "algorithm,n,d,br,bc,threads,median_ms,q1_ms,q3_ms,batch,warmups,repeats,scratch_bytes,io_bytes,max_abs_error,rel_frobenius_error,checksum,samples_ms\n";
        for (auto& x : variants) {
            validate(x);
            std::cout << std::setprecision(10) << x.name << ',' << n << ',' << d << ','
                      << (x.name == "baseline" ? 0 : config.br) << ','
                      << (x.name == "baseline" ? 0 : config.bc) << ",1," << percentile(x.samples, .5) << ','
                      << percentile(x.samples, .25) << ',' << percentile(x.samples, .75) << ','
                      << x.batch << ",5," << repeats << ',' << x.scratch.size()*4 << ',' << 16*n*d << ','
                      << x.max_error << ',' << x.relative_error << ','
                      << std::accumulate(x.out.begin(), x.out.end(), 0.0) << ',';
            for (std::size_t i = 0; i < x.samples.size(); ++i)
                std::cout << (i ? ";" : "") << x.samples[i];
            std::cout << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
