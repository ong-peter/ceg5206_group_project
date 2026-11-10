// Frozen pre-optimization implementation from the local 2026-10-17 version.
// Benchmark fixture only; the production Makefile does not compile this file.
#include "attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("Streaming workspace size overflow");
    return a + b;
}

std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("Streaming workspace size overflow");
    return a * b;
}

struct Layout {
    std::size_t br, bc, blocks, workers, stride, total;
};

Layout layout(std::size_t n, std::size_t d, const AttentionConfig& config) {
    if (!n || !d || !config.br || !config.bc || !config.threads ||
        config.threads > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Streaming requires positive dimensions/tiles and threads fitting int");
#ifndef _OPENMP
    if (config.threads != 1)
        throw std::invalid_argument("Streaming threads > 1 require an OpenMP build (make OPENMP=1)");
#endif
    checked_product(checked_product(n, d), sizeof(float));
    Layout result{};
    result.br = std::min(n, config.br);
    result.bc = std::min(n, config.bc);
    result.blocks = 1 + (n - 1) / result.br;
    result.workers = std::min(config.threads, result.blocks);
    // One score/probability tile, weighted values u, maxima m and sums l.
    // K and V are read directly: no packing, transpose, or persistent arrays.
    result.stride = checked_add(checked_product(result.br, result.bc),
                               checked_product(result.br, d));
    result.stride = checked_add(result.stride, checked_product(2, result.br));
    // Separate workers' live scratch by 128 bytes, including on Apple CPUs.
    // This guard is included in the allocation, even after the last worker.
    if (result.workers > 1)
        result.stride = checked_add(result.stride, 32);
    result.total = checked_product(result.workers, result.stride);
    checked_product(result.total, sizeof(float));
    return result;
}

void query_block(const float* q, const float* k, const float* v, float* out,
                 float* scratch, std::size_t n, std::size_t d,
                 std::size_t begin, const Layout& work, float scale) {
    const std::size_t rows = std::min(work.br, n - begin);
    float* scores = scratch;
    float* values = scores + work.br * work.bc;
    float* maximum = values + work.br * d;
    float* total = maximum + work.br;
    std::fill(values, values + rows * d, 0.0f);
    std::fill(maximum, maximum + rows, -std::numeric_limits<float>::infinity());
    std::fill(total, total + rows, 0.0f);

    for (std::size_t key = 0; key < n;) {
        const std::size_t columns = std::min(work.bc, n - key);
        for (std::size_t i = 0; i < rows; ++i) {
            const float* query = q + (begin + i) * d;
            float* row = scores + i * work.bc;
            for (std::size_t j = 0; j < columns; ++j) {
                const float* key_row = k + (key + j) * d;
                float sum = 0.0f;
                for (std::size_t f = 0; f < d; ++f)
                    sum += query[f] * key_row[f];
                row[j] = sum * scale;
            }
        }

        for (std::size_t i = 0; i < rows; ++i) {
            float* row = scores + i * work.bc;
            float* weighted = values + i * d;
            const float block_maximum = *std::max_element(row, row + columns);
            const float next_maximum = std::max(maximum[i], block_maximum);
            // Explicit first-tile handling avoids exp(-inf - -inf).
            const float alpha = key == 0 ? 0.0f : std::exp(maximum[i] - next_maximum);
            float block_sum = 0.0f;
            for (std::size_t j = 0; j < columns; ++j) {
                row[j] = std::exp(row[j] - next_maximum);
                block_sum += row[j];
            }
            for (std::size_t f = 0; f < d; ++f)
                weighted[f] *= alpha;
            // Contiguous feature loop permits SIMD without reassociating the
            // dot-product or softmax reductions. No fast-math is required.
            for (std::size_t j = 0; j < columns; ++j) {
                const float weight = row[j];
                const float* value_row = v + (key + j) * d;
                for (std::size_t f = 0; f < d; ++f)
                    weighted[f] += weight * value_row[f];
            }
            total[i] = alpha * total[i] + block_sum;
            maximum[i] = next_maximum;
        }
        key += columns;
    }

    for (std::size_t i = 0; i < rows; ++i) {
        float* output = out + (begin + i) * d;
        const float* weighted = values + i * d;
        for (std::size_t f = 0; f < d; ++f)
            output[f] = weighted[f] / total[i];
    }
}

} // namespace

std::size_t attention_streaming_scratch(std::size_t n, std::size_t d,
                                       const AttentionConfig& config) {
    return layout(n, d, config).total;
}

void attention_streaming(const float* q, const float* k, const float* v,
                         float* out, float* scratch, std::size_t n, std::size_t d,
                         const AttentionConfig& config) {
    const Layout work = layout(n, d, config);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    if (work.workers == 1) {
        for (std::size_t block = 0; block < work.blocks; ++block)
            query_block(q, k, v, out, scratch, n, d, block * work.br, work, scale);
        return;
    }
#ifdef _OPENMP
    // The complete recurrence of a query row belongs to one worker.
    // A runtime-created smaller team remains correct; allocation is capacity.
#pragma omp parallel num_threads(static_cast<int>(work.workers))
    {
        float* local = scratch + static_cast<std::size_t>(omp_get_thread_num()) * work.stride;
#pragma omp for schedule(static)
        for (std::size_t block = 0; block < work.blocks; ++block)
            query_block(q, k, v, out, local, n, d, block * work.br, work, scale);
    }
#endif
}
