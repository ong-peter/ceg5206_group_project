#include "attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
std::size_t checked_square(std::size_t n) {
    if (!n || n > std::numeric_limits<std::size_t>::max() / n)
        throw std::invalid_argument("Invalid score matrix dimensions");
    return n * n;
}
void validate(std::size_t n, std::size_t d, const AttentionConfig& config) {
    (void)checked_square(n);
    if (!d || !config.br || !config.bc || !config.threads ||
        config.threads > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Invalid blocked attention dimensions/configuration");
#ifndef _OPENMP
    if (config.threads != 1)
        throw std::invalid_argument("Multithreading requires an OpenMP build (make OPENMP=1)");
#endif
}
}

std::size_t attention_blocked_scratch(std::size_t n, std::size_t d,
                                     const AttentionConfig& config) {
    validate(n, d, config);
    return checked_square(n); // Full FP32 probability / score matrix; no packing buffers.
}

void attention_blocked(const float* q, const float* k, const float* v,
                       float* out, float* scratch, std::size_t n, std::size_t d,
                       const AttentionConfig& config) {
    validate(n, d, config);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const std::size_t br = config.br, bc = config.bc;
    // Each query block owns disjoint score and output rows. Safe for OpenMP.
    // With OpenMP absent, the pragma is ignored and execution is single-threaded.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(config.threads))
#endif
    for (std::size_t ib = 0; ib < n / br + (n % br != 0); ++ib) {
        const std::size_t i0 = ib * br;
        const std::size_t iend = i0 + std::min(br, n - i0);
        // QK^T: cache-block both query and key rows. A contiguous feature reduction
        // computes each dot product; K rows are reused across the query tile.
        for (std::size_t jb = 0; jb < n / bc + (n % bc != 0); ++jb) {
            const std::size_t j0 = jb * bc;
            const std::size_t jend = j0 + std::min(bc, n - j0);
            for (std::size_t i = i0; i < iend; ++i) {
                const float* qi = q + i * d;
                float* si = scratch + i * n;
                for (std::size_t j = j0; j < jend; ++j) {
                    const float* kj = k + j * d;
                    float acc = 0.0f;
                    // SIMD reduction: vectorization report should identify this loop.
#ifdef _OPENMP
                    #pragma omp simd reduction(+:acc)
#endif
                    for (std::size_t f = 0; f < d; ++f)
                        acc += qi[f] * kj[f];
                    si[j] = acc * scale;
                }
            }
        }
        // Stable row-wise softmax, in place: n*n scratch is retained.
        for (std::size_t i = i0; i < iend; ++i) {
            float* si = scratch + i * n;
            const float maximum = *std::max_element(si, si + n);
            float sum = 0.0f;
            for (std::size_t j = 0; j < n; ++j) {
                si[j] = std::exp(si[j] - maximum);
                sum += si[j];
            }
            const float inv_sum = 1.0f / sum;
            for (std::size_t j = 0; j < n; ++j)
                si[j] *= inv_sum;
        }
        // PV: block keys, accumulate on contiguous FP32 feature lanes.
        // No additional workspace; includes non-multiple-of-tile tails.
        for (std::size_t i = i0; i < iend; ++i) {
            float* oi = out + i * d;
            std::fill(oi, oi + d, 0.0f);
            const float* si = scratch + i * n;
            for (std::size_t jb = 0; jb < n / bc + (n % bc != 0); ++jb) {
                const std::size_t j0 = jb * bc;
                const std::size_t jend = j0 + std::min(bc, n - j0);
                for (std::size_t j = j0; j < jend; ++j) {
                    const float weight = si[j];
                    const float* vj = v + j * d;
#ifdef _OPENMP
                    #pragma omp simd
#endif
                    for (std::size_t f = 0; f < d; ++f)
                        oi[f] += weight * vj[f];
                }
            }
        }
    }
}
