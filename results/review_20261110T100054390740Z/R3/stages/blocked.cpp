#include <chrono>
#include "attention.h"
#include "simd.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// PV loop order: key tiles outside query rows, so each V tile is reused across
// the query block (5-19% faster, bitwise identical output; see
// scripts/packing_experiment.py). -DBLOCKED_ORIGINAL_PV restores the row-by-row
// order for that comparison. -DBLOCKED_PACK_K is the K-packing experiment.
#ifndef BLOCKED_ORIGINAL_PV
#define BLOCKED_REORDER_PV
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

#ifdef BLOCKED_PACK_K
// Packing experiment (off by default): transpose each K tile into the worker's
// panel, then compute eight scores at once so one Q value feeds eight
// independent accumulators. Repacked for every query block; included in the call.
void packed_scores(const float* q, const float* k, float* scratch, float* packed,
                   std::size_t n, std::size_t d, std::size_t i0, std::size_t iend,
                   std::size_t bc, float scale) {
    for (std::size_t j0 = 0; j0 < n; j0 += bc) {
        const std::size_t columns = std::min(bc, n - j0);
        for (std::size_t f = 0; f < d; ++f)
            for (std::size_t j = 0; j < columns; ++j)
                packed[f * bc + j] = k[(j0 + j) * d + f];
        for (std::size_t i = i0; i < iend; ++i) {
            const float* qi = q + i * d;
            float* si = scratch + i * n + j0;
            std::size_t j = 0;
            for (; j + 8 <= columns; j += 8) {
                float s[8] = {};
                for (std::size_t f = 0; f < d; ++f) {
                    const float a = qi[f];
                    const float* b = packed + f * bc + j;
                    ATTN_SIMD
                    for (int l = 0; l < 8; ++l)
                        s[l] += a * b[l];
                }
                for (int l = 0; l < 8; ++l)
                    si[j + l] = s[l] * scale;
            }
            for (; j < columns; ++j) {
                const float* kj = k + (j0 + j) * d;
                float acc = 0.0f;
                ATTN_SIMD_SUM(acc)
                for (std::size_t f = 0; f < d; ++f)
                    acc += qi[f] * kj[f];
                si[j] = acc * scale;
            }
        }
    }
}
#endif

#ifdef BLOCKED_REORDER_PV
// Default PV: key tiles outside, query rows inside, so each V tile is reused
// across the query block while it is cached. Each output element still
// accumulates keys in ascending order: bitwise identical to the row-by-row order.
void reordered_values(const float* scratch, const float* v, float* out, std::size_t n,
                      std::size_t d, std::size_t i0, std::size_t iend, std::size_t bc) {
    for (std::size_t i = i0; i < iend; ++i)
        std::fill(out + i * d, out + (i + 1) * d, 0.0f);
    for (std::size_t j0 = 0; j0 < n; j0 += bc) {
        const std::size_t jend = j0 + std::min(bc, n - j0);
        for (std::size_t i = i0; i < iend; ++i) {
            float* oi = out + i * d;
            const float* si = scratch + i * n;
            for (std::size_t j = j0; j < jend; ++j) {
                const float weight = si[j];
                const float* vj = v + j * d;
                ATTN_SIMD
                for (std::size_t f = 0; f < d; ++f)
                    oi[f] += weight * vj[f];
            }
        }
    }
}
#endif
}

std::size_t attention_blocked_scratch(std::size_t n, std::size_t d,
                                     const AttentionConfig& config) {
    validate(n, d, config);
#ifdef BLOCKED_PACK_K
    // Plus one transposed min(Bc, N) x d K panel per worker.
    const std::size_t rows = std::min(n, config.br), columns = std::min(n, config.bc);
    const std::size_t workers = std::min(config.threads, 1 + (n - 1) / rows);
    return checked_square(n) + workers * columns * d;
#else
    return checked_square(n); // Full FP32 probability / score matrix; no packing buffers.
#endif
}

double profile_stage_ms[3];
namespace {
using ProfileClock = std::chrono::steady_clock;
double profile_ms(ProfileClock::time_point start) {
    return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}
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
        auto profile_start = ProfileClock::now();
        // QK^T: cache-block both query and key rows. A contiguous feature reduction
        // computes each dot product; K rows are reused across the query tile.
#ifdef BLOCKED_PACK_K
        std::size_t worker = 0;
#ifdef _OPENMP
        // Static schedule: only threads below the block count receive blocks.
        worker = static_cast<std::size_t>(omp_get_thread_num());
#endif
        const std::size_t columns = std::min(bc, n);
        packed_scores(q, k, scratch, scratch + n * n + worker * columns * d, n, d, i0, iend, columns, scale);
#else
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
                    ATTN_SIMD_SUM(acc)
                    for (std::size_t f = 0; f < d; ++f)
                        acc += qi[f] * kj[f];
                    si[j] = acc * scale;
                }
            }
        }
#endif
        profile_stage_ms[0] += profile_ms(profile_start);
        profile_start = ProfileClock::now();
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
        profile_stage_ms[1] += profile_ms(profile_start);
        profile_start = ProfileClock::now();
        // PV: block keys, accumulate on contiguous FP32 feature lanes.
        // No additional workspace; includes non-multiple-of-tile tails.
#ifdef BLOCKED_REORDER_PV
        reordered_values(scratch, v, out, n, d, i0, iend, bc);
#else
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
                    ATTN_SIMD
                    for (std::size_t f = 0; f < d; ++f)
                        oi[f] += weight * vj[f];
                }
            }
        }
#endif
        profile_stage_ms[2] += profile_ms(profile_start);
    }
}
