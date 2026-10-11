#pragma once

#include <cstddef>

struct AttentionConfig {
    std::size_t br = 32;
    std::size_t bc = 32;
    std::size_t threads = 1;
};

// Row-major FP32 arrays; positive dimensions and non-overlapping buffers.
// Q/K/V/O each contain n*d floats. Temporary arrays belong in scratch.
// Preserve inputs, overwrite output and initialize scratch on every call.
// Reject unsupported settings; do not allocate inside the kernel.
using AttentionFunction = void (*)(const float*, const float*, const float*,
                                  float*, float*, std::size_t, std::size_t,
                                  const AttentionConfig&);

// Baseline scores contains n*n floats. Baseline is always single-threaded.
void attention_scores(const float* q, const float* k, float* scores,
                      std::size_t n, std::size_t d);
void attention_softmax(float* scores, std::size_t n);
void attention_values(const float* scores, const float* v, float* out,
                      std::size_t n, std::size_t d);
void attention_baseline(const float* q, const float* k, const float* v,
                        float* out, float* scores, std::size_t n, std::size_t d,
                        const AttentionConfig& config = {});

// Optional modules return the total scratch size in floats, including all workers
// and packing buffers. The runner allocates it once, outside timing.
std::size_t attention_blocked_scratch(std::size_t n, std::size_t d,
                                     const AttentionConfig& config);
void attention_blocked(const float* q, const float* k, const float* v,
                       float* out, float* scratch, std::size_t n, std::size_t d,
                       const AttentionConfig& config);
std::size_t attention_streaming_scratch(std::size_t n, std::size_t d,
                                       const AttentionConfig& config);
void attention_streaming(const float* q, const float* k, const float* v,
                         float* out, float* scratch, std::size_t n, std::size_t d,
                         const AttentionConfig& config);
