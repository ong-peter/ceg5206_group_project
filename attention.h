#pragma once

#include <cstddef>

struct AttentionConfig {
    std::size_t br = 32;
    std::size_t bc = 32;
    std::size_t threads = 1;
};

// softmax(Q K^T / sqrt(d)) V, row-wise, one head, no mask.
// Contiguous row-major FP32 arrays; positive dimensions, non-overlapping buffers.
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

// B: add blocked.cpp with attention_blocked_scratch and attention_blocked.
// C: add streaming.cpp with attention_streaming_scratch and attention_streaming.
// Makefile detects these files; ./attention --list shows compiled kernels.
// Scratch queries return total floats, including packing and all workers.
// The runner allocates exactly that size before timing, with no extra n*n buffer.
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
