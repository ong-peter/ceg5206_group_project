#pragma once

#include <cstddef>

// Row-major FP32 matrices; n and d are positive. Buffers must not overlap.
// Q, K, V and O contain n*d floats. Scores contains n*n floats.
void attention_scores(const float* q, const float* k, float* scores,
                      std::size_t n, std::size_t d);
void attention_softmax(float* scores, std::size_t n);
void attention_values(const float* scores, const float* v, float* out,
                      std::size_t n, std::size_t d);
void attention_baseline(const float* q, const float* k, const float* v,
                        float* out, float* scores, std::size_t n, std::size_t d);
