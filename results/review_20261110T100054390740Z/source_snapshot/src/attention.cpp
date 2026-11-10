#include "attention.h"

#include <algorithm>
#include <cmath>

void attention_scores(const float* q, const float* k, float* scores,
                      std::size_t n, std::size_t d) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (std::size_t f = 0; f < d; ++f)
                sum += q[i * d + f] * k[j * d + f];
            scores[i * n + j] = sum * scale;
        }
    }
}

void attention_softmax(float* scores, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        float* row = scores + i * n;
        const float maximum = *std::max_element(row, row + n);
        float sum = 0.0f;
        for (std::size_t j = 0; j < n; ++j) {
            row[j] = std::exp(row[j] - maximum);
            sum += row[j];
        }
        for (std::size_t j = 0; j < n; ++j)
            row[j] /= sum;
    }
}

void attention_values(const float* scores, const float* v, float* out,
                      std::size_t n, std::size_t d) {
    for (std::size_t i = 0; i < n; ++i) {
        float* row = out + i * d;
        std::fill(row, row + d, 0.0f);
        for (std::size_t j = 0; j < n; ++j) {
            const float weight = scores[i * n + j];
            for (std::size_t f = 0; f < d; ++f)
                row[f] += weight * v[j * d + f];
        }
    }
}

void attention_baseline(const float* q, const float* k, const float* v,
                        float* out, float* scores, std::size_t n, std::size_t d,
                        const AttentionConfig&) {
    attention_scores(q, k, scores, n, d);
    attention_softmax(scores, n);
    attention_values(scores, v, out, n, d);
}
