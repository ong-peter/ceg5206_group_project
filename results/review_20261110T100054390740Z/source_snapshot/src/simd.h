#pragma once

// Loop-level SIMD hints for optimized kernels.
// The default build passes -fopenmp-simd, which honours "omp simd" without
// needing the OpenMP runtime, so single-thread builds still vectorize.
// "omp simd reduction" lets the compiler keep several partial sums in vector
// lanes. That reorders FP32 additions, so results differ slightly from a
// sequential sum (within the acceptance tolerance).
// ATTN_NO_SIMD (set by the attention-novec target) removes the hints for the
// vectorization ablation.
#ifdef ATTN_NO_SIMD
#define ATTN_SIMD
#define ATTN_SIMD_SUM(var)
#else
#define ATTN_PRAGMA(x) _Pragma(#x)
#define ATTN_SIMD ATTN_PRAGMA(omp simd)
#define ATTN_SIMD_SUM(var) ATTN_PRAGMA(omp simd reduction(+:var))
#endif
