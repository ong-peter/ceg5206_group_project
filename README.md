# CPU attention

Part A: a single-thread FP32 attention baseline and an independent FP64 NumPy reference.

## Run

Requires C++17, Make, Python 3.10+ and NumPy.

```sh
python3 -m pip install numpy
make test       # Edge cases and online softmax checks
make full       # All 12 proposal shapes; save results.csv
make sanitize   # AddressSanitizer and UndefinedBehaviorSanitizer
```

Set `PYTHON=/path/to/python3` if needed. `make clean` removes generated binaries. All source files are in the root.

## Interface

Compute `O = softmax(Q K^T / sqrt(d)) V`, with row-wise softmax, one head and no mask. Matrices are contiguous, row-major N by d arrays. The baseline uses FP32 accumulation and stable softmax, storing probabilities in the score buffer.

`attention_baseline(q, k, v, out, scores, n, d)` requires positive dimensions and separate buffers: N*d floats for each input/output, N*N for scores. It preserves inputs, overwrites output and scratch, and performs no allocation. The three stages are exposed in `attention.h` for profiling. Scratch is 4N^2 bytes; inputs/output occupy 16Nd bytes separately.

```sh
./attention N d input.bin output.bin [repeats]
```

Files contain native-endian FP32 values without headers: Q, K, then V for input; O for output. Repeats defaults to zero. Other runners with this interface can use `python3 check.py --binary ./other-runner --full`.

## Checks and results

The reference converts the exact FP32 inputs to FP64. Every output element must satisfy `abs(O - Oref) <= 1e-4 + 1e-3*abs(Oref)`; NaN/Inf fail. Tests cover known outputs, signed random inputs, non-divisible dimensions and concentrated distributions. The Python online recurrence checks first-block handling, rescaling and tile boundaries. Review of C's actual implementation remains pending.

On 11 October 2026, 21 baseline cases, 198 online tile checks and the smaller sanitized suite passed. Maximum baseline absolute error was 8.28e-6. Initial median latency was 0.261 ms at (128,64) and 4.158 ms at (512,64). `results.csv` includes errors, quartiles and raw timing samples.

Environment: Apple M4, macOS 26.6.2, Apple Clang 21.0.0, Python 3.12.14, NumPy 2.3.5. Compiler flags are in `Makefile`. Timings use preallocated buffers, five warm-ups and 15 samples, excluding allocation, validation and I/O. Batch calibration is excluded; samples are per-call milliseconds. Buffers are reused; scheduling and frequency are OS-controlled. These are initial baseline measurements, not final comparisons.

## Sources

[Attention Is All You Need](https://arxiv.org/abs/1706.03762), Section 3.2.1; [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), Algorithm 3.

AI tools were used for coding assistance.
