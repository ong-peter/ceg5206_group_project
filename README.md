# CPU attention

Part A: a single-thread FP32 baseline and an independent FP64 NumPy reference.
Computes `softmax(Q K^T / sqrt(d)) V` with one head and no mask.

## Run

Requires C++17, Make, Python 3.10+ and NumPy. Set `PYTHON=/path/to/python3` if needed.

```sh
python3 -m pip install numpy
make test       # Available kernels and edge/tile checks
make full       # All 12 shapes; initial timings; results.csv
make sanitize   # AddressSanitizer and UndefinedBehaviorSanitizer
```

Interface notes are in `attention.h`; test options: `python3 check.py --help`.

## Validation and results

The reference converts the exact FP32 inputs to FP64. Each element must satisfy `abs(O - Oref) <= 1e-4 + 1e-3*abs(Oref)`; NaN/Inf fail. Tests cover known outputs, random inputs, tails and concentrated distributions. CSV rows name the actual kernel and configuration. Timed rows include its workspace size, quartiles and raw samples.

On 11 October 2026, 21 baseline cases, 198 Python online-recurrence checks and the sanitized suite passed. Maximum baseline absolute error was 8.28e-6. B/C correctness checks require their implementations.

Recorded environment: Apple M4, macOS 26.6.2, Apple Clang 21.0.0, Python 3.12.14, NumPy 2.3.5. Flags are in Makefile. Timing follows correctness checks, uses preallocated buffers, five warm-ups and 15 samples, and excludes allocation, validation, I/O and batch calibration. Samples are per-call milliseconds with reused buffers; scheduling and frequency are OS-controlled. These are initial measurements.

[Attention Is All You Need](https://arxiv.org/abs/1706.03762), Section 3.2.1; [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), Algorithm 3.

AI tools were used for coding assistance.
