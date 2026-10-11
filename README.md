# CPU attention

Part A: a single-thread FP32 baseline and an independent FP64 NumPy reference.

## Run

Requires C++17, Make, Python 3.10+ and NumPy. Set `PYTHON=/path/to/python3` if needed.

```sh
python3 -m pip install numpy
make test       # Available kernels and edge/tile checks
make full       # All 12 shapes; initial timings; results.csv
make sanitize   # AddressSanitizer and UndefinedBehaviorSanitizer
```

## Add B or C

Add `blocked.cpp` or `streaming.cpp`. Each implements the two functions declared in `attention.h`: a scratch-size query and an attention kernel. Makefile detects these files automatically; no algorithm is supplied here.

The query returns scratch size in **floats**, including packing and all workers. The runner preallocates exactly that buffer; streaming does not receive an N*N matrix. The kernel uses `AttentionConfig {br, bc, threads}`, preserves inputs, overwrites output and initializes scratch on each call without allocation. Reject unsupported settings. Baseline uses N*N scratch floats and one thread. Each `make` rebuild reflects the current modules and compiler flags.

All kernels compute `softmax(Q K^T / sqrt(d)) V`, row-wise, with one head and no mask. Q/K/V/O are separate, contiguous row-major N by d FP32 arrays. Input/output storage is 16Nd bytes.

```sh
./attention --list
python3 check.py --algorithm blocked --full --tiles --br 32 --bc 64
python3 check.py --algorithm streaming --full --tiles --threads 2 --csv streaming.csv
```

`--algorithm all` checks compiled kernels, keeping baseline single-threaded. Missing kernels fail explicitly. `--tiles` sweeps edge cases over the chosen pair, nine pairs from {16,32,64}, (1,1), and tiles larger than N. Core shapes use the chosen pair. `--bench` adds initial timings; `--binary` selects another compatible runner. Defaults: Br=32, Bc=32, threads=1.

The runner accepts `N d input.bin output.bin [repeats]` plus the same algorithm/tile/thread options. Files contain native-endian FP32 values without headers: Q, K, then V for input; O for output. Repeats defaults to zero. `make clean` removes generated binaries.

## Validation and results

The reference converts the exact FP32 inputs to FP64. Each element must satisfy `abs(O - Oref) <= 1e-4 + 1e-3*abs(Oref)`; NaN/Inf fail. Tests cover known outputs, random inputs, tails and concentrated distributions. CSV rows name the actual kernel and configuration. Timed rows include its workspace size, quartiles and raw samples.

On 11 October 2026, 21 baseline cases, 198 Python online-recurrence checks and the sanitized suite passed. Maximum baseline absolute error was 8.28e-6. B/C correctness checks require their implementations.

Recorded environment: Apple M4, macOS 26.6.2, Apple Clang 21.0.0, Python 3.12.14, NumPy 2.3.5. Flags are in Makefile. Timing follows correctness checks, uses preallocated buffers, five warm-ups and 15 samples, and excludes allocation, validation, I/O and batch calibration. Samples are per-call milliseconds with reused buffers; scheduling and frequency are OS-controlled. These are initial measurements.

[Attention Is All You Need](https://arxiv.org/abs/1706.03762), Section 3.2.1; [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), Algorithm 3.

AI tools were used for coding assistance.
