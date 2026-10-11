# CPU attention

Part A: attention semantics, a single-thread materialized baseline, and an independent FP64 reference.

## Run

Requires C++17, Make, Python 3.10+ and NumPy.

```sh
python3 -m pip install numpy
make test
make full
make sanitize
```

`test` covers edge cases and two core shapes. `full` validates all 12 proposal shapes, times the baseline at (128,64) and (512,64), and writes `results.csv`. `sanitize` runs the smaller suite with AddressSanitizer and UndefinedBehaviorSanitizer. Set `PYTHON=/path/to/python3` when invoking Make if needed. `make clean` removes executables.

All files are in the root: `attention.h` defines the interface, `attention.cpp` implements the kernel, `main.cpp` runs it, and `check.py` provides reference checks. Binary inputs are temporary; executables are ignored.

## Computation

Compute `O = softmax(Q K^T / sqrt(d)) V`, with row-wise softmax, one head, batch size one, and no masking or dropout. Q/K/V/O are contiguous, row-major N by d matrices. Inputs and C++ accumulation are FP32. Softmax subtracts the row maximum and replaces scores with probabilities in the same buffer.

Call `attention_baseline(q, k, v, out, scores, n, d)` with positive dimensions, finite inputs and non-overlapping buffers. Each input/output holds N*d floats; scores holds N*N floats. The kernel allocates nothing, preserves Q/K/V, and fully overwrites scores and output on every call. The three stages are exposed for profiling. There is no threading, packing or external numerical library.

Work is O(N^2 d), approximately 4N^2 d FLOPs for the matrix products. Main scratch is 4N^2 bytes; Q/K/V/O occupy 16Nd bytes separately. There is no persistent auxiliary buffer. Buffer sizes are not process RSS or memory traffic. Extremely large finite inputs can overflow FP32 dot products; the runner rejects non-finite outputs.

```sh
./attention N d input.bin output.bin
./attention N d input.bin output.bin 15
```

Files contain native-endian IEEE FP32 values without headers: Q, K, then V for input; O for output. Other kernels can reuse `python3 check.py --binary ./other-runner --full` if their runner accepts the same arguments. Benchmark output must follow the CSV format in `main.cpp` when `--bench` is used.

## Validation

The NumPy reference casts the actual FP32 inputs to FP64, then uses matrix multiplication, stable softmax and multiplication by V. It neither regenerates FP64 inputs nor calls the C++ stages.

Every element must satisfy `abs(O - Oref) <= 1e-4 + 1e-3*abs(Oref)`. Tests report maximum absolute error and relative Frobenius error, dividing by `max(norm(Oref), 1e-12)`. NaN/Inf fail. The tolerance has not been relaxed.

Tests include N=1, zeros, uniform probabilities, identical Q/K rows, signed uniform inputs, (127,33), (513,65), concentrated distributions, and rising or early maxima. Known single-row and uniform-probability outputs also check the reference itself. Repeated baseline runs must agree exactly. The full grid uses N in {128,512,2048,4096} and d in {32,64,128}. Ten input-validation cases cover invalid dimensions, file lengths, non-finite inputs and overflow.

## Online softmax review

After processing keys for one query, the invariants are `m = max(s)`, `l = sum(exp(s-m))` and `u = sum(exp(s-m)*V)`. For a new block:

```text
m_new = max(m, max(s_block))
alpha = exp(m - m_new)
l_new = alpha*l + sum(exp(s_block - m_new))
u_new = alpha*u + exp(s_block - m_new) @ V_block
```

Rescaling puts old and new terms under the same maximum, preserving the invariants. Initialize directly from the first non-empty block; divide u/l only after the last block. Process every key once, handle partial blocks, and keep each query's state in one worker.

The Python recurrence passes FP64 checks at 1e-12 absolute/relative tolerances and FP32 checks at the project tolerance. Each edge case covers nine tile pairs from {16,32,64}, (1,1), and tiles larger than N. This checks the recurrence and boundaries. Review of C's actual kernel and thread counts remains pending its implementation.

## Recorded checks

All 21 baseline cases and 198 online tile checks passed. The smaller sanitized suite also passed. Maximum baseline absolute error was 8.28e-6. Initial median latency was 0.261 ms at (128,64) and 4.158 ms at (512,64); raw samples and quartiles are in `results.csv`.

Run on 11 October 2026: Apple M4 (4 performance and 6 efficiency cores), 16 GiB RAM, macOS 26.6.2, Apple Clang 21.0.0 targeting arm64-apple-darwin25.6.0, Python 3.12.14, NumPy 2.3.5. The OS controlled scheduling and frequency; affinity was not pinned. `sysctl` reported 64 KiB L1 data cache and 4 MiB L2; these do not describe all heterogeneous core caches.

The build uses `-O3 -std=c++17 -Wall -Wextra -Wpedantic`, without fast-math or explicit ISA flags. Apply the same settings to comparison kernels. Timing preallocates buffers and includes all three stages and output initialization, excluding allocation, generation, validation and I/O. Five warm-up calls precede excluded batch calibration. Batches target 10 ms, capped at 1024 calls. Fifteen samples report per-call median and quartiles; `samples_ms` retains chronological samples separated by semicolons. Inputs and buffers are reused, so these are warm repeated-call timings. A checksum and output file retain observable results. Sanitized builds are never timed. Initial timings establish no speedup or thread scaling.

## Sources

[Attention Is All You Need](https://arxiv.org/abs/1706.03762), Section 3.2.1, defines scaled dot-product attention. [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), Algorithm 3, gives online normalization; the weighted-value numerator uses the same rescaling. No external implementation was copied.

AI assistance was used to draft implementation, tests and documentation and run automated checks. Human review is required before submission; the course requires an accurate AI-use declaration in the final report.
