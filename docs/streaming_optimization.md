# Single-thread streaming attention optimization and bottleneck analysis

> This record covers the standalone Part C experiments from 2026-10-17. Parts A and B were integrated on 2026-11-08, with the shared `Makefile` and benchmark runner used for team comparisons. See the [submission guide](member_c_submission.md) for current commands and the [integration record](integration_2026-11-08.md) for the integration checks. The speedups below apply to the October setup.

The optimization was developed locally. All runs used Br=Bc=32 and one thread. Tile sizes, baseline compiler flags, and correctness tolerances were held fixed.

## Where the original implementation spent time

Streaming avoids allocating the N×N score matrix. It still computes every QKᵀ score and the full product with V. The original implementation used an ordinary dot product for each score, then incurred additional work per key tile for the maximum, exponential rescaling, running-state updates, and loop control. The score buffers in the initial tests were 64 KiB at (128, 64) and 1 MiB at (512, 64). The measurements did not include DRAM traffic or hardware cache misses, so they cannot establish a DRAM bottleneck.

The initial tests ran the algorithms separately, making differences of a few percent sensitive to system state. In interleaved runs, the original streaming kernel remained slower than the baseline on larger shapes with d=64 and was slightly faster on some shapes with d=32 or d=128.

1. Temporary instrumentation split the original kernel into score computation and the online update plus PV. Score computation accounted for about 59% of the instrumented runtime at (512, 64) and (2048, 64). The stage timings include instrumentation overhead.
2. The generated assembly used vector multiplication followed by sequential scalar accumulation. The accumulation dependency limited parallelism within each score.

The original kernel also computed alpha and rescaled u on every tile after the first. For random inputs, the running maximum increased on only 15.8% of those updates at N=512 and 6.0% at N=2048. Alpha was exactly 1 on the other updates, making the rescaling redundant.

## Implemented optimizations

### Transposed K tiles and eight-score accumulation

Each worker uses Bc×d extra floats to store the current key tile in transposed form. Eight independent accumulators compute scores for adjacent keys. At each feature, one Q value is multiplied by eight contiguous K values. Each score still accumulates features in order. Apple Clang emitted two four-lane `fmla.4s` instructions to update all eight scores in registers.

The kernel uses portable C++ without ARM-specific intrinsics. Other CPUs require their own vectorization and performance checks. The remainder of fewer than eight keys uses ordinary dot products. Tests cover Bc=7/8/9/31/32/33 and uneven tiles and feature dimensions.

Each query block repacks the K tiles it visits. The kernel timing includes that work. The implementation stores only the current transposed tile in scratch and uses no persistent auxiliary array.

### Rescaling when the maximum increases

The first tile starts with l and u set to zero. For later tiles, `m_new == m` gives alpha=1, so the kernel adds the new tile's contribution to the existing state. If `m_new > m`, it rescales both l and u using the original online softmax recurrence. The computation still visits every key and uses standard exp.

## Matched benchmarks and ablations

`scripts/benchmark_streaming.py` compares five implementations in one process.

| Variant | Implementation |
| --- | --- |
| baseline | Original baseline in `attention.cpp` |
| original | Frozen pre-optimization streaming kernel, now in `src/benchmarks/streaming_original.cpp` |
| rescale_only | Ordinary dot products with redundant rescaling skipped |
| packed_only | K packing and eight-score accumulation with rescaling on every subsequent tile |
| optimized | Both optimizations enabled |

All variants used `-O3 -std=c++17 -Wall -Wextra -Wpedantic` without fast-math or LTO. For each shape, the benchmark generated an independent NumPy FP64 reference from the same FP32 inputs and validated every variant before timing it. Buffers were preallocated. Each variant used five warmups and 15 samples, with short calls batched. The benchmark rotated and reversed the execution order across rounds. Timings included packing, initialization, and normalization. Raw samples, errors, scratch sizes, and environment details were recorded. To control for scratch capacity, `rescale_only` reserved the same capacity as `optimized` and left the K panel unused.

The tests ran on an Apple M4 Pro with macOS 26.6.2 and Apple Clang 17.0.0, without fixed CPU affinity or frequency. All 12 main shapes passed the independent FP64 checks. The table reports local warm-call measurements.

| N, d | Baseline ms | Original streaming ms | Optimized streaming ms | Speedup over original | Speedup over baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128, 64 | 0.264 | 0.262 | 0.168 | 1.56× | 1.57× |
| 512, 64 | 4.018 | 4.227 | 2.696 | 1.57× | 1.49× |
| 2048, 64 | 65.067 | 68.165 | 42.845 | 1.59× | 1.52× |
| 4096, 64 | 266.740 | 277.198 | 173.019 | 1.60× | 1.54× |
| 4096, 128 | 608.515 | 595.524 | 311.583 | 1.91× | 1.95× |

Across all 12 shapes, the speedup ranged from 1.31–1.91× over the original streaming kernel and 1.35–1.95× over the baseline. Both ratios use timings from the same interleaved benchmark.

The ablation at (2048, 64) gave the following results.

| Variant | Median ms |
| --- | ---: |
| original | 68.165 |
| rescale_only | 64.784 |
| packed_only | 43.721 |
| optimized | 42.845 |

The SIMD-friendly score layout accounted for most of the speedup. Skipping redundant rescaling alone cut runtime by about 5%. Instrumentation showed that the score and packing stage fell from about 41.46 ms to 18.98 ms, leaving the online update and PV as the larger remaining stage. Instrumentation can affect code generation and runtime, so these stage timings are diagnostic.

## Memory and correctness

Let `R=min(Br,N)`, `C=min(Bc,N)`, and `W=min(threads,ceil(N/R))`.

```text
scratch bytes = 4W × [RC + Rd + 2R + Cd + (W>1 ? 32 : 0)]
Q/K/V/O bytes = 16Nd
persistent auxiliary = 0
```

The optimized kernel adds C×d floats of scratch per worker. For `N=4096,d=64,Br=Bc=32`:

| Implementation | Scratch |
| --- | ---: |
| Baseline | 64 MiB |
| Original streaming, one thread | 12.25 KiB |
| Optimized streaming, one thread | 20.25 KiB |
| Optimized streaming, four-thread capacity | 81.5 KiB |

The workspace queries and test assertions account for the packed K panel. Scratch is preallocated, and workers have private packing buffers. A single worker still performs the full online recurrence for each query row.

The tolerance remains `abs(O-Oref) <= 1e-4 + 1e-3*abs(Oref)`. SIMD and FMA may change the last few bits relative to the original kernel. With the same implementation, tile sizes, and inputs, outputs must match bit for bit across thread counts.

The 111 full reference checks passed at both one and four threads. The maximum absolute error was `1.1178691438741908e-5` and the maximum relative Frobenius error was `1.458256254275443e-6`. The 201 serial API configurations, 601 multithreaded API configurations, and UBSan checks also passed.

The local logs for this phase are in `results/optimization/`.

- `correctness_serial.csv` covers all 12 shapes, edge cases, and special inputs for baseline and streaming.
- `correctness_openmp4.csv` records the full four-thread streaming reference checks.
- `api_serial.log` and `openmp_build.log` cover 201 and 601 API configurations, including SIMD tails, input protection, NaN-filled scratch, buffer reuse, and thread consistency.
- `ubsan.log` records the UBSan checks. See the [integration record](integration_2026-11-08.md) for the later LLVM 22.1.8 ASan run.
- `memory_serial.csv` and `memory_openmp4.csv` record the updated scratch counts.
- `comparison.csv` contains interleaved measurements of the five uninstrumented implementations across all 12 shapes.
- `profile.csv` contains the separately measured stage timings and counts of running-maximum changes.
- `streaming.s`, `streaming_original.s`, and `vectorization.txt` contain assembly and vectorization evidence.

## Reproduction and dependencies

For the current production build and shared tests, use the [Part C implementation guide](member_c.md). To run the historical comparison with the sources in the current layout, run these commands from the repository root. The default comparison uses three representative shapes. The `--full` run uses all 12.

```sh
.venv/bin/python scripts/benchmark_streaming.py
.venv/bin/python scripts/benchmark_streaming.py --full
.venv/bin/python scripts/profile_streaming.py
```

Use `--output results/optimization/another_run` with the benchmark script to keep separate comparison results. The default overwrites CSVs with the same names. The profiler compiles temporary instrumented copies and leaves production sources untouched.

The standalone Part C kernel requires `src/attention.h`, `src/streaming.cpp`, and the C++17 standard library. Multithreading also requires OpenMP. The existing repository supplies the CLI, baseline, and FP64 reference. The frozen kernel in `src/benchmarks/` serves the historical comparisons and is excluded from the normal build.

Part B owns the separate blocked materialized implementation. The local K packing in this record was implemented inside Part C. Part D can use the shared threading interface for tuning, affinity, scaling, and hardware profiling. The measurements above cover the Part C diagnostics.
