# Part C implementation and usage

Part C was originally integrated against upstream commit `855179728f95b755974f7a1ee5ecc6c462bb8771` (2026-11-07), which included Parts A and B and the shared benchmark and threading tools. See the [submission guide](member_c_submission.md) for contribution details and the [integration record](integration_2026-11-08.md) for validation. The [October optimization record](streaming_optimization.md) documents the original Part C experiments.

## Computation and implementation

All three algorithms compute `O = softmax(QKᵀ / sqrt(d)) V` for dense, unmasked, single-head attention with FP32 row-major tensors. Part A provides the baseline and an independent FP64 reference. Part B stores the full N×N score matrix. Part C streams tiles of Br query rows and Bc key/value rows.

Part C maintains a running maximum m, normalization sum l, and weighted vector u for each query row. It starts with l and u set to zero. When a later tile increases the running maximum, the kernel rescales both l and u by `alpha=exp(m-m_new)`. When the maximum stays the same, alpha equals 1 and the kernel skips the rescaling. After processing all keys, it writes u/l to the output.

The kernel packs the current K tile in transposed form into worker scratch and uses eight independent score accumulators to enable SIMD. It uses ordinary dot products for a remainder of fewer than eight keys. The timed kernel call includes packing, initialization, exponentiation, normalization, and threading overhead. The implementation uses standard exp, with no fast-math, cross-call K cache, or allocation inside the kernel. FMA may affect the last few bits. The acceptance criterion is `abs(O-Oref) <= 1e-4 + 1e-3*abs(Oref)`.

## Memory contract

Let R=min(Br,N), C=min(Bc,N), and W=min(threads,ceil(N/R)).

| Component | Floats per worker |
| --- | ---: |
| Shared tile for scores and exponentiated weights | RC |
| Weighted vector u | Rd |
| m and l | 2R |
| Current transposed K tile | Cd |
| Padding between workers (W>1) | 32 |

```text
streaming scratch bytes = 4W × (RC + Rd + 2R + Cd + (W>1 ? 32 : 0))
baseline / current blocked scratch bytes = 4N²
Q/K/V/O bytes = 16Nd
persistent auxiliary arrays = 0
```

The workspace query returns a float count, which the runner converts to bytes. The count covers preallocated capacity, including unused space in partial tiles and padding between workers. Thread stacks, runtime and validation memory, and DRAM traffic fall outside this calculation. With fixed tiles, d, and thread count, Part C scratch avoids quadratic growth in N.

For `N=4096,d=64,Br=Bc=32`, Parts A and B each require 64 MiB of scratch. Part C requires 20.25 KiB with one thread and 81.5 KiB of capacity with four threads. Q/K/V/O total 4 MiB. `scripts/memory_report.py` requires all three algorithms and checks 36 queries across 12 shapes.

## Threading and SIMD

One worker processes each query block and owns the entire recurrence for each row. Q/K/V are shared read-only, and each worker owns its output rows and scratch. Scratch padding is 128 B. Very short output rows can still share a cache line in the common output layout. With static scheduling, small N or a small number of query blocks can leave workers idle.

Part C uses the shared Makefile's ISA and SIMD settings and has its own microkernel. The no-vectorization build disables automatic vectorization and SLP. Part C uses portable C++ without explicit SIMD intrinsics. Part B's reduction and Part C's per-feature accumulation also differ, so the runtime difference includes more than the effect of online softmax. The historical diagnostic scripts isolate Part C's own optimizations.

## Running the current checkout

From the repository root, use the Python environment set up in the [submission guide](member_c_submission.md).

```sh
make PYTHON=.venv/bin/python test test-streaming-api
make PYTHON=.venv/bin/python attention attention-novec
.venv/bin/python scripts/bench.py --threads 1 --shapes 128x64,512x64 --rounds 2 --novec-binary ./attention-novec --csv results/member_c_bench.csv
.venv/bin/python scripts/memory_report.py --csv results/member_c_memory.csv
```

On macOS with Anaconda libomp, build and check the four-thread configuration with:

```sh
make PYTHON=.venv/bin/python OPENMP=1 LIBOMP=/opt/anaconda3 attention test-streaming-api
OMP_DYNAMIC=FALSE .venv/bin/python test/check.py --algorithm all --tiles --threads 4 --csv results/member_c_quick_t4.csv
```

With GCC on Linux, omit `LIBOMP`. The Makefile still accepts `OMP_PREFIX` as an alias. The submission guide covers the full correctness suite, tuning, scaling, and sanitizers.

## Integration with Parts B and D

Part B is on `main`. The shared reference checks cover all three algorithms, dimensions that do not divide evenly into tiles, tiles of size 1 or larger than N, and odd thread counts or counts greater than the number of query blocks. The Part C kernel builds independently of Parts B and D.

Part D can use `scripts/bench.py`, `test/check.py --tune`, and the thread-scaling targets. For the team experiments, select tile sizes on the target machine, record thread affinity and CPU frequency, and run the full performance grid. See the [benchmarking guide](benchmarking.md) for the current workflow.
