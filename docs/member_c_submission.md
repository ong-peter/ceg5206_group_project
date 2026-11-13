# Part C submission and integration with Parts A, B, and D

The original Part C submission was prepared on 2026-11-08 against upstream commit `855179728f95b755974f7a1ee5ecc6c462bb8771`. Upstream provided Part B, the shared benchmark, the ISA and SIMD settings, the no-vectorization build, and the thread-scaling targets.

## 1. Part C contributions

The table uses paths from the current repository layout.

| File | Part C contribution |
| --- | --- |
| `src/streaming.cpp` | Online softmax, K packing, eight-score SIMD-friendly accumulation, partial tiles, private worker scratch, OpenMP, and argument checks |
| `test/streaming_test.cpp` | 201 serial and 601 multithreaded API configurations, covering input protection, nonzero and NaN scratch, buffer reuse, SIMD tails, overflow, and thread consistency |
| `scripts/run_member_c.sh` | Original reproduction script for the shared build, three-algorithm correctness checks, a small benchmark, and the memory report |
| `scripts/memory_report.py` | 36 memory queries covering 12 shapes across Parts A, B, and C |
| `src/benchmarks/streaming_original.cpp` | Frozen pre-optimization Part C kernel for ablations, excluded from the normal build |
| `src/benchmarks/compare.cpp`, `scripts/benchmark_streaming.py` | Historical comparisons of the original, optimized, and ablated Part C variants under a separate fixed build configuration |
| `scripts/profile_streaming.py` | Diagnostic profiling with temporary instrumented source copies |
| `scripts/package_member_c.py` | Original submission packager with an explicit file list, a patch against the original upstream base, and a manifest with hashes |
| `docs/member_c.md`, `docs/member_c_submission.md` | Design, memory accounting, run instructions, and contribution boundaries |
| `docs/streaming_optimization.md` | Historical optimization record dated 2026-10-17, including the experimental setup and scope |
| `docs/integration_2026-11-08.md` | Parts A, B, and C integration and validation record |

| Shared file | Part C edits | Upstream components |
| --- | --- | --- |
| `src/attention.h` | Implementation status and comments on scratch and packing | Common API and configuration |
| `src/main.cpp` | `--memory` query | `--build-info`, three-algorithm dispatch, timing, and I/O |
| `test/check.py` | Rejection case for overflow from large finite inputs | FP64 reference, tuning, thread consistency, and correctness framework |
| `Makefile` | Part C API target, optional sanitizers, libomp compatibility, support for Python paths with spaces, and thread-test failure propagation | ISA and SIMD settings, no-vectorization build, benchmark, scaling, and vectorization report targets |
| `README.md`, `.gitignore` | Part C navigation and integration status, plus ignore rules for Part C outputs | Upstream usage instructions and AI assistance disclosure |

The original submission also reused the baseline, blocked kernel, SIMD hints, benchmark runner, requirements, and validation workflow from upstream.

## 2. Quick start

Run these commands from the repository root. The build requires a C++17 compiler, Make, and Python 3.12 or 3.13.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make PYTHON=.venv/bin/python test test-streaming-api
```

The default build runs serially and needs no OpenMP runtime. The [Part C implementation guide](member_c.md) also shows a small vectorized versus non-vectorized benchmark and the 36 memory queries. The original reproduction script produced `results/member_c_quick.csv`, `results/member_c_bench.csv`, `results/member_c_memory.csv`, and logs for the environment and build flags. Its original entry points are documented in the [integration record](integration_2026-11-08.md).

On macOS with Anaconda libomp, build and test the OpenMP kernels with:

```sh
OMP_DYNAMIC=FALSE make PYTHON=.venv/bin/python OPENMP=1 LIBOMP=/opt/anaconda3 test-threads test-streaming-api
```

With GCC on Linux, use:

```sh
OMP_DYNAMIC=FALSE make PYTHON=.venv/bin/python OPENMP=1 test-threads test-streaming-api
```

For other macOS installations, set `LIBOMP` to the installed prefix, such as `/opt/homebrew/opt/libomp`. `OMP_PREFIX` is a compatibility alias. Changing build flags triggers a rebuild. Use `./attention --build-info` to verify the active flags.

## 3. Full correctness, tuning, and benchmarks

```sh
mkdir -p results
make PYTHON=.venv/bin/python attention
./attention --list
.venv/bin/python test/check.py --algorithm all --full --tiles --csv results/full_correctness.csv
```

Check that the list includes `baseline`, `blocked`, and `streaming`. The `all` option only tests compiled implementations. Use `scripts/bench.py` for timing.

```sh
.venv/bin/python test/check.py --algorithm all --tune --tiles --tuning-output results/tuned-tiles.json --csv results/tuning.csv
make PYTHON=.venv/bin/python attention attention-novec
.venv/bin/python scripts/bench.py --full --threads 1 --rounds 3 --tile-config results/tuned-tiles.json --novec-binary ./attention-novec --csv results/ladder.csv
```

To check four-thread correctness and run a small scaling experiment on macOS:

```sh
make PYTHON=.venv/bin/python OPENMP=1 LIBOMP=/opt/anaconda3 attention attention-novec test-streaming-api
OMP_DYNAMIC=FALSE .venv/bin/python test/check.py --algorithm all --full --tiles --threads 4 --csv results/full_t4.csv
make PYTHON=.venv/bin/python OPENMP=1 LIBOMP=/opt/anaconda3 test-threads
OMP_PLACES=cores OMP_PROC_BIND=close .venv/bin/python scripts/bench.py --threads 1,2,4 --shapes 512x64,4096x64 --rounds 3 --tile-config results/tuned-tiles.json --csv results/scaling.csv
```

With GCC on Linux, omit `LIBOMP`. For a full scaling experiment, choose the upper thread count based on the target machine's physical cores. See the [benchmarking guide](benchmarking.md) for the team workflow.

```sh
./attention 4096 64 --memory --algorithm streaming --br 32 --bc 32
.venv/bin/python scripts/memory_report.py --csv results/memory.csv
make PYTHON=.venv/bin/python sanitize
```

The sanitizer target enables ASan and UBSan by default. On macOS, a standalone LLVM 22.1.8 toolchain passed the full reference checks and 601 Part C API configurations. A Linux ARM64 container also passed the edge and API checks. See the [integration record](integration_2026-11-08.md). To run only UBSan, set `SANITIZERS=undefined`.

```sh
make PYTHON=.venv/bin/python OPENMP=1 LIBOMP=/opt/anaconda3 SANITIZERS=undefined sanitize
```

## 4. Coordination with Parts B and D

- Parts B and C use the same `src/attention.h` interface. The common runner converts the scratch float count to bytes. Parts A and B allocate the full N×N matrix. Part C's scratch includes the packed K tile, as described in the [implementation guide](member_c.md).
- One worker handles the entire recurrence for each Part C row. Part D can use the existing OpenMP entry point and should record dynamic team settings, affinity, physical cores, and CPU frequency policy in the experiment setup.
- All algorithms pass the same FP64 reference checks before timing. Timings include packing, initialization, and threading overhead. Differences in arithmetic and SIMD across the three implementations affect family-level comparisons.
- The historical five-variant ablation isolates Part C optimizations. Use `scripts/bench.py` and `test/check.py` for the team's comparisons, no-vectorization runs, tuning, and scaling.
- The original Part C validation covered all shapes and scaling for two shapes on a local Mac. Target-server tile selection, performance measurements, and race detection were tracked separately in the team experiments.

## 5. Original submission archive

The following command was used in the original Part C checkout (`a0f6fda`). The packager still expects that checkout's directory layout.

```sh
python3 scripts/package_member_c.py
```

The resulting `dist/member_c_submission.zip` contains a runnable A/B/C snapshot, a Part C patch against `8551797`, an attribution and SHA-256 manifest, and local validation evidence. The file list excludes environments, build outputs, Git metadata, and files outside the manifest. Use `--code-only` if the original local evidence is unavailable.

Before applying the patch, run `git apply --check` on the matching baseline. Use the upstream versions of shared files when integrating with later changes. The original local backup was stored in `dist/pre_integration_2026-11-08/`.
