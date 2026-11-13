# Parts A, B, and C integration record (2026-11-08)

## Integration status

Part C was based on upstream commit `855179728f95b755974f7a1ee5ecc6c462bb8771` (2026-11-07, `add blocked threading, and scaling sweep`), which already included Part B.

A dated Git stash saved pre-integration edits to shared files. A separate local archive, `dist/pre_integration_2026-11-08/local_changes.zip`, saved the full source tree. A byte-for-byte comparison confirmed that `streaming.cpp` matched the backup. The integration preserved the algorithm and tolerance.

## Changes integrated

- Reused upstream `attention.cpp`, `blocked.cpp`, `simd.h`, and `bench.py`.
- Added Part C's `--memory` query alongside the upstream `--build-info` command in `main.cpp`.
- Added the overflow-input rejection check to the upstream tuning, thread consistency, and correctness framework.
- Added Part C API and sanitizer test targets to the upstream build, which already provided native ISA, SIMD/no-vectorization, threading, and benchmarking targets.
- Kept `LIBOMP` as the macOS OpenMP setting and `OMP_PREFIX` as an alias, and added the runtime library search path. The Makefile quotes `PYTHON` so interpreter paths can contain spaces.
- Fixed `test-threads` to return a failing status from the Python checker. A test that injected a failing checker confirmed the nonzero exit status.
- Updated the reproduction script to use `check.py` for correctness, `bench.py` for timing, and the memory report to verify all three algorithms.
- Updated the submission inventory, README, design notes, and packaging baseline. The October optimization report uses its own build conditions and benchmark results.

## Validation results

The logs for this integration are in the local directory `results/integration_2026-11-08/`. The file `validation_summary.json` records the results and source hashes.

| Check | Result | Evidence |
| --- | --- | --- |
| All 12 main shapes and edge cases, one thread | A: 21, B: 111, C: 111 passed | `correctness_serial.csv` / `.log` |
| Same suite with four OpenMP threads for B and C and one thread for A | A: 21, B: 111, C: 111 passed. B and C each matched their single-thread output bit for bit. | `correctness_openmp4.csv` / `.log` |
| Full suite with vectorization disabled | A: 21, B: 111, C: 111 passed | `correctness_novec.csv` / `.log` |
| Direct Part C API tests | 201 serial and 601 multithreaded configurations passed | `build_serial.log`, `build_openmp.log` |
| Part B and C threading edge cases | 40 passed at 2, 3, 4, 14, or 64 threads, including more threads than query blocks, N=1, and uneven tile boundaries | `thread_edges.log` |
| Multithreaded build with UBSan | Reference checks and 601 Part C API configurations passed | `ubsan.log` |
| Thread test failure propagation | Make returned a failure when the checker was forced to fail | `thread_failure_guard.log` |
| Memory queries for all three algorithms | 36 passed with one thread and 36 with four threads | `memory_serial.csv`, `memory_openmp4.csv` |
| Upstream tile tuning | Validated and timed 36 candidates across two shapes, nine tile configurations, and both B and C. The selected configurations passed reference checks. | `tuning.csv`, `tuned-tiles.json`, `tuning.log` |
| Updated `bench.py` | Two representative shapes, vectorized and non-vectorized builds, one and four threads, and two interleaved rounds. Validation preceded timing. | `bench.csv`, `bench.machine.json`, `bench.log` |
| Updated reproduction script | Both the serial and four-thread OpenMP runs passed | `reproduction_serial.log`, `reproduction_openmp4.log` |

Part C's maximum absolute error in the full reference check was `1.1178691438741908e-5`. The elementwise tolerance remained `1e-4 + 1e-3*abs(Oref)`. With the new SIMD settings, Part B's reduction can round differently from a sequential sum and also satisfies this criterion.

The tuning run selected 64×64 tiles for both B and C using the upstream geometric mean of normalized runtimes over two shapes. This result applies to that local run. The shared default stayed at 32×32. The benchmark successfully loaded the selected configuration from JSON.

### Follow-up validation

A standalone LLVM 22.1.8 toolchain ran ASan and UBSan natively on an Apple Silicon Mac. A deliberately out-of-bounds probe reported a heap buffer overflow. The project then passed all 12 main shapes and the edge cases, with 21 reference checks for A and 111 each for B and C. B and C each produced bitwise identical outputs at one and four threads. All 601 Part C API configurations passed, and neither sanitizer reported an issue during the project tests. A Linux ARM64 container also passed the edge cases and 601 API checks. The raw logs are in the local directory `results/asan_retest_20261108_210821/`.

The extended local benchmark covered 12 shapes and 96 configurations. It measured requested thread counts of 1, 2, 4, 8, and 14 for 512×64 and 4096×64. That tuning run selected 32×16 for B and 64×16 for C. The raw results are in the local directory `results/extended_test_20261108_203128/`.

## Reproducing the integration checks

At the original Part C commit, the first command ran serially and the second used Anaconda libomp on macOS. With GCC on Linux, `LIBOMP` could be omitted.

```sh
bash scripts/run_member_c.sh
OPENMP=1 LIBOMP=/opt/anaconda3 OMP_DYNAMIC=FALSE THREADS=4 bash scripts/run_member_c.sh
```

See the [submission guide](member_c_submission.md) for commands that use the current directory layout and for contribution details.

## Scope of the measurements

Part C shares the interface, build, reference checks, memory query, tile configuration, and benchmark runner with Parts A and B.

The measurements in this record come from the local Mac. Target-server performance and dedicated data-race testing were still pending at the end of this integration. See the [benchmarking guide](benchmarking.md) for the later team workflow.

The three implementations differ in SIMD arithmetic and packing, so the measured family-level runtime differences combine several effects. Experiments with a shared microkernel or controlled ablations must preserve the common correctness, memory accounting, and timing rules.
