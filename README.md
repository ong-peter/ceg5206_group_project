# CPU attention

Part A: a single-thread FP32 baseline and an independent FP64 NumPy reference.
Part B: blocked materialized attention with SIMD and OpenMP query-block workers.
Part C: streaming attention with online softmax, packed K tiles, independent
worker scratch, API tests, and explicit memory accounting.
Computes `softmax(Q K^T / sqrt(d)) V` with one head and no mask.

The local integration is based on upstream `8551797` (7 November 2026).
Start with the [C handoff](docs/member_c_submission.md) for contribution ownership
and portable commands, or the [integration record](docs/integration_2026-11-08.md)
for checks against the current A/B implementations. With dependencies installed,
`bash scripts/run_member_c.sh` builds and validates all three families and runs
a small comparison using the shared `bench.py`.

## Run

Requires C++17, Make, and Python with NumPy. CI uses Python 3.12;
local validation also works with Python 3.13.9 in Conda base. The NumPy version
is pinned in `requirements.txt` and shared with CI.

For an existing Conda base environment:

```sh
conda activate base
python -m pip install -r requirements.txt
python -c "import sys, numpy; print(sys.executable); print(sys.version); print('NumPy', numpy.__version__)"
export PYTHON="$CONDA_PREFIX/bin/python"
make test       # Available kernels and edge/tile checks
make sanitize   # AddressSanitizer and UndefinedBehaviorSanitizer
make tsan       # ThreadSanitizer on the OpenMP kernels (see Data races)
make full       # All 12 core shapes and edge cases; errors in results.csv
```

Setting `PYTHON` ensures Make uses the same interpreter as the active Conda
environment. For another environment, set `PYTHON` to its Python executable.
`check.py` checks correctness only: core shapes are N in {128, 512, 2048, 4096}
and d in {32, 64, 128} with `--full`, otherwise (128, 64) and (512, 64).
Timing is done by `bench.py` (see Benchmark ladder) and by tile tuning.

Interface notes are in `attention.h`; test options: `"$PYTHON" check.py --help`.

### Data races (ThreadSanitizer)

```sh
make tsan                    # or: bash scripts/tsan_check.sh
make tsan THREADS="2 7" API=0
```

Results go to `results/tsan_<time>/triage.txt`; the exit status is non-zero on
failures or reports that need review. `scripts/tsan_check.sh` builds separate TSan binaries of the OpenMP kernels and runs the C
API test and `check.py --tiles` at the `make test-threads` thread counts in
parallel. It prefers Clang with LLVM libomp and Archer (on Ubuntu,
`apt install clang-21 libomp-21-dev`), which makes OpenMP synchronization
visible to TSan, so any report is a finding. Without them it falls back to GCC;
libgomp then produces fork/join false positives, which `scripts/tsan_triage.py`
separates from reports that need review. Set `TOOLCHAIN=clang|gcc` to force one,
`THREADS="2 7"` to choose thread counts, `API=0` to skip the API test (hours
under GCC), and `FULL=1` for all 12 core shapes. The TSan build does not affect
`attention` or the benchmarks, which keep the default compiler.

## Build flags and vectorization

Every kernel, the baseline included, is built with `-O3` and `-march=native`
(`-mcpu=native` on ARM), so they all share ISA settings; use `ARCH_FLAGS=` for a
portable build. `-fopenmp-simd` enables the `omp simd` loop hints in `simd.h`
without needing the OpenMP runtime, so single-thread builds vectorize the dot
product reduction too. `make OPENMP=1` adds threading. A flag change triggers
a rebuild, and `./attention --build-info` prints the flags the binary was built
with. `make attention-novec` builds the vectorization ablation: same flags, but
with auto-vectorization and SIMD hints disabled.
`make vec-report` lists vectorized loops (GCC `-fopt-info`, Clang `-Rpass`).

The Makefile detects the compiler and CPU. On Apple silicon it uses `-mcpu=native`
and the Clang ablation flags. For `OPENMP=1`, run `brew install libomp`; the
Makefile then adds `-Xpreprocessor -fopenmp -lomp`. On x86 CPUs with AVX-512,
`make OPENMP=1 bench-width` compares 256- and 512-bit vectors (`VEC_WIDTH=512`
switches the main build). ARM NEON is fixed at 128 bits, so this target is x86-only.
An existing libomp installation can be selected with `LIBOMP=/opt/anaconda3`
on this Mac; the previous `OMP_PREFIX` variable remains an alias. `PYTHON` may
point to an interpreter whose path contains spaces.

## Benchmark ladder

```sh
make clean && make OPENMP=1 bench-quick   # (512, 64) and (2048, 64)
make OPENMP=1 bench                       # all 12 core shapes
```

`bench.py` compares stages cumulatively on the same inputs: 1 (baseline),
1+2 (blocked), 1+2+3 (streaming), and 1+2+4 / 1+2+3+4 (threaded, physical cores
by default), plus a 1-thread no-vectorization ablation of every family. It
checks each variant against the FP64 reference before timing it. The output,
`ladder.csv`, includes the median and IQR, speedup over the baseline, thread
speedup, GFLOP/s, scratch bytes and build flags. Pass `--tile-config tuned-tiles.json`
to use the tuned tiles (the Make targets do this when the file exists), and
`--rounds R` to interleave variants over R rounds.

### Threads (Task 4)

```sh
make clean && make OPENMP=1 test-threads   # outputs must match one thread bit for bit
make OPENMP=1 bench-scaling                # 1, 2, 4, ... cores on 512x64 and 4096x64
```

`--threads` takes a list such as `1,2,4,8`, and `--shapes 512x64,4096x64` picks
shapes. Each run records `OMP_PLACES`/`OMP_PROC_BIND` per row and saves machine
details to `<csv>.machine.json`. See [C's design notes](docs/member_c.md) for its
worker ownership and scratch layout, and the integration record for actual checks.

`ladder.csv` also has `peak_rss_bytes`: the process peak resident set of one
extra untimed run per variant, measured with `/usr/bin/time` (GNU `-f %M` on
Linux, `-l` on macOS; empty if unavailable). It includes input, output, scratch
and runtime memory (about 4.5 MiB for small shapes), so it supplements
`scratch_bytes` rather than replacing it.

### Stage profile

```sh
make profile-stages   # results/stages.csv: 512x64 and 4096x64, one thread
```

`scripts/profile_stages.py` times QKᵀ, softmax and PV separately for the
baseline (its public stage functions) and the blocked kernel (timers in a
temporary copy of `blocked.cpp`), and splits fused streaming into score+K
packing and online update+PV. Outputs are checked against the FP64 reference.
Timers add overhead, so the instrumented total is reported with the stages;
use `bench.py` for timings. `--shapes` picks other shapes.

### Plots

```sh
make plots   # PNGs in plots/ from ladder.csv, scaling.csv, tuning-results.csv, results/stages.csv
```

`scripts/plot_results.py` draws runtime, allocated scratch and process peak RSS
against N (one panel per d), single-thread speedup over the baseline for every
shape, the no-vec ablation (SIMD gain per family, and tiling gain with SIMD
disabled), thread scaling against ideal, the tuning grid relative to 32x32 with
the selected tiles outlined, and the per-stage breakdown. Figures whose CSV is
missing are skipped; paths can be changed with `--ladder`, `--scaling`,
`--tuning`, `--tiles`, `--stages` and `--out`. Run `make OPENMP=1 bench` (all 12
shapes plus the ablation), `make OPENMP=1 bench-scaling` and
`make profile-stages` first. Scratch is allocated capacity, not measured memory
traffic.

## Automated tile tuning

Once `blocked.cpp` or `streaming.cpp` is implemented and compiled, run:

```sh
python check.py --algorithm all --tune --tiles --csv tuning-results.csv
make OPENMP=1 bench   # picks up tuned-tiles.json
```

Tuning checks correctness before timing each of the nine Br/Bc pairs in
{16, 32, 64} on (512, 64) and (2048, 64), using one thread, five warm-ups and
15 samples. All candidates must pass. It chooses one pair per optimized family
by minimizing the geometric mean of median times relative to 32x32 on each
shape, giving both shapes equal weight. The baseline is not tuned.

Raw timing samples and errors are saved as `tuning` rows in the CSV. Selected
tiles, the selection rule, and wall-clock tuning costs (including validation
and process/file overhead) are saved to `tuned-tiles.json`. The tuning command
then runs the usual correctness suite with the selected tiles; `--tiles` also
checks edge cases across the tile sweep.
Use `--tuning-output PATH` to change the JSON output location.

`--tile-config` reuses the saved pair for each optimized family across shapes;
it takes precedence over `--br`/`--bc`. It can also be combined with `--threads`
once threaded kernels exist. Retune after changing kernels, compiler settings,
or hardware. Tuning requires at least one compiled optimized kernel.

## Validation and results

The reference converts the exact FP32 inputs to FP64. Each element must satisfy `abs(O - Oref) <= 1e-4 + 1e-3*abs(Oref)`; NaN/Inf fail. Tests cover known outputs, random inputs, tails and concentrated distributions. CSV rows name the actual kernel and configuration. Timed rows include its workspace size, quartiles and raw samples.

Historical Part A record (11 October 2026): 21 baseline cases, 198 Python online-recurrence checks and the sanitized suite passed. Maximum baseline absolute error was 8.28e-6. Current joint A/B/C validation is recorded separately in the integration record.

Historical Part A environment: Apple M4, macOS 26.6.2, Apple Clang 21.0.0, Python 3.12.14, NumPy 2.3.5. Flags are in Makefile. Timing follows correctness checks, uses preallocated buffers, five warm-ups and 15 samples, and excludes allocation, validation, I/O and batch calibration. Samples are per-call milliseconds with reused buffers; scheduling and frequency are OS-controlled. These are initial measurements, not current joint results.

[Attention Is All You Need](https://arxiv.org/abs/1706.03762), Section 3.2.1; [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), Algorithm 3.

AI tools were used for coding assistance.
