# CPU attention

Part A: a single-thread FP32 baseline and an independent FP64 NumPy reference.
Computes `softmax(Q K^T / sqrt(d)) V` with one head and no mask.

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
"$PYTHON" check.py --algorithm all --full --tiles  # All 12 shapes, no timings
```

Setting `PYTHON` ensures Make uses the same interpreter as the active Conda
environment. For another environment, set `PYTHON` to its Python executable.
For the full benchmark, run `make full`; this overwrites `results.csv` and
times all 12 core shapes: N in {128, 512, 2048, 4096} and d in {32, 64, 128}.
Without `--full`, `--bench` times only (128, 64) and (512, 64).
Edge cases are checked for correctness but are not timed.

Interface notes are in `attention.h`; test options: `"$PYTHON" check.py --help`.

## Automated tile tuning

Once `blocked.cpp` or `streaming.cpp` is implemented and compiled, run:

```sh
python check.py --algorithm all --tune --tiles --csv tuning-results.csv
python check.py --algorithm all --tile-config tuned-tiles.json --full --bench --csv results.csv
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
checks edge cases across the tile sweep. `--tune --full --bench` additionally
benchmarks all core shapes using the selected tiles in that same invocation.
Use `--tuning-output PATH` to change the JSON output location.

`--tile-config` reuses the saved pair for each optimized family across shapes;
it takes precedence over `--br`/`--bc`. It can also be combined with `--threads`
once threaded kernels exist. Retune after changing kernels, compiler settings,
or hardware. Tuning requires at least one compiled optimized kernel.

## Validation and results

The reference converts the exact FP32 inputs to FP64. Each element must satisfy `abs(O - Oref) <= 1e-4 + 1e-3*abs(Oref)`; NaN/Inf fail. Tests cover known outputs, random inputs, tails and concentrated distributions. CSV rows name the actual kernel and configuration. Timed rows include its workspace size, quartiles and raw samples.

On 11 October 2026, 21 baseline cases, 198 Python online-recurrence checks and the sanitized suite passed. Maximum baseline absolute error was 8.28e-6. B/C correctness checks require their implementations.

Recorded environment: Apple M4, macOS 26.6.2, Apple Clang 21.0.0, Python 3.12.14, NumPy 2.3.5. Flags are in Makefile. Timing follows correctness checks, uses preallocated buffers, five warm-ups and 15 samples, and excludes allocation, validation, I/O and batch calibration. Samples are per-call milliseconds with reused buffers; scheduling and frequency are OS-controlled. These are initial measurements.

[Attention Is All You Need](https://arxiv.org/abs/1706.03762), Section 3.2.1; [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), Algorithm 3.

AI tools were used for coding assistance.
