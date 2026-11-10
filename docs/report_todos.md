# Running the report TODO experiments

Run these on the original Ubuntu i7-11700K benchmark machine for report evidence.
The Windows smoke run verifies that the scripts and compiled diagnostic kernels
work; its timings are not replacements for the Ubuntu results.

The report's R1--R6 identifiers are the experiment names used below. S1--S6
are submission checks requiring the team's actual information. Scripts never
remove TODO markers or insert experimental conclusions into the report.

## Setup

From the repository root, activate the Python environment used for the final
benchmarks, install the existing pinned dependencies if needed, and select it:

```bash
python -m pip install -r requirements.txt
export PYTHON="$(command -v python)"
```

Use a C++17 compiler with OpenMP. The default build uses
`c++ -O3 -std=c++17 -march=native -fopenmp-simd -fopenmp`, with no fast-math.
Set `CXX` or pass `--cxx` to choose a compiler. Overrides are recorded:

```bash
bash scripts/run_review_todos.sh --plan --only all
```

Prepare the machine as described in [benchmarking.md](benchmarking.md): idle
system, performance governor, turbo disabled, and Linux `perf` installed.
For system-wide DRAM counters, the existing benchmark setup uses:

```bash
sudo sysctl kernel.perf_event_paranoid=0
```

Restore the machine's prior settings afterward as described in that guide.
The runner records the frequency policy before and after, but does not change
it. Inspect `manifest.json`; a recorded policy is not proof of the achieved
clock. Also inspect `counter_coverage.json` and `counters.log` for unavailable
events. Unsupported optional groups are recorded as missing, not as zero.

## Quick validation before collecting results

```bash
bash scripts/run_review_todos.sh --only R2,R3,R4,R5,R6 --smoke
bash scripts/run_review_todos.sh --only R1 --smoke
```

Smoke mode uses small/tail shapes, 1 and 2 threads by default, one round and at
most three timed samples. Its output directory starts with `review_smoke_`.
It is not a performance experiment for the report. R1 still needs Linux perf.

The existing output directory must not be reused. By default each invocation
creates `results/review_<UTC timestamp>/`; use `--out NEW_DIRECTORY` if desired.
The runner stops on a failed compile, reference check, changed timed output or
thread-invariance failure and preserves partial evidence and an error manifest.
To retry a failed task, use `--only Rn` with a new output directory.

## R1: required corrected counter measurements

First start an independent clock monitor in terminal A, from the repository
root. Use the same path in terminal B. `results/` already exists in this project:

```bash
sudo turbostat --quiet --show Time_Of_Day_Seconds,Busy%,Bzy_MHz --interval 1 \
  --out results/review_clocks.txt
```

If that turbostat version does not expose `Time_Of_Day_Seconds`, omit that name
and retain the startup time and elapsed intervals. Keep this monitor running
through the experiment. Use a new clock-log filename for each session.

In terminal B:

```bash
bash scripts/run_review_todos.sh --only R1 --threads 1,8 \
  --clock-log results/review_clocks.txt
```

After the experiment, stop turbostat with Ctrl+C and keep its completed log
with the result folder. The runner associates the supplied file in its
manifest; supplying a path does not start or validate a clock monitor.

R1 builds an isolated copy of the current sources and runs full FP64-reference,
edge and tile checks at the requested thread counts before collecting counters.
The baseline stays single-threaded. The counter shapes are `(128,128)`,
`(512,64)`, `(4096,64)` and `(4096,128)`, with saved `64x32` tiles from:

```text
results/benchmark_20261109T113037Z/tuned-tiles.json
```

Use `--tiles PATH` if that file is elsewhere. Outputs:

- `R1/correctness_t*.csv` and `.log`: reference validation.
- `R1/counters.csv`: matched cache-pass timing/cycles, IPC, caches, DRAM and
  separate timings for each optional pass.
- `R1/counters.passes.jsonl`: UTC boundaries, full/control raw counts, call
  counts and runner output for each pass; align these with turbostat.
- `R1/counter_coverage.json`: missing standard counter groups.

Update Table 4 and affected text only after reviewing coverage and independent
clocks. `ghz_per_thread` remains an approximate average-cycle/median-time
ratio. Separate passes and control subtraction introduce noise; negative or
very small differences are not evidence of negative traffic or exact zero.
This rerun alone does not establish an L3-latency cause.

## R2: query-tile/cache hypothesis

```bash
bash scripts/run_review_todos.sh --only R2 --threads 1,8 --cache-counters \
  --clock-log results/review_clocks.txt
```

The timing experiment interleaves blocked and streaming at `Br=32,64,128,256`,
`Bc=32`, on `(4096,64)` and `(4096,128)`, using 1 and 8 threads. With
`--cache-counters`, each tile also gets a counter profile in `R2/br*/`.

Optional CPU-specific events can be requested after checking `perf list` on
that machine. On the i7-11700K used here, `l2_rqsts.pf_miss` does not exist
(there is no direct L2 hardware-prefetch miss event), so use events that
appear in `perf list`. Start a new clock monitor (e.g. `review_clocks_R2b.txt`),
then run:

```bash
bash scripts/run_review_todos.sh --only R2 --threads 1,8 --cache-counters \
  --events l2_rqsts.demand_data_rd_miss,l2_lines_out.useless_hwpf,ocr.hwpf_l2_data_rd.any_response,cycle_activity.stalls_l2_miss \
  --clock-log results/review_clocks_R2b.txt
```

These cover demand L2 misses, unused L2 hardware prefetches, L2 hardware-prefetch
requests and L2-miss stalls. Do not repeat events from the standard groups
(e.g. `l2_rqsts.miss`, `l2_rqsts.references`); the script rejects duplicates. Keep to about four extra events: with hyper-threading enabled
each core has about four general-purpose counters, and `ocr.*` events share a
limited set of offcore registers. Check the set first with
`perf stat -e <events> -- true`.

The extra events are checked as one group: if any name is unsupported, the
whole group is skipped and logged. Afterwards, confirm `extra_group_present` is
`true` in each `R2/br*/counter_coverage.json`, then inspect
`extra_counts_json` and `counters.log`. Choose events for demand, prefetch or
stalls that the CPU actually supports. The script does not invent unsupported
event mappings. Compare scratch bytes, runtime, scaling and miss counts. Larger
tiles improve reuse but change scratch and parallelism too; a speedup alone
does not prove a cache-latency mechanism.

## R3: matched comparison and stage diagnostics

```bash
bash scripts/run_review_todos.sh --only R3 --threads 1,8
```

One build enables `BLOCKED_PACK_K`, leaving reordered PV enabled. It compares
packed blocked and streaming in rotated interleaved rounds with identical
`64x32` tiles, at `N=2048,4096` and `d=32,64,128`. Threaded output must match
the same kernel/build on one thread, bit for bit.

`R3/stages/stages.csv` separately measures serial score/packing, softmax and PV
for packed blocked, and score/packing versus online-update/PV for streaming.
These temporary instrumented builds validate against FP64. Their intrusive
times must not replace uninstrumented `summary.csv` results. Even after this
experiment, the residual gap includes normalization/update differences; the
study does not isolate pure score-buffer traffic cost.

## R4: numerical-error attribution

```bash
bash scripts/run_review_todos.sh --only R4 --seeds 1,2,3
```

This is an **untimed correctness experiment**. Generated copies form a full
2x2x2 comparison, with no production kernel changes:

| Factor | Choices |
| --- | --- |
| Score loop | Existing packed eight-score loop / existing scalar score path |
| Normalizer | Tile-local sums / sequential accumulation into the running total |
| Maximum schedule | Existing online updates / first determine the final maximum with the same score loop, then recompute scores using that maximum |

The final-maximum variant performs an extra score pass to remove changes of
normalization scale. It is a numerical diagnostic, not a proposed faster kernel.
All variants keep the same FP32 types, exponential and acceptance tolerance.
Baseline and default blocked outputs are also measured.

The grid covers all 12 core shapes for three deterministic seed offsets, plus
concentrated, rising-maximum and early-maximum cases. Read `R4/validation.csv`.
Compare pairs differing in only one factor for the same case/seed, rather than
attributing the whole baseline/streaming gap to one operation. Interactions
between factors may prevent a unique explanation. There is no timing CSV for R4.

## R5: intermittent slow rounds

Start the independent clock monitor as for R1, then run:

```bash
bash scripts/run_review_todos.sh --only R5 --threads 1,8 --rounds 6 \
  --clock-log results/review_clocks.txt
```

At `(4096,32)`, run baseline 1T and both optimized families at 1T/8T, with at
least six rotated rounds and a fresh process per variant/round.
`R5/rounds.csv` retains each round and samples; `summary.csv` pools them.
`telemetry.jsonl` records UTC boundaries, process affinity/status, load, CPU
statistics, available per-core frequency readings and thermal readings every
0.5 seconds while a timed process is alive, plus start/end snapshots.
Unavailable fields are null/empty. Frequency snapshots are not substitutes for
turbostat. Telemetry is diagnostic overhead, so compare these sessions carefully
with the original unmonitored run. Correlation with a slowdown is not proof of
its cause.

## R6: small-shape alignment/process sensitivity

```bash
bash scripts/run_review_todos.sh --only R6 --processes 6
```

At `(128,128)`, single-thread blocked and streaming are compared with:

- offsets of 0, 4, 16, 32 and 60 bytes from a 64-byte boundary for the combined
  Q/K/V allocation, output and scratch;
- `-falign-functions=16` and `-falign-functions=64` builds;
- six fresh processes for every variant, with rotated order.

Allocation/alignment/copying happen before timing. K/V addresses derive from
Q plus their matrix sizes, and actual pointer residues are saved in each
validation/round CSV's `diagnostic` field. The build uses the unchanged kernels
and a generated allocating runner. If `objdump` is available, each alignment
build also saves `disassembly.txt` for checking actual function placement.
These flags change padding/code layout; they do not control every instruction
address or ASLR. A controlled alignment effect supports sensitivity, not a
unique explanation for the original rank reversal.

## Running several experiments

```bash
# Required rerun only (also the default):
bash scripts/run_review_todos.sh --only R1

# All six, including the optional R2 counter sweep:
bash scripts/run_review_todos.sh --only all --threads 1,8 --cache-counters \
  --clock-log results/review_clocks.txt

# A subset, without changing scripts:
bash scripts/run_review_todos.sh --only R2,R3,R5 --threads 1,8
```

R2--R6 use fixed tiles explicitly stated above, not new tile tuning. Default
timing uses five warmups, batched calls, 15 samples and three rotated rounds;
R5 has at least six rounds and R6 uses `--processes` rounds. Every primary
timing is preceded by FP64 validation, and timed output must remain bitwise
identical to its validated output. Timing summaries print median/Q1/Q3 in ms.

Each folder retains `manifest.json`, `uncommitted.patch`, `source_snapshot/`,
generated sources, build logs/commands/hashes, exact input/reference files and
CSV data. A successful manifest says measurements completed **pending
interpretation**; it does not certify a hypothesis or submission readiness.
`results/` is ignored by Git: preserve and explicitly add selected final
evidence after review. Inputs and executables can be archived separately if
large; keep configuration, source provenance, validation and raw samples.

## S1--S6: submission checks

```bash
"$PYTHON" scripts/check_report_submission.py
"$PYTHON" scripts/check_report_submission.py --final
```

The first command lists visible TODO markers and missing/untracked report
assets. `--final` exits unsuccessfully while any remain. This does not stage or
commit files, access the network, or verify human declarations.

| TODO | Required team action |
| --- | --- |
| S1 | State whether Figure 1 is original/adapted/AI-assisted; supply actual source attribution. |
| S2 | Have each member verify completed deliverables and reused code. |
| S3 | Record the 10--15-minute video; add its clickable URL; test access as an instructor/TA would. |
| S4 | Include the report/assets/data in the assessed version and identify it. A release tag chosen before the final commit avoids trying to embed a PDF's own commit hash into itself. |
| S5 | Name the actual AI tools, uses and human verification across all members. |
| S6 | Verify the repository is public, both links work without unintended account restrictions, and reproduce from a clean checkout of the final version. |

Replace the relevant TODO with the completed content, rather than merely
deleting the marker. For example, add `\textbf{Video:} \url{ACTUAL_URL}` and
`\textbf{Final release tag:} ACTUAL_TAG`, with real values. No script can infer
authorship, assess the video, approve member contributions or attest to another
person's permissions. Rebuild the PDF after editing it, then run the final
submission check. Removing TODOs alone is not proof that these checks passed.

## Checks for the scripts themselves

```bash
"$PYTHON" -m unittest discover -s test -p test_review_todos.py -v
```

The unit checks exercise orchestration and failure gates; the smoke runs above
also compile and execute real kernel variants. Neither constitutes the full
Ubuntu performance/counter experiment.
