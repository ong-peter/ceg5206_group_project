#!/usr/bin/env bash
# Follow-up measurements for the four [TODO] markers in report/main.tex.
# Run on the benchmark machine, prepared as for `make benchmark`
# (performance governor, turbo off, idle; perf_event_paranoid <= 0 for DRAM
# counters): see docs/benchmarking.md. Takes roughly 10-20 minutes.
#
#   TODO 1  Rerun streaming 8T at (4096,32): does the slow round recur?
#   TODO 2  perf counters at (128,128), 1 thread: blocked vs streaming
#   TODO 3  perf counters at (4096,128) and (4096,64), 1 and 8 threads,
#           incl. GHz per thread and L2/L3: why streaming scales 7.1x at d=128
#   TODO 4  Sustainable DRAM bandwidth (STREAM-style triad), 1 and 8 threads
#
# Environment: PYTHON; THREADS (default physical cores); TILES (default: the
# final run's tuned-tiles.json); OUT (default results/todo_<UTC time>);
# ONLY="1 3" runs a subset of the steps.
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import numpy' || { echo "Set PYTHON to an interpreter with requirements.txt installed."; exit 1; }
cores="$("$PYTHON" -c 'import sys; sys.path.insert(0, "scripts"); from bench import physical_cores; print(physical_cores())')"
threads="${THREADS:-$cores}"
tiles_file="${TILES:-results/benchmark_20261109T113037Z/tuned-tiles.json}"
[[ -f "$tiles_file" ]] || { echo "Tuned tiles not found: $tiles_file (set TILES)"; exit 1; }
only="${ONLY:-1 2 3 4}"
out="${OUT:-results/todo_$(date -u '+%Y%m%dT%H%M%SZ')}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
exec > >(tee -a "$out/run.log") 2>&1
step() { echo; echo "== [$(date '+%H:%M:%S')] $*"; }
want() { [[ " $only " == *" $1 "* ]]; }
failed=()
pin=(env OMP_PLACES=cores OMP_PROC_BIND=close)
tiles=(--tile-config "$tiles_file")

step "Preflight"
base=/sys/devices/system/cpu
{
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty=$([[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo yes || echo no)"
    uname -a
    "${CXX:-c++}" --version | head -1
    echo "governor=$(cat $base/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unavailable)"
    echo "intel_no_turbo=$(cat $base/intel_pstate/no_turbo 2>/dev/null || echo unavailable)"
    echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unavailable)"
    echo "load_average=$(cut -d' ' -f1-3 /proc/loadavg)"
    echo "THREADS=$threads TILES=$tiles_file ONLY=$only"
} | tee "$out/environment.txt"
git diff > "$out/uncommitted.patch" 2>/dev/null || true
grep -q '^governor=performance' "$out/environment.txt" || echo "WARNING: governor is not 'performance'"
grep -q '^intel_no_turbo=1' "$out/environment.txt" || echo "WARNING: turbo appears to be on"

step "Build (OpenMP)"
make clean >/dev/null
make "PYTHON=$PYTHON" OPENMP=1 attention || { echo "Build failed"; exit 1; }
./attention --build-info | tee "$out/build-info.txt"

if want 1; then
    step "TODO 1: streaming ${threads}T at 4096x32, 6 interleaved rounds"
    if "${pin[@]}" "$PYTHON" scripts/bench.py --shapes 4096x32 --threads "$threads" "${tiles[@]}" \
            --rounds 6 --csv "$out/rerun_4096x32.csv"; then
        # Samples are pooled in round order, 15 per round: report each round's median.
        "$PYTHON" - "$out/rerun_4096x32.csv" <<'PY' | tee "$out/rerun_4096x32_rounds.txt"
import csv, statistics, sys
for row in csv.DictReader(open(sys.argv[1])):
    samples = [float(x) for x in row["samples_ms"].split(";")]
    rounds = [samples[i:i + 15] for i in range(0, len(samples), 15)]
    medians = [statistics.median(r) for r in rounds]
    print(f"{row['algorithm']:9} T={row['threads']:<2} pooled median {float(row['median_ms']):9.3f} ms  "
          f"per-round medians: " + " ".join(f"{m:.3f}" for m in medians)
          + f"  (max/min {max(medians) / min(medians):.3f})")
PY
    else
        failed+=("TODO 1")
    fi
fi

if want 2; then
    step "TODO 2: perf counters at 128x128, 1 thread"
    "$PYTHON" scripts/perf_profile.py --shapes 128x128 --threads 1 "${tiles[@]}" \
        --csv "$out/perf_128x128.csv" || failed+=("TODO 2")
fi

if want 3; then
    step "TODO 3: perf counters at 4096x128 and 4096x64, 1 and ${threads} threads (pinned)"
    "${pin[@]}" "$PYTHON" scripts/perf_profile.py --shapes 4096x64,4096x128 --threads "1,$threads" "${tiles[@]}" \
        --csv "$out/perf_4096_d64_d128.csv" || failed+=("TODO 3")
fi

if want 4; then
    step "TODO 4: sustainable DRAM bandwidth (triad, 768 MiB working set)"
    if "${CC:-cc}" -O3 -march=native -fopenmp scripts/stream_triad.c -o "$out/stream_triad"; then
        for t in 1 "$threads"; do
            OMP_NUM_THREADS=$t "${pin[@]}" "$out/stream_triad"
        done | awk 'NR == 1 || !/^threads/' | tee "$out/stream_triad.csv"
    else
        failed+=("TODO 4")
    fi
fi

step "Done: $out"
if ((${#failed[@]})); then echo "Failed: ${failed[*]}"; exit 1; fi
cat <<'MSG'
Where each result goes in report/main.tex:
  TODO 1  rerun_4096x32_rounds.txt: if every round agrees, the earlier slow round
          was a one-off; otherwise report the per-round medians.
  TODO 2  perf_128x128.csv: IPC, instructions per call, vector share,
          blocked vs streaming (Part C, the (128,128) exception).
  TODO 3  perf_4096_d64_d128.csv: compare ghz_per_thread and l2_miss_rate /
          llc_* for streaming at 1 vs 8 threads, d=64 vs d=128 (Part D).
  TODO 4  stream_triad.csv: triad GB/s at 1 and 8 threads; machine balance =
          922 GFLOP/s / triad GB/s (Section 4.7).
MSG
