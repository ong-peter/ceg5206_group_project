#!/usr/bin/env bash
# Final benchmark run: every experiment in the proposal, in order, into one folder.
# Prepare the machine first (fixed frequency, idle): see docs/benchmarking.md.
#
#   build -> correctness gate -> tile tuning -> ladder (12 shapes, no-vec ablation)
#   -> thread scaling -> memory report -> stage profile -> hardware counters
#   -> packing experiment -> plots
#
# Environment: PYTHON; ROUNDS (default 3); THREADS (threaded stages, default
# physical cores); OUT (default results/benchmark_<UTC time>); STRICT=1 stops
# unless the governor is performance and turbo is off; PERF=0 and PACKING=0
# skip those steps; LIBOMP for macOS OpenMP builds, as in the Makefile.
# QUICK=1 is a smoke test of the whole pipeline (small shapes, one round, a few
# minutes); its numbers are not results.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PYTHON="$PWD/.venv/bin/python"
    else
        PYTHON=python3
    fi
fi
"$PYTHON" -c 'import numpy, matplotlib' || {
    echo "Install requirements.txt in a Python environment and set PYTHON to its interpreter."
    exit 1
}
rounds="${ROUNDS:-3}"
cores="$("$PYTHON" -c 'from bench import physical_cores; print(physical_cores())')"
threads="${THREADS:-$cores}"
scaling="$("$PYTHON" -c "c=$threads; print(','.join(str(t) for t in sorted({c, *(2**i for i in range(c.bit_length()) if 2**i < c)})))")"
quick="${QUICK:-0}"
if [[ "$quick" == 1 ]]; then
    rounds="${ROUNDS:-1}"
    correctness=() ladder=(--shapes 128x64,512x64) scaling_shapes=512x64 small=(--shapes 128x64,512x64)
    perf_shapes=(--shapes 512x64) packing=(--shapes 128x32,512x64) packing_small=(--shapes 128x32)
else
    correctness=(--full) ladder=(--full) scaling_shapes=512x64,4096x64 small=() perf_shapes=()
    packing=() packing_small=(--shapes 128x32,128x64,128x128)
fi
out="${OUT:-results/benchmark$([[ "$quick" == 1 ]] && echo _quick)_$(date -u '+%Y%m%dT%H%M%SZ')}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
exec > >(tee -a "$out/run.log") 2>&1
started=$(date +%s)
step() { echo; echo "== [$(date '+%H:%M:%S')] $*"; }

frequency_policy() {
    local base=/sys/devices/system/cpu
    echo "governor=$(cat $base/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unavailable)"
    echo "intel_no_turbo=$(cat $base/intel_pstate/no_turbo 2>/dev/null || echo unavailable)"
    echo "cpufreq_boost=$(cat $base/cpufreq/boost 2>/dev/null || echo unavailable)"
    echo "energy_performance_preference=$(cat $base/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo unavailable)"
    echo "power_profile=$(powerprofilesctl get 2>/dev/null || echo unavailable)"
    echo "load_average=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || sysctl -n vm.loadavg 2>/dev/null || echo unavailable)"
}

step "Preflight"
policy="$(frequency_policy)"
echo "$policy"
warnings=()
grep -q '^governor=performance' <<< "$policy" || warnings+=("CPU governor is not 'performance'")
grep -Eq '^(intel_no_turbo=1|cpufreq_boost=0)' <<< "$policy" || warnings+=("turbo appears to be on")
load="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)"
awk -v l="$load" 'BEGIN { exit !(l > 1.0) }' && warnings+=("1-minute load average is $load; close other programs")
for warning in "${warnings[@]}"; do echo "WARNING: $warning (docs/benchmarking.md)"; done
if [[ "${STRICT:-0}" == 1 && ${#warnings[@]} -gt 0 ]]; then
    echo "STRICT=1: stopping. Fix the frequency policy or unset STRICT."
    exit 1
fi
{
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty=$([[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo yes || echo no)"
    uname -a
    "${CXX:-c++}" --version | head -1
    "$PYTHON" -c 'import sys, numpy, matplotlib; print(sys.version.split()[0], "NumPy", numpy.__version__, "matplotlib", matplotlib.__version__)'
    echo "ROUNDS=$rounds THREADS=$threads scaling=$scaling physical_cores=$cores"
    echo "OMP_PLACES=cores OMP_PROC_BIND=close (timing steps)"
    echo "$policy"
    command -v lscpu >/dev/null && lscpu
} > "$out/environment.txt"
git diff > "$out/uncommitted.patch" 2>/dev/null || true
echo "Output: $out"
[[ "$quick" == 1 ]] && echo "QUICK=1: smoke test only, not results." ||
    echo "Expect roughly 3-4 hours with ROUNDS=3 (the 12-shape ladder dominates)."

build=("PYTHON=$PYTHON" "OPENMP=1")
[[ -n "${LIBOMP:-}" ]] && build+=("LIBOMP=$LIBOMP")
step "Build (OpenMP, vectorized and no-vec)"
make clean >/dev/null
make "${build[@]}" attention attention-novec
./attention --build-info | tee "$out/build-info.txt"

step "Correctness gate: all 12 shapes, edge cases and tile sweep; thread invariance"
"$PYTHON" check.py --algorithm all "${correctness[@]}" --tiles --csv "$out/correctness.csv" | tail -1
make "${build[@]}" test-threads

step "Tile tuning (one thread, 9 pairs on 512x64 and 2048x64)"
"$PYTHON" check.py --algorithm all --tune --tiles --csv "$out/tuning-results.csv" \
    --tuning-output "$out/tuned-tiles.json" | grep -E 'Selected|passed'
tiles=(--tile-config "$out/tuned-tiles.json")

pin=(env OMP_PLACES=cores OMP_PROC_BIND=close)
step "Ladder: 12 shapes, every family, $threads threads, no-vec ablation, $rounds rounds"
"${pin[@]}" "$PYTHON" bench.py "${ladder[@]}" --novec-binary ./attention-novec "${tiles[@]}" --threads "$threads" \
    --rounds "$rounds" --csv "$out/ladder.csv"

step "Thread scaling: $scaling threads on $scaling_shapes"
"${pin[@]}" "$PYTHON" bench.py --threads "$scaling" --shapes "$scaling_shapes" "${tiles[@]}" \
    --rounds "$rounds" --csv "$out/scaling.csv"

step "Memory report (allocated scratch and Q/K/V/O bytes)"
"$PYTHON" scripts/memory_report.py --threads "$threads" --csv "$out/memory.csv"

step "Stage profile (instrumented, one thread)"
"$PYTHON" scripts/profile_stages.py "${small[@]}" "${tiles[@]}" --csv "$out/stages.csv"

if [[ "${PERF:-1}" != 0 ]] && command -v perf >/dev/null; then
    step "Hardware counters (perf; DRAM traffic needs perf_event_paranoid <= 0)"
    "$PYTHON" scripts/perf_profile.py "${perf_shapes[@]}" "${tiles[@]}" --threads "1,$threads" --csv "$out/perf_counters.csv" ||
        echo "WARNING: hardware counters failed; see docs/benchmarking.md"
else
    step "Hardware counters skipped (PERF=0 or perf not installed)"
fi

if [[ "${PACKING:-1}" != 0 ]]; then
    step "Blocked-kernel packing experiment (tuned tiles, then small N at fixed 32x32)"
    make "${build[@]}" packing-experiment \
        PACKING_ARGS="${packing[*]} --rounds $rounds --threads 1,$threads --out $out/packing_experiment"
    make "${build[@]}" packing-experiment PACKING_ARGS="${packing_small[*]} --rounds $rounds --threads 1,$threads \
--fixed-tiles 32x32 --out $out/packing_experiment_fixed32"
fi

step "Plots"
"$PYTHON" scripts/plot_results.py --ladder "$out/ladder.csv" --scaling "$out/scaling.csv" \
    --tuning "$out/tuning-results.csv" --tiles "$out/tuned-tiles.json" --stages "$out/stages.csv" \
    --packing "$out/packing_experiment/packing_experiment.csv" --out "$out/plots"

step "Done"
after="$(frequency_policy)"
echo "$after" > "$out/frequency_after.txt"
[[ "$(grep -v load_average <<< "$policy")" == "$(grep -v load_average <<< "$after")" ]] ||
    echo "WARNING: the frequency policy changed during the run; compare environment.txt and frequency_after.txt"
elapsed=$(( $(date +%s) - started ))
echo "Finished in $((elapsed / 3600))h $((elapsed % 3600 / 60))m. Results: $out"
