#!/usr/bin/env bash
# ThreadSanitizer check of the OpenMP kernels (B/C) and the C streaming API.
# Prefers Clang with LLVM libomp and Archer, which makes OpenMP synchronization
# visible to TSan, so any report is treated as a real finding. Falls back to GCC
# with libgomp, whose fork/join false positives are classified by tsan_triage.py.
#
# Environment: PYTHON, TSAN_CXX (force a compiler), TOOLCHAIN=clang|gcc,
# THREADS (default "2 3 <physical> <logical> 64", as make test-threads),
# API=0 to skip the slow API test, FULL=1 for all 12 core shapes, OUT (output dir).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PYTHON="$PWD/.venv/bin/python"
    else
        PYTHON=python3
    fi
fi
"$PYTHON" -c 'import numpy' || {
    echo "Install requirements.txt in a Python environment and set PYTHON to its interpreter."
    exit 1
}
out="${OUT:-results/tsan_$(date -u '+%Y%m%dT%H%M%SZ')}"
mkdir -p "$out/bin" "$out/reports" "$out/probe"
out="$(cd "$out" && pwd)"

# A candidate counts only if it compiles and runs an OpenMP+TSan program;
# for Clang, Archer must also announce itself at startup.
probe_src="$out/probe/probe.cpp"
printf '%s\n' '#include <omp.h>' 'int main() { int x[64] = {};' \
    '#pragma omp parallel for num_threads(2)' 'for (int i = 0; i < 64; ++i) x[i] = i;' \
    'return x[63] == 63 ? 0 : 1; }' > "$probe_src"
probe() {  # probe <compiler> <extra flags...>
    local cxx="$1"; shift
    "$cxx" -fopenmp -fsanitize=thread -g "$@" "$probe_src" -o "$out/probe/a.out" >/dev/null 2>&1 &&
        TSAN_OPTIONS=report_bugs=0 "$out/probe/a.out" >/dev/null 2>&1
}
toolchain="" cxx="" omp_flags=()
find_clang() {
    local candidates=("${TSAN_CXX:-}" clang++ clang++-22 clang++-21 clang++-20 clang++-19 clang++-18)
    for c in "${candidates[@]}"; do
        [[ -n "$c" ]] && command -v "$c" >/dev/null 2>&1 || continue
        "$c" --version 2>/dev/null | grep -qi clang || continue
        local archer
        archer="$("$c" -print-file-name=libarcher.so 2>/dev/null)"
        [[ "$archer" == /* && -f "$archer" ]] || continue
        archer="$(readlink -f "$archer")"
        local libdir
        libdir="$(dirname "$archer")"
        probe "$c" "-L$libdir" "-Wl,-rpath,$libdir" || continue
        local banner
        banner="$(OMP_TOOL_LIBRARIES="$archer" ARCHER_OPTIONS=verbose=1 TSAN_OPTIONS=report_bugs=0 \
            "$out/probe/a.out" 2>&1)" || continue
        [[ "$banner" == *"Archer detected"* ]] || continue
        toolchain=clang cxx="$c" omp_flags=("-L$libdir" "-Wl,-rpath,$libdir")
        export OMP_TOOL_LIBRARIES="$archer"
        return 0
    done
    return 1
}
find_gcc() {
    for c in "${TSAN_CXX:-}" g++ c++; do
        [[ -n "$c" ]] && command -v "$c" >/dev/null 2>&1 || continue
        "$c" --version 2>/dev/null | grep -qi clang && continue
        probe "$c" || continue
        toolchain=gcc cxx="$c"
        return 0
    done
    return 1
}
case "${TOOLCHAIN:-}" in
    clang) find_clang || { echo "No working Clang with Archer found."; exit 1; } ;;
    gcc) find_gcc || { echo "No working GCC with TSan found."; exit 1; } ;;
    "") find_clang || {
            echo "Clang with Archer not found; falling back to GCC (expect fork/join false positives)."
            find_gcc || { echo "No working TSan toolchain found."; exit 1; }
        } ;;
    *) echo "TOOLCHAIN must be clang or gcc"; exit 1 ;;
esac

arch="$("$cxx" -march=native -Werror -x c++ -c /dev/null -o /dev/null >/dev/null 2>&1 && echo -march=native || true)"
flags=(-DWITH_BLOCKED -DWITH_STREAMING -std=c++17 -Wall -Wextra $arch -fopenmp-simd -fopenmp
       -O1 -g -fsanitize=thread -fno-omit-frame-pointer "${omp_flags[@]}")
"$cxx" "${flags[@]}" attention.cpp main.cpp blocked.cpp streaming.cpp -o "$out/bin/attention-tsan"
"$cxx" "${flags[@]}" -I. attention.cpp streaming.cpp tests/streaming_test.cpp -o "$out/bin/streaming-test-tsan"

cores="$("$PYTHON" -c 'from bench import physical_cores; print(physical_cores())' 2>/dev/null || echo 4)"
logical="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo "$cores")"
threads="${THREADS:-$(printf '%s\n' 2 3 "$cores" "$logical" 64 | sort -nu | tr '\n' ' ')}"
check_options=(--algorithm all --tiles)
[[ "${FULL:-0}" == 1 ]] && check_options+=(--full)
{
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    uname -a
    "$cxx" --version | head -1
    echo "toolchain=$toolchain OMP_TOOL_LIBRARIES=${OMP_TOOL_LIBRARIES:-unset}"
    echo "flags=${flags[*]}"
    echo "threads=$threads api=${API:-1} check.py ${check_options[*]}"
} > "$out/environment.txt"
cat "$out/environment.txt"

# Reports never change exit codes, so every case still runs; triage decides.
tsan="halt_on_error=0 exitcode=0 history_size=4 ignore_noninstrumented_modules=1"
pids=() names=()
if [[ "${API:-1}" != 0 ]]; then
    TSAN_OPTIONS="$tsan log_path=$out/reports/api" "$out/bin/streaming-test-tsan" > "$out/api.log" 2>&1 &
    pids+=($!) names+=(api)
fi
for t in $threads; do
    TSAN_OPTIONS="$tsan log_path=$out/reports/check_t$t" "$PYTHON" check.py \
        --binary "$out/bin/attention-tsan" "${check_options[@]}" --threads "$t" > "$out/check_t$t.log" 2>&1 &
    pids+=($!) names+=("check_t$t")
done
echo "Running ${names[*]} in parallel; logs in $out"
failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "${names[$i]}: passed ($(tail -1 "$out/${names[$i]}.log"))"
    else
        echo "${names[$i]}: FAILED (see $out/${names[$i]}.log)"
        failed=1
    fi
done

"$PYTHON" scripts/tsan_triage.py "$out/reports" --toolchain "$toolchain" \
    --samples "$out/review_samples.txt" | tee "$out/triage.txt" || failed=1
if [[ "$failed" == 0 ]]; then
    echo "TSan check passed ($toolchain). Summary: $out/triage.txt"
else
    echo "TSan check FAILED ($toolchain). See $out/triage.txt and the logs above."
fi
exit "$failed"
