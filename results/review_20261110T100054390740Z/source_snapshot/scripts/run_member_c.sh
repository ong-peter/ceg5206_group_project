#!/usr/bin/env bash
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
threads="${THREADS:-1}"
build_options=("PYTHON=$PYTHON" "CXX=${CXX:-c++}" "OPENMP=${OPENMP:-0}")
if [[ -n "${LIBOMP:-${OMP_PREFIX:-}}" ]]; then
    build_options+=("LIBOMP=${LIBOMP:-$OMP_PREFIX}")
fi
mkdir -p results
{
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    uname -a
    "${CXX:-c++}" --version
    "$PYTHON" -c 'import sys, numpy; print(sys.version); print("NumPy", numpy.__version__)'
    echo "OPENMP=${OPENMP:-0} LIBOMP=${LIBOMP:-${OMP_PREFIX:-}} THREADS=$threads"
    echo "OMP_DYNAMIC=${OMP_DYNAMIC:-unset} OMP_PROC_BIND=${OMP_PROC_BIND:-unset} OMP_PLACES=${OMP_PLACES:-unset}"
    echo "CXXFLAGS=${CXXFLAGS:--O3 -std=c++17 -Wall -Wextra -Wpedantic}"
} > results/member_c_environment.txt
make "${build_options[@]}" attention attention-novec test-streaming-api 2>&1 | tee results/member_c_build.log
./attention --build-info >> results/member_c_environment.txt
"$PYTHON" check.py --algorithm all --tiles --threads "$threads" \
    --csv results/member_c_quick.csv 2>&1 | tee results/member_c_quick.log
"$PYTHON" bench.py --threads "$threads" --shapes 128x64,512x64 --rounds 2 \
    --novec-binary ./attention-novec --csv results/member_c_bench.csv 2>&1 | tee results/member_c_bench.log
"$PYTHON" scripts/memory_report.py --threads "$threads" --csv results/member_c_memory.csv
echo "Correctness saved to results/member_c_quick.csv; preliminary timings to results/member_c_bench.csv."
echo "Environment and build flags saved to results/member_c_environment.txt and member_c_build.log."
