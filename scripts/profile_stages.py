"""Per-stage time of each family on one small and one large shape (single thread).

Baseline: QK^T, softmax and PV are separate public functions, timed around each call.
Blocked: stages are interleaved per query block, so a temporary copy of blocked.cpp
is instrumented with timers that accumulate over blocks. Streaming fuses its stages;
it is split into score+packing and online update+PV with profile_streaming.py's
instrumentation. Timers add overhead; the instrumented total is reported next to
the stages and these diagnostics never replace bench.py timings. Every profiled
call is checked against the FP64 reference. No production source file is modified.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check import inputs, reference  # noqa: E402
from profile_streaming import instrument as instrument_streaming, replace_once  # noqa: E402
import numpy as np  # noqa: E402

STAGES = {
    "baseline": ["QK^T scores", "softmax", "PV"],
    "blocked": ["QK^T scores", "softmax", "PV"],
    "streaming": ["QK^T scores + K packing", "online update + PV"],
}

DRIVER = r'''
#include "attention.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
#if defined(PROFILE_BLOCKED)
extern double profile_stage_ms[3];
#elif defined(PROFILE_STREAMING)
extern double profile_scores_ms, profile_update_ms;
#else
double profile_stage_ms[3];
#endif
template<class T> std::vector<T> read(const char* path, std::size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(T)))
        throw std::runtime_error("Incorrect profile input size");
    file.seekg(0);
    std::vector<T> result(count);
    file.read(reinterpret_cast<char*>(result.data()), count * sizeof(T));
    return result;
}
int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::invalid_argument("profile N d br bc repeats inputs reference");
        const std::size_t n = std::stoull(argv[1]), d = std::stoull(argv[2]);
        const AttentionConfig config{std::stoull(argv[3]), std::stoull(argv[4]), 1};
        const int repeats = std::stoi(argv[5]);
        const auto data = read<float>(argv[6], 3 * n * d);
        const auto expected = read<double>(argv[7], n * d);
        const float *q = data.data(), *k = q + n * d, *v = k + n * d;
        std::vector<float> out(n * d);
#if defined(PROFILE_BLOCKED)
        std::vector<float> scratch(attention_blocked_scratch(n, d, config));
        const auto call = [&] { attention_blocked(q, k, v, out.data(), scratch.data(), n, d, config); };
#elif defined(PROFILE_STREAMING)
        std::vector<float> scratch(attention_streaming_scratch(n, d, config));
        const auto call = [&] { attention_streaming(q, k, v, out.data(), scratch.data(), n, d, config); };
#else
        std::vector<float> scratch(n * n);
        const auto call = [&] {
            auto start = Clock::now();
            attention_scores(q, k, scratch.data(), n, d);
            profile_stage_ms[0] += ms_since(start);
            start = Clock::now();
            attention_softmax(scratch.data(), n);
            profile_stage_ms[1] += ms_since(start);
            start = Clock::now();
            attention_values(scratch.data(), v, out.data(), n, d);
            profile_stage_ms[2] += ms_since(start);
        };
#endif
        for (int i = 0; i < 5; ++i) call();
#if defined(PROFILE_STREAMING)
        profile_scores_ms = profile_update_ms = 0;
#else
        std::fill(profile_stage_ms, profile_stage_ms + 3, 0.0);
#endif
        std::vector<double> totals;
        for (int i = 0; i < repeats; ++i) {
            const auto start = Clock::now();
            call();
            totals.push_back(ms_since(start));
        }
        for (std::size_t i = 0; i < n * d; ++i)
            if (!std::isfinite(out[i]) || std::abs(out[i] - expected[i]) > 1e-4 + 1e-3 * std::abs(expected[i]))
                throw std::runtime_error("Profile correctness failed");
        double mean = 0;
        for (double t : totals) mean += t / repeats;
        std::sort(totals.begin(), totals.end());
        std::cout << std::setprecision(10);
#if defined(PROFILE_STREAMING)
        std::cout << profile_scores_ms / repeats << ',' << profile_update_ms / repeats;
#else
        std::cout << profile_stage_ms[0] / repeats << ',' << profile_stage_ms[1] / repeats << ','
                  << profile_stage_ms[2] / repeats;
#endif
        std::cout << ',' << mean << ',' << totals[totals.size() / 2] << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
'''


def instrument_blocked(source):
    """Accumulate QK^T, softmax and PV time over query blocks (serial build only)."""
    source = "#include <chrono>\n" + source
    point = "void attention_blocked("
    source = replace_once(source, point, """double profile_stage_ms[3];
namespace {
using ProfileClock = std::chrono::steady_clock;
double profile_ms(ProfileClock::time_point start) {
    return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}
}
""" + point)
    point = "        // QK^T: cache-block both query and key rows."
    source = replace_once(source, point, "        auto profile_start = ProfileClock::now();\n" + point)
    point = "        // Stable row-wise softmax, in place: n*n scratch is retained."
    source = replace_once(source, point, """        profile_stage_ms[0] += profile_ms(profile_start);
        profile_start = ProfileClock::now();
""" + point)
    point = "        // PV: block keys, accumulate on contiguous FP32 feature lanes."
    source = replace_once(source, point, """        profile_stage_ms[1] += profile_ms(profile_start);
        profile_start = ProfileClock::now();
""" + point)
    # After the #endif closing the PV loop-order choice, so either order is timed.
    point = "                        oi[f] += weight * vj[f];\n                }\n            }\n        }\n#endif\n"
    return replace_once(source, point, point + "        profile_stage_ms[2] += profile_ms(profile_start);\n")


def shape_list(text):
    return [tuple(int(x) for x in item.split("x")) for item in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", type=shape_list, default=[(512, 64), (4096, 64)],
                        help="Small and large shapes (default: 512x64,4096x64)")
    parser.add_argument("--tile-config", type=Path, help="Tuned tiles from check.py --tune (default 32x32)")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--csv", type=Path, default=Path("results/stages.csv"))
    args = parser.parse_args()
    tiles = json.loads(args.tile_config.read_text())["selected"] if args.tile_config else {}
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    arch = ["-march=native"] if subprocess.run([*compiler, "-march=native", "-Werror", "-x", "c++", "-c",
                                                 os.devnull, "-o", os.devnull], capture_output=True).returncode == 0 else []
    flags = ["-O3", "-std=c++17", *arch, "-fopenmp-simd", "-I", str(ROOT)]
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        (temp / "driver.cpp").write_text(DRIVER)
        (temp / "blocked.cpp").write_text(instrument_blocked((ROOT / "blocked.cpp").read_text()))
        (temp / "streaming.cpp").write_text(instrument_streaming((ROOT / "streaming.cpp").read_text()))
        sources = {"baseline": [], "blocked": [temp / "blocked.cpp"], "streaming": [temp / "streaming.cpp"]}
        for algorithm, extra in sources.items():
            binary = temp / f"profile-{algorithm}"
            subprocess.run([*compiler, *flags, f"-DPROFILE_{algorithm.upper()}", str(temp / "driver.cpp"),
                            str(ROOT / "attention.cpp"), *map(str, extra), "-o", str(binary)], check=True)
        for n, d in args.shapes:
            q, k, v = inputs(n, d, n + d)
            np.concatenate([q.ravel(), k.ravel(), v.ravel()]).tofile(temp / "inputs.bin")
            reference(q, k, v).tofile(temp / "reference.bin")
            for algorithm, stages in STAGES.items():
                config = tiles.get(algorithm, {})
                br, bc = (0, 0) if algorithm == "baseline" else (config.get("br", 32), config.get("bc", 32))
                output = subprocess.check_output(
                    [str(temp / f"profile-{algorithm}"), str(n), str(d), str(br or 32), str(bc or 32),
                     str(args.repeats), str(temp / "inputs.bin"), str(temp / "reference.bin")], text=True)
                *times, mean, median = (float(x) for x in output.strip().split(","))
                for stage, ms in zip(stages, times):
                    rows.append(dict(algorithm=algorithm, n=n, d=d, br=br, bc=bc, threads=1, stage=stage,
                                     stage_ms=ms, share=ms / mean, instrumented_mean_ms=mean,
                                     instrumented_median_ms=median, repeats=args.repeats))
                summary = ", ".join(f"{s} {t:.3f} ms ({t / mean:.0%})" for s, t in zip(stages, times))
                print(f"{n}x{d} {algorithm:9} {summary}; total {mean:.3f} ms", flush=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.csv}; flags: {' '.join(compiler + flags[:-2])}")


if __name__ == "__main__":
    main()
