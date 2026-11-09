"""Blocked-kernel design experiment: PV loop order and K packing.

Builds four variants of the blocked kernel, tunes each with check.py --tune
(nine Br/Bc pairs on 512x64 and 2048x64), checks each against the FP64 reference,
then times them in interleaved rounds at each variant's own selected tiles:
  original      row-by-row PV           (-DBLOCKED_ORIGINAL_PV)
  pack          original + packed K     (-DBLOCKED_ORIGINAL_PV -DBLOCKED_PACK_K)
  reorder       key-tile-outer PV       (the default build)
  reorder+pack  reorder + packed K      (-DBLOCKED_PACK_K)
This is a manual design decision, not part of tile tuning: the tuning shapes
all have d=64, while packing helps at d=32 and slows d=128 on the first machine
measured. Tuned tiles can differ per variant; at small N with many threads
a larger Br leaves fewer query blocks than threads, so compare such shapes with
--fixed-tiles too. Output: <out>/packing_experiment.csv (speedup over "original"),
tune_<variant>.csv and tiles_<variant>.json. Packing and tile settings are
included in every timed call, as in bench.py.

With make, the Makefile's compiler and flags are passed in (OPENMP=1 for threads):
  make OPENMP=1 packing-experiment
"""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import physical_cores, shape_list  # noqa: E402
from check import errors, inputs, reference, run  # noqa: E402
import numpy as np  # noqa: E402

VARIANTS = {
    "original": ["-DBLOCKED_ORIGINAL_PV"],
    "pack": ["-DBLOCKED_ORIGINAL_PV", "-DBLOCKED_PACK_K"],
    "reorder": [],
    "reorder+pack": ["-DBLOCKED_PACK_K"],
}


def accepts(compiler, flags):
    probe = subprocess.run([*compiler, *flags, "-x", "c++", "-", "-o", os.devnull], input="int main(){}",
                           text=True, capture_output=True)
    return probe.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--cxxflags", help="Compile flags (default: -O3 -std=c++17 -march=native "
                                           "-fopenmp-simd, plus -fopenmp when the compiler accepts it)")
    parser.add_argument("--ldflags", help="Link flags (default: -fopenmp when the compiler accepts it)")
    parser.add_argument("--shapes", type=shape_list,
                        help="Shapes such as 512x64,2048x128 (default: all 12 core shapes)")
    parser.add_argument("--threads", help="Comma-separated thread counts (default: 1 and physical cores)")
    parser.add_argument("--fixed-tiles", metavar="BRxBC",
                        help="Skip tuning and time every variant at these tiles. Use it where tuned tiles "
                             "change the number of query blocks per thread (small N with many threads)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--out", type=Path, default=Path("results/packing_experiment"))
    args = parser.parse_args()
    compiler = shlex.split(args.cxx)
    openmp = accepts(compiler, ["-fopenmp"])
    if args.cxxflags is None:
        arch = ["-march=native"] if accepts(compiler, ["-march=native"]) else []
        cxxflags = ["-O3", "-std=c++17", *arch, "-fopenmp-simd", *(["-fopenmp"] if openmp else [])]
    else:
        cxxflags = shlex.split(args.cxxflags)
    ldflags = (["-fopenmp"] if openmp else []) if args.ldflags is None else shlex.split(args.ldflags)
    shapes = args.shapes or list(itertools.product((128, 512, 2048, 4096), (32, 64, 128)))
    threads = [int(t) for t in args.threads.split(",")] if args.threads else sorted({1, physical_cores()})
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()

    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        binaries = {}
        for name, defines in VARIANTS.items():
            binaries[name] = temp / f"attention-{name.replace('+', '-')}"
            flags = ["-DWITH_BLOCKED", "-DWITH_STREAMING", *cxxflags, *defines]
            command = [*compiler, *flags, f"-DATTN_BUILD_FLAGS=\"{' '.join(compiler + flags)}\"",
                       *(str(ROOT / s) for s in ("attention.cpp", "main.cpp", "blocked.cpp", "streaming.cpp")),
                       *ldflags, "-o", str(binaries[name])]
            subprocess.run(command, check=True)
        info = subprocess.run([str(binaries["reorder"]), "--build-info"], capture_output=True, text=True).stdout
        if "openmp=1" not in info:
            threads = [1]
            print("No OpenMP in this build: timing one thread only (use make OPENMP=1)", flush=True)

        tiles = {}
        if args.fixed_tiles:
            br, bc = (int(x) for x in args.fixed_tiles.lower().split("x"))
            tiles = {name: (br, bc) for name in binaries}
        for name, binary in binaries.items():
            if args.fixed_tiles:
                break
            safe = name.replace("+", "-")
            subprocess.run([sys.executable, str(ROOT / "check.py"), "--binary", str(binary), "--algorithm", "blocked",
                            "--tune", "--csv", str(out / f"tune_{safe}.csv"),
                            "--tuning-output", str(out / f"tiles_{safe}.json")],
                           check=True, stdout=subprocess.DEVNULL, cwd=ROOT)
            selected = json.loads((out / f"tiles_{safe}.json").read_text())["selected"]["blocked"]
            tiles[name] = (selected["br"], selected["bc"])
            print(f"Tuned {name}: Br={tiles[name][0]} Bc={tiles[name][1]}", flush=True)

        fields = ["n", "d", "threads", "variant", "br", "bc", "median_ms", "q1_ms", "q3_ms",
                  "speedup_vs_original", "max_abs_error", "samples", "build_flags", "samples_ms"]
        rows = []
        names = list(VARIANTS)
        for n, d in shapes:
            q, k, v = inputs(n, d, n + d)
            expected = reference(q, k, v)
            for t in threads:
                settings = {name: (*tiles[name], t) for name in names}
                worst = {}
                for name in names:
                    actual, _ = run(binaries[name], temp, q, k, v, 0, "blocked", settings[name])
                    absolute, _, passed = errors(actual, expected)
                    if not passed:
                        raise RuntimeError(f"Correctness failure before timing: {name} at {n}x{d}, {t} threads")
                    worst[name] = absolute
                samples = {name: [] for name in names}
                for round_index in range(args.rounds):
                    shift = round_index % len(names)
                    for name in names[shift:] + names[:shift]:
                        _, timing = run(binaries[name], temp, q, k, v, args.repeats, "blocked", settings[name])
                        samples[name] += [float(x) for x in timing["samples_ms"].split(";")]
                original = float(np.median(samples["original"]))
                line = []
                for name in names:
                    values = np.array(samples[name])
                    median = float(np.median(values))
                    rows.append(dict(n=n, d=d, threads=t, variant=name, br=settings[name][0], bc=settings[name][1],
                                     median_ms=median, q1_ms=float(np.percentile(values, 25)),
                                     q3_ms=float(np.percentile(values, 75)), speedup_vs_original=original / median,
                                     max_abs_error=worst[name], samples=len(values),
                                     build_flags=" ".join(VARIANTS[name]) or "default",
                                     samples_ms=";".join(f"{x:.6g}" for x in values)))
                    line.append(f"{name} {original / median:.3f}x")
                print(f"{n}x{d} T={t}: original {original:.3f} ms; " + ", ".join(line), flush=True)
                with (out / "packing_experiment.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
    print(f"Saved {out / 'packing_experiment.csv'} with tuning results; compiler: {' '.join(compiler + cxxflags)}")


if __name__ == "__main__":
    main()
