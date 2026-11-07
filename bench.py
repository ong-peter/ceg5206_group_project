"""Cumulative benchmark ladder for the four tasks.

Stages, all on the same shapes and inputs:
  1        baseline                  materialized, 1 thread
  1+2      blocked                   cache blocking + SIMD, 1 thread
  1+2+3    streaming                 online softmax, 1 thread
  1+2+4    blocked, threaded         blocked over T threads
  1+2+3+4  streaming, threaded       streaming over T threads
Ablation (with --novec-binary): every family at 1 thread built without
vectorization. No-vec blocked vs no-vec baseline isolates the tiling gain;
vec vs no-vec of the same family isolates the SIMD gain.

Kernels that are not compiled are skipped, and threaded stages need an
OpenMP build (make OPENMP=1). --threads takes a list (e.g. 1,2,4,8) for a
thread-scaling sweep. Threaded output must be bitwise identical to one thread,
since each query row is computed entirely by one worker in the same order.
OMP_PLACES/OMP_PROC_BIND are recorded per row, and machine details are saved
next to the CSV as <csv>.machine.json. Every variant is checked against the FP64
reference before timing. Rounds interleave variants so slow drifts in system
conditions affect them all; samples from all rounds are pooled.
"""
import argparse
import csv
import itertools
import json
import os
import platform
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from check import errors, inputs, load_tiles, reference, run


def physical_cores():
    try:
        out = subprocess.run(["lscpu", "-p=CORE,SOCKET"], capture_output=True, text=True, check=True).stdout
        cores = {line for line in out.splitlines() if line and not line.startswith("#")}
        if cores:
            return len(cores)
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True,
                                  text=True, check=True).stdout)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return os.cpu_count() or 1


def build_info(binary):
    out = subprocess.run([str(binary), "--build-info"], capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def command(*arguments):
    try:
        return subprocess.run(arguments, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def machine_info(compiler):
    info = dict(platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                physical_cores=physical_cores(), logical_cpus=os.cpu_count(),
                compiler=command(compiler, "--version").splitlines()[:1],
                omp_env={key: value for key, value in os.environ.items() if key.startswith(("OMP_", "GOMP_", "KMP_"))})
    if platform.system() == "Darwin":
        keys = ["machdep.cpu.brand_string", "hw.physicalcpu", "hw.logicalcpu", "hw.perflevel0.physicalcpu",
                "hw.perflevel1.physicalcpu", "hw.l1dcachesize", "hw.l2cachesize", "hw.cachelinesize"]
        info["sysctl"] = {key: command("sysctl", "-n", key) for key in keys}
    else:
        info["lscpu"] = command("lscpu")
        governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        info["governor"] = governor.read_text().strip() if governor.exists() else "unavailable"
    return info


def thread_list(text):
    values = sorted({int(item) for item in text.split(",") if item.strip()})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("threads must be positive integers, e.g. 1,2,4,8")
    return values


def shape_list(text):
    try:
        shapes = [tuple(int(x) for x in item.lower().split("x")) for item in text.split(",") if item.strip()]
    except ValueError:
        shapes = []
    if not shapes or any(len(shape) != 2 or min(shape) <= 0 for shape in shapes):
        raise argparse.ArgumentTypeError("shapes look like 512x64,4096x64")
    return shapes


def kernels(binary):
    return subprocess.run([str(binary), "--list"], capture_output=True, text=True, check=True).stdout.split()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=Path("./attention"))
    parser.add_argument("--novec-binary", type=Path, help="Build with `make attention-novec` for the ablation")
    parser.add_argument("--extra-binary", action="append", default=[], metavar="NAME=PATH",
                        help="Compare another build of every kernel (e.g. w512=./attention-w512); repeatable")
    parser.add_argument("--threads", type=thread_list, default=[physical_cores()],
                        help="Thread counts for threaded stages, e.g. 8 or 1,2,4,8 (default: physical cores)")
    parser.add_argument("--shapes", type=shape_list, help="Shapes such as 512x64,4096x64 (overrides --full)")
    parser.add_argument("--br", type=int, default=32)
    parser.add_argument("--bc", type=int, default=32)
    parser.add_argument("--tile-config", type=Path, help="Tuned tiles from check.py --tune")
    parser.add_argument("--full", action="store_true", help="All 12 core shapes (default: 512x64, 2048x64)")
    parser.add_argument("--repeats", type=int, default=15, help="Timed samples per variant per round")
    parser.add_argument("--rounds", type=int, default=1, help="Interleaved rounds; samples are pooled")
    parser.add_argument("--csv", type=Path, default=Path("ladder.csv"))
    args = parser.parse_args()
    if min(args.br, args.bc, args.repeats, args.rounds) <= 0:
        parser.error("Tiles, repeats and rounds must be positive")
    threaded = [t for t in args.threads if t > 1]

    tiles = load_tiles(args.tile_config, parser) if args.tile_config else {}

    def tile(algorithm):
        config = tiles.get(algorithm, {})
        return config.get("br", args.br), config.get("bc", args.bc)

    binaries = {"vec": args.binary.resolve()}
    if args.novec_binary:
        binaries["novec"] = args.novec_binary.resolve()
    for item in args.extra_binary:
        name, sep, path = item.partition("=")
        if not sep or not name or name in binaries:
            parser.error(f"--extra-binary needs a unique NAME=PATH: {item}")
        binaries[name] = Path(path).resolve()
    info = {name: build_info(path) for name, path in binaries.items()}
    available = kernels(binaries["vec"])
    openmp = info["vec"].get("openmp") == "1"
    if args.novec_binary and info["novec"].get("vectorized") != "0":
        parser.error("--novec-binary was not built with vectorization disabled")

    variants = [("1", "baseline", "vec", 1)]
    if "blocked" in available:
        variants.append(("1+2", "blocked", "vec", 1))
    if "streaming" in available:
        variants.append(("1+2+3", "streaming", "vec", 1))
    if openmp:
        variants += [("1+2+4", "blocked", "vec", t) for t in threaded if "blocked" in available]
        variants += [("1+2+3+4", "streaming", "vec", t) for t in threaded if "streaming" in available]
    if "novec" in binaries:
        variants += [("ablation", name, "novec", 1) for name in ("baseline", "blocked", "streaming")
                     if name in available]
    for build in binaries:
        if build in ("vec", "novec"):
            continue
        extra = kernels(binaries[build])
        variants += [(build, name, build, 1) for name in ("baseline", "blocked", "streaming") if name in extra]
        if info[build].get("openmp") == "1":
            variants += [(build, name, build, t) for t in threaded
                         for name in ("blocked", "streaming") if name in extra]
    for name in ("blocked", "streaming"):
        if name not in available:
            print(f"Skipping {name}: not compiled", flush=True)
    if not openmp and threaded:
        print("Skipping threaded stages: rebuild with `make clean && make OPENMP=1`", flush=True)

    shapes = args.shapes or (list(itertools.product((128, 512, 2048, 4096), (32, 64, 128)))
                             if args.full else [(512, 64), (2048, 64)])
    omp_env = " ".join(f"{key}={os.environ[key]}" for key in ("OMP_PLACES", "OMP_PROC_BIND") if key in os.environ)
    machine_path = args.csv.with_suffix(".machine.json")
    machine_path.write_text(json.dumps(machine_info(info["vec"].get("flags", "c++").split()[0] or "c++"), indent=2) + "\n")
    fields = ["n", "d", "stage", "algorithm", "build", "vectorized", "br", "bc", "threads",
              "median_ms", "q1_ms", "q3_ms", "speedup_vs_baseline", "speedup_vs_1_thread",
              "gflops", "scratch_bytes", "io_bytes", "max_abs_error", "rel_frobenius_error",
              "samples", "omp_env", "build_flags", "samples_ms"]
    with tempfile.TemporaryDirectory() as temporary, args.csv.open("w", newline="") as handle:
        directory = Path(temporary)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for n, d in shapes:
            q, k, v = inputs(n, d, n + d)
            expected = reference(q, k, v)
            outputs, metrics, samples, meta = {}, {}, {}, {}
            for variant in variants:
                _, algorithm, build, threads = variant
                settings = (*tile(algorithm), threads)
                actual, _ = run(binaries[build], directory, q, k, v, 0, algorithm, settings)
                metrics[variant] = errors(actual, expected)
                if not metrics[variant][2]:
                    raise RuntimeError(f"Correctness failure before timing: {variant} at {n}x{d}")
                outputs[variant], samples[variant] = actual, []
            for variant, actual in outputs.items():
                # Same kernel and build at one thread (ablation rows are a different build).
                matches = [other for other in outputs if other[0] != "ablation"
                           and other[1:3] == variant[1:3] and other[3] == 1]
                if variant[3] > 1 and matches and not np.array_equal(actual, outputs[matches[0]]):
                    raise RuntimeError(f"Threaded output differs from one thread: {variant} at {n}x{d}")
            for round_index in range(args.rounds):
                # Rotate the order each round so no variant always runs first.
                shift = round_index % len(variants)
                for variant in variants[shift:] + variants[:shift]:
                    _, algorithm, build, threads = variant
                    settings = (*tile(algorithm), threads)
                    timed, timing = run(binaries[build], directory, q, k, v, args.repeats, algorithm, settings)
                    if not np.array_equal(outputs[variant], timed):
                        raise RuntimeError(f"Timed output changed: {variant} at {n}x{d}")
                    samples[variant] += [float(x) for x in timing["samples_ms"].split(";")]
                    meta[variant] = timing
            medians = {variant: float(np.median(values)) for variant, values in samples.items()}
            baseline = medians[variants[0]]
            flops = 4.0 * n * n * d  # two matrix products; excludes softmax and scaling
            print(f"\n{n}x{d}", flush=True)
            for variant in variants:
                stage, algorithm, build, threads = variant
                values = np.array(samples[variant])
                median = medians[variant]
                single = next((medians[other] for other in variants if threads > 1 and other[0] != "ablation"
                               and other[1:] == (algorithm, build, 1)), None)
                br, bc = tile(algorithm) if algorithm != "baseline" else (0, 0)
                row = dict(n=n, d=d, stage=stage, algorithm=algorithm, build=build, vectorized=info[build].get("vectorized", ""),
                           br=br, bc=bc, threads=threads, median_ms=median,
                           q1_ms=float(np.percentile(values, 25)), q3_ms=float(np.percentile(values, 75)),
                           speedup_vs_baseline=baseline / median,
                           speedup_vs_1_thread=single / median if single else "",
                           gflops=flops / (median * 1e6), scratch_bytes=meta[variant]["scratch_bytes"],
                           io_bytes=meta[variant]["io_bytes"], max_abs_error=metrics[variant][0],
                           rel_frobenius_error=metrics[variant][1], samples=len(values), omp_env=omp_env if threads > 1 else "",
                           build_flags=info[build].get("flags", "unknown").strip(),
                           samples_ms=";".join(f"{x:.6g}" for x in values))
                writer.writerow(row)
                handle.flush()
                label = f"{stage:9} {algorithm:10} {'' if build == 'vec' else f'({build})':8} T={threads:<3}"
                thread_note = f"  {row['speedup_vs_1_thread']:.2f}x vs 1T" if single else ""
                print(f"  {label} {median:10.3f} ms  IQR {row['q1_ms']:.3f}-{row['q3_ms']:.3f}  "
                      f"{row['speedup_vs_baseline']:6.2f}x vs baseline  {row['gflops']:6.2f} GFLOP/s"
                      f"  scratch {int(row['scratch_bytes']) / 2**20:8.2f} MiB{thread_note}", flush=True)
    print(f"\nSaved {args.csv} and {machine_path}", flush=True)


if __name__ == "__main__":
    main()
