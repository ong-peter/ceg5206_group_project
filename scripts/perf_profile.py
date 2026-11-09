"""Hardware counters per attention call with Linux perf (one small and one large shape).

For each algorithm and thread count, the attention binary runs under perf stat
twice: with timed repeats and with none. The difference, divided by the extra
calls, removes input loading, allocation and validation, so every count is per
kernel call. Counted per process (all threads):
  cache group  cycles, instructions, L1D loads/misses, LLC loads/misses
  FP group     single-precision scalar, 128-bit and 256-bit FP instructions;
               FLOPs = scalar + 4*128b + 8*256b (FMA counts twice in these events)
  DRAM group   uncore_imc data reads/writes, system-wide: needs
               perf_event_paranoid <= 0 (or root); skipped otherwise. Other
               processes add traffic, so keep the machine idle.
The working set (scratch + Q/K/V/O bytes) is reported next to the measured
traffic: allocated storage is not memory traffic. Counter runs are not timing runs.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import shape_list  # noqa: E402
from check import inputs, options  # noqa: E402
import numpy as np  # noqa: E402

GROUPS = {
    "cache": ["cycles", "instructions", "L1-dcache-loads", "L1-dcache-load-misses", "LLC-loads", "LLC-load-misses"],
    "fp": ["fp_arith_inst_retired.scalar_single", "fp_arith_inst_retired.128b_packed_single",
           "fp_arith_inst_retired.256b_packed_single"],
    "dram": ["uncore_imc/data_reads/", "uncore_imc/data_writes/"],
}
UNITS = {"MiB": 2**20, "KiB": 2**10, "B": 1, "": 1}


def perf_stat(events, command, system_wide=False):
    with tempfile.NamedTemporaryFile("r", suffix=".csv") as output:
        completed = subprocess.run(["perf", "stat", "-x,", "-o", output.name, *(["-a"] if system_wide else []),
                                    "-e", ",".join(events), "--", *command],
                                   capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip().splitlines()[-1] if completed.stderr else "perf failed")
        counts = {}
        for line in Path(output.name).read_text().splitlines():
            fields = line.split(",")
            if len(fields) < 3 or line.startswith("#"):
                continue
            value, unit, event = fields[0], fields[1], fields[2]
            if value.startswith("<"):
                raise RuntimeError(f"perf could not count {event}: {value}")
            counts[event] = float(value) * UNITS.get(unit, 1)
    return counts, completed.stdout


def calls(stdout):
    """Kernel calls made by main.cpp: 2 checks, 5 warm-ups, batch calibration, repeats x batch."""
    if not stdout.strip():
        return 2
    row = next(csv.DictReader(io.StringIO(stdout)))
    batch, repeats = int(row["batch"]), int(row["repeats"])
    return 2 + int(row["warmups"]) + (2 * batch - 1) + repeats * batch


def available(events, system_wide=False):
    try:
        perf_stat(events, ["true"], system_wide)
        return True
    except RuntimeError as error:
        reason = ("system-wide counters need `sudo sysctl kernel.perf_event_paranoid=0`" if system_wide
                  else str(error))
        print(f"Skipping {', '.join(events)}: {reason}", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=Path("./attention"))
    parser.add_argument("--shapes", type=shape_list, default=[(512, 64), (4096, 64)])
    parser.add_argument("--threads", default="1", help="Comma-separated thread counts for blocked/streaming")
    parser.add_argument("--tile-config", type=Path, help="Tuned tiles from check.py --tune (default 32x32)")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--csv", type=Path, default=Path("results/perf_counters.csv"))
    args = parser.parse_args()
    if not shutil.which("perf"):
        parser.error("perf not found (Ubuntu: linux-tools-$(uname -r))")
    binary = str(args.binary.resolve())
    tiles = json.loads(args.tile_config.read_text())["selected"] if args.tile_config else {}
    groups = {name: events for name, events in GROUPS.items() if available(events, name == "dram")}
    if "cache" not in groups:
        parser.error("perf cannot count cycles; set kernel.perf_event_paranoid to 1 or lower")
    algorithms = subprocess.run([binary, "--list"], capture_output=True, text=True, check=True).stdout.split()
    thread_counts = [int(t) for t in args.threads.split(",")]
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        source, target = Path(temporary) / "input.bin", Path(temporary) / "output.bin"
        for n, d in args.shapes:
            q, k, v = inputs(n, d, n + d)
            np.concatenate((q.ravel(), k.ravel(), v.ravel())).tofile(source)
            for algorithm in algorithms:
                for threads in ([1] if algorithm == "baseline" else thread_counts):
                    config = tiles.get(algorithm, {})
                    settings = (config.get("br", 32), config.get("bc", 32), threads)
                    base = [binary, str(n), str(d), str(source), str(target)]
                    flags = options(algorithm, settings)
                    memory = next(csv.DictReader(io.StringIO(subprocess.run(
                        [binary, str(n), str(d), "--memory", *flags], capture_output=True, text=True,
                        check=True).stdout)))
                    per_call, median_ms = {}, None
                    for name, events in groups.items():
                        full, stdout = perf_stat(events, [*base, str(args.repeats), *flags], name == "dram")
                        empty, _ = perf_stat(events, [*base, "0", *flags], name == "dram")
                        extra = calls(stdout) - 2
                        median_ms = float(next(csv.DictReader(io.StringIO(stdout)))["median_ms"])
                        for event in events:
                            per_call[event] = (full[event] - empty[event]) / extra
                    fp = (per_call.get("fp_arith_inst_retired.scalar_single", 0)
                          + 4 * per_call.get("fp_arith_inst_retired.128b_packed_single", 0)
                          + 8 * per_call.get("fp_arith_inst_retired.256b_packed_single", 0))
                    row = dict(
                        algorithm=algorithm, n=n, d=d, br=0 if algorithm == "baseline" else settings[0],
                        bc=0 if algorithm == "baseline" else settings[1], threads=threads, median_ms=median_ms,
                        cycles=per_call["cycles"], instructions=per_call["instructions"],
                        ipc=per_call["instructions"] / per_call["cycles"],
                        ghz_per_thread=per_call["cycles"] / (median_ms * 1e6 * threads),
                        l1d_loads=per_call["L1-dcache-loads"],
                        l1d_miss_rate=per_call["L1-dcache-load-misses"] / per_call["L1-dcache-loads"],
                        llc_loads=per_call["LLC-loads"], llc_misses=per_call["LLC-load-misses"],
                        llc_miss_bytes=64 * per_call["LLC-load-misses"],
                        fp_flops=fp if "fp" in groups else "",
                        vector_flop_share=(1 - per_call["fp_arith_inst_retired.scalar_single"] / fp)
                        if "fp" in groups and fp else "",
                        model_flops=4 * n * n * d,
                        dram_read_bytes=per_call.get("uncore_imc/data_reads/", ""),
                        dram_write_bytes=per_call.get("uncore_imc/data_writes/", ""),
                        working_set_bytes=int(memory["scratch_bytes"]) + int(memory["io_bytes"]),
                        scratch_bytes=int(memory["scratch_bytes"]), repeats=args.repeats)
                    rows.append(row)
                    dram = (f"  DRAM {(row['dram_read_bytes'] + row['dram_write_bytes']) / 2**20:8.1f} MiB"
                            if row["dram_read_bytes"] != "" else "")
                    print(f"{n}x{d} {algorithm:9} T={threads:<2} IPC {row['ipc']:.2f}  "
                          f"{row['ghz_per_thread']:.2f} GHz  L1D miss {row['l1d_miss_rate']:.1%}  "
                          f"LLC miss {row['llc_miss_bytes'] / 2**20:8.2f} MiB{dram}  "
                          f"working set {row['working_set_bytes'] / 2**20:7.2f} MiB", flush=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.csv} (counts per kernel call)")


if __name__ == "__main__":
    main()
