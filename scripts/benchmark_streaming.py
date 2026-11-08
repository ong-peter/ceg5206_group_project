"""Historical C-only ablations with fixed flags; use bench.py for integrated A/B/C results."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check import inputs, reference  # Independent FP64 reference, exact FP32 inputs.
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="All 12 proposal shapes")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path, default=ROOT / "results/optimization")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000:
        parser.error("repeats must be in [1, 1000]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    # Same flags for every variant; single-threaded, no fast-math, no LTO.
    flags = ["-O3", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-I", str(ROOT)]
    builds = [
        ("original", "benchmarks/streaming_original.cpp", ["-Dattention_streaming=attention_original", "-Dattention_streaming_scratch=attention_original_scratch"]),
        ("rescale_only", "streaming.cpp", ["-DSTREAMING_SCALAR_SCORES", "-Dattention_streaming=attention_rescale_only", "-Dattention_streaming_scratch=attention_rescale_only_scratch"]),
        ("packed_only", "streaming.cpp", ["-DSTREAMING_ALWAYS_RESCALE", "-Dattention_streaming=attention_packed_only", "-Dattention_streaming_scratch=attention_packed_only_scratch"]),
        ("optimized", "streaming.cpp", []),
    ]
    commands = []
    objects = []
    with (output / "build.log").open("w") as log:
        for name, source, defines in builds:
            obj = output / f"{name}.o"
            cmd = [*compiler, *flags, *defines, "-c", str(ROOT/source), "-o", str(obj)]
            commands.append(cmd)
            subprocess.run(cmd, check=True, stdout=log, stderr=log)
            objects.append(str(obj))
        binary = output / "compare"
        cmd = [*compiler, *flags, str(ROOT/"attention.cpp"), str(ROOT/"benchmarks/compare.cpp"), *objects, "-o", str(binary)]
        commands.append(cmd)
        subprocess.run(cmd, check=True, stdout=log, stderr=log)
    (output / "environment.json").write_text(json.dumps({
        "platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
        "compiler": subprocess.check_output([*compiler, "--version"], text=True),
        "commands": commands, "br": 32, "bc": 32, "threads": 1,
        "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                          ["attention.cpp", "attention.h", "streaming.cpp", "benchmarks/streaming_original.cpp", "benchmarks/compare.cpp"]},
        "affinity": "OS controlled", "frequency": "OS controlled", "cache": "repeated warm calls",
    }, indent=2) + "\n")
    shapes = list(itertools.product((128,512,2048,4096),(32,64,128))) if args.full else [(128,64),(512,64),(2048,64)]
    with tempfile.TemporaryDirectory() as temp, (output / "comparison.csv").open("w", newline="") as file:
        data_path, ref_path = Path(temp)/"inputs.bin", Path(temp)/"reference.bin"
        writer = None
        for n,d in shapes:
            q,k,v = inputs(n,d,n+d)
            np.concatenate([q.ravel(),k.ravel(),v.ravel()]).tofile(data_path)
            reference(q,k,v).tofile(ref_path)
            result = subprocess.run([str(binary),str(n),str(d),str(data_path),str(ref_path),"32","32",str(args.repeats)],
                                    capture_output=True,text=True,check=True)
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            if writer is None:
                writer = csv.DictWriter(file,fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            file.flush()
            times = {row["algorithm"]:float(row["median_ms"]) for row in rows}
            print(f"{n}x{d}: baseline={times['baseline']:.3f} ms, original={times['original']:.3f} ms, "
                  f"optimized={times['optimized']:.3f} ms; "
                  f"{times['original']/times['optimized']:.2f}x over original, "
                  f"{times['baseline']/times['optimized']:.2f}x over baseline",flush=True)
    print(f"Saved {output / 'comparison.csv'}")


if __name__ == "__main__":
    main()
