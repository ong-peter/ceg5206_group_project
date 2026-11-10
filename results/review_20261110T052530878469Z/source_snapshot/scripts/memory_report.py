"""Query kernel workspace sizes; no tensors or attention computations are allocated."""
import argparse
import csv
import io
import itertools
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./attention")
    parser.add_argument("--br", type=int, default=32)
    parser.add_argument("--bc", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--csv", type=Path, default=Path("results/memory.csv"))
    args = parser.parse_args()
    if min(args.br, args.bc, args.threads) < 1 or args.threads > 2147483647:
        parser.error("Positive tiles and a thread count fitting int are required")
    binary = str(Path(args.binary).resolve())
    algorithms = subprocess.check_output([binary, "--list"], text=True).split()
    if set(algorithms) != {"baseline", "blocked", "streaming"} or len(algorithms) != 3:
        raise RuntimeError("The integrated memory report requires baseline, blocked and streaming")
    rows = []
    for n, d in itertools.product((128, 512, 2048, 4096), (32, 64, 128)):
        for algorithm in algorithms:
            threads = 1 if algorithm == "baseline" else args.threads
            result = subprocess.run(
                [binary, str(n), str(d), "--memory", "--algorithm", algorithm,
                 "--br", str(args.br), "--bc", str(args.bc), "--threads", str(threads)],
                text=True, capture_output=True, check=True,
            )
            row = next(csv.DictReader(io.StringIO(result.stdout)))
            r, c = min(n, args.br), min(n, args.bc)
            workers = min(threads, (n + r - 1) // r)
            expected = 4 * n * n if algorithm in ("baseline", "blocked") else 4 * workers * (
                r * c + r * d + 2 * r + c * d + (32 if workers > 1 else 0))
            if (int(row["scratch_bytes"]) != expected or int(row["io_bytes"]) != 16 * n * d
                    or int(row["persistent_aux_bytes"]) != 0):
                raise RuntimeError(f"Unexpected memory accounting: {row}")
            rows.append(row)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} workspace queries; saved {args.csv}")
    print("Counts are allocated kernel scratch and Q/K/V/O bytes, not process RSS or DRAM traffic.")


if __name__ == "__main__":
    main()
