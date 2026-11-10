"""PyTorch reference timings for comparison with our kernels.

For each of the 12 core shapes, times PyTorch's default CPU attention,
    torch.nn.functional.scaled_dot_product_attention(q, k, v),
on the same FP32 inputs as the benchmark ladder (test/check.py, seed n+d),
with PyTorch's default settings (including its default thread count).
After 5 warm-up calls, 15 calls are timed individually (as in src/main.cpp);
the CSV reports median/Q1/Q3 in ms and the error against the FP64 reference.
"""
import argparse
import csv
import itertools
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))
from check import errors, inputs, reference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--csv", type=Path, default=Path("results/torch_sdpa.csv"))
    args = parser.parse_args()

    rows = []
    for n, d in itertools.product((128, 512, 2048, 4096), (32, 64, 128)):
        q, k, v = inputs(n, d, n + d)
        qt, kt, vt = (torch.from_numpy(x) for x in (q, k, v))

        for _ in range(args.warmups):
            F.scaled_dot_product_attention(qt, kt, vt)
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            out = F.scaled_dot_product_attention(qt, kt, vt)
            times.append(1000 * (time.perf_counter() - start))

        max_abs, rel, _ = errors(out.numpy(), reference(q, k, v))
        q1, median, q3 = statistics.quantiles(times, n=4, method="inclusive")
        rows.append(dict(n=n, d=d, algorithm="torch-sdpa", threads=torch.get_num_threads(),
                         median_ms=median, q1_ms=q1, q3_ms=q3, max_abs_error=max_abs,
                         rel_frobenius_error=rel, samples=len(times), torch=torch.__version__))
        print(f"{n:>5}x{d:<3} median {median:9.3f} ms  (Q1 {q1:.3f}, Q3 {q3:.3f})", flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.csv} (PyTorch default threads: {torch.get_num_threads()})")


if __name__ == "__main__":
    main()
