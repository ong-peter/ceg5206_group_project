import argparse
import csv
import io
import itertools
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def reference(q, k, v):
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    scores = (q @ k.T) / math.sqrt(q.shape[1])
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ v


def online(q, k, v, br, bc, dtype):
    q, k, v = (x.astype(dtype) for x in (q, k, v))
    n, d = q.shape
    out = np.empty((n, d), dtype=dtype)
    scale = dtype(1) / np.sqrt(dtype(d))
    for start in range(0, n, br):
        query = q[start:start + br]
        maximum = None
        for key in range(0, n, bc):
            scores = (query @ k[key:key + bc].T) * scale
            new_maximum = scores.max(axis=1, keepdims=True)
            if maximum is not None:
                new_maximum = np.maximum(maximum, new_maximum)
            weights = np.exp(scores - new_maximum)
            block_sum = weights.sum(axis=1, keepdims=True)
            block_values = weights @ v[key:key + bc]
            if maximum is None:
                total, values = block_sum, block_values
            else:
                rescale = np.exp(maximum - new_maximum)
                total = rescale * total + block_sum
                values = rescale * values + block_values
            maximum = new_maximum
        out[start:start + br] = values / total
    return out


def errors(actual, expected, atol=1e-4, rtol=1e-3):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        return float("inf"), float("inf"), False
    delta = actual.astype(np.float64) - expected
    absolute = float(np.max(np.abs(delta)))
    relative = float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-12))
    passed = bool(np.all(np.abs(delta) <= atol + rtol * np.abs(expected)))
    return absolute, relative, passed


def inputs(n, d, seed):
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(-1, 1, (n, d)).astype(np.float32) for _ in range(3))


def cases(full):
    yield "single", inputs(1, 33, 1)
    yield "zeros", tuple(np.zeros((17, 7), dtype=np.float32) for _ in range(3))
    q, k, v = inputs(17, 7, 2)
    yield "zero-scores", (np.zeros_like(q), k, v)
    yield "identical-rows", (np.repeat(q[:1], 17, axis=0), np.repeat(k[:1], 17, axis=0), v)
    yield "tail-127-33", inputs(127, 33, 3)
    yield "tail-513-65", inputs(513, 65, 4)
    q, k, v = inputs(127, 33, 5)
    yield "concentrated", (q * np.float32(10), k * np.float32(10), v)
    _, _, v = inputs(65, 5, 6)
    q = np.full((65, 5), 10, dtype=np.float32)
    k = np.repeat(np.linspace(-10, 10, 65, dtype=np.float32)[:, None], 5, axis=1)
    yield "rising-maximum", (q, k, v)
    yield "early-maximum", (q, k[::-1].copy(), v)
    for n, d in (itertools.product((128, 512, 2048, 4096), (32, 64, 128))
                 if full else ((128, 64), (512, 64))):
        yield "core", inputs(n, d, n + d)


def options(algorithm, settings):
    br, bc, threads = settings
    return ["--algorithm", algorithm, "--br", str(br), "--bc", str(bc), "--threads", str(threads)]


def run(binary, directory, q, k, v, repeats, algorithm, settings):
    n, d = q.shape
    source, target = directory / "input.bin", directory / "output.bin"
    np.concatenate((q.ravel(), k.ravel(), v.ravel())).tofile(source)
    target.unlink(missing_ok=True)
    completed = subprocess.run([str(binary), str(n), str(d), str(source), str(target), str(repeats),
                                *options(algorithm, settings)],
                               text=True, capture_output=True, check=True)
    data = np.fromfile(target, dtype=np.float32)
    if data.size != n * d:
        raise RuntimeError("Incorrect output size")
    timing = next(csv.DictReader(io.StringIO(completed.stdout))) if repeats else {}
    if repeats:
        br, bc, threads = settings
        metadata = dict(algorithm=algorithm, n=n, d=d, br=0 if algorithm == "baseline" else br,
                        bc=0 if algorithm == "baseline" else bc, threads=threads)
        if any(timing.get(key) != str(value) for key, value in metadata.items()):
            raise RuntimeError("Timing metadata does not match the selected configuration")
        if int(timing["scratch_bytes"]) < 0 or int(timing["io_bytes"]) != 16 * n * d:
            raise RuntimeError("Incorrect buffer accounting")
    return data.reshape(n, d), timing


def invalid_inputs(binary, directory, algorithm, settings):
    source, target = directory / "invalid.bin", directory / "invalid-out.bin"
    trials = [("0", "1", b""), ("-1", "1", b""), ("x", "1", b""),
              ("1", "0", b""), ("18446744073709551615", "2", b""),
              ("1", "1", b""), ("1", "1", bytes(16)),
              ("1", "1", np.array([np.nan, 0, 0], dtype=np.float32).tobytes()),
              ("1", "1", np.array([np.inf, 0, 0], dtype=np.float32).tobytes())]
    if algorithm == "baseline":
        trials.append(("1", "1", np.array([3e38, 3e38, 1], dtype=np.float32).tobytes()))
    for n, d, data in trials:
        source.write_bytes(data)
        target.unlink(missing_ok=True)
        result = subprocess.run([str(binary), n, d, str(source), str(target),
                                 *options(algorithm, settings)], capture_output=True)
        if result.returncode != 1 or target.exists():
            raise RuntimeError(f"Invalid input was not rejected cleanly: N={n}, d={d}")
    source.write_bytes(np.zeros(3, dtype=np.float32).tobytes())
    invalid_options = [["--br", "0"], ["--bc", "0"], ["--threads", "0"],
                       ["--threads", "2147483648"], ["--algorithm", "unknown"],
                       ["--unknown", "1"], ["--br"]]
    for extra in invalid_options:
        target.unlink(missing_ok=True)
        result = subprocess.run([str(binary), "1", "1", str(source), str(target),
                                 *options(algorithm, settings), *extra], capture_output=True)
        if result.returncode != 1 or target.exists():
            raise RuntimeError(f"Invalid options were not rejected cleanly: {extra}")
    print(f"{algorithm}: {len(trials) + len(invalid_options)} input/option checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./attention")
    parser.add_argument("--algorithm", choices=("baseline", "blocked", "streaming", "all"), default="baseline")
    parser.add_argument("--br", type=int, default=32)
    parser.add_argument("--bc", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tiles", action="store_true", help="Sweep edge-case tile sizes for optimized kernels")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--bench", action="store_true", help="Time selected kernels at (128,64) and (512,64)")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if min(args.br, args.bc, args.threads) <= 0 or args.threads > 2147483647:
        parser.error("Tile sizes and thread count must be positive; threads must fit int")
    if args.algorithm == "baseline" and args.threads != 1:
        parser.error("Baseline supports one thread only")
    binary = Path(args.binary).resolve()
    available = subprocess.run([str(binary), "--list"], text=True, capture_output=True, check=True).stdout.split()
    if not available or len(available) != len(set(available)) or not set(available) <= {"baseline", "blocked", "streaming"}:
        raise RuntimeError("Invalid kernel list")
    algorithms = available if args.algorithm == "all" else [args.algorithm]
    if any(name not in available for name in algorithms):
        parser.error(f"Algorithm unavailable; compiled kernels: {', '.join(available)}")
    fields = ["case", "algorithm", "n", "d", "br", "bc", "threads", "tile_cases", "max_abs_error", "rel_frobenius_error",
              "passed", "median_ms", "q1_ms", "q3_ms", "batch", "warmups", "repeats",
              "scratch_bytes", "io_bytes", "checksum", "samples_ms"]
    csv_file = args.csv.open("w", newline="") if args.csv else None
    writer = csv.DictWriter(csv_file, fieldnames=fields, lineterminator="\n") if csv_file else None
    if writer:
        writer.writeheader()

    def record(name, algorithm, n, d, metrics, settings=(0, 0, 1), tiles=0, timing=None):
        absolute, relative, passed = metrics
        br, bc, threads = settings
        row = dict(case=name, algorithm=algorithm, n=n, d=d, br=br, bc=bc, threads=threads, tile_cases=tiles,
                   max_abs_error=absolute, rel_frobenius_error=relative, passed=passed)
        if timing:
            row.update(timing)
        if writer:
            writer.writerow(row)
            csv_file.flush()
        configuration = f" Br={br} Bc={bc} T={threads}" if algorithm in ("blocked", "streaming") else ""
        print(f"{algorithm:12} {name:16} {n:4}x{d:<3} abs={absolute:.3e} rel={relative:.3e} "
              f"{'PASS' if passed else 'FAIL'}{configuration}", flush=True)
        if not passed:
            raise RuntimeError(f"Correctness failure: {algorithm}, {name}, {n}x{d}{configuration}")

    try:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for algorithm in algorithms:
                settings = (args.br, args.bc, 1 if algorithm == "baseline" else args.threads)
                invalid_inputs(binary, directory, algorithm, settings)
            for name, (q, k, v) in cases(args.full):
                n, d = q.shape
                expected = reference(q, k, v)
                if name == "single":
                    np.testing.assert_array_equal(expected, v.astype(np.float64))
                if name in ("zeros", "zero-scores", "identical-rows"):
                    mean = np.broadcast_to(v.astype(np.float64).mean(axis=0), (n, d))
                    np.testing.assert_allclose(expected, mean, atol=1e-12, rtol=1e-12)
                tiles = [(1, 1), *itertools.product((16, 32, 64), repeat=2), (n + 1, n + 1)]
                for algorithm in algorithms:
                    pairs = [(args.br, args.bc)]
                    if args.tiles and name != "core" and algorithm != "baseline":
                        pairs = list(dict.fromkeys([*pairs, *tiles]))
                    for br, bc in pairs:
                        settings = (br, bc, 1 if algorithm == "baseline" else args.threads)
                        repeats = 15 if args.bench and name == "core" and d == 64 and n in (128, 512) else 0
                        actual, _ = run(binary, directory, q, k, v, 0, algorithm, settings)
                        metrics = errors(actual, expected)
                        reported = (0, 0, 1) if algorithm == "baseline" else settings
                        timing = {}
                        if not metrics[2]:
                            record(name, algorithm, n, d, metrics, settings=reported)
                        if repeats:
                            timed, timing = run(binary, directory, q, k, v, repeats, algorithm, settings)
                            if not np.array_equal(actual, timed):
                                raise RuntimeError(f"Timed {algorithm} output changed")
                        record(name, algorithm, n, d, metrics, settings=reported, timing=timing)
                        repeated, _ = run(binary, directory, q, k, v, 0, algorithm, settings)
                        if not np.array_equal(actual, repeated):
                            raise RuntimeError(f"Repeated {algorithm} runs produced different outputs")
                if name != "core":
                    for dtype in (np.float64, np.float32):
                        measurements = []
                        for br, bc in tiles:
                            candidate = online(q, k, v, br, bc, dtype)
                            tolerance = (1e-12, 1e-12) if dtype == np.float64 else (1e-4, 1e-3)
                            measurement = errors(candidate, expected, *tolerance)
                            measurements.append(measurement)
                            if not measurement[2]:
                                print(f"Failed tile: Br={br}, Bc={bc}, dtype={dtype.__name__}", flush=True)
                        metrics = (max(m[0] for m in measurements), max(m[1] for m in measurements),
                                   all(m[2] for m in measurements))
                        record(name, "online-fp64" if dtype == np.float64 else "online-fp32",
                               n, d, metrics, tiles=len(tiles))
    finally:
        if csv_file:
            csv_file.close()
    print("All checks passed.", flush=True)


if __name__ == "__main__":
    main()
