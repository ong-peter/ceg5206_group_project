"""Run the report's R1--R6 experiments; see docs/report_todos.md.

Experimental sources/binaries, input data, validation, raw samples and provenance
are saved in a new output directory. Production kernel sources are never edited.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
# Import bench first: it restores affinity after importing MKL NumPy.
from bench import machine_info, physical_cores, thread_list  # noqa: E402
sys.path.insert(0, str(ROOT / "test"))
from check import cases, errors, inputs, options, reference  # noqa: E402
import numpy as np  # noqa: E402

TASKS = {
    "R1": "Corrected counters and independent clock-log association",
    "R2": "Query-tile sweep, scratch and optional cache counters",
    "R3": "Interleaved packed-blocked/streaming comparison and serial stages",
    "R4": "Numerical score/summation/rescaling factorial ablation (untimed)",
    "R5": "Independent-process rounds with system/process telemetry",
    "R6": "Controlled buffer offsets and function-alignment builds",
}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Diagnostic insertion point changed: " + old[:90])
    return text.replace(old, new, 1)


def numerical_source(source, sequential_sum=False, global_max=False):
    """Change one numerical factor at a time in a generated source copy."""
    if sequential_sum:
        source = replace_once(source, "            float block_sum = 0.0f;",
                              "            total[i] *= alpha;")
        source = replace_once(source, "                block_sum += row[j];",
                              "                total[i] += row[j];")
        source = replace_once(source, "            total[i] = alpha * total[i] + block_sum;", "")
    if global_max:
        # First pass computes maxima with EXACTLY the selected score micro-kernel.
        # The second pass recomputes scores, now using the final maximum throughout.
        source = replace_once(source, "    for (std::size_t key = 0; key < n;)",
                              "    for (int review_pass = 0; review_pass < 2; ++review_pass) {\n"
                              "    for (std::size_t key = 0; key < n;)")
        point = "        for (std::size_t i = 0; i < rows; ++i) {\n            float* row = scores + i * work.bc;"
        source = replace_once(source, point, """        if (review_pass == 0) {
            for (std::size_t i = 0; i < rows; ++i) {
                const float* row = scores + i * work.bc;
                maximum[i] = std::max(maximum[i], *std::max_element(row, row + columns));
            }
            key += columns;
            continue;
        }
""" + point)
        source = replace_once(source, "        key += columns;\n    }\n\n    for",
                              "        key += columns;\n    }\n    }\n\n    for")
    return source


ALIGNMENT_BUFFER = r'''
#include <cstdint>
#include <cstdlib>
// Diagnostic buffer with an explicitly selected offset from a 64-byte boundary.
// Allocation/copying happen before the production timing loop.
class ReviewBuffer {
    std::vector<float> storage;
    std::size_t count;
    float* pointer;
public:
    explicit ReviewBuffer(std::size_t n) : storage(n + 32), count(n) {
        const char* raw = std::getenv("REVIEW_OFFSET_BYTES");
        const int offset = raw ? std::stoi(raw) : 0;
        if (offset < 0 || offset >= 64 || offset % sizeof(float))
            throw std::invalid_argument("Review offset must be 0,4,...,60 bytes");
        const auto aligned = (reinterpret_cast<std::uintptr_t>(storage.data()) + 63) & ~std::uintptr_t(63);
        pointer = reinterpret_cast<float*>(aligned + offset);
        if (reinterpret_cast<std::uintptr_t>(pointer) % 64 != static_cast<unsigned>(offset))
            throw std::runtime_error("Unexpected diagnostic alignment");
    }
    ReviewBuffer(const ReviewBuffer& other) : ReviewBuffer(other.count) {
        std::copy(other.begin(), other.end(), begin());
    }
    ReviewBuffer& operator=(const ReviewBuffer&) = delete;
    float* data() { return pointer; }
    const float* data() const { return pointer; }
    float* begin() { return pointer; }
    const float* begin() const { return pointer; }
    float* end() { return pointer + count; }
    const float* end() const { return pointer + count; }
    bool operator!=(const ReviewBuffer& other) const {
        return count != other.count || !std::equal(begin(), end(), other.begin());
    }
};
'''


def alignment_source(source):
    source = replace_once(source, "using Clock =", ALIGNMENT_BUFFER + "\nusing Clock =")
    source = replace_once(source, "std::vector<float> data(input_count);", "ReviewBuffer data(input_count);")
    source = replace_once(source, "std::vector<float> out(nd), scratch(scratch_count);",
                          "ReviewBuffer out(nd), scratch(scratch_count);")
    point = "        const float* v = k + nd;"
    return replace_once(source, point, point + r'''
        std::cerr << "alignment_mod64 q=" << reinterpret_cast<std::uintptr_t>(q) % 64
                  << " k=" << reinterpret_cast<std::uintptr_t>(k) % 64
                  << " v=" << reinterpret_cast<std::uintptr_t>(v) % 64
                  << " out=" << reinterpret_cast<std::uintptr_t>(out.data()) % 64
                  << " scratch=" << reinterpret_cast<std::uintptr_t>(scratch.data()) % 64 << '\n';
''')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_csv(path, row):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def digest(data):
    return hashlib.sha256(data.tobytes()).hexdigest()


def system_snapshot(pid=None):
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    value = dict(utc=datetime.now(timezone.utc).isoformat(), monotonic=time.monotonic(),
                 loadavg=read("/proc/loadavg"), proc_stat=read("/proc/stat"))
    value["current_khz"] = {p.parent.parent.name: read(p) for p in
                            Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")}
    value["thermal_millidegrees"] = {str(p): read(p) for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")}
    if pid is not None:
        value.update(pid=pid, status=read(f"/proc/{pid}/status"))
        try:
            value["affinity"] = sorted(os.sched_getaffinity(pid))
        except (AttributeError, OSError):
            value["affinity"] = None
    return value


class Experiments:
    def __init__(self, args):
        self.args, self.out = args, args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.bin = self.out / "bin"
        self.bin.mkdir()
        self.data = self.out / "inputs"
        self.data.mkdir()
        self.env = dict(os.environ, OMP_PLACES="cores", OMP_PROC_BIND="close", OMP_DYNAMIC="FALSE")
        self.compiler = shlex.split(args.cxx)
        self.flags = shlex.split(args.cxxflags)
        self.ldflags = shlex.split(args.ldflags)
        self.binaries, self.builds, self.case_cache = {}, {}, {}
        self.manifest = dict(smoke=args.smoke, tasks=args.only, arguments=vars(args).copy(),
                             machine=machine_info(self.compiler[0]), builds=self.builds,
                             started_utc=datetime.now(timezone.utc).isoformat(), status="running")
        self.manifest["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        self.manifest["git_commit"] = self.capture(["git", "rev-parse", "HEAD"])
        self.manifest["git_status"] = self.capture(["git", "status", "--short"])
        (self.out / "uncommitted.patch").write_text(self.capture(["git", "diff"]), encoding="utf-8")
        # Also preserve untracked experimental sources, which git diff omits.
        snapshot = self.out / "source_snapshot"
        for folder in ["src", "scripts", "test"]:
            for path in (ROOT / folder).rglob("*"):
                if path.is_file() and path.suffix in (".cpp", ".h", ".py", ".sh"):
                    target = snapshot / path.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
        write_json(self.out / "manifest.json", self.manifest)

    @staticmethod
    def capture(command):
        try:
            return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    def command(self, command, logfile):
        with logfile.open("w", encoding="utf-8") as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=self.env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}); see {logfile}")

    def build(self, name="default", defines=(), streaming=None, driver=None, extra_flags=()):
        if name in self.binaries:
            return self.binaries[name]
        folder = self.bin / name
        folder.mkdir()
        sources = {}
        for filename in ["attention.cpp", "blocked.cpp", "streaming.cpp", "main.cpp"]:
            text = (ROOT / "src" / filename).read_text()
            if filename == "streaming.cpp" and streaming is not None:
                text = streaming
            if filename == "main.cpp" and driver is not None:
                text = driver
            path = folder / filename
            path.write_text(text, encoding="utf-8")
            sources[filename] = path
        binary = folder / ("attention.exe" if os.name == "nt" else "attention")
        flags = [*self.flags, *extra_flags, "-DWITH_BLOCKED", "-DWITH_STREAMING", *defines]
        command = [*self.compiler, *flags, "-I", str(ROOT / "src"),
                   '-DATTN_BUILD_FLAGS="' + " ".join(self.compiler + flags) + '"',
                   *map(str, sources.values()), *self.ldflags, "-o", str(binary)]
        print(f"Building {name}", flush=True)
        self.command(command, folder / "build.log")
        info = self.capture([str(binary), "--build-info"])
        if "openmp=1" not in info:
            raise RuntimeError("Review experiments require an OpenMP build")
        self.builds[name] = dict(command=command, build_info=info, source_sha256={
            k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sources.items()})
        write_json(self.out / "manifest.json", self.manifest)
        self.binaries[name] = binary
        return binary

    def case(self, n, d, seed=0, special=None):
        key = f"{special[0] if special else 'random'}_{n}x{d}_s{seed}"
        if key not in self.case_cache:
            q, k, v = inputs(n, d, n + d + seed) if special is None else special[1]
            name = key if special is None else f"{special[0]}_{n}x{d}"
            expected = reference(q, k, v)
            folder = self.data / name
            folder.mkdir(exist_ok=True)
            np.concatenate([q.ravel(), k.ravel(), v.ravel()]).tofile(folder / "input.bin")
            expected.tofile(folder / "reference.bin")
            self.case_cache[key] = dict(n=n, d=d, seed=seed, name=name, expected=expected,
                                        folder=folder, source=folder / "input.bin")
        return self.case_cache[key]

    def invoke(self, job, case, repeats, folder, label, telemetry=False):
        output = folder / "last_output.bin"
        output.unlink(missing_ok=True)
        command = [str(job["binary"]), str(case["n"]), str(case["d"]), str(case["source"]),
                   str(output), str(repeats), *options(job["algorithm"], (job["br"], job["bc"], job["threads"]))]
        environment = dict(self.env, **job.get("env", {}))
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) as process:
            snapshots = [system_snapshot(process.pid)] if telemetry else []
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.5 if telemetry else None)
                    break
                except subprocess.TimeoutExpired:
                    snapshots.append(system_snapshot(process.pid))
            if telemetry:
                snapshots.append(system_snapshot())
                with (folder / "telemetry.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(label=label, command=command, samples=snapshots)) + "\n")
        if process.returncode:
            raise RuntimeError(f"{label}: kernel failed: {stderr}")
        actual = np.fromfile(output, dtype=np.float32)
        if actual.size != case["n"] * case["d"]:
            raise RuntimeError(f"{label}: wrong output size")
        actual = actual.reshape(case["n"], case["d"])
        absolute, relative, passed = errors(actual, case["expected"])
        if not passed:
            raise RuntimeError(f"{label}: FP64-reference validation failed: abs={absolute} rel={relative}")
        timing = next(csv.DictReader(io.StringIO(stdout))) if repeats else {}
        if repeats:
            for key, value in dict(algorithm=job["algorithm"], n=case["n"], d=case["d"],
                                   br=0 if job["algorithm"] == "baseline" else job["br"],
                                   bc=0 if job["algorithm"] == "baseline" else job["bc"], threads=job["threads"]).items():
                if timing.get(key) != str(value):
                    raise RuntimeError(f"{label}: incorrect timing metadata: {key}")
            values = [float(x) for x in timing["samples_ms"].split(";")]
            if len(values) != repeats or any(not math.isfinite(x) or x <= 0 for x in values):
                raise RuntimeError(f"{label}: invalid timing samples")
        return actual, timing, dict(max_abs_error=absolute, rel_frobenius_error=relative,
                                    output_sha256=digest(actual), diagnostic=stderr.strip())

    @staticmethod
    def job(name, binary, algorithm="streaming", br=64, bc=32, threads=1, **extra):
        return dict(name=name, binary=binary, algorithm=algorithm, br=br, bc=bc, threads=threads, **extra)

    def measure(self, task, case, jobs, rounds=None, telemetry=False, timed=True):
        folder = self.out / task
        folder.mkdir(exist_ok=True)
        validated, collected, metrics = {}, {}, {}
        for job in jobs:
            name = job["name"]
            actual, _, metric = self.invoke(job, case, 0, folder, name + ":gate")
            if job["threads"] > 1:
                one, _, _ = self.invoke(dict(job, threads=1), case, 0, folder, name + ":one-thread")
                if digest(one) != digest(actual):
                    raise RuntimeError(f"{name}: threaded output is not bitwise invariant")
            validated[name], collected[name], metrics[name] = metric["output_sha256"], [], metric
            append_csv(folder / "validation.csv", dict(case=case["name"], variant=name,
                       algorithm=job["algorithm"], br=job["br"], bc=job["bc"], threads=job["threads"], **metric))
        if not timed:
            return
        for round_index in range(rounds or self.args.rounds):
            shift = round_index % len(jobs)
            for job in jobs[shift:] + jobs[:shift]:
                name = job["name"]
                actual, timing, metric = self.invoke(job, case, self.args.repeats, folder,
                                                      f"{name}:round{round_index + 1}", telemetry)
                if metric["output_sha256"] != validated[name]:
                    raise RuntimeError(f"{name}: timed output changed bitwise from validation")
                values = [float(x) for x in timing["samples_ms"].split(";")]
                collected[name].extend(values)
                row = dict(case=case["name"], n=case["n"], d=case["d"], variant=name,
                           algorithm=job["algorithm"], br=job["br"], bc=job["bc"], threads=job["threads"],
                           round=round_index + 1, median_ms=float(np.median(values)),
                           q1_ms=float(np.percentile(values, 25)), q3_ms=float(np.percentile(values, 75)),
                           scratch_bytes=int(timing["scratch_bytes"]), samples=len(values),
                           samples_ms=";".join(map(str, values)), **metric)
                append_csv(folder / "rounds.csv", row)
                print(f"{task} {case['name']} {name} round={round_index + 1} "
                      f"median={row['median_ms']:.4f} Q1={row['q1_ms']:.4f} Q3={row['q3_ms']:.4f} ms", flush=True)
        for job in jobs:
            values = collected[job["name"]]
            append_csv(folder / "summary.csv", dict(case=case["name"], variant=job["name"],
                       algorithm=job["algorithm"], br=job["br"], bc=job["bc"], threads=job["threads"],
                       median_ms=float(np.median(values)), q1_ms=float(np.percentile(values, 25)),
                       q3_ms=float(np.percentile(values, 75)), samples=len(values)))

    def counters(self, folder, shapes, br=None):
        folder.mkdir(exist_ok=True)
        command = [sys.executable, str(ROOT / "scripts/perf_profile.py"), "--binary", str(self.build()),
                   "--shapes", ",".join(f"{n}x{d}" for n, d in shapes),
                   "--threads", ",".join(map(str, self.args.threads)), "--repeats", str(self.args.repeats),
                   "--tile-config", str(self.args.tiles.resolve()), "--csv", str(folder / "counters.csv")]
        if br is not None:
            command += ["--br", str(br), "--bc", "32"]
        if self.args.events:
            command += ["--extra-events", self.args.events]
        self.command(command, folder / "counters.log")
        rows = list(csv.DictReader((folder / "counters.csv").open()))
        missing = sorted({field for field in ["l2_references", "dram_read_bytes", "fp_flops"]
                          if any(row[field] == "" for row in rows)})
        write_json(folder / "counter_coverage.json", dict(missing_fields=missing,
                   extra_events_requested=self.args.events,
                   extra_group_present=all(row.get("extra_median_ms", "") != "" for row in rows)
                   if self.args.events else None, all_standard_groups_present=not missing))
        if missing:
            print("Counter coverage is partial; inspect " + str(folder / "counter_coverage.json"), flush=True)

    def r1(self):
        if platform.system() != "Linux":
            raise RuntimeError("R1 requires Linux perf on the original benchmark machine")
        folder = self.out / "R1"
        folder.mkdir()
        # perf_profile itself is not a reference validator: gate it explicitly.
        for t in self.args.threads:
            command = [sys.executable, str(ROOT / "test/check.py"), "--binary", str(self.build()),
                       "--algorithm", "all", "--tiles", "--threads", str(t),
                       "--tile-config", str(self.args.tiles.resolve()), "--csv", str(folder / f"correctness_t{t}.csv")]
            if not self.args.smoke:
                command.append("--full")
            self.command(command, folder / f"correctness_t{t}.log")
        shapes = [(128, 128), (512, 64), (4096, 64), (4096, 128)] if not self.args.smoke else [(128, 64)]
        self.counters(folder, shapes)
        if not self.args.clock_log:
            print("R1: no independent clock log supplied; clock verification remains pending", flush=True)

    def r2(self):
        binary = self.build()
        shapes = [(4096, 64), (4096, 128)] if not self.args.smoke else [(65, 17)]
        for n, d in shapes:
            jobs = [self.job(f"{alg}_br{br}_t{t}", binary, alg, br=br, threads=t)
                    for br in [32, 64, 128, 256] for alg in ["blocked", "streaming"] for t in self.args.threads]
            self.measure("R2", self.case(n, d), jobs)
        if self.args.cache_counters:
            for br in [32, 64, 128, 256]:
                self.counters(self.out / "R2" / f"br{br}", shapes, br)

    def r3(self):
        packed = self.build("packed", defines=["-DBLOCKED_PACK_K"])
        jobs = [self.job(f"{alg}_t{t}", packed, alg, threads=t)
                for alg in ["blocked", "streaming"] for t in self.args.threads]
        shapes = list(itertools.product([2048, 4096], [32, 64, 128])) if not self.args.smoke else [(65, 17)]
        for n, d in shapes:
            self.measure("R3", self.case(n, d), jobs)
        # Intrusive serial stage timings are diagnostic, never primary timings.
        from profile_stages import DRIVER, STAGES, instrument_blocked, instrument_streaming
        folder = self.out / "R3" / "stages"
        folder.mkdir()
        (folder / "driver.cpp").write_text(DRIVER, encoding="utf-8")
        for alg, instrument in [("blocked", instrument_blocked), ("streaming", instrument_streaming)]:
            source = folder / f"{alg}.cpp"
            source.write_text(instrument((ROOT / "src" / f"{alg}.cpp").read_text()), encoding="utf-8")
            binary = folder / (alg + (".exe" if os.name == "nt" else ""))
            command = [*self.compiler, *self.flags, "-DBLOCKED_PACK_K", f"-DPROFILE_{alg.upper()}",
                       "-I", str(ROOT / "src"), str(folder / "driver.cpp"), str(ROOT / "src/attention.cpp"),
                       str(source), *self.ldflags, "-o", str(binary)]
            self.command(command, folder / f"{alg}_build.log")
            write_json(folder / f"{alg}_build.json", dict(command=command))
            for n, d in shapes:
                case = self.case(n, d)
                command = [str(binary), str(n), str(d), "64", "32", str(self.args.repeats),
                           str(case["source"]), str(case["folder"] / "reference.bin")]
                result = subprocess.run(command, text=True, capture_output=True, env=self.env, check=True)
                *stages, mean, median = map(float, result.stdout.strip().split(","))
                for stage, ms in zip(STAGES[alg], stages):
                    append_csv(folder / "stages.csv", dict(algorithm=alg, n=n, d=d, threads=1, stage=stage,
                               stage_ms=ms, instrumented_mean_ms=mean, instrumented_median_ms=median))

    def r4(self):
        source = (ROOT / "src/streaming.cpp").read_text()
        jobs = [self.job("baseline", self.build(), "baseline"), self.job("blocked", self.build(), "blocked")]
        for scalar, sequential, global_max in itertools.product([False, True], repeat=3):
            name = f"scores{'scalar' if scalar else 'packed'}_sum{'seq' if sequential else 'tile'}_max{'global' if global_max else 'online'}"
            binary = self.build(name, defines=["-DSTREAMING_SCALAR_SCORES"] if scalar else [],
                                streaming=numerical_source(source, sequential, global_max))
            jobs.append(self.job(name, binary))
        shapes = list(itertools.product([128, 512, 2048, 4096], [32, 64, 128])) if not self.args.smoke else [(17, 9), (33, 17)]
        for n, d in shapes:
            for seed in self.args.seeds if not self.args.smoke else [0]:
                self.measure("R4", self.case(n, d, seed), jobs, timed=False)
        for name, data in cases(False):
            if name in ["concentrated", "rising-maximum", "early-maximum"]:
                n, d = data[0].shape
                self.measure("R4", self.case(n, d, special=(name, data)), jobs, timed=False)
        print("R4: reference-error measurements saved; diagnostic variants are untimed", flush=True)

    def r5(self):
        binary = self.build()
        jobs = [self.job("baseline_t1", binary, "baseline")]
        jobs += [self.job(f"{alg}_t{t}", binary, alg, threads=t)
                 for alg in ["blocked", "streaming"] for t in self.args.threads]
        self.measure("R5", self.case(4096, 32) if not self.args.smoke else self.case(65, 17),
                     jobs, rounds=1 if self.args.smoke else max(6, self.args.rounds), telemetry=True)

    def r6(self):
        main = alignment_source((ROOT / "src/main.cpp").read_text())
        jobs = []
        for alignment in [16, 64]:
            binary = self.build(f"align{alignment}", driver=main,
                                extra_flags=[f"-falign-functions={alignment}"])
            objdump = shutil.which("objdump")
            if objdump:
                self.command([objdump, "-d", str(binary)], binary.parent / "disassembly.txt")
            for offset in [0, 4, 16, 32, 60]:
                for alg in ["blocked", "streaming"]:
                    jobs.append(self.job(f"{alg}_align{alignment}_offset{offset}", binary, alg,
                                         env={"REVIEW_OFFSET_BYTES": str(offset)}))
        self.measure("R6", self.case(128, 128) if not self.args.smoke else self.case(33, 17), jobs,
                     rounds=1 if self.args.smoke else self.args.processes)

    def run(self):
        try:
            for task in self.args.only:
                print(f"\n{task}: {TASKS[task]}", flush=True)
                getattr(self, task.lower())()
            self.manifest["status"] = "measurements_completed_pending_interpretation"
        except BaseException as error:
            self.manifest.update(status="failed", error=str(error))
            raise
        finally:
            self.manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            self.manifest["machine_after"] = machine_info(self.compiler[0])
            if self.args.clock_log:
                clock = self.args.clock_log.resolve()
                self.manifest["clock_log"] = dict(path=str(clock), exists=clock.is_file(),
                                                  bytes=clock.stat().st_size if clock.is_file() else 0)
            else:
                self.manifest["clock_log"] = dict(path=None, note="No independent clock log associated")
            write_json(self.out / "manifest.json", self.manifest)
        print(f"Saved {self.out}\nReview evidence and update report TODOs manually; none are automatically cleared.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="R1", help="R1,...,R6 or all (default: R1)")
    parser.add_argument("--out", type=Path, help="New directory; existing directories are rejected")
    parser.add_argument("--tiles", type=Path, default=ROOT / "results/benchmark_20261109T113037Z/tuned-tiles.json")
    parser.add_argument("--threads", type=thread_list, help="Counts for R1/R2/R3/R5 (default: 1,physical cores)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--processes", type=int, default=6, help="Independent processes per R6 variant")
    parser.add_argument("--seeds", type=thread_list, default=[1, 2, 3], help="Positive seed offsets for R4")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--cxxflags", default="-O3 -std=c++17 -march=native -fopenmp-simd -fopenmp")
    parser.add_argument("--ldflags", default="-fopenmp")
    parser.add_argument("--cache-counters", action="store_true", help="Also collect counters per R2 tile")
    parser.add_argument("--events", default="", help="Optional CPU-specific perf events; unsupported groups are logged")
    parser.add_argument("--clock-log", type=Path, help="Path to an independent turbostat log (see run guide)")
    parser.add_argument("--smoke", action="store_true", help="Small shapes/one round: validation only, not report results")
    parser.add_argument("--plan", action="store_true", help="Print the requested experiment plan without running")
    args = parser.parse_args(argv)
    args.only = list(TASKS) if args.only.lower() == "all" else list(dict.fromkeys(x.strip().upper() for x in args.only.split(",")))
    if not args.only or any(x not in TASKS for x in args.only):
        parser.error("--only must contain R1,...,R6 or all")
    if min(args.rounds, args.repeats, args.processes) <= 0:
        parser.error("rounds, repeats and processes must be positive")
    if args.smoke:
        args.rounds, args.repeats = 1, min(args.repeats, 3)
    args.threads = args.threads or ([1, 2] if args.smoke else sorted({1, physical_cores()}))
    if 1 not in args.threads:
        parser.error("Include 1 in --threads for a same-kernel reference")
    if args.cache_counters and platform.system() != "Linux" and not args.plan:
        parser.error("--cache-counters needs Linux perf")
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.out = ROOT / "results" / (("review_smoke_" if args.smoke else "review_") + stamp)
    return args


def main():
    args = parse_args()
    if args.plan:
        print(json.dumps({task: TASKS[task] for task in args.only}, indent=2))
        return
    if ("R1" in args.only or args.cache_counters) and platform.system() != "Linux":
        raise SystemExit("Counter experiments require Linux perf on the benchmark machine; use --plan here")
    if not args.tiles.is_file() and ("R1" in args.only or args.cache_counters):
        raise SystemExit("Saved tiles are missing; use --tiles PATH")
    Experiments(args).run()


if __name__ == "__main__":
    main()
