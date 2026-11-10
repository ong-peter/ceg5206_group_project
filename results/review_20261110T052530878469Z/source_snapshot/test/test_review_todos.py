"""Orchestration/failure checks, not hardware performance evidence."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import review_todos as review
import check_report_submission as submission
import perf_profile as perf
import numpy as np


class ReviewTests(unittest.TestCase):
    def temporary(self):
        (ROOT / "results").mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=ROOT / "results")

    def engine(self, folder):
        engine = review.Experiments.__new__(review.Experiments)
        engine.out = Path(folder)
        engine.args = SimpleNamespace(rounds=1, repeats=3)
        engine.env = dict(review.os.environ)
        return engine

    def test_source_transforms_fail_if_anchors_drift(self):
        source = (ROOT / "src/streaming.cpp").read_text()
        variant = review.numerical_source(source, True, True)
        self.assertIn("review_pass < 2", variant)
        self.assertNotIn("block_sum += row[j]", variant)
        self.assertIn("total[i] += row[j]", variant)
        with self.assertRaisesRegex(RuntimeError, "insertion point changed"):
            review.numerical_source(source.replace("float block_sum = 0.0f", "float changed = 0"), True)

    def test_smoke_plan_and_existing_directory_protection(self):
        args = review.parse_args(["--only", "R2,R4", "--smoke"])
        self.assertEqual(args.only, ["R2", "R4"])
        self.assertEqual(args.threads, [1, 2])
        self.assertEqual((args.rounds, args.repeats), (1, 3))
        with self.temporary() as temp:
            marker = Path(temp) / "keep.txt"
            marker.write_text("preserve")
            args.out = Path(temp)
            with self.assertRaises(FileExistsError):
                review.Experiments(args)
            self.assertEqual(marker.read_text(), "preserve")

    def test_nan_output_fails_reference_gate(self):
        with self.temporary() as temp:
            engine = self.engine(temp)
            case = dict(n=1, d=1, source=Path(temp) / "input.bin", expected=np.zeros((1, 1)))
            class Process:
                pid, returncode = 1, 0
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def communicate(self, timeout=None):
                    np.array([np.nan], dtype=np.float32).tofile(Path(temp) / "last_output.bin")
                    return "", ""
            with patch.object(review.subprocess, "Popen", return_value=Process()):
                with self.assertRaisesRegex(RuntimeError, "FP64-reference validation failed"):
                    engine.invoke(review.Experiments.job("bad", Path("fake")), case, 0, Path(temp), "bad")

    def test_timed_bit_change_rejected_even_within_reference_tolerance(self):
        with self.temporary() as temp:
            engine = self.engine(temp)
            actual = np.ones((1, 1), dtype=np.float32)
            changed = np.nextafter(actual, np.float32(2))
            self.assertTrue(review.errors(changed, actual.astype(np.float64))[2])
            metrics = [dict(max_abs_error=0.0, rel_frobenius_error=0.0,
                            output_sha256=review.digest(x), diagnostic="") for x in [actual, changed]]
            with patch.object(engine, "invoke", side_effect=[(actual, {}, metrics[0]), (changed, {}, metrics[1])]):
                with self.assertRaisesRegex(RuntimeError, "changed bitwise"):
                    engine.measure("R2", dict(n=1, d=1, name="case"),
                                   [engine.job("variant", Path("fake"))])
            self.assertFalse((Path(temp) / "R2/summary.csv").exists())

    def test_thread_bit_change_rejected(self):
        with self.temporary() as temp:
            engine = self.engine(temp)
            actual = np.ones((1, 1), dtype=np.float32)
            changed = np.nextafter(actual, np.float32(2))
            metric = dict(max_abs_error=0.0, rel_frobenius_error=0.0,
                          output_sha256=review.digest(actual), diagnostic="")
            with patch.object(engine, "invoke", side_effect=[(actual, {}, metric), (changed, {}, metric)]):
                with self.assertRaisesRegex(RuntimeError, "bitwise invariant"):
                    engine.measure("R2", dict(n=1, d=1, name="case"),
                                   [engine.job("variant", Path("fake"), threads=2)])

    def test_submission_check_ignores_commented_markers(self):
        with self.temporary() as temp:
            folder = Path(temp) / "report"
            folder.mkdir()
            (folder / "main.tex").write_text("% \\todo{S1}{old}\n\\todo{S2}{verify}\n"
                                              "text % \\todo{S3}{hidden}\n"
                                              "50\\% \\todo{S4}{visible}\n")
            with patch.object(submission.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
                todos, problems = submission.check(Path(temp))
            self.assertEqual(todos, [(2, "S2"), (4, "S4")])
            self.assertTrue(any("main.pdf" in item for item in problems))

    def test_counter_pass_pairing_extra_groups_and_tile_overrides(self):
        medians = {"cache": 2.0, "l2": 4.0, "fp": 8.0, "dram": 100.0, "extra": 200.0}
        values = {"cycles": 4_000_000, "instructions": 8_000_000, "L1-dcache-loads": 100,
                  "L1-dcache-load-misses": 1, "LLC-loads": 10, "LLC-load-misses": 2,
                  "l2_rqsts.references": 20, "l2_rqsts.miss": 5,
                  "fp_arith_inst_retired.scalar_single": 1,
                  "fp_arith_inst_retired.128b_packed_single": 2,
                  "fp_arith_inst_retired.256b_packed_single": 3,
                  "uncore_imc/data_reads/": 123, "uncore_imc/data_writes/": 456, "review.extra": 7}
        def fake_perf(events, command, system_wide=False):
            group = next((k for k, v in perf.GROUPS.items() if v == events), "extra")
            if command[5] == "0":
                return {event: 17 for event in events}, ""
            self.assertEqual(command[command.index("--br") + 1], "128")
            self.assertEqual(command[command.index("--bc") + 1], "32")
            return ({event: 17 + 21 * values[event] for event in events},
                    "median_ms,batch,repeats,warmups\n%s,1,15,5\n" % medians[group])
        def fake_run(command, **kwargs):
            stdout = "baseline\n" if "--list" in command else "scratch_bytes,io_bytes\n4,16\n"
            return subprocess.CompletedProcess(command, 0, stdout, "")
        with self.temporary() as temp:
            for mode in ["all", "cache-only"]:
                target = Path(temp) / (mode + ".csv")
                argv = ["perf_profile.py", "--shapes", "1x1", "--csv", str(target), "--br", "128",
                        "--bc", "32", "--extra-events", "review.extra"]
                def enabled(events, system_wide=False):
                    return mode == "all" or events == perf.GROUPS["cache"]
                with patch.object(sys, "argv", argv), patch.object(perf.shutil, "which", return_value="perf"), \
                     patch.object(perf.tempfile, "tempdir", temp), \
                     patch.object(perf, "available", side_effect=enabled), \
                     patch.object(perf, "perf_stat", side_effect=fake_perf), \
                     patch.object(perf.subprocess, "run", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                    perf.main()
                with target.open() as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(float(row["median_ms"]), 2.0)
                self.assertEqual(float(row["ghz_per_thread"]), 2.0)
                self.assertEqual(json.loads(row["extra_counts_json"]), {"review.extra": 7} if mode == "all" else {})
                passes = [json.loads(line) for line in target.with_suffix(".passes.jsonl").read_text().splitlines()]
                self.assertEqual(len(passes), 5 if mode == "all" else 1)
                self.assertTrue(all(p["extra_kernel_calls"] == 21 and p["started_utc"] for p in passes))


if __name__ == "__main__":
    unittest.main()
