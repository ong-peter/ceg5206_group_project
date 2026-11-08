"""Package an explicit C contribution file list, runnable dependencies and evidence.

No git index/commit/remote is changed. Generated binaries and environments are excluded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "855179728f95b755974f7a1ee5ecc6c462bb8771"
NEW = [
    "streaming.cpp", "tests/streaming_test.cpp",
    "scripts/run_member_c.sh", "scripts/memory_report.py",
    "benchmarks/streaming_original.cpp", "benchmarks/compare.cpp",
    "scripts/benchmark_streaming.py", "scripts/profile_streaming.py",
    "docs/member_c.md", "docs/streaming_optimization.md",
    "docs/member_c_submission.md", "scripts/package_member_c.py",
    "docs/integration_2026-11-08.md",
]
MODIFIED = ["attention.h", "main.cpp", "check.py", "Makefile", "README.md", ".gitignore"]
REUSED = ["attention.cpp", "blocked.cpp", "simd.h", "bench.py", "requirements.txt", ".github/workflows/validation.yml"]
HISTORICAL_EVIDENCE = [
    "results/optimization/correctness_serial.csv",
    "results/optimization/correctness_serial.log",
    "results/optimization/correctness_openmp4.csv",
    "results/optimization/correctness_openmp4.log",
    "results/optimization/api_serial.log", "results/optimization/openmp_build.log",
    "results/optimization/ubsan.log",
    "results/optimization/memory_serial.csv", "results/optimization/memory_openmp4.csv",
    "results/optimization/comparison.csv", "results/optimization/comparison.log",
    "results/optimization/environment.json",
    "results/optimization/profile.csv", "results/optimization/profile.log",
    "results/optimization/vectorization.txt",
    "results/optimization/recheck/comparison.csv",
    "results/optimization/recheck/environment.json",
    "results/optimization/reproduction.log",
    "results/asan_startup_sample.txt",
]


EVIDENCE = [
    "results/integration_2026-11-08/validation_summary.json",
    "results/integration_2026-11-08/build_serial.log",
    "results/integration_2026-11-08/correctness_serial.csv",
    "results/integration_2026-11-08/correctness_serial.log",
    "results/integration_2026-11-08/build_openmp.log",
    "results/integration_2026-11-08/correctness_openmp4.csv",
    "results/integration_2026-11-08/correctness_openmp4.log",
    "results/integration_2026-11-08/correctness_novec.csv",
    "results/integration_2026-11-08/correctness_novec.log",
    "results/integration_2026-11-08/ubsan.log",
    "results/integration_2026-11-08/thread_edges.log",
    "results/integration_2026-11-08/thread_failure_guard.log",
    "results/integration_2026-11-08/memory_serial.csv",
    "results/integration_2026-11-08/memory_openmp4.csv",
    "results/integration_2026-11-08/tuned-tiles.json",
    "results/integration_2026-11-08/tuning.csv",
    "results/integration_2026-11-08/tuning.log",
    "results/integration_2026-11-08/bench.csv",
    "results/integration_2026-11-08/bench.machine.json",
    "results/integration_2026-11-08/bench.log",
    "results/integration_2026-11-08/reproduction_serial.log",
    "results/integration_2026-11-08/reproduction_openmp4.log",
]


def git(*args, allowed=(0,)):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if result.returncode not in allowed:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/member_c_submission.zip")
    parser.add_argument("--code-only", action="store_true", help="Omit historical local results")
    args = parser.parse_args()
    categories = [("c_new", NEW), ("c_edits_to_shared_file", MODIFIED), ("reused_unchanged", REUSED)]
    if not args.code_only:
        categories.append(("historical_evidence_2026-10-17", HISTORICAL_EVIDENCE))
        categories.append(("integration_evidence_2026-11-08", EVIDENCE))
    members = {}
    manifest = []
    for category, names in categories:
        for name in names:
            path = ROOT / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Required regular file missing: {name}")
            content = path.read_bytes()
            if category == "reused_unchanged" and content != git("show", f"{BASE}:{name}"):
                raise RuntimeError(f"Reused file differs from the base; review attribution: {name}")
            members[name] = content
            manifest.append(dict(path=name, category=category, bytes=len(content),
                                 sha256=hashlib.sha256(content).hexdigest()))
    # Diff only the declared C paths. The working index is never modified.
    patch = git("diff", "--binary", BASE, "--", *MODIFIED, *NEW)
    base_paths = set(git("ls-tree", "-r", "--name-only", BASE).decode().splitlines())
    tracked = set(git("ls-files").decode().splitlines())
    for name in NEW:
        if name not in base_paths and name not in tracked:
            patch += git("diff", "--no-index", "--binary", "--", "/dev/null", name, allowed=(0, 1))
    description = {
        "base_commit": BASE,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_note": ("Code only; evidence omitted." if args.code_only else
                          "Historical C-only evidence: 2026-10-17. Joint A/B/C integration checks: 2026-11-08. "
                          "Small benchmarks and tuning are not a complete target-server performance study."),
        "patch": "member_c_changes.patch",
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "start_here": "docs/member_c_submission.md",
        "files": manifest,
    }
    members["member_c_changes.patch"] = patch
    members["MANIFEST.json"] = (json.dumps(description, indent=2, ensure_ascii=False) + "\n").encode()
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr("member_c_submission/" + name, content)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        for name, content in members.items():
            if archive.read("member_c_submission/" + name) != content:
                raise RuntimeError(f"ZIP content differs: {name}")
    print(f"Created {destination}: {len(manifest)} source/evidence files + manifest + patch")
    print("No binaries, .venv, .git, commits, remote changes or unlisted files included.")


if __name__ == "__main__":
    main()
