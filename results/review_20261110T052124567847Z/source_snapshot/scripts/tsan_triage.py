"""Classify ThreadSanitizer reports from scripts/tsan_check.sh.

A data race where one access runs outside every parallel region is ordered by
the region's fork/join; GCC's libgomp hides that synchronization from TSan, so
these are reported as "fork-join" false positives. Any other report (both
accesses inside parallel regions, or a non-race warning) is listed for review.
With Clang and Archer the runtime's synchronization is visible, so every report
needs review.
"""
import argparse
import collections
from pathlib import Path
import re

REGION = re.compile(r"_omp_fn\.|\.omp_outlined\.|__kmp_invoke_microtask")
ACCESS = re.compile(r"^  (?:Previous )?(?:[Rr]ead|[Ww]rite|Atomic \w+) of size \d+ at \S+ by "
                    r"(main thread|thread T\d+)")
FRAME = re.compile(r"#\d+ (.+?) (\S+:\d+)")


def reports(paths):
    for path in paths:
        for block in path.read_text(errors="replace").split("==================\n"):
            if "WARNING: ThreadSanitizer" in block:
                yield path, block


def accesses(block):
    # Sections start with a two-space indent; only access stacks are kept.
    result = []
    for section in re.split(r"\n(?=  \S)", block):
        match = ACCESS.match(section)
        if match:
            frames = FRAME.findall(section)
            in_region = any(REGION.search(name) for name, _ in frames)
            location = next((place for name, place in frames
                             if not place.startswith(("/usr/", "/build/", "../"))
                             and "sanitizer" not in place), "?")
            result.append((match.group(1), in_region, location))
    return result


def classify(block, toolchain):
    kind = re.search(r"WARNING: ThreadSanitizer: ([^(\n]+)", block).group(1).strip()
    pair = accesses(block)
    if toolchain == "clang":
        return "review", kind, pair
    if kind == "data race" and len(pair) >= 2 and not (pair[0][1] and pair[1][1]):
        return "fork-join", kind, pair
    return "review", kind, pair


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", type=Path, help="Directory of TSan log_path files")
    parser.add_argument("--toolchain", choices=("clang", "gcc"), required=True)
    parser.add_argument("--samples", type=Path, help="Write one full report per review group here")
    args = parser.parse_args()
    files = sorted(p for p in args.reports.iterdir() if p.is_file())
    counts = collections.Counter()
    groups = collections.Counter()
    samples = {}
    for path, block in reports(files):
        verdict, kind, pair = classify(block, args.toolchain)
        counts[verdict] += 1
        key = (verdict, kind, tuple((who == "main thread", region, place) for who, region, place in pair[:2]))
        groups[key] += 1
        if verdict == "review":
            samples.setdefault(key, f"# {path.name}\n{block}")
    total = sum(counts.values())
    print(f"toolchain: {args.toolchain}")
    print(f"report files: {len(files)}")
    print(f"warnings: {total}; fork-join false positives: {counts['fork-join']}; "
          f"need review: {counts['review']}")
    for (verdict, kind, pair), n in groups.most_common():
        sides = " <-> ".join(f"{'main' if main else 'worker'}"
                             f"{' in region' if region else ''} @ {place}" for main, region, place in pair)
        print(f"{n:7d}  {verdict:9s} {kind}: {sides}")
    if args.samples and samples:
        args.samples.write_text("\n==================\n".join(samples.values()))
        print(f"review samples: {args.samples}")
    return 1 if counts["review"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
