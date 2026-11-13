"""List unresolved report TODOs and check tracked submission assets.

This local check cannot attest to authorship, contribution accuracy, video
quality or another person's access. See the S1--S6 checklist in the run guide.
"""
import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANUAL = {
    "S1": "Confirm original/adapted/AI-assisted figure provenance and attribution.",
    "S2": "Have every member confirm completed contributions and reused code.",
    "S3": "Add the clickable 10--15 minute video link; test instructor/TA access.",
    "S4": "Set a final release tag/commit identifying the assessed package.",
    "S5": "Complete the actual AI tools, uses and human-verification declaration.",
    "S6": "Check public repository/video access and reproduce from the final checkout.",
}


def check(root=ROOT):
    text = (root / "report/main.tex").read_text(encoding="utf-8")
    # Strip LaTeX comments while preserving source line numbers and literal \%.
    visible = "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in text.splitlines())
    todos = [(visible.count("\n", 0, match.start()) + 1, match.group(1))
             for match in re.finditer(r"\\todo\{([^}]+)\}", visible)]
    assets = ["report/main.tex", "report/main.pdf", "report/transformer-connected.png"]
    assets += ["results/benchmark_20261109T113037Z/plots/" + name for name in
               ["speedup_1thread.png", "vectorization.png", "stage_breakdown.png",
                "scratch_vs_n.png", "thread_scaling.png"]]
    problems = []
    for asset in assets:
        if not (root / asset).is_file():
            problems.append("Missing: " + asset)
        else:
            result = subprocess.run(["git", "ls-files", "--error-unmatch", "--", asset],
                                    cwd=root, capture_output=True)
            if result.returncode:
                problems.append("Not tracked in Git: " + asset)
    return todos, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="Fail if TODOs or missing/untracked assets remain")
    parser.add_argument("--allow-video-pending", action="store_true",
                        help="With --final, permit only S3 while the video is being prepared")
    args = parser.parse_args()
    todos, problems = check()
    for line, label in todos:
        print(f"report/main.tex:{line}: TODO {label}")
    for item in problems:
        print(item)
    print("\nHuman checks (the script cannot complete these):")
    for label, action in MANUAL.items():
        print(f"{label}: {action}")
    print(f"\n{len(todos)} visible TODO markers; {len(problems)} asset issues.")
    blocking_todos = [(line, label) for line, label in todos
                      if not (args.allow_video_pending and label == "S3")]
    if args.allow_video_pending:
        print("Video exception: S3 may remain; video submission/access is not certified.")
    if args.final and (blocking_todos or problems):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
