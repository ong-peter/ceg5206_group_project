# Submission package checks - 13 November 2026

## Version and outstanding item

The assessed source/report package is tagged `ceg5206-report-2026-11-13`.
It includes the report source/PDF, all figure assets, kernel sources, tests,
reproduction scripts and the previously committed benchmark/follow-up data.
The only visible report TODO is S3: the video link and instructor/TA access.
This tag freezes the non-video package; adding the video later requires a new
commit/tag and an updated version identifier in the report.

## Access

An unauthenticated request to GitHub's repository API returned HTTP 200 with
`private: false` on 13 November 2026. The tag and report PDF are also checked
without credentials after publication. Video access is not yet checked.

## Contributions

The existing member responsibilities are retained. Chen Xinzhe's row is
shortened to "Baseline, FP64 reference and correctness tests". No new claims
of another member's approval, cross-review or code authorship are added.
The report distinguishes established cited algorithms from the group's CPU
implementation, validation, experiments and analysis.

## Clean-source portability checks

A new worktree at `d1f56ae7841531235f08f71cae4eb21ef2845e7e` was used.
Production sources, tests, Makefile and pinned requirements are unchanged.
Validation used Apple Clang 21.0.0, Python 3.12, NumPy 2.3.5,
Matplotlib 3.11.2 and conda-forge llvm-openmp 23.1.5 on Apple Silicon.
The OpenMP package was extracted into an isolated dependency folder.
Use `brew install libomp` or an existing installation on other Macs; set
`LIBOMP` to its prefix and `PYTHON` to the environment with requirements.txt.
Choose a LIBOMP path without spaces because the current Makefile does not
quote its include/linker paths.

Commands run from the worktree (paths are relative to this local validation
layout; replace them with your installed dependencies):

```sh
make OPENMP=1 LIBOMP=../tmp/openmp PYTHON=../tmp/repro-venv/bin/python test test-streaming-api test-threads
../tmp/repro-venv/bin/python test/check.py --algorithm all --full --tiles --csv results/submission_20261113/correctness.csv
make PYTHON=../tmp/repro-venv/bin/python test-streaming-api
```

Results:

- FP64-reference full shape/edge/tile suite: 261 CSV rows, all passed.
- Streaming C API: 201 serial and 601 OpenMP configurations passed.
- Thread-invariance suites: 2, 3, 10 and 64 threads passed (the Make target
  repeats 10 because physical/logical core detection returns the same value).
- Nine figures regenerated successfully from the original recorded CSVs.
- Report rebuilt successfully using Tectonic 0.17.0 and Overleaf; 10 pages, only S3 visible. The packaged PDF is the compiled Overleaf download.
- The submission checker accepts `--final --allow-video-pending` and rejects
  ordinary `--final` while S3 remains.

Logs: [validation directory](submission_validation_2026-11-13/).
Full reference results: `results/submission_20261113/correctness.csv`.

Plot regeneration:

```sh
python scripts/plot_results.py \
  --ladder results/benchmark_20261109T113037Z/ladder.csv \
  --scaling results/benchmark_20261109T113037Z/scaling.csv \
  --tuning results/benchmark_20261109T113037Z/tuning-results.csv \
  --tiles results/benchmark_20261109T113037Z/tuned-tiles.json \
  --stages results/benchmark_20261109T113037Z/stages.csv \
  --packing results/benchmark_20261109T113037Z/packing_experiment/packing_experiment.csv \
  --out results/submission_20261113/plots
```

These checks verify portability, correctness and reuse of saved experimental
inputs. They do not repeat the Intel i7-11700K performance/counter experiment,
TSan or the sanitizer runs. Those reported results retain their original
provenance and machine-specific reproduction instructions in
[benchmarking.md](benchmarking.md) and [report_todos.md](report_todos.md).

To inspect this package from a clean checkout:

```sh
git clone --branch ceg5206-report-2026-11-13 https://github.com/ong-peter/ceg5206_group_project.git
cd ceg5206_group_project
python -m pip install -r requirements.txt
python scripts/check_report_submission.py --final --allow-video-pending
```

For a fresh correctness run, follow the platform-specific build setup in the
README and then run the test commands above with your own PYTHON/LIBOMP paths.
For performance reproduction, use the documented controlled Intel/Linux setup.
