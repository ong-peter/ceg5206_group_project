# Running the final benchmark

One command runs every experiment in the proposal, in order, into one folder:

```sh
make benchmark
```

It takes roughly 3-4 hours on an 8-core machine with the default 3 rounds. The
numbers are only as good as the machine state, so prepare the machine first.

## 1. One-time setup

- Python with the pinned packages (NumPy and matplotlib):

  ```sh
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  ```

  The scripts use `.venv/bin/python` when it exists; otherwise set `PYTHON`.
- A C++17 compiler with OpenMP (GCC on Linux; on macOS set `LIBOMP`, see the README).
- Linux `perf` for hardware counters (Ubuntu: `sudo apt install linux-tools-$(uname -r)`).
  Without it, the counter step is skipped.
- Optional: Clang and Archer for `make tsan` (`sudo apt install clang-21 libomp-21-dev`).

## 2. Prepare the machine (before every session)

These settings reset at reboot.

**Close other programs.** Browsers, editors, chat and sync clients all take CPU
time from the threaded runs, which use every physical core. Disable automatic
suspend (Settings, Power) so the machine does not sleep during the run.

**Fix the CPU frequency.** By default Ubuntu runs the `powersave` governor in the
`balanced` power profile with turbo on. With turbo, one busy core runs faster
(about 4.9 GHz on the i7-11700K) than eight busy cores (about 4.4-4.6 GHz), and
heat lowers clocks over a long run. That makes one-thread runs look relatively
fast and thread scaling look worse than it is. Fix it:

```sh
powerprofilesctl set performance
sudo cpupower frequency-set -g performance
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

Set the power profile first; its daemon otherwise resets the energy preference.
On AMD CPUs, turn boost off with
`echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost` instead of `no_turbo`.
If you keep turbo on, say so in the report; the run records the setting either way.

Check it while something is running (`Bzy_MHz` should stay near the base clock,
3600 MHz on the i7-11700K):

```sh
sudo turbostat --quiet --show Busy%,Bzy_MHz --interval 5
```

**Allow hardware counters.** Per-process counters (cycles, IPC, caches, FP
instructions) need `perf_event_paranoid` at 1 or lower. DRAM traffic is counted
system-wide by the memory controller and needs 0:

```sh
sudo sysctl kernel.perf_event_paranoid=0
```

## 3. Smoke test (a few minutes)

```sh
make benchmark QUICK=1
```

This runs the whole pipeline on small shapes with one round, to catch setup
problems before the long run. Its numbers are not results; it writes to
`results/benchmark_quick_<time>/`.

## 4. Full run

```sh
STRICT=1 make benchmark
```

`STRICT=1` stops before starting unless the governor is `performance` and turbo
is off. Run it where the terminal will not close, for example:

```sh
nohup make benchmark > benchmark.out 2>&1 &
```

Options (environment or `make` variables):

| Variable | Default | Meaning |
| --- | --- | --- |
| `ROUNDS` | 3 | Interleaved rounds per timing step; each round rotates the variant order |
| `THREADS` | physical cores | Threads for the threaded ladder stages; scaling uses 1, 2, 4, ... up to it |
| `OUT` | `results/benchmark_<UTC time>` | Output folder |
| `STRICT` | 0 | 1: refuse to start without a fixed frequency policy |
| `PERF` | 1 | 0: skip hardware counters |
| `PACKING` | 1 | 0: skip the blocked-kernel packing experiment |
| `QUICK` | 0 | 1: smoke test (small shapes, one round) |
| `PYTHON` | `.venv/bin/python` or `python3` | Interpreter with the requirements |
| `LIBOMP` | | macOS OpenMP installation, as in the README |

## 5. What the run does

| Step | Command it runs | Output in the folder |
| --- | --- | --- |
| Preflight: frequency policy, load, git commit, compiler, CPU | | `environment.txt`, `uncommitted.patch`, `run.log` |
| Clean OpenMP build, vectorized and no-vec | `make OPENMP=1 attention attention-novec` | `build-info.txt` |
| Correctness gate: 12 shapes, edge cases, tile sweep; identical output at 2-64 threads | `test/check.py --full --tiles`, `make test-threads` | `correctness.csv` |
| Tile tuning: 9 pairs on 512x64 and 2048x64, one thread | `test/check.py --tune` | `tuning-results.csv`, `tuned-tiles.json` |
| Ladder: 12 shapes, three families at 1 thread, threaded stages, no-vec ablation | `scripts/bench.py --full` | `ladder.csv`, `ladder.machine.json` |
| Thread scaling: 1, 2, 4, ... threads on 512x64 and 4096x64 | `scripts/bench.py --threads` | `scaling.csv`, `scaling.machine.json` |
| Allocated memory per algorithm and shape | `scripts/memory_report.py` | `memory.csv` |
| Stage profile: QKᵀ, softmax, PV per call, 512x64 and 4096x64 | `scripts/profile_stages.py` | `stages.csv` |
| Hardware counters per call, 512x64 and 4096x64, 1 and all threads | `scripts/perf_profile.py` | `perf_counters.csv` |
| Blocked-kernel design: PV order and K packing | `scripts/packing_experiment.py` | `packing_experiment/`, `packing_experiment_fixed32/` |
| Figures | `scripts/plot_results.py` | `plots/*.png` |
| Postflight: frequency policy again | | `frequency_after.txt` |

All timing steps pin threads to physical cores (`OMP_PLACES=cores`,
`OMP_PROC_BIND=close`), check every variant against the FP64 reference before
timing it, use 5 warm-ups and 15 timed samples per round, and time warm,
repeated calls on preallocated buffers. `ladder.csv` also records each
variant's process peak RSS. Sanitizer builds are never timed.

## 6. After the run

1. Read the warnings in `run.log`: frequency policy, load, and whether the policy
   changed during the run (`frequency_after.txt`).
2. Look at the figures in `plots/`.
3. Restore the machine:

   ```sh
   echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
   sudo cpupower frequency-set -g powersave
   powerprofilesctl set balanced
   sudo sysctl kernel.perf_event_paranoid=4
   ```

   (4 is Ubuntu's default; check yours with `cat /proc/sys/kernel/perf_event_paranoid` beforehand.)
4. Keep the data. `results/` and `*.csv` are git-ignored, but the submission
   needs the raw CSVs, the tuned tiles and the figures. Copy the folder somewhere
   tracked, for example `data/final_<date>/`, and add it with `git add -f`.

## 7. Rerunning one piece

| Target | What it does |
| --- | --- |
| `make OPENMP=1 bench ROUNDS=3` | Ladder into `ladder.csv` (uses `tuned-tiles.json` when present) |
| `make OPENMP=1 bench-scaling ROUNDS=3` | Thread scaling into `scaling.csv` |
| `make profile-stages` | `results/stages.csv` |
| `make OPENMP=1 perf-profile` | `results/perf_counters.csv` |
| `make OPENMP=1 packing-experiment` | `results/packing_experiment/` |
| `make plots` | `plots/` from the files above |
| `make tsan` | ThreadSanitizer check (not a timing run) |

Run `python test/check.py --algorithm all --tune --tiles --csv tuning-results.csv`
first so these targets use tuned tiles.

## 8. Reading the results

- Tiles are tuned at one thread on N >= 512. At N = 128 with eight threads, Br = 64
  leaves two query blocks, so at most two threads have work. The proposal's
  scaling shapes (512 and 4096) are not affected.
- `llc_misses` counts demand loads that miss the last-level cache; hardware
  prefetches are not included, so streaming access patterns show few misses.
  Use `dram_read_bytes` and `dram_write_bytes` for memory traffic. Those are
  system-wide and include any other activity on the machine.
- `ghz_per_thread` in `perf_counters.csv` is cycles divided by median time and
  threads. With turbo off it should sit near the base clock; if it varies
  between one and eight threads, the frequency policy was not fixed.
- `peak_rss_bytes` includes buffers, scratch and runtime memory (about 4.5 MiB
  minimum), so it supplements `scratch_bytes` rather than replacing it.
