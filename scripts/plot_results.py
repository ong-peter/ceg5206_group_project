"""Plot the benchmark CSVs for the report.

Inputs (each figure is skipped when its CSV is missing):
  --ladder   scripts/bench.py --full --novec-binary ...       (make OPENMP=1 bench -> ladder.csv)
  --scaling  scripts/bench.py --threads 1,2,4,... --shapes .. (make OPENMP=1 bench-scaling -> scaling.csv)
  --tuning   test/check.py --tune                     (tuning-results.csv, with tuned-tiles.json)
  --stages   scripts/profile_stages.py                (results/stages.csv)
  --packing  scripts/packing_experiment.py            (results/packing_experiment/packing_experiment.csv)

Figures written to --out:
  runtime_vs_n.png         median time per call against N, one panel per d, 1 thread, IQR bars
  scratch_vs_n.png         allocated kernel scratch against N, with Q/K/V/O bytes for reference
  peak_rss_vs_n.png        process peak RSS against N (ladder.csv from scripts/bench.py)
  stage_breakdown.png      per-stage share of one call: QK^T, softmax, PV; streaming's fused split
  packing_experiment.png   blocked variants (PV order, K packing) over the original, per shape and thread count
  speedup_1thread.png      blocked and streaming speedup over the baseline, every shape, 1 thread
  vectorization.png        no-vec ablation: SIMD gain per family, tiling gain without SIMD
  thread_scaling.png       speedup over the same kernel at 1 thread (IQR bars), one panel per shape
  tile_tuning.png          tuning grid, time relative to 32x32, selected tiles outlined
Scratch is allocated capacity reported by the kernels, not measured DRAM traffic or RSS.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
# Fixed categorical order: a family keeps its color in every figure.
FAMILIES = {"baseline": ("Materialized baseline", "#2a78d6"),
            "blocked": ("Blocked materialized", "#eb6834"),
            "streaming": ("Streaming", "#1baf7a")}
DIVERGING = LinearSegmentedColormap.from_list("faster_slower", ["#2a78d6", "#f0efec", "#e34948"])
SINGLE_STAGES = {"1": "baseline", "1+2": "blocked", "1+2+3": "streaming"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": INK_3, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def read(path):
    if not path or not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key in ("stage", "case", "algorithm", "build"):  # labels such as stage "1" stay text
                continue
            try:
                row[key] = float(value) if "." in value or "e" in value else int(value)
            except (TypeError, ValueError):
                pass
    return rows


def shape(n, d):
    return f"{n}×{d}"


def bytes_label(value, _=None):
    for unit, size in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if value >= size:
            return f"{value / size:g} {unit}"
    return f"{value:g} B"


def log2_axis(ax, values, label):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(sorted(set(values))))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel(label)


def label_ends(ax, items, gap=11):
    """Direct-label line ends in text ink, nudged apart vertically to avoid collisions."""
    to_display, to_data = ax.transData.transform, ax.transData.inverted().transform
    points = sorted(((to_display((x, y)), text) for x, y, text in items), key=lambda p: p[0][1])
    placed = []
    for (px, py), text in points:
        py = max(py, placed[-1][1] + gap) if placed else py
        placed.append((px, py, text))
    for px, py, text in placed:
        x, y = to_data((px, py))
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK_2, fontsize=8, annotation_clip=False)


def legend(fig, families, extra=()):
    handles = [plt.Line2D([], [], color=FAMILIES[f][1], marker="o", label=FAMILIES[f][0]) for f in families]
    handles += list(extra)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.06))


def single_thread(ladder):
    return [r for r in ladder if r["build"] == "vec" and r["threads"] == 1 and r["stage"] in SINGLE_STAGES]


def present(rows):
    return [f for f in FAMILIES if any(r["algorithm"] == f for r in rows)]


def runtime_vs_n(ladder, out):
    rows = single_thread(ladder)
    ds = sorted({r["d"] for r in rows})
    fig, axes = plt.subplots(1, len(ds), figsize=(3.3 * len(ds), 3.3), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], ds):
        ends = []
        for family in present(rows):
            series = sorted((r for r in rows if r["d"] == d and r["algorithm"] == family), key=lambda r: r["n"])
            ns = [r["n"] for r in series]
            medians = [r["median_ms"] for r in series]
            spread = [[m - r["q1_ms"] for m, r in zip(medians, series)], [r["q3_ms"] - m for m, r in zip(medians, series)]]
            ax.errorbar(ns, medians, yerr=spread, color=FAMILIES[family][1], marker="o", capsize=0, elinewidth=1)
            ends.append((ns[-1], medians[-1], family))
        ax.set_yscale("log")
        log2_axis(ax, [r["n"] for r in rows if r["d"] == d], "Sequence length N")
        ax.set_title(f"d = {d}", color=INK)
        if ax is axes[0][-1]:
            label_ends(ax, ends)
    axes[0][0].set_ylabel("Median time per call (ms, log)")
    legend(fig, present(rows))
    fig.savefig(out)
    plt.close(fig)


def memory_vs_n(ladder, out, column="scratch_bytes", ylabel="Allocated scratch (log)"):
    rows = [r for r in single_thread(ladder) if isinstance(r.get(column), int)]
    threaded = [r for r in ladder if r["stage"] == "1+2+3+4" and r["build"] == "vec"]
    workers = max((r["threads"] for r in threaded), default=None)
    ds = sorted({r["d"] for r in rows})
    fig, axes = plt.subplots(1, len(ds), figsize=(3.3 * len(ds), 3.3), sharey=True, squeeze=False)
    baseline = {(r["n"], r["d"]): r[column] for r in rows if r["algorithm"] == "baseline"}
    materialized_same = all(r[column] == baseline.get((r["n"], r["d"]))
                            for r in rows if r["algorithm"] == "blocked")
    for ax, d in zip(axes[0], ds):
        ends = []

        def plot(series, color, text, style="-"):
            series = sorted(series, key=lambda r: r["n"])
            if series:
                xs, ys = [r["n"] for r in series], [r[column] for r in series]
                ax.plot(xs, ys, style, color=color, marker="o")
                ends.append((xs[-1], ys[-1], text))
        families = ["baseline", "streaming"] if materialized_same else present(rows)
        for family in families:
            text = "baseline = blocked" if family == "baseline" and materialized_same else family
            plot([r for r in rows if r["d"] == d and r["algorithm"] == family], FAMILIES[family][1], text)
        if workers:
            plot([r for r in threaded if r["d"] == d and r["threads"] == workers], FAMILIES["streaming"][1],
                 f"streaming, {workers} workers", "--")
        ns = sorted({r["n"] for r in rows if r["d"] == d})
        ax.plot(ns, [16 * n * d for n in ns], ":", color=INK_3, marker="o")
        ends.append((ns[-1], 16 * ns[-1] * d, "Q/K/V/O"))
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(FuncFormatter(bytes_label))
        log2_axis(ax, [r["n"] for r in rows if r["d"] == d], "Sequence length N")
        ax.set_title(f"d = {d}", color=INK)
        if ax is axes[0][-1]:
            label_ends(ax, ends)
    axes[0][0].set_ylabel(ylabel)
    extra = [plt.Line2D([], [], color=INK_3, linestyle=":", marker="o", label="Q/K/V/O (16Nd bytes)")]
    if workers:
        extra.insert(0, plt.Line2D([], [], color=FAMILIES["streaming"][1], linestyle="--", marker="o",
                                   label=f"Streaming, {workers} workers"))
    names = ["baseline", "streaming"] if materialized_same else present(rows)
    handles = [plt.Line2D([], [], color=FAMILIES[f][1], marker="o",
                          label="Baseline and blocked" + (" (4N²)" if column == "scratch_bytes" else "") if f == "baseline" and materialized_same
                          else FAMILIES[f][0] + (", 1 thread" if f == "streaming" else "")) for f in names]
    fig.legend(handles=handles + extra, loc="upper center", ncol=len(handles) + len(extra),
               bbox_to_anchor=(0.5, 1.06))
    fig.savefig(out)
    plt.close(fig)


def grouped_bars(ax, groups, series, values, labels, colors=None):
    """Bars grouped by `groups`, one bar per series with a small surface gap."""
    width = 0.8 / len(series)
    for i, name in enumerate(series):
        xs = [g + (i - (len(series) - 1) / 2) * width for g in range(len(groups))]
        ax.bar(xs, [values.get((group, name), math.nan) for group in groups], width * 0.92,
               color=(colors or {})[name] if colors else FAMILIES[name][1], label=labels[name])
    ax.set_xticks(range(len(groups)))
    ax.axhline(1, color=INK_3, linewidth=1, linestyle="--", zorder=0)


def speedup_1thread(ladder, out):
    rows = single_thread(ladder)
    shapes = sorted({(r["n"], r["d"]) for r in rows})
    families = [f for f in present(rows) if f != "baseline"]
    values = {((r["n"], r["d"]), r["algorithm"]): r["speedup_vs_baseline"] for r in rows}
    fig, ax = plt.subplots(figsize=(max(6, 0.75 * len(shapes)), 3.3))
    grouped_bars(ax, shapes, families, values, {f: FAMILIES[f][0] for f in families})
    ax.set_xticklabels([shape(*s) for s in shapes], rotation=45, ha="right")
    ax.set_xlabel("Shape N×d")
    ax.set_ylabel("Speedup over baseline (×)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILIES[f][1], label=FAMILIES[f][0]) for f in families]
    handles.append(plt.Line2D([], [], color=INK_3, linewidth=1, linestyle="--", label="Baseline (1×)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.04))
    fig.savefig(out)
    plt.close(fig)


def vectorization(ladder, out, d):
    vec = {(r["n"], r["d"], r["algorithm"]): r["median_ms"] for r in single_thread(ladder)}
    novec = {(r["n"], r["d"], r["algorithm"]): r["median_ms"] for r in ladder if r["stage"] == "ablation"}
    if not novec:
        return False
    shapes = sorted({(n, dd) for n, dd, _ in novec if dd == d}) or sorted({(n, dd) for n, dd, _ in novec})
    families = [f for f in FAMILIES if any(key[2] == f for key in novec)]
    simd = {(s, f): novec[(*s, f)] / vec[(*s, f)] for s in shapes for f in families if (*s, f) in vec}
    tiling = {(s, f): novec[(*s, "baseline")] / novec[(*s, f)] for s in shapes for f in families
              if f != "baseline" and (*s, f) in novec}
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.3))
    grouped_bars(left, shapes, families, simd, {f: FAMILIES[f][0] for f in families})
    left.set_title("SIMD gain: no-vec time ÷ vec time", color=INK)
    left.set_ylabel("Gain (×)")
    optimized = [f for f in families if f != "baseline"]
    grouped_bars(right, shapes, optimized, tiling, {f: FAMILIES[f][0] for f in optimized})
    right.set_title("Gain over baseline, both built without SIMD", color=INK)
    for ax in (left, right):
        ax.set_xticklabels([shape(*s) for s in shapes])
        ax.set_xlabel("Shape N×d, 1 thread")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILIES[f][1], label=FAMILIES[f][0]) for f in families]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.06))
    fig.savefig(out)
    plt.close(fig)
    return True


# Blocked-kernel variants (categorical slots 7, 6, 5), apart from the family colors.
VARIANT_COLORS = {"pack": "#6250d6", "reorder": "#008300", "reorder+pack": "#e87ba4"}
VARIANT_LABELS = {"pack": "Packed K", "reorder": "Reordered PV (default)", "reorder+pack": "Reordered PV + packed K"}


def packing_experiment(rows, out):
    threads = sorted({r["threads"] for r in rows})
    shapes = sorted({(r["n"], r["d"]) for r in rows})
    variants = [v for v in VARIANT_COLORS if any(r["variant"] == v for r in rows)]
    fig, axes = plt.subplots(len(threads), 1, figsize=(max(6, 0.75 * len(shapes)), 2.9 * len(threads)),
                             squeeze=False, sharey=True, layout="constrained")
    for ax, t in zip(axes[:, 0], threads):
        values = {((r["n"], r["d"]), r["variant"]): r["speedup_vs_original"] for r in rows if r["threads"] == t}
        grouped_bars(ax, shapes, variants, values, VARIANT_LABELS, VARIANT_COLORS)
        ax.set_xticklabels([shape(*s) for s in shapes], rotation=45 if len(shapes) > 6 else 0,
                           ha="right" if len(shapes) > 6 else "center")
        used = {r["variant"]: f"{r['br']}×{r['bc']}" for r in rows if r["threads"] == t}
        tiles = (f"all at {next(iter(used.values()))}" if len(set(used.values())) == 1 else
                 "tiles " + ", ".join(f"{name} {tile}" for name, tile in used.items()))
        ax.set_title(f"{t} thread{'s' if t > 1 else ''}; {tiles}", color=INK, loc="left")
        ax.set_ylabel("Speedup over original (×)")
    axes[-1][0].set_xlabel("Shape N×d")
    handles = [plt.Rectangle((0, 0), 1, 1, color=VARIANT_COLORS[v], label=VARIANT_LABELS[v]) for v in variants]
    handles.append(plt.Line2D([], [], color=INK_3, linewidth=1, linestyle="--", label="Original blocked (1×)"))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    fig.savefig(out)
    plt.close(fig)


def thread_scaling(scaling, out):
    rows = [r for r in scaling if r["build"] == "vec" and r["algorithm"] in ("blocked", "streaming")
            and r["stage"] != "ablation"]
    shapes = sorted({(r["n"], r["d"]) for r in rows})
    threads = sorted({r["threads"] for r in rows})
    # Extra room between panels for the direct labels at each line's end.
    fig, axes = plt.subplots(1, len(shapes), figsize=(4.2 * len(shapes), 3.3), sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.55})
    for ax, (n, d) in zip(axes[0], shapes):
        ax.plot(threads, threads, ":", color=INK_3, linewidth=1)
        ends = [(threads[-1], threads[-1], "ideal")]
        for family in ("blocked", "streaming"):
            series = sorted((r for r in rows if (r["n"], r["d"]) == (n, d) and r["algorithm"] == family),
                            key=lambda r: r["threads"])
            if not series:
                continue
            xs = [r["threads"] for r in series]
            ys = [1.0 if r["threads"] == 1 else r["speedup_vs_1_thread"] for r in series]
            # IQR of each thread count's samples, as speedup over the 1-thread median:
            # [median_1 / q3_T, median_1 / q1_T].
            one = next((r["median_ms"] for r in series if r["threads"] == 1), None)
            if one is not None:
                low = [y - one / r["q3_ms"] for r, y in zip(series, ys)]
                high = [one / r["q1_ms"] - y for r, y in zip(series, ys)]
                ax.errorbar(xs, ys, yerr=[low, high], fmt="none", ecolor=FAMILIES[family][1],
                            elinewidth=1.2, capsize=3, zorder=3)
            ax.plot(xs, ys, color=FAMILIES[family][1], marker="o")
            ends.append((xs[-1], ys[-1], f"{family} {ys[-1]:.1f}×"))
        log2_axis(ax, threads, "Threads")
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_locator(FixedLocator(threads))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.set_title(shape(n, d), color=INK)
        label_ends(ax, ends)
    axes[0][0].set_ylabel("Speedup over 1 thread (×, log)")
    legend(fig, ["blocked", "streaming"],
           [plt.Line2D([], [], color=INK_3, linestyle=":", linewidth=1, label="Ideal (linear)")])
    fig.savefig(out)
    plt.close(fig)


def tile_tuning(tuning, tiles, out):
    rows = [r for r in tuning if r["case"] == "tuning" and isinstance(r.get("median_ms"), float)]
    families = [f for f in ("blocked", "streaming") if any(r["algorithm"] == f for r in rows)]
    shapes = sorted({(r["n"], r["d"]) for r in rows})
    sizes = sorted({r["br"] for r in rows})
    ratio = {}
    for f in families:
        for s in shapes:
            cells = {(r["br"], r["bc"]): r["median_ms"] for r in rows if r["algorithm"] == f and (r["n"], r["d"]) == s}
            base = cells.get((32, 32)) or min(cells.values())
            ratio[(f, s)] = {pair: value / base for pair, value in cells.items()}
    spread = [v for grid in ratio.values() for v in grid.values()]
    norm = TwoSlopeNorm(vcenter=1.0, vmin=min(min(spread), 0.99), vmax=max(max(spread), 1.01))
    fig, axes = plt.subplots(len(families), len(shapes), figsize=(3.0 * len(shapes) + 1.4, 2.9 * len(families)),
                             squeeze=False, layout="constrained")
    for i, f in enumerate(families):
        for j, s in enumerate(shapes):
            ax = axes[i][j]
            grid = [[ratio[(f, s)].get((br, bc), math.nan) for bc in sizes] for br in sizes]
            image = ax.imshow(grid, cmap=DIVERGING, norm=norm, origin="lower")
            for a, br in enumerate(sizes):
                for b, bc in enumerate(sizes):
                    value = grid[a][b]
                    ax.text(b, a, f"{value:.2f}", ha="center", va="center", fontsize=8, color=INK)
            selected = tiles.get(f)
            if selected and selected["br"] in sizes and selected["bc"] in sizes:
                ax.add_patch(plt.Rectangle((sizes.index(selected["bc"]) - 0.5, sizes.index(selected["br"]) - 0.5),
                                           1, 1, fill=False, edgecolor=INK, linewidth=2))
            ax.set_xticks(range(len(sizes)), [str(x) for x in sizes])
            ax.set_yticks(range(len(sizes)), [str(x) for x in sizes])
            ax.grid(False)
            if i == len(families) - 1:
                ax.set_xlabel("Bc (keys per tile)")
            if j == 0:
                ax.set_ylabel(f"{FAMILIES[f][0]}\nBr (queries per tile)")
            ax.set_title(shape(*s), color=INK)
    bar = fig.colorbar(image, ax=axes, shrink=0.8)
    bar.set_label("Time relative to 32×32\n(blue faster, red slower)")
    if tiles:
        fig.suptitle("Outlined: selected pair per family (geometric mean over both shapes)",
                     color=INK_2, fontsize=9)
    fig.savefig(out)
    plt.close(fig)


# Stages get their own hues (categorical slots 7, 5, 6, 4), apart from the family colors.
STAGE_COLORS = {"QK^T scores": ("#6250d6", SURFACE), "softmax": ("#e87ba4", INK), "PV": ("#008300", SURFACE),
                "QK^T scores + K packing": ("#6250d6", SURFACE), "online update + PV": ("#eda100", INK)}


def stage_breakdown(stages, out):
    shapes = sorted({(r["n"], r["d"]) for r in stages})
    families = [f for f in FAMILIES if any(r["algorithm"] == f for r in stages)]
    fig, axes = plt.subplots(len(shapes), 1, figsize=(7.5, 1.0 + 0.55 * len(families) * len(shapes)),
                             squeeze=False, layout="constrained")
    for ax, (n, d) in zip(axes[:, 0], shapes):
        for y, family in enumerate(reversed(families)):
            left = 0.0
            rows = [r for r in stages if (r["n"], r["d"]) == (n, d) and r["algorithm"] == family]
            for r in rows:
                fill, ink = STAGE_COLORS.get(r["stage"], (INK_3, SURFACE))
                ax.barh(y, r["stage_ms"], left=left, height=0.62, color=fill, edgecolor=SURFACE, linewidth=1.5)
                if r["share"] >= 0.08:
                    ax.text(left + r["stage_ms"] / 2, y, f"{r['share']:.0%}", ha="center", va="center",
                            fontsize=8, color=ink)
                left += r["stage_ms"]
            if rows:
                tiles = "" if family == "baseline" else f", {rows[0]['br']}×{rows[0]['bc']}"
                ax.text(left, y, f"  {rows[0]['instrumented_mean_ms']:.3g} ms{tiles}", va="center",
                        fontsize=8, color=INK_2)
        ax.set_yticks(range(len(families)), [FAMILIES[f][0] for f in reversed(families)])
        ax.set_title(f"{shape(n, d)}, 1 thread", color=INK, loc="left")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.18)
    axes[-1][0].set_xlabel("Instrumented time per call (ms); labels are each stage's share")
    names = list(dict.fromkeys(r["stage"] for r in stages))
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLORS.get(s, (INK_3,))[0], label=s) for s in names]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ladder", type=Path, default=Path("ladder.csv"))
    parser.add_argument("--scaling", type=Path, default=Path("scaling.csv"))
    parser.add_argument("--tuning", type=Path, default=Path("tuning-results.csv"))
    parser.add_argument("--tiles", type=Path, default=Path("tuned-tiles.json"), help="Selected tiles to outline")
    parser.add_argument("--stages", type=Path, default=Path("results/stages.csv"))
    parser.add_argument("--packing", type=Path, default=Path("results/packing_experiment/packing_experiment.csv"))
    parser.add_argument("--ablation-d", type=int, default=64, help="Feature dimension of the ablation shapes")
    parser.add_argument("--out", type=Path, default=Path("plots"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    written = []

    def save(name, function, *inputs):
        path = args.out / name
        if function(*inputs, path) is not False:
            written.append(path)
    ladder = read(args.ladder)
    if ladder:
        save("runtime_vs_n.png", runtime_vs_n, ladder)
        save("scratch_vs_n.png", memory_vs_n, ladder)
        if any(isinstance(r.get("peak_rss_bytes"), int) for r in ladder):
            save("peak_rss_vs_n.png", lambda rows, path: memory_vs_n(
                rows, path, "peak_rss_bytes", "Process peak RSS (log)"), ladder)
        save("speedup_1thread.png", speedup_1thread, ladder)
        save("vectorization.png", lambda rows, path: vectorization(rows, path, args.ablation_d), ladder)
    else:
        print(f"Skipping ladder figures: {args.ladder} not found (make OPENMP=1 bench)")
    scaling = read(args.scaling)
    if scaling:
        save("thread_scaling.png", thread_scaling, scaling)
    else:
        print(f"Skipping thread scaling: {args.scaling} not found (make OPENMP=1 bench-scaling)")
    tuning = read(args.tuning)
    if tuning:
        tiles = json.loads(args.tiles.read_text())["selected"] if args.tiles.exists() else {}
        save("tile_tuning.png", tile_tuning, tuning, tiles)
    else:
        print(f"Skipping tile tuning: {args.tuning} not found (test/check.py --algorithm all --tune)")
    stages = read(args.stages)
    if stages:
        save("stage_breakdown.png", stage_breakdown, stages)
    else:
        print(f"Skipping stage breakdown: {args.stages} not found (scripts/profile_stages.py)")
    packing = read(args.packing)
    if packing:
        save("packing_experiment.png", packing_experiment, packing)
    else:
        print(f"Skipping packing experiment: {args.packing} not found (make OPENMP=1 packing-experiment)")
    for path in written:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
