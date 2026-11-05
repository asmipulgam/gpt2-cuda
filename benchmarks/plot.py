"""Charts from the benchmark CSVs.

    python benchmarks/plot.py   # reads results/kernels.csv and results/e2e.csv, writes PNGs next to them
"""
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils import RESULTS_DIR, read_csv  # noqa: E402

# Categorical slots in fixed order; an implementation keeps its color across charts.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=12)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def grouped_bars(ax, groups, impls, values, colors):
    n = len(impls)
    width = 0.8 / n
    for i, impl in enumerate(impls):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        ys = [values.get((grp, impl), 0) for grp in groups]
        ax.bar(xs, ys, width * 0.92, label=impl, color=colors[impl], edgecolor="white", linewidth=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)


def plot_kernels():
    path = RESULTS_DIR / "kernels.csv"
    if not path.exists():
        print(f"skip: {path} not found (run bench_kernels.py)")
        return
    by_kernel = defaultdict(list)
    for r in read_csv(path):
        by_kernel[r["kernel"]].append(r)
    for kernel, rows in by_kernel.items():
        shapes = list(dict.fromkeys(r["shape"] for r in rows))
        impls = list(dict.fromkeys(r["impl"] for r in rows))
        colors = {impl: SERIES[i % len(SERIES)] for i, impl in enumerate(impls)}
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
        grouped_bars(a1, shapes, impls, {(r["shape"], r["impl"]): float(r["ms"]) for r in rows}, colors)
        style(a1, f"{kernel}: time per call (lower is better)", "ms")
        grouped_bars(a2, shapes, impls, {(r["shape"], r["impl"]): float(r["gbps"]) for r in rows}, colors)
        style(a2, f"{kernel}: effective bandwidth (higher is better)", "GB/s")
        fig.tight_layout()
        out = RESULTS_DIR / f"kernel_{kernel}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def plot_e2e():
    path = RESULTS_DIR / "e2e.csv"
    if not path.exists():
        print(f"skip: {path} not found (run bench_e2e.py)")
        return
    rows = read_csv(path)
    configs = list(dict.fromkeys(f"B={r['batch']} T={r['prompt_len']}" for r in rows))
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    colors = {v: SERIES[i % len(SERIES)] for i, v in enumerate(variants)}
    values = {(f"B={r['batch']} T={r['prompt_len']}", r["variant"]): float(r["tokens_per_s"]) for r in rows}
    fig, ax = plt.subplots(figsize=(13, 4.8))
    grouped_bars(ax, configs, variants, values, colors)
    style(ax, "GPT-2 generation throughput (higher is better)", "tokens / s")
    fig.tight_layout()
    out = RESULTS_DIR / "e2e_tokens_per_s.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_kernels()
    plot_e2e()
