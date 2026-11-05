"""Plot adc_bench tables output: ns per sample against total table footprint.

    bench/build/adc_bench tables              # writes bench/out/tables.csv
    python bench/plot_tables.py               # writes bench/out/tables*.png

One panel per access pattern. Lines are one table size N each, joining the
instance counts K (footprint = K * 3 * (N + 1) * 4 bytes); solid with
nothing running between rounds, dashed with 16 MB of other work. The flat
lines are the polynomial chain, vectorized and scalar -- they have no
footprint. Cache sizes come from sysctl when available.
"""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "examples"))
from common import plots  # noqa: E402

BETWEEN_STYLE = {0: "-", 16: "--"}


def cache_sizes():
    """(label, bytes) for the performance cores' L1D and L2, if sysctl has them."""
    out = []
    for key, label in (("hw.perflevel0.l1dcachesize", "L1D"), ("hw.perflevel0.l2cachesize", "L2")):
        try:
            v = int(subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout)
            out.append((f"{label} {v >> 10} KB" if v < 1 << 20 else f"{label} {v >> 20} MB", v))
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return out


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["k"], r["n"], r["between_mb"] = int(r["k"]), int(r["n"]), int(r["between_mb"])
        r["footprint_kb"], r["ns_per_sample"] = float(r["footprint_kb"]), float(r["ns_per_sample"])
    return rows


def plot_footprint(rows):
    caches = cache_sizes()
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    sizes = sorted({r["n"] for r in rows if r["kind"] == "table"})
    for ax, pattern in zip(axes, ("osc", "random")):
        for color, n in zip(plots.SERIES, sizes):
            for between, ls in BETWEEN_STYLE.items():
                pts = sorted((r["footprint_kb"], r["ns_per_sample"]) for r in rows if r["kind"] == "table"
                             and r["pattern"] == pattern and r["n"] == n and r["between_mb"] == between)
                if pts:
                    x, y = zip(*pts)
                    ax.plot(x, y, ls, color=color, lw=1.6, marker="o", ms=3,
                            label=f"table N={n}" if between == 0 else None)
        for kind, label in (("math", "polynomials, vectorized"), ("scalar", "polynomials, scalar")):
            ys = [r["ns_per_sample"] for r in rows if r["kind"] == kind and r["pattern"] == pattern
                  and r["between_mb"] == 0 and r["k"] >= 4]
            ax.axhline(np.median(ys), color=plots.TRUTH if kind == "math" else plots.MUTED, lw=1.4, ls=":")
            ax.annotate(f"{label}: {np.median(ys):.2f} ns", (0.01, np.median(ys)), xycoords=("axes fraction", "data"),
                        va="bottom", fontsize=9, color=plots.TRUTH if kind == "math" else plots.MUTED)
        for label, b in caches:
            ax.axvline(b / 1024, color=plots.MUTED, lw=0.8)
            ax.annotate(label, (b / 1024, 0.97), xycoords=("data", "axes fraction"), ha="right", va="top",
                        rotation=90, fontsize=8, color=plots.MUTED)
        ax.set_xscale("log")
        ax.set_ylim(0, None)
        ax.set_ylabel("ns per sample per instance")
        ax.set_title(f"pattern: {pattern}" + ("  (oscillator phase: sequential reads)" if pattern == "osc"
                                              else "  (hashed phase: unpredictable reads)"), fontsize=10)
        plots.clean_axes(ax)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    axes[-1].set_xlabel("total table footprint, K instances × 3 tables × (N + 1) floats (KB)")
    fig.suptitle("sin → tanh → dB gain chain: tables vs. polynomials as tables multiply\n"
                 "solid: nothing between rounds; dashed: 16 MB of other work between rounds", fontsize=11)
    fig.tight_layout()
    return fig


def plot_slowdown(rows, n=4096):
    """Tables at one size against K, relative to scalar and vector math."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ref = defaultdict(dict)
    for r in rows:
        if r["kind"] in ("math", "scalar"):
            ref[(r["pattern"], r["between_mb"])][(r["kind"], r["k"])] = r["ns_per_sample"]
    i = 0
    for pattern in ("osc", "random"):
        for between, ls in BETWEEN_STYLE.items():
            pts = sorted((r["k"], r["ns_per_sample"]) for r in rows if r["kind"] == "table" and r["n"] == n
                         and r["pattern"] == pattern and r["between_mb"] == between)
            k, y = zip(*pts)
            scalar = [ref[(pattern, between)][("scalar", kk)] for kk in k]
            ax.plot(k, np.array(y) / np.array(scalar), ls, color=plots.SERIES[i // 2], lw=1.6, marker="o", ms=3,
                    label=f"{pattern}, {between} MB between")
            i += 1
    ax.axhline(1, color=plots.MUTED, lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("instances K (each with its own three tables)")
    ax.set_ylabel("table time / scalar polynomial time")
    ax.set_title(f"N = {n}: below 1 the tables win, above it the polynomials do (vectorized math is ~3.5× lower still)",
                 fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    plots.clean_axes(ax)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", default=str(ROOT / "out" / "tables.csv"))
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    rows = load(args.csv)
    out = Path(args.csv).parent
    for name, fig in (("tables", plot_footprint(rows)), ("tables_slowdown", plot_slowdown(rows))):
        fig.savefig(out / f"{name}.png", dpi=130)
    print(f"Saved plots to {out}/")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
