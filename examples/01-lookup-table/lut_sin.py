"""Lookup table approximation of sin(x) on [-pi, pi].

Precompute one period of sin into a table of N entries,

    table[i] = sin(-pi + 2*pi * i / N),   i = 0 .. N   (entry N is a guard
                                                       point equal to entry 0)

then map x to a fractional table position pos = (x + pi) * N / (2*pi) and
read it back:

  nearest  table[round(pos)]                              error ~ step/2
  linear   table[i] + frac * (table[i+1] - table[i])      error ~ step^2/8

where step = 2*pi / N. Doubling the table halves the nearest error (6 dB per
doubling) and quarters the linear error (12 dB per doubling) -- memory buys
accuracy, but slowly.

The table and the index arithmetic are in f32 (what we'd ship in C++) or f64
(to isolate table resolution from f32 rounding). Error metrics and spectrum
analysis live in ../common (see common/metrics.py).

Usage:
    python lut_sin.py                    # SIZE and INTERP below
    python lut_sin.py -n 256             # one table size
    python lut_sin.py -i linear          # nearest | linear
    python lut_sin.py --compare          # overlay COMPARE_SIZES
    python lut_sin.py --eval f64         # table and lookup in f64
    python lut_sin.py --no-show          # just write the PNGs
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, harness, plots, report
from common.metrics import EVAL_DTYPES, N_SAMPLES, PI
from common.sinpoly import eval_lut, make_table

SIZE = 1024
INTERP = "nearest"
COMPARE_SIZES = [64, 256, 1024, 4096]
INTERPS = ["nearest", "linear"]

ZOOM = (np.pi / 4 - 0.08, np.pi / 4 + 0.08)  # x window for the zoomed function view
OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Approximation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def size_label(results, interp):
    return f"{interp}, N = " + ", ".join(str(r.key) for r in results)


def plot_function(ctx, results, interp):
    """Full period on top; a zoomed window below where the steps (nearest) or
    chords (linear) are actually visible."""
    x, truth = ctx.x, ctx.truth
    fig, (ax, ax_z) = plt.subplots(2, 1, figsize=(10, 8))
    zoom = (x >= ZOOM[0]) & (x <= ZOOM[1])
    ax.plot(x, truth, color=plots.TRUTH, lw=2.5, label="sin(x)")
    ax_z.plot(x[zoom], truth[zoom], color=plots.TRUTH, lw=2.5, label="sin(x)")
    for color, r in zip(plots.SERIES, results):
        ax.plot(x, r.approx, color=color, lw=2, ls="--", label=f"LUT, {r.label}")
        ax_z.plot(x[zoom], r.approx[zoom], color=color, lw=1.8, label=f"LUT, {r.label}")
    ax.axvspan(*ZOOM, color=plots.MUTED, alpha=0.15, lw=0)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"sin(x) vs. lookup table, {size_label(results, interp)}")
    ax.legend(loc="upper left", frameon=False)
    plots.style_axes(ax)

    ax_z.set_xlim(*ZOOM)
    ax_z.set_xlabel("x")
    ax_z.set_ylabel("y")
    ax_z.set_title(f"zoom: x in [{ZOOM[0]:.3f}, {ZOOM[1]:.3f}] (shaded above)", fontsize=10)
    plots.clean_axes(ax_z)
    fig.tight_layout()
    return fig


def fmt_bytes(n):
    return f"{n:,} B"


def print_table(size, dtype):
    table = make_table(size, dtype)
    print(f"Table ({table.dtype.name}):")
    print(f"  entries       : {size:,} (+1 guard point)")
    print(f"  memory        : {fmt_bytes(table.nbytes)}")
    print(f"  step          : 2π/N = {2 * PI / size:.3e}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-n", "--size", type=int, default=SIZE, help=f"table entries (default {SIZE})")
    mode.add_argument("--compare", action="store_true", help=f"overlay sizes {COMPARE_SIZES}")
    parser.add_argument("-i", "--interp", choices=INTERPS, default=INTERP, help=f"lookup method (default {INTERP})")
    parser.add_argument("--eval", choices=EVAL_DTYPES, default="f32", help="precision of the table and lookup")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--emit-cpp", action="store_true", help="print the C++ function for this configuration and exit")
    args = parser.parse_args()

    sizes = COMPARE_SIZES if args.compare else [args.size]
    if min(sizes) < 2:
        parser.error("table size must be at least 2")
    dtype = EVAL_DTYPES[args.eval]

    if args.emit_cpp:
        if args.compare:
            parser.error("--emit-cpp emits one configuration; drop --compare")
        cppgen.print_emitted(cppgen.emit_lut(sizes[0], args.interp)[1])
        return

    ctx = harness.make_context()
    report.print_header(f"Lookup table approximation ({'compare' if args.compare else f'N = {sizes[0]}'})", [
        ("function", "sin(x)"),
        ("range", "[-π, π]"),
        ("interp", args.interp),
        ("eval", f"{args.eval}, {N_SAMPLES:,} samples"),
        ("truth", "f64 sin(x)"),
    ])

    results = []
    for size in sizes:
        table = make_table(size, dtype)
        results.append(harness.evaluate(lambda x: eval_lut(x, table, args.interp, dtype), size, f"N = {size}", ctx))

    if args.compare:
        memory = lambda r: fmt_bytes(make_table(r.key, dtype).nbytes)
        report.print_compare(results, "N", extra_cols=[("memory", 10, memory)])
    else:
        print_table(sizes[0], dtype)
        report.print_result(results[0])
    report.print_reference(ctx)

    what = f"{size_label(results, args.interp)}, evaluated in {args.eval}"
    harness.save_figures(OUT_DIR, f"lut_sin_{args.interp}_{'compare' if args.compare else sizes[0]}_{args.eval}", {
        "": plot_function(ctx, results, args.interp),
        "error": plots.plot_errors(ctx.x, results, f"Lookup table sin(x) error, {what}"),
        "spectrum": plots.plot_spectrum(results, ctx.ref, f"Lookup table sin(x) harmonics, {what}"),
        "oscillator": plots.plot_oscillator(results, ctx.ref_osc, f"Lookup table sin(x) oscillator, {what}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
