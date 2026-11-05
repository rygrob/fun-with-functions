"""2^x by splitting x into an integer part (exponent bits) and a fraction
(a polynomial).

A float is already 2^e * m with e an integer field, so 2^x = 2^n * 2^f
needs only the fraction approximated: 2^n is written straight into the
exponent bits as (n + 127) << 23. Audio uses it everywhere: mtof is
440 * 2^((note - 69) / 12), dB to gain is 2^(dB * log2(10) / 20).

  bits            Schraudolph: write x * 2^23 into the bits as an integer.
                  floor(x) lands in the exponent field and the fraction in
                  the significand, so 2^f ~ 1 + f, the chord: 6% error with no
                  arithmetic at all, 3% with a tuned offset
  floor + P(f)    n = floor(x), P(f) ~ 2^f on [0, 1). vsl::exp2 (degree 6)
  round + P(f)    n = round(x), f on [-1/2, 1/2]. The reduction is exact
                  (Sterbenz) but the fit is *no* better: 2^f on any unit
                  interval is the same curve times a constant
  table + P(f)    k = round(N x): the top bits of k give n, the low log2 N bits
                  index a table of 2^(j/N), and the rest of x gets a degree-3
                  polynomial on [-1/2N, 1/2N]. 16 entries reach < 1 ULP

The range reduction is exact, so the error is periodic in x: every unit
interval looks the same, and the error columns below hold for any n in the
normal range. The edges don't: 2^n from bits is only right for n in
[-126, 127], and the edge table shows what happens outside.

Error is shown as ULP, and as the audio units it turns into: a relative error
in 2^x is a pitch error of 1200 * log2(1 + rel) cents (mtof) and a gain error
of 20 * log10(1 + rel) dB.

Fitting and evaluation live in ../common/exp2poly.py, bit fields in
../common/bits.py.

Usage:
    python exp2_bits.py                  # DEGREE below, side by side
    python exp2_bits.py -d 5             # another degree
    python exp2_bits.py --compare        # degree sweep per method
    python exp2_bits.py --eval f64       # evaluate in f64
    python exp2_bits.py --emit-cpp       # C++ for the polynomial methods
    python exp2_bits.py --no-show        # just write the PNGs
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, exp2poly, fit, harness, metrics, plots, report
from common.metrics import EVAL_DTYPES, N_SAMPLES

# Degree of P(f). 5 is the slides' (with 1 + f Q(f)); 6 is vsl::exp2. The
# table method uses TABLE_DEGREE.
DEGREE = 5
TABLE_DEGREE = 3
TABLE_SIZE = 16
COMPARE_DEGREES = [3, 4, 5, 6, 7]
COMPARE_TABLE_DEGREES = [1, 2, 3, 4]

STATS_RANGE = (-125.0, 126.0)   # normal results with margin; edges are tabled separately
PLOT_RANGE = (-3.0, 3.0)
EDGE_INPUTS = [-151, -150, -149.5, -140, -127, -126.5, -126, 127.4, 127.6, 127.99, 128, 129, 200]

OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Inputs and evaluation
# ---------------------------------------------------------------------------

@dataclass
class Context:
    x: np.ndarray       # f32 inputs, sorted
    truth: np.ndarray   # f64 2^x


@dataclass
class Result:
    name: str
    m: exp2poly.Method
    y: np.ndarray
    ulp: np.ndarray
    rel: np.ndarray     # signed
    stats: dict


def make_context():
    """Dense over PLOT_RANGE (a few periods), plus uniform over STATS_RANGE."""
    rng = np.random.default_rng(1)
    x = np.concatenate([np.linspace(*PLOT_RANGE, 2 * N_SAMPLES), rng.uniform(*STATS_RANGE, N_SAMPLES)])
    x = np.unique(x.astype(np.float32))
    return Context(x=x, truth=np.exp2(x.astype(np.float64)))


def evaluate(name, m, ctx):
    y = m.f(ctx.x).astype(np.float64)
    rel = y / ctx.truth - 1
    ratio = np.abs(1 + rel)
    ulp = metrics.ulp_error_f32(y, ctx.truth)
    return Result(name, m, y, ulp, rel, {
        "ulp": ulp.max(),
        "rel": np.abs(rel).max(),
        "cents": np.abs(1200 * np.log2(ratio)).max(),
        "db": np.abs(20 * np.log10(ratio)).max(),
    })


def run(steps, ctx, dtype):
    return [evaluate(exp2poly.STEP_NAMES[s], exp2poly.method(s, d, dtype, n), ctx) for s, d, n in steps]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(results, title):
    print(title)
    header = (f"  {'method':<20} {'degree':>9} {'coeffs':>6} {'mul':>4} {'add':>4}"
              f"  {'max ULP':>9}  {'max rel':>9}  {'cents':>9}  {'dB':>9}")
    print(header)
    for r in results:
        s, m = r.stats, r.m
        print(f"  {r.name:<20} {m.degree:>9} {m.ncoef:>6} {m.mul:>4} {m.add:>4}"
              f"  {report.fmt_ulp(s['ulp']):>9}  {s['rel']:>9.2e}  {s['cents']:>9.2e}  {s['db']:>9.2e}")
    print(f"  (x in [{STATS_RANGE[0]:g}, {STATS_RANGE[1]:g}]; cents = mtof pitch error, dB = gain error;"
          " ops exclude the float -> int and integer bit ops)")
    print()


def fmt_coeff(c, dtype):
    """Print the coefficient as the eval dtype actually stores it. 9 significant
    digits round-trip an f32; 17 round-trip an f64."""
    return f"{float(dtype(c)): .8e}" if dtype == np.float32 else f"{float(c): .16e}"


def print_coeffs(deg, dtype):
    for reduction, (lo, hi) in exp2poly.REDUCTIONS.items():
        print(f"Coefficients, {reduction} ({np.dtype(dtype).name}):  2^f ~ P(f), f in [{lo:g}, {hi:g}]")
        for k, c in enumerate(exp2poly.fit_exp2(deg, reduction)):
            print(f"  P f^{k:<2d}  {fmt_coeff(c, dtype)}")
        print()
    o = exp2poly.bits_offset()
    print(f"Schraudolph offset: x + {o:.6f} before writing the bits "
          f"(= subtracting {-o * 2**23:,.0f} from the integer, scales by 2^{o:.4f} = {2**o:.4f})")
    print()


def print_edges(deg, dtype):
    """What each edge strategy returns where 2^n leaves the normal range."""
    c_floor = exp2poly.fit_exp2(deg, "floor")
    c_round = exp2poly.fit_exp2(deg, "round")
    x = np.array(EDGE_INPUTS, dtype=dtype)
    cols = [
        ("true 2^x", np.exp2(x.astype(np.float64))),
        ("floor, raw", exp2poly.eval_poly(x, c_floor, "floor", dtype)),
        ("floor, vsl", exp2poly.eval_vsl_edges(x, c_floor, dtype)),
        ("round, raw", exp2poly.eval_poly(x, c_round, "round", dtype)),
        ("round, full", exp2poly.eval_full_range(x, c_round, "round", dtype)),
    ]
    print(f"Edges, degree {deg} ({np.dtype(dtype).name}): 2^n is only right for n in [-126, 127]")
    print("  " + f"{'x':>8}" + "".join(f"  {name:>13}" for name, _ in cols))
    for i, xi in enumerate(x):
        print("  " + f"{float(xi):>8g}" + "".join(f"  {float(v[i]):>13.6g}" for _, v in cols))
    print("  raw:  n wraps the exponent field: below -126 garbage, n = 128 inf, n >= 129 into the sign bit")
    print("        (round reaches n = 128 already at x = 127.5)")
    print("  vsl:  select 0 below -126 -- subnormal results flush, x >= 129 still wraps")
    print("  full: clamp x, then 2^n = 2^(n/2) * 2^(n - n/2): subnormals and overflow right, +1 mul")
    print()


def label(r):
    return r.name if r.m.degree == "—" else f"{r.name}, {r.m.degree}"


def plot_function(ctx, results, title):
    """2^x and the bit trick's chords; relative error in cents underneath."""
    fig, (ax, ax_e) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
    sel = (ctx.x >= PLOT_RANGE[0]) & (ctx.x <= PLOT_RANGE[1])
    x = ctx.x[sel]
    ax.plot(x, ctx.truth[sel], color=plots.TRUTH, lw=2.5, label="2^x")
    for color, r in zip(plots.SERIES, results):
        ax.plot(x, r.y[sel], color=color, lw=1.8, ls="--", label=label(r))
        ax_e.plot(x, 1200 * np.log2(1 + r.rel[sel]), color=color, lw=1.5, label=label(r))
    for k in range(int(PLOT_RANGE[0]), int(PLOT_RANGE[1]) + 1):
        ax.axvline(k, color=plots.MUTED, lw=0.5, ls=":")
        ax_e.axvline(k, color=plots.MUTED, lw=0.5, ls=":")
    ax.set_yscale("log", base=2)
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax_e.axhline(0, color=plots.MUTED, lw=0.8)
    ax_e.set_ylabel("error (cents)")
    ax_e.set_xlabel("x  (dotted: integers, where the exponent field steps)")
    for a in (ax, ax_e):
        plots.clean_axes(a)
        a.set_xlim(*PLOT_RANGE)
    fig.tight_layout()
    return fig


def plot_errors(ctx, results, title):
    """ULP error over a few periods: binned max envelope over faint samples."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sel = (ctx.x >= PLOT_RANGE[0]) & (ctx.x <= PLOT_RANGE[1])
    x = ctx.x[sel]
    for color, r in zip(plots.SERIES, results):
        ax.plot(x, r.ulp[sel], ".", color=color, ms=1, alpha=0.15, rasterized=True)
    for color, r in zip(plots.SERIES, results):
        ax.plot(*plots.binned_max(x, r.ulp[sel]), color=color, lw=1.8, label=label(r))
    plots._label_line(ax, 0.5, "0.5 ULP (correctly rounded)")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_xlim(*PLOT_RANGE)
    ax.set_xlabel("x")
    ax.set_ylabel("error (f32 ULPs)")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    plots.clean_axes(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-d", "--degree", type=int, default=DEGREE, help=f"degree of P(f) (default {DEGREE})")
    mode.add_argument("--compare", action="store_true", help=f"sweep degrees {COMPARE_DEGREES}")
    parser.add_argument("--eval", choices=EVAL_DTYPES, default="f32", help="precision to evaluate in")
    fit.add_argument(parser)
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--emit-cpp", action="store_true", help="print the C++ for this configuration and exit")
    args = parser.parse_args()
    fit.set_method(args.fit)
    if args.degree < 1:
        parser.error("degree must be at least 1")
    dtype = EVAL_DTYPES[args.eval]

    if args.emit_cpp:
        if args.compare:
            parser.error("--emit-cpp emits one configuration; drop --compare")
        code = "\n".join([cppgen.bit_helpers()] + [e[1] for e in (
            cppgen.emit_exp2(args.degree, "floor"), cppgen.emit_exp2(args.degree, "round"),
            cppgen.emit_exp2_unit(args.degree),
            cppgen.emit_exp2_table(TABLE_DEGREE, TABLE_SIZE), cppgen.emit_exp2_bits(True))])
        cppgen.print_emitted(code)
        print("// (also needs <bit>)")
        return

    ctx = make_context()
    report.print_header(f"2^x from exponent bits + polynomial ({'compare' if args.compare else f'degree {args.degree}'})", [
        ("function", "2^x = 2^n * 2^f, 2^n written into the exponent field"),
        ("range", f"{len(ctx.x):,} f32 inputs: dense on [{PLOT_RANGE[0]:g}, {PLOT_RANGE[1]:g}], "
                  f"uniform on [{STATS_RANGE[0]:g}, {STATS_RANGE[1]:g}]"),
        ("eval", f"{args.eval}"),
        ("truth", "f64 2^x"),
    ])

    if args.compare:
        steps = ([(s, d, TABLE_SIZE) for s in ("floor", "round", "unit") for d in COMPARE_DEGREES]
                 + [("table", d, TABLE_SIZE) for d in COMPARE_TABLE_DEGREES])
        results = run(steps, ctx, dtype)
        print_table(results, "Degree sweep:")
        plotted = [r for r in results if r.m.degree in ("4", "6") or r.m.degree.startswith(f"{TABLE_DEGREE},")]
        tag, what = "compare", f"degree sweep, evaluated in {args.eval}"
    else:
        print_coeffs(args.degree, dtype)
        steps = [("bits", None, TABLE_SIZE), ("bits-tuned", None, TABLE_SIZE), ("floor", args.degree, TABLE_SIZE),
                 ("round", args.degree, TABLE_SIZE), ("unit", args.degree, TABLE_SIZE),
                 ("table", TABLE_DEGREE, TABLE_SIZE)]
        results = run(steps, ctx, dtype)
        print_table(results, f"Side by side, degree {args.degree}:")
        print_edges(args.degree, dtype)
        plotted = results
        tag, what = str(args.degree), f"degree {args.degree}, evaluated in {args.eval}"

    harness.save_figures(OUT_DIR, f"exp2_bits_{tag}_{args.eval}", {
        "": plot_function(ctx, [r for r in plotted if r.m.ncoef == 0] or plotted[:2],
                          "2^x: the exponent field steps at integers, the bits fill in a chord"),
        "error": plot_errors(ctx, [r for r in plotted if r.m.ncoef > 0], f"2^x error, {what}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
