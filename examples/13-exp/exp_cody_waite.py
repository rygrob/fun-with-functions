"""e^x: exp2's bit split, plus getting from base e to base 2 exactly.

e^x = 2^n * e^r with n = round(x / ln 2) and r = x - n ln 2. The 2^n is
exponent bits as in 10-exp2; the trouble is r. n ln 2 is as big as ~87 while
r is at most 0.35, so any rounding in n ln 2 -- or in x log2 e, if you go
via exp2 -- is a large relative error in r, and so in the result:

  exp2(x log2 e)   x log2 e rounds by up to half an ulp of ~127: ~64 ULP at
                   |x| ~ 87, growing linearly with |x|
  single ln2       r = x - n * f32(ln 2): the same, from n * ln2's rounding
  Cody–Waite       ln 2 = ln2_hi + ln2_lo with ln2_hi on 16 bits, so
                   n * ln2_hi is exact, x - n * ln2_hi is exact (Sterbenz) and
                   n * ln2_lo is tiny: ~1 ULP everywhere. It is split-π again
                   (05), with the same weakness: -ffast-math may fold the two
                   subtractions back into one
  vsl              vsl::exp: floor reduction, the same split, then reuse
                   exp2's polynomial on r log2 e

For audio the arguments are usually small -- a one-pole's exp(-2π fc/fs)
has |x| <= π, an envelope's exp(-t/τ) a few units -- and there the naive
route is within a few ULP. The split buys ULPs at large |x|, not dB.

Fitting and evaluation live in ../common/exppoly.py.

Usage:
    python exp_cody_waite.py              # DEGREE below, side by side
    python exp_cody_waite.py --compare    # degree sweep
    python exp_cody_waite.py --eval f64   # evaluate in f64
    python exp_cody_waite.py --emit-cpp   # C++ for the reduced forms
    python exp_cody_waite.py --no-show    # just write the PNGs
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, exppoly, fit, harness, metrics, plots, report
from common.metrics import EVAL_DTYPES, N_SAMPLES

DEGREE = 6
COMPARE_DEGREES = [5, 6, 7]
X_RANGE = (-87.0, 88.0)     # results stay normal
SMALL = 10.0                # "audio" arguments: |x| <= SMALL
ONE_POLE = np.pi            # one-pole coefficients: |x| <= pi

OUT_DIR = Path(__file__).parent / "out"


def make_inputs():
    x = np.concatenate([np.linspace(*X_RANGE, 3 * N_SAMPLES), np.linspace(-SMALL, SMALL, N_SAMPLES)])
    x = np.unique(x.astype(np.float32))
    return x, np.exp(x.astype(np.float64))


def evaluate(step, degree, x, truth, dtype):
    m = exppoly.method(step, degree, dtype)
    y = m.f(x).astype(np.float64)
    ulp = metrics.ulp_error_f32(y, truth)
    rel = np.abs(y / truth - 1)
    worst = int(np.argmax(ulp))
    small, pole = np.abs(x) <= SMALL, np.abs(x) <= ONE_POLE
    return exppoly.STEP_NAMES[step], m, ulp, {
        "ulp": ulp[worst], "at": float(x[worst]), "ulp_small": ulp[small].max(),
        "ulp_pole": ulp[pole].max(), "db": 20 * np.log10(1 + rel.max()),
    }


def print_table(rows, title):
    print(title)
    print(f"  {'method':<20} {'degree':>6} {'mul':>4} {'add':>4}  {'max ULP':>9} {'at x':>9}"
          f"  {'|x|<=10':>9}  {'|x|<=π':>9}  {'max dB':>9}")
    for name, m, _, s in rows:
        print(f"  {name:<20} {m.degree:>6} {m.mul:>4} {m.add:>4}  {report.fmt_ulp(s['ulp']):>9} {s['at']:>9.3f}"
              f"  {report.fmt_ulp(s['ulp_small']):>9}  {report.fmt_ulp(s['ulp_pole']):>9}  {s['db']:>9.2e}")
    print(f"  (x in [{X_RANGE[0]:g}, {X_RANGE[1]:g}]; |x| <= π covers one-pole coefficients exp(-2π fc/fs))")
    print()


def plot_errors(x, rows, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    for color, (name, m, ulp, _) in zip(plots.SERIES, rows):
        ax.plot(*plots.binned_max(x, ulp), color=color, lw=1.8, label=f"{name}, {m.degree}")
    plots._label_line(ax, 0.5, "0.5 ULP (correctly rounded)")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_xlim(*X_RANGE)
    ax.set_xlabel("x")
    ax.set_ylabel("error (f32 ULPs)")
    ax.set_title(title)
    ax.legend(loc="upper center", frameon=False, fontsize=9, ncol=2)
    plots.clean_axes(ax)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-d", "--degree", type=int, default=DEGREE, help=f"polynomial degree (default {DEGREE})")
    mode.add_argument("--compare", action="store_true", help=f"sweep degrees {COMPARE_DEGREES}")
    parser.add_argument("--eval", choices=EVAL_DTYPES, default="f32", help="precision to evaluate in")
    fit.add_argument(parser)
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--emit-cpp", action="store_true", help="print the C++ for this configuration and exit")
    args = parser.parse_args()
    fit.set_method(args.fit)
    dtype = EVAL_DTYPES[args.eval]

    if args.emit_cpp:
        if args.compare:
            parser.error("--emit-cpp emits one configuration; drop --compare")
        code = "\n".join([cppgen.bit_helpers(), cppgen.emit_exp_cody_waite(args.degree)[1],
                          cppgen.emit_exp_via_exp2(args.degree)[1]])
        cppgen.print_emitted(code)
        print("// (also needs <bit>)")
        return

    x, truth = make_inputs()
    report.print_header(f"e^x with Cody–Waite reduction ({'compare' if args.compare else f'degree {args.degree}'})", [
        ("function", "e^x = 2^n * e^r, n = round(x / ln 2), r = x - n ln 2"),
        ("range", f"{len(x):,} f32 inputs on [{X_RANGE[0]:g}, {X_RANGE[1]:g}], dense on [-{SMALL:g}, {SMALL:g}]"),
        ("eval", args.eval),
        ("truth", "f64 e^x"),
    ])
    hi, lo = exppoly.split_ln2(dtype)
    print(f"ln 2 = ln2_hi + ln2_lo:  {float(hi):.9g} ({exppoly.HI_BITS} bits, = vsl's) + {float(lo):.9g}")
    print()

    degrees = COMPARE_DEGREES if args.compare else [args.degree]
    rows = [evaluate(s, d, x, truth, dtype) for s in exppoly.STEP_NAMES for d in degrees]
    print_table(rows, "Degree sweep:" if args.compare else f"Side by side, degree {args.degree}:")

    plotted = rows if not args.compare else [r for r in rows if r[1].degree == DEGREE]
    tag = "compare" if args.compare else str(args.degree)
    harness.save_figures(OUT_DIR, f"exp_cody_waite_{tag}_{args.eval}", {
        "error": plot_errors(x, plotted, f"e^x error: rounding in n ln 2 grows with |x|, evaluated in {args.eval}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
