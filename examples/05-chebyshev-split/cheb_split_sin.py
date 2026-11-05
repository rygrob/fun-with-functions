"""Chebyshev approximation of sin(x) on [-pi, pi], zeros factored, pi split.

04 builds the zeros in as (x - pi)(x + pi), but in f32 "pi" is 3.14159274,
8.7e-8 above true pi. Near x = pi the true answer is about pi - x, so a zero
in the wrong place is a relative error of 100% at the last few inputs.

Fix: carry pi as two f32 values,

    pi_hi = f32(pi)             =  3.14159274
    pi_lo = f32(pi - pi_hi)     = -8.74227766e-08

and evaluate

    ((x - pi_hi) - pi_lo) * ((x + pi_hi) + pi_lo) * P(x^2) * x

Near x = pi, x - pi_hi is exact (Sterbenz: the two are within a factor of 2),
so subtracting the tiny pi_lo afterwards leaves the true distance to pi
instead of rounding it away. Same coefficients as 04, two extra adds.

Fitting and evaluation live in ../common/sinpoly.py; error metrics and
spectrum analysis in ../common/metrics.py.
After https://mooooo.ooo/chebyshev-sine-approximation/

Usage:
    python cheb_split_sin.py                 # DEGREE below
    python cheb_split_sin.py -d 11           # one degree, with coefficients
    python cheb_split_sin.py --compare       # overlay COMPARE_DEGREES
    python cheb_split_sin.py --eval f64      # evaluate in f64
    python cheb_split_sin.py --no-show       # just write the PNGs
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, fit, harness, plots, report, sinpoly
from common.metrics import EVAL_DTYPES, N_SAMPLES, PI

# Total degree of the sin approximation; odd, >= 3. P gets (degree - 1) / 2 terms.
DEGREE = 13
COMPARE_DEGREES = [7, 9, 11, 13]

STEP = "split"
OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def degree_label(results):
    degs = [r.key for r in results]
    return f"degree {degs[0]}" if len(degs) == 1 else "degrees " + ", ".join(map(str, degs))


def plot_function(ctx, results, dtype):
    """sin vs. the approximation, and the near-pi close-up with 04's
    single-constant version alongside for contrast."""
    fig, (ax, ax_pi) = plt.subplots(2, 1, figsize=(10, 8.5))

    ax.plot(ctx.x, ctx.truth, color=plots.TRUTH, lw=2.5, label="sin(x)")
    for color, r in zip(plots.SERIES, results):
        ax.plot(ctx.x, r.approx, color=color, lw=2, ls="--", label=f"Chebyshev split-π, {r.label}")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"sin(x) vs. split-π factored Chebyshev polynomial, {degree_label(results)}")
    ax.legend(loc="upper left", frameon=False)
    plots.style_axes(ax)

    deg = results[-1].key
    plots.plot_near_pi(ax_pi, [
        (f"factored (04), degree {deg}", plots.MUTED, sinpoly.method("factored", deg, dtype).f),
        (f"split-π, degree {deg}", plots.SERIES[len(results) - 1], sinpoly.method(STEP, deg, dtype).f),
    ])
    fig.tight_layout()
    return fig


def fmt_coeff(c, dtype):
    """Print the coefficient as the eval dtype actually stores it. 9 significant
    digits round-trip an f32; 17 round-trip an f64."""
    return f"{float(dtype(c)): .8e}" if dtype == np.float32 else f"{float(c): .16e}"


def print_coeffs(deg, dtype):
    print(f"Coefficients ({np.dtype(dtype).name}):  sin(x) ~ ((x - π_hi) - π_lo)((x + π_hi) + π_lo) * P(x^2) * x")
    for k, c in enumerate(sinpoly.fit_cheb_factored(deg)):
        print(f"  P x^{2 * k:<2d}  {fmt_coeff(c, dtype)}")
    hi, lo = sinpoly.split_pi(dtype)
    print(f"  π_hi  {fmt_coeff(hi, dtype)}")
    print(f"  π_lo  {fmt_coeff(lo, dtype)}")
    print(f"  π_hi + π_lo - π = {float(hi) + float(lo) - PI:+.3e}  (π_hi alone: {float(hi) - PI:+.3e})")
    print()
    m = sinpoly.method(STEP, deg, dtype)
    print(f"Cost: {m.mul} mul, {m.add} add")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-d", "--degree", type=int, default=DEGREE, help=f"total polynomial degree (default {DEGREE})")
    mode.add_argument("--compare", action="store_true", help=f"overlay degrees {COMPARE_DEGREES}")
    parser.add_argument("--eval", choices=EVAL_DTYPES, default="f32", help="precision to evaluate the polynomial in")
    fit.add_argument(parser)
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--emit-cpp", action="store_true", help="print the C++ function for this configuration and exit")
    args = parser.parse_args()
    fit.set_method(args.fit)

    raw = COMPARE_DEGREES if args.compare else [args.degree]
    degrees = sorted({d if d % 2 else d - 1 for d in raw})
    if min(degrees) < 3:
        parser.error("degree must be at least 3")
    dtype = EVAL_DTYPES[args.eval]

    if args.emit_cpp:
        if args.compare:
            parser.error("--emit-cpp emits one configuration; drop --compare")
        cppgen.print_emitted(cppgen.emit_cheb_split(degrees[0])[1])
        return

    ctx = harness.make_context()
    report.print_header(f"Split-π factored Chebyshev approximation ({'compare' if args.compare else f'degree {degrees[0]}'})", [
        ("function", "sin(x)"),
        ("range", "[-π, π]"),
        ("fit", f"Chebyshev {fit.get_method()} of sin(x) / (x (x² − π²)), converted to monomials"),
        ("eval", f"{args.eval}, ((x - π_hi) - π_lo)((x + π_hi) + π_lo) * P(x^2) * x, {N_SAMPLES:,} samples"),
        ("truth", "f64 sin(x)"),
    ])

    results = []
    for deg in degrees:
        results.append(harness.evaluate(sinpoly.method(STEP, deg, dtype).f, deg, f"degree {deg}", ctx))

    if args.compare:
        report.print_compare(results, "degree")
    else:
        deg = degrees[0]
        print_coeffs(deg, dtype)
        report.print_result(results[0])
        rows = sinpoly.progression(STEP, deg, dtype, ctx)
        report.print_progression(rows, f"Progression, degree {deg}:")
    report.print_reference(ctx)

    what = f"{degree_label(results)}, evaluated in {args.eval}"
    harness.save_figures(OUT_DIR, f"cheb_split_sin_{'compare' if args.compare else degrees[0]}_{args.eval}", {
        "": plot_function(ctx, results, dtype),
        "error": plots.plot_errors(ctx.x, results, f"Split-π Chebyshev sin(x) error, {what}"),
        "spectrum": plots.plot_spectrum(results, ctx.ref, f"Split-π Chebyshev sin(x) harmonics, {what}"),
        "oscillator": plots.plot_oscillator(results, ctx.ref_osc, f"Split-π Chebyshev sin(x) oscillator, {what}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
