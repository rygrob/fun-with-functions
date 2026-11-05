"""Chebyshev approximation of sin(x) on [-pi, pi] with the zeros factored out.

Vanilla Chebyshev (03) spreads *absolute* error evenly, so wherever sin(x)
is small -- near ±pi -- the *relative* error explodes, and the waveform
doesn't quite reach zero at the wrap. sin has known zeros at 0 and ±pi, so
build them in:

    sin(x) = x (x - pi)(x + pi) * f(x),   f(x) = sin(x) / (x (x^2 - pi^2))

f is smooth, even, and never far from a constant (it ranges over about a
factor of 2), so a Chebyshev fit of f with small *absolute* error gives sin
small *relative* error everywhere. The fit is in t = x/pi; P(x^2) is f
converted to monomials:

    sin(x) ~ (x - pi)(x + pi) * P(x^2) * x

pi here is one constant in the eval dtype. In f32 that's 3.14159274, which
is 8.7e-8 *above* true pi, so the zero lands on the wrong float -- 05 fixes
that by splitting pi into two f32 terms.

Fitting and evaluation live in ../common/sinpoly.py; error metrics and
spectrum analysis in ../common/metrics.py.

Usage:
    python cheb_factored_sin.py              # DEGREE below
    python cheb_factored_sin.py -d 11        # one degree, with coefficients
    python cheb_factored_sin.py --compare    # overlay COMPARE_DEGREES
    python cheb_factored_sin.py --eval f64   # evaluate in f64
    python cheb_factored_sin.py --no-show    # just write the PNGs
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

STEP = "factored"
OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def degree_label(results):
    degs = [r.key for r in results]
    return f"degree {degs[0]}" if len(degs) == 1 else "degrees " + ", ".join(map(str, degs))


def plot_function(ctx, results, dtype):
    """The quotient f being fitted, sin vs. the approximation, and a close-up
    of the last few f32 inputs below pi where the zero placement shows."""
    fig, (ax_f, ax, ax_pi) = plt.subplots(3, 1, figsize=(10, 12))

    x = ctx.x.astype(np.float64)
    ax_f.plot(x, sinpoly.factored_quotient(x), color=plots.TRUTH, lw=2.5)
    ax_f.set_xlabel("x")
    ax_f.set_ylabel("f(x)")
    ax_f.set_title("What gets fitted: f(x) = sin(x) / (x (x² − π²)) -- smooth, even, no zeros", fontsize=10)
    plots.style_axes(ax_f)

    ax.plot(ctx.x, ctx.truth, color=plots.TRUTH, lw=2.5, label="sin(x)")
    for color, r in zip(plots.SERIES, results):
        ax.plot(ctx.x, r.approx, color=color, lw=2, ls="--", label=f"Chebyshev factored, {r.label}")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"sin(x) vs. factored Chebyshev polynomial, {degree_label(results)}")
    ax.legend(loc="upper left", frameon=False)
    plots.style_axes(ax)

    deg = results[-1].key
    near = [(f"factored, degree {deg}", plots.SERIES[len(results) - 1], sinpoly.method(STEP, deg, dtype).f)]
    plots.plot_near_pi(ax_pi, near)
    fig.tight_layout()
    return fig


def fmt_coeff(c, dtype):
    """Print the coefficient as the eval dtype actually stores it. 9 significant
    digits round-trip an f32; 17 round-trip an f64."""
    return f"{float(dtype(c)): .8e}" if dtype == np.float32 else f"{float(c): .16e}"


def print_coeffs(deg, dtype):
    a = sinpoly.cheb_factored_series(deg)
    print(f"Chebyshev series of f(πt), degree {deg - 3}:  f(πt) = Σ aₙ Tₙ(t)")
    for k in range(0, deg - 2, 2):
        print(f"  a_{k:<2d}  {a[k]: .9e}")
    print()
    print(f"Coefficients ({np.dtype(dtype).name}):  sin(x) ~ (x - π)(x + π) * P(x^2) * x")
    for k, c in enumerate(sinpoly.fit_cheb_factored(deg)):
        print(f"  P x^{2 * k:<2d}  {fmt_coeff(c, dtype)}")
    pi = dtype(PI)
    print(f"  π     {fmt_coeff(pi, dtype)}  ({float(pi) - PI:+.3e} from true π)")
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
        cppgen.print_emitted(cppgen.emit_cheb_factored(degrees[0])[1])
        return

    ctx = harness.make_context()
    report.print_header(f"Factored Chebyshev approximation ({'compare' if args.compare else f'degree {degrees[0]}'})", [
        ("function", "sin(x)"),
        ("range", "[-π, π]"),
        ("fit", f"Chebyshev {fit.get_method()} of sin(x) / (x (x² − π²)), converted to monomials"),
        ("eval", f"{args.eval}, (x - π)(x + π) * P(x^2) * x, {N_SAMPLES:,} samples"),
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
        report.print_progression(rows, f"Progression so far, degree {deg}:")
    report.print_reference(ctx)

    what = f"{degree_label(results)}, evaluated in {args.eval}"
    harness.save_figures(OUT_DIR, f"cheb_factored_sin_{'compare' if args.compare else degrees[0]}_{args.eval}", {
        "": plot_function(ctx, results, dtype),
        "error": plots.plot_errors(ctx.x, results, f"Factored Chebyshev sin(x) error, {what}"),
        "spectrum": plots.plot_spectrum(results, ctx.ref, f"Factored Chebyshev sin(x) harmonics, {what}"),
        "oscillator": plots.plot_oscillator(results, ctx.ref_osc, f"Factored Chebyshev sin(x) oscillator, {what}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
