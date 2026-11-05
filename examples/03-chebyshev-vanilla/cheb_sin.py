"""Chebyshev approximation of sin(x) on [-pi, pi] -- the vanilla version.

Write sin(pi t), t in [-1, 1], as a Chebyshev series

    sin(pi t) = sum a_n T_n(t),   T_n(cos θ) = cos(n θ)

For a smooth function the a_n fall off geometrically, so truncating after
degree N leaves an error about the size of the first dropped term -- spread
evenly across the whole range, not piled up at the edges like Taylor. Here
the a_n come from interpolating at Chebyshev nodes, which is near-minimax in
*absolute* error.

The series is then converted to monomials and evaluated like Taylor,
x * P(x^2) via Horner, in f32 or f64. Nothing is done about the zeros at
0 and ±pi or about f32 rounding: that's 04 and 05.

Fitting and evaluation live in ../common/sinpoly.py; error metrics and
spectrum analysis in ../common/metrics.py.

Usage:
    python cheb_sin.py                   # DEGREE below
    python cheb_sin.py -d 9              # one degree, with coefficients
    python cheb_sin.py --compare         # overlay COMPARE_DEGREES
    python cheb_sin.py --eval f64        # evaluate in f64
    python cheb_sin.py --no-show         # just write the PNGs
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, fit, harness, plots, report, sinpoly
from common.metrics import EVAL_DTYPES, N_SAMPLES

# Odd numbers only; even ones round down since sin is odd.
DEGREE = 13
COMPARE_DEGREES = [7, 9, 11, 13]
COEFF_PLOT_TERMS = 31  # how far out to show the Chebyshev coefficient decay

STEP = "vanilla"
OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def degree_label(results):
    degs = [r.key for r in results]
    return f"degree {degs[0]}" if len(degs) == 1 else "degrees " + ", ".join(map(str, degs))


def plot_function(ctx, results):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ctx.x, ctx.truth, color=plots.TRUTH, lw=2.5, label="sin(x)")
    for color, r in zip(plots.SERIES, results):
        ax.plot(ctx.x, r.approx, color=color, lw=2, ls="--", label=f"Chebyshev, {r.label}")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"sin(x) vs. Chebyshev polynomial, {degree_label(results)}")
    ax.legend(loc="upper left", frameon=False)
    plots.style_axes(ax)
    fig.tight_layout()
    return fig


def plot_coefficients(results):
    """|a_n| of sin(pi t) on a log scale: geometric decay, and the truncation
    error at degree N is about |a_(N+2)|, the first odd term dropped."""
    a = np.abs(sinpoly.cheb_vanilla_series(COEFF_PLOT_TERMS))
    n = np.arange(len(a))
    odd = n % 2 == 1
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(n[odd], a[odd], "o-", color=plots.TRUTH, lw=1.5, ms=5, label="|aₙ|, odd n (even n are 0)")
    for color, r in zip(plots.SERIES, results):
        ax.axvline(r.key + 1, color=color, lw=1.5, ls="--", label=f"truncate after {r.label}")
    ax.axhline(2.0**-24, color=plots.MUTED, ls=":", lw=1)
    ax.annotate("2⁻²⁴", (COEFF_PLOT_TERMS, 2.0**-24), ha="right", va="bottom", color=plots.MUTED, fontsize=9)
    ax.set_xlim(0, COEFF_PLOT_TERMS)
    ax.set_xlabel("n")
    ax.set_ylabel("|aₙ|")
    ax.set_title("Chebyshev coefficients of sin(πt): sin(πt) = Σ aₙ Tₙ(t)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    plots.clean_axes(ax)
    fig.tight_layout()
    return fig


def fmt_coeff(c, dtype):
    """Print the coefficient as the eval dtype actually stores it. 9 significant
    digits round-trip an f32; 17 round-trip an f64."""
    return f"{float(dtype(c)): .8e}" if dtype == np.float32 else f"{float(c): .16e}"


def print_coeffs(deg, dtype):
    a = sinpoly.cheb_vanilla_series(deg)
    print(f"Chebyshev series, degree {deg}:  sin(πt) = Σ aₙ Tₙ(t)")
    for k in range(1, deg + 1, 2):
        print(f"  a_{k:<2d}  {a[k]: .9e}")
    print(f"  first dropped term |a_{deg + 2}| ≈ {abs(sinpoly.cheb_vanilla_series(deg + 2)[deg + 2]):.2e}")
    print()
    print(f"Monomial coefficients ({np.dtype(dtype).name}):  sin(x) ~ x * P(x^2)")
    for k, c in enumerate(sinpoly.fit_cheb_vanilla(deg)):
        print(f"  x^{2 * k + 1:<2d}  {fmt_coeff(c, dtype)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-d", "--degree", type=int, default=DEGREE, help=f"polynomial degree (default {DEGREE})")
    mode.add_argument("--compare", action="store_true", help=f"overlay degrees {COMPARE_DEGREES}")
    parser.add_argument("--eval", choices=EVAL_DTYPES, default="f32", help="precision to evaluate the polynomial in")
    fit.add_argument(parser)
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--emit-cpp", action="store_true", help="print the C++ function for this configuration and exit")
    args = parser.parse_args()
    fit.set_method(args.fit)

    raw = COMPARE_DEGREES if args.compare else [args.degree]
    degrees = sorted({d if d % 2 else d - 1 for d in raw})
    dtype = EVAL_DTYPES[args.eval]

    if args.emit_cpp:
        if args.compare:
            parser.error("--emit-cpp emits one configuration; drop --compare")
        cppgen.print_emitted(cppgen.emit_cheb_vanilla(degrees[0])[1])
        return

    ctx = harness.make_context()
    report.print_header(f"Chebyshev approximation ({'compare' if args.compare else f'degree {degrees[0]}'})", [
        ("function", "sin(x)"),
        ("range", "[-π, π]"),
        ("fit", f"Chebyshev {fit.get_method()} of sin(πt), converted to monomials"),
        ("eval", f"{args.eval}, Horner in x^2, {N_SAMPLES:,} samples"),
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
    harness.save_figures(OUT_DIR, f"cheb_sin_{'compare' if args.compare else degrees[0]}_{args.eval}", {
        "": plot_function(ctx, results),
        "coefficients": plot_coefficients(results),
        "error": plots.plot_errors(ctx.x, results, f"Chebyshev sin(x) error, {what}"),
        "spectrum": plots.plot_spectrum(results, ctx.ref, f"Chebyshev sin(x) harmonics, {what}"),
        "oscillator": plots.plot_oscillator(results, ctx.ref_osc, f"Chebyshev sin(x) oscillator, {what}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
