"""Taylor series approximation of sin(x) on [-pi, pi].

The Taylor (Maclaurin) series of sin around x = 0 is

    sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...

so a degree-N approximation keeps every odd power up to x^N. It is exact at
x = 0 and gets steadily worse as we move away from the expansion point --
which is the whole story of this example.

The polynomial is evaluated with Horner in x^2, in f32 (what we'd ship in
C++) or f64 (to isolate the truncation error from f32 rounding). Error
metrics and spectrum analysis live in ../common (see common/metrics.py).

Usage:
    python taylor_sin.py                 # DEGREE below
    python taylor_sin.py -d 5            # one degree, with coefficients
    python taylor_sin.py --compare       # overlay COMPARE_DEGREES
    python taylor_sin.py --eval f64      # evaluate the polynomial in f64
    python taylor_sin.py --no-show       # just write the PNGs
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, harness, plots, report
from common.metrics import EVAL_DTYPES, N_SAMPLES

# Odd numbers only; even ones round down since sin has no even terms.
DEGREE = 7
COMPARE_DEGREES = [3, 5, 7, 9]

OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Approximation
# ---------------------------------------------------------------------------

def taylor_sin_coeffs(degree):
    """Coefficients c_k so that sin(x) ~ x * sum c_k * (x^2)^k.

    c_k = (-1)^k / (2k+1)!, for 2k+1 <= degree.
    """
    return np.array([(-1) ** k / math.factorial(2 * k + 1) for k in range(degree // 2 + 1)])


def eval_odd_poly(x, coeffs, dtype):
    """Evaluate x * P(x^2) via Horner, entirely in dtype."""
    x = np.asarray(x, dtype=dtype)
    c = coeffs.astype(dtype)
    x2 = x * x
    p = np.full_like(x, c[-1])
    for ck in reversed(c[:-1]):
        p = p * x2 + ck
    return x * p


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
        ax.plot(ctx.x, r.approx, color=color, lw=2, ls="--", label=f"Taylor, {r.label}")
    ax.set_ylim(-2, 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"sin(x) vs. Taylor polynomial, {degree_label(results)}")
    ax.legend(loc="upper left", frameon=False)
    plots.style_axes(ax)
    fig.tight_layout()
    return fig


def fmt_coeff(c, dtype):
    """Print the coefficient as the eval dtype actually stores it. 9 significant
    digits round-trip an f32; 17 round-trip an f64."""
    return f"{float(dtype(c)): .8e}" if dtype == np.float32 else f"{float(c): .16e}"


def print_coeffs(deg, dtype):
    print(f"Coefficients, degree {deg} ({np.dtype(dtype).name}):")
    for k, c in enumerate(taylor_sin_coeffs(deg)):
        print(f"  x^{2 * k + 1:<2d}  {fmt_coeff(c, dtype)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-d", "--degree", type=int, default=DEGREE, help=f"approximation degree (default {DEGREE})")
    mode.add_argument("--compare", action="store_true", help=f"overlay degrees {COMPARE_DEGREES}")
    parser.add_argument("--eval", choices=EVAL_DTYPES, default="f32", help="precision to evaluate the polynomial in")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--emit-cpp", action="store_true", help="print the C++ function for this configuration and exit")
    args = parser.parse_args()

    raw = COMPARE_DEGREES if args.compare else [args.degree]
    degrees = sorted({d if d % 2 else d - 1 for d in raw})
    dtype = EVAL_DTYPES[args.eval]

    if args.emit_cpp:
        if args.compare:
            parser.error("--emit-cpp emits one configuration; drop --compare")
        cppgen.print_emitted(cppgen.emit_taylor(degrees[0])[1])
        return

    ctx = harness.make_context()
    report.print_header(f"Taylor series approximation ({'compare' if args.compare else f'degree {degrees[0]}'})", [
        ("function", "sin(x)"),
        ("range", "[-π, π]"),
        ("eval", f"{args.eval}, Horner in x^2, {N_SAMPLES:,} samples"),
        ("truth", "f64 sin(x)"),
    ])

    results = []
    for deg in degrees:
        coeffs = taylor_sin_coeffs(deg)
        results.append(harness.evaluate(lambda x: eval_odd_poly(x, coeffs, dtype), deg, f"degree {deg}", ctx))

    if args.compare:
        report.print_compare(results, "degree")
    else:
        print_coeffs(degrees[0], dtype)
        report.print_result(results[0])
    report.print_reference(ctx)

    what = f"{degree_label(results)}, evaluated in {args.eval}"
    harness.save_figures(OUT_DIR, f"taylor_sin_{'compare' if args.compare else degrees[0]}_{args.eval}", {
        "": plot_function(ctx, results),
        "error": plots.plot_errors(ctx.x, results, f"Taylor sin(x) error, {what}"),
        "spectrum": plots.plot_spectrum(results, ctx.ref, f"Taylor sin(x) harmonics, {what}"),
        "oscillator": plots.plot_oscillator(results, ctx.ref_osc, f"Taylor sin(x) oscillator, {what}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
