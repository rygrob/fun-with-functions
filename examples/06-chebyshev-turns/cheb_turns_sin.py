"""Chebyshev approximation of sin(2 pi x), x in turns on [0, 1].

05 needed two f32 constants for pi because the zeros of sin(x) sit at
irrational points. Measure phase in turns instead -- which is what an
oscillator's phase accumulator holds anyway -- and the zeros move to
0, 1/2 and 1, all exact floats. Nothing to split, and no 2*pi multiply to
get from the accumulator to the function's input.

With u = x - 1/2 in [-1/2, 1/2], sin(2 pi x) = -sin(2 pi u), and

    sin(2 pi u) = u (u^2 - 1/4) * g(u),   g(u) = sin(2 pi u) / (u (u^2 - 1/4))

g is smooth and even; fit it with Chebyshev in t = 2u. Since
u^2 - 1/4 = (u - 1/2)(u + 1/2) = (x - 1) x, the factors come straight from x:

    sin(2 pi x) ~ -u * x * (x - 1) * P(u^2)

x is exact, and x - 1 is exact for x >= 1/2 (Sterbenz), so the zeros at 0
and 1 are exact; the one at 1/2 comes from u = x - 1/2, also exact there.
This is vsl::sin_2pi (degree 15).

Fitting and evaluation live in ../common/sinpoly.py; error metrics and
spectrum analysis in ../common/metrics.py.

Usage:
    python cheb_turns_sin.py                 # DEGREE below
    python cheb_turns_sin.py -d 15           # one degree, with coefficients
    python cheb_turns_sin.py --compare       # overlay COMPARE_DEGREES
    python cheb_turns_sin.py --eval f64      # evaluate in f64
    python cheb_turns_sin.py --no-show       # just write the PNGs
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, fit, harness, plots, report, sinpoly
from common.metrics import EVAL_DTYPES, N_SAMPLES

# Total degree in u; odd, >= 3. P gets (degree - 1) / 2 terms.
DEGREE = 13
COMPARE_DEGREES = [9, 11, 13, 15]

STEP = "turns"
DOMAIN = "turns"
OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def degree_label(results):
    degs = [r.key for r in results]
    return f"degree {degs[0]}" if len(degs) == 1 else "degrees " + ", ".join(map(str, degs))


def near_one_inputs(n_ulps=24):
    """Every f32 from n_ulps below 1 up to 1, the top of the range."""
    xs = [np.float32(1)]
    for _ in range(n_ulps):
        xs.append(np.nextafter(xs[-1], np.float32(0)))
    return np.array(xs[::-1], dtype=np.float32)


def plot_function(ctx, results, dtype):
    """sin(2 pi x) vs. the approximation, and the last few f32 inputs below 1
    -- the same close-up as 04/05, where the zero now lands exactly."""
    fig, (ax, ax_z) = plt.subplots(2, 1, figsize=(10, 8.5))

    ax.plot(ctx.x, ctx.truth, color=plots.TRUTH, lw=2.5, label="sin(2πx)")
    for color, r in zip(plots.SERIES, results):
        ax.plot(ctx.x, r.approx, color=color, lw=2, ls="--", label=f"Chebyshev turns, {r.label}")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x (turns)")
    ax.set_ylabel("y")
    ax.set_title(f"sin(2πx) vs. factored Chebyshev polynomial in turns, {degree_label(results)}")
    ax.legend(loc="upper right", frameon=False)
    plots.style_axes(ax, DOMAIN)

    x = near_one_inputs()
    k = np.arange(len(x)) - (len(x) - 1)
    deg = results[-1].key
    f = sinpoly.method(STEP, deg, dtype).f
    ax_z.plot(k, np.sin(2 * np.pi * x.astype(np.float64)) * 1e7, "o-", color=plots.TRUTH, lw=1.5, ms=4,
              label="sin(2πx)")
    ax_z.plot(k, f(x).astype(np.float64) * 1e7, "o--", color=plots.SERIES[len(results) - 1], lw=1.5, ms=4,
              label=f"turns, degree {deg}")
    ax_z.axhline(0, color=plots.MUTED, lw=0.8)
    ax_z.set_xlabel("x, in f32 steps below 1")
    ax_z.set_ylabel("y  (× 10⁻⁷)")
    ax_z.set_title("Last f32 inputs below 1 turn: the zero is an exact float", fontsize=10)
    ax_z.legend(loc="upper left", frameon=False, fontsize=9)
    plots.clean_axes(ax_z)
    fig.tight_layout()
    return fig


def fmt_coeff(c, dtype):
    """Print the coefficient as the eval dtype actually stores it. 9 significant
    digits round-trip an f32; 17 round-trip an f64."""
    return f"{float(dtype(c)): .8e}" if dtype == np.float32 else f"{float(c): .16e}"


def print_coeffs(deg, dtype):
    print(f"Coefficients ({np.dtype(dtype).name}):  sin(2πx) ~ -u * x * (x - 1) * P(u^2),  u = x - 1/2")
    for k, c in enumerate(sinpoly.fit_cheb_turns(deg)):
        print(f"  P u^{2 * k:<2d}  {fmt_coeff(c, dtype)}")
    print("  zeros at x = 0, 1/2, 1 -- exact in f32, no split constants")
    print()
    m = sinpoly.method(STEP, deg, dtype)
    print(f"Cost: {m.mul} mul, {m.add} add  (radians versions also need x = 2π·phase − π first)")
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
        cppgen.print_emitted(cppgen.emit_cheb_turns(degrees[0])[1])
        return

    ctx = harness.make_context(DOMAIN)
    report.print_header(f"Chebyshev approximation in turns ({'compare' if args.compare else f'degree {degrees[0]}'})", [
        ("function", "sin(2πx)"),
        ("range", "[0, 1] turns"),
        ("fit", f"Chebyshev {fit.get_method()} of sin(2πu) / (u (u² − ¼)), converted to monomials"),
        ("eval", f"{args.eval}, -u * x * (x - 1) * P(u^2), u = x - 1/2, {N_SAMPLES:,} samples"),
        ("truth", "f64 sin(2πx)"),
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
        report.print_progression(rows, f"Progression, degree {deg} (radians rows on [-π, π], turns on [0, 1]):")
    report.print_reference(ctx)

    what = f"{degree_label(results)}, evaluated in {args.eval}"
    harness.save_figures(OUT_DIR, f"cheb_turns_sin_{'compare' if args.compare else degrees[0]}_{args.eval}", {
        "": plot_function(ctx, results, dtype),
        "error": plots.plot_errors(ctx.x, results, f"Chebyshev sin(2πx) error in turns, {what}", DOMAIN),
        "spectrum": plots.plot_spectrum(results, ctx.ref, f"Chebyshev sin(2πx) harmonics in turns, {what}"),
        "oscillator": plots.plot_oscillator(results, ctx.ref_osc, f"Chebyshev sin(2πx) oscillator in turns, {what}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
