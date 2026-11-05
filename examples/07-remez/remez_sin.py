"""Remez minimax approximation of sin(2 pi x), x in turns on [0, 1].

Same form as 06,

    sin(2 pi x) ~ -u * x * (x - 1) * P(u^2),   u = x - 1/2

but P comes from a hand-rolled Remez exchange (../common/remez.py) instead of
Chebyshev interpolation. Chebyshev is near-minimax for *absolute* error in
the quotient g; what we care about is *relative* error of sin, and since the
factors u, x, x - 1 are exact, that's exactly the relative error of g. So:

    minimize  max |(g(v) - P(v)) / g(v)|   over v = u^2 in [0, 1/4]

g varies by about 2x over the range, so weighting matters. The optimum
equioscillates: the error touches ±E at n + 1 alternating points.

Where it helps depends on the error budget. At degree 11 the fit error
dominates, so a better fit shows up in f32. At degree 13 the fit error is
already well below f32 rounding, so it barely does -- 08 goes after that.

Usage:
    python remez_sin.py                      # DEGREE below
    python remez_sin.py -d 13                # one degree
    python remez_sin.py --weight absolute    # minimax absolute error instead
    python remez_sin.py --compare            # overlay COMPARE_DEGREES
    python remez_sin.py --eval f64           # evaluate in f64
    python remez_sin.py --emit-cpp           # print the C++ function
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, fit, harness, plots, report, sinpoly
from common.metrics import EVAL_DTYPES, N_SAMPLES, sin_turns, ulp_f32

DEGREE = 11
COMPARE_DEGREES = [9, 11, 13, 15]
WEIGHTS = ["relative", "absolute"]

DOMAIN = "turns"
OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def degree_label(results):
    degs = [r.key for r in results]
    return f"degree {degs[0]}" if len(degs) == 1 else "degrees " + ", ".join(map(str, degs))


def fits(degree):
    """(label, color, P) for the three fits compared at one degree."""
    return [
        ("Chebyshev (06)", plots.MUTED, sinpoly.fit_cheb_turns(degree)),
        ("Remez, absolute", plots.SERIES[1], sinpoly.fit_remez_turns(degree, "absolute")),
        ("Remez, relative", plots.SERIES[0], sinpoly.fit_remez_turns(degree, "relative")),
    ]


def plot_equioscillation(ctx, degree, dtype):
    """Signed error, linear axis: the f64 fit (Remez relative touches ±E at
    n + 1 alternating points; Chebyshev bulges at the ends) and the f32
    result (how much of that shape survives rounding)."""
    x = ctx.x[(ctx.x > 0) & (ctx.truth != 0)]
    truth = sin_turns(x)
    ulp = ulp_f32(truth)
    _, E, _ = sinpoly.fit_remez_turns(degree, "relative", full=True)
    fit_rows, f32_rows = [], []
    for label, color, p in fits(degree):
        fit_rows.append((label, color, (sinpoly.eval_turns(x, p, np.float64) - truth) / truth))
        f32_rows.append((label, color, (sinpoly.eval_turns(x, p, dtype).astype(np.float64) - truth) / ulp))
    return plots.plot_signed_error(
        x, fit_rows, f32_rows,
        f"Signed error, degree {degree}: fit (f64) vs. shipped ({np.dtype(dtype).name}, no FMA)",
        DOMAIN, levels=E)


def fmt_coeff(c, dtype):
    """Print the coefficient as the eval dtype actually stores it. 9 significant
    digits round-trip an f32; 17 round-trip an f64."""
    return f"{float(dtype(c)): .8e}" if dtype == np.float32 else f"{float(c): .16e}"


def print_fit(degree, weight, dtype):
    p, E, history = sinpoly.fit_remez_turns(degree, weight, full=True)
    print(f"Remez exchange ({weight} error), degree {degree}, {len(p)} coefficients in v = u^2:")
    for i, (level, peak) in enumerate(history):
        print(f"  iter {i}:  |E| = {level:.6e}   max |err| = {peak:.6e}")
    unit = "relative error" if weight == "relative" else "absolute error in g"
    print(f"  levelled {unit}: {abs(E):.3e}"
          + (f"  (~{abs(E) / 2.0**-24:.2f} ULP before rounding)" if weight == "relative" else ""))
    print()
    print(f"Coefficients ({np.dtype(dtype).name}):  sin(2πx) ~ -u * x * (x - 1) * P(u^2),  u = x - 1/2")
    for k, c in enumerate(p):
        print(f"  P u^{2 * k:<2d}  {fmt_coeff(c, dtype)}")
    print()


def print_fit_comparison(degree, ctx):
    """Fit error (f64) vs. shipped error (f32), for each fit."""
    print(f"Fit vs. rounding, degree {degree} (max ULP over {N_SAMPLES:,} samples):")
    print(f"  {'':<18} {'f64 eval':>10} {'f32 eval':>10}")
    for label, _, p in fits(degree):
        row = [harness.evaluate(lambda x: sinpoly.eval_turns(x, p, dt), degree, "", ctx).stats["max_ulp"]
               for dt in (np.float64, np.float32)]
        print(f"  {label:<18} {row[0]:>10.3f} {row[1]:>10.3f}")
    print("  f64 is the fit alone; f32 adds evaluation rounding.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-d", "--degree", type=int, default=DEGREE, help=f"total polynomial degree (default {DEGREE})")
    mode.add_argument("--compare", action="store_true", help=f"overlay degrees {COMPARE_DEGREES}")
    parser.add_argument("--weight", choices=WEIGHTS, default="relative", help="error to minimax (default relative)")
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
        cppgen.print_emitted(cppgen.emit_remez_turns(degrees[0], args.weight)[1])
        return

    ctx = harness.make_context(DOMAIN)
    report.print_header(f"Remez approximation in turns ({'compare' if args.compare else f'degree {degrees[0]}'})", [
        ("function", "sin(2πx)"),
        ("range", "[0, 1] turns"),
        ("fit", f"Remez minimax of g(u) = sin(2πu) / (u (u² − ¼)) in u², {args.weight} error"),
        ("eval", f"{args.eval}, -u * x * (x - 1) * P(u^2), u = x - 1/2, {N_SAMPLES:,} samples"),
        ("truth", "f64 sin(2πx)"),
    ])

    results = []
    for deg in degrees:
        p = sinpoly.fit_remez_turns(deg, args.weight)
        results.append(harness.evaluate(lambda x: sinpoly.eval_turns(x, p, dtype), deg, f"degree {deg}", ctx))

    if args.compare:
        report.print_compare(results, "degree")
    else:
        deg = degrees[0]
        print_fit(deg, args.weight, dtype)
        report.print_result(results[0])
        print_fit_comparison(deg, ctx)
        rows = sinpoly.progression("remez", deg, dtype, ctx)
        report.print_progression(rows, f"Progression, degree {deg} (radians rows on [-π, π], turns on [0, 1]):")
    report.print_reference(ctx)

    what = f"{degree_label(results)}, {args.weight}, evaluated in {args.eval}"
    figs = {f"equioscillation_{d}": plot_equioscillation(ctx, d, np.float32) for d in degrees}
    figs.update({
        "error": plots.plot_errors(ctx.x, results, f"Remez sin(2πx) error in turns, {what}", DOMAIN),
        "spectrum": plots.plot_spectrum(results, ctx.ref, f"Remez sin(2πx) harmonics in turns, {what}"),
        "oscillator": plots.plot_oscillator(results, ctx.ref_osc, f"Remez sin(2πx) oscillator in turns, {what}"),
    })
    tag = f"{'compare' if args.compare else degrees[0]}_{args.weight}_{args.eval}"
    harness.save_figures(OUT_DIR, f"remez_sin_{tag}", figs, show=not args.no_show)


if __name__ == "__main__":
    main()
