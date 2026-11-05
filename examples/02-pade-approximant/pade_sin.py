"""Padé approximant of sin(x) on [-pi, pi].

A Padé approximant [m/n] is a rational function P(x) / Q(x), with P of
degree m and Q of degree n, chosen so its Taylor series matches f's through
order m + n:

    P(x) - f(x) Q(x) = O(x^(m+n+1)),   Q(0) = 1

So it uses the same information as a degree m+n Taylor polynomial, but
spends some of it on a denominator. Rational functions bend in ways
polynomials can't, so [m/n] usually beats Taylor m+n away from the expansion
point -- at the cost of a division. Like Taylor, it is still built entirely
from derivatives at x = 0, so the error still grows toward ±pi.

sin is odd, so P is odd (m odd) and Q is even (n even):

    sin(x) ~ x * Pe(x^2) / Qe(x^2)

[7/6] is the form used by JUCE's dsp::FastMathApproximations::sin.

Coefficients are solved exactly with Fractions. Evaluation is Horner in x^2
for both polynomials, in f32 or f64. Error metrics and spectrum analysis
live in ../common (see common/metrics.py).

Usage:
    python pade_sin.py                   # ORDER below
    python pade_sin.py -m 5 -n 4         # one [m/n], with coefficients
    python pade_sin.py --compare         # overlay COMPARE_ORDERS
    python pade_sin.py --eval f64        # evaluate in f64
    python pade_sin.py --no-show         # just write the PNGs
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, harness, metrics, plots, report
from common.metrics import EVAL_DTYPES, N_SAMPLES
from common.sinpoly import eval_pade, horner, pade_sin, sin_taylor_fractions

# m odd, n even; other values round down since sin is odd.
ORDER = (7, 6)
COMPARE_ORDERS = [(3, 2), (5, 4), (7, 6), (9, 8)]

OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Approximation
# ---------------------------------------------------------------------------

def eval_taylor(x, c, dtype):
    """Same-order Taylor polynomial, for comparison."""
    x = np.asarray(x, dtype=dtype)
    return x * horner(np.array([dtype(ck) for ck in c[1::2]]), x * x)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def name(m, n):
    return f"[{m}/{n}]"


def order_label(results):
    return ", ".join(r.label for r in results)


def plot_function(ctx, results):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ctx.x, ctx.truth, color=plots.TRUTH, lw=2.5, label="sin(x)")
    for color, r in zip(plots.SERIES, results):
        ax.plot(ctx.x, r.approx, color=color, lw=2, ls="--", label=f"Padé {r.label}")
    ax.set_ylim(-2, 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"sin(x) vs. Padé {order_label(results)}")
    ax.legend(loc="upper left", frameon=False)
    plots.style_axes(ax)
    fig.tight_layout()
    return fig


def fmt_coeff(c, dtype):
    """Print the coefficient as the eval dtype actually stores it. 9 significant
    digits round-trip an f32; 17 round-trip an f64."""
    return f"{float(dtype(c)): .8e}" if dtype == np.float32 else f"{float(c): .16e}"


def integer_form(pe, qe):
    """Scale numerator and denominator by a common factor so every coefficient
    is an integer, then divide out their gcd (the JUCE-style constants)."""
    scale = math.lcm(*(c.denominator for c in pe + qe))
    ints = [int(c * scale) for c in pe + qe]
    g = math.gcd(*ints)
    ints = [i // g for i in ints]
    return ints[:len(pe)], ints[len(pe):]


def print_coeffs(m, n, dtype):
    pe, qe = pade_sin(m, n)
    print(f"Coefficients, {name(m, n)} ({np.dtype(dtype).name}):  sin(x) ~ x * P(x^2) / Q(x^2)")
    for k, c in enumerate(pe):
        print(f"  P x^{2 * k:<2d}  {fmt_coeff(c, dtype)}")
    for k, c in enumerate(qe):
        print(f"  Q x^{2 * k:<2d}  {fmt_coeff(c, dtype)}")
    print()
    ip, iq = integer_form(pe, qe)
    print("Integer form (same ratio):")
    print("  P : " + ", ".join(f"{i:,}" for i in ip))
    print("  Q : " + ", ".join(f"{i:,}" for i in iq))
    print()
    # Odd x, even x^2: one mul for x^2, Horner on each, one mul by x, one div.
    muls = 1 + (len(pe) - 1) + (len(qe) - 1) + 1
    adds = (len(pe) - 1) + (len(qe) - 1)
    print(f"Cost: {muls} mul, {adds} add, 1 div")
    print()


def print_denominator(m, n, ctx):
    """A rational approximation is only safe if Q has no zeros on the range."""
    _, qe = pade_sin(m, n)
    q = horner(np.array([float(c) for c in qe]), ctx.x.astype(np.float64) ** 2)
    print(f"Denominator: min Q(x^2) on range = {q.min():.4f}  ({'no poles' if q.min() > 0 else 'POLE ON RANGE'})")
    print()


def print_taylor_comparison(m, n, dtype, ctx, result):
    """Two fair fights: Taylor using the same Taylor information (order m+n),
    and Taylor storing the same number of coefficients (no division)."""
    pe, qe = pade_sin(m, n)
    same_order = m + n
    same_count = 2 * (len(pe) + len(qe)) - 1
    print("vs. Taylor:")
    print(f"  {'':<16} {'coeffs':>6}  {'max abs':>9}  {'SINAD dB':>8}  note")
    print(f"  {'Padé ' + name(m, n):<16} {len(pe) + len(qe):>6}  {result.stats['max_abs']:>9.2e}"
          f"  {result.spec['sinad']:>8.1f}  1 div")
    for deg, note in ((same_order, "same Taylor order"), (same_count, "same coefficient count")):
        t = harness.evaluate(lambda x: eval_taylor(x, sin_taylor_fractions(deg), dtype), deg, "", ctx)
        print(f"  {'Taylor ' + str(deg):<16} {(deg + 1) // 2:>6}  {t.stats['max_abs']:>9.2e}"
              f"  {t.spec['sinad']:>8.1f}  {note}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def to_odd_even(m, n):
    return (m if m % 2 else m - 1), (n if n % 2 == 0 else n - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-m", type=int, default=ORDER[0], help=f"numerator degree, odd (default {ORDER[0]})")
    mode.add_argument("--compare", action="store_true",
                      help="overlay " + ", ".join(name(*o) for o in COMPARE_ORDERS))
    parser.add_argument("-n", type=int, default=ORDER[1], help=f"denominator degree, even (default {ORDER[1]})")
    parser.add_argument("--eval", choices=EVAL_DTYPES, default="f32", help="precision to evaluate in")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--emit-cpp", action="store_true", help="print the C++ function for this configuration and exit")
    args = parser.parse_args()

    orders = COMPARE_ORDERS if args.compare else [to_odd_even(args.m, args.n)]
    if min(m for m, _ in orders) < 1 or min(n for _, n in orders) < 0:
        parser.error("need m >= 1 and n >= 0")
    dtype = EVAL_DTYPES[args.eval]

    if args.emit_cpp:
        if args.compare:
            parser.error("--emit-cpp emits one configuration; drop --compare")
        cppgen.print_emitted(cppgen.emit_pade(*orders[0])[1])
        return

    ctx = harness.make_context()
    report.print_header(f"Padé approximation ({'compare' if args.compare else name(*orders[0])})", [
        ("function", "sin(x)"),
        ("range", "[-π, π]"),
        ("eval", f"{args.eval}, Horner in x^2 for P and Q, {N_SAMPLES:,} samples"),
        ("truth", "f64 sin(x)"),
    ])

    results = []
    for m, n in orders:
        pe, qe = pade_sin(m, n)
        results.append(harness.evaluate(lambda x: eval_pade(x, pe, qe, dtype), name(m, n), name(m, n), ctx))

    if args.compare:
        report.print_compare(results, "[m/n]")
    else:
        m, n = orders[0]
        print_coeffs(m, n, dtype)
        print_denominator(m, n, ctx)
        report.print_result(results[0])
        print_taylor_comparison(m, n, dtype, ctx, results[0])
    report.print_reference(ctx)

    what = f"{order_label(results)}, evaluated in {args.eval}"
    tag = "compare" if args.compare else f"{orders[0][0]}_{orders[0][1]}"
    harness.save_figures(OUT_DIR, f"pade_sin_{tag}_{args.eval}", {
        "": plot_function(ctx, results),
        "error": plots.plot_errors(ctx.x, results, f"Padé sin(x) error, {what}"),
        "spectrum": plots.plot_spectrum(results, ctx.ref, f"Padé sin(x) harmonics, {what}"),
        "oscillator": plots.plot_oscillator(results, ctx.ref_osc, f"Padé sin(x) oscillator, {what}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
