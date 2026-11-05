"""tanh(x) for saturation: where does the asymptote come from?

tanh levels off at ±1, and no polynomial or odd rational can: a polynomial
grows without bound, and an odd rational's numerator outgrows its
denominator by one degree. So every approximation needs the asymptote from
somewhere:

  poly + clamp       clamp the input to ±L, fit a polynomial on [-L, L].
                     The clamp is the asymptote, with a kink at L and an error
                     of at least 1 - tanh(L); L is chosen per degree to balance
                     that against the fit
  Padé + clamp       [7/6] is JUCE's FastMathApproximations::tanh. Excellent
                     near 0, passes 1 at x ~ 4.97 and keeps climbing, so the
                     output needs clamping
  sigmoid            y = s / sqrt(1 + s^2) is exactly tanh when s = sinh x, so
                     approximate s with an odd polynomial: the asymptote, the
                     bound |y| < 1 and monotonicity all come from the algebra.
                     vsl::tanh. The polynomial's error matters less as y
                     saturates (dy/ds = (1 + s^2)^-1.5 ~ sech^3 x), so fit it
                     by Remez with that weight (07's machinery) -- 5.5 dB
                     better than vsl's coefficients, and 14 dB better than
                     sinh's own Taylor series
  exp2-based         1 - 2 / (2^(2x log2 e) + 1), on 10-exp2. Saturates via IEEE
                     inf, but 1 - 2/(e + 1) cancels near 0: an absolute noise
                     floor of ~ulp(1), so quiet signals see -144 dB of junk
                     rather than relative accuracy
  exp2, expm1 form   the same split, keeping the -1 exact: e - 1 =
                     2^n (f Q(f)) + (2^n - 1), so n = 0 has full relative
                     precision. tanh = (e - 1) / (e - 1 + 2), input clamped

The same lesson as sin's zeros: put the structure (asymptotes, bounds,
monotonicity) into the form, and leave the polynomial only the smooth rest.

Saturator metrics, beyond max error: max |y| (overshooting 1 in a feedback
loop can blow up), monotonicity, small-signal gain at 0 (the slope sets the
gain of a ladder filter's feedback), and the error in a driven sine's
output, the saturator's version of SINAD.

Fitting and evaluation live in ../common/tanhpoly.py.

Usage:
    python tanh_sat.py                   # DEGREE / PADE_ORDER below, side by side
    python tanh_sat.py -d 11             # another degree
    python tanh_sat.py --compare         # degree sweep per method
    python tanh_sat.py --eval f64        # evaluate in f64
    python tanh_sat.py --emit-cpp        # C++ for the sigmoid, Padé and exp2 forms
    python tanh_sat.py --no-show         # just write the PNGs
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, fit, harness, plots, report, tanhpoly
from common.metrics import EVAL_DTYPES, N_FFT, N_SAMPLES

# Odd degree of the polynomial in x (clamp, sigmoid's s). 9 is vsl::tanh.
DEGREE = 9
PADE_ORDER = (7, 6)
COMPARE_DEGREES = [5, 7, 9, 11, 13]
COMPARE_PADE = [(5, 4), (7, 6), (9, 8)]

X_RANGE = 20.0                      # error measured on [-X_RANGE, X_RANGE]
SMALL = 2.0**-10                    # "quiet" inputs: |x| below this
DRIVES = [1.0, 4.0]                 # driven-sine amplitudes
EDGE_INPUTS = [5.0, 10.0, 100.0, 600.0, 1e4, 1e30, np.inf]

OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Inputs and evaluation
# ---------------------------------------------------------------------------

@dataclass
class Context:
    x: np.ndarray       # f32 inputs on [-X_RANGE, X_RANGE], sorted
    truth: np.ndarray   # f64 tanh x
    small: np.ndarray   # f32 inputs with 0 < |x| <= SMALL
    sine: np.ndarray    # one cycle of a unit sine, f64, N_FFT points


@dataclass
class Result:
    name: str
    m: tanhpoly.Method
    y: np.ndarray
    err: np.ndarray     # signed absolute error
    stats: dict
    drive_err: dict     # amplitude -> error spectrum (dBc per harmonic)


def make_context():
    xs = np.linspace(-X_RANGE, X_RANGE, 4 * N_SAMPLES)
    small = np.geomspace(1e-8, SMALL, N_SAMPLES // 10)
    x = np.unique(np.concatenate([xs, small, -small]).astype(np.float32))
    small = np.unique(np.concatenate([small, -small]).astype(np.float32))
    return Context(x=x, truth=np.tanh(x.astype(np.float64)), small=small,
                   sine=np.sin(2 * np.pi * np.arange(N_FFT) / N_FFT))


def driven(fn, amp, sine):
    """Output of a saturator driven by one cycle of amp * sin; input rounded
    to f32 as a plug-in's buffer would be."""
    x = (amp * sine).astype(np.float32)
    return x, fn(x).astype(np.float64)


def evaluate(name, m, ctx):
    y = m.f(ctx.x).astype(np.float64)
    err = y - ctx.truth
    worst = int(np.argmax(np.abs(err)))
    ys = m.f(ctx.small).astype(np.float64)
    ts = np.tanh(ctx.small.astype(np.float64))
    x0 = np.float32(2.0**-12)
    gain0 = float(m.f(np.array([x0]))[0]) / np.tanh(float(x0))
    stats = {
        "abs": abs(err[worst]),
        "at": float(ctx.x[worst]),
        "max_y": np.abs(y).max(),
        "drop": max(0.0, -np.diff(y).min()),
        "gain0_db": 20 * np.log10(abs(gain0)),
        "small_rel": np.abs(ys / ts - 1).max(),
    }
    drive_err = {}
    for amp in DRIVES:
        x, yd = driven(m.f, amp, ctx.sine)
        td = np.tanh(x.astype(np.float64))
        e = yd - td
        stats[f"drive_{amp:g}"] = 10 * np.log10(max((e**2).sum(), 1e-300) / (td**2).sum())
        E, T = np.abs(np.fft.rfft(e)), np.abs(np.fft.rfft(td))
        drive_err[amp] = (20 * np.log10(np.maximum(E, 1e-300) / T[1]), 20 * np.log10(np.maximum(T, 1e-300) / T[1]))
    return Result(name, m, y, err, stats, drive_err)


def run(steps, ctx, dtype):
    return [evaluate(tanhpoly.STEP_NAMES[s], tanhpoly.method(s, size, dtype), ctx) for s, size in steps]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def db(e):
    return 20 * np.log10(max(e, 1e-300))


def print_table(results, title):
    print(title)
    drives = "".join(f" {'A=' + format(a, 'g'):>7}" for a in DRIVES)
    print(f"  {'method':<20} {'size':>11} {'coef':>4} {'mul':>4} {'add':>4} {'div':>3} {'sqrt':>4}"
          f"  {'max abs':>9} {'dB':>7} {'at x':>7}  {'max |y|':>10} {'max drop':>8} {'gain at 0':>10}"
          f"  {'quiet rel':>9}  {'driven sine err, dB':>{8 * len(DRIVES)}}")
    print(f"  {'':<20} {'':>11} {'':>4} {'':>4} {'':>4} {'':>3} {'':>4}"
          f"  {'':>9} {'':>7} {'':>7}  {'':>10} {'':>8} {'(dB)':>10}  {'(dB)':>9} {drives}")
    for r in results:
        s, m = r.stats, r.m
        print(f"  {r.name:<20} {m.size:>11} {m.ncoef:>4} {m.mul:>4} {m.add:>4} {m.div:>3} {m.sqrt:>4}"
              f"  {s['abs']:>9.2e} {db(s['abs']):>7.1f} {s['at']:>7.3f}  {s['max_y']:>10.8f} {s['drop']:>8.1e}"
              f" {s['gain0_db']:>+10.6f}  {db(s['small_rel']):>9.1f} "
              + "".join(f" {s[f'drive_{a:g}']:>7.1f}" for a in DRIVES))
    print(f"  (x in [-{X_RANGE:g}, {X_RANGE:g}]; max drop is the largest decrease between neighbours (6e-8 is 1 ulp near 1); gain at 0 is"
          f" y/tanh at x = 2^-12;\n   quiet rel is max relative error for |x| <= 2^-10; driven sine is error"
          " power / output power for A sin)")
    print()


def fmt_coeff(c, dtype):
    """Print the coefficient as the eval dtype actually stores it. 9 significant
    digits round-trip an f32; 17 round-trip an f64."""
    return f"{float(dtype(c)): .8e}" if dtype == np.float32 else f"{float(c): .16e}"


def print_coeffs(deg, dtype):
    r, E, history = tanhpoly.fit_sigmoid(deg, full=True)
    print(f"Sigmoid, Remez degree {deg} ({np.dtype(dtype).name}):  s = x * R(x^2), y = s / sqrt(1 + s^2),"
          f" weight x sech^3 x on [0, {tanhpoly.SIGMOID_RANGE:g}]")
    print(f"  {'':8} {'Remez':>16} {'vsl':>16} {'Taylor (sinh)':>16}")
    taylor = tanhpoly.sigmoid_taylor(deg)
    vsl = tanhpoly.VSL_SIGMOID if deg == 9 else [np.nan] * len(r)
    for k, (a, b, c) in enumerate(zip(r, vsl, taylor)):
        print(f"  R x^{2 * k:<3d} {fmt_coeff(a, dtype):>16} {fmt_coeff(b, dtype):>16} {fmt_coeff(c, dtype):>16}")
    print(f"  levelled error E = {abs(E):.3e} ({db(abs(E)):.1f} dB) after {len(history)} exchanges")
    print()
    print(f"Clamp point per degree (minimizes max(fit error, 1 - tanh L)): "
          + ", ".join(f"{d}: {tanhpoly.best_clamp_limit(d):.2f}" for d in COMPARE_DEGREES))
    pe, qe = tanhpoly.pade_tanh(*PADE_ORDER)
    x = np.linspace(0, 10, 100001)
    y = tanhpoly.eval_pade(x, pe, qe, np.float64, clamp=False)
    print(f"Padé {PADE_ORDER[0]}/{PADE_ORDER[1]} unclamped: reaches 1 at x = {x[np.argmax(y >= 1)]:.3f}, "
          f"{y[-1]:.3f} at x = 10")
    print()


def print_edges(results, dtype):
    x = np.array(EDGE_INPUTS, dtype=dtype)
    print(f"Large inputs ({np.dtype(dtype).name}): a blown-up signal should still saturate to 1")
    print(f"  {'x':>8}" + "".join(f"  {r.name[:16]:>16}" for r in results))
    for xi in x:
        print(f"  {float(xi):>8g}" + "".join(f"  {float(r.m.f(np.array([xi]))[0]):>16.9g}" for r in results))
    print("  sigmoid: s * s overflows past |x| ~ 500 (f32), so y = s * (1 / inf) = 0 -- clamp x first (2 ops)")
    print("  Padé:    input clamped (best_pade_clamp); clamping the output instead passes nan through:")
    print("           past ~1e7 P and Q both overflow, inf / inf = nan (the bench's pade_tanh_7_6)")
    print()


def label(r):
    return f"{r.name} {r.m.size}"


def plot_function(ctx, results, title):
    fig, (ax, ax_e) = plt.subplots(2, 1, figsize=(10, 8.5))
    sel = np.abs(ctx.x) <= 6
    x = ctx.x[sel]
    ax.plot(x, ctx.truth[sel], color=plots.TRUTH, lw=2.5, label="tanh x")
    pe, qe = tanhpoly.pade_tanh(*PADE_ORDER)
    ax.plot(x, tanhpoly.eval_pade(x, pe, qe, np.float64, clamp=False), color=plots.MUTED, lw=1.2, ls=":",
            label=f"Padé [{PADE_ORDER[0]}/{PADE_ORDER[1]}], no clamp")
    for color, r in zip(plots.SERIES, results):
        ax.plot(x, r.y[sel], color=color, lw=1.6, ls="--", label=label(r))
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlim(-6, 6)
    ax.set_xlabel("x")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    plots.clean_axes(ax)

    pos = (ctx.x > 0) & (ctx.x <= 12)
    for color, r in zip(plots.SERIES, results):
        ax_e.plot(*plots.binned_max(ctx.x[pos], np.abs(r.err[pos])), color=color, lw=1.6, label=label(r))
    plots._label_line(ax_e, 2.0**-24, "f32 rounding near 1 (2⁻²⁴)")
    ax_e.set_yscale("log")
    ax_e.set_ylim(1e-9, 1)
    ax_e.set_xlim(0, 12)
    ax_e.set_xlabel("x")
    ax_e.set_ylabel("absolute error")
    plots.clean_axes(ax_e)
    fig.tight_layout()
    return fig


def plot_harmonics(results, amp, title):
    """Odd harmonics of tanh(A sin) and each method's error at those
    harmonics, in dB relative to the output's fundamental."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    h = np.arange(1, 64, 2)
    _, true_dbc = results[0].drive_err[amp]
    ax.plot(h, true_dbc[h], "o-", color=plots.TRUTH, lw=1.5, ms=4, label=f"tanh({amp:g} sin): harmonics")
    for color, r in zip(plots.SERIES, results):
        err_dbc, _ = r.drive_err[amp]
        ax.plot(h, err_dbc[h], "o--", color=color, lw=1, ms=3, label=f"{label(r)}: error")
    ax.set_xlabel("harmonic")
    ax.set_ylabel("level (dB re fundamental)")
    ax.set_ylim(-220, 0)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    plots.clean_axes(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-d", "--degree", type=int, default=DEGREE, help=f"odd polynomial degree (default {DEGREE})")
    mode.add_argument("--compare", action="store_true", help=f"sweep degrees {COMPARE_DEGREES}")
    parser.add_argument("--eval", choices=EVAL_DTYPES, default="f32", help="precision to evaluate in")
    fit.add_argument(parser)
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--emit-cpp", action="store_true", help="print the C++ for this configuration and exit")
    args = parser.parse_args()
    fit.set_method(args.fit)
    deg = args.degree if args.degree % 2 else args.degree - 1
    if deg < 3:
        parser.error("degree must be at least 3")
    dtype = EVAL_DTYPES[args.eval]

    if args.emit_cpp:
        if args.compare:
            parser.error("--emit-cpp emits one configuration; drop --compare")
        code = "\n".join([cppgen.bit_helpers(), cppgen.emit_exp2_full(tanhpoly.EXP2_DEGREE)[1]] + [e[1] for e in (
            cppgen.emit_tanh_sigmoid(deg), cppgen.emit_tanh_pade(*PADE_ORDER), cppgen.emit_tanh_clamp(deg),
            cppgen.emit_tanh_exp2(), cppgen.emit_tanh_expm1())])
        cppgen.print_emitted(code)
        print("// (also needs <algorithm> and <bit>)")
        return

    ctx = make_context()
    report.print_header(f"tanh x for saturation ({'compare' if args.compare else f'degree {deg}'})", [
        ("function", "tanh x"),
        ("range", f"{len(ctx.x):,} f32 inputs on [-{X_RANGE:g}, {X_RANGE:g}], plus log-spaced |x| down to 1e-8"),
        ("eval", f"{args.eval}"),
        ("truth", "f64 tanh x"),
        ("drive", f"one cycle of A sin, {N_FFT:,} points, A in {DRIVES}"),
    ])

    if args.compare:
        steps = ([("clamp", d) for d in COMPARE_DEGREES] + [("pade", o) for o in COMPARE_PADE]
                 + [("sigmoid", d) for d in COMPARE_DEGREES])
        results = run(steps, ctx, dtype)
        print_table(results, "Degree sweep:")
        plotted = [r for r in results if r.m.size.split(",")[0] == "9" or r.m.size == "[7/6]"]
        tag, what = "compare", f"degree sweep, evaluated in {args.eval}"
    else:
        print_coeffs(deg, dtype)
        steps = [("clamp", deg), ("pade", PADE_ORDER), ("sigmoid-taylor", deg), ("sigmoid-vsl", None),
                 ("sigmoid", deg), ("exp2", None), ("expm1", None)]
        results = run(steps, ctx, dtype)
        print_table(results, f"Side by side, degree {deg} and Padé [{PADE_ORDER[0]}/{PADE_ORDER[1]}]:")
        print_edges(results, dtype)
        plotted = results
        tag, what = str(deg), f"degree {deg}, evaluated in {args.eval}"

    amp = DRIVES[-1]
    harness.save_figures(OUT_DIR, f"tanh_sat_{tag}_{args.eval}", {
        "": plot_function(ctx, plotted, f"tanh x, {what}"),
        "harmonics": plot_harmonics(plotted, amp, f"tanh({amp:g} sin): harmonics and the error at each, {what}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
