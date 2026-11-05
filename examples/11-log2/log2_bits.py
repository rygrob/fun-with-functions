"""log2(x) by reading the exponent bits and approximating log2 of the
significand: exp2's split, run backwards.

A positive float is x = 2^e * m, with e an integer field and m in [1, 2).
So log2(x) = e + log2(m): the integer part is free, and only log2(m) on one
octave needs a polynomial. Audio uses it for gain to dB (20 log10 x =
6.02 log2 x), metering, and frequency to pitch (12 log2(f / 440)).

  bits            Mitchell: the bit pattern read as a number is
                  2^23 (e + 127) + 2^23 (m - 1), so bits / 2^23 - 127 =
                  e + (m - 1): log2(m) ~ m - 1, a chord. 103 cents of error,
                  52 with a tuned offset, for one multiply and add
  mantissa        log2(m) = u * P(u), u = m - 1 on [0, 1): the zero at m = 1
                  is built in, so x slightly *above* 1 is fine. But x slightly
                  *below* 1 has e = -1 and m ~ 2, and -1 + 0.9999 cancels:
                  10^7 ULP there
  fold            if m >= c, halve m and add 1 to e, so m lands in [c/2, c)
                  around 1 and x near 1 always has e = 0. c = 1.5 is
                  vsl::log2 (one compare against a short constant); c = sqrt 2
                  centres the range in log terms and fits slightly better
  atanh           fold at sqrt 2, s = (m - 1)/(m + 1) in [-0.172, 0.172],
                  log2(m) = s * Q(s^2): only odd terms on a small interval,
                  so half the coefficients -- paid for with a division

The reduction is exact (reading bits, and m - 1 by Sterbenz), so after the
fold the error repeats every octave. Error is shown as ULP (relative: log2 is
0 at x = 1), and in the audio units an absolute error in log2 becomes:
1200 * err cents of pitch, 6.02 * err dB of level -- over x in
[2^-20, 2^10], about -120 to +60 dB.

Fitting and evaluation live in ../common/log2poly.py, bit fields in
../common/bits.py.

Usage:
    python log2_bits.py                  # DEGREE below, side by side
    python log2_bits.py -d 7             # another degree
    python log2_bits.py --compare        # degree sweep per method
    python log2_bits.py --eval f64       # evaluate in f64
    python log2_bits.py --emit-cpp       # C++ for the polynomial methods
    python log2_bits.py --no-show        # just write the PNGs
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, fit, harness, log2poly, metrics, plots, report
from common.metrics import EVAL_DTYPES, N_SAMPLES

# Degree of u * P(u). 7 is the slides'; 9 is vsl::log2. The atanh form uses
# ATANH_DEGREE (odd).
DEGREE = 7
ATANH_DEGREE = 7
COMPARE_DEGREES = [5, 7, 9, 11]
COMPARE_ATANH_DEGREES = [3, 5, 7, 9]

AUDIO_RANGE = (-20, 10)          # log2 x: -120 dB .. +60 dB, or 20 octaves of pitch
PLOT_RANGE = (0.25, 4.0)
N_NEAR_ONE = 1 << 14             # every f32 this many steps either side of 1

OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Inputs and evaluation
# ---------------------------------------------------------------------------

@dataclass
class Context:
    x: np.ndarray       # f32 inputs, sorted
    truth: np.ndarray   # f64 log2 x
    audio: np.ndarray   # mask: x in the audio range


@dataclass
class Result:
    name: str
    m: log2poly.Method
    y: np.ndarray
    ulp: np.ndarray
    err: np.ndarray     # signed absolute error
    stats: dict


def make_context():
    """Log-spaced over PLOT_RANGE, every f32 within N_NEAR_ONE steps of 1, and
    random over the whole normal range."""
    rng = np.random.default_rng(2)
    one = np.float32(1)
    below = one - np.arange(1, N_NEAR_ONE + 1) * np.float32(2.0**-24)
    above = one + np.arange(0, N_NEAR_ONE + 1) * np.float32(2.0**-23)
    x = np.concatenate([np.geomspace(*PLOT_RANGE, 2 * N_SAMPLES), below, above,
                        np.exp2(rng.uniform(-125, 127, N_SAMPLES))])
    x = np.unique(x.astype(np.float32))
    truth = np.log2(x.astype(np.float64))
    return Context(x=x, truth=truth, audio=(truth >= AUDIO_RANGE[0]) & (truth <= AUDIO_RANGE[1]))


def evaluate(name, m, ctx):
    y = m.f(ctx.x).astype(np.float64)
    err = y - ctx.truth
    ulp = metrics.ulp_error_f32(y, ctx.truth)
    worst = int(np.argmax(ulp))
    abs_audio = np.abs(err[ctx.audio]).max()
    return Result(name, m, y, ulp, err, {
        "ulp": ulp[worst],
        "at": float(ctx.x[worst]),
        "abs": abs_audio,
        "cents": 1200 * abs_audio,
        "db": 20 * np.log10(2) * abs_audio,
    })


def run(steps, ctx, dtype):
    return [evaluate(log2poly.STEP_NAMES[s], log2poly.method(s, d, dtype), ctx) for s, d in steps]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(results, title):
    print(title)
    header = (f"  {'method':<20} {'degree':>6} {'coeffs':>6} {'mul':>4} {'add':>4} {'div':>4}"
              f"  {'max ULP':>9} {'at x':>12}  {'max abs':>9}  {'cents':>9}  {'dB':>9}")
    print(header)
    for r in results:
        s, m = r.stats, r.m
        print(f"  {r.name:<20} {m.degree:>6} {m.ncoef:>6} {m.mul:>4} {m.add:>4} {m.div:>4}"
              f"  {report.fmt_ulp(s['ulp']):>9} {s['at']:>12.9g}  {s['abs']:>9.2e}"
              f"  {s['cents']:>9.2e}  {s['db']:>9.2e}")
    print(f"  (ULP over all normal x; abs, cents and dB over log2 x in [{AUDIO_RANGE[0]}, {AUDIO_RANGE[1]}],"
          " where f32 rounding of the result alone is ~5e-7)")
    print()


def fmt_coeff(c, dtype):
    """Print the coefficient as the eval dtype actually stores it. 9 significant
    digits round-trip an f32; 17 round-trip an f64."""
    return f"{float(dtype(c)): .8e}" if dtype == np.float32 else f"{float(c): .16e}"


def print_coeffs(deg, dtype):
    for step in ("fold-1.5", "fold-sqrt2"):
        fold = log2poly.FOLDS[step]
        lo, hi = log2poly.fold_range(fold)
        print(f"Coefficients, {log2poly.STEP_NAMES[step]} ({np.dtype(dtype).name}):  "
              f"log2(1 + u) ~ u * P(u), u in [{lo:.4f}, {hi:.4f})")
        for k, c in enumerate(log2poly.fit_mantissa(deg, fold)):
            print(f"  P u^{k:<2d}  {fmt_coeff(c, dtype)}")
        print()
    print(f"Coefficients, atanh degree {ATANH_DEGREE}:  log2(m) ~ s * Q(s^2), s = (m - 1)/(m + 1), "
          f"|s| <= {log2poly.s_max():.4f}")
    for k, c in enumerate(log2poly.fit_atanh(ATANH_DEGREE)):
        print(f"  Q s^{2 * k:<2d}  {fmt_coeff(c, dtype)}")
    print()
    o = log2poly.bits_offset()
    print(f"Mitchell offset: subtract {o:.6f} (= add {-o * 2**23:,.0f} to the integer bits)")
    print()


def print_edges(deg, dtype):
    """Where reading the exponent field stops working."""
    p = log2poly.fit_mantissa(deg, 1.5)
    fi = np.finfo(dtype)
    x = np.array([0.0, float(fi.smallest_subnormal), 1e-40, float(fi.tiny), 1.0, float(fi.max), np.inf], dtype=dtype)
    with np.errstate(divide="ignore"):
        truth = np.log2(x.astype(np.float64))
    cols = [
        ("true log2 x", truth),
        ("fold 1.5 (vsl)", log2poly.eval_mantissa(x, p, 1.5, dtype)),
        ("full range", log2poly.eval_full_range(x, p, 1.5, dtype)),
    ]
    print(f"Edges, degree {deg} ({np.dtype(dtype).name}):")
    print("  " + f"{'x':>14}" + "".join(f"  {name:>15}" for name, _ in cols))
    for i, xi in enumerate(x):
        print("  " + f"{float(xi):>14.6g}" + "".join(f"  {float(v[i]):>15.7g}" for _, v in cols))
    print("  vsl:  subnormals read e = -127 and a garbage m; 0 gives -127; inf reads as 2^128 -> 128")
    print("  full: scale subnormals by 2^23 first (select + mul), and select -inf for 0, inf for inf")
    print()


def label(r):
    return r.name if r.m.degree == "—" else f"{r.name}, {r.m.degree}"


def plot_function(ctx, results, title):
    """log2(x) and Mitchell's chords, error in cents underneath."""
    fig, (ax, ax_e) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
    sel = (ctx.x >= 0.125) & (ctx.x <= 8)
    x = ctx.x[sel]
    ax.plot(x, ctx.truth[sel], color=plots.TRUTH, lw=2.5, label="log2 x")
    for color, r in zip(plots.SERIES, results):
        ax.plot(x, r.y[sel], color=color, lw=1.8, ls="--", label=label(r))
        ax_e.plot(x, 1200 * r.err[sel], color=color, lw=1.5, label=label(r))
    for k in range(-3, 4):
        for a in (ax, ax_e):
            a.axvline(2.0**k, color=plots.MUTED, lw=0.5, ls=":")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax_e.axhline(0, color=plots.MUTED, lw=0.8)
    ax_e.set_ylabel("error (cents)")
    ax_e.set_xlabel("x  (dotted: powers of two, where the exponent field steps)")
    for a in (ax, ax_e):
        a.set_xscale("log", base=2)
        a.set_xlim(0.125, 8)
        plots.clean_axes(a)
    fig.tight_layout()
    return fig


def plot_errors(ctx, results, title):
    """ULP over a few octaves, and every f32 near 1 where cancellation shows."""
    fig, (ax, ax_z) = plt.subplots(2, 1, figsize=(10, 8.5))
    sel = (ctx.x >= PLOT_RANGE[0]) & (ctx.x <= PLOT_RANGE[1])
    x = ctx.x[sel]
    for color, r in zip(plots.SERIES, results):
        ax.plot(*plots.binned_max(np.log2(x), r.ulp[sel]), color=color, lw=1.8, label=label(r))
    ax.set_xlabel("log2 x")
    ax.set_title(title)

    near = np.abs(ctx.x.astype(np.float64) - 1) <= 2.0**-10
    xn = ctx.x[near].astype(np.float64) - 1
    for color, r in zip(plots.SERIES, results):
        ax_z.plot(xn, r.ulp[near], ".", color=color, ms=2, alpha=0.5, rasterized=True)
        ax_z.plot(*plots.binned_max(xn, r.ulp[near], 200), color=color, lw=1.5)
    ax_z.set_xlabel("x − 1  (every f32 within 2⁻¹⁰ of 1)")
    ax_z.set_title("Near x = 1: below 1 the unfolded mantissa computes −1 + 0.999…", fontsize=10)

    for a in (ax, ax_z):
        plots._label_line(a, 0.5, "0.5 ULP")
        a.set_yscale("symlog", linthresh=1)
        a.set_ylim(bottom=0)
        a.set_ylabel("error (f32 ULPs)")
        plots.clean_axes(a)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-d", "--degree", type=int, default=DEGREE, help=f"degree of u * P(u) (default {DEGREE})")
    mode.add_argument("--compare", action="store_true", help=f"sweep degrees {COMPARE_DEGREES}")
    parser.add_argument("--eval", choices=EVAL_DTYPES, default="f32", help="precision to evaluate in")
    fit.add_argument(parser)
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--emit-cpp", action="store_true", help="print the C++ for this configuration and exit")
    args = parser.parse_args()
    fit.set_method(args.fit)
    if args.degree < 2:
        parser.error("degree must be at least 2")
    dtype = EVAL_DTYPES[args.eval]

    if args.emit_cpp:
        if args.compare:
            parser.error("--emit-cpp emits one configuration; drop --compare")
        code = "\n".join([cppgen.bit_helpers()] + [e[1] for e in (
            cppgen.emit_log2(args.degree, None), cppgen.emit_log2(args.degree, 1.5),
            cppgen.emit_log2(args.degree, np.sqrt(2)), cppgen.emit_log2_atanh(ATANH_DEGREE),
            cppgen.emit_log2_bits(True))])
        cppgen.print_emitted(code)
        print("// (also needs <bit>)")
        return

    ctx = make_context()
    report.print_header(f"log2 x from exponent bits + polynomial ({'compare' if args.compare else f'degree {args.degree}'})", [
        ("function", "log2 x = e + log2 m, e read from the exponent field"),
        ("range", f"{len(ctx.x):,} f32 inputs: log-spaced on [{PLOT_RANGE[0]:g}, {PLOT_RANGE[1]:g}], "
                  f"every f32 within {N_NEAR_ONE:,} steps of 1, random over all normal x"),
        ("eval", f"{args.eval}"),
        ("truth", "f64 log2 x"),
    ])

    if args.compare:
        steps = ([(s, d) for s in ("mantissa", "fold-1.5", "fold-sqrt2") for d in COMPARE_DEGREES]
                 + [("atanh", d) for d in COMPARE_ATANH_DEGREES])
        results = run(steps, ctx, dtype)
        print_table(results, "Degree sweep:")
        plotted = [r for r in results if r.m.degree == "7" and r.name != log2poly.STEP_NAMES["atanh"]
                   or r.name == log2poly.STEP_NAMES["atanh"] and r.m.degree in ("5", "7")]
        tag, what = "compare", f"degree sweep, evaluated in {args.eval}"
    else:
        print_coeffs(args.degree, dtype)
        steps = [("bits", None), ("bits-tuned", None), ("mantissa", args.degree), ("fold-1.5", args.degree),
                 ("fold-sqrt2", args.degree), ("atanh", ATANH_DEGREE)]
        results = run(steps, ctx, dtype)
        print_table(results, f"Side by side, degree {args.degree} (atanh {ATANH_DEGREE}):")
        print_edges(args.degree, dtype)
        plotted = results
        tag, what = str(args.degree), f"degree {args.degree}, evaluated in {args.eval}"

    harness.save_figures(OUT_DIR, f"log2_bits_{tag}_{args.eval}", {
        "": plot_function(ctx, [r for r in plotted if r.m.ncoef == 0] or plotted[:2],
                          "log2 x: the exponent field gives the integer, the bits a chord between"),
        "error": plot_errors(ctx, [r for r in plotted if r.m.ncoef > 0], f"log2 x error, {what}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
