"""tan(pi x) for filter cutoffs: Padé vs. Chebyshev, radians vs. turns.

A bilinear-transform filter (TPT one-pole, SVF, ...) needs the prewarped
cutoff g = tan(pi fc / fs). The input is x = fc / fs in [0, 1/2), and tan has
a pole at Nyquist, x = 1/2. Four ways to approximate it:

  Padé [m/n]      t * Pe(t^2) / Qe(t^2), t = pi x. Rational, so it *has* a
                  pole -- near pi/2, wherever its denominator happens to vanish
  Cheb radians    t * P(t^2) / ((pi/2 - t)(pi/2 + t)): divide the poles out,
                  fit the smooth rest. Like 04: pi/2 is one f32 constant, and
                  t = pi x has already rounded, so near Nyquist the distance
                  to the pole is mostly rounding error
  Cheb split      the same with pi/2 = hi + lo, like 05 and vsl::tan. Fixes
                  the pole for an exact radians input; from fc / fs the
                  rounding is already in t = pi x and the split can't help
  Cheb turns      x * Q(x^2) / ((1/2 - x)(1/2 + x)): the same in x = fc / fs
                  directly. Poles at ±1/2 are exact floats, 1/2 - x is exact
                  near them, and there's no pi multiply. This is vsl::tan_pi

The lesson from sin repeats -- poles, like zeros, belong on exact floats --
but the filter decides how much it matters. A filter given g behaves as if
its cutoff were atan(g) / pi, so the error worth quoting is the cutoff shift
in cents. Near Nyquist tan's slope absorbs relative error in g (cents ~
sinc(2x) * rel), so ULP errors of 10^7 at the pole move the cutoff by
millicents. What *does* show is a low-order fit, and skipping the prewarp.

The tables also give max ULP for exact radians inputs t in (0, pi/2) -- the
case vsl::tan serves -- which is where split and single pi/2 part ways.

Fitting and evaluation live in ../common/tanpoly.py; the truth (exact at the
pole) and the cents metric in ../common/metrics.py.

Usage:
    python tan_pi.py                     # DEGREE / PADE_ORDER below, side by side
    python tan_pi.py -d 11 -m 9 -n 8     # other sizes
    python tan_pi.py --compare           # cost vs. accuracy sweep
    python tan_pi.py --fs 44100          # label cutoffs for another rate
    python tan_pi.py --eval f64          # evaluate in f64
    python tan_pi.py --emit-cpp          # C++ for the Chebyshev forms and Padé
    python tan_pi.py --no-show           # just write the PNGs
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, fit, harness, metrics, plots, report, tanpoly
from common.metrics import EVAL_DTYPES, N_SAMPLES, PI, SAMPLE_RATE

# Numerator degree of the Chebyshev forms (odd); Q gets (degree + 1) / 2
# terms. 9 is vsl::tan_pi. [7/6] is JUCE's FastMathApproximations::tan.
DEGREE = 9
PADE_ORDER = (7, 6)
COMPARE_DEGREES = [3, 5, 7, 9, 11]
COMPARE_PADE = [(3, 2), (5, 4), (7, 6)]  # [9/8] matches [7/6]: f32 rounding, not the fit

AUDIO_TOP = 20_000             # Hz; above this is "near Nyquist"
NEAR_NYQUIST_HZ = [20_000, 22_000, 23_000, 23_900, 23_990, 23_999]  # at 48 kHz; scaled to --fs
N_LAST = 4096                  # every f32 this many steps below 1/2 (and below pi/2)
CENTS_FLOOR = 1e-9             # log-axis floor for exact results

OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Inputs and evaluation
# ---------------------------------------------------------------------------

@dataclass
class Context:
    fs: float
    x: np.ndarray       # f32 fc / fs in (0, 1/2), sorted
    truth: np.ndarray   # f64 tan(pi x), exact at the pole
    t: np.ndarray       # f32 radians in (0, pi/2), for the radians-input column
    truth_t: np.ndarray # f64 tan(t)


@dataclass
class Result:
    name: str
    m: tanpoly.Method
    g: np.ndarray
    ulp: np.ndarray
    cents: np.ndarray
    stats: dict


def cutoff_inputs(fs):
    """f32 cutoffs in (0, 1/2): uniform, log-spaced from 1 Hz, log-spaced in
    distance below Nyquist, and every f32 in the last N_LAST steps below 1/2
    (the step there is 2^-25, ~0.0014 Hz at 48 kHz)."""
    uniform = np.linspace(0, 0.5, N_SAMPLES)
    low = np.geomspace(1 / fs, 0.5, N_SAMPLES // 5)
    high = 0.5 - np.geomspace(2.0**-25, 0.1, N_SAMPLES // 5)
    last = 0.5 - np.arange(1, N_LAST + 1) * 2.0**-25
    x = np.unique(np.concatenate([uniform, low, high, last]).astype(np.float32))
    return x[(x > 0) & (x < 0.5)]


def radians_inputs():
    """f32 radians in (0, pi/2): uniform, plus every f32 in the last N_LAST
    steps below pi/2. f32(pi/2) is above pi/2, so the top input is the one
    below it."""
    top = np.nextafter(np.float32(PI / 2), np.float32(0))
    last = top.view(np.uint32) - np.arange(N_LAST, dtype=np.uint32)
    t = np.unique(np.concatenate([np.linspace(0, float(top), N_SAMPLES).astype(np.float32),
                                  last.view(np.float32)]))
    return t[t > 0]


def make_context(fs):
    x = cutoff_inputs(fs)
    t = radians_inputs()
    return Context(fs=fs, x=x, truth=metrics.tan_pi(x), t=t, truth_t=np.tan(t.astype(np.float64)))


def evaluate(name, m, ctx):
    g = m.f(ctx.x).astype(np.float64)
    ulp = metrics.ulp_error_f32(g, ctx.truth)
    ulp[~np.isfinite(g)] = np.inf
    cents = metrics.cutoff_cents(g, ctx.x)
    audio = ctx.x * ctx.fs <= AUDIO_TOP
    ulp_t = (metrics.ulp_error_f32(m.f_radians(ctx.t).astype(np.float64), ctx.truth_t).max()
             if m.f_radians else np.nan)
    return Result(name, m, g, ulp, cents, {
        "ulp_radians": ulp_t,
        "ulp_audio": ulp[audio].max(),
        "ulp_all": ulp.max(),
        "cents_audio": cents[audio].max(),
        "cents_top": cents[~audio].max(),
    })


def reference(ctx):
    """f32(tan(pi x)), correctly rounded: the floor every f32 method sits on."""
    m = tanpoly.Method(lambda x: metrics.tan_pi(x).astype(np.float32), "—", 0, 0, 0, 0, "½ exact",
                       lambda t: np.tan(t.astype(np.float64)).astype(np.float32))
    return evaluate("f32(tan), rounded", m, ctx)


def run(steps, ctx, dtype):
    """steps: (step, size) pairs -> Results."""
    return [evaluate(tanpoly.STEP_NAMES[s], tanpoly.method(s, size, dtype), ctx) for s, size in steps]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt_cents(c):
    return "∞ (g ≤ 0)" if not np.isfinite(c) else f"{c:.2g}" if c < 1e3 else f"{c:,.0f}"


def fmt_ulp(u):
    return "—" if np.isnan(u) else report.fmt_ulp(u)


def print_table(results, ref, fs, title):
    """pole: where the form puts it. The fc/fs columns feed x = fc / fs (radians
    forms compute pi x first); "radians in" feeds exact f32 radians instead."""
    print(title)
    header = (f"  {'method':<20} {'size':>6} {'coeffs':>6} {'mul':>4} {'add':>4} {'div':>4}  {'pole':<12}"
              f"  {'max ULP':>9} {'max ULP':>9}  {'max cents':>9} {'max cents':>9}  {'max ULP':>9}")
    sub = (f"  {'':<20} {'':>6} {'':>6} {'':>4} {'':>4} {'':>4}  {'':<12}"
           f"  {'≤ 20 kHz':>9} {'to Nyq':>9}  {'≤ 20 kHz':>9} {'> 20 kHz':>9}  {'radians in':>9}")
    print(header)
    print(sub)
    for r in results + [ref]:
        s, m = r.stats, r.m
        ops = (f"{'libm':>16}" if r.name == tanpoly.STEP_NAMES["libm"] else
               f"{'':>16}" if r is ref else f"{m.ncoef:>6} {m.mul:>4} {m.add:>4} {m.div:>4}")
        print(f"  {r.name:<20} {m.size:>6} {ops}  {m.pole:<12}"
              f"  {fmt_ulp(s['ulp_audio']):>9} {fmt_ulp(s['ulp_all']):>9}"
              f"  {fmt_cents(s['cents_audio']):>9} {fmt_cents(s['cents_top']):>9}  {fmt_ulp(s['ulp_radians']):>9}")
    print(f"  (fs = {fs:g} Hz. fc/fs columns: every radians method includes its pi * x multiply."
          f" radians in: exact f32 t in (0, π/2), every f32 in the last {N_LAST:,} below π/2)")
    print()


def print_near_nyquist(results, ref, ctx_fs, dtype):
    """Cutoff error at a few specific cutoffs approaching Nyquist."""
    fcs = [f * ctx_fs / 48_000 for f in NEAR_NYQUIST_HZ]
    x = np.array([fc / ctx_fs for fc in fcs], dtype=np.float32)
    truth = metrics.tan_pi(x)
    print(f"Cutoff error near Nyquist, cents (ULP of g in brackets), {np.dtype(dtype).name}:")
    print(f"  {'fc (Hz)':>9} {'true g':>10}" + "".join(f"  {r.name:>22}" for r in results + [ref]))
    for xi, fc, t in zip(x, fcs, truth):
        row = f"  {fc:>9,.0f} {t:>10.4g}"
        for r in results + [ref]:
            g = r.m.f(np.array([xi])).astype(np.float64)
            c = metrics.cutoff_cents(g, np.array([xi]))[0]
            u = metrics.ulp_error_f32(g, np.array([t]))[0]
            row += f"  {fmt_cents(c):>10} ({report.fmt_ulp(u):>9})"
        print(row)
    print()


def fmt_coeff(c, dtype):
    """Print the coefficient as the eval dtype actually stores it. 9 significant
    digits round-trip an f32; 17 round-trip an f64."""
    return f"{float(dtype(c)): .8e}" if dtype == np.float32 else f"{float(c): .16e}"


def print_coeffs(deg, mn, dtype):
    print(f"Coefficients ({np.dtype(dtype).name}):  tan(πx) ~ x * Q(x^2) / ((1/2 - x)(1/2 + x))")
    for k, c in enumerate(tanpoly.fit_cheb_turns(deg)):
        print(f"  Q x^{2 * k:<2d}  {fmt_coeff(c, dtype)}")
    print("  Q(0) = π/4; poles at x = ±1/2 -- exact in f32")
    print()
    pole = tanpoly.pade_pole(*mn)
    h = float(dtype(PI / 2))
    print("Where the pole lands, as fc / fs (true: 0.5):")
    print(f"  Padé [{mn[0]}/{mn[1]}]  {pole / PI:.10f}  (its denominator's zero, t = {pole:.10f})")
    print(f"  Cheb radians {h / PI:.10f}  (h = {np.dtype(dtype).name}(π/2) = {h:.10f}), reached via a rounded π·x")
    print(f"  Cheb split   0.5000000000  (h_hi + h_lo), but still reached via a rounded π·x")
    print(f"  Cheb turns   0.5000000000  exact")
    print()


def label(r):
    return r.name if r.m.size == "—" else f"{r.name} {r.m.size}"


def log_binned_max(x, y, lo, hi, n_bins=plots.N_BINS):
    """Max of y in n_bins log-spaced bins of x over [lo, hi]; (centers, max)."""
    edges = np.geomspace(lo, hi, n_bins + 1)
    idx = np.digitize(x, edges) - 1
    keep = (idx >= 0) & (idx < n_bins)
    out = np.full(n_bins, np.nan)
    np.fmax.at(out, idx[keep], y[keep])
    centers = np.sqrt(edges[:-1] * edges[1:])
    has = ~np.isnan(out)
    return centers[has], out[has]


def _error_panels(axes, xaxis, lo, hi, results, ref):
    """ULP and cents envelopes against xaxis (Hz, log)."""
    ax_ulp, ax_c = axes
    for ax, key in ((ax_ulp, "ulp"), (ax_c, "cents")):
        ax.plot(*log_binned_max(xaxis, np.maximum(getattr(ref, key), CENTS_FLOOR), lo, hi),
                color=plots.MUTED, lw=1.5, label=ref.name)
        for color, r in zip(plots.SERIES, results):
            y = np.minimum(np.maximum(getattr(r, key), CENTS_FLOOR), 1e12)  # g <= 0 -> drawn at the top
            ax.plot(*log_binned_max(xaxis, y, lo, hi), color=color, lw=1.8, label=label(r))
    plots._label_line(ax_ulp, 0.5, "0.5 ULP (correctly rounded)")
    plots._label_line(ax_c, 1.0, "1 cent")
    ax_ulp.set_yscale("symlog", linthresh=1)
    ax_ulp.set_ylim(bottom=0)
    ax_c.set_yscale("log")
    ax_c.set_ylim(CENTS_FLOOR, 1e4)
    ax_ulp.set_ylabel("error in g (f32 ULPs)")
    ax_c.set_ylabel("cutoff error (cents)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        plots.clean_axes(ax)


def _legend(fig, axes, n):
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               frameon=False, ncol=min(n, 4), fontsize=9)


def plot_errors(ctx, results, ref, title):
    """Error against cutoff frequency, 1 Hz to Nyquist."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    _error_panels(axes, ctx.x * ctx.fs, 1, ctx.fs / 2, results, ref)
    for ax in axes:
        ax.axvline(AUDIO_TOP, color=plots.MUTED, lw=0.8)
    axes[-1].set_xlabel(f"cutoff fc (Hz), fs = {ctx.fs:g}")
    fig.suptitle(title)
    _legend(fig, axes, len(results) + 1)
    return fig


def plot_nyquist(ctx, results, ref, title):
    """Error against distance below Nyquist, log, Nyquist on the right."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    d = (0.5 - ctx.x.astype(np.float64)) * ctx.fs
    lo, hi = 2.0**-25 * ctx.fs, 0.1 * ctx.fs
    _error_panels(axes, d, lo, hi, results, ref)
    for ax in axes:
        ax.invert_xaxis()
        ax.axvline(ctx.fs / 2 - AUDIO_TOP, color=plots.MUTED, lw=0.8)
    axes[-1].set_xlabel("distance below Nyquist, fs/2 − fc (Hz); one f32 step is "
                        f"{2.0**-25 * ctx.fs:.4f} Hz here")
    fig.suptitle(title)
    _legend(fig, axes, len(results) + 1)
    return fig


def last_inputs(n=24):
    """The last n f32 below 1/2, ascending."""
    return (0.5 - np.arange(n, 0, -1) * 2.0**-25).astype(np.float32)


def plot_function(ctx, results, title):
    """g against fc, and the last f32 inputs below Nyquist where the pole is."""
    fig, (ax, ax_z) = plt.subplots(2, 1, figsize=(10, 8.5))
    fc = ctx.x * ctx.fs
    ax.plot(fc, ctx.truth, color=plots.TRUTH, lw=2.5, label="tan(πx)")
    for color, r in zip(plots.SERIES, results):
        ax.plot(fc, np.where(r.g > 0, r.g, np.nan), color=color, lw=1.8, ls="--", label=label(r))
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e8)
    ax.set_xlim(0, ctx.fs / 2)
    ax.set_xlabel(f"cutoff fc (Hz), fs = {ctx.fs:g}")
    ax.set_ylabel("g = tan(π fc / fs)")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    plots.clean_axes(ax)

    x = last_inputs()
    k = np.arange(len(x)) - len(x)
    truth = metrics.tan_pi(x)
    ax_z.axhline(1, color=plots.TRUTH, lw=1.5, label="tan(πx)")
    for color, r in zip(plots.SERIES, results):
        if r.m.pole == "none":
            continue  # no prewarp: ~1e-7 of the truth here, off the scale
        ax_z.plot(k, r.m.f(x).astype(np.float64) / truth, "o--", color=color, lw=1.2, ms=3, label=label(r))
    ax_z.set_ylim(0, 2)
    ax_z.set_xlabel("x, in f32 steps below 1/2 (Nyquist)")
    ax_z.set_ylabel("g / tan(πx)")
    ax_z.set_title("Last f32 inputs below Nyquist: the pole is an exact float only in turns", fontsize=10)
    ax_z.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    plots.clean_axes(ax_z)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-d", "--degree", type=int, default=DEGREE,
                        help=f"numerator degree of the Chebyshev forms, odd (default {DEGREE})")
    parser.add_argument("-m", type=int, default=PADE_ORDER[0], help=f"Padé numerator degree, odd (default {PADE_ORDER[0]})")
    parser.add_argument("-n", type=int, default=PADE_ORDER[1], help=f"Padé denominator degree, even (default {PADE_ORDER[1]})")
    parser.add_argument("--compare", action="store_true",
                        help=f"sweep Padé {COMPARE_PADE} and Chebyshev degrees {COMPARE_DEGREES}")
    parser.add_argument("--fs", type=float, default=SAMPLE_RATE, help=f"sample rate for labels (default {SAMPLE_RATE})")
    parser.add_argument("--eval", choices=EVAL_DTYPES, default="f32", help="precision to evaluate in")
    fit.add_argument(parser)
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--emit-cpp", action="store_true", help="print the C++ for the three forms and exit")
    args = parser.parse_args()
    fit.set_method(args.fit)

    deg = args.degree if args.degree % 2 else args.degree - 1
    mn = (args.m if args.m % 2 else args.m - 1, args.n if args.n % 2 == 0 else args.n - 1)
    if deg < 1 or mn[0] < 1 or mn[1] < 0:
        parser.error("need degree >= 1, m >= 1 and n >= 0")
    dtype = EVAL_DTYPES[args.eval]

    if args.emit_cpp:
        if args.compare:
            parser.error("--emit-cpp emits one configuration; drop --compare")
        code = "\n".join(e[1] for e in (cppgen.emit_tan_turns(deg), cppgen.emit_tan_radians(deg),
                                       cppgen.emit_tan_split(deg), cppgen.emit_tan_pade(*mn)))
        cppgen.print_emitted(code)
        return

    ctx = make_context(args.fs)
    report.print_header(f"tan(πx) for filter cutoffs ({'compare' if args.compare else f'degree {deg}, Padé [{mn[0]}/{mn[1]}]'})", [
        ("function", "g = tan(π fc / fs), x = fc / fs"),
        ("range", f"(0, ½): {len(ctx.x):,} f32 inputs, dense near Nyquist (every f32 in the last {N_LAST})"),
        ("eval", f"{args.eval}; radians forms compute t = π·x in {args.eval} first"),
        ("truth", "f64 tan(πx), via cot(π(½ − x)) near the pole"),
        ("metric", "cutoff error in cents: the filter's cutoff is atan(g) / π"),
    ])

    ref = reference(ctx)
    if args.compare:
        steps = ([("pade", o) for o in COMPARE_PADE] + [("radians", d) for d in COMPARE_DEGREES]
                 + [("split", d) for d in COMPARE_DEGREES] + [("turns", d) for d in COMPARE_DEGREES])
        results = run(steps, ctx, dtype)
        print_table(results, ref, ctx.fs, "Cost vs. accuracy:")
        plotted = [r for r in results if r.name in (tanpoly.STEP_NAMES["pade"], tanpoly.STEP_NAMES["turns"])]
        tag = "compare"
        what = f"Padé vs. Chebyshev turns, evaluated in {args.eval}"
    else:
        print_coeffs(deg, mn, dtype)
        steps = [("prewarp-free", None), ("libm", None), ("pade", mn), ("radians", deg), ("split", deg), ("turns", deg)]
        results = run(steps, ctx, dtype)
        print_table(results, ref, ctx.fs, f"Side by side, degree {deg} and Padé [{mn[0]}/{mn[1]}]:")
        print_near_nyquist(results[1:], ref, ctx.fs, dtype)
        plotted = results
        tag = f"{deg}_{mn[0]}_{mn[1]}"
        what = f"degree {deg}, Padé [{mn[0]}/{mn[1]}], evaluated in {args.eval}"

    harness.save_figures(OUT_DIR, f"tan_pi_{tag}_{args.eval}", {
        "": plot_function(ctx, plotted, f"tan(πx) for filter cutoffs, {what}"),
        "error": plot_errors(ctx, plotted, ref, f"tan(πx) error against cutoff, {what}"),
        "nyquist": plot_nyquist(ctx, plotted, ref, f"tan(πx) error approaching Nyquist, {what}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
