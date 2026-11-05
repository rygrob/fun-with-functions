"""tanh revisited: the sigmoid's square root, its domain, and the clamp.

12-tanh builds the asymptote into the algebra, y = s / sqrt(1 + s^2) with
s = x * R(x^2) ~ sinh x. This example asks what that form does in practice
(a scratch workspace, f32 throughout, simulating the C++):

  the square root   how y is formed from s: s * (1 / sqrt) as the generated
                    C++ and vsl write it, plain s / sqrt, or the hardware
                    reciprocal-sqrt estimate plus 0, 1 or 2 Newton steps.
                    NEON's estimate (FRSQRTE, 8 bits) is modelled from Arm's
                    pseudocode and matches this machine's vrsqrteq_f32 bit for
                    bit; FRSQRTS's step is fused. SSE's rsqrtps (12 bits) is
                    Intel's undocumented table, so it's modelled as the exact
                    value rounded to 12 bits: a stand-in, not Intel
  the domain        s * s overflows once |x| > 555.9 (f32): y = s / inf = 0,
                    and past 76,850 s itself is inf: inf / inf = NaN. Behind
                    a Newton step, inf * 0 = NaN already at 555.9
  drive             moderate (|x| <= 10, +20 dB on a full-scale signal) and
                    high (|x| <= 1000, +60 dB): the high-drive break is the
                    overflow, not the fit
  the clamp         tanh x rounds to 1 in f32 once x > atanh(1 - 2^-25) ~ 9.01:
                    past that no float can tell tanh from 1, so clamping x
                    there (fmin / fmax, two instructions) loses nothing and
                    removes the overflow for every input

Measured in C++ over every float (bench/src/rsqrt.cpp and the scratch
checks, 2026-10-19): the Python here samples, so its maxima are lower bounds,
but the breakpoints are exact (found on the float bit patterns).

Usage:
    python tanh_revisited.py              # degree 9 (sigmoid_tanh_9)
    python tanh_revisited.py -d 11        # another degree
    python tanh_revisited.py --no-show    # just write the PNGs
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import harness, metrics, plots, report, tanhpoly
from common.sinpoly import horner

DEGREE = 9
CLAMP = 9.0                 # fmin / fmax limit for the clamped form
MODERATE, HIGH = 10.0, 1000.0
F32 = np.float32
F32_MAX = float(np.finfo(F32).max)

OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Square roots, as the hardware does them (f32)
# ---------------------------------------------------------------------------

def _recip_sqrt_estimate(a):
    """Arm's RecipSqrtEstimate: a 9-bit a in [128, 512) (the significand
    scaled to [0.25, 1)) to a 9-bit r in [256, 512), r ~ 512 / sqrt(a / 512)."""
    a = np.asarray(a, dtype=np.int64)
    a = np.where(a < 256, a * 2 + 1, (((a >> 1) << 1) + 1) * 2)
    b = np.floor(np.sqrt(2.0**28 / a)).astype(np.int64)
    for _ in range(4):  # the pseudocode's loop: largest b with a (b + 1)^2 < 2^28
        b = np.where(a * (b + 1) ** 2 < 2**28, b + 1, b)
        b = np.where(a * b * b >= 2**28, b - 1, b)
    return (np.maximum(b, 512) + 1) // 2


def frsqrte(a):
    """NEON vrsqrteq_f32 for positive f32 (inf -> 0): the exponent's parity
    picks [0.25, 0.5) or [0.5, 1), eight significand bits index the table,
    and the result keeps eight. Checked against this machine's vrsqrteq_f32
    on every float in [1, 4) and every 997th positive normal."""
    a = np.asarray(a, dtype=F32)
    u = a.view(np.uint32).astype(np.int64)
    exp, frac = (u >> 23) & 0xFF, u & 0x7FFFFF
    scaled = np.where(exp % 2 == 0, 256 + (frac >> 15), 128 + (frac >> 16))
    est = _recip_sqrt_estimate(scaled)
    bits = (((380 - exp) // 2) << 23) | ((est & 0xFF) << 15)
    out = bits.astype(np.uint32).view(F32)
    return np.where(np.isinf(a), F32(0), np.where(np.isnan(a), np.nan, out)).astype(F32)


def rsqrt12(a):
    """A 12-bit reciprocal-sqrt estimate, as SSE's rsqrtps is specified
    (|rel err| <= 1.5 * 2^-12): the exact value rounded to 12 significant
    bits. Intel's table isn't published, so this stands in for it."""
    with np.errstate(divide="ignore", invalid="ignore"):
        e = 1 / np.sqrt(np.asarray(a, dtype=np.float64))
        m, ex = np.frexp(e)
        return np.where(np.isfinite(e) & (e > 0), np.ldexp(np.round(m * 4096) / 4096, ex), e).astype(F32)


def newton(a, e):
    """One step, e (3 - a e^2) / 2 as NEON writes it: a * e rounded (vmulq),
    then FRSQRTS's (3 - (a e) e) / 2 with one rounding, then * e."""
    with np.errstate(invalid="ignore", over="ignore"):
        ae = (a * e).astype(F32)
        step = ((3 - ae.astype(np.float64) * e.astype(np.float64)) / 2).astype(F32)
        return (e * step).astype(F32)


# ---------------------------------------------------------------------------
# Forms: x -> y, all f32
# ---------------------------------------------------------------------------

def sinh_and_a(x, r):
    """s = x R(x^2) and a = s^2 + 1, in f32 (overflow to inf allowed)."""
    x = np.asarray(x, dtype=F32)
    with np.errstate(over="ignore", invalid="ignore"):
        s = x * horner(r.astype(F32), x * x)
        return s, s * s + F32(1)


def forms(r):
    """name -> f(x) for every way of finishing s / sqrt(1 + s^2)."""
    def presented(x):
        s, a = sinh_and_a(x, r)
        with np.errstate(invalid="ignore", divide="ignore"):
            return s * (F32(1) / np.sqrt(a))

    def division(x):
        s, a = sinh_and_a(x, r)
        with np.errstate(invalid="ignore"):
            return s / np.sqrt(a)

    def estimate(steps, est):
        def f(x):
            s, a = sinh_and_a(x, r)
            e = est(a)
            for _ in range(steps):
                e = newton(a, e)
            with np.errstate(invalid="ignore"):
                return (s * e).astype(F32)
        return f

    def clamped(x):
        x = np.fmin(np.fmax(np.asarray(x, dtype=F32), F32(-CLAMP)), F32(CLAMP))
        return division(x)

    return {
        "s * (1 / sqrt)": presented,
        "s / sqrt": division,
        "NEON estimate": estimate(0, frsqrte),
        "NEON + 1 Newton": estimate(1, frsqrte),
        "NEON + 2 Newton": estimate(2, frsqrte),
        "12-bit estimate": estimate(0, rsqrt12),
        "12-bit + 1 Newton": estimate(1, rsqrt12),
        f"s / sqrt, clamp {CLAMP:g}": clamped,
    }


# ---------------------------------------------------------------------------
# Breakpoints, on the float bit patterns
# ---------------------------------------------------------------------------

def f32_range(lo, hi):
    """Every positive f32 bit pattern from lo to hi, as uint32."""
    return np.arange(np.array(lo, F32).view(np.uint32), np.array(hi, F32).view(np.uint32) + 1, dtype=np.uint32)


def first_where(pred, lo, hi):
    """Smallest f32 x in [lo, hi] with pred(x), by bisection on the bits
    (pred must be monotone there); None if pred(hi) is false."""
    a, b = int(np.array(lo, F32).view(np.uint32)), int(np.array(hi, F32).view(np.uint32))
    as_x = lambda u: np.array([u], dtype=np.uint32).view(F32)
    if not pred(as_x(b))[0]:
        return None
    while a < b:
        m = (a + b) // 2
        if pred(as_x(m))[0]:
            b = m
        else:
            a = m + 1
    return float(as_x(a)[0])


def breakpoints(f):
    """Where y stops being near 1 (|y - 1| >= 1/2, or non-finite) and where
    it goes non-finite: both monotone past the overflow, so bisect from
    x = 100 up to the largest float."""
    wrong = first_where(lambda x: ~(np.abs(f(x).astype(np.float64) - 1) < 0.5), 100.0, F32_MAX)
    bad = first_where(lambda x: ~np.isfinite(f(x)), 100.0, F32_MAX)
    return wrong, bad


def tanh_rounds_to_one():
    """Smallest f32 x with f32(tanh x) = 1: tanh x >= 1 - 2^-25."""
    return first_where(lambda x: np.tanh(x.astype(np.float64)).astype(F32) == 1, 1.0, 20.0)


def first_exact_one(f, lo=5.0, hi=20.0):
    """Smallest f32 x in [lo, hi] past which f(x) is exactly 1 for every
    float up to hi (checked on every float in the range)."""
    u = f32_range(lo, hi)
    y = f(u.view(F32))
    not_one = np.flatnonzero(y != 1)
    if len(not_one) and not_one[-1] == len(u) - 1:
        return None
    return float(u.view(F32)[not_one[-1] + 1]) if len(not_one) else lo


# ---------------------------------------------------------------------------
# Errors in a drive range (sampled)
# ---------------------------------------------------------------------------

def inputs(limit):
    """Positive f32 inputs up to limit: dense linear, plus log-spaced down to
    1e-8 so the small-signal (relative) error shows."""
    lin = np.linspace(0, min(limit, 20.0), 2_000_001)[1:]
    log = np.geomspace(1e-8, limit, 1_000_001)
    return np.unique(np.concatenate([lin, log]).astype(F32))


def errors(f, x):
    y = f(x).astype(np.float64)
    t = np.tanh(x.astype(np.float64))
    ok = np.isfinite(y)
    e = metrics.pointwise_errors(y[ok], t[ok])
    normal = x[ok] >= np.finfo(F32).tiny
    return {
        "abs": e["abs"].max(),
        "rel": np.nanmax(e["rel"][normal]),
        "ulp": e["ulp"].max(),
        "max_y": np.abs(y[ok]).max() if ok.any() else np.nan,
        "bad": int((~ok).sum()),
        "at": float(x[ok][np.argmax(e["abs"])]),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def db(e):
    return 20 * np.log10(max(e, 1e-300))


def fmt_x(v):
    return "never" if v is None else f"{v:.6g}"


def print_domain(fs):
    print("Domain (exact, on float bit patterns, x > 100):")
    print(f"  {'form':<22} {'wrong from':>12} {'y there':>9} {'non-finite from':>16}")
    for name, f in fs.items():
        wrong, bad = breakpoints(f)
        there = "" if wrong is None else f"{float(f(np.array([wrong], F32))[0]):g}"
        print(f"  {name:<22} {fmt_x(wrong):>12} {there:>9} {fmt_x(bad):>16}")
    s_limit = np.sqrt(F32_MAX)
    print(f"  (wrong: |y - 1| >= 1/2 or non-finite. s * s overflows once s > sqrt(FLT_MAX) = {s_limit:.4g};"
          f" with s ~ 3.6e-6 x^9 that's x ~ 556)")
    print()


def print_drive(fs, limit, title):
    x = inputs(limit)
    print(f"{title}: {len(x):,} f32 inputs in (0, {limit:g}] (sampled: maxima are lower bounds)")
    print(f"  {'form':<22} {'max abs':>9} {'dB':>7} {'at x':>8}  {'max rel':>9}  {'max ULP':>9}  {'max |y|':>12}  {'non-finite':>10}")
    for name, f in fs.items():
        s = errors(f, x)
        print(f"  {name:<22} {s['abs']:>9.2e} {db(s['abs']):>7.1f} {s['at']:>8.4g}  {s['rel']:>9.2e}"
              f"  {report.fmt_ulp(s['ulp']):>9}  {s['max_y']:>12.9f}  {s['bad']:>10,}")
    print("  (errors over the finite outputs; non-finite counts the rest)")
    print()


def print_saturation(fs, r):
    x1 = tanh_rounds_to_one()
    print("Indistinguishable from 1:")
    print(f"  f32(tanh x) = 1 from x = {x1:.9g} (tanh x >= 1 - 2^-25; atanh(1 - 2^-25) = "
          f"{np.arctanh(1 - 2.0**-25):.6f})")
    print(f"  1 - tanh x at the clamp {CLAMP:g}: {1 - np.tanh(CLAMP):.3e} = {(1 - np.tanh(CLAMP)) / 2.0**-24:.2f} ulp(1-)")
    for name, f in fs.items():
        print(f"  {name:<22} y = 1 exactly for every float in [x, 20] from x = {fmt_x(first_exact_one(f)):>9}"
              f"   y({CLAMP:g}) = {float(f(np.array([CLAMP], F32))[0]):.9f}")
    print(f"  leading coefficient r0 = {r[0]:.10f}: small signals pass at gain {20 * np.log10(r[0]):+.6f} dB,"
          f" relative error {abs(r[0] - 1):.2e} = {abs(r[0] - 1) * 2**24:.0f} ulp near 0")
    print()


def reach_methods(deg):
    """JUCE's Padé [7/6] unclamped, 12-tanh's polynomial with its own clamp
    (a polynomial can't reach far, so it clamps early and pays 1 - tanh L),
    and the sigmoid without and with the clamp at CLAMP."""
    pe, qe = tanhpoly.pade_tanh(7, 6)
    limit = tanhpoly.best_clamp_limit(deg)
    p = tanhpoly.fit_clamp(deg, limit)
    r = tanhpoly.fit_sigmoid(deg)
    div = forms(r)["s / sqrt"]
    return {
        "Padé [7/6] (JUCE)": lambda x: tanhpoly.eval_pade(x, pe, qe, F32, clamp=False),
        f"polynomial {deg}, clamp {limit:.2f}": lambda x: tanhpoly.eval_clamp(x, p, limit, F32),
        f"sigmoid {deg}": div,
        f"sigmoid {deg}, clamp {CLAMP:g}": forms(r)[f"s / sqrt, clamp {CLAMP:g}"],
    }


def print_reach(ms):
    """How far out each unclamped method stays within its own best error:
    clamping at L costs 1 - tanh L, so a method can be clamped once that
    cost falls below its error, if it's still accurate there."""
    x = f32_range(0.01, 12.0)[::16].view(F32)
    t = np.tanh(x.astype(np.float64))
    print(f"Reach (|error| on (0, 12], every 16th float; clamping at L costs 1 - tanh L):")
    print(f"  {'method':<30} {'max |err| on [0, 9]':>20} {'dB':>7}  {'|err| > 1e-4 from':>18}  {'max |y|':>10}")
    for name, f in ms.items():
        with np.errstate(over="ignore", invalid="ignore"):
            e = np.abs(f(x).astype(np.float64) - t)
        in9 = x <= 9
        over = np.flatnonzero(~(e < 1e-4))
        print(f"  {name:<30} {e[in9].max():>20.2e} {db(e[in9].max()):>7.1f}  "
              f"{fmt_x(float(x[over[0]]) if len(over) else None):>18}  {np.nanmax(np.abs(f(x))):>10.6f}")
    for L in (5.0, 7.0, CLAMP):
        print(f"  clamp at {L:g}: costs 1 - tanh {L:g} = {1 - np.tanh(L):.2e} ({db(1 - np.tanh(L)):.0f} dB)")
    print()


def plot_reach(ms, title):
    """Each method's |error| on (0, 12] against 1 - tanh x, the cost of
    clamping at x: a method can be clamped where its error meets that
    line, if it's still accurate there. Padé runs out first."""
    x = f32_range(0.01, 12.0)[::16].view(F32)
    t = np.tanh(x.astype(np.float64))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, 1 - t, color=plots.TRUTH, lw=2.5, ls="--", label="1 − tanh x: the cost of clamping at x")
    for color, (name, f) in zip(plots.SERIES, ms.items()):
        with np.errstate(over="ignore", invalid="ignore"):
            e = np.abs(f(x).astype(np.float64) - t)
        ax.plot(*plots.binned_max(x, e, 600), color=color, lw=1.6, label=name)
    plots._label_line(ax, 2.0**-24, "2⁻²⁴")
    ax.axvline(CLAMP, color=plots.MUTED, ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1)
    ax.set_xlim(0, 12)
    ax.set_xlabel("x")
    ax.set_ylabel("absolute error")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    plots.clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_domain(fs, title):
    """y and |error| against |x| on a log axis out to 1e6: the collapse at
    556 (to 0, or NaN behind a Newton step) and the clamped form holding."""
    x = np.geomspace(1e-2, 1e6, 20001).astype(F32)
    t = np.tanh(x.astype(np.float64))
    fig, (ax, ax_e) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
    for color, (name, f) in zip(plots.SERIES, fs.items()):
        y = f(x).astype(np.float64)
        ax.plot(x, y, color=color, lw=1.6, label=name)
        ax_e.plot(x, np.abs(y - t), color=color, lw=1.2)
    for v, text in [(555.9, "s² overflows"), (76850, "s overflows")]:
        for a in (ax, ax_e):
            a.axvline(v, color=plots.MUTED, ls=":", lw=1)
        ax.annotate(text, (v, 0.5), rotation=90, ha="right", va="center", color=plots.MUTED, fontsize=9)
    ax.plot(x, t, color=plots.TRUTH, lw=2.5, alpha=0.3, label="tanh x")
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.15)
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
    plots.clean_axes(ax)
    plots._label_line(ax_e, 2.0**-24, "2⁻²⁴")
    ax_e.set_yscale("log")
    ax_e.set_ylim(1e-9, 2)
    ax_e.set_xlabel("x (log)")
    ax_e.set_ylabel("absolute error (NaN: gap)")
    plots.clean_axes(ax_e)
    fig.tight_layout()
    return fig


def plot_saturation(fs, title):
    """1 - y near saturation, every float in [4, 12]: where each form
    reaches exactly 1, against 1 - tanh x and the f32 steps below 1."""
    x = f32_range(4.0, 12.0)[::64].view(F32)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, 1 - np.tanh(x.astype(np.float64)), color=plots.TRUTH, lw=2.5, label="1 − tanh x")
    for color, (name, f) in zip(plots.SERIES, fs.items()):
        gap = 1 - f(x).astype(np.float64)
        ax.plot(x, np.where(gap > 0, gap, np.nan), color=color, lw=1.2, label=name)
    plots._label_line(ax, 2.0**-24, "one f32 step below 1 (2⁻²⁴)")
    plots._label_line(ax, 2.0**-25, "rounds to 1 below here (2⁻²⁵)")
    x1 = tanh_rounds_to_one()
    ax.axvline(x1, color=plots.MUTED, ls="--", lw=1)
    ax.annotate(f"f32(tanh x) = 1 from {x1:.3f}", (x1, 1e-4), ha="left", va="center", color=plots.MUTED,
                fontsize=9, xytext=(6, 0), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1e-2)
    ax.set_xlim(4, 12)
    ax.set_xlabel("x")
    ax.set_ylabel("1 − y (gaps: y ≥ 1)")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
    plots.clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_small(fs, title):
    """Relative error for quiet inputs, 1e-8 to 10: the plateau is R's
    leading coefficient minus 1 (the fit minimizes absolute error)."""
    x = np.geomspace(1e-8, 10, 20001).astype(F32)
    t = np.tanh(x.astype(np.float64))
    fig, ax = plt.subplots(figsize=(10, 5))
    for color, (name, f) in zip(plots.SERIES, fs.items()):
        y = f(x).astype(np.float64)
        ax.plot(*plots.binned_max(x, np.abs(y / t - 1), 300), color=color, lw=1.4, label=name)
    plots._label_line(ax, 2.0**-24, "2⁻²⁴")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1e-1)
    ax.set_xlabel("x (log)")
    ax.set_ylabel("relative error")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    plots.clean_axes(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-d", "--degree", type=int, default=DEGREE, help=f"odd degree of s (default {DEGREE})")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    args = parser.parse_args()
    deg = args.degree if args.degree % 2 else args.degree - 1

    r = tanhpoly.fit_sigmoid(deg)
    fs = forms(r)
    report.print_header(f"tanh revisited: sigmoid degree {deg}, f32", [
        ("form", "y = s / sqrt(1 + s^2), s = x R(x^2), R by sech^3-weighted Remez (12-tanh)"),
        ("truth", "f64 tanh x"),
        ("drive", f"moderate |x| <= {MODERATE:g}, high |x| <= {HIGH:g} (tanh is odd: x > 0 shown)"),
    ])
    print_domain(fs)
    print_drive(fs, MODERATE, "Moderate drive")
    print_drive(fs, HIGH, "High drive")
    print_saturation(fs, r)
    ms = reach_methods(deg)
    print_reach(ms)

    what = f"sigmoid degree {deg}, f32"
    harness.save_figures(OUT_DIR, f"tanh_revisited_{deg}", {
        "reach": plot_reach(ms, f"tanh: how far out each fit holds, f32"),
        "domain": plot_domain(fs, f"tanh: the domain, {what}"),
        "saturation": plot_saturation(fs, f"tanh: reaching 1, {what}"),
        "small": plot_small(fs, f"tanh: quiet inputs, {what}"),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
