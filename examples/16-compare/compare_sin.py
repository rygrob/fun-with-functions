"""Every way to fit sin(x) on [-pi, pi] at one degree, signed error side by side.

The fitting arc in one picture -- each method is a choice of *where* the
approximation agrees with sin:

  Taylor            all n + 1 conditions at x = 0 (derivatives)        00
  Padé [m/n]        the same Taylor terms, spent on P/Q, m + n = degree 02
  equally spaced    the value at n + 1 evenly spaced points            15
  Chebyshev points  the value at n + 1 Chebyshev points (first kind)   03
  Chebyshev proj.   the truncated Chebyshev series: no aliasing        15
  linear LUT        not a fit: the smallest table (01's linear lookup)
                    whose max error matches the projection's            01

Everything is f64, so this is fit error only. The top panel is sized to the
worst method; the bottom one to the Chebyshev points fit, so the others run off the edges.
Instructive only -- no C++, no f32 evaluation, no bench.

Usage:
    python compare_sin.py                # DEGREE below
    python compare_sin.py -d 13          # another (odd) degree
    python compare_sin.py --no-show      # just write the PNGs
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import fit, harness, plots, report, sinpoly
from common.metrics import PI

DEGREE = 9
N_GRID = 20_001
N_GRID_LUT = 1_000_001  # ~1000 points per table step, so the size search sees each chord's peak

OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Methods (all f64)
# ---------------------------------------------------------------------------

def pade_order(degree):
    """[m/n] with m + n = degree, m odd, n even, n as close to m as allowed:
    9 -> [5/4], 13 -> [7/6] (02's JUCE form)."""
    n = degree // 2 - (degree // 2) % 2
    return degree - n, n


def equispaced_coeffs(degree):
    """x * P(x^2) through degree + 1 equally spaced points. Symmetric points and
    an odd sin make the even coefficients 0; drop them."""
    nodes = np.linspace(-PI, PI, degree + 1)
    return np.polynomial.polynomial.polyfit(nodes, np.sin(nodes), degree)[1::2]


def cheb_coeffs(degree, method):
    """03's fit by `method`: at the Chebyshev points (interpolation), or the
    true a_1..a_n (projection). fit's method is global, so put it back
    afterwards."""
    before = fit.get_method()
    fit.set_method(method)
    try:
        return sinpoly.fit_cheb_vanilla(degree)
    finally:
        fit.set_method(before)


def methods(degree):
    """label -> f(x), in talk order."""
    m, n = pade_order(degree)
    pe, qe = sinpoly.pade_sin(m, n)
    fits = {
        "Taylor": sinpoly.taylor_coeffs(degree),
        f"Padé [{m}/{n}]": None,
        "equally spaced": equispaced_coeffs(degree),
        "Chebyshev points": cheb_coeffs(degree, "interpolation"),
        "Chebyshev projection": cheb_coeffs(degree, "projection"),
    }
    out = {}
    for label, c in fits.items():
        if c is None:
            out[label] = lambda x: sinpoly.eval_pade(x, pe, qe, np.float64)
        else:
            out[label] = lambda x, c=c: sinpoly.eval_odd(x, c, np.float64)
    return out


def lut_size_to_match(target):
    """Smallest linear table with max error <= target. Chord error is
    step^2 / 8 * max|sin''| = step^2 / 8 with step = 2 pi / N, so start
    there and step up until the measured error agrees."""
    x = np.linspace(-PI, PI, N_GRID_LUT)
    truth = np.sin(x)
    size = math.ceil(2 * PI / math.sqrt(8 * target))
    while np.abs(lut(size)(x) - truth).max() > target:
        size += 1
    return size


def lut(size):
    table = sinpoly.make_table(size, np.float64)
    return lambda x: sinpoly.eval_lut(x, table, "linear", np.float64)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def plot_signed_error(x, errors, degree):
    fig, (ax_all, ax_zoom) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for color, (label, err) in zip(plots.SERIES, errors.items()):
        for ax in (ax_all, ax_zoom):
            # The table's error is one scallop per entry -- a solid band at
            # this scale, so draw it faint and underneath.
            band = label.startswith("linear LUT")
            ax.plot(x, err, color=color, lw=1 if band else 2, alpha=0.35 if band else 1,
                    zorder=1 if band else 2, label=f"{label}  (max {np.abs(err).max():.2e})")
    worst = max(np.abs(e).max() for e in errors.values())
    zoom = np.abs(errors["Chebyshev points"]).max()
    for ax, lim, title in [(ax_all, 1.1 * worst, f"Signed error, degree {degree} -- full scale"),
                           (ax_zoom, 2 * zoom, "Zoomed to the Chebyshev points fit's ripple")]:
        ax.axhline(0, color=plots.MUTED, lw=0.8)
        ax.set_ylim(-lim, lim)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("p(x) − sin(x)")
        ax.set_title(title)
        plots.style_axes(ax)
    ax_all.legend(loc="upper center", frameon=False)
    ax_zoom.set_xlabel("x")
    fig.tight_layout()
    return fig


def print_table(x, errors):
    """Max error, and where it is: the edge for the one-point methods, spread
    out for the many-point ones."""
    mid = np.abs(x) <= PI / 2
    print(f"  {'':28}{'max |error|':>14}{'|x| <= π/2':>14}{'at x = ±π':>14}")
    for label, err in errors.items():
        a = np.abs(err)
        print(f"  {label:28}{a.max():>14.3e}{a[mid].max():>14.3e}{max(a[0], a[-1]):>14.3e}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-d", "--degree", type=int, default=DEGREE, help=f"odd polynomial degree (default {DEGREE})")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    args = parser.parse_args()
    degree = args.degree if args.degree % 2 else args.degree - 1

    x = np.linspace(-PI, PI, N_GRID)
    truth = np.sin(x)
    errors = {label: f(x) - truth for label, f in methods(degree).items()}
    size = lut_size_to_match(np.abs(errors["Chebyshev projection"]).max())
    errors[f"linear LUT, {size} entries"] = lut(size)(x) - truth

    m, n = pade_order(degree)
    report.print_header(f"Fitting methods compared (degree {degree})", [
        ("function", "sin(x)"),
        ("range", "[-π, π]"),
        ("methods", f"Taylor, Padé [{m}/{n}], equally spaced, Chebyshev points, Chebyshev projection"),
        ("table", f"linear LUT sized to the projection: {size} entries = {4 * (size + 1):,} bytes in f32 "
                  f"(with guard), vs {(degree + 1) // 2} coefficients"),
        ("eval", f"f64, {N_GRID:,} points"),
        ("truth", "f64 sin(x)"),
    ])
    print_table(x, errors)

    harness.save_figures(OUT_DIR, f"compare_sin_{degree}", {
        "": plot_signed_error(x, errors, degree),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
