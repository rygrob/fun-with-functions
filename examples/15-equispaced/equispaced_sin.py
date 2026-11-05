"""Why Chebyshev nodes: interpolating sin(x) on [-pi, pi] at equally spaced points.

The obvious way to fit a degree-n polynomial is to pick n + 1 equally spaced
points and pass through sin at each. Chebyshev (03) also interpolates -- the
only difference is *where* the points go: bunched towards the ends,

    t_k = cos(pi (k + 1/2) / (n + 1)),   k = 0..n

For any interpolant the error is

    f(x) - p(x) = f^(n+1)(ξ) / (n+1)!  *  ω(x),   ω(x) = Π (x - x_k)

The first factor doesn't care about the nodes; ω(x) is all the nodes
control. Chebyshev nodes make ω equiripple (it's T_(n+1) scaled down by
2^n); equally spaced nodes leave ω small in the middle and huge at the ends,
so the error piles up at ±pi -- Runge's phenomenon.

sin is entire, so in exact arithmetic the equispaced fit still converges --
the classic divergence needs a function like 1 / (1 + 25 t^2) with poles
near the interval. What bites for sin is the *Lebesgue constant* Λ_n: how
much the interpolant amplifies an error in the sampled values. Chebyshev's
grows like (2/pi) ln n; equispaced grows like 2^n / (e n ln n). Store the
samples in f32 and every fit sees noise of 2^-24; equispaced multiplies it
by Λ_n, so past a modest degree adding terms makes it *worse*.

Both fits use the same barycentric evaluator in f64; only the nodes differ.
This one is instructive only -- no C++, no f32 evaluation, no bench.

Usage:
    python equispaced_sin.py             # DEGREE below
    python equispaced_sin.py -d 7        # another degree
    python equispaced_sin.py --no-show   # just write the PNGs
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import fit, harness, plots, remez, report
from common.metrics import PI

DEGREE = 9
SWEEP_DEGREES = range(1, 61)  # convergence plot
TABLE_DEGREES = [5, 9, 13, 17, 21, 29, 41]
N_GRID = 20_001

OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Nodes and interpolation
# ---------------------------------------------------------------------------

def equispaced_nodes(degree):
    return np.linspace(-PI, PI, degree + 1)


def chebyshev_nodes(degree):
    """First-kind Chebyshev points on [-pi, pi]: the same nodes 03 fits at."""
    k = np.arange(degree + 1)
    return PI * np.cos(PI * (k + 0.5) / (degree + 1))[::-1]


NODES = {"equally spaced": equispaced_nodes, "Chebyshev": chebyshev_nodes}


def lagrange_basis(nodes, x):
    """ℓ_j(x) for every node j, as rows: the polynomial that is 1 at node j and
    0 at the others. Barycentric form, so it's stable for any node set;
    p(x) = Σ y_j ℓ_j(x). Assumes no x lands exactly on a node."""
    diff = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(diff, 1.0)
    w = 1.0 / np.prod(diff, axis=1)
    terms = w[:, None] / (x[None, :] - nodes[:, None])
    return terms / terms.sum(axis=0)


def interpolate(nodes, y, x):
    return y @ lagrange_basis(nodes, x)


def node_polynomial(nodes, x):
    """ω(x) = Π (x - x_k): the only part of the interpolation error the nodes control."""
    return np.prod(x[None, :] - nodes[:, None], axis=0)


def lebesgue(nodes, x):
    """Λ = max Σ |ℓ_j(x)|: worst-case gain from errors in the samples to
    errors in the interpolant."""
    return np.abs(lagrange_basis(nodes, x)).sum(axis=0).max()


def projection(degree, x):
    """The truncated Chebyshev series of sin on [-pi, pi]: the true a_0..a_n,
    not an interpolant, so no nodes. Built as fit.series does it -- a
    high-degree interpolation cut back to degree n. Its error is the tail,
    ~ a_(n+2) T_(n+2), without interpolation's aliased copy of it."""
    full = Chebyshev.interpolate(np.sin, max(fit.PROJECTION_DEGREE, 4 * degree), domain=[-PI, PI])
    return Chebyshev(full.coef[:degree + 1], domain=[-PI, PI])(x)


def minimax_coeffs(degree):
    """Monomial coefficients of the minimax polynomial for absolute error, by
    07's Remez exchange. The error equioscillates, so it's the floor every
    other degree-n fit sits above. sin is odd, so the optimum is odd, of
    degree m = the largest odd number <= degree. Its error alternates at
    m + 3 points, so ask Remez for m + 2 coefficients (degree m + 1, whose
    extra even term comes out 0) to match its n + 1 reference points."""
    odd = degree if degree % 2 else degree - 1
    p, _, _ = remez.remez(np.sin, np.ones_like, -PI, PI, odd + 2)
    p[0::2] = 0.0
    return p[:degree + 1]


def minimax(degree, x):
    return np.polynomial.polynomial.polyval(x, minimax_coeffs(degree))


PROJECTION = "Chebyshev projection"
MINIMAX = "Remez minimax"
EXTRA_FITS = {PROJECTION: projection, MINIMAX: minimax}  # not interpolants: no nodes


def coefficients(degree):
    """Monomial coefficients c_k of every fit (sin(x) ~ Σ c_k x^k), low order
    first. The interpolants solve the Vandermonde system through their nodes:
    fine for printing at these degrees, though it's the ill-conditioned route
    the barycentric evaluator avoids."""
    coeffs = {}
    for name, make in NODES.items():
        nodes = make(degree)
        coeffs[name] = np.polynomial.polynomial.polyfit(nodes, np.sin(nodes), degree)
    full = Chebyshev.interpolate(np.sin, max(fit.PROJECTION_DEGREE, 4 * degree), domain=[-PI, PI])
    coeffs[PROJECTION] = Chebyshev(full.coef[:degree + 1], domain=[-PI, PI]).convert(
        kind=np.polynomial.Polynomial, domain=[-1, 1], window=[-1, 1]).coef
    coeffs[MINIMAX] = minimax_coeffs(degree)
    return coeffs


def max_error(nodes, x, truth, sample_dtype):
    y = np.sin(nodes).astype(sample_dtype).astype(np.float64)
    return np.abs(interpolate(nodes, y, x) - truth).max()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def plot_function(x, truth, degree):
    """sin and both interpolants, nodes marked, with the error underneath."""
    fig, (ax, ax_err) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    ax.plot(x, truth, color=plots.TRUTH, lw=2.5, label="sin(x)")
    for color, (name, make) in zip(plots.SERIES, NODES.items()):
        nodes = make(degree)
        ax.plot(x, interpolate(nodes, np.sin(nodes), x), color=color, lw=2, ls="--",
                label=f"{name}, {degree + 1} points")
        ax.plot(nodes, np.sin(nodes), "o", color=color, ms=6)
        err = np.abs(interpolate(nodes, np.sin(nodes), x) - truth)
        ax_err.semilogy(x, err, color=color, lw=1.5, label=f"{name}  (max {err.max():.1e})")
    for color, (name, approx) in zip(plots.SERIES[len(NODES):], EXTRA_FITS.items()):
        err = np.abs(approx(degree, x) - truth)
        ax_err.semilogy(x, err, color=color, lw=1.5, label=f"{name}  (max {err.max():.1e})")
    ax.set_ylim(-1.2, 1.2)
    ax.set_ylabel("y")
    ax.set_title(f"sin(x) interpolated at {degree + 1} points (degree {degree})")
    ax.legend(loc="upper left", frameon=False)
    plots.style_axes(ax)
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("|error|")
    ax_err.set_title("Absolute error -- 0 at each node, and equally spaced piles it up at the ends", fontsize=10)
    ax_err.legend(loc="upper center", frameon=False)
    plots.style_axes(ax_err)
    fig.tight_layout()
    return fig


def plot_signed_error(x, truth, degree):
    """p(x) - sin(x) on a linear scale, sized to Chebyshev's ripple so the
    equally spaced fit runs off the top and bottom at the ends. The projection
    shows what's left once interpolation's aliasing (the t T_(n+1) envelope)
    is gone; Remez evens out the rest."""
    fig, ax = plt.subplots(figsize=(10, 5))
    peak = 0.0
    for color, (name, make) in zip(plots.SERIES, NODES.items()):
        nodes = make(degree)
        err = interpolate(nodes, np.sin(nodes), x) - truth
        ax.plot(x, err, color=color, lw=2, label=f"{name}  (max {np.abs(err).max():.2e})")
        ax.plot(nodes, np.zeros_like(nodes), "o", color=color, ms=5)
        if name == "Chebyshev":
            peak = np.abs(err).max()
    for color, (name, approx) in zip(plots.SERIES[len(NODES):], EXTRA_FITS.items()):
        err = approx(degree, x) - truth
        ax.plot(x, err, color=color, lw=2, label=f"{name}  (max {np.abs(err).max():.2e})")
    ax.axhline(0, color=plots.MUTED, lw=0.8)
    ax.set_ylim(-2 * peak, 2 * peak)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlabel("x")
    ax.set_ylabel("p(x) − sin(x)")
    ax.set_title(f"Signed error, {degree + 1} points (degree {degree})")
    ax.legend(loc="upper center", frameon=False)
    plots.style_axes(ax)
    fig.tight_layout()
    return fig


def plot_node_polynomial(x, degree):
    """|ω(x)| for both node sets: the shape the error inherits."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for color, (name, make) in zip(plots.SERIES, NODES.items()):
        nodes = make(degree)
        omega = np.abs(node_polynomial(nodes, x))
        ax.semilogy(x, omega, color=color, lw=1.5, label=f"{name}  (max {omega.max():.2g})")
    ax.set_xlabel("x")
    ax.set_ylabel("|ω(x)|")
    ax.set_title(f"Node polynomial |ω(x)| = Π |x − xₖ|, {degree + 1} points -- "
                 "error = f⁽ⁿ⁺¹⁾(ξ) / (n+1)! × ω(x)")
    ax.legend(loc="upper center", frameon=False)
    plots.style_axes(ax)
    fig.tight_layout()
    return fig


def plot_convergence(sweep):
    """Max error vs. degree with samples stored in f64 and in f32."""
    fig, (ax, ax_leb) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    degrees = sweep["degrees"]
    for color, name in zip(plots.SERIES, NODES):
        ax.semilogy(degrees, sweep[name, "f64"], color=color, lw=2, label=f"{name}, f64 samples")
        ax.semilogy(degrees, sweep[name, "f32"], color=color, lw=1.5, ls="--", label=f"{name}, f32 samples")
        ax_leb.semilogy(degrees, sweep[name, "lebesgue"], color=color, lw=2, label=name)
    for y, text in [(2.0**-24, "2⁻²⁴"), (2.0**-53, "2⁻⁵³")]:
        ax.axhline(y, color=plots.MUTED, ls=":", lw=1)
        ax.annotate(text, (degrees[-1], y), ha="right", va="bottom", color=plots.MUTED, fontsize=9)
    ax.set_ylim(1e-17, 1e2)
    ax.set_ylabel("max |error|")
    ax.set_title("Max error vs. degree on [-π, π]")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    plots.clean_axes(ax)
    ax_leb.set_xlabel("degree n  (n + 1 points)")
    ax_leb.set_ylabel("Λₙ")
    ax_leb.set_title("Lebesgue constant: how much sample error the interpolant amplifies", fontsize=10)
    ax_leb.legend(loc="upper left", frameon=False)
    plots.clean_axes(ax_leb)
    fig.tight_layout()
    return fig


def run_sweep(x, truth):
    sweep = {"degrees": np.array(SWEEP_DEGREES)}
    for name, make in NODES.items():
        nodes = [make(d) for d in SWEEP_DEGREES]
        sweep[name, "f64"] = np.array([max_error(n, x, truth, np.float64) for n in nodes])
        sweep[name, "f32"] = np.array([max_error(n, x, truth, np.float32) for n in nodes])
        sweep[name, "lebesgue"] = np.array([lebesgue(n, x) for n in nodes])
    return sweep


def print_table(sweep):
    print("Max |error| (f64 samples / f32 samples) and Lebesgue constant Λₙ:")
    print(f"  {'degree':>6}  {'equally spaced':>22}  {'Λₙ':>8}    {'Chebyshev':>22}  {'Λₙ':>5}")
    for d in TABLE_DEGREES:
        i = d - SWEEP_DEGREES[0]
        cols = []
        for name in NODES:
            cols.append(f"{sweep[name, 'f64'][i]:9.2e} / {sweep[name, 'f32'][i]:9.2e}")
            cols.append(f"{sweep[name, 'lebesgue'][i]:.3g}")
        print(f"  {d:>6}  {cols[0]:>22}  {cols[1]:>8}    {cols[2]:>22}  {cols[3]:>5}")
    print()


def print_coeffs(x, truth, degree):
    """Odd powers only: every fit here is symmetric about 0, so the even
    coefficients are 0 (up to rounding). Taylor's 1/k! for reference."""
    coeffs = coefficients(degree)
    names = list(coeffs)
    print(f"Monomial coefficients, degree {degree}:  sin(x) ~ Σ c_k x^k")
    print(f"  {'':5}" + "".join(f"{n:>22}" for n in names) + f"{'Taylor':>22}")
    for k in range(1, degree + 1, 2):
        taylor = (-1) ** (k // 2) / math.factorial(k)
        print(f"  x^{k:<3d}" + "".join(f"{coeffs[n][k]:>22.13e}" for n in names) + f"{taylor:>22.13e}")
    errs = [np.abs(np.polynomial.polynomial.polyval(x, coeffs[n]) - truth).max() for n in names]
    print(f"  {'max':5}" + "".join(f"{e:>22.3e}" for e in errs))
    print()


def to_chebyshev(c):
    """Monomials in x -> Chebyshev coefficients in t = x / pi."""
    return np.polynomial.chebyshev.poly2cheb(c * PI ** np.arange(len(c)))


def print_chebyshev(degree, n_dropped=2):
    """Every fit in the Chebyshev basis, next to the true a_k (the projection,
    which keeps them exactly), with the first few dropped a_k underneath.

    Then the interpolant through n + 1 = N Chebyshev points: on those points
    T_(2Nj ± k) takes the same values as (-1)^j T_k, so every dropped term folds onto
    a kept one,

        c_k = a_k + Σ_j (-1)^j (a_(2Nj - k) + a_(2Nj + k))

    and c_k - a_k should match that sum."""
    a = Chebyshev.interpolate(lambda t: np.sin(PI * t), max(fit.PROJECTION_DEGREE, 4 * degree)).coef
    coeffs = {n: to_chebyshev(c) for n, c in coefficients(degree).items() if n != PROJECTION}
    names = list(coeffs)
    kept = range(1, degree + 1, 2)
    dropped = [k for k in range(degree + 1, len(a)) if k % 2][:n_dropped]

    print(f"Chebyshev coefficients, degree {degree}:  sin(πt) ~ Σ c_k T_k(t),  t = x/π")
    print(f"  {'':5}{'true a_k (projection)':>22}" + "".join(f"{n:>22}" for n in names))
    for k in kept:
        print(f"  T_{k:<3d}{a[k]:>22.13e}" + "".join(f"{coeffs[n][k]:>22.13e}" for n in names))
    for k in dropped:
        print(f"  T_{k:<3d}{a[k]:>22.13e}" + "".join(f"{'(dropped)':>22}" for n in names))
    print()

    N = degree + 1
    c = coeffs["Chebyshev"]
    print(f"Folding in the Chebyshev interpolant ({N} points): c_k - a_k vs. "
          f"Σ (-1)^j (a_(2Nj-k) + a_(2Nj+k)),  first fold onto T_{kept[-1]} is a_{2 * N - kept[-1]}")
    print(f"  {'':5}{'c_k - a_k':>22}{'predicted':>22}{'first term only':>22}")
    for k in kept:
        folded = sum((-1) ** j * (a[2 * N * j - k] + a[2 * N * j + k])
                     for j in range(1, len(a) // (2 * N)) if 2 * N * j + k < len(a))
        first = -a[2 * N - k]
        print(f"  T_{k:<3d}{c[k] - a[k]:>22.3e}{folded:>22.3e}{first:>22.3e}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-d", "--degree", type=int, default=DEGREE, help=f"polynomial degree (default {DEGREE})")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    args = parser.parse_args()

    # Offset by half a step so no grid point lands exactly on a node.
    x = np.linspace(-PI, PI, N_GRID) + PI / (N_GRID - 1)
    x = x[x <= PI]
    truth = np.sin(x)

    report.print_header("Equally spaced vs. Chebyshev interpolation", [
        ("function", "sin(x)"),
        ("range", "[-π, π]"),
        ("fit", "polynomial through n + 1 points, barycentric Lagrange in f64"),
        ("samples", "sin at the nodes, stored in f64 or rounded to f32"),
        ("truth", f"f64 sin(x), {len(x):,} points"),
    ])
    sweep = run_sweep(x, truth)
    print_table(sweep)
    print_coeffs(x, truth, args.degree)
    print_chebyshev(args.degree)

    harness.save_figures(OUT_DIR, f"equispaced_sin_{args.degree}", {
        "": plot_function(x, truth, args.degree),
        "signed": plot_signed_error(x, truth, args.degree),
        "omega": plot_node_polynomial(x, args.degree),
        "convergence": plot_convergence(sweep),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
