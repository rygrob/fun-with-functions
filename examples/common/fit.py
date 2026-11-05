"""Chebyshev fits on an interval, returned as monomial coefficients.

Two ways to choose the coefficients of a degree-n Chebyshev series, picked
for every fit in the repo by set_method (each example's --fit flag):

  interpolation   match f exactly at the n + 1 Chebyshev nodes. n + 1
                  evaluations, no integrals -- but the dropped high-order
                  terms alias onto the kept ones, so the error bound is about
                  2 x sum |a_k| over the tail, and at the interval's ends
                  (where every T_k = +-1) the aliased terms add up
  projection      the truncated Chebyshev series: the true a_0..a_n (the
                  orthogonal projection under the weight 1 / sqrt(1 - t^2)).
                  Error ~ the tail itself, sum |a_k|: up to 2x smaller, and
                  2x smaller exactly where relative error peaks. Computed as
                  a high-degree interpolation truncated to degree n -- the
                  discrete orthogonality of the nodes makes that the
                  projection up to the (negligible) aliasing at degree N

Both are near-minimax in max error; neither is optimal for a particular
metric (that's Remez, 07). The default is projection, as on the slides:
interpolating at n + 1 nodes is what the talk argues against.

The sin and tan fits (sinpoly, tanpoly) map their own intervals and call
series(); the exp2 / log2 / tanh / exp / dB fits use chebyshev() below.
"""

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

METHODS = ("interpolation", "projection")
_method = "projection"
PROJECTION_DEGREE = 128     # interpolation degree truncated for a projection


def set_method(method):
    global _method
    if method not in METHODS:
        raise ValueError(f"fit method must be one of {METHODS}")
    _method = method


def get_method():
    return _method


def add_argument(parser):
    """The --fit flag every Chebyshev example takes."""
    parser.add_argument("--fit", choices=METHODS, default="projection",
                        help="how Chebyshev coefficients are chosen (default projection)")


def series(fn, degree, domain=(-1.0, 1.0)):
    """Chebyshev series of fn on domain, degree `degree`, by the current method."""
    if _method == "interpolation":
        return Chebyshev.interpolate(fn, degree, domain=list(domain))
    full = Chebyshev.interpolate(fn, max(PROJECTION_DEGREE, 4 * degree), domain=list(domain))
    return Chebyshev(full.coef[:degree + 1].copy(), domain=list(domain))


def dct_coeffs(fn, n):
    """Chebyshev coefficients a_0..a_(n-1) of fn on [-1, 1], interpolating at
    n nodes, written out as the DCT-II it is: the nodes are equally spaced
    angles, so this is a cosine transform of the samples. Same numbers as
    Chebyshev.interpolate(fn, n - 1); here to show what that does."""
    theta = (np.arange(n) + 0.5) * np.pi / n        # equally spaced angles
    samples = fn(np.cos(theta))                     # f at the Chebyshev nodes
    k = np.arange(n)[:, None]
    a = 2 / n * np.cos(k * theta) @ samples         # DCT-II
    a[0] /= 2
    return a


def chebyshev(fn, a, b, degree):
    """Monomial coefficients c_k (low order first) of the degree-`degree`
    Chebyshev fit of fn on [a, b] (interpolant or projection, see
    set_method) -- near-minimax in absolute error.
    Converting with the identity domain and window gives powers of x itself."""
    cheb = series(fn, degree, (a, b))
    return cheb.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1]).coef


def minimax_endpoints(fn, a, b, degree):
    """Monomial coefficients of a degree-`degree` polynomial that matches fn
    *exactly* at a and b, and is minimax in relative error in between:

        P(x) = L(x) + (x - a)(x - b) Q(x)

    L is the chord through (a, fn(a)) and (b, fn(b)), so P(a) and P(b) are
    those values whatever Q is (and in floating point too: (x - a) or
    (x - b) is exactly 0 there). Q is a Remez fit (07's) of
    (fn - L) / ((x - a)(x - b)), weighted by |(x - a)(x - b)| / |fn| so the
    error it minimizes is P's relative error. The quotient is 0/0 at the
    ends; it's smooth, so it's sampled a hair inside them.

    Pinning the ends buys continuity wherever two reduced intervals meet,
    and exact values there (2^0 = 1). Below degree 2 there is no Q: P is
    the chord.
    """
    from .remez import remez
    fa, fb = float(fn(np.array([a]))[0]), float(fn(np.array([b]))[0])
    chord = Polynomial([fa - (fb - fa) / (b - a) * a, (fb - fa) / (b - a)])
    ends = Polynomial([a * b, -(a + b), 1.0])
    if degree < 2:
        return chord.coef
    eps = 1e-6 * (b - a)
    inside = lambda x: np.clip(x, a + eps, b - eps)
    quotient = lambda x: (fn(inside(x)) - chord(inside(x))) / ends(inside(x))
    weight = lambda x: np.abs(ends(x)) / np.abs(fn(inside(x))) + 1e-12
    q, _, _ = remez(quotient, weight, a, b, degree - 1)
    return (chord + ends * Polynomial(q)).coef
