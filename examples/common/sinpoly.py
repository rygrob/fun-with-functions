"""Polynomial sin(x) approximations on [-pi, pi], shared so later examples can
compare against earlier steps of the progression.

  taylor            x * P(x^2),  P from the Maclaurin series
  cheb vanilla      x * P(x^2),  P from a Chebyshev fit of sin itself
  cheb factored     x * (x - pi)(x + pi) * P(x^2),  P from a Chebyshev fit of
                    sin(x) / (x (x^2 - pi^2)); pi is a single dtype constant
  cheb split        same P, but pi = pi_hi + pi_lo so the zeros at ±pi land
                    on true pi rather than on the nearest float to it
  pade              x * Pe(x^2) / Qe(x^2), [m/n] Padé from exact Taylor terms
  lut               nearest or linear lookup in a one-period table
  cheb turns        sin(2 pi x) for x in [0, 1] turns: -u x (x - 1) * P(u^2)
                    with u = x - 1/2. The zeros 0, 1/2, 1 are exact floats,
                    so nothing needs splitting

All evaluators take inputs in any dtype and do every operation in `dtype`.
After https://mooooo.ooo/chebyshev-sine-approximation/
"""

import math
from fractions import Fraction
from typing import Callable, NamedTuple

import numpy as np
from numpy.polynomial import Chebyshev, chebyshev

from . import fit
from .metrics import PI


# ---------------------------------------------------------------------------
# Fitting (always in f64)
# ---------------------------------------------------------------------------

def taylor_coeffs(degree):
    """c_k so that sin(x) ~ x * sum c_k (x^2)^k; c_k = (-1)^k / (2k+1)!."""
    return np.array([(-1) ** k / math.factorial(2 * k + 1) for k in range(degree // 2 + 1)])


def _cheb_to_x_monomial(a):
    """Chebyshev series in t = x/pi  ->  monomial coefficients in x."""
    mono_t = chebyshev.cheb2poly(a)
    return np.array([c / PI**k for k, c in enumerate(mono_t)])


def cheb_vanilla_series(degree):
    """Chebyshev coefficients a_0..a_degree of sin(pi t) on [-1, 1].

    Chebyshev interpolation or projection (fit.set_method) -- near-minimax
    in absolute error.
    sin is odd, so the even coefficients are exactly zero; force them so.
    """
    a = fit.series(lambda t: np.sin(PI * t), degree).coef
    a[0::2] = 0.0
    return a


def fit_cheb_vanilla(degree):
    """c_k so that sin(x) ~ x * sum c_k (x^2)^k, from a degree-`degree`
    Chebyshev fit of sin itself."""
    return _cheb_to_x_monomial(cheb_vanilla_series(degree))[1::2]


def factored_quotient(x):
    """f(x) = sin(x) / (x (x^2 - pi^2)): smooth and even on [-pi, pi].

    np.sinc(x/pi) = sin(x)/x handles x = 0; the Chebyshev nodes never land
    on ±pi, so the other two removable singularities never get evaluated.
    """
    return np.sinc(x / PI) / (x * x - PI * PI)


def cheb_factored_series(degree):
    """Chebyshev coefficients of f(pi t) for a degree-`degree` sin
    approximation: f gets degree - 3 (even), so P has (degree - 1) / 2 terms."""
    a = fit.series(lambda t: factored_quotient(PI * t), degree - 3).coef
    a[1::2] = 0.0
    return a


def fit_cheb_factored(degree):
    """p_k so that sin(x) ~ x (x^2 - pi^2) * sum p_k (x^2)^k, total degree
    `degree` (odd, >= 3)."""
    return _cheb_to_x_monomial(cheb_factored_series(degree))[0::2]


def turns_quotient(u):
    """g(u) = sin(2 pi u) / (u (u^2 - 1/4)): smooth and even on [-1/2, 1/2].
    sin(2 pi u) / u = 2 pi sinc(2u) handles u = 0; nodes never land on ±1/2."""
    return 2 * PI * np.sinc(2 * u) / (u * u - 0.25)


def cheb_turns_series(degree):
    """Chebyshev coefficients of g(t/2), t in [-1, 1], for a degree-`degree`
    sin(2 pi x) approximation: g gets degree - 3 (even)."""
    a = fit.series(lambda t: turns_quotient(t / 2), degree - 3).coef
    a[1::2] = 0.0
    return a


def fit_cheb_turns(degree):
    """p_k so that sin(2 pi u) ~ u (u^2 - 1/4) * sum p_k (u^2)^k."""
    mono_t = chebyshev.cheb2poly(cheb_turns_series(degree))
    return np.array([c * 2.0**k for k, c in enumerate(mono_t)])[0::2]  # t = 2u


def turns_quotient_v(v):
    """g as a function of v = u^2 on [0, 1/4], finite at both ends.

    Near u = 0 use g's sinc form. Near u = 1/2, with w = u - 1/2,
    sin(2 pi u) = -sin(2 pi w) and u (u^2 - 1/4) = u w (u + 1/2), so
    g = -2 pi sinc(2w) / (u (u + 1/2)) -- regular there (g(1/2) = -4 pi).
    """
    u = np.sqrt(np.asarray(v, dtype=np.float64))
    near_half = u > 0.25
    out = np.empty_like(u)
    out[~near_half] = turns_quotient(u[~near_half])
    uh = u[near_half]
    out[near_half] = -2 * PI * np.sinc(2 * (uh - 0.5)) / (uh * (uh + 0.5))
    return out


def fit_remez_turns(degree, weight="relative", full=False):
    """p_k so that sin(2 pi u) ~ u (u^2 - 1/4) * sum p_k (u^2)^k, minimax on
    v = u^2 in [0, 1/4]. weight="relative" minimizes max relative error of
    g -- which *is* the relative error of sin, since the factors are exact.
    weight="absolute" minimizes max |g - P|, the Remez cousin of Chebyshev.
    full=True also returns the levelled error E and the iteration history.
    """
    from .remez import remez
    n = (degree - 1) // 2
    w = (lambda v: 1 / np.abs(turns_quotient_v(v))) if weight == "relative" else (lambda v: np.ones_like(v))
    p, E, history = remez(turns_quotient_v, w, 0.0, 0.25, n)
    return (p, E, history) if full else p


# ---------------------------------------------------------------------------
# Evaluation (in dtype)
# ---------------------------------------------------------------------------

def horner(c, t):
    r = np.full_like(t, c[-1])
    for ck in reversed(c[:-1]):
        r = r * t + ck
    return r


def eval_odd(x, c, dtype):
    """x * P(x^2)."""
    x = np.asarray(x, dtype=dtype)
    return x * horner(c.astype(dtype), x * x)


def eval_factored(x, p, dtype):
    """(x - pi)(x + pi) * P(x^2) * x, with pi rounded to a single dtype value."""
    x = np.asarray(x, dtype=dtype)
    pi = dtype(PI)
    return (x - pi) * (x + pi) * horner(p.astype(dtype), x * x) * x


def split_pi(dtype):
    """pi_hi = nearest dtype to pi, pi_lo = dtype(pi - pi_hi).

    For f32 the residual is computed in f64, which is plenty. For f64 the
    residual is below f64 resolution, so use sin(fl(pi)) = sin(pi - d) ~ d.
    """
    hi = dtype(PI)
    lo = dtype(PI - float(hi)) if dtype == np.float32 else dtype(math.sin(math.pi))
    return hi, lo


def eval_split(x, p, dtype):
    """((x - pi_hi) - pi_lo)((x + pi_hi) + pi_lo) * P(x^2) * x.

    Near x = +pi, x - pi_hi is exact (Sterbenz), so subtracting the tiny
    pi_lo leaves the true distance to pi instead of rounding it away.
    """
    x = np.asarray(x, dtype=dtype)
    hi, lo = split_pi(dtype)
    return ((x - hi) - lo) * ((x + hi) + lo) * horner(p.astype(dtype), x * x) * x


def eval_turns(x, p, dtype):
    """sin(2 pi x) for x in [0, 1] turns: -u * x * (x - 1) * P(u^2), u = x - 1/2.

    u^2 - 1/4 = (u - 1/2)(u + 1/2) = (x - 1) x, so the three zeros are built
    from x itself: x is exact, and x - 1 is exact for x >= 1/2 (Sterbenz).
    Since sin(2 pi (u + 1/2)) = -sin(2 pi u), the sign flips.
    """
    x = np.asarray(x, dtype=dtype)
    u = x - dtype(0.5)
    return -u * x * (x - dtype(1)) * horner(p.astype(dtype), u * u)


# ---------------------------------------------------------------------------
# Padé (see 02-pade-approximant)
# ---------------------------------------------------------------------------

def sin_taylor_fractions(order):
    """Exact Taylor coefficients c_0..c_order of sin about 0."""
    return [Fraction(0) if k % 2 == 0 else Fraction((-1) ** (k // 2), math.factorial(k))
            for k in range(order + 1)]


def solve_exact(A, b):
    """Gaussian elimination over Fractions. A is square, b a vector."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col] / M[col][col]
                M[r] = [a - f * c for a, c in zip(M[r], M[col])]
    return [M[i][n] / M[i][i] for i in range(n)]


def pade(c, m, n):
    """[m/n] Padé coefficients (p_0..p_m, q_0..q_n) with q_0 = 1, from
    Taylor coefficients c. Exact if c is exact."""
    coef = lambda k: c[k] if k >= 0 else Fraction(0)
    # Orders m+1..m+n of P - f*Q must vanish: sum_j q_j c_{k-j} = 0.
    A = [[coef(k - j) for j in range(1, n + 1)] for k in range(m + 1, m + n + 1)]
    b = [-coef(k) for k in range(m + 1, m + n + 1)]
    q = [Fraction(1)] + (solve_exact(A, b) if n else [])
    p = [sum(q[j] * coef(k - j) for j in range(min(k, n) + 1)) for k in range(m + 1)]
    return p, q


def pade_sin(m, n):
    """Odd numerator and even denominator in x^2: sin ~ x * Pe(x^2) / Qe(x^2)."""
    p, q = pade(sin_taylor_fractions(m + n), m, n)
    return p[1::2], q[0::2]


def eval_pade(x, pe, qe, dtype):
    """x * Pe(x^2) / Qe(x^2), entirely in dtype."""
    x = np.asarray(x, dtype=dtype)
    x2 = x * x
    num = horner(np.array([dtype(c) for c in pe]), x2)
    den = horner(np.array([dtype(c) for c in qe]), x2)
    return x * num / den


# ---------------------------------------------------------------------------
# Lookup table (see 01-lookup-table)
# ---------------------------------------------------------------------------

def make_table(size, dtype):
    """size entries over one period starting at -pi, plus a guard point so
    linear interpolation never has to wrap i+1."""
    i = np.arange(size + 1)
    return np.sin(-PI + 2 * PI * i / size).astype(dtype)


def eval_lut(x, table, interp, dtype):
    """Look x up in table, entirely in dtype (index math included)."""
    size = len(table) - 1
    x = np.asarray(x, dtype=dtype)
    pos = (x + dtype(PI)) * dtype(size / (2 * PI))
    if interp == "nearest":
        # floor(pos + 0.5), like a C++ static_cast<int>(pos + 0.5f) for pos >= 0.
        i = np.floor(pos + dtype(0.5)).astype(np.int64) % size
        return table[i]
    i0 = np.floor(pos)
    frac = pos - i0
    i = i0.astype(np.int64) % size
    a, b = table[i], table[i + 1]
    return a + frac * (b - a)


# ---------------------------------------------------------------------------
# The progression, for side-by-side tables
# ---------------------------------------------------------------------------

STEPS = ["taylor", "vanilla", "factored", "split", "turns", "remez"]
STEP_NAMES = {
    "taylor": "Taylor",
    "vanilla": "Chebyshev",
    "factored": "Cheb factored",
    "split": "Cheb split-π",
    "turns": "Cheb turns",
    "remez": "Remez turns (rel)",
}


class Method(NamedTuple):
    f: Callable      # f(x) -> approx, all arithmetic in the eval dtype
    ncoef: int       # stored polynomial coefficients (not counting pi)
    mul: int
    add: int
    domain: str = "radians"  # input domain, see metrics.DOMAINS


def method(step, degree, dtype):
    """One step of the progression at a given total degree.

    Operation counts, with K coefficients in P:
      x * P(x^2)                    x^2, K-1 Horner steps, times x
                                    -> K+1 mul, K-1 add
      (x - pi)(x + pi) * P * x      also two subtractions/additions and two
                                    more products -> K+3 mul, K+1 add
      split pi                      two more adds for pi_lo -> K+3 mul, K+3 add
      -u * x * (x - 1) * P(u^2)     u = x - 1/2 and x - 1, three products
                                    -> K+3 mul, K+1 add (sign folds into P)
    Each Horner step is a multiply-add, so K-1 of them can be FMAs.
    """
    if step in ("turns", "remez"):
        p = fit_cheb_turns(degree) if step == "turns" else fit_remez_turns(degree)
        k = len(p)
        return Method(lambda x: eval_turns(x, p, dtype), k, k + 3, k + 1, "turns")
    if step in ("taylor", "vanilla"):
        c = taylor_coeffs(degree) if step == "taylor" else fit_cheb_vanilla(degree)
        k = len(c)
        return Method(lambda x: eval_odd(x, c, dtype), k, k + 1, k - 1)
    p = fit_cheb_factored(degree)
    k = len(p)
    if step == "factored":
        return Method(lambda x: eval_factored(x, p, dtype), k, k + 3, k + 1)
    return Method(lambda x: eval_split(x, p, dtype), k, k + 3, k + 3)


def progression(upto, degree, dtype, ctx):
    """(name, Method, Result) for every step up to and including `upto`.
    Each step is measured on its own input domain; a context for any other
    domain than ctx's is built on demand."""
    from . import harness
    contexts = {ctx.domain: ctx}
    rows = []
    for step in STEPS[:STEPS.index(upto) + 1]:
        m = method(step, degree, dtype)
        if m.domain not in contexts:
            contexts[m.domain] = harness.make_context(m.domain)
        rows.append((STEP_NAMES[step], m, harness.evaluate(m.f, degree, "", contexts[m.domain])))
    return rows
