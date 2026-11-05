"""tanh(x) approximations for saturation: how to build the asymptote in
(see 12-tanh and notes/tanh-asymptotes.md).

  clamp             odd polynomial x * P(x^2) fit to tanh on [-L, L], input
                    clamped to ±L. A polynomial can't level off, so the
                    clamp makes the asymptote -- with a kink, and an error of
                    at least 1 - tanh(L). L is chosen per degree to balance
                    the two
  pade              [m/n] Padé from exact Taylor terms, output clamped to ±1.
                    An odd rational can't have a horizontal asymptote either
                    (numerator and denominator degrees differ by 1), so [7/6]
                    -- JUCE's FastMathApproximations::tanh -- runs past 1 at
                    x ~ 4.97 without the clamp
  sigmoid           s / sqrt(1 + s^2) with s = x * R(x^2) ~ sinh x: the
                    asymptote is in the algebra, |y| < 1 and monotonic for
                    free. vsl::tanh. R comes from a Remez fit weighted by how
                    much an error in s moves y: dy/ds = (1 + s^2)^(-3/2)
                    ~ sech^3 x. "sigmoid, Taylor" uses sinh's own series
  exp2              1 - 2 / (2^(2x log2 e) + 1), with 10-exp2's polynomial.
                    Saturates through IEEE inf: 2^big = inf gives 1, 2^-big =
                    0 gives -1. Near 0, 1 - 2/(e + 1) cancels: absolute error
                    ~ulp(1), so quiet signals see a noise floor near -144 dB
  exp2, expm1       the same split, but keep the -1 exact: with t = 2x log2 e
                    = n + f, e - 1 = 2^n (f Q(f)) + (2^n - 1), and
                    tanh = (e - 1) / (e - 1 + 2). For |t| < 1/2, n = 0 and
                    e - 1 = f Q(f) has full relative precision. Input clamped
                    to ±9.1 (tanh rounds to 1 there) so e - 1 stays finite

All evaluation is in `dtype`; numpy can't model a hardware rsqrt estimate,
so the sigmoid uses 1 / sqrt (see the bench for the estimate).
"""

import math
from fractions import Fraction
from typing import Callable, NamedTuple

import numpy as np

from . import exp2poly, fit
from .remez import remez
from .sinpoly import horner, pade

LOG2E = 1 / np.log(2)
SIGMOID_RANGE = 10.0    # Remez fits s on [0, SIGMOID_RANGE]; tanh(10) = 1 - 4e-9


# ---------------------------------------------------------------------------
# Fitting (always in f64)
# ---------------------------------------------------------------------------

def tanh_quotient_v(v):
    """tanh(x) / x as a function of v = x^2 (1 at v = 0)."""
    x = np.sqrt(np.asarray(v, dtype=np.float64))
    safe = np.where(x == 0, 1.0, x)
    return np.where(x == 0, 1.0, np.tanh(safe) / safe)


def fit_clamp(degree, limit):
    """p_k so that tanh(x) ~ x * sum p_k (x^2)^k on [-limit, limit]
    (Chebyshev in v = x^2 on [0, limit^2]); degree odd."""
    return fit.chebyshev(tanh_quotient_v, 0.0, limit * limit, (degree - 1) // 2)


def clamp_error(degree, limit):
    """Max |error| of the clamped polynomial: the fit on [0, L], or the gap
    1 - p(L) beyond it."""
    x = np.linspace(0, limit, 4001)
    p = fit_clamp(degree, limit)
    y = x * np.polynomial.polynomial.polyval(x * x, p)
    return max(np.abs(y - np.tanh(x)).max(), 1 - y[-1] if y[-1] < 1 else y[-1] - 1, 1 - np.tanh(limit))


def best_clamp_limit(degree):
    """The clamp point that minimizes the max error for this degree: a longer
    range costs fit error, a shorter one leaves 1 - tanh(L) on the table."""
    limits = np.linspace(1.0, 8.0, 141)
    errs = [clamp_error(degree, L) for L in limits]
    return float(limits[int(np.argmin(errs))])


def tanh_taylor_fractions(order):
    """Exact Taylor coefficients of tanh about 0: sinh / cosh, term by term."""
    s = [Fraction(0) if k % 2 == 0 else Fraction(1, math.factorial(k)) for k in range(order + 1)]
    c = [Fraction(0) if k % 2 else Fraction(1, math.factorial(k)) for k in range(order + 1)]
    t = []
    for k in range(order + 1):
        t.append(s[k] - sum(c[j] * t[k - j] for j in range(1, k + 1)))
    return t


def pade_tanh(m, n):
    """Odd numerator and even denominator: tanh ~ x * Pe(x^2) / Qe(x^2)."""
    p, q = pade(tanh_taylor_fractions(m + n), m, n)
    return p[1::2], q[0::2]


def best_pade_clamp(m, n):
    """The input clamp L for Padé [m/n] that minimizes the max error: the
    rational on [0, L], then its value at L held beyond. Past ~5 the rational
    climbs through 1; clamping the input (not the output) also keeps x^2 small,
    so the Horners can't overflow to inf / inf = NaN."""
    pe, qe = pade_tanh(m, n)
    def err(L):
        x = np.linspace(0, L, 20001)
        y = x * np.polynomial.polynomial.polyval(x * x, pe) / np.polynomial.polynomial.polyval(x * x, qe)
        return max(np.abs(y - np.tanh(x)).max(), abs(1 - y[-1]))
    limits = np.round(np.arange(3.0, 6.0, 0.01), 2)
    return float(limits[int(np.argmin([err(L) for L in limits]))])


def sinh_quotient_v(v):
    """sinh(x) / x as a function of v = x^2."""
    x = np.sqrt(np.asarray(v, dtype=np.float64))
    safe = np.where(x == 0, 1.0, x)
    return np.where(x == 0, 1.0, np.sinh(safe) / safe)


SIGMOID_CHEB_LIMIT = 3.4  # plain Chebyshev R: fit x on [0, 3.4] (scanned; flat optimum, 3.4-3.43)


def fit_sigmoid_cheb(degree, limit=SIGMOID_CHEB_LIMIT):
    """r_k for the sigmoid by a plain Chebyshev fit, no weighting: R = 1 +
    v Q(v), so R(0) = 1 exactly (y = x for tiny x, as tanh rounds), and Q =
    (sinh x / x - 1) / v interpolated at a high degree on v in [0, limit^2],
    then truncated (oversampled, as every slide fit). The short range does
    the weight's job: past about 3.4, tanh is flat and an error in R barely
    moves y, while fitting sinh's growth out to 9 swamps the fit (-15 dB).
    Past the range R keeps growing like sinh, so y still saturates. Degree 9:
    about -117 dB, 24 ULP, against the weighted Remez fit's -121 dB, 78."""
    from numpy.polynomial import Chebyshev, Polynomial
    def q(v):
        safe = np.where(v == 0, 1.0, v)
        return np.where(v == 0, 1 / 6, (sinh_quotient_v(v) - 1) / safe)
    n = (degree + 1) // 2
    c = Chebyshev.interpolate(q, fit.PROJECTION_DEGREE, domain=[0, limit * limit]).truncate(n - 1)
    return np.concatenate([[1.0], c.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1]).coef])


def fit_sigmoid(degree, full=False, relative=False, pinned=False):
    """r_k so that sinh(x) ~ x * sum r_k (x^2)^k, minimax in the error it
    causes in y = s / sqrt(1 + s^2): |dy| = x |dR| (1 + s^2)^(-3/2) with
    s = sinh x, i.e. weight x sech^3 x, on [0, SIGMOID_RANGE]. The weight is
    0 at x = 0, which the exchange can't divide by, so add a small floor.
    (Plain sech^3 also works but spends accuracy near 0 that y doesn't need.)

    relative: minimax in |dy| / tanh x instead (weight x sech^3 x / tanh x,
    1 at x = 0, no floor): the ULP error, where the absolute fit leaves
    R(0) = 1.0000046 and 78 ULP near 0 at degree 9. pinned: R(0) = 1
    exactly, so y = x for tiny x (tanh x rounds to x there); the rest,
    R = 1 + v Q(v), is fitted with the weight times v.
    """
    n = (degree + 1) // 2
    if relative:
        def w(v):
            x = np.sqrt(v)
            safe = np.where(x == 0, 1.0, x)
            return np.where(x == 0, 1.0, safe / np.cosh(safe) ** 3 / np.tanh(safe))
    else:
        w = lambda v: (np.sqrt(v) + 1e-3) / np.cosh(np.sqrt(v)) ** 3
    if not pinned:
        r, E, history = remez(sinh_quotient_v, w, 0.0, SIGMOID_RANGE**2, n, grid_size=40001)
        return (r, E, history) if full else r

    def g(v):
        safe = np.where(v == 0, 1.0, v)
        return np.where(v == 0, 1 / 6, (sinh_quotient_v(v) - 1) / safe)
    q, E, history = remez(g, lambda v: w(v) * np.maximum(v, 1e-12), 0.0, SIGMOID_RANGE**2, n - 1,
                          grid_size=40001)
    r = np.concatenate([[1.0], q])
    return (r, E, history) if full else r


def sigmoid_taylor(degree):
    """sinh's own series: r_k = 1 / (2k + 1)!."""
    return np.array([1 / math.factorial(2 * k + 1) for k in range((degree + 1) // 2)])


# ---------------------------------------------------------------------------
# Evaluation (in dtype)
# ---------------------------------------------------------------------------

def eval_clamp(x, p, limit, dtype):
    """x * P(x^2) with x clamped to ±limit first."""
    x = np.clip(np.asarray(x, dtype=dtype), dtype(-limit), dtype(limit))
    return x * horner(p.astype(dtype), x * x)


def eval_pade(x, pe, qe, dtype, clamp=True, limit=None):
    """x * Pe(x^2) / Qe(x^2), output clamped to ±1 (or not). With limit, the
    input is clamped to ±limit instead and the output is left alone."""
    x = np.asarray(x, dtype=dtype)
    if limit is not None:
        x, clamp = np.clip(x, dtype(-limit), dtype(limit)), False
    with np.errstate(over="ignore", invalid="ignore"):
        x2 = x * x
        y = x * horner(np.array([dtype(c) for c in pe]), x2) / horner(np.array([dtype(c) for c in qe]), x2)
    return np.clip(y, dtype(-1), dtype(1)) if clamp else y


def eval_sigmoid(x, r, dtype, limit=None):
    """s / sqrt(1 + s^2), s = x * R(x^2). With limit, x is clamped first:
    without it s * s overflows for |x| beyond ~500 (f32) and y = s * 0 = 0."""
    x = np.asarray(x, dtype=dtype)
    if limit is not None:
        x = np.clip(x, dtype(-limit), dtype(limit))
    with np.errstate(over="ignore", invalid="ignore"):
        s = x * horner(r.astype(dtype), x * x)
        return s * (dtype(1) / np.sqrt(s * s + dtype(1)))


def eval_exp2(x, c, dtype):
    """1 - 2 / (2^(2 x log2 e) + 1), 2^ from 10-exp2 (round, full range)."""
    x = np.asarray(x, dtype=dtype)
    e = exp2poly.eval_full_range(x * dtype(2 * LOG2E), c, "round", dtype)
    return dtype(1) - dtype(2) / (e + dtype(1))


EXPM1_LIMIT = 9.1   # tanh(9.1) = 1 - 2.5e-8: rounds to 1 in f32


def eval_expm1(x, q, dtype):
    """(e - 1) / (e - 1 + 2), e - 1 = 2^n (f Q(f)) + (2^n - 1), t = 2x log2 e.

    q fits (2^f - 1) / f on [-1/2, 1/2] (exp2poly.fit_table's quotient). The
    small part f Q(f) and the exact 2^n - 1 are added last, so n = 0 keeps
    relative precision -- no 1 + tiny - 1.
    """
    from . import bits
    x = np.clip(np.asarray(x, dtype=dtype), dtype(-EXPM1_LIMIT), dtype(EXPM1_LIMIT))
    t = x * dtype(2 * LOG2E)
    n = np.rint(t)
    f = t - n
    p = bits.pow2i(n.astype(bits.layout(dtype).sint), dtype)
    em1 = p * (f * horner(q.astype(dtype), f)) + (p - dtype(1))
    return em1 / (em1 + dtype(2))


def fit_expm1(degree):
    """q_k so that 2^f - 1 ~ f * sum q_k f^k on [-1/2, 1/2]; total degree `degree`."""
    return fit.chebyshev(exp2poly.expm1_quotient, -0.5, 0.5, degree - 1)


# ---------------------------------------------------------------------------
# Methods, for side-by-side tables
# ---------------------------------------------------------------------------

STEP_NAMES = {
    "clamp": "poly + clamp",
    "pade": "Padé + clamp",
    "sigmoid-taylor": "sigmoid, Taylor s",
    "sigmoid-vsl": "sigmoid, vsl s",
    "sigmoid": "sigmoid, Remez s",
    "exp2": "exp2-based",
    "expm1": "exp2, expm1 form",
}
EXP2_DEGREE = 6

# vsl::tanh's R, for comparison (fit by hand: github.com/sketch-audio/vsl, work/cheb_tanh.py).
VSL_SIGMOID = np.array([9.999960065e-01, 1.666701734e-01, 8.333356120e-03, 1.962047536e-04, 3.292487918e-06])


class Method(NamedTuple):
    f: Callable      # f(x) -> tanh(x), all arithmetic in the eval dtype
    size: str
    ncoef: int
    mul: int
    add: int
    div: int
    sqrt: int


def method(step, size, dtype):
    """size: odd degree for clamp / sigmoid, (m, n) for Padé, ignored for
    exp2 and vsl (degree 9).

    Operation counts (min / max for a clamp count as adds):
      clamp            2 for the clamp, x^2, Horner (k - 1), * x
      pade             2 for the input clamp, x^2, two Horners, * x, div
      sigmoid          x^2, Horner, * x, s^2 + 1, sqrt, div (or rsqrt), * s
      exp2             * 2 log2 e, exp2 (round + degree 6, full range), + 1,
                       div, 1 -
      expm1            clamp, * 2 log2 e, round, t - n, Horner (5), f * Q,
                       p * (...), p - 1, +, + 2, div
    """
    if step == "clamp":
        limit = best_clamp_limit(size)
        p = fit_clamp(size, limit)
        k = len(p)
        return Method(lambda x: eval_clamp(x, p, limit, dtype), f"{size}, L={limit:.2f}", k, k + 1, k + 1, 0, 0)
    if step == "pade":
        m, n = size
        pe, qe = pade_tanh(m, n)
        limit = best_pade_clamp(m, n)
        a, b = len(pe), len(qe)
        return Method(lambda x: eval_pade(x, pe, qe, dtype, limit=limit), f"[{m}/{n}], L={limit:.2f}", a + b,
                      1 + (a - 1) + (b - 1) + 1, (a - 1) + (b - 1) + 2, 1, 0)
    if step in ("sigmoid", "sigmoid-taylor", "sigmoid-vsl"):
        if step == "sigmoid":
            r = fit_sigmoid(size)
        elif step == "sigmoid-taylor":
            r = sigmoid_taylor(size)
        else:
            r, size = VSL_SIGMOID, 2 * len(VSL_SIGMOID) - 1
        k = len(r)
        return Method(lambda x: eval_sigmoid(x, r, dtype), str(size), k, k + 2, k, 1, 1)
    if step == "expm1":
        q = fit_expm1(EXP2_DEGREE)
        return Method(lambda x: eval_expm1(x, q, dtype), f"exp2 {EXP2_DEGREE}", len(q),
                      EXP2_DEGREE + 3, EXP2_DEGREE + 5, 1, 0)
    c = exp2poly.fit_exp2(EXP2_DEGREE, "round")
    return Method(lambda x: eval_exp2(x, c, dtype), f"exp2 {EXP2_DEGREE}", len(c),
                  EXP2_DEGREE + 4, EXP2_DEGREE + 3, 1, 0)
