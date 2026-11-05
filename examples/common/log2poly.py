"""log2(x) approximations: read the integer part from the exponent bits and
approximate log2 of the significand with a polynomial (see 11-log2).

    x = 2^e * m,   log2(x) = e + log2(m)

  bits              Mitchell: the bit pattern read as a number,
                    bits / 2^23 - 127 = e + (m - 1). log2(m) ~ m - 1, the
                    chord; exp2's bit trick run backwards
  mantissa          m in [1, 2), u = m - 1, log2(m) = u * P(u): the zero at
                    m = 1 built in, like sin's. But x just below 1 has e = -1
                    and m near 2: e + log2(m) = -1 + 0.99... cancels
  fold c            if m >= c, halve m and add 1 to e: m in [c/2, c), which
                    puts x = 1 in the middle of the range, so x near 1 has
                    e = 0 and keeps its relative precision. c = 1.5 is
                    vsl::log2; c = sqrt(2) centres it in log terms
  atanh             fold at sqrt(2), then s = (m - 1)/(m + 1) and
                    log2(m) = 2 atanh(s) / ln 2 = s * Q(s^2): odd, and |s| <=
                    0.172, so half the coefficients -- for a division

Degrees count the polynomial in u (or s) including the factored zero:
u * P(u) with P of degree d - 1 has degree d. All evaluation is in `dtype`.
Subnormal inputs read the wrong e and m; eval_full_range scales them first.
"""

from typing import Callable, NamedTuple

import numpy as np

from . import bits, fit
from .sinpoly import horner

LN2 = np.log(2)
SQRT2 = np.sqrt(2)


# ---------------------------------------------------------------------------
# Fitting (always in f64)
# ---------------------------------------------------------------------------

def log2_quotient(u):
    """g(u) = log2(1 + u) / u, smooth through u = 0 (1/ln 2 there)."""
    u = np.asarray(u, dtype=np.float64)
    safe = np.where(u == 0, 1.0, u)
    return np.where(u == 0, 1 / LN2, np.log1p(u) / (safe * LN2))


def atanh_quotient_v(v):
    """h as a function of v = s^2: log2((1+s)/(1-s)) / s = 2 atanh(s) / (s ln 2)."""
    s = np.sqrt(np.asarray(v, dtype=np.float64))
    safe = np.where(s == 0, 1.0, s)
    return np.where(s == 0, 2 / LN2, 2 * np.arctanh(safe) / (safe * LN2))


def fold_range(fold):
    """u = m - 1 range for a fold threshold (None: no fold, m in [1, 2))."""
    return (0.0, 1.0) if fold is None else (fold / 2 - 1, fold - 1)


def fit_mantissa(degree, fold=None):
    """p_k so that log2(1 + u) ~ u * sum p_k u^k on the fold's u range."""
    return fit.chebyshev(log2_quotient, *fold_range(fold), degree - 1)


def s_max(fold=SQRT2):
    """Largest |s| = |(m - 1)/(m + 1)| for m in [fold/2, fold)."""
    return max(abs((fold / 2 - 1) / (fold / 2 + 1)), abs((fold - 1) / (fold + 1)))


def fit_atanh(degree, fold=SQRT2):
    """q_k so that log2(m) ~ s * sum q_k (s^2)^k; degree is odd, in s."""
    return fit.chebyshev(atanh_quotient_v, 0.0, s_max(fold) ** 2, (degree - 1) // 2)


def bits_offset():
    """Mitchell's best constant: log2(m) - (m - 1) runs from 0 (at m = 1, 2)
    up to log2(1/ln2) - (1/ln2 - 1) = 0.0861 at m = 1/ln 2. Adding half of
    that centres the error: +-0.043, i.e. +-52 cents."""
    m = 1 / LN2
    return -0.5 * float(np.log2(m) - (m - 1))


# ---------------------------------------------------------------------------
# Evaluation (in dtype)
# ---------------------------------------------------------------------------

def eval_bits(x, dtype, offset=0.0):
    """Mitchell: float(bits) * 2^-23 - 127 - offset."""
    return bits.bits_as_log2(x, dtype, offset)


def reduce(x, fold, dtype):
    """(e, m) with x = 2^e * m, m in [fold/2, fold) (or [1, 2)); e in dtype."""
    e, m = bits.split_exponent(x, dtype)
    e = e.astype(dtype)
    if fold is not None:
        big = m >= dtype(fold)
        m = np.where(big, m * dtype(0.5), m)
        e = e + big.astype(dtype)
    return e, m


def eval_mantissa(x, p, fold, dtype):
    """e + u * P(u), u = m - 1 (exact: Sterbenz, m within a factor 2 of 1)."""
    e, m = reduce(x, fold, dtype)
    u = m - dtype(1)
    return e + u * horner(p.astype(dtype), u)


def eval_atanh(x, q, fold, dtype):
    """e + s * Q(s^2), s = (m - 1) / (m + 1)."""
    e, m = reduce(x, fold, dtype)
    s = (m - dtype(1)) / (m + dtype(1))
    return e + s * horner(q.astype(dtype), s * s)


def eval_full_range(x, p, fold, dtype):
    """eval_mantissa with subnormal inputs scaled by 2^sig_bits first (one
    select and a multiply), so their e and m read right. Zero and inf get
    selects of their own (-inf, inf); negative and NaN are out of scope."""
    L = bits.layout(dtype)
    x = np.asarray(x, dtype=dtype)
    tiny = x < dtype(np.finfo(dtype).tiny)
    with np.errstate(over="ignore"):
        xs = np.where(tiny, x * dtype(2.0**L.sig_bits), x)
    y = eval_mantissa(xs, p, fold, dtype) - np.where(tiny, dtype(L.sig_bits), dtype(0))
    return np.where(x == 0, dtype(-np.inf), np.where(np.isinf(x), x, y))


# ---------------------------------------------------------------------------
# Methods, for side-by-side tables
# ---------------------------------------------------------------------------

STEP_NAMES = {
    "bits": "bits (Mitchell)",
    "bits-tuned": "bits, tuned offset",
    "mantissa": "mantissa [1, 2)",
    "fold-1.5": "fold [0.75, 1.5)",
    "fold-sqrt2": "fold [√½, √2)",
    "atanh": "atanh, fold √2",
}
FOLDS = {"mantissa": None, "fold-1.5": 1.5, "fold-sqrt2": SQRT2}


class Method(NamedTuple):
    f: Callable      # f(x) -> log2(x), all arithmetic in the eval dtype
    degree: str
    ncoef: int
    mul: int
    add: int
    div: int


def method(step, degree, dtype):
    """Operation counts, beyond the bit masks and the int -> float of e:
      bits             float(bits) * 2^-23 - 127: 1 mul, 1 add
      mantissa         m - 1, Horner (d - 1), u * P, e + ...: d mul, d + 1 add
      fold             + m * 0.5 (exact) and e + 1, by select: +1 mul, +1 add
      atanh            m - 1, m + 1, one div, s^2, Horner (k - 1), s * Q, e + ...
    """
    if step == "bits":
        return Method(lambda x: eval_bits(x, dtype), "—", 0, 1, 1, 0)
    if step == "bits-tuned":
        o = bits_offset()
        return Method(lambda x: eval_bits(x, dtype, o), "—", 0, 1, 1, 0)
    if step == "atanh":
        q = fit_atanh(degree)
        k = len(q)
        return Method(lambda x: eval_atanh(x, q, SQRT2, dtype), str(degree), k, k + 2, k + 3, 1)
    fold = FOLDS[step]
    p = fit_mantissa(degree, fold)
    extra = 0 if fold is None else 1
    return Method(lambda x: eval_mantissa(x, p, fold, dtype), str(degree), degree,
                  degree + extra, degree + 1 + extra, 0)
