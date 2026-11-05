"""e^x approximations: exp2's split, plus the problem of getting from base e
to base 2 without losing the fraction (see 13-exp).

    e^x = 2^n * e^r,   n = round(x / ln 2),   r = x - n ln 2 in [-ln2/2, ln2/2]

  via exp2          2^(x log2 e) with 10-exp2's polynomial. x * log2 e
                    rounds by up to half an ulp of a number as big as 127, and
                    that rounding error lands in the fraction: ~64 ULP at
                    |x| ~ 87
  single ln2        r = x - n * f32(ln 2): n * ln2 rounds the same way
  Cody–Waite        ln 2 = ln2_hi + ln2_lo, ln2_hi with 16 significant bits so
                    n * ln2_hi is exact for |n| < 256, x - n * ln2_hi exact by
                    Sterbenz, and the tiny n * ln2_lo carries the rest. Split-π
                    for exp -- with the same fast-math fragility
  vsl               vsl::exp: floor(x / ln 2), the same split, then
                    2^(r log2 e) on 10-exp2's floor polynomial (one table of
                    coefficients for both functions)

All evaluation in `dtype`; results only right while n stays in the normal
range (|x| below ~87, as 10-exp2's edges).
"""

from typing import Callable, NamedTuple

import numpy as np

from . import bits, exp2poly, fit
from .sinpoly import horner

LN2 = np.log(2)
LOG2E = 1 / LN2
HI_BITS = 16


def split_ln2(dtype, hi_bits=HI_BITS):
    """ln2_hi = ln 2 rounded to hi_bits significant bits, ln2_lo = the rest
    in dtype. 16 bits leaves 8 for n in f32, so n * ln2_hi is exact for
    |n| < 256. (For f64 the same 16 bits leave 37 spare: plenty.)"""
    hi = np.round(LN2 * 2.0**hi_bits) / 2.0**hi_bits
    return dtype(hi), dtype(LN2 - hi)


def fit_exp(degree):
    """c_k so that e^r ~ sum c_k r^k on [-ln2/2, ln2/2]."""
    return fit.chebyshev(np.exp, -LN2 / 2, LN2 / 2, degree)


def eval_via_exp2(x, c2, dtype):
    """2^(x log2 e), round reduction, c2 from exp2poly.fit_exp2(..., "round")."""
    return exp2poly.eval_poly(np.asarray(x, dtype=dtype) * dtype(LOG2E), c2, "round", dtype)


def eval_reduced(x, c, dtype, split=True):
    """2^n * P(r), n = round(x log2 e), r = x - n ln 2 (split or not)."""
    x = np.asarray(x, dtype=dtype)
    n = np.rint(x * dtype(LOG2E))
    if split:
        hi, lo = split_ln2(dtype)
        r = (x - n * hi) - n * lo
    else:
        r = x - n * dtype(LN2)
    return bits.pow2i(n.astype(bits.layout(dtype).sint), dtype) * horner(c.astype(dtype), r)


def eval_vsl(x, c2, dtype):
    """vsl::exp: n = floor(x / ln 2), split reduction, then 2^(r log2 e) on
    the floor exp2 polynomial c2."""
    x = np.asarray(x, dtype=dtype)
    hi, lo = split_ln2(dtype)
    n = np.floor(x / dtype(LN2))
    r = (x - n * hi) - n * lo
    return bits.pow2i(n.astype(bits.layout(dtype).sint), dtype) * horner(c2.astype(dtype), r * dtype(LOG2E))


STEP_NAMES = {
    "via-exp2": "exp2(x log2 e)",
    "single": "single ln2",
    "cody-waite": "Cody–Waite",
    "vsl": "vsl (floor, split)",
}


class Method(NamedTuple):
    f: Callable
    degree: int
    mul: int
    add: int


def method(step, degree, dtype):
    """Operation counts (float only; the rint / floor and bit ops aside):
      via exp2         x * log2e, x - n, Horner, * 2^n
      single           x * log2e, n * ln2, x - ..., Horner, * 2^n
      Cody–Waite       two more: n * lo and its subtraction
      vsl              x / ln2 (a mul by 1/ln2), split, r * log2e, Horner, * 2^n
    """
    if step == "via-exp2":
        c2 = exp2poly.fit_exp2(degree, "round")
        return Method(lambda x: eval_via_exp2(x, c2, dtype), degree, degree + 2, degree + 1)
    if step == "vsl":
        c2 = exp2poly.fit_exp2(degree, "floor")
        return Method(lambda x: eval_vsl(x, c2, dtype), degree, degree + 5, degree + 2)
    c = fit_exp(degree)
    if step == "single":
        return Method(lambda x: eval_reduced(x, c, dtype, split=False), degree, degree + 3, degree + 1)
    return Method(lambda x: eval_reduced(x, c, dtype), degree, degree + 4, degree + 2)
