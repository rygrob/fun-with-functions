"""2^x approximations: split x into an integer part, done with exponent bits,
and a fraction, done with a polynomial (see 10-exp2).

    2^x = 2^n * 2^f,   n integer,   2^n = bits((n + bias) << sig_bits)

  bits              Schraudolph: write (x + offset) * 2^23 straight into the
                    bits. The exponent field takes floor(x) and the
                    significand takes the fraction, so 2^f ~ 1 + f: the chord
  floor             n = floor(x), f = x - n in [0, 1), P(f) ~ 2^f. vsl::exp2
                    (degree 6)
  round             n = round(x), f = x - n in [-1/2, 1/2]: half the interval,
                    so a degree lower for the same error -- and x - n is exact
                    (Sterbenz), which x - floor(x) isn't for x in (-1, 0).
                    It does *not* make the fit better: 2^f on any unit
                    interval is the same curve times a constant
  unit              round, with 2^f = 1 + f Q(f): the 1 is built in, so every
                    integer x gives exactly 2^n (the fit alone misses them by
                    an ULP or so). The same idea as log2's u P(u)
  table             k = round(N x): n = k >> log2 N from the bits, j = k mod N
                    picks T[j] = 2^(j/N) from a table, and f = x - k/N in
                    [-1/2N, 1/2N] gets a tiny polynomial. The interval really
                    shrinks, so a low degree goes a long way (glibc's exp2f:
                    N = 32, degree 3)

The polynomials are fit to 2^f in f64; every evaluation runs in `dtype`.
None of these handle the edges: 2^n is only right for n in [-126, 127] (f32).
eval_full_range shows what handling them costs.
"""

from typing import Callable, NamedTuple

import numpy as np

from . import bits, fit
from .sinpoly import horner

REDUCTIONS = {"floor": (0.0, 1.0), "round": (-0.5, 0.5)}


# ---------------------------------------------------------------------------
# Fitting (always in f64)
# ---------------------------------------------------------------------------

def fit_exp2(degree, reduction):
    """c_k so that 2^f ~ sum c_k f^k on the reduction's interval."""
    return fit.chebyshev(np.exp2, *REDUCTIONS[reduction], degree)


def expm1_quotient(f):
    """(2^f - 1) / f, smooth through f = 0 (ln 2 there)."""
    f = np.asarray(f, dtype=np.float64)
    safe = np.where(f == 0, 1.0, f)
    return np.where(f == 0, np.log(2), np.expm1(f * np.log(2)) / safe)


def fit_table(degree, size):
    """q_k so that 2^f ~ 1 + f * sum q_k f^k on [-1/2N, 1/2N]; P = 1 + f Q
    has total degree `degree`."""
    h = 0.5 / size
    return fit.chebyshev(expm1_quotient, -h, h, degree - 1)


def fit_exp2_unit(degree):
    """q_k so that 2^f ~ 1 + f * sum q_k f^k on [-1/2, 1/2]; P = 1 + f Q has
    total degree `degree`."""
    return fit.chebyshev(expm1_quotient, -0.5, 0.5, degree - 1)


def table_values(size, dtype):
    """T[j] = 2^(j/N), rounded to dtype."""
    return np.exp2(np.arange(size) / size).astype(dtype)


def bits_offset():
    """The offset that minimizes Schraudolph's max relative error.

    The chord 1 + f over 2^f peaks at f = 1/ln2 - 1 (ratio 1.0861) and is 1 at
    both ends. Shifting x by o scales the whole thing by 2^o, so pick 2^o to
    centre the ratio around 1: max relative error 4.1%.
    """
    f = 1 / np.log(2) - 1
    peak = (1 + f) / 2**f
    return float(np.log2(2 / (1 + peak)))


# ---------------------------------------------------------------------------
# Evaluation (in dtype)
# ---------------------------------------------------------------------------

def eval_bits(x, dtype, offset=0.0):
    """Schraudolph: bits = int((x + offset) * 2^23) + (127 << 23)."""
    return bits.log2_as_bits(x, dtype, offset)


def reduce(x, reduction, dtype):
    """(n, f) with x = n + f: n as integers, f in dtype."""
    x = np.asarray(x, dtype=dtype)
    n = np.floor(x) if reduction == "floor" else np.rint(x)
    return n.astype(bits.layout(dtype).sint), x - n


def eval_poly(x, c, reduction, dtype):
    """2^n * P(f). Valid for n in the normal range."""
    n, f = reduce(x, reduction, dtype)
    return bits.pow2i(n, dtype) * horner(c.astype(dtype), f)


def eval_unit(x, q, dtype):
    """2^n * (1 + f Q(f)), n = round(x). Exact at every integer x, for n in
    the normal range."""
    n, f = reduce(x, "round", dtype)
    return bits.pow2i(n, dtype) * (dtype(1) + f * horner(q.astype(dtype), f))


def eval_table(x, q, table, dtype):
    """2^n * T[j] * (1 + f Q(f)), k = round(N x), n = k >> log2 N, j = k & (N - 1).

    N x is exact (N a power of two), and so is f = x - k/N (Sterbenz again).
    T[j] + T[j] * (f Q) adds the small correction last, so T[j]'s own
    rounding is most of the error.
    """
    L = bits.layout(dtype)
    size = len(table)
    shift = size.bit_length() - 1
    x = np.asarray(x, dtype=dtype)
    k = np.rint(x * dtype(size))
    f = x - k * dtype(1.0 / size)
    k = k.astype(L.sint)
    t = table[k & (size - 1)]
    p = f * horner(q.astype(dtype), f)
    return bits.pow2i(k >> shift, dtype) * (t + t * p)


def eval_full_range(x, c, reduction, dtype):
    """2^n * P(f) right at the edges too:

      clamp x to [min, max]   so n can't wrap the exponent field: below
                              2^-150 the answer rounds to 0 anyway, above
                              2^128 it's inf
      2^n = 2^(n/2) * 2^(n - n/2)   two exponent-field writes, each in range,
                              so subnormal results come out right (rounded
                              twice) and n = 128 overflows to inf by itself
    """
    L = bits.layout(dtype)
    lo, hi = -(L.bias + L.sig_bits + 1), L.bias + 1
    x = np.clip(np.asarray(x, dtype=dtype), dtype(lo), dtype(hi))
    n, f = reduce(x, reduction, dtype)
    n1 = n >> 1
    with np.errstate(over="ignore"):
        return bits.pow2i(n1, dtype) * horner(c.astype(dtype), f) * bits.pow2i(n - n1, dtype)


def eval_vsl_edges(x, c, dtype):
    """vsl::exp2's edge handling: floor reduction, select 0 for x < -126.
    Subnormal results flush to 0; x >= 129 writes 256 into the exponent
    field, which carries into the sign bit."""
    x = np.asarray(x, dtype=dtype)
    return np.where(x < dtype(-126), dtype(0), eval_poly(x, c, "floor", dtype))


# ---------------------------------------------------------------------------
# Methods, for side-by-side tables
# ---------------------------------------------------------------------------

STEP_NAMES = {
    "bits": "bits (Schraudolph)",
    "bits-tuned": "bits, tuned offset",
    "floor": "floor + P(f)",
    "round": "round + P(f)",
    "unit": "round + 1 + f Q(f)",
    "table": "table + P(f)",
}


class Method(NamedTuple):
    f: Callable      # f(x) -> 2^x, all arithmetic in the eval dtype
    degree: str
    ncoef: int       # polynomial coefficients (plus table entries, for table)
    mul: int         # float multiplies
    add: int         # float adds (x - n included)


TABLE_SIZE = 32


def method(step, degree, dtype, table_size=TABLE_SIZE):
    """Operation counts, beyond the float -> int conversion and the integer
    add and shift that build 2^n:
      bits             x * 2^23 (one mul; the + offset is one add)
      floor / round    floor or rint, x - n, Horner (degree mul + add), * 2^n
      unit             rint, x - n, Horner on Q (degree - 1 each), f * Q, 1 +, * 2^n
      table            N x, k / N, x - k/N, Horner on Q (degree - 1 each),
                       f * Q, t + t * p, * 2^n; plus one table load
    """
    if step == "bits":
        return Method(lambda x: eval_bits(x, dtype), "—", 0, 1, 0)
    if step == "bits-tuned":
        o = bits_offset()
        return Method(lambda x: eval_bits(x, dtype, o), "—", 0, 1, 1)
    if step == "table":
        q, t = fit_table(degree, table_size), table_values(table_size, dtype)
        return Method(lambda x: eval_table(x, q, t, dtype), f"{degree}, N={table_size}", degree,
                      degree + 4, degree + 1)
    if step == "unit":
        q = fit_exp2_unit(degree)
        return Method(lambda x: eval_unit(x, q, dtype), str(degree), degree, degree + 1, degree + 1)
    c = fit_exp2(degree, step)
    return Method(lambda x: eval_poly(x, c, step, dtype), str(degree), degree + 1, degree + 1, degree + 1)
