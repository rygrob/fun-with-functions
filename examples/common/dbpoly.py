"""dB <-> linear gain conversions (see 14-db): exp2 and log2 with a scale,
sized to a dB budget instead of an ULP one.

    gain = 10^(dB / 20) = 2^(dB * K),   K = log2(10) / 20 = 0.1661
    dB   = 20 log10(x)  = log2(x) / K,  1 / K = 20 log10(2) = 6.0206

dB -> gain (all floor reduction, n = floor(dB K), f in [0, 1)):
  bits              Schraudolph: (dB K) * 2^23 written into the bits. 2^f ~ 1 + f
                    -- which is the degree-1 pinned fit below
  bits, tuned       the same with the offset that centres the error: half the
                    max error, but 0 dB no longer gives a gain of exactly 1
  free d            P(f) ~ 2^f, Chebyshev (10-exp2): P(1) != 2 P(0), so the
                    gain steps at every 6.02 dB
  pinned d          P(0) = 1, P(1) = 2 exactly, minimax relative error in
                    between (fit.minimax_endpoints): continuous across the
                    steps, exact unity at 0 dB, and no worse for it

gain -> dB (fold at 1.5, as 11-log2; degree counts u * P(u)):
  bits              Mitchell: (float(bits) / 2^23 - 127) / K
  fold d            u * P(u), Chebyshev: P's two ends disagree at the fold, so
                    dB steps at x = 1.5 * 2^k
  fold pinned d     log2(1 + u) exact at both ends of [-1/4, 1/2]: seamless

Safe variants handle the whole float range: silence (-inf dB -> 0, 0 ->
-inf dB), subnormal gains, and overflow, for a clamp, a split 2^n and a few
selects. The plain ones are only right for gains in the normal range and,
for the bit tricks, |dB| below ~1540 (dB K 2^23 must fit an int32).

All evaluation in `dtype`.
"""

from typing import Callable, NamedTuple

import numpy as np

from . import bits, exp2poly, fit, log2poly
from .sinpoly import horner

K = np.log2(10) / 20          # dB -> log2 units
DB_PER_LOG2 = 1 / K           # 6.0206 dB per octave of gain
FOLD = 1.5


# ---------------------------------------------------------------------------
# Fitting (always in f64)
# ---------------------------------------------------------------------------

def fit_gain(degree, pinned):
    """c_k so that 2^f ~ sum c_k f^k on [0, 1]."""
    return fit.minimax_endpoints(np.exp2, 0.0, 1.0, degree) if pinned else exp2poly.fit_exp2(degree, "floor")


def fit_db(degree, pinned):
    """p_k so that log2(1 + u) ~ u * sum p_k u^k on the 1.5 fold's u range;
    degree counts u * P(u)."""
    lo, hi = log2poly.fold_range(FOLD)
    if pinned:
        return fit.minimax_endpoints(log2poly.log2_quotient, lo, hi, degree - 1)
    return log2poly.fit_mantissa(degree, FOLD)


# ---------------------------------------------------------------------------
# Evaluation (in dtype)
# ---------------------------------------------------------------------------

def to_log2(db, dtype):
    return np.asarray(db, dtype=dtype) * dtype(K)


def eval_gain_bits(db, dtype, offset=0.0):
    return bits.log2_as_bits(to_log2(db, dtype), dtype, offset)


def eval_gain(db, c, dtype):
    """2^n * P(f), n = floor(dB K)."""
    return exp2poly.eval_poly(to_log2(db, dtype), c, "floor", dtype)


def eval_gain_safe(db, c, dtype):
    """2^n * P(f) right over the whole float range: clamp dB K to [-151, 128]
    (-inf dB clamps too, and 2^-151 rounds to 0), then build 2^n in two
    halves so subnormal gains and overflow come out right."""
    L = bits.layout(dtype)
    t = np.clip(to_log2(db, dtype), dtype(-(L.bias + L.sig_bits + 1)), dtype(L.bias + 1))
    n = np.floor(t)
    f = t - n
    n = n.astype(L.sint)
    n1 = n >> 1
    with np.errstate(over="ignore"):
        return bits.pow2i(n1, dtype) * horner(c.astype(dtype), f) * bits.pow2i(n - n1, dtype)


def eval_db_bits(x, dtype, offset=0.0):
    return bits.bits_as_log2(x, dtype, offset) * dtype(DB_PER_LOG2)


def eval_db(x, p, dtype):
    """(e + u * P(u)) / K with the 1.5 fold."""
    return log2poly.eval_mantissa(x, p, FOLD, dtype) * dtype(DB_PER_LOG2)


def eval_db_safe(x, p, dtype):
    """eval_db for any x >= 0: subnormals scaled first, 0 -> -inf, inf -> inf
    (log2poly.eval_full_range)."""
    return log2poly.eval_full_range(x, p, FOLD, dtype) * dtype(DB_PER_LOG2)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

class Method(NamedTuple):
    f: Callable
    name: str
    degree: str
    mul: int
    add: int


GAIN_STEPS = ["pow", "bits", "bits-tuned", "free", "pinned", "pinned-safe"]
DB_STEPS = ["log10", "bits", "bits-tuned", "fold", "fold-pinned", "fold-pinned-safe"]


def gain_method(step, degree, dtype):
    """dB -> gain. Counts: * K, then exp2's floor, f, Horner (degree), * 2^n.
    Safe adds the clamp (2) and the second 2^n multiply."""
    if step == "pow":
        return Method(lambda db: np.power(dtype(10), np.asarray(db, dtype=dtype) / dtype(20)),
                      "pow(10, dB/20)", "—", 0, 0)
    if step == "bits":
        return Method(lambda db: eval_gain_bits(db, dtype), "bits (Schraudolph)", "—", 2, 0)
    if step == "bits-tuned":
        o = exp2poly.bits_offset()
        return Method(lambda db: eval_gain_bits(db, dtype, o), "bits, tuned offset", "—", 2, 1)
    c = fit_gain(degree, step != "free")
    if step == "pinned-safe":
        return Method(lambda db: eval_gain_safe(db, c, dtype), "exp2, pinned, safe", str(degree),
                      degree + 3, degree + 3)
    name = "exp2, pinned ends" if step == "pinned" else "exp2, free fit"
    return Method(lambda db: eval_gain(db, c, dtype), name, str(degree), degree + 2, degree + 1)


def db_method(step, degree, dtype):
    """gain -> dB. Counts: log2's (fold, m - 1, Horner, u * P, e + ...), * 1/K.
    Safe adds the subnormal scale (1 mul) and selects."""
    if step == "log10":
        return Method(lambda x: dtype(20) * np.log10(np.asarray(x, dtype=dtype)), "20 log10(x)", "—", 1, 0)
    if step == "bits":
        return Method(lambda x: eval_db_bits(x, dtype), "bits (Mitchell)", "—", 2, 1)
    if step == "bits-tuned":
        o = log2poly.bits_offset()
        return Method(lambda x: eval_db_bits(x, dtype, o), "bits, tuned offset", "—", 2, 1)
    p = fit_db(degree, step != "fold")
    if step == "fold-pinned-safe":
        return Method(lambda x: eval_db_safe(x, p, dtype), "log2 fold, pinned, safe", str(degree),
                      degree + 3, degree + 2)
    name = "log2 fold, pinned" if step == "fold-pinned" else "log2 fold, free fit"
    return Method(lambda x: eval_db(x, p, dtype), name, str(degree), degree + 2, degree + 2)
