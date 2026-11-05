"""IEEE-754 bit fields: the exponent field is already a base-2 logarithm.

A normal float is

    x = 2^(E - bias) * (1 + M / 2^sig_bits)

with E the biased exponent field and M the significand field. Read as an
integer, the bits are E * 2^sig_bits + M -- so (bits / 2^sig_bits - bias) is
log2(x) up to a piecewise-linear error, and writing (n + bias) << sig_bits
builds 2^n exactly. exp2 and log2 split into an integer part done with bits
and a fractional part done with a polynomial.

  f32: 8 exponent bits, bias 127,  23 significand bits
  f64: 11 exponent bits, bias 1023, 52 significand bits

Every helper takes the float dtype and works on numpy arrays.
"""

from typing import NamedTuple

import numpy as np


class Layout(NamedTuple):
    uint: type
    sint: type
    bias: int
    sig_bits: int
    exp_mask: int   # exponent field, after shifting down


LAYOUTS = {
    np.float32: Layout(np.uint32, np.int32, 127, 23, 0xFF),
    np.float64: Layout(np.uint64, np.int64, 1023, 52, 0x7FF),
}


def layout(dtype):
    return LAYOUTS[np.dtype(dtype).type]


def as_bits(x, dtype):
    """The float's bit pattern as an unsigned integer."""
    return np.ascontiguousarray(x, dtype=dtype).view(layout(dtype).uint)


def from_bits(b, dtype):
    """Reinterpret unsigned integer bits as floats."""
    return np.ascontiguousarray(b, dtype=layout(dtype).uint).view(dtype)


def pow2i(n, dtype):
    """2^n by writing the exponent field: (n + bias) << sig_bits.

    Exact for n in the normal range, [1 - bias, bias] (f32: [-126, 127]).
    Outside it the field wraps: n = bias + 1 gives inf, larger n spills into
    the sign bit, and n <= -bias gives zero or garbage. Callers clamp.
    """
    L = layout(dtype)
    n = np.asarray(n).astype(L.sint)
    return from_bits((n + L.bias).astype(L.uint) << L.uint(L.sig_bits), dtype)


def split_exponent(x, dtype):
    """(e, m) with x = 2^e * m, m in [1, 2), for positive normal x.

    e is read straight from the exponent field; m keeps x's significand with
    the exponent field set to bias (i.e. 2^0). Subnormals read e = -bias and
    a wrong m: callers that care scale them up first.
    """
    L = layout(dtype)
    b = as_bits(x, dtype)
    e = ((b >> L.uint(L.sig_bits)) & L.uint(L.exp_mask)).astype(L.sint) - L.bias
    sig = b & L.uint((1 << L.sig_bits) - 1)
    m = from_bits(sig | L.uint(L.bias << L.sig_bits), dtype)
    return e, m


def bits_as_log2(x, dtype, offset=0.0):
    """Mitchell's approximation: the bits as a number, scaled,
    float(bits) * 2^-sig_bits - bias = e + (m - 1), minus `offset`.

    Exact at powers of two with offset 0. Done as C would --
    `(float)bits * 0x1p-23f - 127` -- so the int -> float conversion rounds
    away the low bits (f32 keeps 24 of the 31).
    """
    L = layout(dtype)
    b = as_bits(x, dtype).astype(dtype)
    return b * dtype(2.0**-L.sig_bits) - dtype(L.bias + offset)


def log2_as_bits(y, dtype, offset=0.0):
    """Schraudolph's approximation, Mitchell's inverse: the bits are
    int((y + offset) * 2^sig_bits) + (bias << sig_bits), so the exponent
    field gets floor(y) and the significand the fraction: 2^n * (1 + f).
    Done in dtype then integer, as C would; |y| must stay below 2^(31 - 23)
    for f32 or the int conversion overflows."""
    L = layout(dtype)
    t = np.trunc((np.asarray(y, dtype=dtype) + dtype(offset)) * dtype(2.0**L.sig_bits))
    return from_bits((t.astype(L.sint) + L.sint(L.bias << L.sig_bits)).astype(L.uint), dtype)
