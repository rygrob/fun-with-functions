"""tan(pi x) approximations for filter cutoffs, x = fc / fs in [0, 1/2).

Every method takes the same f32 input x -- the normalized cutoff a filter
actually has -- so the radians methods pay for their x * pi themselves:

  prewarp-free      g = pi x, no tan at all: what the prewarp fixes
  pade              t * Pe(t^2) / Qe(t^2), t = pi x; [m/n] Padé from exact
                    Taylor terms of tan. JUCE's FastMathApproximations::tan
                    is [7/6]
  cheb radians      t * P(t^2) / ((h - t)(h + t)), t = pi x, h = pi/2: the
                    poles divided out, P from a Chebyshev fit of the smooth
                    quotient. h is a single dtype constant, like 04's pi
  cheb split        same P, but h = h_hi + h_lo so the poles land on true
                    pi/2, like 05. This is vsl::tan (radians in)
  cheb turns        x * Q(x^2) / ((1/2 - x)(1/2 + x)): poles at ±1/2, exact
                    floats, and no pi multiply. This is vsl::tan_pi (degree 9)

The radians methods also take radians directly (Method.f_radians), for the
case where the caller already has an exact t -- that's where the split helps.

Sizes are the numerator degree in t or x (odd); the Chebyshev forms' P
gets (degree + 1) / 2 terms. All evaluators do every operation in `dtype`.
"""

import math
from fractions import Fraction
from typing import Callable, NamedTuple

import numpy as np
from numpy.polynomial import chebyshev

from . import fit
from .metrics import PI
from .sinpoly import horner, pade


# ---------------------------------------------------------------------------
# Fitting (always in f64)
# ---------------------------------------------------------------------------

def tan_taylor_fractions(order):
    """Exact Taylor coefficients c_0..c_order of tan about 0, by dividing the
    sin series by the cos series: tan * cos = sin, solved term by term."""
    s = [Fraction(0) if k % 2 == 0 else Fraction((-1) ** (k // 2), math.factorial(k)) for k in range(order + 1)]
    c = [Fraction(0) if k % 2 else Fraction((-1) ** (k // 2), math.factorial(k)) for k in range(order + 1)]
    t = []
    for k in range(order + 1):
        t.append(s[k] - sum(c[j] * t[k - j] for j in range(1, k + 1)))
    return t


def pade_tan(m, n):
    """Odd numerator and even denominator in t^2: tan ~ t * Pe(t^2) / Qe(t^2)."""
    p, q = pade(tan_taylor_fractions(m + n), m, n)
    return p[1::2], q[0::2]


def pade_pole(m, n):
    """Smallest positive zero of the [m/n] denominator: where the Padé puts
    the pole that should be at pi/2."""
    _, qe = pade_tan(m, n)
    if len(qe) < 2:
        return np.inf
    roots = np.polynomial.Polynomial([float(c) for c in qe]).roots()
    v = np.real(roots[(np.abs(np.imag(roots)) < 1e-12) & (np.real(roots) > 0)])
    return float(np.sqrt(v.min())) if len(v) else np.inf


def radians_quotient(t):
    """f(t) = tan(t) (pi^2/4 - t^2) / t: smooth and even on [-pi/2, pi/2].
    tan(t) / t = sinc(t/pi) / cos(t); nodes never land on ±pi/2."""
    return np.sinc(t / PI) * (PI * PI / 4 - t * t) / np.cos(t)


def turns_quotient(x):
    """g(x) = tan(pi x) (1/4 - x^2) / x: smooth and even on [-1/2, 1/2],
    g(0) = pi/4. Written as pi sinc(x) (1/4 - x^2) / cos(pi x)."""
    return PI * np.sinc(x) * (0.25 - x * x) / np.cos(PI * x)


def _fit_even(quotient, half_width, degree):
    """Monomial coefficients p_k of quotient(u) ~ sum p_k (u^2)^k on
    [-half_width, half_width], from Chebyshev interpolation in t = u / half_width.
    A numerator of degree `degree` gives the quotient degree - 1 (even)."""
    a = fit.series(lambda t: quotient(t * half_width), degree - 1).coef
    a[1::2] = 0.0
    mono_t = chebyshev.cheb2poly(a)
    return np.array([c / half_width**k for k, c in enumerate(mono_t)])[0::2]


def fit_cheb_radians(degree):
    """p_k so that tan(t) ~ t * sum p_k (t^2)^k / (pi^2/4 - t^2)."""
    return _fit_even(radians_quotient, PI / 2, degree)


def fit_cheb_turns(degree):
    """q_k so that tan(pi x) ~ x * sum q_k (x^2)^k / (1/4 - x^2)."""
    return _fit_even(turns_quotient, 0.5, degree)


# ---------------------------------------------------------------------------
# Evaluation (in dtype)
# ---------------------------------------------------------------------------

def eval_prewarp_free(x, dtype):
    """g = pi x: the small-angle tan, i.e. no prewarp."""
    return dtype(PI) * np.asarray(x, dtype=dtype)


def to_radians(x, dtype):
    """t = pi x, rounded to dtype: what every radians method does first."""
    return dtype(PI) * np.asarray(x, dtype=dtype)


def eval_pade_t(t, pe, qe, dtype):
    """t * Pe(t^2) / Qe(t^2), entirely in dtype."""
    t = np.asarray(t, dtype=dtype)
    t2 = t * t
    num = horner(np.array([dtype(c) for c in pe]), t2)
    den = horner(np.array([dtype(c) for c in qe]), t2)
    return t * num / den


def eval_cheb_radians_t(t, p, dtype):
    """t * P(t^2) / ((h - t)(h + t)), h = pi/2 rounded to dtype: the poles
    land on dtype(pi/2), not pi/2."""
    t = np.asarray(t, dtype=dtype)
    h = dtype(PI / 2)
    return t * horner(p.astype(dtype), t * t) / ((h - t) * (h + t))


def split_half_pi(dtype):
    """h_hi = nearest dtype to pi/2, h_lo = what's left (see sinpoly.split_pi)."""
    hi = dtype(PI / 2)
    lo = dtype(PI / 2 - float(hi)) if dtype == np.float32 else dtype(math.cos(math.pi / 2))
    return hi, lo


def eval_cheb_split_t(t, p, dtype):
    """t * P(t^2) / (((h_hi - t) + h_lo)((h_hi + t) + h_lo)).

    Near t = pi/2, h_hi - t is exact (Sterbenz), so adding the tiny h_lo
    leaves the true distance to the pole -- *if t is exact*. Fed from
    t = pi x, the rounding is already in t and the split can't remove it.
    """
    t = np.asarray(t, dtype=dtype)
    hi, lo = split_half_pi(dtype)
    return t * horner(p.astype(dtype), t * t) / (((hi - t) + lo) * ((hi + t) + lo))


def eval_cheb_turns(x, q, dtype):
    """x * Q(x^2) / ((1/2 - x)(1/2 + x)).

    1/2 - x is exact for x in [1/4, 1/2] (Sterbenz), so the distance to the
    pole is exact, and so is x itself: no pi in the input at all.
    """
    x = np.asarray(x, dtype=dtype)
    return x * horner(q.astype(dtype), x * x) / ((dtype(0.5) - x) * (dtype(0.5) + x))


def eval_libm_t(t, dtype):
    """The library tan in dtype."""
    return np.tan(np.asarray(t, dtype=dtype))


# ---------------------------------------------------------------------------
# Methods, for side-by-side tables
# ---------------------------------------------------------------------------

STEPS = ["prewarp-free", "libm", "pade", "radians", "split", "turns"]
STEP_NAMES = {
    "prewarp-free": "g = πx (no tan)",
    "libm": "tan(π·x) libm",
    "pade": "Padé",
    "radians": "Cheb radians",
    "split": "Cheb split π/2",
    "turns": "Cheb turns (tanπ)",
}


class Method(NamedTuple):
    f: Callable      # f(x) -> g, x = fc / fs, all arithmetic in the eval dtype
    size: str        # "[7/6]", "9", ... for the tables
    ncoef: int       # stored coefficients (not counting pi or pi/2)
    mul: int
    add: int
    div: int
    pole: str        # where the pole lands, for the tables
    f_radians: Callable = None  # f(t) -> tan(t) for radians t; None if the form has no radians input


def _from_turns(f_t, dtype):
    return lambda x: f_t(to_radians(x, dtype))


def method(step, size, dtype):
    """One method at a given size: a numerator degree (odd) for the Chebyshev
    forms, an (m, n) pair for Padé; ignored by the fixed ones.

    Operation counts, with K coefficients in P, from x = fc / fs:
      pi x                          1 mul (every radians method)
      t * Pe(t^2) / Qe(t^2)         t^2, two Horners, times t, one div
                                    -> 1 + 1 + (a-1) + (b-1) + 1 mul
      t * P(t^2) / ((h-t)(h+t))     t^2, K-1 Horner steps, two adds, two
                                    products -> 1 + K + 2 mul, K+1 add, 1 div
      split h                       two more adds for h_lo -> K+3 add
      x * Q(x^2) / ((½-x)(½+x))     the same without pi x -> K+2 mul
    """
    if step == "prewarp-free":
        return Method(lambda x: eval_prewarp_free(x, dtype), "—", 0, 1, 0, 0, "none")
    if step == "libm":
        f_t = lambda t: eval_libm_t(t, dtype)
        return Method(_from_turns(f_t, dtype), "—", 0, 1, 0, 0, "true π/2", f_t)
    if step == "pade":
        m, n = size
        pe, qe = pade_tan(m, n)
        a, b = len(pe), len(qe)
        f_t = lambda t: eval_pade_t(t, pe, qe, dtype)
        return Method(_from_turns(f_t, dtype), f"[{m}/{n}]", a + b,
                      1 + 1 + (a - 1) + (b - 1) + 1, (a - 1) + (b - 1), 1, "Q's zero", f_t)
    if step in ("radians", "split"):
        p = fit_cheb_radians(size)
        k = len(p)
        if step == "radians":
            f_t = lambda t: eval_cheb_radians_t(t, p, dtype)
            return Method(_from_turns(f_t, dtype), str(size), k, k + 3, k + 1, 1, "rounded π/2", f_t)
        f_t = lambda t: eval_cheb_split_t(t, p, dtype)
        return Method(_from_turns(f_t, dtype), str(size), k, k + 3, k + 3, 1, "π/2 hi+lo", f_t)
    q = fit_cheb_turns(size)
    k = len(q)
    return Method(lambda x: eval_cheb_turns(x, q, dtype), str(size), k, k + 2, k + 1, 1, "½ exact")
