# exp2, log2, tanh, exp: bits for the integer part, polynomials for the rest

Written when the slides used log₂ degree 9 and 2^x degree 6; they now use degree 7
and degree 5 with 1 + f Q(f) (`degree-effects.md`). Numbers are as of writing.

Notes from building `examples/10-exp2`, `11-log2`, `12-tanh` and `13-exp`,
and their benches (`adc_bench exp2|log2|tanh|exp`).

- **Error numbers** are the bench's exhaustive f32 check unless marked
  Python (a sampled set). The bench is clang 21 on arm64, which fuses Horner
  steps into FMAs, so ULPs differ slightly from the Python model.
- **Timings** are on AC power, ns per sample. *buffer* is a loop the compiler
  can vectorize.

## The shared idea

A float is 2^e · m: the exponent field is already a base-2 log
(`common/bits.py`).

- **exp2:** 2^x = 2^n · 2^f. Write `(n + 127) << 23` for 2^n; only 2^f on a
  unit interval needs a polynomial.
- **log2:** log2 x = e + log2 m. Read e from the field; only log2 m on one
  octave needs a polynomial.
- **The bit tricks are the zero-polynomial limit of the same split.**
  - Schraudolph: write x · 2²³ into the bits, giving 2^f ≈ 1 + f.
  - Mitchell: read the bits as a number, giving log2 m ≈ m − 1.
  - Both are chords. ±3% / ±0.043, about **±52 cents** with a tuned offset,
    for one multiply and add. log2's runs at 38× `std::log2` in a buffer.
- **The reduction is exact** (bit reads, Sterbenz), so the error repeats
  every unit of x (exp2) or every octave (log2). Edges aside, any unit
  interval is the whole story.

## exp2 (10)

| method | mul/add | max ULP | cents | buffer ns |
|---|---|---|---|---|
| `std::exp2` | — | 0.51 | 1.0e-4 | 1.42 |
| bits, tuned offset | 1/1 | 3.7e5 | **51** | **0.09** |
| floor + P, degree 6 (vsl) | 7/7 | 1.29 | 1.4e-4 | 0.31 |
| round + P, degree 6 | 7/7 | 0.99 | 1.5e-4 | 0.31 |
| round + P, degree 5 | 6/6 | 3.00 | 3.8e-4 | 0.27 |
| table N=16 + P, degree 3 | 7/4 + load | 1.00 | 1.9e-4 | 0.85 (gather) |
| round 6, full range | +1 mul, clamp | 0.99 | 1.5e-4 | 0.58 |

- **Centring doesn't help exp2's fit.** 2^f on [−½, ½] is 2^f on [0, 1]
  times √2⁻¹: the same curve, so the same relative error at the same degree
  (Python: identical max rel at degree 3). Round's only real win is that
  x − round(x) is exact (Sterbenz) where x − floor(x) isn't for x in (−1, 0).
  This contradicts the talk plan's "P(f) on [−½, ½]" note.
- **To actually shrink the interval, use a table:** 2^(j/N) for the top bits
  of the fraction. 16 entries (64 bytes) plus degree 3 reaches 1.0 ULP, as
  good as degree 6. But the table becomes a gather and won't vectorize, so it
  is 3× slower in a buffer. Same lesson as the sin LUTs.
- **Edges** (Python's edge table):
  - 2^n from bits is only right for n in [−126, 127].
  - Below −126 the field wraps to garbage (−4e31).
  - n = 128 gives inf; n ≥ 129 carries into the sign bit and gives −0.0.
  - Round reduction reaches n = 128 already at x = 127.5.
  - The fix: clamp x, then build 2^n as 2^(n/2) · 2^(n−n/2). Subnormals and
    overflow come out right, for one multiply and about 2× the buffer time.
  - None of this matters for mtof or dB → gain.
- **In audio units:** the degree-5 polynomial is 0.0004 cents and 2e-6 dB.
  Anything at degree 4 or above is far below audibility. The bit trick's 52
  cents isn't.

## log2 (11)

| method | mul/add | max ULP | cents (log2 x in [−20, 10]) | buffer ns |
|---|---|---|---|---|
| `std::log2` | — | 0.50 | 0.0011 | 1.91 |
| bits, tuned offset | 1/1 | 6e12 | **52** | **0.05** |
| mantissa [1, 2), degree 9 | 9/10 | **4.7e6** at x = 1⁻ | 0.0013 | 0.53 |
| fold at 1.5, degree 9 (vsl) | 10/11 | 2.46 | 0.0012 | 0.68 |
| fold at √2, degree 9 | 10/11 | 2.10 | 0.0012 | 0.68 |
| fold at 1.5, degree 7 | 8/9 | 20.9 | 0.0019 | 0.63 |
| atanh, degree 5 (3 coeffs) | 5/6 + div | 5.29 | 0.0013 | 0.65 |
| atanh, degree 7 | 6/7 + div | 2.99 | 0.0012 | 0.70 |

- **Why the fold matters** (the talk plan's "why centring matters"):
  - u · P(u) builds in the zero at m = 1, so x just *above* 1 is fine.
  - x just *below* 1 reads e = −1, m ≈ 2, and −1 + 0.99999 cancels: 10⁶–10⁷
    ULP.
  - Folding m ≥ c down an octave makes x near 1 always read e = 0.
  - √2 centres the range in log terms and fits slightly better than vsl's
    1.5.
- **atanh form:** s = (m−1)/(m+1) is small (|s| ≤ 0.172) and log2 m is odd
  in s. 3 coefficients give 5 ULP, at the cost of a division; similar speed.
- **For audio, absolute error is the metric.** Every polynomial row is about
  0.001 cents. That floor is the f32 rounding of the *result* at log2 x =
  −20 (0.5 ulp of 20 ≈ 1e-6), not the approximation. ULP matters only if log2
  feeds something that needs relative accuracy near x = 1 (e.g. pow).
- **Edges:**
  - Subnormal inputs read e = −127 and a garbage m (1e-40 → −127 instead of
    −133). Fixed by scaling them by 2²³ first (a select and a multiply).
  - 0 gives −127, inf gives 128. Each needs a select.

## tanh (12)

| method | max abs | dB | max \|y\| | gain at 0 | quiet rel (Python) | buffer ns |
|---|---|---|---|---|---|---|
| `std::tanh` | 4.9e-8 | −146 | 1 | 0 | — | 2.61 |
| poly + clamp, degree 9 (L = 2.85) | 6.6e-3 | **−44** | 0.9989 | −0.073 dB | −42 dB | 0.19 |
| Padé [7/6] + clamp (JUCE) | 9.6e-5 | −80 | 1 | 0 | −136 dB | 0.29 |
| Padé [9/8] + clamp | 6.9e-6 | −103 | 1 | 0 | −136 dB | 0.36 |
| sigmoid, vsl's s | 1.6e-6 | −116 | 1 | −0.00004 dB | −108 dB | 0.31 |
| **sigmoid, Remez s, degree 9** | 8.8e-7 | **−121** | 1 | +0.00004 dB | −107 dB | 0.31 |
| sigmoid, Remez s, degree 11 | 1.9e-7 | −134 | 1 | 0 | −127 dB | 0.34 |
| exp2-based 1 − 2/(e + 1) | 1.9e-7 | −135 | 1 | −0.002 dB | **+4 dB** | 0.78 |
| **exp2, expm1 form** | 1.5e-7 | **−136** | 1 | 0 | −134 dB | 0.51 |

- **Every form needs its asymptote from somewhere:**
  - Polynomial + clamp tops out near −44 dB at degree 9, and the kink sprays
    error to high harmonics. At h = 19 the error exceeds tanh's own harmonic
    (`_harmonics.png`).
  - Degrees 7 and 11 overshoot 1 before the clamp (max |y| 1.0005).
  - Padé [7/6] is superb near 0 but passes 1 at x = 4.97 and needs an
    output clamp.
- **The sigmoid builds the asymptote in.** s/√(1+s²) with s ≈ sinh: bounded,
  monotonic, smooth. Fitting s by Remez weighted with dy/ds · x =
  x·sech³x (07's machinery, `tanhpoly.fit_sigmoid`) gives **−121 dB with
  vsl's own cost**: 5 dB better than vsl's coefficients and 17 dB better than
  sinh's Taylor series. The levelled error E (7.4e-7) matches the measured
  max, so the linearised weight is right. Plain sech³ is 4 dB worse (f64
  fit error 1.2e-6 vs 7.5e-7).
- **exp2-based:** the most accurate per op, and it saturates through IEEE
  inf. But 1 − 2/(e + 1) cancels near 0: its error is ~ulp(1) absolute, so
  quiet signals get *relative* error of +4 dB, i.e. pure noise below about
  −140 dBFS. The **expm1 form** keeps the −1 exact: e − 1 = 2^n · f·Q(f) +
  (2^n − 1), so n = 0 has full relative precision. That fixes it (−134 dB
  quiet, exact gain at 0) at 0.51 ns.
- **Large inputs (Python edge table):**
  - Sigmoid: s·s overflows past |x| ≈ 500 in f32, so y = s · (1/inf) =
    **0**, a saturator that returns silence for a blown-up signal. At inf
    it's NaN. A 2-op input clamp fixes it.
  - Padé: inf/inf gives NaN past ~1e19.
  - The exp2 forms and poly + clamp are fine.

## exp (13): Cody–Waite

e^x = 2^n · e^r, r = x − n ln 2. Any rounding in n ln 2 (or in x log2 e, if
you go via exp2) is a big relative error in r, so in the result.

| method | mul/add | max ULP | \|x\| ≤ π (Python) | dB | buffer ns | fast-math max ULP |
|---|---|---|---|---|---|---|
| `std::exp` | — | 0.51 | — | 5e-7 | 1.45 | 0.51 |
| exp2(x log2 e), degree 6 | 8/7 | **64** at x = 87 | 3.8 | 3.4e-5 | 0.33 | 64 |
| single ln2, degree 6 (Python) | 9/7 | 68 | 1.2 | 3.6e-5 | — | — |
| **Cody–Waite**, degree 6 | 10/8 | **1.10** | 1.2 | 8e-7 | 0.37 | **4.52** |
| Cody–Waite, degree 7 | 11/9 | 0.89 | 1.0 | 6.5e-7 | 0.39 | 4.52 |
| vsl::exp (floor, split, exp2 poly; Python) | 11/8 | 1.89 | 1.6 | 1e-6 | — | — |

- **ln2_hi has 16 significant bits** (= vsl's 0.693145752), so n · ln2_hi is
  exact for |n| < 256 and x − n · ln2_hi is exact by Sterbenz.
- **The unsplit error grows linearly with |x|.** It's a few ULP for audio
  arguments (one-pole coefficients |x| ≤ π, envelopes a few units) and
  3e-5 dB at worst. The split buys ULPs at large |x|, not anything audible.
- **Fast-math:** Cody–Waite degrades 1.1 → 4.5 ULP. That's worse, but not
  split-π's collapse (4.4 → 1.9e5), and still far better than no split (64).
  The compiler rewrote it partially; the asm wasn't checked. exp2, log2 and
  tanh moved by only a few tenths of a ULP under fast-math. The tanh check
  stays in |x| ≤ 20, so the exp2-based form's inf saturation path wasn't
  exercised under `-ffinite-math-only`.

## vsl findings to look at

These are *in vsl*, not this repo; nothing in vsl was changed.

1. **Vectorized `vsl::tanh` (and anything else on `rsqrt`) uses the bare
   hardware estimate.** `vsl_impl.hpp` maps vector `rsqrt` to `vrsqrteq_f32`
   / `vrsqrteq_f64` on NEON and `_mm_rsqrt_ps` on SSE, with no Newton step.
   Measured on this M-series, exhaustive over [−20, 20]:

   | `vsl::tanh` arithmetic | max abs | max \|y\| | per call ns |
   |---|---|---|---|
   | scalar path (1/sqrt) | −116 dB | 1 | 1.14 |
   | vector path (vrsqrte) | **−49.7 dB** | **1.0033** | 0.77 |
   | + 1 Newton step | −95 dB | 1.0000001 | 1.06 |
   | + 2 Newton steps | −116 dB | 1.0000001 | 1.34 |

   - The vector path is 66 dB worse than the fit allows, overshoots 1 by
     0.3%. SSE's estimate is 12-bit, so x86 would land somewhere else
     again (not measured here).
   - Two Newton steps are needed to match the scalar path, and per call
     that's *slower* than `1/sqrt` here.
   - Worth checking vectorized `1 / sqrt` against the estimate + 2 steps in a
     real SIMD loop before choosing. Our rows call the scalar intrinsic, so
     their buffer timing isn't vectorized.
2. **`vsl::exp2(x)` for x ≥ 129** writes 256 into the exponent field, which
   carries into the sign bit: returns −0.0 (x = 200 gives −1.4e-17).
   By the same arithmetic `vsl::exp` should do this from x ≈ 89.4 (n = 129;
   not run). Only matters for overflow inputs, but a clamp is two ops.
3. **`vsl::tanh` returns 0 for |x| ≳ 500** and NaN for inf (above). A clamp
   at ±9 or so fixes it at no accuracy cost.
4. **Not a bug, an opportunity:** a sech³-weighted Remez fit of vsl's own
   form is 5 dB better (`tanhpoly.fit_sigmoid(9)`, coefficients printed by
   `12-tanh/tanh_sat.py`).

## Not done

- ~~**Vectorized rsqrt timing**~~ Done 2026-10-19: `bench/src/rsqrt.cpp`
  (`adc_rsqrt`), sigmoid_tanh_9 four wide, every float in [−20, 20], M2 Pro
  on AC. NEON: std::sqrt (clang vectorizes it to fsqrt.4s + fdiv.4s, even
  with -fmath-errno) −121.2 dB, 0.30 ns; estimate −49.7 dB, |y| 1.0033,
  0.28 ns; + 1 step −95.5 dB, 0.37; + 2 steps −120.9 dB, 0.50: slower
  than sqrt + div. SSE under Rosetta (accuracy only, may not match Intel):
  estimate −76.3 dB, one step −120.6. Keep std::sqrt; GCC needs
  -fno-math-errno to vectorize it. Slide: Square root in SIMD (scratch).
- **Remez for exp2 / log2 with relative weight.** Chebyshev is already
  within ~1 ULP of the f32 floor at the vsl degrees, so probably not worth a
  slide.
