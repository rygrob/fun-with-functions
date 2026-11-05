# tanh and horizontal asymptotes

Notes from the end of the 2026-10-05 session, for a future `09-tanh`
example. Nothing here has been built in this repo yet.

## What vsl does

```cpp
p = x · R(x²)              // odd polynomial ≈ sinh(x): coefficients ≈ 1, 1/6, 1/120, 1/5040, …
return p * rsqrt(p*p + 1)  // tanh = sinh / sqrt(1 + sinh²), exact algebra
```

Measured in f64 over [0, 12] with vsl's coefficients:

- max abs error **1.4e-6 (−117 dB)**, at x ≈ 4.9
- **monotonic**, and **bounded below 1**, with no clamp and no branch
- slope at 0 is 0.999996, not 1; the fit traded it for accuracy elsewhere.
  That's −0.00003 dB of small-signal gain.

vsl's `work/tanh_compare.py` (github.com/sketch-audio/vsl) also tried polynomial + clamp, Padé, and
"from R" before settling on this sinh-proxy form.

## Options

1. **Polynomial + clamp.**
   - Branchless via `min`/`max`.
   - Leaves a kink at ±L unless the fit is constrained to meet 1 with zero
     slope there; kinks create harmonics.
   - Full f32 needs L ≈ 9. Chebyshev converges slowly over that range
     (poles at ±iπ/2), so in practice L is 1–3.
2. **Rational + clamp** (Lambert / Padé, as in JUCE `fastTanh` [7/6]).
   - **An odd rational can't have a horizontal asymptote**: numerator and
     denominator degrees differ by at least 1. So it always needs a clamp,
     plus a division. That's a good counterintuitive slide.
3. **Algebraic sigmoid, s / √(1 + s²), with s an odd polynomial ≈ sinh**
   (vsl's choice).
   - The asymptote is part of the structure: |y| < 1, monotonic when s is,
     smooth, branchless, SIMD-friendly. Costs one `rsqrt`.
   - Forgiving: dy/ds = (1 + s²)^(−3/2), so s's error shrinks by about 1/s³
     once tanh saturates.
   - **Fits 07's Remez directly:** target sinh, weight by the sensitivity
     sech³(x) (times one more sech for relative error). It's a linear problem
     in s's coefficients, so it's the same machinery with a new weight.
   - Lesson: the same as the sin story. Put the known structure (zeros,
     asymptotes) into the form, and let the polynomial do only the smooth
     part.
4. **exp-based: 1 − 2/(e^{2x} + 1),** reusing the exp2 example.
   - IEEE inf gives branchless saturation: overflow → 1, underflow → −1.
     **Fast-math trap:** `-ffinite-math-only` assumes no infs.
   - Cancellation near 0: use em1/(em1 + 2) with expm1, or a polynomial
     selected for small |x|.
   - The route to near-1-ULP accuracy across the full range. The lesson is
     composition and error propagation.
5. **Piecewise with selects:** a polynomial near 0 and an exp or asymptotic
   form beyond, merged by mask. Branchless, but you pay for both halves.

## Horizontal asymptotes in general

- **Structural transforms:**
  - s/√(1+s²): smooth, needs rsqrt
  - s/(1+|s|): no sqrt, but approaches the asymptote slowly
- **Compactify the domain:** t = x/(1+|x|) or x/√(1+x²) maps ℝ onto (−1, 1);
  then fit a polynomial in t on that finite interval.
- **Clamp,** with a constrained fit so value and slope match at the clamp
  point.
- **Asymptotic expansion for large |x|,** plus a select.

## Metrics a tanh example should add

- max abs error (dB)
- slope at 0: small-signal gain, which matters in feedback such as ladder
  filters
- monotonicity
- max |y| (overshooting 1 in a feedback loop can blow up)
- smoothness at any clamp point (C0 / C1 / C2)
- **harmonics of a driven sine** against true tanh, the saturator's
  version of SINAD
- **ADAA:** x/√(1+x²) has a closed-form antiderivative, √(1+x²). tanh's
  antiderivative, log cosh, needs its own careful approximation (cancellation
  at large |x|).

## Suggested 09 progression

clamp + polynomial → rational + clamp (JUCE) → algebraic sigmoid fitted
with sech³-weighted Remez (the hero for audio) → exp-based (optional, after
exp2; for ULP accuracy).
