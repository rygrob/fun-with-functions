# Why factoring out the zeros improves relative error

Example 04 / slide "Factor out the zeros". Written 2026-10-26 from a
conversation with Claude, so the argument is on file.

## The code

```cpp
const auto x2 = x * x;
auto p = c0;  p = p * x2 + c1;  ...           // Horner on P(x²): 6 coefficients
return (x - pi) * (x + pi) * p * x;            // the factors, outside Horner
```

Horner still evaluates P(x²); the factors are multiplied on afterwards, on
purpose:

- Multiplying them into one degree-13 polynomial and running Horner on
  that is the vanilla fit again: the zero at π exists only as big terms
  cancelling.
- Writing (x − π)(x + π) as x² − π² (reusing `x2`) is worse in float: see
  "x² − π² in one factor" below.

Cost against vanilla Chebyshev 13: the factors are 2 adds and 2 extra
multiplies, but P has 6 coefficients instead of 7, so Horner is one
multiply-add shorter. Net: **one multiply and one add**.

## The through line

1. **Relative error divides by |f|.** Near a zero of f, |f| is tiny, so any
   absolute error that doesn't also go to zero there blows up.
2. **A plain fit doesn't know the zeros are special.** Chebyshev spreads its
   absolute error evenly; at π it's off by some ε instead of exactly 0. Near
   π the relative error is about ε / |x − π|: unbounded as x → π, limited
   only by the floats nearest π (rel ≈ 1, 1.8e7 ULP for vanilla 13). At 0 it
   was already fine: an odd polynomial is exactly 0 at 0.
3. **Factoring makes p's relative error equal P's.** Write
   sin x = x(x − π)(x + π)·Q(x), Q the true quotient (the plot on the
   slide), and p = x(x − π)(x + π)·P. Then

       p − f = x(x − π)(x + π)·(P − Q)
       |p − f| / |f| = |P − Q| / |Q|

   The factors cancel out of the ratio exactly. Q stays between −1/π² and
   −1/2π², nowhere near zero, so a P with small absolute error has small
   relative error everywhere, ±π included. The problem moved from a
   function with zeros to one without.
4. **Rounding follows the same logic** ("the polynomial never has to
   produce a tiny answer"). Float addition is accurate relative to the size
   of its *inputs*: vanilla makes its tiny output near π by adding terms of
   size ~1 that nearly cancel, leaving ~1e-7 absolute rounding, which is
   huge relative to the output. Multiplication keeps relative accuracy
   (relative errors just add). Factored: P outputs ~−0.1 with full relative
   precision, `x - pi` is exact (Sterbenz: two floats within a factor of 2),
   and the tiny result comes from multiplying a small *exact* number by
   well-conditioned ones, never from cancellation.

So: zeros built into the form → the factors carry the small values,
exactly → everything else is a product of well-conditioned numbers →
relative error stays small.

## Where it still fails: the constant

Step 4 assumes `x - pi` is the distance to π. It's exact, but it's the exact
distance to **f32(π)**, about 8.7e-8 away from π. Within a few floats of π,
that representation error is as large as x − π itself, so the relative error
is back to ~1: factoring alone goes 1.8e7 → 1.2e7 ULP. The subtraction
doesn't create the error; it exposes the error already in the stored π.
Split π (π_hi + π_lo) supplies the missing 8.7e-8: 4.44 ULP.

- Factoring fixes the polynomial's problem (it doesn't pass through zero).
- Split π fixes the constant's problem (where the zero actually is).

## x² − π² in one factor (checked 2026-10-26, not on the slides)

Could the two split factors become one, x² − π², with π² split in hi/lo?
Exhaustive over every normal float in (0, π] (the forms are odd, so all of
[−π, π]), f32, same P, against `std::sin` in double; M2 Pro, on AC. Scratch
program, not in the repo.

| variant | after P | max ULP (default clang) |
|---|---|---|
| A: `((x - pi_hi) - pi_lo) * ((x + pi_hi) + pi_lo)` (slides) | 4 add, 3 mul | 4.44 |
| B: `fma(x, x, -pi2_hi) - pi2_lo` | 1 fma, 1 sub, 2 mul | 4.44 (same worst x) |
| E: `((x - pi_hi) - pi_lo) * (x + pi_hi)` | 3 add, 3 mul | 4.44 |
| C: `(x2 - pi2_hi) - pi2_lo` (rounded x²) | 2 sub, 2 mul | 3.3e6 |
| D: `fma(x, x, -pi2_hi)`, no lo | 1 fma, 2 mul | 6.5e6 |

- **C fails by catastrophic cancellation.** `x2 - pi2_hi` is itself exact
  (Sterbenz), but `x2` already carries up to half an ulp of ~9.87 (~5e-7),
  and near π the true x² − π² ≈ 2π(x − π) is that small. Cancellation
  exposes the earlier rounding, the same mechanism as f32(π).
- **B works** because the FMA forms x·x − π²_hi exactly before its single
  rounding: the x² analogue of Sterbenz for `x - pi_hi`. The lo part is
  still needed (D): f32(π²) is off by about as much as f32(π).
- **E:** x + π doesn't cancel, so its lo correction is wasted. One add
  fewer, same accuracy, portable.
- Timing, indicative (one sample at a time, 65,536 sorted inputs, median of
  100): A 1.13–1.16 ns, B 0.92–0.95, E 1.05–1.08.
- Fast-math: B kept 4.46 ULP; A and E fell to 1.2e7 as on the slides. The
  explicit `fma` seems to block the reassociation (one compiler, one build:
  not a guarantee).
- Catch: B needs hardware FMA (arm64; x86 with `-mfma` / Haswell+). Without
  it `std::fma` is a slow libm call. A non-fusing build is fine, since the
  `fma` is explicit (4.39 ULP with `-ffp-contract=off`). f64 would need π²
  split in double; untested.

Decision: the slides keep form A. One idea at a time (factors, then
Sterbenz and split π); B is a speaker-note / Q&A answer.
