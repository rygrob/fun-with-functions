# Goldberg's "Fast Approximate Logarithms" (eBay, 2015): what it adds

Three posts by David Goldberg (I: the basics, II: rounding error, III: the
formulas; archived:
[I](https://web.archive.org/web/20250420103624/https://innovation.ebayinc.com/stories/fast-approximate-logarithms-part-i-the-basics/),
[II](https://web.archive.org/web/20250425155522/https://innovation.ebayinc.com/stories/fast-approximate-logarithms-part-ii-rounding-error/),
[III](https://web.archive.org/web/20250420101937/https://innovation.ebayinc.com/stories/fast-approximate-logarithms-part-iii-the-formulas/)).
Read 2026-10-06 against the repo's
11-log2 and 14-db work.

## Where it agrees with what we found

- **Reduce to [0.75, 1.5), not [1, 2).**
  - On [1, 2), x just below 1 reads e = −1, m ≈ 2, and cancels. He measures
    1.7 bits there; we measured 4.7e6 ULP.
  - The fold is where vsl's 1.5 comes from.
- **His extra reason for the shifted interval:** on [1, 2) a good *global*
  relative-error fit must satisfy g(1) = 0 *and* g(2) = 1. The shifted
  interval needs only g(1) = 0, which frees a degree of freedom.
  - That's the same "pin the ends so reduced intervals join up" idea as our
    `fit.minimax_endpoints` in 14.
  - He doesn't enforce continuity at the 1.5 fold itself. We measured that
    step (0.002 dB) and pinned it away.
- **Write the polynomial in (x − 1),** which is exact by Sterbenz for
  ½ ≤ x ≤ 2. The rule of thumb generalizes: near a zero x₀, approximate in
  terms of an *exact* x − x₀. That's the same principle as sin's turns form
  (factors built from exact differences).
- **Special-casing the region near 1** (a Taylor branch, even with bit tests)
  was *slower* than just using the (x − 1) form. Fix the form; don't add
  branches.
- **Sampling hides the worst case.** At input spacing 1/1024 every form looks
  fine. Only spacing 2⁻²² exposes the cancellation just below 1 (5.5 → 1.7
  bits). This is our "a sampled max is a lower bound; only the exhaustive
  check is real".
- **For tables, linear interpolation keeps relative error bounded near a
  zero; nearest lookup can't.** His log table also splits into lo/hi halves
  to dodge cancellation near 1, the table version of the fold. It's the same
  finding as our sin phase-LUT.

## Points we hadn't made

1. **The fold test is one mantissa bit.** m ≥ 1.5 exactly when the top
   fraction bit (0x00400000) is set: an integer AND, no float compare. That's
   a clean reason to fold at 1.5 rather than √2, even though √2 fits slightly
   better (2.10 vs 2.46 ULP at degree 9).
   - Our `split_exponent` + `m >= 1.5` compare is the float-compare
     version.
   - For SIMD either is fine; for scalar code, his is free.
2. **Rational forms: "a division is worth about one coefficient".** His
   line 8, (x−1)(a(x−1)+b)/((x−1)+c), gets 11.6 bits from three coefficients.
   A polynomial needs four for that (line 4, 11.3 bits). A cubic over a
   quadratic reaches 20.7 bits.
   - We only tried the atanh rational, which is a fixed structure, and it
     does better at similar cost:

     | form | relative bits | max ULP | ops |
     |---|---|---|---|
     | Goldberg quadratic (2 coefficients) | 5.5 | 3.5e5 | 2 mul, 2 add |
     | ours, fold 1.5, degree 3 (Chebyshev) | 8.2 | 4.6e4 | 3 mul, 4 add |
     | Goldberg line 8, rational (3 coefficients) | 11.6 | 4939 | 2 mul, 3 add, 1 div |
     | **ours, atanh, degree 3 (2 coefficients)** | **15.4** | 378 | 4 mul, 5 add, 1 div |
     | ours, atanh, degree 5 (3 coefficients) | 21.5 | 4.95 | 5 mul, 6 add, 1 div |

   - A general *minimax rational* (his method) in the atanh variable is
     untested and might beat both.
3. **Degenerate rationals.** One of his forms (line 9) costs the same as
   line 3 but is 1 bit worse. Its optimal coefficients run to about 1e4, so
   it collapses into a simpler rational. An extra parameter can be useless,
   and you only find out by fitting. A good caution for "just add a term".
4. **Minimax as a convex feasibility problem.** For relative error and
   rationals, he skips Remez. With the denominator multiplied through, |P −
   f·Q| ≤ γ·|f·Q| on a grid is linear in the coefficients for fixed γ, so you
   bisect on γ with an LP feasibility solver (CVX). It handles rationals,
   weights and extra constraints (pinned ends, monotonicity, sign) uniformly,
   and works on a discrete grid, even the f32 grid itself. It's a possible
   alternative to our hand-rolled Remez, and natural for the pinned fits.
5. **Cost per bit.** He reports *bits per nanosecond* (and ns per bit) per
   form and plots the trade-off curve. That's a one-number efficiency
   measure our summary tables could add: for example log2 fold 9 at 0.68 ns
   per sample and about 22 bits.
6. **Accuracy regime.** He targets 5–11 bits (3× and 2× speedups over
   `log2f`), where we target near-f32. In audio terms, 7 bits of *relative*
   log2 accuracy is about 0.05 dB near ±6 dB, but absolute dB error is the
   meter's metric. Our 14-db budget table is the audio-native version of his
   trade-off table.

## Possible follow-ups (not done)

- A minimax rational in the atanh variable (or his (x−1) form at higher
  order), fitted by the LP/bisection method, against atanh degree 3 / 5.
- The bit-test fold in the C++ emitters (integer AND instead of a float
  compare), and a timing comparison.
- Adding a bits-per-ns column to the bench summaries.
