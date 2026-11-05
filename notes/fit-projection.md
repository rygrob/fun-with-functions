# Chebyshev projection vs. interpolation: the landscape

Every Chebyshev fit in the repo can now be either (`common/fit.py`):

- **Examples:** every example from 03 to 14 takes `--fit projection|interpolation`.
  Projection became the default on 2026-11-05; this note was written when it was
  interpolation, and its "before" numbers are that. Projection isn't always the
  winner, as the tables below show: at degree 13 turns, max ULP goes the other way.
- **Bench:** `bench/generated` is built with projection. `python bench/generate.py
  --fit interpolation --out bench/generated-interpolation`, then build with
  `-DADC_GENERATED_DIR=generated-interpolation`, for the other. That set picks up
  `tuned/interpolation/*.json`, which is where 08 saves when run with
  `--fit interpolation`.

This note collects how the headlines move. It's for the consolidation pass;
the per-topic notes still quote the interpolation numbers.

## The two fits

For f = Σ aₖTₖ, a degree-n fit:

| | interpolation (the repo's default) | projection |
|---|---|---|
| what | match f at n + 1 Chebyshev nodes | keep the true a₀…aₙ: the truncated series |
| cost to fit | n + 1 evaluations | integrals, or interpolate at high N and truncate (what `fit.series` does) |
| error | dropped terms *alias* onto kept ones; bound ≈ 2·Σ\|aₖ\| over the tail | the tail itself, ≈ Σ\|aₖ\| |
| cost to evaluate | same | same |

## When it matters: the parity trick

**Projection is about 2× better in relative error and ULP for sin and tan,
and no better (sometimes a few percent worse) for exp2, log2, tanh, exp and
dB.**

- **The cause is the even/odd trick.** sin's quotients are even or odd, so we
  zero half the coefficients and fit with n + 1 nodes.
- **The aliased term:** the first dropped *non-zero* term is then
  T_{n+2}, and at the nodes (the zeros of T_{n+1}) T_{n+2} = −T_n. It
  aliases into the kept terms, and the interpolation error's leading term
  becomes a_{n+2}(T_{n+2} + T_n).
  - At t = ±1 that's **2·a_{n+2}**, against projection's a_{n+2}.
  - The interval's ends are exactly where relative error and ULP peak
    (sin → 0 at ±π, and at the turns form's 0, ½, 1).
- **Without parity** (2^f, log2 in u, anything fit in v = x²), the leading
  dropped term is T_{n+1}. It's zero at the nodes, so it doesn't alias:
  both methods share the same leading error, and they differ only at second
  order, in either direction.
- **Confirmed directly:** interpolating at **one more node** (degree n+1,
  then dropping the zero term) matches projection. Factored sin, degree 9:
  2.25e-5 → 1.64e-5 absolute, 5.77e-5 → 2.84e-5 relative.

So the practical rule is **don't interpolate an even or odd function at n + 1
nodes. Use n + 2, or project.** It's free at runtime, the same polynomial
form and cost.

- **Jatin Chowdhury's coefficients** (the lookup-table post) are exactly the
  degree-9 projection of Colin's quotient. That's why his fit reached
  1.64e-5 where ours was 2.25e-5.

## Python results (sampled f32; max ULP is a lower bound)

### sin, `--compare` (f32 eval)

| example | degree | max abs | max ULP | SINAD dB |
|---|---|---|---|---|
| 03 vanilla | 7 | 4.9e-4 → **2.6e-4** | 6.9e10 → 3.4e10 | 70.7 → 72.0 |
| | 9 | 1.2e-5 → **6.2e-6** | 1.6e9 → 8.3e8 | 103.3 → 104.6 |
| | 11 | 5.6e-7 → 5.4e-7 | 6.7e7 → 1.4e7 | 135.7 → 136.7 |
| | 13 | 5.8e-7 → 5.0e-7 | 1.2e7 → 1.2e7 | 137.4 → 138.1 |
| 04 factored | 9 | 2.3e-5 → 1.7e-5 | (zeros on f32 π) | 97.1 → 98.0 |
| | 11 | 6.4e-7 → 4.5e-7 | | 131.4 → 132.9 |
| 05 split-π | 7 | 9.2e-4 → 7.4e-4 | 4.0e4 → **2.0e4** | 64.1 → 66.7 |
| | 9 | 2.3e-5 → 1.7e-5 | 935 → **463** | 97.0 → 97.9 |
| | 11 | 5.6e-7 → 4.5e-7 | 17.8 → **10.4** | 131.2 → 132.2 |
| | 13 | 2.1e-7 → 2.2e-7 | 3.56 → 3.74 | 141.6 → 141.5 |
| 06 turns | 9 | 2.3e-5 → 1.7e-5 | 934 → **463** | 97.1 → 97.9 |
| | 11 | 5.3e-7 → 5.1e-7 | 16.1 → **9.9** | 132.2 → 133.4 |
| | 13 | 2.2e-7 → 2.2e-7 | 3.74 → 4.30 | 146.2 → 146.2 |
| | 15 | 2.1e-7 → 2.1e-7 | 3.54 → 3.54 | 146.3 → 146.3 |

- **Degrees 7–11** (fit-limited): ULP halves, as the parity argument
  predicts, and SINAD rises 1–2.6 dB.
- **Degree 13** (rounding-limited): a wash, and this sample is slightly
  *worse* for projection. Which way six coefficients round to f32 decides
  it, the same "fit vs. rounding" effect as Remez at 13 (session 1).

### Other functions (`--compare`, f32)

| function | projection's effect |
|---|---|
| tan(πx), turns | degree 5: 1746 → 873 ULP to Nyquist (2×); degree 7: 52.5 → 28.1; degree 9: 4.66 → **3.74**; degree 11: unchanged |
| tan, radians / split | same pattern at low degree; degree 9 split: 4.85 → 4.09 (radians in) |
| exp2 (no parity) | ±5%, either way: floor 6: 1.40 → 1.56 ULP; round 5: 2.94 → 3.08 |
| log2 (no parity) | fold 1.5, degree 5: 884 → 1034 (worse); degree 9: 2.77 → 2.74; √2 fold: 1.89 → 1.89 |
| tanh, poly + clamp (fit in x²) | slightly worse in max error (−43.6 → −42.1 dB at degree 9); the clamp points move |
| exp, Cody–Waite (no parity) | degree 6: 1.32 → 1.28 ULP; degree 5: 3.09 → 3.25 |
| dB free fits | slightly worse (degree 3: 8.7e-4 → 9.0e-4 dB). The pinned fits are Remez, so unaffected |
| tanh sigmoid, Remez, pinned fits | unaffected (not Chebyshev) |

## Exhaustive f32 (bench)

Every normal float; interpolation vs projection builds.
Methods that aren't Chebyshev fits (Taylor, Padé, LUTs, Remez, the pinned
dB fits, the sigmoid tanh, the bit tricks) are identical and left out.

| method | interpolation | projection |
|---|---|---|
| **sin, split-π 11** | 18.7 ULP | **11.5** |
| **sin, split-π 13** | 4.44 | 4.44 (same f32 coefficients after rounding) |
| **sin, turns 11** | 19.2 | **11.9** (Remez relative: 9.24) |
| **sin, turns 13** | **4.62** | 4.97 |
| sin, phase 13 (integer factors) | **5.20** | 5.70 |
| sin, tuned 13 (08's search) | 4.061 (Remez start) | 4.061 (projection start, different coefficients) |
| sin, tuned 11 (08's search) | **9.24** (Remez start) | 11.1 (projection start) |
| **tan(πx), turns 5** | 1192 (ULP ≤ 20 kHz) | **712** |
| **tan(πx), turns 9** (= vsl) | 4.61 | **3.93** |
| tan(πx), turns 11 | **3.97** | 4.24 |
| exp2, floor 6 (vsl) | **1.29** | 1.34 |
| exp2, round 6 | 0.99 | 0.98 |
| log2, fold 1.5, degree 9 (vsl) | 2.46 | **2.29** |
| log2, fold 1.5, degree 7 | **20.9** | 24.5 |
| log2, fold √2, degree 9 | 2.10 | 2.10 |
| log2, atanh 5 / 7 | 5.29 / 2.99 | 5.29 / 2.99 |
| tanh, poly + clamp 9 | −43.6 dB, max \|y\| 0.9989 | −42.1 dB, max \|y\| **1.00048** (overshoots) |
| exp, Cody–Waite 5 / 6 / 7 | 3.00 / 1.10 / 0.89 | 3.03 / 1.09 / 0.89 |
| dB → gain, free 3 | 8.9e-4 dB | 9.2e-4 dB |

## Fit vs. rounding, seen through 07

07's degree-11 table (max ULP over its sample):

| fit | f64 (fit only) | f32 (shipped) |
|---|---|---|
| Chebyshev interpolation | 15.05 | 16.1 |
| **Chebyshev projection** | **7.45** | **9.88** |
| Remez, absolute | 7.54 | 9.84 |
| Remez, relative | 5.62 | 8.36 |

- **Projection is effectively the absolute-error minimax fit.** The n + 1
  node interpolation was 2× off it.
- **Only relative-weighted Remez does better**, because it targets the
  metric (ULP).

## 08's search from a projection start

- `--start cheb` saves to `tuned/`, `--start cheb --fit interpolation` to `tuned/interpolation/`.
- **Degree 13:** the search ends at different coefficients from the
  Remez-started one (3.85 ULP on its sample), but the exhaustive check gives
  **4.061 at x = 0.4168, exactly the Remez-start result**. That worst case
  looks like a floor set by rounding in the factors −u·x·(x − 1), not by the
  polynomial. No coefficient search gets below it, so a lower number needs a
  different evaluation form.
- **Degree 11:** 11.1 from projection vs 9.24 from Remez. The fit limits
  degree 11, and relative Remez is the better fit, so it's the better start.

## What changes in the headlines

- **Nothing at the headline degrees.**
  - sin 13: split 4.44, turns 4.62 (projection 4.97), tuned 4.06.
  - exp2 6, log2 9 (vsl), Cody–Waite 6, tanh, dB: within a few percent
    either way.
  - These are rounding-limited or have no parity, so the fit method doesn't
    matter.
- **Low and mid degrees change a lot for sin and tan:** about 2× in ULP
  where the fit dominates.
  - sin 11: 19 → 12 ULP. sin 9: about 935 → 463.
  - tan turns 9 (vsl's degree): 4.61 → 3.93. tan turns 5: 1192 → 712.
  - SINAD +1–2.6 dB at degrees 7–11.
- **Talk framing:**
  - "Chebyshev" names a basis, not a fit.
  - **For an even or odd function, interpolate at one more node, or
    project.** Otherwise you give away half your accuracy at the ends, which
    are where relative error and ULP live.
  - Projection gets you to absolute-minimax for free. Relative Remez is the
    last step when ULP is the metric.
  - At the degrees where f32 rounding dominates, none of it matters; search
    (08) or a different form does.
- **Possible repo default for the consolidation:** projection for the sin and
  tan fits (or interpolation at n + 2 nodes), keeping in mind it makes turns
  13 slightly worse (4.62 → 4.97). The headline there is the tuned 4.06
  anyway.
