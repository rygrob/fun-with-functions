# tan(πx) for filter cutoffs (09-tan)

Notes from building `examples/09-tan/tan_pi.py`. Numbers are f32 eval over
~133K inputs (dense near Nyquist, every f32 in the last 4096 steps below ½),
fs = 48 kHz. That's a sampled set, not exhaustive, so max ULP is a lower
bound. Reproduce with `python3 examples/09-tan/tan_pi.py [--compare] --no-show`.

## Setup

- A bilinear-transform filter (TPT one-pole, SVF) takes g = tan(π fc/fs),
  with x = fc/fs in [0, ½). The pole is at Nyquist.
- **The metric that matters:** for any g, the filter's cutoff lands exactly
  at x' = atan(g)/π. So the honest error is the cutoff shift in cents,
  `metrics.cutoff_cents`.
- **Sensitivity:** d(cents) ≈ 1731 · sinc(2x) · rel(g). At low fc, relative
  error maps 1:1 to cutoff error. Towards Nyquist the factor goes to 0: tan's
  slope absorbs the error.
- **Truth:** `metrics.tan_pi` uses cot(π(½ − x)) for |x| > ¼. ½ − x is
  exact, so the truth stays exact at the pole (cf. `sin_turns`).

## Forms compared

| form | expression | pole lands at (fc/fs) |
|---|---|---|
| no prewarp | g = πx | none |
| libm | `tanf(f32(π) * x)` | rounded input |
| Padé [m/n] | t·Pe(t²)/Qe(t²), t = πx | Q's zero: [3/2] 0.5033, [5/4] 0.5000037, [7/6] 0.5000000009 |
| Cheb radians | t·P(t²)/((h−t)(h+t)), h = f32(π/2) | 0.50000001 (f32(π/2)/π), reached through a rounded πx |
| Cheb split | same, h = h_hi + h_lo (as `vsl::tan`) | true π/2, but still reached through a rounded πx |
| **Cheb turns** | x·Q(x²)/((½−x)(½+x)) | **½ exactly**. Degree 9 reproduces `vsl::tan_pi` bit for bit |

- [7/6] is JUCE's `FastMathApproximations::tan`.
- Padé coefficients come from exact Taylor terms (sin/cos series division
  over `Fraction`s).

## Results

Degree 9 / Padé [7/6] side by side:

| method | mul/add/div | max ULP ≤ 20 kHz | max ULP to Nyquist | max cents ≤ 20k | max cents > 20k |
|---|---|---|---|---|---|
| g = πx | 1/0/0 | 1e7 | 1.7e7 | **610** | **780** |
| libm tan(π·x) | libm | 6.3 | 6.3e6 | 2.5e-4 | 1.4e-4 |
| Padé [7/6] | 9/6/1 | 9.2 | 1.0e7 | 3.8e-4 | 2.2e-4 |
| Cheb radians 9 | 8/6/1 | 6.0 | 2.5e6 | 4.7e-4 | 1.3e-4 |
| Cheb split 9 | 8/8/1 | 6.2 | 6.3e6 | 4.7e-4 | 1.5e-4 |
| **Cheb turns 9** | **7/6/1** | **4.4** | **4.7** | 4.3e-4 | 7.2e-5 |
| f32(tan), rounded | — | 0.5 | 0.5 | 1.0e-4 | 1.8e-5 |

Cost vs. accuracy (`--compare`):

| method | coeffs | mul/add/div | max cents ≤ 20k | > 20k |
|---|---|---|---|---|
| Padé [3/2] | 4 | 5/2/1 | 4.2 | 11 |
| Padé [5/4] | 6 | 7/4/1 | 0.0023 | 0.013 |
| **Cheb turns 3** | **2** | **4/3/1** | **1.7** | **0.2** |
| Cheb turns 5 | 3 | 5/4/1 | 0.038 | 0.025 |
| Cheb turns 7 | 4 | 6/5/1 | 0.0014 | 0.00056 |

### Split π/2 helps only for exact radians input

The table's last column (`max ULP, radians in`) feeds exact f32 t in
(0, π/2), every f32 in the last 4096 below π/2:

| method | fc/fs in, max ULP | radians in, max ULP |
|---|---|---|
| Padé [7/6] | 1.0e7 | 4.0e6 |
| Cheb radians 9 (single h) | 2.5e6 | 4.9e6 |
| Cheb split 9 | 6.3e6 | **4.85** |
| Cheb turns 9 | **4.7** | — |
| libm | 6.3e6 | 0.65 |

- **Split fixes vsl's case:** the caller has an exact t, so `(h_hi − t) + h_lo`
  is the true distance to the pole.
- **Split can't fix the filter case:** t = f32(π)·x has already rounded by
  up to ~1e-7 near π/2. That's as large as the distance to the pole for the
  last inputs, and f32 near π/2 is coarser (1.2e-7) than fc/fs steps mapped
  to radians (9.4e-8). Distinct cutoffs collapse onto one t.
- **The fix is the input domain, not the constant:** take the distance as
  ½ − x, which is exact. Turns does this.

## Bench (C++, exhaustive f32)

`bench/build/adc_bench tan`: every normal float in [FLT_MIN, ½), clang 21
on arm64 (Horner fused into FMAs), **on AC power**. Times are ns per sample.

| method | coeffs | per call | buffer | cold | max ULP ≤ 20k | max ULP to Nyq | cents ≤ 20k | cents > 20k |
|---|---|---|---|---|---|---|---|---|
| `std::tan(π·x)` | — | 5.12 | 3.63 | 2.25 | 6.57 | 6.3e6 | 2.5e-4 | 1.4e-4 |
| Padé [3/2] | 4 | 0.53 | 0.15 | 0.14 | 2.0e5 | 1.7e7 | 4.2 | 11 |
| Padé [5/4] | 6 | 0.72 | 0.19 | 0.20 | 109 | 1.4e7 | 0.0023 | 0.013 |
| Padé [7/6] | 8 | 0.91 | 0.23 | 0.23 | 9.63 | 1.4e7 | 4.4e-4 | 2.2e-4 |
| Padé [9/8] | 10 | 1.13 | 0.26 | 0.27 | 10.1 | 1.4e7 | 4.3e-4 | 2.2e-4 |
| turns 3 | 2 | 0.54 | 0.16 | 0.15 | 2.8e4 | 6.7e4 | 1.7 | 0.2 |
| turns 5 | 3 | 0.65 | 0.18 | 0.17 | 1192 | 1746 | 0.038 | 0.025 |
| turns 7 | 4 | 0.72 | 0.20 | 0.20 | 25.5 | 52.5 | 0.0014 | 5.8e-4 |
| turns 9 | 5 | 0.84 | 0.22 | 0.23 | 4.61 | **4.98** | 4.6e-4 | 8.8e-5 |
| turns 11 | 6 | 0.94 | 0.24 | 0.25 | 3.97 | **3.97** | 4.6e-4 | 7.8e-5 |

- **Turns 9 timing:** the first run said 1.07 / 0.28; two reruns gave
  0.84 / 0.22. That was noise.
- **Cost:** per call, turns degree d costs about the same as Padé
  [d−2/d−3]. At the same accuracy turns is cheaper: turns 9 beats [7/6] on
  time *and* ULP. Everything is 5–10× faster than `std::tan` per call and
  15–20× faster in a vectorized buffer.
- **f32 vs Python:** exhaustive ULP is slightly above the Python sample
  (turns 9: 4.98 vs 4.66), from FMA contraction and the denser input set.
  Cents agree to 2 significant figures.

## Talk points

- **Poles belong on exact floats, like zeros.** The sin lesson carries over:
  turns holds ~4.7 ULP right up to the pole. Every radians form reaches
  10⁶–10⁷ ULP there, because πx rounds by about an ulp of π/2, and that is
  the whole distance to the pole for the last f32 inputs.
- **The filter barely notices.** All those 10⁷-ULP errors move the cutoff by
  less than a millicent: the sinc(2x) factor. ULP is the wrong metric for
  this use; cents is the right one.
- **Near Nyquist, the shape of the error is the tell** (`_nyquist.png`):
  - Padé's cents error is *flat* to Nyquist. The pole is in the wrong
    place, so the cutoff carries a constant offset.
  - Turns' error *falls to zero* at Nyquist, because the pole is exact.
  - Radians forms sit on a ~1e-4 cent floor from rounding πx.
- **Where it audibly matters:**
  - skipping the prewarp: 600–800 cents near the top;
  - very low order: Padé [3/2] reaches 11 cents.
- **Turns lets you go cheaper.** Degree 3 (2 coefficients, 4 mul, 1 div)
  gives 1.7 cents. That's better than Padé [3/2] at fewer ops, and a sensible
  modulated-cutoff choice. Degree 5 is already 0.04 cents.
- **Fit vs. rounding:**
  - With `--eval f64`, Padé [7/6] still has 3e5 f32-ULP error near Nyquist:
    the misplaced pole is in the fit itself.
  - Turns 9 in f64 is 1.45 ULP. That's its fit error; f32 rounding makes up
    the rest.
  - Padé [9/8] is no better than [7/6] in f32.

## Not done yet

- Remez with relative weight on the turns quotient (07's machinery should
  just work), and the 08 coefficient search, if degree-9 ULP matters.
- Checking cents against an actual filter: measure the −3 dB point of a TPT
  one-pole, to back up the atan(g)/π claim on a slide.
