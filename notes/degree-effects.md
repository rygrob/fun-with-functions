# What raising the degree buys (Q&A background)

From `examples/18-degree-effects`: every float for f32 error, an even grid in
f64 for fit error, `loops.cpp`'s three loops on sorted inputs. M2 Pro on power,
clang 21, projection fits, median of five launches. Not on the slides.

A *step* is one Horner multiply-add. Compare per step, not per degree: an odd
function gets two degrees per step.

## The short version

- **Cost is the same everywhere.** One step costs about 0.02 ns in a buffer, 0.08
  ns per sample, 1.2 ns in feedback, whatever the function.
- **The gain depends on the function.** Bits per step: sin 5 to 7, log₂ 2.7, tanh
  as a polynomial about 0.9, as a sigmoid 3 to 5. The nearest complex singularity
  sets the rate.
- **Then it stops.** f32 rounding ends the gain at 20 to 22 bits, and each further
  step only costs.
- **Degree can't fix the wrong form.** Relative error and ULP near a zero, and
  tanh's asymptote, are fixed by the form.

## Cost: linear, and predictable from the core

| per step | predicted, 3.5 GHz | measured |
|---|---:|---:|
| buffer: 1 fma, 4 lanes, 4 pipes = 1/16 cycle | 0.018 ns | 0.02 |
| sample: 1 fma, 4 pipes = 1/4 cycle | 0.071 ns | mostly 0.03 to 0.12, about 0.08 |
| feedback: 4 cycles on the chain | 1.14 ns | 1.2 |

So 1 : 4 : 64. The feedback lines are parallel across every function. Feedback is
where degree is expensive, and it sets the break-even against the library: sin
degree 13 (10.8 vs 10.9 ns), log₂ fold degree 5 (13.2 vs 13.5), tanh sigmoid
degree 9 to 11 (16.1, 17.35 vs 17.3).

## Gain: singularities set the rate

A Chebyshev series on [a, b] falls like ρ⁻ⁿ, where the ellipse with foci a, b
through the nearest singularity has parameter ρ (Bernstein; Trefethen's ATAP).

| target | nearest singularity | predicted ρ | measured | bits per degree |
|---|---|---:|---:|---:|
| log₂ m / u on an octave | u = −1 (m = 0) | 5.83 | 6.5 at these degrees | 2.54 |
| tanh x on ±2.8 | x = iπ/2 | 1.71 | 1.71 | 0.77 |
| tanh x on ±4 | x = iπ/2 | 1.47 | 1.47 | 0.55 |
| sin x on ±π | none (entire) | | speeds up | 3.9 to 7 per coefficient |

- **log₂:** an octave [c, 2c) always puts m = 0 at t = −3, so ρ = 3 + √8 wherever
  the octave sits. The fold doesn't change the rate (2.7 bits per step, with or
  without it). It fixes ULP near 1. That's the "each degree is about 7× better" we
  measured; 6.5 tends to 5.83 as the degree grows.
- **sin:** nothing limits it, and each step is two degrees: 5 to 7 bits.
- **tanh's poles are why a polynomial is bad at it.** Widening the fit to reach
  the saturation also moves the poles closer, relative to the interval.

## The floor

| | fit reaches f32 floor | abs floor | best max ULP |
|---|---|---:|---:|
| sin, turns | degree 11 to 13 | 2.3e-7 (22 bits) | 4.17 at degree 15 |
| sin, split π | degree 11 to 13 | 2.4e-7 (22 bits) | 4.44 at degree 13 |
| sin, plain fit | degree 11 | 4.5e-7 (21 bits) | 7e6: never |
| log₂, fold | degree 7 to 8 | 1.0e-6 (20 bits) | 1.79 at degree 10 |
| log₂, [1, 2) | degree 7 to 9 | 1.0e-6 (20 bits) | 4.7e6: never |
| tanh, sigmoid | degree 11 to 13 | under 1.4e-7 (23 bits) | 3.67 at degree 13 |

- **Absolute error stops before ULP does.** log₂ fold degree 7 is 0.7 bits off the
  absolute floor but 24.5 ULP; degree 9 gets to 2.29. Spend the last degrees only
  if ULP matters.
- **log₂'s floor is higher** because the output is large: at log₂ x near −20 a
  float's own spacing is about 2e-6.
- **Sometimes more degree is worse:** the plain sin fit from 11 to 13 goes 7.4e6
  to 1.8e7 ULP, split π from 13 to 15 goes 4.44 to 4.79, log₂ fold from 10 to 11
  goes 1.79 to 2.15. It's rounding noise.
- **Split π and turns are the same fit, scaled:** their f64 errors match, and they
  differ only in rounding and speed (2.0× against 2.3× per sample at degree 13).
- **Rule of thumb:** compare the f64 fit error to the floor. When the fit is below
  it, you've finished; one more step can tidy ULP.

## Relative error: mostly form, not degree

Where the form gets zeros right, relative tracks absolute and ULP keeps falling
(split π, turns, fold). Where it doesn't, degree can't help: the plain sin fit stays
at relative error about 1 near ±π (0.69, 1.73, 1.33 at 11, 13, 15), and log₂ on [1, 2) stops at
0.39 (4.7e6 ULP) near 1. Split π, turns and the fold each did in one change what no
degree could. The same goes for tanh: degree 17 as a polynomial is 9.8 bits, where
degree 9 as a sigmoid is 19.4.

## Two oddities, explained

- **tanh polynomial's stairs:** 5 to 7, 9 to 11 and 13 to 15 gain nothing. The
  total error past the clamp is the fit's error at L plus 1 − tanh L. At 7, 11
  and 15 those add (7: 1.56e-2 + 1.21e-2 = 2.78e-2 measured); at 5, 9, 13, 17 they
  oppose. Averaged, about 0.9 bits per step.
- **Sigmoid slows past degree 9**, 4.7 to 3 bits per step. The fit is on [0, 3.4],
  and inside it the error keeps falling fast (2.6e-10 at degree 13 in f64). The
  maximum moves outside it, into the tail (2.2e-8), which the fit doesn't control.
