# Reproducing "Lookup Tables Are Bad Approximators" (Chowdhury, 2025)

Written when the bench's Chebyshev fits were interpolations. `cheb_factored_sin_9`
is now a projection, so it matches his fit (1.7e-5 here against 2.3e-5 below).

`bench/build/adc_bench lutpoly` (src/lutpoly.hpp). M2 Pro, clang 21, on AC
power, best of 5 over 10M floats.

## The setup

| | his | here |
|---|---|---|
| target | sin on [−π, π], 2e-5 absolute | same |
| polynomial | `sin_poly_7` (from the post's figure): odd degree-7 fit of sin(x)/(π² − x²), total degree 9, 4 coefficients | **his coefficients**, outer factor (π² − x²) assumed; **1.65e-5** |
| LUT-320k | 320,000 entries, no interpolation (1.25 MB) | same; 1.05e-5 |
| LUT_lin-500 | 500 entries, linear (2 KB) | same; 1.98e-5 |
| inputs | 10M, ramp and random | ramp = 440 Hz oscillator phase, precomputed; uniform random |
| SIMD reference | xsimd sin | Accelerate `vvsinf` (no xsimd here) |

- **His "7th-order" is the quotient's polynomial.** Our `cheb_factored_sin_9`
  has the same form and cost, but plain Chebyshev interpolation reaches
  2.3e-5. His fit is closer to minimax.

## Results (speedup vs std::sin, or vs the SIMD reference)

| | his Ryzen 7 | **M2 Pro** | his M1 |
|---|---|---|---|
| scalar ramp: poly / LUT-320k / LUT_lin | 3.67 / 1.97 / 1.12 | **3.72 / 4.79 / 3.09** | 1.46 / 1.32 / 1.12 |
| scalar random | 10.1 / 5.02 / 2.76 | **10.8 / 8.77 / 8.46** | 7.18 / 5.16 / 3.14 |
| SIMD ramp | 3.98 / 1.39 / 1.79 | **2.07 / 1.13 / 0.83** | 2.0 / 1.75 / 1.48 |
| SIMD random | 4.4 / 1.49 / 2.0 | **2.07 / 0.61 / 0.84** | 4.5 / 1.94 / 2.22 |
| auto-vectorized loop, random (not in the post) | — | poly **42.5×**, tables 9× | — |

Absolute times on the M2 Pro, random inputs:

| method | ms per 10M | ns per sample |
|---|---|---|
| `std::sin` scalar | 57 | 5.7 |
| poly, scalar | 5.3 | 0.53 |
| poly, auto-vectorized | **1.4** | **0.14** |
| poly, explicit 4-wide | 1.8 | 0.18 |
| `vvsinf` | 3.8 | 0.38 |
| LUT-320k, any harness | 6.2–6.6 | 0.62–0.66 |
| LUT_lin-500, any harness | 4.5–6.8 | 0.45–0.68 |

## What reproduces

- **The polynomial's scalar speedups:** 3.72× vs his 3.67× (ramp) and 10.8×
  vs 10.1× (random). Nearly exact.
- **The polynomial wins once vectorized:** 1.8× the big table and 2.5× the
  small one with explicit SIMD; 4.7× when the compiler vectorizes the loop
  itself.
- **std::sin is ~2.7× slower on random input** (57 vs 21 ms). That's branch
  mispredictions in libm's range reduction, as the post's commenter found
  with perf on glibc. Apple's libm branches the same way.
- **His M1 ramp anomaly:** ramp 3–5× slower than random for every method,
  and all speedups squashed to 1.1–1.5×. Generating the ramp *inside* the
  timed loop (`x += dx; if (x >= π) x -= 2π`) reproduces it exactly: every
  method lands at 23–31 ms, 1.23–1.31×. The serial float add and wrap
  dominate, so his M1 ramp numbers measure the ramp generator, not sin.
  (His Ryzen ramp numbers don't show it as strongly: wider out-of-order
  core, or a different loop.)

## What doesn't: the scalar tables are machine dependent

- **On the M2 Pro the 1.25 MB table *beats* the scalar polynomial on a
  ramp** (4.39 vs 5.67 ms) and nearly matches it on random input. It fits
  the 16 MB L2.
- **On his Ryzen it doesn't.** Ryzen 7's L2 is 512 KB–1 MB per core, so the
  table lives in L3.
- **His LUT_lin-500 at 1.12× is surprisingly slow** for a 2 KB table that
  sits in L1. Here it's 3.1×. The difference is probably in the index
  computation or codegen (not visible in the post), not in the cache.
- **So:** "tables are bad" holds firmly for **vectorized** code (gathers),
  and for tables pushed out of the near caches (see
  `notes/tables-vs-math.md`). For scalar, warm, single-table code it
  depends on the cache hierarchy, and can go the other way on Apple
  silicon.
