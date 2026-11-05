# dB ↔ gain (14-db)

Notes from building `examples/14-db/db_gain.py` and `adc_bench dbgain|gaindb`.

- **Error numbers** are Python, over 2M f32 inputs per direction, unless
  marked bench (exhaustive f32, clang arm64 with FMA).
- **Timings** are the bench on AC power, ns per sample, in a loop the
  compiler can vectorize ("buffer").

## The setup

- gain = 10^(dB/20) = 2^(dB·K), and dB = log2(x)/K, with K = log2(10)/20.
  These are 10-exp2 and 11-log2 plus one multiply.
- **The specialised part is the budget.** 0.01 dB is far below audibility,
  so the question is the cheapest form per budget, not maximum accuracy:

  | budget | dB → gain | gain → dB |
  |---|---|---|
  | 0.1 dB | degree 2 (7 ops) | degree 2 (8 ops) |
  | 0.01 dB | degree 3 (9 ops) | degree 4 (12 ops) |
  | 0.001 dB | degree 3 (9 ops) | degree 5 (14 ops) |

- **Bench, buffer ns:**
  - `std::pow(10, dB/20)`: 1.54. `db_to_gain_3`: 0.25 (6×).
  - `20·log10`: 1.98. `gain_to_db_4`: 0.59 (3.4×).

## Pin the ends: continuity and exact unity for free

- **Floor reduction steps at every 6.02 dB unless P(1) = 2·P(0).** A free
  Chebyshev fit misses:
  - degree 2: **a 0.031 dB step** at every 6.02 dB (zoom panel in
    `db_gain_3_f32_gain_error.png`)
  - degree 3: a 3.7e-4 dB step, and the gain is *non-monotonic* there
    (3e-4 dB drop)
- **The free fit isn't exactly 1 at 0 dB** (degree 3: −0.00087 dB). That's
  inaudible, but a "bypassed" processor no longer nulls.
- **`fit.minimax_endpoints`** writes P = chord + (x−a)(x−b)·Q, so P(0) = 1 and
  P(1) = 2 *exactly in floating point* (the (f − 1) factor is exactly 0
  there). Q is a Remez fit weighted for P's relative error. Result:
  - no step, monotonic, unity exact
  - max error equal to or *better than* the free fit from degree 3 up (8.95e-4
    vs 8.66e-4 dB at 3; 2.9e-5 vs 3.0e-5 at 4)
- **The same trick for log2:** pinning u·P(u) at both ends of the fold range
  makes the 1.5 fold seamless. A free fit steps 0.0021 dB at x = 1.5·2^k
  (degree 4).
- **A pinned degree-1 fit is exactly Schraudolph's chord**, so the bit trick
  is the bottom row of this table, not a separate idea.

## The bit tricks: fine for meters, wrong for gains

- **Schraudolph (dB → gain) and Mitchell (gain → dB):** ±0.52 dB, exact at
  0 dB, continuous, about 2 ops.
- **The tuned offsets halve that, but break unity:**
  - Schraudolph tuned gives **0.978 (−0.19 dB) at 0 dB**: a gain that should
    be untouched isn't.
  - Mitchell tuned reads +0.26 dB at gain 1.
  - Fine for a meter, wrong for anything that must pass audio at unity.

## Ramps: don't evaluate per sample

A fade linear in dB is geometric in gain: g[n+1] = g[n]·r, one multiply.

| method | −60→0 dB, 1 s | 0→−60 dB, 10 s | −96→0 dB, 60 s |
|---|---|---|---|
| exp2 pinned 3, every sample | 9.0e-4 | 9.0e-4 | 9.0e-4 |
| g ·= r, f32 | 0.015 | **0.11** | **0.58** |
| g ·= r, f64 state | 5e-7 | 5e-7 | 5e-7 |
| g ·= r, f32, re-anchored every 64 | 9.2e-4 | 9.4e-4 | 9.3e-4 |

- **f32 drift is linear in time.** It's mostly r's own rounding compounded
  n times, so a long fade ends up 0.5 dB off.
- **Two fixes, both cheap:**
  - keep the state in f64
  - re-anchor once per block with the polynomial (1/64 of an exp2 per
    sample)

## Compressor: both conversions in one loop

Threshold −24 dB, ratio 4:1, hard knee, static curve. Output gain error vs
exact, in dB:

| detector + gain | below threshold | gain == 1 below | above threshold |
|---|---|---|---|
| std (log10 + pow) | 0 | yes | 2.5e-6 |
| Mitchell + Schraudolph | 0 | yes | **0.91** |
| both tuned offsets | 0.19 | **no** | 0.44 |
| Mitchell + exact pow | 0 | yes | 0.39 |
| exact log10 + Schraudolph | 0 | yes | 0.52 |
| free fits (4, 3) | 8.7e-4 | **no** | 1.4e-3 |
| pinned fits (4, 3) | 0 | yes | 1.9e-3 |

- **A matched bit-trick pair doesn't cancel.** Mitchell and Schraudolph are
  exact inverses (S(M(x)) = x), but a compressor applies a gain that's a
  *function* of the level. Errors add, roughly (1 − 1/R) × detector error +
  gain error, so the pair is 0.91 dB off: worse than either alone. (I had
  guessed they'd cancel; measured, they don't.)
- **Unity below threshold** needs the gain path's P(0) = 1 exactly. Pinned
  fits and the untuned bit trick have it; free fits and tuned offsets don't.

## Extended range (tiny and extreme gains)

f32 gains run from −897 dB (the smallest subnormal) through −758.6 dB
(FLT_MIN) to +770.6 dB (FLT_MAX).

- **dB → gain, plain forms:**
  - **−inf dB (a fader at the bottom) gives NaN** from the polynomial,
    because floor(−inf) − (−inf) is NaN.
  - **The bit tricks give −1**: a gain with the polarity flipped.
  - Below −758 dB the exponent field wraps to garbage (−1e37). Above +770 dB
    they give garbage or −0, not inf.
- **dB → gain, safe form** (`eval_gain_safe`, `db_to_gain_N_safe`): clamp
  dB·K to [−151, 128], then build 2^n in two halves.
  - Right everywhere: −inf → 0, subnormal gains to 0.01%, overflow → inf.
  - Cost: +2 ops; about 2× the buffer time (0.53 vs 0.25 ns).
- **gain → dB, plain forms:**
  - Subnormal gains read e = −127: 1e-40 comes out −764 dB instead of −800.
  - Silence (0) gives −765 dB rather than −inf. Bench: 132 dB max error over
    subnormals.
- **gain → dB, safe form** (`eval_db_safe`, `gain_to_db_N_safe`): scale
  subnormals by 2²³ first, and select 0 → −inf and inf → inf.
  - Right everywhere: 0.0016 dB max over every positive float.
  - Cost: about 1.5× the buffer time (0.86 vs 0.59 ns).
- **Even `std::pow(10, dB/20)` loses accuracy at extreme dB.** Bench: 60 ULP
  (3e-5 dB) near ±700 dB, because dB/20 rounds and pow amplifies it. The
  same happens in our dB·K. It's inaudible.
- **The audio take:**
  - Nothing below about −150 dB is audible, so the polynomials' accuracy
    never matters out there.
  - **But −inf dB and 0 gain do happen** (faders, silence, gates), and the
    plain forms turn them into NaN, −1 or −765 dB.
  - Use the safe forms at API boundaries, or clamp dB to a floor (e.g.
    −150 dB → 0) before converting.

## Power domain (aside)

- RMS in dB is 10·log10(mean square) = 3.0103·log2(ms), so there's no sqrt.
- The same log2 error costs half as many dB as 20·log10(rms) would.
