# tanh in a feedback loop (2026-10-21)

Background, not (yet) on the slides; a thread to revisit for the tanh
section. Follows `latency-horner-estrin.md` (the three loops, Horner vs
Estrin). The point: tanh is the transcendental most likely to sit *on* a
feedback path (a ladder filter, tape or hysteresis models, a feedback
saturator), and there the shipped sigmoid barely beats std::tanh. A tanh
designed for the dependent case is 1.5–2.2× faster, and the best
all-rounder is a minimax rational.

Code in `tanh-feedback/`:

- `tanh_feedback.cpp`: every candidate in the three loops of
  `bench/simple/loops.cpp`, plus max abs error over every float in [0, 9]
  (build line in its header).
- `fit_pw.py`: the piecewise segments (Chebyshev, near-minimax), writes
  `pw_tanh_table.inc`.
- `remez_rational.py`: a hand-rolled rational Remez exchange, writes
  `rational_coeffs.inc`.

M2 Pro, Apple clang 21 -O3, fast-math off, x in [−5, 5] sorted, 65,536
samples, median of 100 passes; two launches agreed to ~1%.

## Results

| tanh | vectorizable | independent | dependent | max abs error, [0, 9] | max y |
|------|---:|---:|---:|---:|---:|
| std::tanh | 2.00 | 2.01 | 16.5 | | |
| poly + clamp 9 | 0.19 (10.3×) | 0.88 (2.3×) | 9.8 (1.7×) | −42.1 dB | |
| sigmoid 9 (shipped) | 0.31 (6.4×) | 1.10 (1.8×) | 15.6 (1.1×) | −121.4 dB | **1** |
| sigmoid 9, Estrin R | 0.29 (6.8×) | 1.16 (1.7×) | 14.6 (1.1×) | −121.3 dB | |
| Padé [7/6] + clamp | 0.29 (6.9×) | 1.30 (1.5×) | 11.0 (1.5×) | −80.3 dB | |
| piecewise 18 × 5 | 1.07 (1.9×) | 1.09 (1.8×) | **7.7 (2.2×)** | −125.3 dB | 0.99999994 |
| minimax [7/6], Estrin | **0.25 (8.0×)** | **0.99 (2.0×)** | 9.5 (1.7×) | −105.2 dB | 1.0000054 |
| minimax [9/6], Estrin | 0.27 (7.4×) | 1.06 (1.9×) | 10.8 (1.5×) | −122.3 dB | 1.0000006 |
| minimax [9/8], Estrin | 0.29 (6.9×) | 1.13 (1.8×) | 10.5 (1.6×) | **−130.2 dB** | 1.00000012 |

ns per sample (speedup vs std::tanh). Max y over every float (stride 7).

## Why the sigmoid is slow in feedback

Its chain: clamp → x² → 4 Horner fma → x·R → s² + 1 → **sqrt → divide**.
About 55 cycles at ~3.5 GHz; the sqrt and divide back to back are ~20 of
them, with nothing to overlap. Estrin on R (5 coefficients) saves one fma
level, 4 cycles: 15.6 → 14.6 ns. std::tanh's own latency is ~59 cycles,
so in feedback the sigmoid is 1.1×.

## Designed for the dependent case

**Piecewise table** (`pw_tanh`): |x| picks one of 18 segments of width
1/2 on [0, 9]; each is degree 5 in t = |x| − center, by Estrin.

```cpp
alignas(64) inline constexpr float pw_tanh_table[18][7] = { /* center, c0..c5 */ };

inline auto pw_tanh(float x) -> float
{
    const auto a = std::fmin(std::fabs(x), 8.9999990f);
    const auto& c = pw_tanh_table[static_cast<int>(a + a)];
    const auto t = a - c[0];
    const auto t2 = t * t;
    const auto p0 = c[1] + c[2] * t;
    const auto p1 = c[3] + c[4] * t;
    const auto p2 = c[5] + c[6] * t;
    return std::copysign(p0 + t2 * (p1 + t2 * p2), x);
}
```

No sqrt, no divide; in scalar code the table is one L1 load (504 bytes,
8 lines); narrow segments need only degree 5, three fma levels. ~27
cycles a sample with the feedback fma: the latency winner, and −125 dB.
But vectorized the load is a gather: 3.5× slower than the sigmoid. The
Lookup tables slide's lesson, turned around.

Width / degree for the segments (f64, max abs):

| h | d3 | d4 | d5 | d6 | d7 |
|---|---:|---:|---:|---:|---:|
| 1 | −58.6 | −85.7 | −92.1 | −110.9 | −129.1 |
| 1/2 | −83.2 | −106.4 | **−125.8** | −153.4 | −169.3 |
| 1/4 | −105.8 | −133.0 | −161.6 | −189.9 | −219.2 |

**Minimax rational**: tanh x ≈ x P(s) / Q(s), s = x², clamped to [−9, 9],
P and Q by Estrin side by side, one divide.

```cpp
inline auto rat_tanh_9_8(float x) -> float
{
    const auto& p = rat_p_9_8;  // 0.999999906, 0.133731969, 0.00348658101, 2.04718171e-05, 1.31842075e-08
    const auto& q = rat_q_9_8;  // 1, 0.467064938, 0.0258419757, 0.000327138399, 7.70265634e-07
    const auto xc = std::fmin(std::fmax(x, -9.0f), 9.0f);
    const auto s = xc * xc;
    const auto s2 = s * s;
    const auto P = (p[0] + p[1] * s) + s2 * ((p[2] + p[3] * s) + p[4] * s2);
    const auto Q = (q[0] + q[1] * s) + s2 * ((q[2] + q[3] * s) + q[4] * s2);
    return xc * P / Q;
}
```

The fit: at the m + n + 2 reference points solve
x P(s) − f Q(s) − (−1)ⁱ E Q_prev(s) = 0 (the error term's Q from the last
iterate, so it stays linear), iterate to a fixed point, exchange to the
error's alternating extrema; powers in s/81 for conditioning. It levels
exactly (levelled E equals the max over a 220k-point grid):

| [num/den] | [7/6] | [9/6] | [9/8] | [11/8] | [11/10] |
|---|---:|---:|---:|---:|---:|
| f64 minimax | −105.6 | −126.1 | −154.7 | −178.8 | −209.9 dB |

[9/8] in float lands at −130.2 dB (float's own rounding).

## Takeaways

- **Minimax beats Padé by 25 dB at the same degree and cost** ([7/6]:
  −105 vs −80 dB). Padé spends everything at x = 0; Remez spreads it over
  [0, 9]. Taylor vs Chebyshev again, for rationals.
- **Minimax [9/8] matches or beats the shipped sigmoid in every loop**:
  9 dB more accurate, about as fast vectorized and one at a time, 1.5×
  faster in feedback (one divide instead of sqrt + divide).
- **Its catch: it can overshoot 1 by one float step** (1.00000012). The
  sigmoid can't exceed 1 by construction (the scoreboard's Max |y|
  column). Fix with an output clamp (2–3 cycles on the chain) or a fit
  constrained to stay ≤ 1.
- **Which tanh depends on the loop**: vectorizable → sigmoid or minimax
  rational; independent → about even; dependent → piecewise table +
  Estrin (scalar), minimax rational (if it must also vectorize, or for
  `vsl` lanes across voices, where the table becomes a gather).

## Threads to pull

- Minimax [9/8] with an output clamp (or a constrained fit) as a row on the
  tanh Improvements slide; that means moving the rational Remez into
  `examples/common/` and cppgen so the numbers come from the repo.
- Measure a real feedback loop (a ladder filter with tanh in it) rather
  than the synthetic chain.
- A 4-lane NEON version: piecewise (gather) vs rational across voices.
