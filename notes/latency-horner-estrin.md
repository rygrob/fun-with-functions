# Latency vs throughput: Horner, Estrin, and the three loops (2026-10-21)

Background, not on the slides. How fast a polynomial
is depends on the loop around it as much as on the polynomial. Two
programs reproduce everything here:

- `bench/simple/loops.cpp` (`adc_simple_loops`, the Three loops slide): the three
  inner loops, for std::sin, Taylor 13 and turns 13.
- `bench/simple/estrin.cpp` (`adc_simple_estrin`): Horner vs Estrin for
  the shipped `cheb_sin_13` / `cheb_turns_sin_13`, same coefficients, in
  all three loops, plus max ULP over every float.

M2 Pro P-core (~3.5 GHz), Apple clang 21 -O3, fast-math off, sorted
inputs, 65,536 samples, median of 100 passes.

## Three loops

| loop | how | what it measures |
|------|-----|------------------|
| buffer, vectorizable | `escape(out.data())` once per pass | SIMD throughput: clang runs 4 lanes per NEON instruction and unrolls 4×, 16 samples per iteration |
| per sample, independent | `escape(&out[i])` after every sample | scalar throughput: no vectorizing, but the out-of-order core overlaps calls that don't depend on each other. Same as `#pragma clang loop vectorize(disable)`. The scoreboards quote this one |
| per sample, dependent | `y = f(in[i] + 0.0f * y)`, `escape(&y)` once | latency: each call waits for the last. Without fast-math `0 * y` can't be folded (NaN, infinity, −0); with `-ffast-math` it vanishes and you silently measure the independent loop |

ns per sample (`adc_simple_loops`):

| | vectorizable | independent | dependent |
|---|---:|---:|---:|
| std::sin | 1.8 | 1.85 | 10.6 |
| Taylor 13 | 0.17 | 0.62 | 10.5 |
| turns 13 | 0.23 | 0.81 | 10.3 |

So: about 11× / 3× / 1× for Taylor against std::sin. Vectorizing is worth
almost exactly 4× to the polynomial (the lanes); std::sin is a scalar
library call and never vectorizes. In the dependent loop everything costs
the same: it's the length of the longest dependent chain, and Horner's is
about as long as `sinf`'s.

## Throughput vs latency: count different things

- **Throughput** is total instructions over the vector pipes (4 on the
  M2's P-cores). Degree 13 in Horner form: x², 6 fma, × x = 8 ops per 4
  samples, 2 per sample, ~0.5 cycle: the 0.16 ns we measure. At the
  hardware limit, not below it.
- **Latency** is the longest chain of dependent ops, at ~4 cycles per
  fmul / fma (fsub ~3). Horner is a pure chain, each step waiting on the
  last.

## Estrin's scheme

Evaluate P(s), s = x², as a tree instead of a chain: pair the
coefficients, combine the pairs with s², the halves with s⁴. The pairs
don't depend on each other, so they run side by side while s² and s⁴ are
formed. More total work (s², s⁴, one more fma), shorter chain.

Taylor degree 13 in both forms:

```cpp
// Horner: one chain, 6 dependent fma after x2
const auto x2 = x * x;
auto p = T(1.0 / 6227020800);
p = p * x2 + T(-1.0 / 39916800);
p = p * x2 + T(1.0 / 362880);
p = p * x2 + T(-1.0 / 5040);
p = p * x2 + T(1.0 / 120);
p = p * x2 + T(-1.0 / 6);
p = p * x2 + T(1.0);
return x * p;

// Estrin: x2 -> x4 -> x8 -> one fma -> x *
const auto x2 = x * x;
const auto x4 = x2 * x2;
const auto x8 = x4 * x4;
const auto a = T(1.0) + T(-1.0 / 6) * x2;                  //  1      - x^2/3!
const auto b = T(1.0 / 120) + T(-1.0 / 5040) * x2;         //  x^4/5! - x^6/7!   (/ x^4)
const auto c = T(1.0 / 362880) + T(-1.0 / 39916800) * x2;  //  x^8/9! - x^10/11! (/ x^8)
const auto lo = a + b * x4;
const auto hi = c + T(1.0 / 6227020800) * x4;              //  + x^12/13!        (/ x^8)
return x * (lo + hi * x8);
```

(Matches the generated `taylor_sin_13` to 3.9e-7 over 10⁶ points: a few
roundings, different order.)

ns per sample (`adc_simple_estrin`; "fold" is Estrin with the final
multiply folded into the halves, x·lo + (x·hi)·s⁴):

| loop | std::sin | rad Horner | rad Estrin | rad fold | turns Horner | turns Estrin | turns fold |
|------|---:|---:|---:|---:|---:|---:|---:|
| vector | 1.76 | **0.16** | 0.19 | 0.21 | **0.23** | 0.26 | 0.28 |
| scalar | 1.75 | **0.62** | 0.73 | 0.80 | **0.81** | 0.94 | 1.02 |
| chain  | 10.6 | 10.4 | **7.2** | 7.2 | 10.1 | **7.9** | 8.1 |

Max ULP, turns, every float in [0, 1]: Horner 4.969, Estrin 5.234, fold
4.840.

Taylor 13 itself (2026-11-05, now in `adc_simple_estrin`; median of five
launches, `slides/bench/simple-estrin.txt`): chain 10.75 Horner, 7.40
Estrin, 31% off, the number on the Three loops slide. Measured separately
with `loops.cpp`'s exact loops, seven launches:

| ns per sample | buffer | sample | feedback |
|---|---:|---:|---:|
| sorted std::sin | 1.733 | 1.841 | 10.63 |
| sorted Horner | 0.163 | 0.616 | 10.53 |
| sorted Estrin | 0.196 | 0.732 | 7.27 |
| random std::sin | 5.941 | 5.774 | 15.50 |
| random Horner | 0.164 | 0.614 | 10.55 |
| random Estrin | 0.196 | 0.732 | 7.24 |

Same max abs (2.13e-5 Horner, 2.14e-5 Estrin); max ULP for |x| <= 1.5,
2.21 Horner, 3.20 Estrin.

The cycle counts predict the chain row (+4 cycles for the chain's own fma):

| form | critical path | cycles | predicted | measured |
|------|---------------|-------:|----------:|---------:|
| radians Horner | s, 6 fma, × x | 32 + 4 | 10.3 ns | 10.4 |
| radians Estrin | s, s², s⁴, fma, × x | 20 + 4 | 6.9 ns | 7.2 |
| turns Horner | u, s, 5 fma, × factors | 31 + 4 | 10.0 ns | 10.1 |
| turns Estrin | u, s, s², s⁴, fma, × factors | 23 + 4 | 7.7 ns | 7.9 |

- Estrin is ~30% faster where the result feeds back, ~20% slower
  everywhere else. Rule: Horner by default, Estrin on a feedback path.
- In a feedback loop Horner gains nothing over libm (std::sin's latency
  is ~37 cycles); Estrin gains ~1.45×.
- Folding the final multiply in doesn't shorten the chain: Estrin's
  halves are already balanced (s⁴ and hi both ready at ~12 cycles), so
  the multiply lengthens hi by what it saves at the end. ~20 cycles is
  the floor for this shape. Fold does round slightly better (4.84 ULP).
- The other latency lever is degree: each Horner step is 4 cycles.

## Which loop is per-sample DSP?

Per sample isn't the question; whether the function's input depends on
its own previous output is. The core's out-of-order window (hundreds of
instructions) overlaps any work that doesn't.

From shipped plug-in code:

| code | transcendental | on the sample-to-sample chain? | loop |
|------|----------------|------|------|
| sine oscillators, LFO shapes | sin of the phase | no: only the phase is carried | independent |
| state-variable filter | tan of the cutoff | no: g comes from the cutoff, not the filter's output | independent |
| biquad coefficients | sin, cos, pow | no | independent |
| compressor, feedforward | log10 to dB, exp2 from dB | no | independent |
| compressor, feedback | log10 of the output, exp2 for the gain | **yes** | dependent |
| hysteresis, tanh inside a filter loop, FM feedback | tanh / coth of the state | **yes** | dependent |
| waveshapers on the input | of the input x | no, unless inside a feedback loop | independent |

A real feedback path also carries the detector, the gain computer and the
`x * gain`, so the transcendentals are part of the chain, not all of it.

## SIMD across channels

`vsl::float32x4` / `float64x2` put channels or voices in the lanes. A
vector instruction costs about what its scalar version does, so per
channel the cost is the independent number over the lane count: Taylor
0.62 / 4 ≈ 0.16 ns, the same as the vectorized loop (the same hardware,
filled with channels instead of consecutive samples).

It also rescues the dependent loop: four voices' chains run side by side
in the lanes, ~10 ns per step for all four, ~2.6 ns each. For recursive
filters and feedback compressors, SIMD across voices is the only
parallelism there is, and it stacks with Estrin's shorter chain.

Conditions: fill the lanes (stereo in float32x4 is 2×); the function
needs a vector implementation (std::sin has none, hence vsl's own);
branches become selects (both paths' work); tables need a gather (one
load per lane); x86 AVX2 is 8 floats wide.

## tanh: designing for the dependent case

Continued in `tanh-feedback.md`: in feedback the shipped sigmoid is 1.1×
std::tanh (sqrt + divide on the chain); a piecewise table + Estrin is
2.2× at −125 dB, and a minimax rational [9/8] is 1.6× at −130 dB while
matching the sigmoid vectorized.

## Threads to pull

- A real feedback compressor (log2 → gain computer → exp2) with Horner
  vs Estrin inside, rather than the synthetic chain.
- A 4-lane NEON version of the independent and dependent loops, to
  measure the across-channels claim rather than derive it.
- Lower degree on a feedback path.
