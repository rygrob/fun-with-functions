# Why std::sin keeps up in the feedback loop

2026-10-23. Disassembled from the running `adc_simple_loops` (lldb,
`disassemble -n sinf`, Apple libsystem_m on the M2 Pro), so this is the real
code, not a guess. Cycle counts are indicative: roughly 4 cycles per
dependent fmul / fma / fcvt, 3 for frintn; real per-op latencies vary.

## What the compiler does in each loop (`bench/simple/loops.cpp`)

From `objdump -d` of the same binary; everything inlines into `main`.

- **Buffer.** Taylor 13 is inlined and vectorized: per iteration, four
  NEON registers of four lanes, 16 samples. Each register: `fmul.4s` for
  x², six `fmla.4s` (Horner), `fmul.4s` by x. std::sin stays a scalar `bl
  sinf` per sample.
- **Sample.** The same 8 ops per sample, scalar (`fmul`, 6 `fmadd`, `fmul`),
  one sample per iteration, store, escape. Nothing in sample i + 1 needs
  sample i, so the out-of-order core overlaps several: throughput-bound,
  8 FP ops over 4 FP pipes ≈ 2 cycles ≈ 0.6 ns (measured 0.62).
  std::sin overlaps the same way, it just has more instructions per sample
  (call, sign/magnitude tests, double conversions): 1.84 ns.
- **Feedback.** Same 8 ops, plus `fmadd s0, y, 0, x` for `x + 0 * y` (FMA
  fused it). Every op waits on the last: ≈ 9 × 4 = 36 cycles ≈ 10.3 ns
  (measured 10.47). std::sin's call sits in the same chain: 10.50.

## Apple's sinf, sketched

Branch on |x| (integer compares on the bits, thresholds in its constant pool):

| |x| | path |
|---|---|
| < 2⁻¹² | x, tiny correction |
| < π/4 | no reduction: degree-7 odd polynomial in **double** |
| < 120 | one-step reduction by π/2 in double, then sin or cos polynomial |
| ≥ 120 | table-driven reduction (not hit on [−π, π]) |

Main path (π/4 ≤ |x| < 120), in double:

```
x  = double(x)                       fcvt
n  = round(x * 2/π)                  fmul, frintn   (and fcvtns for the int)
r  = x - n * π/2                     fmsub          (one constant: f32 input, f64 work)
if n odd:  cos(r) = 1 + r²((c2 + c4 r²) + r⁴(c6 + c8 r²))   degree 8, Estrin
else:      sin(r) = r + c3 r³ + r⁵(c5 + c7 r²)              degree 7, Estrin
flip sign from n's bit 1                eor
return float(result)                    fcvt
```

Constants read from memory: 2/π = 0.6366…, π/2 = 1.5708…, cos coefficients
2.44e-5, −1.389e-3, 0.04167, −0.5, sin coefficients −1.95e-4, 8.33e-3,
−0.16667 (Taylor-like, slightly tuned).

### Its dependent chain (sin branch)

fcvt → fmul → frintn → fmsub → r² → r³ → r⁵ → fmadd → eor → fcvt:
about 10 ops, ≈ 35 cycles. Below π/4 it skips the reduction: ≈ 22. Our
Taylor 13: x + 0·y, x², six Horner fmadds, × x: 9 ops, ≈ 36. Same ballpark,
hence the same 10.5 ns.

### Why it's short despite the extra work

- **Range reduction trades degree for a few ops.** On [−π/4, π/4] a degree-7
  polynomial is plenty (in double), against our degree 13 on [−π, π].
- **Estrin, not Horner.** The polynomial is split so r², r³, r⁵ and the
  partial sums run in parallel: 4 dependent steps (r², r³, r⁵, last fmadd)
  where Horner in r² would take 5 (r², three fmadds, × r).
- **The branches are off the chain.** The magnitude tests and the quadrant
  test (sin or cos branch) are predicted, and the core speculates past them.
  The data chain is all that's left. Sorted inputs keep them predictable;
  random inputs mispredict (sample loop: 5.75 ns against 1.84), and in
  feedback each miss would add a pipeline flush (~13+ cycles) to the chain.
- **Measured, random inputs** (2026-10-23, loops.cpp without the sort, on
  power, three runs each): feedback std::sin 14.95–15.15 ns, Taylor
  10.32–10.43, turns 10.04–10.13; sorted, all three 10.0–10.6. Sample loop
  std::sin 5.73–5.83 against 1.84; buffer 6.0 against 1.76. Taylor and turns
  don't move: no branches.
- **Double precision is cheap.** fcvt each way costs ~2 × 4 cycles, and buys
  a one-constant reduction and short polynomial without float error worries.

So the lesson for the feedback case: what counts is the dependent chain,
and libm's is short because of range reduction plus Estrin. Our turns form
reduces too; Estrin on Taylor 13 brings ours to 7.2 ns
(`notes/latency-horner-estrin.md`).
