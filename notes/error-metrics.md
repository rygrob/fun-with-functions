# Measuring approximation error

Reference notes for the "Toolkit → Error" part of the talk. The numbers below
come from `examples/00-taylor-series/taylor_sin.py` (Taylor sin on [-π, π]),
so they can be reproduced with `--compare`, `-d N` and `--eval f32|f64`.

---

## 1. Bits of precision: absolute vs. relative

Both readouts use the same conversion:

```
bits = -log2(error)
```

An error of 2⁻ᵇ means about *b* binary digits of the answer are right. The two
metrics count **different kinds of digit**.

### Absolute error → fixed-point bits

```
abs_error = |approx − truth|
```

- Measured in the output's own units. For sin (|y| ≤ 1), 2⁻ᵇ means the answer
  is right to *b* bits **after the binary point**, wherever you are in the
  range.
- **The audio reading:** each bit is about 6.02 dB. Absolute bits × 6 dB is
  roughly the worst-case error level relative to full scale. Absolute error is
  "what you'll hear".
- Example: degree-7 Taylor has max absolute error 0.075 ≈ 2⁻³·⁷, so 3.7 bits,
  or about −22 dBFS of error at its worst point.
- f32 ceiling: about 24 bits near |y| = 1, more for small outputs.

### Relative error → significant (floating-point) bits

```
rel_error = |approx − truth| / |truth|
```

- Error measured against the size of the answer. That is how floats store
  numbers: a fixed count of **significant** bits, wherever the binary point
  sits.
- 2⁻ᵇ means the leading *b* mantissa bits are right.
- f32 has 24 significant bits (23 stored plus the implicit 1). Correct rounding
  gives relative error ≤ 2⁻²⁴, so **24 bits is the ceiling**.
- It matters when outputs are small (quiet signals, values near a function's
  zeros) or feed into `log`, division, or long chains of multiplies, where
  relative error carries through.

### Why they disagree

Near a zero of the function (sin at ±π), the truth → 0. A modest absolute
error divided by a tiny truth is a huge relative error.

| degree 7 | value | bits |
|---|---|---|
| max abs error | 7.5e-2 | 3.7 |
| max rel error | 8.6e+5 | 0 (not even the sign is right) |

The bits readout floors at 0. An error ≥ 1 means no correct bits.

**Talk framing:** Taylor looks *mediocre* by absolute error and *catastrophic*
by relative error. A good approximation (Chebyshev with the zeros factored
out) can be good by **both**.

### ULP error (the libm unit)

A **ULP** (unit in the last place) is the gap between two neighbouring f32
values at a given size: `2^(exponent − 23)`. It is 2⁻²³ ≈ 1.2e-7 just above
1.0, and halves at every power of 2 going down.

```
ulp_error = |approx − truth| / ulp_f32(truth)       (truth exact, here f64)
```

- **≤ 0.5 ULP:** correctly rounded. Nothing can do better, because the exact
  value sits between two f32s.
- **< 1 ULP:** faithfully rounded. One of the two neighbouring f32s.
- **N ULP:** about log₂(N) bits of the mantissa are wrong.
- Relative bits ≈ `24 − log2(ULP error)`, give or take a bit depending on
  where the value sits between powers of 2.
- This is the scale libm documentation uses ("max error 0.5 ulp", "1 ulp").
- We use the **fractional** form, not the integer "number of f32s in between"
  distance. The integer form scores both 0.49 and 0.01 ULP as 0, and plateaus
  at about 2³¹ when the sign is wrong.
- `ulp_f32(t)` comes from the power-of-2 range of the *exact* value (via
  `frexp`). `np.spacing(f32(t))` gives the wrong gap when `t` rounds up across
  a power of 2.
- The ULP plots step down every time |sin| crosses a power of 2, because the
  gap halves. That makes a good visual for explaining ULPs.

---

## 2. Evaluating in f32 vs. f64

`--eval` sets the precision of **the polynomial arithmetic**: the coefficients
are stored as that type, and every Horner multiply and add rounds to it. The
inputs are the same f32 `x` values either way. The truth is always f64 `sin`.

### Two separate sources of error

1. **Approximation (truncation) error.** The polynomial is not sin. This is
   the math, and it depends only on the method and degree.
2. **Evaluation (rounding) error.** Each operation rounds its result, and the
   coefficients themselves are rounded. In f32, 1/6 is stored as
   `1.66666672e-01`.

The total is roughly the sum. **f64 evaluation pushes (2) down by about 2⁻²⁹,
so it shows the pure approximation error.** f32 evaluation is what ships.

### How to read the comparison

- **f32 ≈ f64:** truncation dominates, so more degree (or a better method)
  will help.
- **f32 ≫ f64:** you're at the **rounding floor**. More terms won't help. To
  improve, change how you evaluate: term order or Horner variant, FMA,
  splitting awkward constants (the π_major + π_minor trick in vsl), or
  coefficients optimized for f32 instead of rounded from exact values.

### Taylor numbers

| degree | SINAD f32 | SINAD f64 | max abs f32 | max abs f64 | max ULP, abs(x) ≤ 1, f32 | f64 |
|---|---|---|---|---|---|---|
| 9  | 53.8 dB  | 53.8 dB  | 6.93e-3 | 6.93e-3 | 1.38 | 0.42 |
| 13 | 105.4 dB | 105.4 dB | 2.12e-5 | 2.11e-5 | 1.15 | 0.00 |
| 15 | 133.1 dB | 134.4 dB | 1.02e-6 | 7.73e-7 | 1.15 | 0.00 |
| 17 | 140.2 dB | 146.5 dB | 3.53e-7 | 2.24e-8 | 1.15 | 0.00 |

- Up to about degree 13 the two match: truncation is the whole story.
- At degree 17, f64 has about 16× less absolute error and beats the f32 sin
  reference (145 dB). f32 stalls at 140 dB because of its own rounding.
- Near the origin the math is essentially exact (f64 shows 0.00 ULP), but f32
  evaluation sits at about 1.15 ULP. That's pure rounding.
- Caveat: in f64 mode the output is not rounded to f32 at the end, so f64
  numbers can exceed the "f32(sin)" reference. That is expected, because it is
  showing the math, not a shippable f32 result.

**Talk framing:** "Your error budget has two line items. f64 tells you what
the math costs; f32 tells you what the hardware costs. Knowing which one
dominates tells you whether to change the polynomial or the code."

---

## 3. SINAD / harmonic analysis

The time-domain metrics answer "how wrong is each sample?" For an oscillator
the audio question is "**what does the error sound like?**", meaning how much
energy goes to harmonics and noise, and where.

### Method

1. Take **one cycle**: N = 65,536 evenly spaced phases over [-π, π), as f32
   inputs.
2. Evaluate the approximation and take an FFT. With exactly one cycle in the
   buffer, **bin h is harmonic h**. There is no windowing and no leakage,
   because the signal is exactly periodic in the buffer.
3. `rfft` folds the negative frequencies in, so double the power of every bin
   except DC and Nyquist. Without this, SINAD is off by about 3 dB.
4. Compute:
   ```
   SINAD = 10·log10( P_fundamental / P_everything_else )     (DC counts as error)
   ENOB  = (SINAD − 1.76) / 6.02
   dBc_h = 10·log10( P_h / P_fundamental )
   ```
   ENOB is "effective number of bits", the bits-of-precision idea again, this
   time from total error power instead of the worst sample.
5. Reference: sin rounded to f32 gives **145.4 dB (23.9 ENOB)**. That is the
   ceiling for any f32 sine.

### Why one cycle is the same as a real oscillator

The approximation's output depends only on phase. An oscillator at K cycles
per N samples, with K coprime to N, visits all the same N phases, just in a
different order. The energy at harmonic h lands in bin `h·K mod N` instead of
bin h. The bins get shuffled and the energies are identical, so **SINAD is
identical**.

Checked: the 48 kHz / 996.8 Hz oscillator (K = 1361) gives exactly the same
SINAD as the one-cycle analysis.

It also means **every error, including f32 rounding, is periodic in phase**,
so all of it shows up as harmonics. Nothing is truly random noise here. SINAD
is effectively THD+N, where the "N" is deterministic.

Cross-check: projecting the output onto sin/cos in the time domain gives the
same SINAD for degree 3 and degree 7. At degree 15 the two differ by 0.2 dB,
because the projection treats f32 rounding of the phase input as exact. The
FFT version counts it as error, the same way the f32 reference does.

### Two views in the script

- **`_spectrum.png`**: harmonic number on a log axis, binned max. This is the
  clean analytical view, and the slope is easy to read.
- **`_oscillator.png`**: a real ~1 kHz oscillator at 48 kHz, showing 3 ms of
  the waveform plus the spectrum in Hz. Harmonics above Nyquist alias back
  down and fill the whole band. This is the one to show an audio audience,
  because it's what they'd see on an analyzer.

### Taylor results

| degree | SINAD | ENOB | worst harmonic |
|---|---|---|---|
| 3  | −1.5 dB  | —    | error carries more power than the fundamental |
| 5  | 14.8 dB  | 2.2  | |
| 7  | 32.3 dB  | 5.1  | h3 at −41.9 dBc |
| 9  | 53.8 dB  | 8.7  | |
| 15 | 133.1 dB | 21.8 | h5 at −146.4 dBc |

### The story the spectrum tells

- **The harmonics fall off at about 6 dB per octave**, the signature of a
  **jump**. Taylor's value at −π and +π differs (+0.075 vs −0.075 at
  degree 7), so the oscillator jumps once per cycle. You can see the step in
  the waveform panel. The spectrum looks like a sawtooth's: harmonics all the
  way up, aliasing everywhere.
- A continuous waveform with a kink falls off at 12 dB per octave; one that
  is smooth through the wrap falls off much faster. **Pinning the zeros at
  ±π removes the jump.** That's an argument for factoring out the zeros you
  can see right on a spectrum slide.
- **Both odd and even harmonics** show up at similar levels (degree 7: h2
  −43.2, h3 −41.9, h4 −42.0 dBc), again like a sawtooth. Being an odd
  function only removes the cosine components. Getting rid of the even
  harmonics would need half-wave symmetry, g(x + π) = −g(x), and the jump
  breaks that.
- Even degree 15 is about 12 dB short of the 145 dB f32 ceiling.

### Caveats to remember

- This measures the **function** only. Phase-accumulator precision (f32
  phase drifting at long run times) is a separate topic, and the f64 phase
  accumulator here avoids it on purpose.
- The SINAD of a discontinuous approximation doesn't depend on f0, but *where*
  the aliased harmonics land does. Pick a prime K so they don't pile into a
  few bins.
- SINAD weights error by power (an RMS average). A single bad sample can
  dominate max-ULP while barely moving SINAD, and vice versa. That is why we
  show both.
