# Bench

C++ timing and exhaustive ULP error for the sin approximations, generated
from the Python examples.

```sh
python bench/generate.py                 # examples -> generated/*_approx.hpp
cmake -S bench -B bench/build && cmake --build bench/build
bench/build/adc_bench                    # timing + ULP, f32 and f64
bench/build/adc_bench timing f32
bench/build/adc_bench ulp f32 --stride 64   # every 64th input, about 1 s
bench/build/adc_bench ulp f32 --threads 2   # the exhaustive checks use half the cores by default
bench/build/adc_bench ulp f32 --domain turns --dump-worst 1000 --dump-dir /tmp/worst
bench/build/adc_bench tan                # tan(pi x) cutoffs: timing f32/f64 + exhaustive f32 error, ~17 s
bench/build/adc_bench tan ulp f32 --stride 64
bench/build/adc_bench exp2               # also log2, tanh, exp, dbgain, gaindb: timing + exhaustive f32 error, ~15-30 s each
bench/build/adc_bench tanh ulp f32 --stride 256
bench/build/adc_bench tables             # tables vs math as tables multiply, ~13 s -> bench/out/tables.csv
python bench/plot_tables.py              # -> bench/out/tables.png, tables_slowdown.png
bench/build/adc_bench lutpoly            # reproduce "Lookup Tables Are Bad Approximators" (notes/lut-reproduction.md)
```

- **Methods** are listed in `METHODS` in `generate.py`; edit and regenerate.
  Each example can also print its own function with `--emit-cpp`.
- **Domains:** *radians* is float x in [-π, π], *turns* is float x in [0, 1],
  and *phase* is a `uint32_t` fixed-point accumulator (2³² = one turn), the
  way shipped table oscillators index. The phase LUTs come in 256 to 64K
  entries, and tables are built at startup from `std::sin`.
- **Timing** is the method on the slides (`simple/main.cpp`, built as
  `adc_simple`): `out[i] = f(in[i])` over 65,536 sorted inputs (like an
  oscillator), the median of 100 passes, in ns per sample; `out` stays a
  buffer from the caller. `simple/loops.cpp` (`adc_simple_loops`) times
  all three loop shapes.
  - *indep* puts Carruth's `escape` after every sample: one at a time,
    the way per-sample DSP is written. The column the slides quote.
  - *vector* escapes once per pass, so the compiler may vectorize:
    polynomials do (4 lanes, unrolled 4×), table lookups and `std::`
    library calls don't.
  - *cold* is a 256-sample, 997 Hz oscillator block straight after evicting
    64 MB. The block's phases and the output buffer are re-touched, so only
    the method's own table, constants and code start cold. Trimmed mean.
- **Phase-domain ULP** is over every 32-bit phase. Its inputs have absolute
  precision, so compare those methods by max abs dB.
- **ULP f32** checks every normal float in range: |x| ≤ f32(π) for radians,
  [FLT_MIN, 1] for turns. Truth is f64, and the error is fractional ULPs of
  the true value, the same as `examples/common/metrics.py`.
  - Mean and exact % are weighted by float count, so they're dominated by tiny
    inputs (half of all floats in range are below about 1e-19). Max is the
    number to quote.
- **ULP f64** uses 2²⁴ random inputs, in f64 ULPs against `std::sin`. These
  degree-13 methods are built for f32, so expect large numbers.
- `-DADC_FAST_MATH=ON` builds with `-ffast-math` / `/fp:fast` and shows the
  split-π method breaking; see `notes/fast-math-split-pi.md`.
- The emitted code is plain `a * b + c`. Whether it becomes FMA depends on
  the compiler's `-ffp-contract` default, so C++ errors can differ slightly
  from the Python model, which never fuses.
- `--domain radians|turns|phase` limits the ULP pass to one domain (turns
  alone takes ~7 s). `--dump-worst n` writes each method's n worst f32
  inputs to `worst_<method>.txt`; `examples/08-coefficient-search` uses this
  to feed the exhaustive check back into its search (`--bench-rounds`).
- **tan(πx)** (`adc_bench tan`, `src/tan.hpp`): the filter-cutoff methods from
  `examples/09-tan`, listed in `TAN_METHODS` and generated into
  `generated/tan_approx.hpp`.
  - Inputs are x = fc/fs. Timing uses uniform [0, ½), and the cold block is
    a 20 Hz–20 kHz cutoff sweep. The baseline is `std::tan(π·x)`.
  - The error pass checks every normal float in [FLT_MIN, ½). It reports max
    ULP in the audio band (≤ 20 kHz at 48 kHz) and up to Nyquist, plus the
    cutoff shift in cents, using `err::cutoff_cents`, the same as
    `metrics.cutoff_cents`. f32 only.
- **exp2, log2, tanh, exp, dB ↔ gain** (`adc_bench exp2|log2|tanh|exp|dbgain|gaindb`,
  `src/unary.hpp`): the methods from `examples/10-exp2` to `14-db`. They're
  listed in `EXP2_METHODS` … `GAINDB_METHODS` and generated into one header
  each. `bit_helpers.hpp` holds the shared `pow2i` / `split_exponent`.
  - Each function is a traits struct: its truth (f64 `std::` function), the
    float ranges checked exhaustively, the timing inputs, and the audio
    unit for its error. exp2 reports cents and dB from relative error. log2
    reports cents and dB from absolute error over log2 x in [−20, 10]. tanh
    reports absolute error in dB plus max |y|. exp reports dB from relative
    error.
  - dbgain checks dB over the whole normal-gain range, [−750, 770]. gaindb
    checks every positive float *including subnormals*, where only the
    `_safe` form is right.
  - **NEON rows (arm64 only):** tanh adds `vsl::tanh`'s exact arithmetic with
    the hardware rsqrt estimate its vector path uses: raw, then +1 and +2
    Newton steps. These rows call the scalar intrinsic, so their *buffer*
    timing isn't vectorized and isn't comparable.
- **Tables vs math** (`adc_bench tables`, `src/tables.hpp`): K instances
  each run sin → tanh → dB-gain into a bus. The table version gives each
  instance three float LUTs (linear interpolation); the math version uses
  the generated polynomials, vectorized or scalar.
  - Swept: K, table size N, sequential vs hashed phase, and 0/2/16 MB of
    other work between rounds.
  - Output: CSV plus a summary; `plot_tables.py` draws ns per sample
    against total footprint with the cache sizes marked. See
    `notes/tables-vs-math.md`.
- **Fit variants:** the headers use Chebyshev projection. `generate.py --fit interpolation
  --out bench/generated-interpolation`, then `cmake -S bench -B bench/build-interpolation
  -DADC_GENERATED_DIR=generated-interpolation`, builds every method with interpolation
  fits (tuned rows from `tuned/interpolation/`). See `notes/fit-projection.md`.
- Tuned coefficients saved by 08 (`examples/08-coefficient-search/tuned/*.json`)
  are picked up by `generate.py` automatically.
