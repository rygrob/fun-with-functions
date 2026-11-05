# FMA contraction: when `a * b + c` becomes one instruction

Short notes to come back to for the optimization part of the talk. Items
marked **verify** are from memory, not tested.

## Measured: Apple clang 21

From a test file with `p * x2 + c` (one expression) and
`t = p * x2; t + c` (two statements):

| flags | one expression | two statements |
|---|---|---|
| arm64 `-O0` / `-O2` / `-O3` | **fmadd** | fmul + fadd |
| arm64 `-ffp-contract=off` | fmul + fadd | fmul + fadd |
| arm64 `-ffp-contract=fast` or `-ffast-math` | fmadd | **fmadd** |
| x86-64 `-O2` (baseline, no FMA) | mulss + addss | mulss + addss |
| x86-64 `-O2 -mfma` | **vfmadd** | vmulss + vaddss |
| x86-64 `-mfma -ffp-contract=fast` | vfmadd | vfmadd |

- Clang's default is `-ffp-contract=on`: it fuses only **within one
  expression**. The front end emits `llvm.fmuladd`, so it happens even at
  `-O0`. It's not an optimization-level thing and it's not fast-math.
- Whether FMA is used depends on the **target**. arm64 always has FMA;
  baseline x86-64 doesn't, so you need `-mfma` / `-march=haswell` or newer.
- `fast` (and fast-math) also fuses **across statements**.
- Scoped control: `#pragma clang fp contract(on|off|fast)`, or the standard
  `#pragma STDC FP_CONTRACT ON|OFF`.

## Other compilers (verify all of these)

- **GCC**: I believe the default is `-ffp-contract=fast` in GNU modes and
  `off` with ISO `-std=c++NN`. CMake defaults to `-std=gnu++20`
  (`CMAKE_CXX_EXTENSIONS ON`), which would make a plain CMake GCC build fuse
  across statements.
- **MSVC**: I believe that since VS2022, `/fp:precise` no longer contracts by
  default and `/fp:contract` turns it on. `/fp:fast` contracts.

## Why it matters here

- Our emitted Horner steps are one expression each (`p = p * x2 + c;`), so
  the same source gives **different results per platform**. The bench on this
  Mac measured FMA Horner; the Python model rounds twice per step; a baseline
  x86 build would round twice too.
- FMA is usually *more* accurate (one rounding instead of two) and faster,
  but "usually" isn't "always". With a small enough tolerance, a test that
  passed on x86 can fail on ARM, and the reverse.
- It interacts with the split-π trick only through fast-math, which brings
  reassociation along with contraction. Contraction alone doesn't touch
  `(x - pi_hi) - pi_lo`, because there's no multiply in it.

## Options for reproducible numbers

1. **Emit `std::fma` explicitly** (what `vsl::sin` does). Every platform gives
   the same answer. On x86 without `-mfma` it becomes a **slow library
   call**, so it needs to be guarded or the target required.
2. **`-ffp-contract=off` for the bench.** Matches the Python model, but on ARM
   it measures slower code than would actually ship.
3. **Make the Python model fuse.** Emulate an FMA with an exact product (in
   f64 for f32 inputs, the product is exact) and a single rounding, so
   Python and the ARM C++ agree.
4. **Report it.** At minimum, the bench header should print the contraction
   mode it was built with.

## For the talk

- A slide on "the same source, two answers": `p * x + c` across
  arm64 / x86 / x86 + FMA, with the ULP difference shown.
- How it fits with fast-math (reassociation, contraction, `nnan`, …). That's a
  good place to bring in the split-π collapse from
  `notes/fast-math-split-pi.md`.
