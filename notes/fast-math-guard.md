# Guarding split π under fast-math (2026-11-02)

Answers steps 2–3 of `fast-math-split-pi.md`'s exploration order. MSVC was
tested in github.com/rygrob/adc-msvc (`WRITEUP.md` there has
every variant). The clang half is reproduced here on Apple clang.

## The guard

`cppgen.fp_precise_macros()` / `emit_cheb_split_guarded` (bench method
`cheb_split_guarded_sin_13`), the repo's recommended header:

- `float_control(precise, on, push)` / `pop` around the definition
  (`__pragma` on MSVC, `_Pragma` on clang)
- `__forceinline` on MSVC, `inline __attribute__((always_inline))` on clang
- GCC: no such pragma; the `#else` branch leaves it unguarded

Why `__forceinline`: MSVC tracks the FP model per function and won't
auto-inline a precise function into a `/fp:fast` caller (a call per sample,
scalar, 4.8× slower). Forced, it inlines *and* keeps precise semantics:
vectorized, bit-identical to `/fp:precise`. Observed on 19.44, not documented.
Clang flags each instruction, so the pragma alone survives inlining.

## Results, every normal float in [−f32(π), f32(π)]

| | unguarded, fast | guarded, fast | precise build |
|---|---|---|---|
| MSVC 19.44 `/fp:fast`, x64 (repo) | 12,303,662 | 4.904 | 4.904 |
| clang-cl 19.1 `/fp:fast`, x64 (repo) | 12,303,662 | 4.904 | 4.904 |
| Apple clang 21 `-ffast-math`, arm64 (repo harness) | 12,303,662 | 4.437 | 4.437 |
| Apple clang 21, our bench (projection fit) | 1.23e7 | 4.437 | 4.437 |

4.437 vs 4.904: clang on arm64 (and on x64 with AVX2) contracts Horner into
FMA; `float_control(precise)` keeps contraction on in clang, so the guarded
build matches the default build bit for bit. The guarded `kernel_f32` on
arm64: no calls, NEON `.4s`, five `fmla`, both π_lo constants present (the
unguarded fast build has none).

Timing, our bench, M2 Pro on AC power, indicative: guarded under fast-math
0.97 ns one at a time / 0.25 vectorized, the same as unguarded in the precise
build (0.97 / 0.24). The unguarded fast build's 0.82 / 0.21 is the factored
version's speed: it dropped the π_lo work.

## Reproduce

```sh
python3 bench/generate.py
cmake --build bench/build-fast --target adc_bench   # -DADC_FAST_MATH=ON
bench/build-fast/adc_bench ulp f32 --domain radians
```

## Caveats (from the repo's write-up)

- Guard at the definition; a pragma on the caller does nothing.
- MSVC: every inline wrapper between the guarded function and the hot loop
  needs `__forceinline` too, or it stops inlining.
- `__forceinline` is a request (C4714, `/Ob0`): it degrades to a correct call.
- Clang: intrinsics' bodies were parsed outside the pragma, so a guarded
  helper built from `_mm_sub_ps` is not protected; use vector operators.
