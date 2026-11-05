# Keeping split-π alive under fast-math

Directions for exploring, not settled answers. Items marked **verify** are
things I believe but haven't confirmed on a current compiler. Check them on
Compiler Explorer (godbolt.org) before relying on them or putting them on a
slide.

---

## The problem

The split-π factors (05, `vsl::sin`):

```cpp
const auto lo = (x - pi_major) - pi_minor;   // ~ x - pi
const auto hi = (x + pi_major) + pi_minor;   // ~ x + pi
```

Under reassociation (`-ffast-math`, `-Ofast`, `-fassociative-math`,
`/fp:fast`), the compiler may rewrite this as

```cpp
x - (pi_major + pi_minor)     // pi_major + pi_minor folds at compile time...
                              // ...and rounds straight back to pi_major in f32
```

pi_minor is smaller than half an f32 step at π, so `f32(pi_major + pi_minor) ==
pi_major`. The split silently turns into example 04, the single-constant
version.

**What that costs:** from our tables, max error goes from 3.56 ULP to 1.2e7 ULP
at x = f32(±π), with relative error at 100% there. Nothing crashes and nothing
looks wrong on a plot of the waveform, so it's an easy regression to miss.

**Why per-file flags aren't enough:** vsl is header-only and `force_inline`.
The code is compiled with **the user's** flags, in **their** translation unit.
Anything we do has to live in the source and survive being inlined into a
fast-math caller.

---

## Measured: it does collapse on Apple clang 21 (arm64)

From the bench (`bench/`, built with `-DADC_FAST_MATH=ON`, `ulp f32 --stride 64`):

| method | fast-math off | fast-math on |
|---|---|---|
| cheb_split_sin_13 max ULP | 4.39 | **1.9e5** (at x = 3.1415863) |
| cheb_turns_sin_13 max ULP | 4.15 | 4.04 |

The assembly for `(x - hi) - lo` goes from two `fadd`s to one under
`-ffast-math`: the constants are folded together. An exhaustive run would
reach about 1.2e7 ULP at f32(π), the example 04 number. The turns version
isn't affected. Repeat this on gcc and MSVC to fill in the per-compiler
table at the end.

---

## First: a test that catches it

Before trying fixes, add a test that fails when the split collapses. Compile
it once normally and once with fast-math:

```cpp
// sin(f32(pi)) is about -8.742278e-08; the collapsed version returns exactly 0.
const float y = vsl::sin<false>(3.14159274f);
assert(y != 0.0f);
assert(std::abs(y - -8.742278e-08f) <= 4 * ulp_of(-8.742278e-08f));
```

The same check works for `-pi` and for the last few floats below π (see
`plots.near_pi_inputs` in the examples).

**Check the assembly too.** In the split version, both constants (pi_major and
pi_minor) should appear and there should be two subtractions per factor. If
you only see one π constant, it has been folded. On godbolt, compare
`-O2` against `-O2 -ffast-math` for clang and gcc, and `/O2` against
`/O2 /fp:fast` for MSVC.

---

## Clang (easiest)

In LLVM IR, fast-math permissions are **flags on each instruction** (`reassoc`,
`contract`, `nnan`, …), not a per-function mode. An `fsub` emitted without
`reassoc` stays unreassociable even after it's inlined into a fast-math
function. So scoped pragmas should survive `force_inline`. **verify** with an
inlined call site.

Options, in rough order of preference:

1. **`#pragma clang fp reassociate(off)`** inside the function body (or a
   block). This targets exactly the problem and leaves contraction (FMA) alone,
   so the Horner `fma`s are unaffected. Available since clang 12. **verify**
   the version.
   **Verified 2026-10-15, Apple clang 21 (arm64), scalar:** `(x - hi) - lo`
   at x = f32(π), in an `always_inline` function called from a `-O2
   -ffast-math` TU: without the pragma it returns exactly 0 (collapsed);
   with it, 8.74227766e-08 (intact). The pragma survives inlining.
   **Re-verified 2026-10-19, exhaustively:** the slide's `cheb_split_sin_13`
   (`always_inline`) in a loop over every float in [−f32(π), f32(π)], built
   `-O3 -ffast-math` and `-O2 -ffast-math`, harness and ULP in a separate
   precise TU. Max ULP: no pragma 1.23e7 (factored, as measured by the
   bench), pragma 4.6 (precise build: 4.44 either way). At f32(π): −0 vs
   −8.74e−8. No calls left in the loops, so it's inlined. The 4.6 vs 4.44
   is fast-math reordering the rest (Horner, the final product), not the
   split. The pragma must open a compound statement (clang errors
   otherwise), hence the block above. MSVC is still unverified.
   ```cpp
   {
   #pragma clang fp reassociate(off)
       const auto lo = (x - pi_major) - pi_minor;
       const auto hi = (x + pi_major) + pi_minor;
   }
   ```
2. **`#pragma float_control(precise, on, push)` … `(pop)`**: clang accepts the
   MSVC-style pragma too. It's heavier than option 1 (it turns off all the
   fast-math relaxations), but the same spelling works on both compilers.
   **verify** whether clang accepts it inside a function body as well as at
   file scope.
3. **`__arithmetic_fence(expr)`**: a clang builtin that stops reassociation
   across the fence, e.g. `__arithmetic_fence(x - pi_major) - pi_minor`.
   **verify** which targets support it; I believe x86 only, which would rule
   it out for ARM / Apple Silicon.
4. **`-fprotect-parens`**: makes clang honour parentheses as reassociation
   barriers. It's a flag, so it has the user-flags problem above; mainly
   useful for understanding the behaviour.

---

## GCC

GCC applies optimization options **per function**. That's the root of why it's
harder here: there's no per-instruction flag to survive inlining.

- **`__attribute__((optimize("no-associative-math")))`** or
  **`#pragma GCC optimize`**: GCC's docs say these are for debugging, not
  production. When two functions have different optimize options, GCC may
  refuse to inline one into the other, and for an `always_inline` function
  that's a **hard error**. **verify** with `force_inline` called from a
  `-ffast-math` TU.
- **Inline-asm value barrier** (GCC and clang): hide the intermediate from the
  optimizer so it can't reassociate across it.
  ```cpp
  auto d = x - pi_major;
  asm("" : "+x"(d));          // x86 SSE register; ARM64 would be "+w"
  const auto lo = d - pi_minor;
  ```
  It costs no instructions, but the constraint depends on the architecture,
  and with vsl's SIMD types each width needs the right register class. A
  generic `"+r"` forces a round trip through a general-purpose register, which
  is slow, and won't work for vectors.
- `-fno-associative-math` as a flag: same user-flags problem.

---

## MSVC (hardest)

- **`#pragma float_control(precise, on, push)` … `#pragma float_control(pop)`**:
  this is the documented way to get precise semantics locally under
  `/fp:fast`. The catch is that it must be at **file scope**, around whole
  function definitions, not inside a function.
  - That's workable for a header: wrap the `sin` definition.
  - **verify** the two things that matter: (a) that it really disables
    reassociation under `/fp:fast`, not just contraction, and (b) what happens
    when the function is **inlined** into a `/fp:fast` caller. Does the
    callee's precise model survive inlining, or does MSVC refuse to inline?
    This is the crux.
- **`#pragma fp_contract(off)`** controls only FMA contraction. It doesn't help
  with reassociation.
- **No inline asm on x64.** The asm-barrier trick doesn't exist. `_ReadWriteBarrier`
  is a memory barrier for the compiler and doesn't affect values held in
  registers.
- **Intrinsics:** worth testing whether MSVC reassociates `_mm_sub_ps` /
  `_mm_add_ps` under `/fp:fast`. I don't know. Clang turns these intrinsics
  into ordinary `fsub`/`fadd` IR, so in clang they get **no** protection; MSVC
  may treat them more opaquely. **verify** on godbolt.
- **`volatile`** as a last resort: store `x - pi_major` to a `volatile` local and
  read it back. It's portable and it works, but it forces a memory round trip
  per call and doesn't fit SIMD types. Only useful to prove a point.

---

## Sidestepping the problem

These are worth weighing against compiler-specific workarounds, and possibly
worth a slide.

1. **Work in turns (example 06).** `sin_2pi` has no compensation constants at
   all. Its zeros are exact floats, and its factors come straight from x:
   `-u * x * (x - 1) * p` with `u = x - 0.5f`. There's nothing for
   reassociation to fold away. Reordering the products changes the result by
   about a rounding error, not 1e7 ULP. It's the same accuracy (3.74 vs 3.56
   ULP) with fewer operations, and fast-math can't break it. That might be the
   strongest argument for turns in the whole talk.
2. **Making pi_minor a runtime value doesn't help.** Even if the compiler can't
   fold `pi_major + pi_minor` at compile time, reassociation still computes
   `x - (pi_major + pi_minor)` at runtime, and the sum still rounds to
   pi_major. The fix has to prevent reassociation, not just constant folding.
3. **Folding the correction into the product is fragile too.**
   `(x - π̂)(x + π̂) + (π̂² - π²)` can be rewritten by fast-math as
   `x² - (π̂² - c)`. That folds to `x² - f32(π²)`, which has catastrophic
   cancellation near π, arguably worse than the original.

---

## Suggested exploration order

1. Write the fails-under-fast-math test. Confirm on godbolt that clang, gcc and
   MSVC all actually collapse the split with fast-math on. If one doesn't, note
   which version.
2. Clang: try `#pragma clang fp reassociate(off)`, inlined into a `-ffast-math`
   caller, scalar and SIMD.
3. MSVC: try file-scope `float_control(precise, on, push)` around the function
   with `/fp:fast`, inlined. If it doesn't hold, try intrinsics, then decide
   whether `volatile` is acceptable or turns is the answer.
4. GCC: check whether the `optimize` attribute plays nicely with `always_inline`.
   If not, the asm barrier is the pragmatic route on x86 and ARM.
5. Write up which mechanism held on which compiler. A small table is good
   slide material: "this is why `sin_2pi` is the default".
