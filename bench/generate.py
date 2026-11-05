"""Generate bench/generated/*_approx.hpp from the Python examples.

Emits every benchmarked method as a template<typename T> function (see
examples/common/cppgen.py), plus a compile-time list the bench iterates:

    adc::for_each_method<T>(f)   calls f(index, name, domain, lambda) per method

Lambdas rather than function pointers so each method inlines into the
timing and ULP loops.

Tables are built at startup from std::sin (sin_table / sin_turns_table) so
the header stays small whatever the sizes; the examples' --emit-cpp prints
literal tables instead.

The other functions get a header each, listing their methods the same way:
adc::for_each_<fn>_method<T>(f) calls f(index, name, fn), every fn taking T.

  tan_approx.hpp    tan(pi x) for filter cutoffs, x = fc / fs (09-tan)
  exp2_approx.hpp   2^x from exponent bits + polynomial (10-exp2)
  log2_approx.hpp   log2 x from exponent bits + polynomial (11-log2)
  tanh_approx.hpp   tanh x for saturation (12-tanh)
  exp_approx.hpp    e^x with Cody–Waite reduction (13-exp)
  dbgain_approx.hpp, gaindb_approx.hpp   dB <-> gain (14-db)

Domains (sin):
  radians  T x in [-pi, pi]
  turns    T x in [0, 1]
  phase    std::uint32_t phase, 2^32 = one turn -- a fixed-point phase
           accumulator, the way shipped table oscillators index

Chebyshev fits are projections unless --fit interpolation (see common/fit.py).

Usage:
    python bench/generate.py            # writes bench/generated/*_approx.hpp
    python bench/generate.py --fit interpolation --out bench/generated-interpolation
    python bench/generate.py --stdout   # print instead
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "examples"))
from common import cppgen, fit, tanhpoly  # noqa: E402
from common.log2poly import SQRT2  # noqa: E402
from common.tanhpoly import VSL_SIGMOID  # noqa: E402

TUNED_DIR = ROOT.parent / "examples" / "08-coefficient-search" / "tuned"


def tuned_files():
    """08's saved coefficients: tuned/*.json, or tuned/<fit>/*.json when the
    fits aren't the default (08 saves interpolation-started searches there)."""
    d = TUNED_DIR if fit.get_method() == "projection" else TUNED_DIR / fit.get_method()
    return sorted(d.glob("turns_*.json"))

# (emitter, args, domain). Edit to change what gets benchmarked.
PHASE_LUT_SIZES = [256, 1024, 4096, 65536]

TUNED_PLACEHOLDER = ("tuned", (), "turns")

METHODS = [
    (cppgen.emit_taylor, (13,), "radians"),
    (cppgen.emit_pade, (7, 6), "radians"),
    (cppgen.emit_lut, (1024, "nearest", False), "radians"),
    (cppgen.emit_lut, (1024, "linear", False), "radians"),
    (cppgen.emit_cheb_vanilla, (13,), "radians"),
    (cppgen.emit_cheb_factored, (13,), "radians"),
    (cppgen.emit_cheb_split, (11,), "radians"),
    (cppgen.emit_cheb_split, (13,), "radians"),
    (cppgen.emit_cheb_split_guarded, (13,), "radians"),
    (cppgen.emit_cheb_turns, (11,), "turns"),
    (cppgen.emit_cheb_turns, (13,), "turns"),
    (cppgen.emit_remez_turns, (11,), "turns"),
    (cppgen.emit_remez_turns, (13,), "turns"),
    # Any coefficients 08-coefficient-search has saved (expanded in generate()).
    TUNED_PLACEHOLDER,
    (cppgen.emit_cheb_turns_phase, (13,), "phase"),
    (cppgen.emit_cheb_phase, (13,), "phase"),
    *[(cppgen.emit_lut_phase, (n, interp), "phase")
      for interp in ("nearest", "linear") for n in PHASE_LUT_SIZES],
]

# tan(pi x) for filter cutoffs: Padé vs. Chebyshev turns at a few sizes.
TAN_METHODS = [
    *[(cppgen.emit_tan_pade, mn) for mn in [(3, 2), (5, 4), (7, 6), (9, 8)]],
    *[(cppgen.emit_tan_turns, (d,)) for d in [3, 5, 7, 9, 11]],
]

EXP2_METHODS = [
    (cppgen.emit_exp2_bits, (True,)),
    (cppgen.emit_exp2, (5, "floor")),
    (cppgen.emit_exp2, (6, "floor")),   # vsl::exp2
    (cppgen.emit_exp2, (5, "round")),
    (cppgen.emit_exp2, (6, "round")),
    (cppgen.emit_exp2_unit, (5,)),      # on the slides
    (cppgen.emit_exp2_table, (3, 16)),
    (cppgen.emit_exp2_full, (6,)),
]

LOG2_METHODS = [
    (cppgen.emit_log2_bits, (True,)),
    (cppgen.emit_log2, (7, None)),
    (cppgen.emit_log2, (9, None)),
    (cppgen.emit_log2, (7, 1.5)),       # on the slides
    (cppgen.emit_log2, (9, 1.5)),       # vsl::log2
    (cppgen.emit_log2, (7, SQRT2)),
    (cppgen.emit_log2, (9, SQRT2)),
    (cppgen.emit_log2_atanh, (5,)),
    (cppgen.emit_log2_atanh, (7,)),
]

TANH_METHODS = [
    (cppgen.emit_tanh_clamp, (9,)),
    (cppgen.emit_tanh_pade, (7, 6)),
    (cppgen.emit_tanh_pade, (7, 6, tanhpoly.best_pade_clamp(7, 6))),  # input clamped instead
    (cppgen.emit_tanh_pade, (9, 8)),
    (cppgen.emit_tanh_sigmoid, (9, VSL_SIGMOID, "vsl_sigmoid_tanh_9")),
    (cppgen.emit_tanh_sigmoid, (9,)),
    (cppgen.emit_tanh_sigmoid, (11,)),
    (cppgen.emit_tanh_sigmoid, (9, None, None, 9.0)),
    (cppgen.emit_tanh_sigmoid, (11, None, None, 9.0)),
    # Plain Chebyshev R on [0, 3.4], R(0) = 1: no Remez (tanhpoly.fit_sigmoid_cheb).
    (cppgen.emit_tanh_sigmoid, (9, tanhpoly.fit_sigmoid_cheb(9), "sigmoid_cheb_clamp_tanh_9", 9.0)),
    # Fitted for relative error (ULP), and with R(0) = 1 pinned too: see tanhpoly.fit_sigmoid.
    (cppgen.emit_tanh_sigmoid, (9, tanhpoly.fit_sigmoid(9, relative=True), "sigmoid_rel_clamp_tanh_9", 9.0)),
    (cppgen.emit_tanh_sigmoid, (9, tanhpoly.fit_sigmoid(9, relative=True, pinned=True), "sigmoid_relpin_clamp_tanh_9", 9.0)),
    (cppgen.emit_tanh_sigmoid, (11, tanhpoly.fit_sigmoid(11, relative=True), "sigmoid_rel_clamp_tanh_11", 9.0)),
    (cppgen.emit_tanh_sigmoid, (11, tanhpoly.fit_sigmoid(11, relative=True, pinned=True), "sigmoid_relpin_clamp_tanh_11", 9.0)),
    (cppgen.emit_tanh_exp2, ()),
    (cppgen.emit_tanh_expm1, ()),
]

EXP_METHODS = [
    (cppgen.emit_exp_via_exp2, (6,)),
    (cppgen.emit_exp_cody_waite, (5,)),
    (cppgen.emit_exp_cody_waite, (6,)),
    (cppgen.emit_exp_cody_waite, (7,)),
]

DBGAIN_METHODS = [
    (cppgen.emit_db_to_gain, (2, False)),
    (cppgen.emit_db_to_gain, (3, False, False)),
    (cppgen.emit_db_to_gain, (3, False)),
    (cppgen.emit_db_to_gain, (4, False)),
    (cppgen.emit_db_to_gain, (3, True)),
]

GAINDB_METHODS = [
    (cppgen.emit_gain_to_db, (3, False)),
    (cppgen.emit_gain_to_db, (4, False)),
    (cppgen.emit_gain_to_db, (5, False)),
    (cppgen.emit_gain_to_db, (4, True)),
]

# fn -> (methods, extra includes). bit_helpers.hpp holds pow2i and
# split_exponent once; tanh's exp2 forms use exp2_approx.hpp's round_exp2_6_full.
UNARY = {
    "tan": (TAN_METHODS, []),
    "exp2": (EXP2_METHODS, ["<algorithm>", '"bit_helpers.hpp"']),  # round_exp2_6_full clamps
    "log2": (LOG2_METHODS, ['"bit_helpers.hpp"']),
    "tanh": (TANH_METHODS, ["<algorithm>", '"exp2_approx.hpp"']),
    "exp": (EXP_METHODS, ['"bit_helpers.hpp"']),
    "dbgain": (DBGAIN_METHODS, ["<algorithm>", '"bit_helpers.hpp"']),
    "gaindb": (GAINDB_METHODS, ["<limits>", '"bit_helpers.hpp"']),
}
BASE_INCLUDES = ["<cmath>", "<cstdint>", "<type_traits>"]

ARG_TYPE = {"radians": "T", "turns": "T", "phase": "std::uint32_t"}

TABLES = """
/// One period of sin starting at `start` radians: N entries plus a guard
/// point. Built from std::sin in double at startup.
template<typename T, std::size_t N>
inline auto make_sin_table(double start) -> std::array<T, N + 1>
{
    auto t = std::array<T, N + 1>{};
    for (std::size_t i = 0; i <= N; ++i) {
        t[i] = static_cast<T>(std::sin(start + 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(N)));
    }
    return t;
}

/// sin(-pi + 2 pi i / N), for radians lookups.
template<typename T, std::size_t N>
inline const auto sin_table = make_sin_table<T, N>(-std::numbers::pi);

/// sin(2 pi i / N), for phase-accumulator lookups.
template<typename T, std::size_t N>
inline const auto sin_turns_table = make_sin_table<T, N>(0.0);
"""

OUT = ROOT / "generated" / "sin_approx.hpp"


def generate():
    prologue = cppgen.prologue().replace("#include <cmath>", "#include <array>\n#include <cmath>\n#include <cstddef>\n#include <numbers>")
    parts = [prologue, """
/// Input domain of a method: radians x in [-pi, pi] approximating sin(x),
/// turns x in [0, 1] approximating sin(2 pi x), or a 32-bit phase
/// accumulator approximating sin(2 pi phase / 2^32).
enum class Domain { radians, turns, phase };
""", TABLES]
    names, emitted_tables = [], set()
    methods = []
    for m in METHODS:
        methods += [(cppgen.emit_tuned_turns, (path,), "turns") for path in tuned_files()] \
            if m is TUNED_PLACEHOLDER else [m]
    for emit, args, domain in methods:
        name, code = emit(*args)
        # LUTs of the same size share one table.
        if code.startswith("/// sin(-pi"):
            table, fn = code.split("\n\n", 1)
            if table in emitted_tables:
                code = fn
            emitted_tables.add(table)
        parts.append(code)
        names.append((name, domain))

    calls = "\n".join(
        f'    f(std::integral_constant<int, {i}>{{}}, "{name}", std::integral_constant<Domain, Domain::{domain}>{{}}, '
        f"[]({ARG_TYPE[domain]} x) {{ return {name}<T>(x); }});"
        for i, (name, domain) in enumerate(names)
    )
    parts.append(f"""inline constexpr int method_count = {len(names)};

/// Calls f(index, name, domain, fn) for every generated method, in order.
/// index and domain are integral_constants, so callers can branch on them
/// with if constexpr (phase methods take std::uint32_t, the others T).
template<typename T, typename F>
inline void for_each_method(F&& f)
{{
{calls}
}}
""")
    parts.append(cppgen.epilogue())
    return "\n".join(parts)


def with_includes(includes):
    """The standard prologue with its include list replaced: system headers
    sorted, then local ones."""
    system = sorted(BASE_INCLUDES + [i for i in includes if i.startswith("<")])
    local = [i for i in includes if i.startswith('"')]
    lines = [f"#include {i}" for i in system] + ([""] if local else []) + [f"#include {i}" for i in local]
    base = "\n".join(f"#include {i}" for i in BASE_INCLUDES)
    return cppgen.prologue().replace(base, "\n".join(lines))


def generate_bit_helpers():
    return "\n".join([with_includes(["<bit>"]), cppgen.bit_helpers(), cppgen.epilogue()])


def generate_unary(fn):
    methods, includes = UNARY[fn]
    prologue = with_includes(includes)
    names, parts = [], [prologue]
    for emit, args in methods:
        name, code = emit(*args)
        parts.append(code)
        names.append(name)
    calls = "\n".join(
        f'    f(std::integral_constant<int, {i}>{{}}, "{name}", [](T x) {{ return {name}<T>(x); }});'
        for i, name in enumerate(names)
    )
    what = {"tan": "tan(pi x)", "exp2": "2^x", "log2": "log2 x", "tanh": "tanh x", "exp": "e^x",
            "dbgain": "dB -> gain", "gaindb": "gain -> dB"}[fn]
    takes = "\n/// Each takes x = fc / fs in [0, 1/2)." if fn == "tan" else ""
    parts.append(f"""inline constexpr int {fn}_method_count = {len(names)};

/// Calls f(index, name, fn) for every generated {what} method, in order.{takes}
template<typename T, typename F>
inline void for_each_{fn}_method(F&& f)
{{
{calls}
}}
""")
    parts.append(cppgen.epilogue())
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stdout", action="store_true", help="print instead of writing the header")
    fit.add_argument(parser)
    parser.add_argument("--out", type=Path, default=OUT.parent,
                        help="directory for the headers (default bench/generated; build against another"
                             " with -DADC_GENERATED_DIR=...)")
    args = parser.parse_args()
    fit.set_method(args.fit)
    outputs = {args.out / OUT.name: (generate(), len(METHODS) - 1 + len(tuned_files()))}
    outputs[args.out / "bit_helpers.hpp"] = (generate_bit_helpers(), 0)
    for fn, (methods, _) in UNARY.items():
        outputs[args.out / f"{fn}_approx.hpp"] = (generate_unary(fn), len(methods))
    if args.stdout:
        for src, _ in outputs.values():
            print(src)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for path, (src, n) in outputs.items():
        path.write_text(src)
        print(f"Wrote {path} ({n} methods)")


if __name__ == "__main__":
    main()
