#pragma once
// Reproducing Jatin Chowdhury, "Lookup Tables Are Bad Approximators" (2025):
// sin(x) on [-pi, pi] to 2e-5 absolute, four ways --
//
//   std        std::sin (scalar); Accelerate's vvsinf as the SIMD reference
//              (his used xsimd)
//   poly-7     his code (sin_poly_7, from the post's figure): an odd degree-7
//              fit of sin(x) / (pi^2 - x^2), so (pi^2 - x^2) * x * P(x^2),
//              total degree 9, 4 coefficients, 1.65e-5. Same form and cost as
//              our cheb_factored_sin_9 (same form; Chebyshev projection, 1.7e-5)
//   LUT-320k   320,000 floats over [-pi, pi], nearest entry (1.25 MB)
//   LUT_lin    500 floats, linear interpolation (2 KB)
//
// -- over 10M inputs, a ramp (an oscillator's phase, precomputed) and uniform
// random, in three harnesses:
//
//   scalar     one call per sample, auto-vectorization disabled
//   auto       out[i] = f(in[i]); the compiler may vectorize (only the poly can)
//   simd       explicit 4-wide (clang vector extensions, NEON / SSE): the poly
//              runs on vectors, the tables split the vector into 4 scalar
//              loads, the linear table interpolates on vectors
//
// plus "ramp generated in the loop": x += dx with a wrap, inside the timed
// loop -- a serial float chain, to test why his M1 ramp runs were 3-5x
// slower than random for every method.
//
// Reported like the post: time for 10M, and speedup over std.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#include <audio_bench/audio_bench.hpp>


namespace adc::lutpoly {

using f4 = float __attribute__((ext_vector_type(4)));
using i4 = std::int32_t __attribute__((ext_vector_type(4)));

inline constexpr auto COUNT = std::size_t{10'000'000};
inline constexpr auto REPS = 5;
inline constexpr auto PI = std::numbers::pi_v<float>;
inline constexpr auto RAMP_HZ = 440.0;
inline constexpr auto FS = 48000.0;

/// sin over [-pi, pi] at size + 1 points (the guard makes x = pi safe).
struct Table {
    std::vector<float> v;
    float scale;  // entries per radian
    explicit Table(std::size_t size) : v(size + 1), scale(static_cast<float>(size) / (2.0f * PI))
    {
        for (auto i = std::size_t{}; i <= size; ++i) {
            v[i] = static_cast<float>(std::sin(-std::numbers::pi + 2.0 * std::numbers::pi * static_cast<double>(i)
                                               / static_cast<double>(size)));
        }
    }
};

/// Jatin Chowdhury's sin_poly_7, as in the post (outer factor ours: the
/// figure stops at the polynomial).
template<typename T>
constexpr T sin_poly_7(T x, T x_sq)
{
    const auto x_5_7 = T(0.000170965340046f) + T(-2.09843101304e-6f) * x_sq;
    const auto x_3_5_7 = T(-0.00661594021539f) + x_5_7 * x_sq;
    const auto x_1_3_5_7 = T(0.101319673615f) + x_3_5_7 * x_sq;
    return x * x_1_3_5_7;
}

template<typename T>
constexpr T sin_jc(T x)
{
    constexpr auto pi_sq = std::numbers::pi_v<float> * std::numbers::pi_v<float>;
    const auto x_sq = x * x;
    return (T(pi_sq) - x_sq) * sin_poly_7(x, x_sq);
}

inline auto lut_nearest(const Table& t, float x) -> float
{
    return t.v[static_cast<std::size_t>((x + PI) * t.scale + 0.5f)];
}

inline auto lut_linear(const Table& t, float x) -> float
{
    const auto pos = (x + PI) * t.scale;
    const auto i = static_cast<std::size_t>(pos);
    const auto f = pos - static_cast<float>(i);
    return t.v[i] + f * (t.v[i + 1] - t.v[i]);
}

inline auto lut_nearest4(const Table& t, f4 x) -> f4
{
    const auto i = __builtin_convertvector((x + PI) * t.scale + 0.5f, i4);
    return f4{t.v[static_cast<std::size_t>(i[0])], t.v[static_cast<std::size_t>(i[1])],
              t.v[static_cast<std::size_t>(i[2])], t.v[static_cast<std::size_t>(i[3])]};
}

inline auto lut_linear4(const Table& t, f4 x) -> f4
{
    const auto pos = (x + PI) * t.scale;
    const auto i = __builtin_convertvector(pos, i4);
    const auto f = pos - __builtin_convertvector(i, f4);
    const auto at = [&](int k, int off) { return t.v[static_cast<std::size_t>(i[k] + off)]; };
    const auto a = f4{at(0, 0), at(1, 0), at(2, 0), at(3, 0)};
    const auto b = f4{at(0, 1), at(1, 1), at(2, 1), at(3, 1)};
    return a + f * (b - a);
}

/// Best of REPS runs of fn() over COUNT samples, in ms.
template<typename F>
inline auto time_ms(F&& fn) -> double
{
    auto best = 1e300;
    for (auto r = 0; r < REPS; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

/// out[i] = f(in[i]) with vectorization off: one call per sample.
template<typename F>
inline auto scalar(const std::vector<float>& in, std::vector<float>& out, F f) -> void
{
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (auto i = std::size_t{}; i < in.size(); ++i) out[i] = f(in[i]);
    audio_bench::do_not_optimize(out[0]);
}

/// out[i] = f(in[i]), the compiler free to vectorize.
template<typename F>
inline auto autovec(const std::vector<float>& in, std::vector<float>& out, F f) -> void
{
    for (auto i = std::size_t{}; i < in.size(); ++i) out[i] = f(in[i]);
    audio_bench::do_not_optimize(out[0]);
}

/// Explicit 4-wide: f takes and returns f4.
template<typename F>
inline auto simd(const std::vector<float>& in, std::vector<float>& out, F f) -> void
{
    for (auto i = std::size_t{}; i + 4 <= in.size(); i += 4) {
        f4 x;
        __builtin_memcpy(&x, &in[i], sizeof x);
        const f4 y = f(x);
        __builtin_memcpy(&out[i], &y, sizeof y);
    }
    audio_bench::do_not_optimize(out[0]);
}

/// The ramp generated inside the timed loop: x += dx, wrapped -- serial.
template<typename F>
inline auto ramp_inline(std::vector<float>& out, F f) -> void
{
    auto x = -PI;
    const auto dx = static_cast<float>(2.0 * std::numbers::pi * RAMP_HZ / FS);
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (auto i = std::size_t{}; i < out.size(); ++i) {
        out[i] = f(x);
        x += dx;
        if (x >= PI) x -= 2.0f * PI;
    }
    audio_bench::do_not_optimize(out[0]);
}

struct Row {
    std::string name;
    double ms;
    double max_abs;
};

inline auto print(const std::string& title, const std::vector<Row>& rows) -> void
{
    std::cout << std::format("\n{}\n", title);
    for (const auto& r : rows) {
        std::cout << std::format("  {:<16} {:>9.2f} ms  ({:.2f}x)   max abs err {:.2e}\n", r.name, r.ms,
                                 rows[0].ms / r.ms, r.max_abs);
    }
}

inline auto max_abs(const std::vector<float>& in, const std::vector<float>& out) -> double
{
    auto e = 0.0;
    for (auto i = std::size_t{}; i < in.size(); ++i) {
        e = std::max(e, std::abs(static_cast<double>(out[i]) - std::sin(static_cast<double>(in[i]))));
    }
    return e;
}

inline auto run() -> void
{
    const auto big = Table{320'000};
    const auto small = Table{500};

    auto ramp = std::vector<float>(COUNT);
    auto random = std::vector<float>(COUNT);
    auto rng = std::mt19937_64{12345};
    auto dist = std::uniform_real_distribution<float>{-PI, PI};
    for (auto i = std::size_t{}; i < COUNT; ++i) {
        const auto ph = std::fmod(static_cast<double>(i) * RAMP_HZ / FS, 1.0);
        ramp[i] = static_cast<float>(-std::numbers::pi + 2.0 * std::numbers::pi * ph);
        random[i] = dist(rng);
    }
    auto out = std::vector<float>(COUNT);

    const auto std_sin = [](float x) { return std::sin(x); };
    const auto poly = [](float x) { return sin_jc<float>(x); };
    const auto lut320 = [&](float x) { return lut_nearest(big, x); };
    const auto lut500 = [&](float x) { return lut_linear(small, x); };

    std::cout << std::format("sin(x) on [-pi, pi], {}M floats, best of {} runs; ramp = {:g} Hz oscillator at {:g} kHz\n",
                             COUNT / 1'000'000, REPS, RAMP_HZ, FS / 1000);

    for (const auto* name : {"RAMP", "RANDOM"}) {
        const auto& in = std::string{name} == "RAMP" ? ramp : random;
        auto rows = std::vector<Row>{};
        const auto add = [&](const char* label, auto&& harness) {
            const auto ms = time_ms([&] { harness(); });
            rows.push_back({label, ms, max_abs(in, out)});
        };

        add("std", [&] { scalar(in, out, std_sin); });
        add("poly-7", [&] { scalar(in, out, poly); });
        add("LUT-320k", [&] { scalar(in, out, lut320); });
        add("LUT_lin-500", [&] { scalar(in, out, lut500); });
        print(std::format("{}, scalar (no vectorization):", name), rows);

        rows.clear();
        add("std", [&] { autovec(in, out, std_sin); });
        add("poly-7", [&] { autovec(in, out, poly); });
        add("LUT-320k", [&] { autovec(in, out, lut320); });
        add("LUT_lin-500", [&] { autovec(in, out, lut500); });
        print(std::format("{}, auto (compiler may vectorize the loop):", name), rows);

        rows.clear();
#if defined(__APPLE__)
        add("vvsinf", [&] {
            const auto n = static_cast<int>(in.size());
            vvsinf(out.data(), in.data(), &n);
            audio_bench::do_not_optimize(out[0]);
        });
#else
        add("std (scalar)", [&] { scalar(in, out, std_sin); });
#endif
        add("poly-7", [&] { simd(in, out, [](f4 x) { return sin_jc<f4>(x); }); });
        add("LUT-320k", [&] { simd(in, out, [&](f4 x) { return lut_nearest4(big, x); }); });
        add("LUT_lin-500", [&] { simd(in, out, [&](f4 x) { return lut_linear4(small, x); }); });
        print(std::format("{}, explicit 4-wide SIMD (reference: {}):", name,
#if defined(__APPLE__)
                          "Accelerate vvsinf"
#else
                          "scalar std::sin"
#endif
                          ),
              rows);
    }

    auto rows = std::vector<Row>{};
    const auto add_inline = [&](const char* label, auto f) {
        rows.push_back({label, time_ms([&] { ramp_inline(out, f); }), 0.0});
    };
    add_inline("std", std_sin);
    add_inline("poly-7", poly);
    add_inline("LUT-320k", lut320);
    add_inline("LUT_lin-500", lut500);
    std::cout << "\nRAMP generated in the loop (x += dx, wrap; serial), scalar:\n";
    for (const auto& r : rows) {
        std::cout << std::format("  {:<16} {:>9.2f} ms  ({:.2f}x)\n", r.name, r.ms, rows[0].ms / r.ms);
    }
}

} // namespace adc::lutpoly
