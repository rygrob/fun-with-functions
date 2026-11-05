#pragma once
// exp2, log2, tanh, exp and dB <-> gain (examples/10-14): timing and exhaustive f32 error of
// every generated method, one function at a time.
//
// Each function is a traits struct (Exp2, Log2, Tanh below) giving its
// methods, f64 truth, std:: reference, the float ranges to check, the inputs
// to time, and which audio unit its error turns into:
//
//   exp2   relative error -> cents of pitch (mtof), dB of gain
//   log2   absolute error -> cents of pitch, dB of level, over log2 x in
//          [-20, 10] (larger results are limited by their own f32 rounding)
//   tanh   absolute error in dB re full scale, and max |y|
//   exp    relative error -> dB (envelopes, one-pole coefficients)
//   dbgain relative error -> dB, over the whole normal-gain range of dB
//   gaindb absolute error in dB, over every positive float, subnormals
//          included (where only the safe form is right)
//
// Timing: as timing.hpp, inputs uniform over the traits' timing range; the
//   cold block ramps through that range. Speedups are against std::<fn>.
// Error, f32: every normal float in the traits' ranges (or every stride-th).
//   Truth is the f64 std:: function. Error columns as in the examples.
//
// tanh on arm64 adds vsl::tanh's exact arithmetic (FMA Horner, vsl's
// coefficients) with the NEON reciprocal-square-root estimate vsl's vector
// path uses -- raw, and with one and two Newton steps -- next to the scalar
// 1 / sqrt.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "errors.hpp"
#include "exp2_approx.hpp"
#include "dbgain_approx.hpp"
#include "exp_approx.hpp"
#include "gaindb_approx.hpp"
#include "log2_approx.hpp"
#include "tanh_approx.hpp"
#include "timing.hpp"

namespace adc::unary {

/// Every normal float in [lo, hi] (both positive), negated if `negate`.
struct Range {
    float lo;
    float hi;
    bool negate;
};

/// Running statistics for one method.
struct Stats {
    double ulp{};
    double ulp_x{};
    double abs{};       // only where the traits' abs_mask holds
    double abs_x{};
    double rel{};
    double max_y{};
    std::uint64_t count{};

    void add(double y, double t, double x, bool masked)
    {
        const auto bad = !std::isfinite(y) && std::isfinite(t);
        const auto e = bad ? std::numeric_limits<double>::infinity() : std::abs(y - t);
        const auto u = e / err::ulp_f32(t);
        if (u > ulp || count == 0) {
            ulp = u;
            ulp_x = x;
        }
        if (masked && e > abs) {
            abs = e;
            abs_x = x;
        }
        if (t != 0.0) rel = std::max(rel, e / std::abs(t));
        max_y = std::max(max_y, bad ? e : std::abs(y));
        ++count;
    }

    void merge(const Stats& o)
    {
        if (o.count && (o.ulp > ulp || count == 0)) {
            ulp = o.ulp;
            ulp_x = o.ulp_x;
        }
        if (o.abs > abs) {
            abs = o.abs;
            abs_x = o.abs_x;
        }
        rel = std::max(rel, o.rel);
        max_y = std::max(max_y, o.max_y);
        count += o.count;
    }
};

inline auto db(double e) -> double { return 20.0 * std::log10(std::max(e, 1e-300)); }

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

struct Exp2 {
    static constexpr const char* ref_name = "std::exp2";
    static constexpr const char* name = "exp2";
    static constexpr const char* what = "2^x";
    static constexpr int generated = exp2_method_count;
    template<typename T, typename F> static void for_each(F&& f) { for_each_exp2_method<T>(f); }
    template<typename T> static auto reference(T x) -> T { return std::exp2(x); }
    static auto truth(double x) -> double { return std::exp2(x); }
    /// Normal results with margin, as the example; the edges are its own table.
    static auto ranges() -> std::vector<Range>
    {
        return {{std::numeric_limits<float>::min(), 126.0f, false}, {std::numeric_limits<float>::min(), 125.0f, true}};
    }
    static auto range_text() -> std::string { return "normal x in [-125, 126]"; }
    static constexpr bool show_abs = false;  // 2^x spans 2^+-126; relative is the metric
    static auto abs_mask(double) -> bool { return false; }
    static auto input(double u) -> double { return -10.0 + 20.0 * u; }  // +-60 dB of gain, +-10 octaves
    static auto timing_text() -> std::string { return "x uniform in [-10, 10]"; }
    static auto audio_header() -> std::string { return std::format("{:>10} {:>10}", "cents", "dB"); }
    static auto audio(const Stats& s) -> std::string
    {
        return std::format("{:>10.3g} {:>10.3g}", 1200.0 * std::log2(1.0 + s.rel), 20.0 * std::log10(1.0 + s.rel));
    }
};

struct Log2 {
    static constexpr const char* ref_name = "std::log2";
    static constexpr const char* name = "log2";
    static constexpr const char* what = "log2 x";
    static constexpr int generated = log2_method_count;
    template<typename T, typename F> static void for_each(F&& f) { for_each_log2_method<T>(f); }
    template<typename T> static auto reference(T x) -> T { return std::log2(x); }
    static auto truth(double x) -> double { return std::log2(x); }
    static auto ranges() -> std::vector<Range>
    {
        return {{std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), false}};
    }
    static auto range_text() -> std::string { return "every normal x > 0"; }
    static constexpr bool show_abs = true;
    static auto abs_mask(double t) -> bool { return t >= -20.0 && t <= 10.0; }
    static auto input(double u) -> double { return std::exp2(-20.0 + 30.0 * u); }  // -120 .. +60 dB
    static auto timing_text() -> std::string { return "log2 x uniform in [-20, 10]"; }
    static auto audio_header() -> std::string { return std::format("{:>10} {:>10}", "cents", "dB"); }
    static auto audio(const Stats& s) -> std::string
    {
        return std::format("{:>10.3g} {:>10.3g}", 1200.0 * s.abs, 20.0 * std::log10(2.0) * s.abs);
    }
};

struct Exp {
    static constexpr const char* ref_name = "std::exp";
    static constexpr const char* name = "exp";
    static constexpr const char* what = "e^x";
    static constexpr int generated = exp_method_count;
    template<typename T, typename F> static void for_each(F&& f) { for_each_exp_method<T>(f); }
    template<typename T> static auto reference(T x) -> T { return std::exp(x); }
    static auto truth(double x) -> double { return std::exp(x); }
    /// Normal results, as the example.
    static auto ranges() -> std::vector<Range>
    {
        return {{std::numeric_limits<float>::min(), 88.0f, false}, {std::numeric_limits<float>::min(), 87.0f, true}};
    }
    static auto range_text() -> std::string { return "normal x in [-87, 88]"; }
    static constexpr bool show_abs = false;
    static auto abs_mask(double) -> bool { return false; }
    static auto input(double u) -> double { return -10.0 + 20.0 * u; }
    static auto timing_text() -> std::string { return "x uniform in [-10, 10]"; }
    static auto audio_header() -> std::string { return std::format("{:>10}", "dB"); }
    static auto audio(const Stats& s) -> std::string { return std::format("{:>10.3g}", 20.0 * std::log10(1.0 + s.rel)); }
};

struct DbToGain {
    static constexpr const char* ref_name = "std::pow(10, dB / 20)";
    static constexpr const char* name = "dbgain";
    static constexpr const char* what = "dB -> gain";
    static constexpr int generated = dbgain_method_count;
    template<typename T, typename F> static void for_each(F&& f) { for_each_dbgain_method<T>(f); }
    template<typename T> static auto reference(T x) -> T { return std::pow(T(10), x / T(20)); }
    static auto truth(double x) -> double { return std::pow(10.0, x / 20.0); }
    /// The whole normal-gain range, not just audio: errors repeat every 6.02 dB.
    static auto ranges() -> std::vector<Range>
    {
        return {{std::numeric_limits<float>::min(), 770.0f, false}, {std::numeric_limits<float>::min(), 750.0f, true}};
    }
    static auto range_text() -> std::string { return "normal dB in [-750, 770]"; }
    static constexpr bool show_abs = false;
    static auto abs_mask(double) -> bool { return false; }
    static auto input(double u) -> double { return -120.0 + 144.0 * u; }
    static auto timing_text() -> std::string { return "dB uniform in [-120, 24]"; }
    static auto audio_header() -> std::string { return std::format("{:>10}", "dB"); }
    static auto audio(const Stats& s) -> std::string { return std::format("{:>10.3g}", 20.0 * std::log10(1.0 + s.rel)); }
};

struct GainToDb {
    static constexpr const char* ref_name = "20 * std::log10";
    static constexpr const char* name = "gaindb";
    static constexpr const char* what = "gain -> dB";
    static constexpr int generated = gaindb_method_count;
    template<typename T, typename F> static void for_each(F&& f) { for_each_gaindb_method<T>(f); }
    template<typename T> static auto reference(T x) -> T { return T(20) * std::log10(x); }
    static auto truth(double x) -> double { return 20.0 * std::log10(x); }
    /// Every positive float, subnormals included.
    static auto ranges() -> std::vector<Range>
    {
        return {{std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(), false}};
    }
    static auto range_text() -> std::string { return "every positive x, subnormals included"; }
    static constexpr bool show_abs = true;
    static auto abs_mask(double) -> bool { return true; }
    static auto input(double u) -> double { return std::exp2(-20.0 + 24.0 * u); }
    static auto timing_text() -> std::string { return "log2 x uniform in [-20, 4]"; }
    static auto audio_header() -> std::string { return std::format("{:>10}", "max dB"); }
    static auto audio(const Stats& s) -> std::string { return std::format("{:>10.3g}", s.abs); }
};

#if defined(__ARM_NEON)
/// vsl::tanh's arithmetic: FMA Horner on vsl's coefficients, then
/// p * rsqrt(p * p + 1). Newton = -1 is the scalar 1 / sqrt; 0 is the bare
/// NEON estimate (vrsqrte, as vsl's vector path); 1 and 2 add Newton steps.
template<int Newton>
inline auto vsl_tanh(float x) -> float
{
    const auto u = x * x;
    auto r = 3.292487918e-06f;
    r = std::fma(r, u, 1.962047536e-04f);
    r = std::fma(r, u, 8.333356120e-03f);
    r = std::fma(r, u, 1.666701734e-01f);
    r = std::fma(r, u, 9.999960065e-01f);
    const auto p = x * r;
    const auto q = p * p + 1.0f;
    if constexpr (Newton < 0) {
        return p * (1.0f / std::sqrt(q));
    }
    else {
        auto e = vrsqrtes_f32(q);
        for (auto i = 0; i < Newton; ++i) e *= vrsqrtss_f32(q * e, e);
        return p * e;
    }
}
inline constexpr int neon_extras = 4;
#else
inline constexpr int neon_extras = 0;
#endif

struct Tanh {
    static constexpr const char* ref_name = "std::tanh";
    static constexpr const char* name = "tanh";
    static constexpr const char* what = "tanh x";
    static constexpr int generated = tanh_method_count;
    template<typename T, typename F> static void for_each(F&& f)
    {
        for_each_tanh_method<T>(f);
#if defined(__ARM_NEON)
        if constexpr (std::is_same_v<T, float>) {
            f(generated + 0, "vsl::tanh, 1/sqrt (scalar)", [](float x) { return vsl_tanh<-1>(x); });
            f(generated + 1, "vsl::tanh, vrsqrte (vector)", [](float x) { return vsl_tanh<0>(x); });
            f(generated + 2, "  + 1 Newton step", [](float x) { return vsl_tanh<1>(x); });
            f(generated + 3, "  + 2 Newton steps", [](float x) { return vsl_tanh<2>(x); });
        }
#endif
    }
    template<typename T> static auto reference(T x) -> T { return std::tanh(x); }
    static auto truth(double x) -> double { return std::tanh(x); }
    static auto ranges() -> std::vector<Range>
    {
        // Every normal float: past |x| ~ 556 the unclamped sigmoid's s * s overflows.
        constexpr auto lo = std::numeric_limits<float>::min(), hi = std::numeric_limits<float>::max();
        return {{lo, hi, false}, {lo, hi, true}};
    }
    static auto range_text() -> std::string { return "every normal x"; }
    static constexpr bool show_abs = true;
    static auto abs_mask(double) -> bool { return true; }
    static auto input(double u) -> double { return -5.0 + 10.0 * u; }
    static auto timing_text() -> std::string { return "x uniform in [-5, 5]"; }
    static auto audio_header() -> std::string { return std::format("{:>10} {:>12}", "abs dB", "max |y|"); }
    static auto audio(const Stats& s) -> std::string { return std::format("{:>10.1f} {:>12.9f}", db(s.abs), s.max_y); }
};

/// Methods for type T: the generated ones, plus any extras (float only).
template<typename Fn, typename T>
inline auto method_count() -> int
{
    auto n = 0;
    Fn::template for_each<T>([&](auto, const char*, auto) { ++n; });
    return n;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// The cold block's inputs: a ramp through the timing range over one second
/// at 48 kHz, as an envelope or a sweep would ask.
template<typename Fn, typename T>
struct Ramp {
    double u = 0.0;
    auto operator()() -> T
    {
        const auto y = static_cast<T>(Fn::input(u));
        u += 1.0 / 48000.0;
        if (u >= 1.0) u -= 1.0;
        return y;
    }
};

template<typename Fn, typename T>
inline auto run_timing(const char* type_name) -> void
{
    auto rng = std::mt19937_64{12345};
    auto dist = std::uniform_real_distribution<double>{0.0, 1.0};
    auto xs = std::vector<T>{};
    for (auto i = std::size_t{}; i < timing::N; ++i) xs.push_back(static_cast<T>(Fn::input(dist(rng))));

    auto rows = std::vector<timing::Row>{};
    rows.push_back(timing::time_method<T>(std::string{Fn::ref_name}, xs,
        [](T x) { return Fn::template reference<T>(x); }, Ramp<Fn, T>{}));
    Fn::template for_each<T>([&](auto, const char* name, auto fn) {
        rows.push_back(timing::time_method<T>(name, xs, fn, Ramp<Fn, T>{}));
    });
    timing::print_rows(rows, std::format("{}, {}: ns per sample, speedup vs {}; {}",
                                         Fn::what, type_name, Fn::ref_name, Fn::timing_text()),
                       std::format("{}-sample ramp through that range", timing::BLOCK));
}

// ---------------------------------------------------------------------------
// Error, f32
// ---------------------------------------------------------------------------

template<typename Fn>
inline auto run_error(std::uint32_t stride) -> void
{
    const auto n_methods = method_count<Fn, float>();
    const auto n_rows = static_cast<std::size_t>(n_methods + 1);  // + std:: reference
    auto names = std::vector<std::string>(n_rows);
    Fn::template for_each<float>([&](auto i, const char* name, auto) { names[static_cast<std::size_t>(int(i))] = name; });
    names.back() = std::string{Fn::ref_name};

    const auto start = std::chrono::steady_clock::now();
    const auto n_threads = err::thread_count;
    auto total = std::vector<Stats>(n_rows);
    for (const auto& range : Fn::ranges()) {
        const auto lo = std::bit_cast<std::uint32_t>(range.lo);
        const auto hi = std::bit_cast<std::uint32_t>(range.hi);
        const auto steps = (std::uint64_t{hi} - lo) / stride + 1;
        auto tables = std::vector<std::vector<Stats>>(n_threads, std::vector<Stats>(n_rows));
        auto threads = std::vector<std::thread>{};
        for (auto th = 0u; th < n_threads; ++th) {
            threads.emplace_back([&, th]() {
                auto& table = tables[th];
                for (auto k = steps * th / n_threads; k < steps * (th + 1) / n_threads; ++k) {
                    auto x = std::bit_cast<float>(static_cast<std::uint32_t>(lo + k * stride));
                    if (range.negate) x = -x;
                    const auto xd = static_cast<double>(x);
                    const auto t = Fn::truth(xd);
                    const auto masked = Fn::abs_mask(t);
                    Fn::template for_each<float>([&](auto i, const char*, auto fn) {
                        table[static_cast<std::size_t>(int(i))].add(static_cast<double>(fn(x)), t, xd, masked);
                    });
                    table.back().add(static_cast<double>(Fn::template reference<float>(x)), t, xd, masked);
                }
            });
        }
        for (auto& th : threads) th.join();
        for (const auto& table : tables) {
            for (auto i = std::size_t{}; i < n_rows; ++i) total[i].merge(table[i]);
        }
    }

    const auto what = stride == 1 ? std::string{"every"} : std::format("every {}th", stride);
    std::cout << std::format("\nError, {}, f32: {} float, {}\n", Fn::what, what, Fn::range_text());
    std::cout << std::format("  {:<28} {:>10} {:>14} {:>10} {:>10} {:>14}  {}\n",
                             "method", "max ULP", "at x", "max rel", "max abs", "at x", Fn::audio_header());
    for (auto i = std::size_t{}; i < n_rows; ++i) {
        const auto& s = total[i];
        const auto ulp = !std::isfinite(s.ulp) ? std::string{"inf"}
                         : s.ulp < 1e4         ? std::format("{:.2f}", s.ulp)
                                               : std::format("{:.2e}", s.ulp);
        const auto abs = Fn::show_abs ? std::format("{:>10.3g} {:>14.8g}", s.abs, s.abs_x)
                                      : std::format("{:>10} {:>14}", "—", "");
        std::cout << std::format("  {:<28} {:>10} {:>14.8g} {:>10.3g} {}  {}\n", names[i], ulp,
                                 s.ulp_x, s.rel, abs, Fn::audio(s));
    }
    const auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << std::format("  ({:.1f} s on {} threads)\n", secs, n_threads);
}

template<typename Fn>
inline auto run(const std::string& mode, bool f32, bool f64, std::uint32_t stride) -> void
{
    if (mode != "ulp") {
        if (f32) run_timing<Fn, float>("f32");
        if (f64) run_timing<Fn, double>("f64");
    }
    if (mode != "timing" && f32) run_error<Fn>(stride);
}

} // namespace adc::unary
