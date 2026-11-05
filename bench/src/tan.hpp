#pragma once
// tan(pi x) for filter cutoffs (examples/09-tan): timing and exhaustive f32
// error of every generated tan method, against std::tan(pi * x).
//
// Every method takes x = fc / fs in [0, 1/2); the pole is at Nyquist.
//
// Timing: as timing.hpp -- independent, vectorized, and a cold block -- with inputs
//   uniform in [0, 1/2). The cold block is a cutoff sweep, 20 Hz to 20 kHz
//   exponentially over one second at 48 kHz, the way a modulated filter asks
//   for a new g every sample.
//
// Error, f32: every normal float with FLT_MIN <= x < 1/2 (~1.06e9 inputs).
//   Truth is err::tan_pi (exact at the pole). Two metrics, as in 09-tan:
//     ULP     fractional f32 ULPs of the true g, in the audio band
//             (fc <= 20 kHz at 48 kHz) and all the way to Nyquist
//     cents   how far the filter's cutoff moves: it lands at atan(g) / pi
//   Methods are built for f32, so there's no f64 error pass.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "errors.hpp"
#include "tan_approx.hpp"
#include "timing.hpp"

namespace adc::tan_bench {

inline constexpr int STD_ROW = tan_method_count;  // accumulator index for the std::tan row
inline constexpr double AUDIO_TOP = 20000.0 / 48000.0;  // fc / fs; above this is "near Nyquist"

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Exponential cutoff sweep, 20 Hz to 20 kHz over one second at 48 kHz, then
/// back to the bottom.
template<typename T>
struct CutoffSweep {
    T x = T(20.0 / 48000.0);
    T ratio = T(std::pow(1000.0, 1.0 / 48000.0));
    auto operator()() -> T
    {
        const auto y = x;
        x *= ratio;
        if (x > T(20000.0 / 48000.0)) x = T(20.0 / 48000.0);
        return y;
    }
};

template<typename T>
inline auto run_timing(const char* type_name) -> void
{
    auto rng = std::mt19937_64{12345};
    auto dist = std::uniform_real_distribution<double>{0.0, 0.5};
    auto xs = std::vector<T>{};
    for (auto i = std::size_t{}; i < timing::N; ++i) xs.push_back(static_cast<T>(dist(rng)));

    auto rows = std::vector<timing::Row>{};
    rows.push_back(timing::time_method<T>("std::tan(pi x)", xs,
        [](T x) { return std::tan(T(std::numbers::pi) * x); }, CutoffSweep<T>{}));
    for_each_tan_method<T>([&](auto, const char* name, auto fn) {
        rows.push_back(timing::time_method<T>(name, xs, fn, CutoffSweep<T>{}));
    });
    timing::print_rows(rows, std::format("tan(pi x), {}: ns per sample, speedup vs std::tan(pi x)", type_name),
                       std::format("{}-sample 20 Hz - 20 kHz cutoff sweep", timing::BLOCK));
}

// ---------------------------------------------------------------------------
// Error, f32
// ---------------------------------------------------------------------------

/// Running statistics for one method.
struct Stats {
    double ulp_audio{};
    double ulp_all{};
    double arg_max{};       // input with the largest ULP error
    double cents_audio{};
    double cents_top{};
    std::uint64_t count{};

    void add(double ulps, double cents, double x)
    {
        const auto audio = x <= AUDIO_TOP;
        if (audio) {
            ulp_audio = std::max(ulp_audio, ulps);
            cents_audio = std::max(cents_audio, cents);
        }
        else {
            cents_top = std::max(cents_top, cents);
        }
        if (ulps > ulp_all || count == 0) {
            ulp_all = ulps;
            arg_max = x;
        }
        ++count;
    }

    void merge(const Stats& o)
    {
        if (o.count && (o.ulp_all > ulp_all || count == 0)) {
            ulp_all = o.ulp_all;
            arg_max = o.arg_max;
        }
        ulp_audio = std::max(ulp_audio, o.ulp_audio);
        cents_audio = std::max(cents_audio, o.cents_audio);
        cents_top = std::max(cents_top, o.cents_top);
        count += o.count;
    }
};

using Table = std::vector<Stats>;  // tan_method_count methods, then std::tan

inline auto fmt_ulp(double u) -> std::string
{
    if (!std::isfinite(u)) return "inf";
    return u < 1e4 ? std::format("{:.2f}", u) : std::format("{:.2e}", u);
}

inline auto fmt_cents(double c) -> std::string
{
    return std::isfinite(c) ? std::format("{:.2g}", c) : "inf (g<=0)";
}

inline auto run_ulp(std::uint32_t stride) -> void
{
    constexpr auto lo = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::min());
    constexpr auto hi = std::bit_cast<std::uint32_t>(0.5f) - 1;  // largest float below 1/2
    const auto start = std::chrono::steady_clock::now();
    const auto n_threads = err::thread_count;
    const auto steps = (std::uint64_t{hi} - lo) / stride + 1;

    auto tables = std::vector<Table>(n_threads, Table(tan_method_count + 1));
    auto names = std::vector<std::string>(tan_method_count + 1);
    for_each_tan_method<float>([&](auto i, const char* name, auto) { names[static_cast<std::size_t>(int(i))] = name; });
    names[STD_ROW] = "std::tan(pi x)";

    auto threads = std::vector<std::thread>{};
    for (auto t = 0u; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            auto& table = tables[t];
            const auto begin = steps * t / n_threads;
            const auto end = steps * (t + 1) / n_threads;
            for (auto k = begin; k < end; ++k) {
                const auto x = std::bit_cast<float>(static_cast<std::uint32_t>(lo + k * stride));
                const auto xd = static_cast<double>(x);
                const auto truth = err::tan_pi(xd);
                const auto u = err::ulp_f32(truth);
                const auto one = [&](std::size_t i, float y) {
                    const auto g = static_cast<double>(y);
                    const auto ulps = std::isfinite(g) ? std::abs(g - truth) / u : std::numeric_limits<double>::infinity();
                    table[i].add(ulps, err::cutoff_cents(g, xd), xd);
                };
                for_each_tan_method<float>([&](auto i, const char*, auto fn) { one(static_cast<std::size_t>(int(i)), fn(x)); });
                one(STD_ROW, std::tan(std::numbers::pi_v<float> * x));
            }
        });
    }
    for (auto& th : threads) th.join();
    auto total = tables[0];
    for (auto t = 1u; t < n_threads; ++t) {
        for (auto i = std::size_t{}; i < total.size(); ++i) total[i].merge(tables[t][i]);
    }

    const auto what = stride == 1 ? std::string{"every"} : std::format("every {}th", stride);
    std::cout << std::format("\nError, tan(pi x), f32: {} normal float with FLT_MIN <= x < 1/2 (x = fc / fs)\n", what);
    std::cout << std::format("  {:<22} {:>10} {:>10} {:>14}  {:>10} {:>10}\n",
                             "method", "max ULP", "max ULP", "worst x", "max cents", "max cents");
    std::cout << std::format("  {:<22} {:>10} {:>10} {:>14}  {:>10} {:>10}\n",
                             "", "<= 20 kHz", "to Nyq", "", "<= 20 kHz", "> 20 kHz");
    for (auto i = std::size_t{}; i < total.size(); ++i) {
        const auto& s = total[i];
        std::cout << std::format("  {:<22} {:>10} {:>10} {:>14.9g}  {:>10} {:>10}\n", names[i],
                                 fmt_ulp(s.ulp_audio), fmt_ulp(s.ulp_all), s.arg_max,
                                 fmt_cents(s.cents_audio), fmt_cents(s.cents_top));
    }
    const auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << std::format("  (20 kHz at fs = 48 kHz; {:.1f} s on {} threads)\n", secs, n_threads);
}

} // namespace adc::tan_bench
