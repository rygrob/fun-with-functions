#pragma once
// ULP error of every generated method, plus max absolute error in dB.
//
// f32: every normal float in the method's range (or every stride-th one).
//   radians  FLT_MIN <= |x| <= f32(pi), both signs   ~2.1e9 inputs
//   turns    FLT_MIN <= x <= 1                         ~1.1e9 inputs
//   phase    every 32-bit phase (stride-th)            ~4.3e9 inputs
//   Truth is f64: std::sin(x), or an exactly-reduced sin(2 pi x). Error is
//   fractional ULPs of the true value, as in the Python examples.
//
//   Phase inputs have *absolute* precision (2^-32 turns), so near the zeros
//   no method can have small relative error and ULP numbers there mean
//   little; compare phase methods by max abs error (dB).
//
// f64: exhaustive is out of reach (and there's no reference more precise
//   than f64 here), so 2^24 random inputs per domain, measured in f64 ULPs
//   against std::sin -- itself within about 1 ULP, so read small numbers
//   loosely.

#include <algorithm>
#include <bit>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "errors.hpp"
#include "sin_approx.hpp"

namespace adc::ulp {

using err::UlpStats;

inline constexpr int STD_ROW = method_count;  // accumulator index for the std::sin row
inline constexpr double TWO_POW_M32 = 1.0 / 4294967296.0;

struct Options {
    std::uint32_t stride = 1;
    std::string domain = "all";     // radians | turns | phase | all
    std::size_t dump_worst = 0;     // keep this many worst inputs per method
    std::string dump_dir = ".";     // ...and write them to <dir>/worst_<method>.txt
};

inline auto dump_worst_n = std::size_t{0};

using Worst = std::vector<std::pair<double, double>>;  // (ulps, x), a min-heap on ulps

inline auto keep_worst(Worst& h, double ulps, double x) -> void
{
    if (h.size() < dump_worst_n) {
        h.emplace_back(ulps, x);
        std::ranges::push_heap(h, std::greater{});
    }
    else if (ulps > h.front().first) {
        std::ranges::pop_heap(h, std::greater{});
        h.back() = {ulps, x};
        std::ranges::push_heap(h, std::greater{});
    }
}

struct Table {
    std::vector<std::string> names = std::vector<std::string>(method_count + 1);
    std::vector<UlpStats> stats = std::vector<UlpStats>(method_count + 1);
    std::vector<bool> used = std::vector<bool>(method_count + 1);
    std::vector<Worst> worst = std::vector<Worst>(method_count + 1);

    void add(int i, const char* name, double y, double truth, double ulp, double x, bool exact)
    {
        const auto ui = static_cast<std::size_t>(i);
        const auto abs_err = std::abs(y - truth);
        stats[ui].add(abs_err / ulp, abs_err, x, exact);
        if (dump_worst_n) keep_worst(worst[ui], abs_err / ulp, x);
        if (!used[ui]) { used[ui] = true; names[ui] = name; }
    }
};

/// Write each method's worst inputs, worst first: "x<TAB>ulps" per line.
inline auto dump(const Table& t, const std::string& dir) -> void
{
    if (!dump_worst_n) return;
    for (auto i = 0; i <= method_count; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        if (!t.used[ui]) continue;
        auto w = t.worst[ui];
        std::ranges::sort(w, std::greater{});
        auto name = t.names[ui];
        std::ranges::replace_if(name, [](char c) { return !std::isalnum(static_cast<unsigned char>(c)) && c != '_'; }, '_');
        auto out = std::ofstream{dir + "/worst_" + name + ".txt"};
        for (const auto& [ulps, x] : w) out << std::format("{:.9g}\t{:.6g}\n", x, ulps);
    }
}

inline auto print(const Table& t, const std::string& title) -> void
{
    std::cout << "\n" << title << "\n";
    std::cout << std::format("  {:<26} {:>12} {:>16} {:>10} {:>9} {:>9} {:>10}\n",
                             "method", "max ULP", "at x", "mean ULP", "exact %", "<1 ULP %", "max abs dB");
    for (auto i = 0; i <= method_count; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        if (!t.used[ui]) continue;
        const auto& s = t.stats[ui];
        std::cout << std::format("  {:<26} {:>12.4g} {:>16.9g} {:>10.4g} {:>9.3f} {:>9.3f} {:>10.1f}\n",
                                 t.names[ui], s.max, s.arg_max, s.mean(),
                                 s.pct(s.exact), s.pct(s.under_1), s.max_abs_db());
    }
}

/// Visit every stride-th 32-bit pattern in [lo, hi], split across threads;
/// fn(bits, table) accumulates into a per-thread table.
template<typename F>
inline auto parallel_bits(std::uint32_t lo, std::uint32_t hi, std::uint32_t stride, F fn) -> Table
{
    const auto n_threads = err::thread_count;
    const auto steps = (std::uint64_t{hi} - lo) / stride + 1;
    auto tables = std::vector<Table>(n_threads);
    auto threads = std::vector<std::thread>{};
    for (auto t = 0u; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            const auto begin = steps * t / n_threads;
            const auto end = steps * (t + 1) / n_threads;
            for (auto k = begin; k < end; ++k) {
                fn(static_cast<std::uint32_t>(lo + k * stride), tables[t]);
            }
        });
    }
    for (auto& th : threads) th.join();
    auto out = tables[0];
    for (auto t = 1u; t < n_threads; ++t) {
        for (auto i = 0; i <= method_count; ++i) {
            const auto ui = static_cast<std::size_t>(i);
            out.stats[ui].merge(tables[t].stats[ui]);
            out.used[ui] = out.used[ui] || tables[t].used[ui];
            if (out.names[ui].empty()) out.names[ui] = tables[t].names[ui];
            for (const auto& [ulps, x] : tables[t].worst[ui]) keep_worst(out.worst[ui], ulps, x);
        }
    }
    return out;
}

template<Domain D>
inline auto f32_domain(std::uint32_t stride) -> Table
{
    if constexpr (D == Domain::phase) {
        return parallel_bits(0u, 0xFFFFFFFFu, stride, [](std::uint32_t phase, Table& t) {
            const auto x = static_cast<double>(phase) * TWO_POW_M32;
            const auto truth = err::sin_turns(x);
            const auto truth_f = static_cast<float>(truth);
            const auto u = err::ulp_f32(truth);
            for_each_method<float>([&](auto i, const char* name, auto domain, auto fn) {
                if constexpr (domain == D) {
                    const auto y = fn(phase);
                    t.add(i, name, y, truth, u, x, y == truth_f);
                }
            });
            const auto y = std::sin(static_cast<float>(2.0 * std::numbers::pi) * (static_cast<float>(phase) * float(TWO_POW_M32)));
            t.add(STD_ROW, "std::sin(2pi phase/2^32)", y, truth, u, x, y == truth_f);
        });
    }
    else {
        constexpr auto lo = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::min());
        constexpr auto hi = std::bit_cast<std::uint32_t>(D == Domain::turns ? 1.0f : std::numbers::pi_v<float>);
        return parallel_bits(lo, hi, stride, [](std::uint32_t bits, Table& t) {
            const auto one = [&](float x) {
                const auto truth = D == Domain::turns ? err::sin_turns(x) : std::sin(static_cast<double>(x));
                const auto truth_f = static_cast<float>(truth);
                const auto u = err::ulp_f32(truth);
                for_each_method<float>([&](auto i, const char* name, auto domain, auto fn) {
                    if constexpr (domain == D) {
                        const auto y = fn(x);
                        t.add(i, name, y, truth, u, x, y == truth_f);
                    }
                });
                const auto y = D == Domain::turns ? std::sin(static_cast<float>(2.0 * std::numbers::pi) * x) : std::sin(x);
                t.add(STD_ROW, D == Domain::turns ? "std::sin(2pi x)" : "std::sin(x)", y, truth, u, x, y == truth_f);
            };
            const auto xp = std::bit_cast<float>(bits);
            one(xp);
            if constexpr (D == Domain::radians) one(-xp);  // turns range is [0, 1] only
        });
    }
}

template<Domain D>
inline auto f64_domain(std::size_t samples) -> Table
{
    auto rng = std::mt19937_64{static_cast<std::uint64_t>(D) + 1};
    auto unit = std::uniform_real_distribution<double>{0.0, 1.0};
    auto t = Table{};
    for (auto k = std::size_t{}; k < samples; ++k) {
        const auto u01 = unit(rng);
        const auto phase = static_cast<std::uint32_t>(u01 * 4294967296.0);
        const auto x = D == Domain::radians ? 2.0 * std::numbers::pi * u01 - std::numbers::pi
                     : D == Domain::turns   ? u01
                                            : static_cast<double>(phase) * TWO_POW_M32;
        const auto truth = D == Domain::radians ? std::sin(x) : err::sin_turns(x);
        const auto ulp = err::ulp_f64(truth);
        for_each_method<double>([&](auto i, const char* name, auto domain, auto fn) {
            if constexpr (domain == D) {
                double y = 0.0;
                if constexpr (D == Domain::phase) y = fn(phase);
                else y = fn(x);
                t.add(i, name, y, truth, ulp, x, y == truth);
            }
        });
        if constexpr (D != Domain::radians) {  // radians: std::sin is the reference, so no row
            const auto y = std::sin(2.0 * std::numbers::pi * x);
            t.add(STD_ROW, "std::sin(2pi x)", y, truth, ulp, x, y == truth);
        }
    }
    return t;
}

inline auto run_f32(const Options& opt) -> void
{
    dump_worst_n = opt.dump_worst;
    const auto stride = opt.stride;
    const auto want = [&](const char* d) { return opt.domain == "all" || opt.domain == d; };
    const auto what = stride == 1 ? std::string{"every"} : std::format("every {}th", stride);
    const auto start = std::chrono::steady_clock::now();
    const auto finish = [&](const Table& t, const std::string& title) { print(t, title); dump(t, opt.dump_dir); };
    if (want("radians")) finish(f32_domain<Domain::radians>(stride), std::format("ULP error, f32, radians: {} normal float with FLT_MIN <= |x| <= f32(pi)", what));
    if (want("turns")) finish(f32_domain<Domain::turns>(stride), std::format("ULP error, f32, turns: {} normal float with FLT_MIN <= x <= 1", what));
    if (want("phase")) finish(f32_domain<Domain::phase>(stride), std::format("ULP error, f32, phase: {} 32-bit phase (compare by max abs dB)", what));
    if (opt.dump_worst) std::cout << std::format("  (wrote the {} worst inputs per method to {}/worst_<method>.txt)\n", opt.dump_worst, opt.dump_dir);
    const auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << std::format("  ({:.1f} s on {} threads)\n", secs, err::thread_count);
}

inline auto run_f64(std::size_t samples) -> void
{
    print(f64_domain<Domain::radians>(samples),
          std::format("ULP error, f64, radians: {} random x in [-pi, pi], f64 ULPs vs std::sin", samples));
    print(f64_domain<Domain::turns>(samples),
          std::format("ULP error, f64, turns: {} random x in [0, 1], f64 ULPs vs exact-reduced sin(2 pi x)", samples));
    print(f64_domain<Domain::phase>(samples),
          std::format("ULP error, f64, phase: {} random 32-bit phases, f64 ULPs vs exact-reduced sin(2 pi x)", samples));
}

} // namespace adc::ulp
