// What each degree buys: exhaustive f32 error, f64 fit error and the three
// loops from bench/simple/loops.cpp, for every method in degree_approx.hpp.
// See degree_effects.py, which emits the header and reads the output.
//
//   clang++ -std=c++23 -O3 -DNDEBUG examples/18-degree-effects/degree_bench.cpp -o /tmp/degree_bench
//   /tmp/degree_bench errors > examples/18-degree-effects/results/errors.txt
//   for i in 1 2 3 4 5; do /tmp/degree_bench timing; done > examples/18-degree-effects/results/timing.txt
//
// A family name after the mode runs only that family's methods. The error checks
// use half the cores; --threads n, last on the line, changes that.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "degree_approx.hpp"

// ---------------------------------------------------------------------------
// The functions: the exact value, and where errors are taken
// ---------------------------------------------------------------------------

auto sin_turns(double x) -> double
{
    const auto n = std::nearbyint(4.0 * x);
    const auto r = 2.0 * std::numbers::pi * (x - n / 4.0);
    switch (static_cast<std::int64_t>(n) & 3) {
        case 0: return std::sin(r);
        case 1: return std::cos(r);
        case 2: return -std::sin(r);
        default: return -std::cos(r);
    }
}

struct Domain {
    float hi;                   // every normal float in (0, hi]
    float abs_lo, abs_hi;       // max abs over this part (log2's grows with |e|)
    double grid_lo, grid_hi;    // f64 fit error, on an even grid
};

auto domain(adc::Family family) -> Domain
{
    using enum adc::Family;
    constexpr auto inf = std::numeric_limits<float>::infinity();
    switch (family) {
        case sin_radians:
        case sin_split: return {std::numbers::pi_v<float>, 0.0f, inf, 0.0, std::numbers::pi};
        case sin_turns: return {1.0f, 0.0f, inf, 0.0, 1.0};
        case log2_fold:
        case log2_plain: return {std::numeric_limits<float>::max(), 0x1p-20f, 0x1p10f, 0.5, 2.0};
        default: return {20.0f, 0.0f, inf, 0.0, 9.5};
    }
}

auto truth(adc::Family family, double x) -> double
{
    using enum adc::Family;
    switch (family) {
        case sin_radians:
        case sin_split: return std::sin(x);
        case sin_turns: return ::sin_turns(x);
        case log2_fold:
        case log2_plain: return std::log2(x);
        default: return std::tanh(x);
    }
}

// ---------------------------------------------------------------------------
// Errors: every float in f32, an even grid in f64 (the fit alone, no rounding)
// ---------------------------------------------------------------------------

auto ulp_f32(double t) -> double
{
    auto e = 0;
    std::frexp(std::abs(t), &e);
    return std::ldexp(1.0, std::max(e - 24, -149));
}

struct Error { double abs = 0, rel = 0, ulp = 0, ulp_at = 0; };

auto thread_count = std::max(1u, std::thread::hardware_concurrency() / 2);  // --threads

auto f32_error(adc::Family family, auto f) -> Error
{
    const auto d = domain(family);
    const auto lo = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::min());
    const auto hi = std::bit_cast<std::uint32_t>(d.hi);
    const auto n_threads = thread_count;
    auto worst = std::vector<Error>(n_threads);
    auto threads = std::vector<std::thread>{};
    for (auto t = 0u; t < n_threads; ++t) {
        threads.emplace_back([&, t] {
            const auto span = std::uint64_t{hi} - lo + 1;
            const auto b0 = lo + span * t / n_threads, b1 = lo + span * (t + 1) / n_threads;
            auto e = Error{};
            for (auto b = b0; b < b1; ++b) {
                const auto x = std::bit_cast<float>(static_cast<std::uint32_t>(b));
                const auto tr = truth(family, double(x));
                const auto err = std::abs(double(f(x)) - tr);
                if (x >= d.abs_lo && x <= d.abs_hi) e.abs = std::max(e.abs, err);
                if (tr == 0.0) continue;  // the exact zeros: rel and ULP are about err / tiny
                e.rel = std::max(e.rel, err / std::abs(tr));
                const auto ulp = err / ulp_f32(tr);
                if (ulp > e.ulp) { e.ulp = ulp; e.ulp_at = x; }
            }
            worst[t] = e;
        });
    }
    for (auto& th : threads) th.join();
    auto r = Error{};
    for (const auto& e : worst) {
        r.abs = std::max(r.abs, e.abs);
        r.rel = std::max(r.rel, e.rel);
        if (e.ulp > r.ulp) { r.ulp = e.ulp; r.ulp_at = e.ulp_at; }
    }
    return r;
}

auto f64_error(adc::Family family, auto f) -> Error
{
    constexpr auto N = 1 << 22;
    const auto d = domain(family);
    auto e = Error{};
    for (auto i = 0; i < N; ++i) {
        const auto x = d.grid_lo + (d.grid_hi - d.grid_lo) * (i + 0.5) / N;
        const auto tr = truth(family, x);
        const auto err = std::abs(f(x) - tr);
        e.abs = std::max(e.abs, err);
        if (std::abs(tr) > 1e-300) e.rel = std::max(e.rel, err / std::abs(tr));
    }
    return e;
}

// ---------------------------------------------------------------------------
// Timing: bench/simple/loops.cpp's three loops, sorted inputs
// ---------------------------------------------------------------------------

constexpr auto SAMPLES = std::size_t{65536};
constexpr auto RUNS = std::size_t{100};

// Chandler Carruth, CppCon 2015: https://www.youtube.com/watch?v=nXaxk27zwlk
auto escape(void* p) -> void { asm volatile("" : : "g"(p) : "memory"); }

auto bench(auto loop) -> double
{
    using clock = std::chrono::steady_clock;
    auto ns = std::vector<double>(RUNS);
    for (auto& t : ns) {
        const auto start = clock::now();
        loop();
        const auto elapsed = clock::now() - start;
        t = std::chrono::duration<double, std::nano>(elapsed).count();
    }
    std::ranges::sort(ns);
    return ns[RUNS / 2] / double(SAMPLES);
}

auto buffer(auto f, std::span<const float> in, std::span<float> out) -> void
{
    for (auto i = 0uz; i < in.size(); ++i)
        out[i] = f(in[i]);
    escape(out.data());
}

auto sample(auto f, std::span<const float> in, std::span<float> out) -> void
{
    for (auto i = 0uz; i < in.size(); ++i) {
        out[i] = f(in[i]);
        escape(&out[i]);
    }
}

auto feedback(auto f, std::span<const float> in) -> void
{
    auto y = 0.0f;
    for (auto i = 0uz; i < in.size(); ++i)
        y = f(in[i] + 0.0f * y);
    escape(&y);
}

auto inputs(adc::Family family) -> std::vector<float>
{
    using enum adc::Family;
    constexpr auto pi = std::numbers::pi_v<float>;
    auto rng = std::mt19937{1};
    auto unit = std::uniform_real_distribution<float>{0, 1};
    auto in = std::vector<float>(SAMPLES);
    for (auto& x : in) {
        const auto u = unit(rng);
        switch (family) {
            case sin_radians:
            case sin_split: x = 2 * pi * u - pi; break;
            case sin_turns: x = u; break;
            case log2_fold:
            case log2_plain: x = std::exp2(30 * u - 20); break;
            default: x = 10 * u - 5; break;
        }
    }
    std::ranges::sort(in);
    return in;
}

// ---------------------------------------------------------------------------

auto main(int argc, char** argv) -> int
{
    if (argc >= 3 && !std::strcmp(argv[argc - 2], "--threads")) {
        thread_count = std::max(1u, unsigned(std::atoi(argv[argc - 1])));
        argc -= 2;
    }
    const auto errors = argc > 1 && !std::strcmp(argv[1], "errors");
    const auto only = argc > 2 ? argv[2] : nullptr;
    auto out = std::vector<float>(SAMPLES);
    auto row = [&](adc::Family family, const char* fam, int degree, const char* name, auto f32, auto f64) {
        if (only && std::strcmp(only, fam)) return;
        if (errors) {
            const auto e = f32_error(family, f32);
            const auto g = f64_error(family, f64);
            std::printf("E %s %d %s %.4g %.4g %.4g %.9g %.4g %.4g\n", fam, degree, name, e.abs, e.rel, e.ulp, e.ulp_at, g.abs, g.rel);
        }
        else {
            const auto in = inputs(family);
            const auto b = bench([&] { buffer(f32, in, out); });
            const auto s = bench([&] { sample(f32, in, out); });
            const auto fb = bench([&] { feedback(f32, in); });
            std::printf("T %s %d %s %.3f %.3f %.3f\n", fam, degree, name, b, s, fb);
        }
        std::fflush(stdout);
    };
    using enum adc::Family;
    row(sin_radians, "sin_radians", 0, "std::sin", [](float x) { return std::sin(x); }, [](double x) { return std::sin(x); });
    row(sin_turns, "sin_turns", 0, "std::sin", [](float x) { return std::sin(2 * std::numbers::pi_v<float> * x); },
        [](double x) { return std::sin(2 * std::numbers::pi * x); });
    row(log2_fold, "log2_fold", 0, "std::log2", [](float x) { return std::log2(x); }, [](double x) { return std::log2(x); });
    row(tanh_sigmoid, "tanh_sigmoid", 0, "std::tanh", [](float x) { return std::tanh(x); }, [](double x) { return std::tanh(x); });
    adc::for_each_degree_method([&](adc::Family family, const char* fam, int degree, const char* name, auto f32, auto f64) {
        row(family, fam, degree, name, f32, f64);
    });
}
