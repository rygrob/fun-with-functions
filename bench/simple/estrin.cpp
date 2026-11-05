// Horner vs Estrin for the degree-13 sin polynomials: speed in three
// settings (vectorized, scalar independent, each sample waiting on the last)
// and max ULP over every float in range. Background for notes/latency-horner-estrin.md;
// not on the slides.
//
//   cmake --build bench/build && bench/build/adc_simple_estrin [--threads n]
//
// Critical path, at ~4 cycles per fmul/fma (fsub ~3):
//   taylor_sin_13  Horner  s=x*x, 6 fma, *x               = 8 ops ~ 32 cycles
//                  Estrin  s, s2, s4, 1 fma, *x            = 5 ops ~ 20 cycles
//   cheb_sin_13    Horner  s=x*x, 6 fma, *x               = 8 ops ~ 32 cycles
//                  Estrin  s, s2, s4, 1 fma, *x            = 5 ops ~ 20 cycles
//   cheb_turns_13  Horner  u=x-.5, s, 5 fma, *factors     = 8 ops ~ 31 cycles
//                  Estrin  u, s, s2, s4, 1 fma, *factors  = 6 ops ~ 23 cycles
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include "sin_approx.hpp"

// ---------------------------------------------------------------------------
// Estrin forms, same coefficients as the generated Horner ones
// ---------------------------------------------------------------------------

inline auto estrin_taylor_sin_13(float x) -> float
{
    const auto s = x * x;
    const auto s2 = s * s;
    const auto s4 = s2 * s2;
    const auto a = 1.0f + -0.16666666666666666f * s;
    const auto b = 0.008333333333333333f + -0.0001984126984126984f * s;
    const auto c = 2.7557319223985893e-06f + -2.505210838544172e-08f * s;
    const auto lo = a + b * s2;
    const auto hi = c + 1.6059043836821613e-10f * s2;
    return x * (lo + hi * s4);
}

inline auto estrin_sin_13(float x) -> float
{
    const auto s = x * x;
    const auto s2 = s * s;
    const auto s4 = s2 * s2;
    const auto a = 0.9999999944722884f + -0.16666664569429987f * s;
    const auto b = 0.008333310291714583f + -0.00019840151861339985f * s;
    const auto c = 2.752939532139805e-06f + -2.4676492417399503e-08f * s;
    const auto lo = a + b * s2;
    const auto hi = c + 1.344998965622804e-10f * s2;
    return x * (lo + hi * s4);
}

inline auto estrin_turns_sin_13(float x) -> float
{
    const auto u = x - 0.5f;
    const auto s = u * u;
    const auto s2 = s * s;
    const auto s4 = s2 * s2;
    const auto a = -25.132741155387375f + 64.83582285642966f * s;
    const auto b = -67.07662737709583f + 38.49587885367295f * s;
    const auto c = -14.049663725541881f + 3.161601856720294f * s;
    const auto p = (a + b * s2) + c * s4;
    return -u * x * (x - 1.0f) * p;
}

/// Estrin with the final multiply folded into both halves, off the s4 path:
/// x * (lo + hi * s4) = x * lo + (x * hi) * s4.
inline auto estrin_fold_sin_13(float x) -> float
{
    const auto s = x * x;
    const auto s2 = s * s;
    const auto s4 = s2 * s2;
    const auto a = 0.9999999944722884f + -0.16666664569429987f * s;
    const auto b = 0.008333310291714583f + -0.00019840151861339985f * s;
    const auto c = 2.752939532139805e-06f + -2.4676492417399503e-08f * s;
    const auto lo = x * (a + b * s2);
    const auto hi = x * (c + 1.344998965622804e-10f * s2);
    return lo + hi * s4;
}

inline auto estrin_fold_turns_sin_13(float x) -> float
{
    const auto u = x - 0.5f;
    const auto f = -u * x * (x - 1.0f);
    const auto s = u * u;
    const auto s2 = s * s;
    const auto s4 = s2 * s2;
    const auto a = -25.132741155387375f + 64.83582285642966f * s;
    const auto b = -67.07662737709583f + 38.49587885367295f * s;
    const auto c = -14.049663725541881f + 3.161601856720294f * s;
    return f * (a + b * s2) + (f * c) * s4;
}

// ---------------------------------------------------------------------------
// Timing: the slide method, three loop shapes
// ---------------------------------------------------------------------------

constexpr auto SAMPLES = std::size_t{65536};
constexpr auto RUNS = std::size_t{100};
auto escape(void* p) -> void { asm volatile("" : : "g"(p) : "memory"); }

enum class Loop { vector, scalar, chain };

template<Loop L>
[[gnu::noinline]] auto bench(auto f, std::span<const float> in, std::span<float> out) -> double
{
    using clock = std::chrono::steady_clock;
    auto ns = std::vector<double>(RUNS);
    for (auto& t : ns) {
        const auto start = clock::now();
        if constexpr (L == Loop::vector) {
            for (auto i = 0uz; i < in.size(); ++i)
                out[i] = f(in[i]);
        }
        else if constexpr (L == Loop::scalar) {
#pragma clang loop vectorize(disable) interleave(disable)
            for (auto i = 0uz; i < in.size(); ++i)
                out[i] = f(in[i]);
        }
        else {
            auto y = 0.0f;  // 0 * y can't be folded without fast-math
            for (auto i = 0uz; i < in.size(); ++i) {
                y = f(in[i] + 0.0f * y);
                out[i] = y;
            }
        }
        escape(out.data());
        t = std::chrono::duration<double, std::nano>(clock::now() - start).count();
    }
    std::ranges::sort(ns);
    return ns[RUNS / 2] / double(SAMPLES);
}

template<Loop L>
void timing_row(const char* loop, std::span<const float> rad, std::span<const float> trn, std::span<float> out)
{
    const auto lib = bench<L>([](float x) { return std::sin(x); }, rad, out);
    const auto th = bench<L>([](float x) { return adc::taylor_sin_13(x); }, rad, out);
    const auto te = bench<L>([](float x) { return estrin_taylor_sin_13(x); }, rad, out);
    const auto h = bench<L>([](float x) { return adc::cheb_sin_13(x); }, rad, out);
    const auto e = bench<L>([](float x) { return estrin_sin_13(x); }, rad, out);
    const auto ht = bench<L>([](float x) { return adc::cheb_turns_sin_13(x); }, trn, out);
    const auto et = bench<L>([](float x) { return estrin_turns_sin_13(x); }, trn, out);
    const auto ef = bench<L>([](float x) { return estrin_fold_sin_13(x); }, rad, out);
    const auto eft = bench<L>([](float x) { return estrin_fold_turns_sin_13(x); }, trn, out);
    std::printf("  %-8s %6.2f   %6.2f  %6.2f   %6.2f  %6.2f  %6.2f   %6.2f  %6.2f  %6.2f\n", loop, lib, th, te, h, e, ef, ht, et, eft);
}

// ---------------------------------------------------------------------------
// Max ULP over every normal float in range (both signs for radians)
// ---------------------------------------------------------------------------

auto ulp_f32(double t) -> double
{
    auto e = 0;
    std::frexp(std::abs(t), &e);
    return std::ldexp(1.0, std::max(e - 24, -149));
}

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

auto thread_count = std::max(1u, std::thread::hardware_concurrency() / 2);  // --threads

auto max_ulp(auto f, auto truth, float hi, bool both_signs) -> double
{
    const auto lo_bits = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::min());
    const auto hi_bits = std::bit_cast<std::uint32_t>(hi);
    const auto n_threads = thread_count;
    auto worst = std::vector<double>(n_threads);
    auto threads = std::vector<std::thread>{};
    for (auto t = 0u; t < n_threads; ++t) {
        threads.emplace_back([&, t] {
            const auto span = std::uint64_t{hi_bits} - lo_bits + 1;
            const auto b0 = lo_bits + span * t / n_threads, b1 = lo_bits + span * (t + 1) / n_threads;
            auto w = 0.0;
            for (auto b = b0; b < b1; ++b) {
                const auto x = std::bit_cast<float>(static_cast<std::uint32_t>(b));
                for (auto s : {x, -x}) {
                    const auto tr = truth(double(s));
                    w = std::max(w, std::abs(double(f(s)) - tr) / ulp_f32(tr));
                    if (!both_signs) break;
                }
            }
            worst[t] = w;
        });
    }
    for (auto& th : threads) th.join();
    return *std::ranges::max_element(worst);
}

auto main(int argc, char** argv) -> int
{
    if (argc == 3 && std::string_view{argv[1]} == "--threads") thread_count = std::max(1u, unsigned(std::atoi(argv[2])));
    constexpr auto pi = std::numbers::pi_v<float>;
    auto rng = std::mt19937{1};
    auto dist = std::uniform_real_distribution<float>{0.0f, 1.0f};
    auto trn = std::vector<float>(SAMPLES), rad = std::vector<float>(SAMPLES), out = std::vector<float>(SAMPLES);
    for (auto& x : trn) x = dist(rng);
    std::ranges::sort(trn);
    for (auto i = 0uz; i < SAMPLES; ++i) rad[i] = 2 * pi * trn[i] - pi;

    std::printf("ns per sample            Taylor            radians                  turns\n");
    std::printf("  loop     std::sin  Horner  Estrin   Horner  Estrin    fold   Horner  Estrin    fold\n");
    timing_row<Loop::vector>("vector", rad, trn, out);
    timing_row<Loop::scalar>("scalar", rad, trn, out);
    timing_row<Loop::chain>("chain", rad, trn, out);

    std::printf("max ULP, turns, every float in [0, 1]: Horner %.3f  Estrin %.3f  fold %.3f\n",
                max_ulp([](float x) { return adc::cheb_turns_sin_13(x); }, sin_turns, 1.0f, false),
                max_ulp([](float x) { return estrin_turns_sin_13(x); }, sin_turns, 1.0f, false),
                max_ulp([](float x) { return estrin_fold_turns_sin_13(x); }, sin_turns, 1.0f, false));
}
