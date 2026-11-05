// tanh in a feedback loop (notes/tanh-feedback.md): std::tanh, the shipped
// sigmoid (Horner and Estrin), Pade, a piecewise table for the dependent case
// and minimax rationals, in the three loops of bench/simple/loops.cpp, with
// the max abs error over every float in [0, 9]. Tables: fit_pw.py and
// remez_rational.py (emit pw_tanh_table.inc and rational_coeffs.inc).
//
//   clang++ -std=c++23 -O3 -I bench/generated \
//       notes/tanh-feedback/tanh_feedback.cpp -o /tmp/tanh_feedback && /tmp/tanh_feedback [--threads n]
#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "tanh_approx.hpp"

/// sigmoid_clamp_tanh_9 with R(x^2) by Estrin: r = (c0 + c1 s) + s2 ((c2 + c3 s) + c4 s2).
inline auto estrin_sigmoid_tanh_9(float x) -> float
{
    const auto xc = std::fmin(std::fmax(x, -9.0f), 9.0f);
    const auto s = xc * xc;
    const auto s2 = s * s;
    const auto a = 1.0000045841180956f + 0.1666433506348483f * s;
    const auto b = 0.008358021824318818f + 0.00019032944649238027f * s;
    const auto r = a + s2 * (b + 3.638146764843207e-06f * s2);
    const auto y = xc * r;
    return y / std::sqrt(y * y + 1.0f);
}

/// tanh for a feedback loop: |x| picks one of 18 segments of width 1/2 on
/// [0, 9]; each is a degree-5 polynomial in t = |x| - center, by Estrin.
/// The chain: |x|, clamp, index, one load, t, three fma levels, copysign.
alignas(64) inline constexpr float pw_tanh_table[18][7] = {
#include "pw_tanh_table.inc"
};

inline auto pw_tanh(float x) -> float
{
    const auto a = std::fmin(std::fabs(x), 8.9999990f);
    const auto& c = pw_tanh_table[static_cast<int>(a + a)];
    const auto t = a - c[0];
    const auto t2 = t * t;
    const auto p0 = c[1] + c[2] * t;
    const auto p1 = c[3] + c[4] * t;
    const auto p2 = c[5] + c[6] * t;
    return std::copysign(p0 + t2 * (p1 + t2 * p2), x);
}

#include "rational_coeffs.inc"

/// Minimax rational tanh x ~ x P(s) / Q(s), s = x^2, clamped to [-9, 9]; P and
/// Q by Estrin, side by side, then one divide.
inline auto rat_tanh_7_6(float x) -> float
{
    const auto& p = rat_p_7_6;
    const auto& q = rat_q_7_6;
    const auto xc = std::fmin(std::fmax(x, -9.0f), 9.0f);
    const auto s = xc * xc;
    const auto s2 = s * s;
    const auto P = (p[0] + p[1] * s) + s2 * (p[2] + p[3] * s);
    const auto Q = (q[0] + q[1] * s) + s2 * (q[2] + q[3] * s);
    return xc * P / Q;
}

inline auto rat_tanh_9_6(float x) -> float
{
    const auto& p = rat_p_9_6;
    const auto& q = rat_q_9_6;
    const auto xc = std::fmin(std::fmax(x, -9.0f), 9.0f);
    const auto s = xc * xc;
    const auto s2 = s * s;
    const auto P = (p[0] + p[1] * s) + s2 * ((p[2] + p[3] * s) + p[4] * s2);
    const auto Q = (q[0] + q[1] * s) + s2 * (q[2] + q[3] * s);
    return xc * P / Q;
}

inline auto rat_tanh_9_8(float x) -> float
{
    const auto& p = rat_p_9_8;
    const auto& q = rat_q_9_8;
    const auto xc = std::fmin(std::fmax(x, -9.0f), 9.0f);
    const auto s = xc * xc;
    const auto s2 = s * s;
    const auto P = (p[0] + p[1] * s) + s2 * ((p[2] + p[3] * s) + p[4] * s2);
    const auto Q = (q[0] + q[1] * s) + s2 * ((q[2] + q[3] * s) + q[4] * s2);
    return xc * P / Q;
}

constexpr auto SAMPLES = std::size_t{65536};
constexpr auto RUNS = std::size_t{100};
auto escape(void* p) -> void { asm volatile("" : : "g"(p) : "memory"); }

enum class Loop { vector, independent, dependent };

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
            escape(out.data());
        }
        else if constexpr (L == Loop::independent) {
            for (auto i = 0uz; i < in.size(); ++i) {
                out[i] = f(in[i]);
                escape(&out[i]);
            }
        }
        else {
            auto y = 0.0f;
            for (auto i = 0uz; i < in.size(); ++i)
                y = f(in[i] + 0.0f * y);
            escape(&y);
        }
        t = std::chrono::duration<double, std::nano>(clock::now() - start).count();
    }
    std::ranges::sort(ns);
    return ns[RUNS / 2] / double(SAMPLES);
}

void method(const char* name, auto f, std::span<const float> in, std::span<float> out, double lib[3])
{
    const double t[3] = {bench<Loop::vector>(f, in, out), bench<Loop::independent>(f, in, out),
                         bench<Loop::dependent>(f, in, out)};
    std::printf("  %-22s", name);
    for (auto k = 0; k < 3; ++k) std::printf("  %6.2f (%4.1fx)", t[k], lib[k] / t[k]);
    std::printf("  %7.1f dB\n", max_abs(f));
}

auto thread_count = std::max(1u, std::thread::hardware_concurrency() / 2);  // --threads

auto max_abs(auto f) -> double
{
    const auto lo = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::min());
    const auto hi = std::bit_cast<std::uint32_t>(9.0f);
    const auto n_threads = thread_count;
    auto worst = std::vector<double>(n_threads);
    auto threads = std::vector<std::thread>{};
    for (auto t = 0u; t < n_threads; ++t) {
        threads.emplace_back([&, t] {
            const auto span = std::uint64_t{hi} - lo + 1;
            auto w = 0.0;
            for (auto b = lo + span * t / n_threads; b < lo + span * (t + 1) / n_threads; ++b) {
                const auto x = std::bit_cast<float>(static_cast<std::uint32_t>(b));
                w = std::max(w, std::abs(double(f(x)) - std::tanh(double(x))));
            }
            worst[t] = w;
        });
    }
    for (auto& th : threads) th.join();
    return 20 * std::log10(*std::ranges::max_element(worst));
}

auto main(int argc, char** argv) -> int
{
    if (argc == 3 && std::string_view{argv[1]} == "--threads") thread_count = std::max(1u, unsigned(std::atoi(argv[2])));
    auto rng = std::mt19937{1};
    auto dist = std::uniform_real_distribution<float>{-5, 5};
    auto in = std::vector<float>(SAMPLES), out = std::vector<float>(SAMPLES);
    for (auto& x : in) x = dist(rng);
    std::ranges::sort(in);

    const auto lib_f = [](float x) { return std::tanh(x); };
    double lib[3] = {bench<Loop::vector>(lib_f, in, out), bench<Loop::independent>(lib_f, in, out),
                     bench<Loop::dependent>(lib_f, in, out)};
    std::printf("ns per sample, x in [-5, 5] sorted; max abs error over every float in [0, 9]\n");
    std::printf("  %-22s  %-14s  %-14s  %-14s  %s\n", "", "vectorizable", "independent", "dependent", "error");
    std::printf("  %-22s  %6.2f          %6.2f          %6.2f\n", "std::tanh", lib[0], lib[1], lib[2]);
    method("poly + clamp 9", [](float x) { return adc::clamp_tanh_9(x); }, in, out, lib);
    method("sigmoid 9 (shipped)", [](float x) { return adc::sigmoid_clamp_tanh_9(x); }, in, out, lib);
    method("sigmoid 9, Estrin", [](float x) { return estrin_sigmoid_tanh_9(x); }, in, out, lib);
    method("Pade [7/6] + clamp", [](float x) { return adc::pade_tanh_7_6(x); }, in, out, lib);
    method("piecewise 18 x 5", [](float x) { return pw_tanh(x); }, in, out, lib);
    method("minimax [7/6], Estrin", [](float x) { return rat_tanh_7_6(x); }, in, out, lib);
    method("minimax [9/6], Estrin", [](float x) { return rat_tanh_9_6(x); }, in, out, lib);
    method("minimax [9/8], Estrin", [](float x) { return rat_tanh_9_8(x); }, in, out, lib);
}
