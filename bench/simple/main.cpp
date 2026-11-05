// The benchmark on the slides: how long does f take per sample, over a buffer?
// The full bench (bench/src) uses the same method for every scoreboard.
//
//   cmake --build bench/build && bench/build/adc_simple

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <span>
#include <vector>

#include "sin_approx.hpp"  // adc::taylor_sin_13

constexpr auto SAMPLES = std::size_t{65536};
constexpr auto RUNS = std::size_t{100};

// Chandler Carruth, CppCon 2015: https://www.youtube.com/watch?v=nXaxk27zwlk
auto escape(void* p) -> void { asm volatile("" : : "g"(p) : "memory"); }

// Average ns per sample for median run.
// in and out each hold SAMPLES values.
auto bench(auto f, std::span<const float> in, std::span<float> out) -> double
{
    using clock = std::chrono::steady_clock;
    auto ns = std::vector<double>(RUNS);
    for (auto& t : ns) {
        const auto start = clock::now();
        for (auto i = 0uz; i < in.size(); ++i) {
            out[i] = f(in[i]);
            escape(&out[i]);
        }
        const auto elapsed = clock::now() - start;
        t = std::chrono::duration<double, std::nano>(elapsed).count();
    }
    std::ranges::sort(ns);
    return ns[RUNS / 2] / double(SAMPLES);
}

auto main() -> int
{
    constexpr auto pi = std::numbers::pi_v<float>;
    auto rng = std::mt19937{1};
    auto dist = std::uniform_real_distribution<float>{-pi, pi};
    auto in = std::vector<float>(SAMPLES);
    auto out = std::vector<float>(SAMPLES);
    for (auto& x : in) x = dist(rng);
    std::ranges::sort(in);  // leave out for random

    const auto a = bench([](float x) { return std::sin(x); }, in, out);
    const auto b = bench([](float x) { return adc::taylor_sin_13(x); }, in, out);
    std::printf("std::sin %.2f ns\nTaylor   %.2f ns (%.1fx)\n", a, b, a / b);
}
