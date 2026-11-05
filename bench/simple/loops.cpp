// Three inner loops for the same bench: a buffer at a time, a sample at a
// time, and a sample at a time in feedback. Companion to main.cpp.
//
//   cmake --build bench/build && bench/build/adc_simple_loops

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <span>
#include <vector>

#include "sin_approx.hpp"  // adc::taylor_sin_13, adc::cheb_turns_sin_13

constexpr auto SAMPLES = std::size_t{65536};
constexpr auto RUNS = std::size_t{100};

// Chandler Carruth, CppCon 2015: https://www.youtube.com/watch?v=nXaxk27zwlk
auto escape(void* p) -> void { asm volatile("" : : "g"(p) : "memory"); }

// Average ns per sample for median run of loop.
auto bench(auto loop) -> double
{
    using clock = std::chrono::steady_clock;
    auto ns = std::vector<double>(RUNS);
    for (auto& t : ns) {
        const auto start = clock::now();
        loop();  // SAMPLES iterations
        const auto elapsed = clock::now() - start;
        t = std::chrono::duration<double, std::nano>(elapsed).count();
    }
    std::ranges::sort(ns);
    return ns[RUNS / 2] / double(SAMPLES);
}

// Buffer: the compiler may optimize within the buffer.
auto buffer(auto f, std::span<const float> in, std::span<float> out) -> void
{
    for (auto i = 0uz; i < in.size(); ++i)
        out[i] = f(in[i]);
    escape(out.data());
}

// Sample: the compiler may optimize within the sample.
auto sample(auto f, std::span<const float> in, std::span<float> out) -> void
{
    for (auto i = 0uz; i < in.size(); ++i) {
        out[i] = f(in[i]);
        escape(&out[i]);
    }
}

// Feedback: the previous output must be known.
auto feedback(auto f, std::span<const float> in) -> void
{
    auto y = 0.0f;
    for (auto i = 0uz; i < in.size(); ++i)
        y = f(in[i] + 0.0f * y);
    escape(&y);
}

auto row(const char* name, auto f, std::span<const float> in, std::span<float> out) -> void
{
    const auto b = bench([&] { buffer(f, in, out); });
    const auto s = bench([&] { sample(f, in, out); });
    const auto fb = bench([&] { feedback(f, in); });
    std::printf("%-12s %9.3f %9.3f %9.3f\n", name, b, s, fb);
}

// One table per input order: sorted like an oscillator's sweep (std::sin's
// best case: its branches go the same way for long stretches), or random.
auto table(const char* order, std::span<const float> turns, std::span<float> out) -> void
{
    constexpr auto pi = std::numbers::pi_v<float>;
    auto radians = std::vector<float>(turns.size());
    for (auto i = 0uz; i < turns.size(); ++i) radians[i] = 2 * pi * turns[i] - pi;

    std::printf("%s inputs, ns per sample\n", order);
    std::printf("              buffer    sample  feedback\n");
    row("std::sin", [](float x) { return std::sin(x); }, radians, out);
    row("Taylor", [](float x) { return adc::taylor_sin_13(x); }, radians, out);
    row("Turns", [](float x) { return adc::cheb_turns_sin_13(x); }, turns, out);
}

auto main() -> int
{
    auto rng = std::mt19937{1};
    auto dist = std::uniform_real_distribution<float>{0, 1};
    auto turns = std::vector<float>(SAMPLES);
    auto out = std::vector<float>(SAMPLES);
    for (auto& x : turns) x = dist(rng);

    auto sorted = turns;
    std::ranges::sort(sorted);
    table("Sorted", sorted, out);
    table("Random", turns, out);
}
