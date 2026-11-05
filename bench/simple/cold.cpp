// The cold-cache benchmark on the slides: one callback right after the caches
// are flushed, as if the host and other plug-ins just ran. Companion to
// main.cpp; the full bench (bench/src/timing.hpp) does the same.
//
//   cmake --build bench/build && bench/build/adc_simple_cold

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <numeric>
#include <span>
#include <vector>

#include "sin_approx.hpp"  // adc::taylor_sin_13, adc::lut_sin_linear_1024

constexpr auto SAMPLES = std::size_t{65536};
constexpr auto BLOCK = std::size_t{256};

auto escape(void* p) -> void { asm volatile("" : : "g"(p) : "memory"); }

// Read one word per cache line of a buffer far bigger than the caches.
auto evict() -> void
{
    static auto junk = std::vector<std::uint64_t>((64 << 20) / 8, 1);
    auto sum = std::uint64_t{};
    for (auto i = 0uz; i < junk.size(); i += 8) sum += junk[i];
    escape(&sum);
}

// Average ns per sample over BLOCK-sample callbacks, each right after an evict.
auto bench_cold(auto f, std::span<const float> in, std::span<float> out) -> double
{
    using clock = std::chrono::steady_clock;
    auto ns = std::vector<double>{};
    for (auto b = 0uz; b < in.size(); b += BLOCK) {
        evict();
        std::ranges::copy(in.subspan(b, BLOCK), &out[b]);  // the host's buffers are warm
        escape(out.data());
        const auto start = clock::now();
        for (auto i = b; i < b + BLOCK; ++i)
            out[i] = f(in[i]);
        escape(out.data());
        const auto elapsed = clock::now() - start;
        ns.push_back(std::chrono::duration<double, std::nano>(elapsed).count());
    }
    std::ranges::sort(ns);
    ns.resize(ns.size() * 9 / 10);  // drop the slowest 10%: interrupts
    return std::reduce(ns.begin(), ns.end()) / double(ns.size() * BLOCK);
}

auto main() -> int
{
    constexpr auto pi = std::numbers::pi;
    auto in = std::vector<float>(SAMPLES);
    auto out = std::vector<float>(SAMPLES);
    for (auto i = 0uz; i < SAMPLES; ++i) {  // a 997 Hz oscillator at 48 kHz
        const auto phase = std::fmod(double(i) * 997.0 / 48000.0, 1.0);
        in[i] = float(2.0 * pi * phase - pi);
    }

    const auto a = bench_cold([](float x) { return std::sin(x); }, in, out);
    const auto b = bench_cold([](float x) { return adc::taylor_sin_13(x); }, in, out);
    const auto c = bench_cold([](float x) { return adc::lut_sin_linear_1024(x); }, in, out);
    std::printf("cold, ns per sample\nstd::sin %.2f\nTaylor   %.2f\nTable    %.2f\n", a, b, c);
}
