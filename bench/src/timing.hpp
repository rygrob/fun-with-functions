#pragma once
// Timing: every generated method against std::sin.
//
// Three measurements per method, ns per sample:
//   indep     out[i] = f(in[i]) over N sorted inputs (like an oscillator),
//             escape after every sample: one at a time, the way per-sample
//             DSP is written, though the core still overlaps independent
//             calls. The median of RUNS passes. The method on the slides
//             (bench/simple/main.cpp), and the column they quote
//   vector    the same loop with one escape per pass: the compiler may
//             vectorize -- polynomials do, table lookups (gathers) and
//             std::sin don't (bench/simple/loops.cpp has both, and a third,
//             dependent loop)
//   cold      one BLOCK-sample oscillator callback right after evicting the
//             caches, as if the host and other plug-ins just ran. The block's
//             phases (an OSC_HZ oscillator) and the output buffer are
//             re-touched after the eviction, so what starts cold is only what
//             belongs to the method: its table, constants and code. The phase
//             accumulator itself isn't timed -- a float accumulator with a wrap
//             is a serial chain that would swamp the cheaper methods, while an
//             integer one is nearly free. Trimmed mean, since a block is only a
//             handful of clock ticks.
//
// Radians methods get x in [-pi, pi]; turns methods the same phases in
// [0, 1]; phase methods the same phases as a 32-bit accumulator value.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <audio_bench/audio_bench.hpp>

#include "sin_approx.hpp"

namespace adc::timing {

inline constexpr auto N = std::size_t{65536};
inline constexpr auto RUNS = std::size_t{100};
inline constexpr auto BLOCK = std::size_t{256};
inline constexpr auto COLD_REPS = 200;
inline constexpr auto OSC_HZ = 997.0;
inline constexpr auto SAMPLE_RATE = 48000.0;
inline constexpr auto EVICT_BYTES = std::size_t{64} << 20;  // well past L2 + SLC on current desktop chips

struct Inputs {
    std::vector<double> turns;           // uniform in [0, 1)
    std::vector<double> radians;         // 2 pi turns - pi
    std::vector<std::uint32_t> phase;    // turns * 2^32
};

inline auto make_inputs() -> Inputs
{
    auto rng = std::mt19937_64{12345};
    auto dist = std::uniform_real_distribution<double>{0.0, 1.0};
    auto in = Inputs{};
    for (auto i = std::size_t{}; i < N; ++i) {
        const auto u = dist(rng);
        in.turns.push_back(u);
        in.radians.push_back(2.0 * std::numbers::pi * u - std::numbers::pi);
        in.phase.push_back(static_cast<std::uint32_t>(u * 4294967296.0));
    }
    return in;
}

/// Read one word per cache line of a buffer much larger than the caches.
inline auto evict() -> void
{
    static auto buf = std::vector<std::uint64_t>(EVICT_BYTES / sizeof(std::uint64_t), 1);
    auto sum = std::uint64_t{};
    for (auto i = std::size_t{}; i < buf.size(); i += 64 / sizeof(std::uint64_t)) {
        sum += buf[i];
    }
    audio_bench::do_not_optimize(sum);
}

struct Row {
    std::string name;
    double indep_ns;
    double vector_ns;
    double cold_ns;
};

/// Chandler Carruth's escape (CppCon 2015), as on the slides: the compiler
/// must assume the asm reads and writes everything p reaches.
inline auto escape(void* p) -> void
{
#if defined(_MSC_VER)
    audio_bench::do_not_optimize(p);
#else
    asm volatile("" : : "g"(p) : "memory");
#endif
}

/// Median over RUNS passes of out[i] = f(in[i]), in ns per sample: with an
/// escape after every sample (Independent), or once per pass. Not inlined,
/// so out stays a buffer from the caller, as in a plug-in: inlined next to
/// its allocation, the compiler can prove out doesn't overlap a lookup
/// table, and vectorized tables look 40% faster than in a plug-in.
template<bool Independent, typename T, typename X, typename F>
[[gnu::noinline]] auto buffer_ns(F f, const std::vector<X>& in, std::vector<T>& out) -> double
{
    using clock = std::chrono::steady_clock;
    const auto src = std::span{in};  // by value, as on the slides: the escape's
    const auto dst = std::span{out};  // clobber can't force a reload of in / out
    auto ns = std::vector<double>(RUNS);
    for (auto& t : ns) {
        const auto start = clock::now();
        for (auto i = std::size_t{}; i < src.size(); ++i) {
            dst[i] = f(src[i]);
            if constexpr (Independent) escape(&dst[i]);
        }
        escape(dst.data());
        const auto elapsed = clock::now() - start;
        t = std::chrono::duration<double, std::nano>(elapsed).count();
    }
    std::ranges::sort(ns);
    return ns[RUNS / 2] / static_cast<double>(in.size());
}

/// Phase generators for the cold oscillator block, one per domain.
template<typename T>
struct RadiansOsc {
    T x = T(-std::numbers::pi);
    T inc = T(2.0 * std::numbers::pi * OSC_HZ / SAMPLE_RATE);
    auto operator()() -> T
    {
        const auto y = x;
        x += inc;
        if (x >= T(std::numbers::pi)) x -= T(2.0 * std::numbers::pi);
        return y;
    }
};

template<typename T>
struct TurnsOsc {
    T x = T(0);
    T inc = T(OSC_HZ / SAMPLE_RATE);
    auto operator()() -> T
    {
        const auto y = x;
        x += inc;
        if (x >= T(1)) x -= T(1);
        return y;
    }
};

struct PhaseOsc {
    std::uint32_t p = 0;
    std::uint32_t inc = static_cast<std::uint32_t>(OSC_HZ / SAMPLE_RATE * 4294967296.0);
    auto operator()() -> std::uint32_t
    {
        const auto y = p;
        p += inc;  // wraps for free
        return y;
    }
};

/// Mean of the fastest 90%: drops interrupts and other outliers.
inline auto trimmed_mean(std::vector<double> v) -> double
{
    std::ranges::sort(v);
    const auto keep = v.size() * 9 / 10;
    auto sum = 0.0;
    for (auto i = std::size_t{}; i < keep; ++i) sum += v[i];
    return sum / static_cast<double>(keep);
}

template<typename T, typename X, typename F, typename Osc>
inline auto time_method(const std::string& name, const std::vector<X>& xs, F fn, Osc osc) -> Row
{
    auto out = std::vector<T>(xs.size());
    auto sorted = xs;
    std::ranges::sort(sorted);
    buffer_ns<true>(fn, sorted, out);  // warm up: the first method timed can catch the clock ramping
    const auto indep_ns = buffer_ns<true>(fn, sorted, out);
    const auto vector_ns = buffer_ns<false>(fn, sorted, out);

    // Timer would include the eviction, so time the cold block by hand.
    auto phases = std::vector<X>(BLOCK);
    auto cold = std::vector<double>{};
    for (auto r = 0; r < COLD_REPS; ++r) {
        for (auto& p : phases) p = osc();  // next block of the oscillator
        evict();
        std::fill(out.begin(), out.begin() + BLOCK, T{});  // the host's buffer is warm
        auto warm = X{};
        for (auto p : phases) warm += p;                  // and so are this block's phases
        audio_bench::do_not_optimize(warm);
        const auto start = std::chrono::steady_clock::now();
        for (auto i = std::size_t{}; i < BLOCK; ++i) {
            out[i] = fn(phases[i]);
        }
        audio_bench::do_not_optimize(out[0]);
        const auto end = std::chrono::steady_clock::now();
        cold.push_back(static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
    }

    return {name, indep_ns, vector_ns, trimmed_mean(cold) / static_cast<double>(BLOCK)};
}

/// The timing table; speedups are against rows[0]. cold describes the block.
inline auto print_rows(const std::vector<Row>& rows, const std::string& what, const std::string& cold) -> void
{
    const auto& base = rows[0];
    std::cout << std::format(
        "\nTiming, {}\n"
        "  indep / vector: {} sorted inputs, escape every sample or once per pass, median of {} passes.\n"
        "  cold: {} after evicting {} MB, trimmed mean of {}\n",
        what, N, RUNS, cold, EVICT_BYTES >> 20, COLD_REPS);
    std::cout << std::format("  {:<34} {:>9} {:>7}  {:>9} {:>7}  {:>9} {:>7}\n",
                             "method", "indep", "", "vector", "", "cold", "");
    for (const auto& r : rows) {
        std::cout << std::format("  {:<34} {:>9.2f} {:>6.1f}x  {:>9.2f} {:>6.1f}x  {:>9.2f} {:>6.1f}x\n", r.name,
                                 r.indep_ns, base.indep_ns / r.indep_ns,
                                 r.vector_ns, base.vector_ns / r.vector_ns,
                                 r.cold_ns, base.cold_ns / r.cold_ns);
    }
}

template<typename T>
inline auto run(const char* type_name) -> void
{
    const auto in = make_inputs();
    auto rad = std::vector<T>{};
    auto trn = std::vector<T>{};
    for (auto i = std::size_t{}; i < N; ++i) {
        rad.push_back(static_cast<T>(in.radians[i]));
        trn.push_back(static_cast<T>(in.turns[i]));
    }

    auto rows = std::vector<Row>{};
    rows.push_back(time_method<T>("std::sin(x)", rad, [](T x) { return std::sin(x); }, RadiansOsc<T>{}));
    rows.push_back(time_method<T>("std::sin(2pi x)  [turns]", trn,
        [](T x) { return std::sin(T(2.0 * std::numbers::pi) * x); }, TurnsOsc<T>{}));
    for_each_method<T>([&](auto, const char* name, auto domain, auto fn) {
        if constexpr (domain == Domain::phase) {
            rows.push_back(time_method<T>(std::string{name} + "  [phase]", in.phase, fn, PhaseOsc{}));
        }
        else if constexpr (domain == Domain::turns) {
            rows.push_back(time_method<T>(std::string{name} + "  [turns]", trn, fn, TurnsOsc<T>{}));
        }
        else {
            rows.push_back(time_method<T>(name, rad, fn, RadiansOsc<T>{}));
        }
    });

    print_rows(rows, std::format("{}: ns per sample, speedup vs std::sin(x)", type_name),
               std::format("{}-sample {:g} Hz oscillator block", BLOCK, OSC_HZ));
}

} // namespace adc::timing
