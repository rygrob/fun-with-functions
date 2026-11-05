#pragma once
// Tables vs. math as the number of tables grows: when do lookups fall out of
// cache, and what does it cost?
//
// The workload is K instances -- synth voices, or plugin instances -- each
// running a mixed chain per sample and summing into a bus:
//
//   oscillator   s = sin(2 pi phase)          32-bit phase accumulator
//   waveshaper   y = tanh(drive * s)
//   envelope     g = 10^(dB / 20)             dB ramping linearly
//   bus         += y * g
//
// "table": every instance owns its own three float LUTs of N + 1 entries
// (linear interpolation, guard point), so the working set is K * 3 * (N + 1)
// floats. "math": the same chain with the generated polynomials
// (cheb_turns_sin_13, sigmoid_tanh_9, db_to_gain_3) -- no memory beyond the
// bus -- vectorized across the block as the compiler likes, and "scalar":
// the same with vectorization disabled (clang), since table lookups can't
// vectorize (no gather on NEON) and per-sample feedback code can't either. The host loop processes a B-sample block for each instance in turn,
// like a plug-in chain or a voice loop.
//
// Swept: K, N, the oscillator's access pattern, and a competing working set
// touched between rounds (other plug-ins running between our callbacks):
//
//   pattern osc     each voice's phase advances by its own (detuned) increment:
//                   sequential reads the prefetchers can follow
//   pattern random  the phase is hashed per sample (noise / FM-like jumps):
//                   every read lands on an unpredictable line
//   between         0, 2 or 16 MB read and written between rounds
//
// Reported: ns per sample per instance (all three functions), as CSV plus a
// summary. Timings are machine specific; rerun on power before quoting.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include <audio_bench/audio_bench.hpp>

#include "dbgain_approx.hpp"
#include "sin_approx.hpp"
#include "tanh_approx.hpp"

namespace adc::tables {

inline constexpr auto BLOCK = 128;
inline constexpr auto TANH_RANGE = 8.0f;          // tanh table covers [-8, 8], clamped
inline constexpr auto DB_LO = -120.0f;            // dB table covers [-120, 24]
inline constexpr auto DB_HI = 24.0f;
inline constexpr auto MAX_FOOTPRINT = std::size_t{256} << 20;

/// One instance's three tables, laid out back to back in one allocation.
struct Tables {
    std::vector<float> data;
    std::size_t n = 0;  // entries per table, excluding the guard point
    auto sin_t() const -> const float* { return data.data(); }
    auto tanh_t() const -> const float* { return data.data() + (n + 1); }
    auto db_t() const -> const float* { return data.data() + 2 * (n + 1); }
};

inline auto make_tables(std::size_t n) -> Tables
{
    auto t = Tables{std::vector<float>(3 * (n + 1)), n};
    for (auto i = std::size_t{}; i <= n; ++i) {
        const auto u = static_cast<double>(i) / static_cast<double>(n);
        t.data[i] = static_cast<float>(std::sin(2.0 * std::numbers::pi * u));
        t.data[n + 1 + i] = static_cast<float>(std::tanh(-TANH_RANGE + 2.0 * TANH_RANGE * u));
        t.data[2 * (n + 1) + i] = static_cast<float>(std::pow(10.0, (DB_LO + (DB_HI - DB_LO) * u) / 20.0));
    }
    return t;
}

/// Per-instance state: a detuned oscillator, a drive and an envelope.
struct Voice {
    std::uint32_t phase;
    std::uint32_t inc;
    float drive;
    float db;
    float ddb;  // dB per sample
};

inline auto make_voices(int k) -> std::vector<Voice>
{
    auto v = std::vector<Voice>{};
    for (auto i = 0; i < k; ++i) {
        const auto hz = 55.0 * std::exp2(static_cast<double>(i % 48) / 12.0) * (1.0 + 0.001 * i);
        v.push_back({static_cast<std::uint32_t>(i) * 0x9E3779B9u,
                     static_cast<std::uint32_t>(hz / 48000.0 * 4294967296.0),
                     1.0f + 0.25f * static_cast<float>(i % 8), -60.0f + static_cast<float>(i % 40), 0.0005f});
    }
    return v;
}

/// Cheap integer hash for the random pattern (same cost in both versions).
inline auto hash(std::uint32_t x) -> std::uint32_t
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return x;
}

/// Linear lookup at position pos in [0, n), table has a guard point.
inline auto lerp(const float* t, float pos) -> float
{
    const auto i = static_cast<std::uint32_t>(pos);
    const auto f = pos - static_cast<float>(i);
    return t[i] + f * (t[i + 1] - t[i]);
}

template<bool Random>
inline auto phase_at(const Voice& v, int i, std::uint32_t salt) -> std::uint32_t
{
    const auto p = v.phase + static_cast<std::uint32_t>(i) * v.inc;
    if constexpr (Random) return hash(p ^ salt);
    else return p;
}

/// One block of one instance, tables.
template<bool Random>
inline auto block_table(const Voice& v, const Tables& t, int shift, std::uint32_t salt, float* bus) -> void
{
    const auto n = static_cast<float>(t.n);
    const auto frac_scale = 1.0f / static_cast<float>(1u << shift);
    const auto tanh_scale = n / (2.0f * TANH_RANGE);
    const auto db_scale = n / (DB_HI - DB_LO);
    const auto top = std::nextafter(n, 0.0f);  // keeps i + 1 on the guard point at most
    for (auto i = 0; i < BLOCK; ++i) {
        const auto p = phase_at<Random>(v, i, salt);
        const auto idx = p >> shift;
        const auto s = lerp(t.sin_t(), static_cast<float>(idx) + static_cast<float>(p & ((1u << shift) - 1)) * frac_scale);
        const auto u = std::clamp((v.drive * s + TANH_RANGE) * tanh_scale, 0.0f, top);
        const auto y = lerp(t.tanh_t(), u);
        const auto d = std::clamp((v.db + static_cast<float>(i) * v.ddb - DB_LO) * db_scale, 0.0f, top);
        bus[i] += y * lerp(t.db_t(), d);
    }
}

/// One block of one instance, polynomials. Vector = false stops the
/// compiler vectorizing across the block: the cost per sample of code that
/// can't (a feedback loop, per-sample branching), and a like-for-like
/// comparison with the tables, whose lookups don't vectorize either.
template<bool Random, bool Vector = true>
inline auto block_math(const Voice& v, std::uint32_t salt, float* bus) -> void
{
#if defined(__clang__)
    if constexpr (!Vector) {
#pragma clang loop vectorize(disable) interleave(disable)
        for (auto i = 0; i < BLOCK; ++i) {
            const auto p = phase_at<Random>(v, i, salt);
            const auto s = cheb_turns_sin_13<float>(static_cast<float>(p) * 0x1p-32f);
            const auto y = sigmoid_tanh_9<float>(v.drive * s);
            bus[i] += y * db_to_gain_3<float>(v.db + static_cast<float>(i) * v.ddb);
        }
        return;
    }
#endif
    for (auto i = 0; i < BLOCK; ++i) {
        const auto p = phase_at<Random>(v, i, salt);
        const auto s = cheb_turns_sin_13<float>(static_cast<float>(p) * 0x1p-32f);
        const auto y = sigmoid_tanh_9<float>(v.drive * s);
        bus[i] += y * db_to_gain_3<float>(v.db + static_cast<float>(i) * v.ddb);
    }
}

/// Read and write `bytes` of someone else's memory: other plug-ins running.
inline auto compete(std::size_t bytes) -> void
{
    if (!bytes) return;
    static auto buf = std::vector<std::uint64_t>((std::size_t{64} << 20) / sizeof(std::uint64_t), 1);
    const auto words = std::min(bytes / sizeof(std::uint64_t), buf.size());
    auto sum = std::uint64_t{};
    for (auto i = std::size_t{}; i < words; i += 8) {  // one word per 64 bytes, read-modify-write
        buf[i] += 1;
        sum += buf[i];
    }
    audio_bench::do_not_optimize(sum);
}

struct Result {
    std::string kind;
    std::string pattern;
    int k;
    std::size_t n;           // 0 for math
    std::size_t between_mb;
    double footprint_kb;
    double ns_per_sample;
};

/// Time `rounds` rounds of K blocks, the competing set touched (untimed)
/// before each. Best of three: we want the effect, not scheduler noise.
template<typename F>
inline auto time_rounds(int k, int rounds, std::size_t between, F&& round) -> double
{
    auto best = 1e300;
    for (auto rep = 0; rep < 3; ++rep) {
        auto total = 0.0;
        for (auto r = 0; r < rounds; ++r) {
            compete(between);
            const auto t0 = std::chrono::steady_clock::now();
            round(r);
            const auto t1 = std::chrono::steady_clock::now();
            total += std::chrono::duration<double, std::nano>(t1 - t0).count();
        }
        best = std::min(best, total);
    }
    return best / (static_cast<double>(k) * BLOCK * rounds);
}

inline auto rounds_for(int k, std::size_t between) -> int
{
    const auto samples = between ? (1 << 17) : (1 << 20);
    return std::clamp(samples / (k * BLOCK), between ? 8 : 16, between ? 64 : 4096);
}

template<bool Random>
inline auto run_one_table(int k, std::size_t n, std::size_t between) -> double
{
    auto tables = std::vector<Tables>{};
    for (auto i = 0; i < k; ++i) tables.push_back(make_tables(n));
    auto voices = make_voices(k);
    auto bus = std::vector<float>(BLOCK);
    const auto shift = 32 - static_cast<int>(std::countr_zero(n));
    const auto rounds = rounds_for(k, between);
    // Warm everything once, as a running host would have.
    for (auto j = std::size_t{}; j < voices.size(); ++j) block_table<Random>(voices[j], tables[j], shift, 0, bus.data());
    const auto ns = time_rounds(k, rounds, between, [&](int r) {
        for (auto j = std::size_t{}; j < voices.size(); ++j) {
            block_table<Random>(voices[j], tables[j], shift, static_cast<std::uint32_t>(r), bus.data());
        }
        audio_bench::do_not_optimize(bus[0]);
    });
    return ns;
}

template<bool Random, bool Vector>
inline auto run_one_math(int k, std::size_t between) -> double
{
    auto voices = make_voices(k);
    auto bus = std::vector<float>(BLOCK);
    const auto rounds = rounds_for(k, between);
    return time_rounds(k, rounds, between, [&](int r) {
        for (auto j = std::size_t{}; j < voices.size(); ++j) {
            block_math<Random, Vector>(voices[j], static_cast<std::uint32_t>(r), bus.data());
        }
        audio_bench::do_not_optimize(bus[0]);
    });
}

inline auto run(const std::string& csv_path, bool quick) -> void
{
    const auto ks = quick ? std::vector<int>{1, 8, 64, 512} : std::vector<int>{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    const auto ns = quick ? std::vector<std::size_t>{1024, 16384} : std::vector<std::size_t>{256, 1024, 4096, 16384, 65536};
    const auto betweens = std::vector<std::size_t>{0, 2, 16};
    auto results = std::vector<Result>{};

    const auto add = [&](std::string kind, std::string pattern, int k, std::size_t n, std::size_t mb, double ns_) {
        const auto kb = static_cast<double>(k) * 3.0 * static_cast<double>(n + (n ? 1 : 0)) * 4.0 / 1024.0;
        results.push_back({std::move(kind), std::move(pattern), k, n, mb, kb, ns_});
        std::cerr << std::format("\r  {:<6} {:<7} K={:<5} N={:<6} between={:>2} MB: {:7.2f} ns   ", results.back().kind,
                                 results.back().pattern, k, n, mb, ns_);
    };

    for (const auto mb : betweens) {
        for (const auto k : ks) {
            add("math", "osc", k, 0, mb, run_one_math<false, true>(k, mb << 20));
            add("math", "random", k, 0, mb, run_one_math<true, true>(k, mb << 20));
            add("scalar", "osc", k, 0, mb, run_one_math<false, false>(k, mb << 20));
            add("scalar", "random", k, 0, mb, run_one_math<true, false>(k, mb << 20));
            for (const auto n : ns) {
                if (static_cast<std::size_t>(k) * 3 * (n + 1) * 4 > MAX_FOOTPRINT) continue;
                add("table", "osc", k, n, mb, run_one_table<false>(k, n, mb << 20));
                add("table", "random", k, n, mb, run_one_table<true>(k, n, mb << 20));
            }
        }
    }
    std::cerr << "\n";

    if (const auto dir = std::filesystem::path(csv_path).parent_path(); !dir.empty()) {
        std::filesystem::create_directories(dir);
    }
    if (auto* f = std::fopen(csv_path.c_str(), "w")) {
        std::fprintf(f, "kind,pattern,k,n,between_mb,footprint_kb,ns_per_sample\n");
        for (const auto& r : results) {
            std::fprintf(f, "%s,%s,%d,%zu,%zu,%.3f,%.4f\n", r.kind.c_str(), r.pattern.c_str(), r.k, r.n, r.between_mb,
                         r.footprint_kb, r.ns_per_sample);
        }
        std::fclose(f);
        std::cout << std::format("Wrote {} rows to {}\n", results.size(), csv_path);
    }

    // Summary: ns per sample at a few points, tables vs. math.
    const auto find = [&](const std::string& kind, const std::string& pat, int k, std::size_t n, std::size_t mb) {
        for (const auto& r : results) {
            if (r.kind == kind && r.pattern == pat && r.k == k && r.n == n && r.between_mb == mb) return r.ns_per_sample;
        }
        return std::nan("");
    };
    std::cout << std::format("\nsin -> tanh -> dB-gain chain, ns per sample per instance (block {}):\n", BLOCK);
    for (const auto mb : betweens) {
        std::cout << std::format("\n  between rounds: {} MB of other work\n", mb);
        std::cout << std::format("  {:>6}  {:>8} {:>8}  {:>8} {:>8}", "K", "math osc", "math rnd", "scal osc", "scal rnd");
        for (const auto n : ns) std::cout << std::format("  {:>9} {:>9}", std::format("N={} osc", n), "rnd");
        std::cout << "\n";
        for (const auto k : ks) {
            std::cout << std::format("  {:>6}  {:>8.2f} {:>8.2f}  {:>8.2f} {:>8.2f}", k, find("math", "osc", k, 0, mb),
                                     find("math", "random", k, 0, mb), find("scalar", "osc", k, 0, mb),
                                     find("scalar", "random", k, 0, mb));
            for (const auto n : ns) {
                std::cout << std::format("  {:>9.2f} {:>9.2f}", find("table", "osc", k, n, mb), find("table", "random", k, n, mb));
            }
            std::cout << "\n";
        }
    }
}

} // namespace adc::tables
