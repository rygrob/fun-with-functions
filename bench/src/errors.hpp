#pragma once
// Error metrics, matching examples/common/metrics.py. Kept here rather than
// using audio_bench's helpers so the C++ and Python numbers mean the same
// thing.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <thread>

namespace adc::err {

/// Threads for the exhaustive checks: half the cores, unless --threads says otherwise.
inline auto thread_count = std::max(1u, std::thread::hardware_concurrency() / 2);

/// Gap between adjacent f32 values in the binade containing t (t is the
/// exact value, here a double). Subnormals bottom out at 2^-149.
inline auto ulp_f32(double t) -> double
{
    int e = 0;
    std::frexp(std::abs(t), &e);
    return std::ldexp(1.0, std::max(e - 24, -149));
}

/// Same for f64: 53-bit significand, subnormals at 2^-1074.
inline auto ulp_f64(double t) -> double
{
    int e = 0;
    std::frexp(std::abs(t), &e);
    return std::ldexp(1.0, std::max(e - 53, -1074));
}

/// sin(2 pi x) in f64, exact at the zeros. Reduce by quarter turns first --
/// n = round(4x) and r = x - n/4 are exact for f32 inputs and for doubles in
/// [0, 1] -- then sin or cos of the small remainder.
inline auto sin_turns(double x) -> double
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

/// tan(pi x) in f64, exact at the pole: for |x| > 1/4 use 1 / tan(pi (1/2 - x)),
/// since 1/2 - |x| is exact for f32 inputs (matches metrics.tan_pi).
inline auto tan_pi(double x) -> double
{
    const auto a = std::abs(x);
    if (a <= 0.25) return std::tan(std::numbers::pi * x);
    return std::copysign(1.0, x) / std::tan(std::numbers::pi * (0.5 - a));
}

/// Cutoff error in cents of a bilinear-transform filter given g ~ tan(pi x):
/// its cutoff lands at atan(g) / pi. A g that isn't positive and finite puts
/// the cutoff nowhere: infinity (matches metrics.cutoff_cents).
inline auto cutoff_cents(double g, double x) -> double
{
    if (!(g > 0.0) || !std::isfinite(g)) return std::numeric_limits<double>::infinity();
    return std::abs(1200.0 * std::log2(std::atan(g) / (std::numbers::pi * x)));
}

/// Running ULP statistics for one method.
struct UlpStats {
    double max{};
    double arg_max{};       // input with the largest error
    double max_abs{};       // largest absolute error, for dB comparisons
    double sum{};
    std::uint64_t count{};
    std::uint64_t exact{};  // result == truth rounded to the eval type
    std::uint64_t under_1{};  // error < 1 ULP (faithfully rounded)

    void add(double ulps, double abs_err, double x, bool is_exact)
    {
        max_abs = std::max(max_abs, abs_err);
        if (ulps > max || count == 0) {
            max = ulps;
            arg_max = x;
        }
        sum += ulps;
        ++count;
        exact += is_exact ? 1 : 0;
        under_1 += ulps < 1.0 ? 1 : 0;
    }

    void merge(const UlpStats& o)
    {
        if (o.count && (o.max > max || count == 0)) {
            max = o.max;
            arg_max = o.arg_max;
        }
        max_abs = std::max(max_abs, o.max_abs);
        sum += o.sum;
        count += o.count;
        exact += o.exact;
        under_1 += o.under_1;
    }

    /// Max absolute error as a level relative to full scale (sin peaks at 1).
    auto max_abs_db() const -> double { return 20.0 * std::log10(std::max(max_abs, 1e-300)); }

    auto mean() const -> double { return count ? sum / static_cast<double>(count) : 0.0; }
    auto pct(std::uint64_t n) const -> double
    {
        return count ? 100.0 * static_cast<double>(n) / static_cast<double>(count) : 0.0;
    }
};

} // namespace adc::err
