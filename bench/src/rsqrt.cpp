// tanh's square root in SIMD: sigmoid_tanh_9 (y = s / sqrt(1 + s^2)) four
// wide, with five ways to take the square root. Exhaustive accuracy over every
// float in [-20, 20] (max abs error against double tanh, max |y|) and buffer
// timing. Standalone (no audio_bench), so it also builds for x86-64:
//
//   adc_rsqrt                         NEON on arm64, SSE on x86-64
//
//   clang++ -std=c++20 -O2 -arch x86_64 -Igenerated src/rsqrt.cpp
//                                     SSE on Apple silicon, run under Rosetta:
//                                     accuracy is Rosetta's rsqrtps, which
//                                     need not match Intel's bit for bit, and
//                                     the timings mean nothing
//
// The SIMD paths repeat sigmoid_tanh_9's coefficients; main() checks the
// scalar path against the generated function first.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tanh_approx.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <immintrin.h>
#else
#error "adc_rsqrt needs NEON or SSE"
#endif

namespace {

constexpr float c4 = 3.638146764843207e-06f;
constexpr float c3 = 0.00019032944649238027f;
constexpr float c2 = 0.008358021824318818f;
constexpr float c1 = 0.1666433506348483f;
constexpr float c0 = 1.0000045841180956f;

// ---------------------------------------------------------------------------
// Plain C++: the compiler vectorizes std::sqrt (fsqrt.4s / sqrtps) or not
// ---------------------------------------------------------------------------

__attribute__((noinline)) void tanh_std(const float* x, float* y, int n)
{
    for (auto i = 0; i < n; ++i) {
        const auto x2 = x[i] * x[i];
        auto r = c4;
        r = r * x2 + c3;
        r = r * x2 + c2;
        r = r * x2 + c1;
        r = r * x2 + c0;
        const auto s = x[i] * r;
        y[i] = s * (1.0f / std::sqrt(s * s + 1.0f));
    }
}

// ---------------------------------------------------------------------------
// Intrinsics
// ---------------------------------------------------------------------------

#if defined(__aarch64__)

using V = float32x4_t;

inline auto load(const float* p) -> V { return vld1q_f32(p); }
inline auto store(float* p, V v) -> void { vst1q_f32(p, v); }
inline auto splat(float v) -> V { return vdupq_n_f32(v); }
inline auto mul(V a, V b) -> V { return vmulq_f32(a, b); }
inline auto madd(V a, V b, V c) -> V { return vaddq_f32(vmulq_f32(a, b), c); }

inline auto sqrt_div(V s, V a) -> V { return vdivq_f32(s, vsqrtq_f32(a)); }

// rsqrt: the hardware estimate (about 8 bits), then Newton steps; vrsqrtsq
// computes (3 - a e^2) / 2, so each step is one multiply more.
template<int Steps>
inline auto rsqrt(V a) -> V
{
    auto e = vrsqrteq_f32(a);
    for (auto k = 0; k < Steps; ++k)
        e = vmulq_f32(e, vrsqrtsq_f32(vmulq_f32(a, e), e));
    return e;
}

#else

using V = __m128;

inline auto load(const float* p) -> V { return _mm_loadu_ps(p); }
inline auto store(float* p, V v) -> void { _mm_storeu_ps(p, v); }
inline auto splat(float v) -> V { return _mm_set1_ps(v); }
inline auto mul(V a, V b) -> V { return _mm_mul_ps(a, b); }
inline auto madd(V a, V b, V c) -> V { return _mm_add_ps(_mm_mul_ps(a, b), c); }

inline auto sqrt_div(V s, V a) -> V { return _mm_div_ps(s, _mm_sqrt_ps(a)); }

// rsqrt: the hardware estimate (about 12 bits), then Newton steps,
// e (3 - a e^2) / 2.
template<int Steps>
inline auto rsqrt(V a) -> V
{
    auto e = _mm_rsqrt_ps(a);
    for (auto k = 0; k < Steps; ++k) {
        const auto t = _mm_mul_ps(_mm_mul_ps(a, e), e);
        e = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(0.5f), e), _mm_sub_ps(_mm_set1_ps(3.0f), t));
    }
    return e;
}

#endif

// s = x R(x^2) and a = 1 + s^2, four at a time.
inline auto sinh_and_a(V x, V& a) -> V
{
    const auto x2 = mul(x, x);
    auto r = splat(c4);
    r = madd(r, x2, splat(c3));
    r = madd(r, x2, splat(c2));
    r = madd(r, x2, splat(c1));
    r = madd(r, x2, splat(c0));
    const auto s = mul(x, r);
    a = madd(s, s, splat(1.0f));
    return s;
}

__attribute__((noinline)) void tanh_sqrt_div(const float* x, float* y, int n)
{
    for (auto i = 0; i < n; i += 4) {
        auto a = splat(0.0f);
        const auto s = sinh_and_a(load(x + i), a);
        store(y + i, sqrt_div(s, a));
    }
}

template<int Steps>
__attribute__((noinline)) void tanh_rsqrt(const float* x, float* y, int n)
{
    for (auto i = 0; i < n; i += 4) {
        auto a = splat(0.0f);
        const auto s = sinh_and_a(load(x + i), a);
        store(y + i, mul(s, rsqrt<Steps>(a)));
    }
}

// ---------------------------------------------------------------------------
// Measure
// ---------------------------------------------------------------------------

using Fn = void (*)(const float*, float*, int);

struct Method
{
    const char* name;
    Fn f;
};

struct Error
{
    double max_abs = 0;
    double max_y = 0;
};

// Every float in [0, 20] (tanh is odd), a chunk at a time.
auto exhaustive(Fn f) -> Error
{
    const auto lim = 20.0f;
    auto top = std::uint32_t{0};
    std::memcpy(&top, &lim, 4);
    const auto chunk = 1 << 16;
    std::vector<float> xs(chunk), ys(chunk);
    auto err = Error{};
    for (auto u0 = std::uint64_t{0}; u0 <= top; u0 += chunk) {
        auto n = 0;
        for (auto u = u0; u < u0 + chunk && u <= top; ++u) {
            const auto b = static_cast<std::uint32_t>(u);
            std::memcpy(&xs[static_cast<std::size_t>(n++)], &b, 4);
        }
        while (n % 4 != 0)
            xs[static_cast<std::size_t>(n++)] = 0.0f;
        f(xs.data(), ys.data(), n);
        for (auto i = 0; i < n; ++i) {
            const auto k = static_cast<std::size_t>(i);
            err.max_abs = std::max(err.max_abs, std::fabs(double(ys[k]) - std::tanh(double(xs[k]))));
            err.max_y = std::max(err.max_y, double(std::fabs(ys[k])));
        }
    }
    return err;
}

// 4096 inputs in [-5, 5], median of 400 runs, ns per sample.
auto buffer_ns(Fn f) -> double
{
    constexpr auto n = 4096;
    std::vector<float> x(n), y(n);
    auto seed = std::uint32_t{1};
    for (auto& v : x) {
        seed = seed * 1664525u + 1013904223u;
        v = static_cast<float>(seed >> 8) * (10.0f / 16777216.0f) - 5.0f;
    }
    std::vector<double> t;
    for (auto r = 0; r < 400; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f(x.data(), y.data(), n);
        const auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / n);
        asm volatile("" : : "r"(y.data()) : "memory");
    }
    std::nth_element(t.begin(), t.begin() + static_cast<std::ptrdiff_t>(t.size() / 2), t.end());
    return t[t.size() / 2];
}

} // namespace

int main()
{
    // The SIMD paths' coefficients must be sigmoid_tanh_9's.
    for (auto x = -20.0f; x <= 20.0f; x += 0.01f) {
        auto y = 0.0f;
        tanh_std(&x, &y, 1);
        if (y != adc::sigmoid_tanh_9(x)) {
            std::printf("coefficients differ from sigmoid_tanh_9 at x = %.9g\n", double(x));
            return 1;
        }
    }

    const Method methods[] = {
        {"std::sqrt (compiler)", tanh_std},
        {"sqrt + div", tanh_sqrt_div},
        {"rsqrt estimate", tanh_rsqrt<0>},
        {"+ 1 Newton step", tanh_rsqrt<1>},
        {"+ 2 Newton steps", tanh_rsqrt<2>},
    };
#if defined(__aarch64__)
    std::printf("adc rsqrt: sigmoid_tanh_9, NEON, every float in [-20, 20]\n");
#else
    std::printf("adc rsqrt: sigmoid_tanh_9, SSE, every float in [-20, 20]\n");
#endif
    std::printf("  %-22s %12s %13s %10s\n", "method", "max abs dB", "max |y|", "buffer ns");
    for (const auto& m : methods) {
        const auto e = exhaustive(m.f);
        std::printf("  %-22s %12.1f %13.9f %10.3f\n", m.name, 20 * std::log10(e.max_abs), e.max_y, buffer_ns(m.f));
    }
}
