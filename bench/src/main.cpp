// Benchmarks and ULP error for the generated sin approximations.
//
//   adc_bench [timing|ulp|all] [f32|f64|both] [--stride k] [--threads n] [--samples n]
//             [--domain radians|turns|phase] [--dump-worst n] [--dump-dir path]
//   adc_bench tan|exp2|log2|tanh|exp|dbgain|gaindb [timing|ulp|all] [f32|f64|both] [--stride k] [--threads n]
//   adc_bench tables [--quick] [--csv path]
//   adc_bench lutpoly          reproduce "Lookup Tables Are Bad Approximators"
//
// A function name runs that function's methods instead of sin: tan (filter
// cutoffs, src/tan.hpp) or exp2 / log2 / tanh / exp / dbgain / gaindb
// (src/unary.hpp). tables sweeps many-tables-vs-math (src/tables.hpp). Their error
// passes are f32 only.
//
// Defaults: half the cores, all, both, stride 1 (exhaustive f32), 2^24 f64 samples, every
// domain. --dump-worst writes each method's n worst f32 inputs to
// <dump-dir>/worst_<method>.txt (08-coefficient-search reads them back).
// Regenerate the methods with: python bench/generate.py

#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>

#include "lutpoly.hpp"
#include "tables.hpp"
#include "tan.hpp"
#include "timing.hpp"
#include "unary.hpp"
#include "ulp.hpp"

namespace {

auto usage() -> int
{
    std::cerr << "usage: adc_bench [timing|ulp|all] [f32|f64|both] [--stride k] [--threads n] [--samples n]\n"
                 "                 [--domain radians|turns|phase] [--dump-worst n] [--dump-dir path]\n"
                 "       adc_bench tan|exp2|log2|tanh|exp|dbgain|gaindb [timing|ulp|all] [f32|f64|both] [--stride k] [--threads n]\n"
                 "       adc_bench tables [--quick] [--csv path]\n"
                 "       adc_bench lutpoly\n";
    return 2;
}

auto build_info() -> std::string
{
#if defined(__clang__)
    auto compiler = std::format("clang {}.{}.{}", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    auto compiler = std::format("gcc {}.{}.{}", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    auto compiler = std::format("msvc {}", _MSC_VER);
#else
    auto compiler = std::string{"unknown compiler"};
#endif
#if defined(ADC_FAST_MATH)
    compiler += ", fast-math ON";
#else
    compiler += ", fast-math off";
#endif
#if defined(NDEBUG)
    compiler += ", release";
#else
    compiler += ", DEBUG (timings meaningless)";
#endif
    return compiler;
}

} // namespace

int main(int argc, char** argv)
{
    auto mode = std::string{"all"};
    auto precision = std::string{"both"};
    auto ulp_opt = adc::ulp::Options{};
    auto samples = std::size_t{1} << 24;
    auto fn = std::string{"sin"};
    auto csv = std::string{"bench/out/tables.csv"};
    auto quick = false;

    for (auto i = 1; i < argc; ++i) {
        const auto arg = std::string{argv[i]};
        if (i == 1 && (arg == "tan" || arg == "exp2" || arg == "log2" || arg == "tanh" || arg == "exp"
                       || arg == "dbgain" || arg == "gaindb" || arg == "tables" || arg == "lutpoly")) fn = arg;
        else if (arg == "--csv" && i + 1 < argc) csv = argv[++i];
        else if (arg == "--quick") quick = true;
        else if (arg == "timing" || arg == "ulp" || arg == "all") mode = arg;
        else if (arg == "f32" || arg == "f64" || arg == "both") precision = arg;
        else if (arg == "--stride" && i + 1 < argc) ulp_opt.stride = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (arg == "--domain" && i + 1 < argc) ulp_opt.domain = argv[++i];
        else if (arg == "--dump-worst" && i + 1 < argc) ulp_opt.dump_worst = std::strtoull(argv[++i], nullptr, 10);
        else if (arg == "--dump-dir" && i + 1 < argc) ulp_opt.dump_dir = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) {
            adc::err::thread_count = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
            if (adc::err::thread_count == 0) return usage();
        }
        else if (arg == "--samples" && i + 1 < argc) samples = std::strtoull(argv[++i], nullptr, 10);
        else return usage();
    }
    if (ulp_opt.stride == 0) return usage();
    if (ulp_opt.domain != "all" && ulp_opt.domain != "radians" && ulp_opt.domain != "turns" && ulp_opt.domain != "phase") return usage();

    const auto f32 = precision != "f64";
    const auto f64 = precision != "f32";
    if (fn == "lutpoly") {
        std::cout << "adc lookup-table reproduction (" << build_info() << ")\n";
        adc::lutpoly::run();
        return 0;
    }
    if (fn == "tables") {
        std::cout << "adc tables-vs-math bench (" << build_info() << ")\n";
        adc::tables::run(csv, quick);
        return 0;
    }
    if (fn != "sin" && fn != "tan") {
        std::cout << "adc " << fn << " bench (" << build_info() << ")\n";
        if (fn == "exp2") adc::unary::run<adc::unary::Exp2>(mode, f32, f64, ulp_opt.stride);
        if (fn == "log2") adc::unary::run<adc::unary::Log2>(mode, f32, f64, ulp_opt.stride);
        if (fn == "tanh") adc::unary::run<adc::unary::Tanh>(mode, f32, f64, ulp_opt.stride);
        if (fn == "exp") adc::unary::run<adc::unary::Exp>(mode, f32, f64, ulp_opt.stride);
        if (fn == "dbgain") adc::unary::run<adc::unary::DbToGain>(mode, f32, f64, ulp_opt.stride);
        if (fn == "gaindb") adc::unary::run<adc::unary::GainToDb>(mode, f32, f64, ulp_opt.stride);
        return 0;
    }
    if (fn == "tan") {
        std::cout << "adc tan(pi x) bench (" << build_info() << ")\n";
        if (mode != "ulp") {
            if (f32) adc::tan_bench::run_timing<float>("f32");
            if (f64) adc::tan_bench::run_timing<double>("f64");
        }
        if (mode != "timing" && f32) adc::tan_bench::run_ulp(ulp_opt.stride);
        return 0;
    }
    std::cout << "adc sin approximation bench (" << build_info() << ")\n";

    if (mode != "ulp") {
        if (f32) adc::timing::run<float>("f32");
        if (f64) adc::timing::run<double>("f64");
    }
    if (mode != "timing") {
        if (f32) adc::ulp::run_f32(ulp_opt);
        if (f64) adc::ulp::run_f64(samples);
    }
    return 0;
}
