"""Everything numeric on the slides, from the same code as the examples.

Each item below writes slides/data/<name>.json: coefficients, the TeX that
shows them, and the curves a figure draws. The deck loads these at startup
(an element with data-src="<name>"), so nothing on a slide is typed in by
hand. The JSON is committed, so the deck runs without Python.

Every Chebyshev fit on the slides is a projection (fit.set_method below):
interpolate at many nodes, keep the first terms: fit.py's default, and what
bench/generated is built from.

Bench numbers (timing, exhaustive ULP) come from the C++ bench's output,
saved as text under slides/bench/ and parsed here:

    python3 bench/generate.py
    cmake -S bench -B bench/build
    cmake -S bench -B bench/build-fast -DADC_FAST_MATH=ON
    B=bench/build/adc_bench
    for i in 1 2 3 4 5; do $B timing f32; done > slides/bench/sin-timing-f32.txt
    $B ulp f32 --domain radians > slides/bench/sin-ulp-f32-radians.txt
    $B ulp f32 --domain turns > slides/bench/sin-ulp-f32-turns.txt
    $B ulp f32 --domain phase > slides/bench/sin-ulp-f32-phase.txt
    bench/build-fast/adc_bench ulp f32 --domain radians > slides/bench/sin-ulp-f32-radians-fast.txt
    for f in tan exp2 log2 tanh; do $B $f > slides/bench/$f-f32.txt; done
    for f in tan exp2 log2 tanh; do
        for i in 1 2 3 4 5; do $B $f timing f32; done > slides/bench/$f-timing-f32.txt
    done
    for i in 1 2 3 4 5; do bench/build/adc_simple_estrin; done > slides/bench/simple-estrin.txt
    bench/build/adc_rsqrt > slides/bench/tanh-rsqrt-neon.txt
    (SSE: build src/rsqrt.cpp for x86_64 as its header says, run under
    Rosetta, > slides/bench/tanh-rsqrt-sse-rosetta.txt)

Timings are the median of five launches (bench_timing): std::sin's time
moves between launches of the same binary (1.73 or about 2 ns).
Run them on AC power.

Usage:
    python3 slides/generate.py            # every item
    python3 slides/generate.py cheb-sin-5 # just these
"""

import importlib.util
import inspect
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import Chebyshev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))  # for common/
from common import (bits, cppgen, exp2poly, fit, harness, log2poly, metrics, plots, sinpoly, tanhpoly,
                    tanpoly)
from common.metrics import PI

fit.set_method("projection")  # every Chebyshev fit on the slides

OUT_DIR = Path(__file__).parent / "data"

# Example 15 isn't a package; load its nodes and barycentric interpolation.
_spec = importlib.util.spec_from_file_location("equispaced", ROOT / "examples/15-equispaced/equispaced_sin.py")
equispaced = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(equispaced)


def load_example(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tan_example = load_example("tan_pi", "09-tan/tan_pi.py")
BENCH_DIR = Path(__file__).parent / "bench"
ITEMS = {}


def item(name):
    def register(fn):
        ITEMS[name] = fn
        return fn
    return register


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def sig(v, digits=6):
    """A float rounded to `digits` significant digits, for JSON and TeX."""
    return float(f"{v:.{digits}g}")


def curve(y):
    """Samples rounded for the SVG; non-finite values become null."""
    return [sig(v, 5) if np.isfinite(v) else None for v in y]


def tex_poly(coeffs, powers, digits=6):
    """c0 x^p0 + c1 x^p1 + ... with signs folded in: 0.99515\\,x - 0.15775\\,x^3."""
    terms = []
    for c, p in zip(coeffs, powers):
        mag = f"{abs(c):#.{digits}g}"  # "#" keeps trailing zeros: 0.005810
        x = "" if p == 0 else ("\\,x" if p == 1 else f"\\,x^{{{p}}}")
        sign = "-" if c < 0 else "+"
        terms.append((sign, mag + x))
    first_sign, first = terms[0]
    out = [("-" if first_sign == "-" else "") + first]
    out += [f"{sign} {t}" for sign, t in terms[1:]]
    return " ".join(out)


def sci(v, digits=2):
    """3.0e9 -> "3.0 \\times 10^{9}" for TeX."""
    m, e = f"{v:.{digits - 1}e}".split("e")
    return f"{m} \\times 10^{{{int(e)}}}"


def envelope(x, y, bins=600):
    """Bin centres and the max of y in each bin, as log10 (nothing where the
    bin's max is 0). Error curves are spiky at the sample level; the envelope
    is the worst case at each x, which is what we quote. NaNs are skipped:
    relative error is 0/0 where an exact zero is hit exactly."""
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    top = np.zeros(bins)
    np.fmax.at(top, idx, y)
    with np.errstate(divide="ignore"):
        return (edges[:-1] + edges[1:]) / 2, np.where(top > 0, np.log10(top), np.nan)


def log_ticks(lo, hi, step):
    return [{"v": k, "label": f"1e{k}" if k else "1"} for k in range(lo, hi + 1, step)]


PI_TICKS = [{"v": k * PI, "label": lab} for k, lab in
            [(-2, "−2π"), (-1, "−π"), (0, "0"), (1, "π"), (2, "2π")]]


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

@item("cheb-sin-5")
def cheb_sin_5():
    """Example 03 at degree 5: a vanilla Chebyshev fit of sin on [-π, π],
    shown past the interval so the divergence outside is visible."""
    degree = 5
    c = sinpoly.fit_cheb_vanilla(degree)  # sin(x) ~ x * P(x^2)
    powers = list(range(1, degree + 1, 2))
    x = np.linspace(-2 * PI, 2 * PI, 1201)
    tex = "p(x) = " + tex_poly(c, powers, digits=4)
    return {
        "degree": degree,
        "interval": [-PI, PI],
        "coeffs": [sig(v) for v in c],
        "powers": powers,
        "tex": tex,
        "plot": {
            "x": curve(x),
            "xlim": [-2 * PI, 2 * PI],
            "ylim": [-1.6, 1.6],
            "xticks": PI_TICKS,
            "yticks": [-1, 0, 1],
            "band": [-PI, PI],
            "series": [
                {"label": "\\sin x", "color": plots.TRUTH, "y": curve(np.sin(x))},
                {"label": tex, "color": plots.SERIES[0], "dashed": True,
                 "y": curve(sinpoly.eval_odd(x, c, np.float64))},
            ],
        },
    }


@item("taylor-sin")
def taylor_sin():
    """Example 00's Maclaurin polynomials of sin at degrees 1, 3, 5, 7, for the
    morph figure: each step is the next polynomial, drawn over sin. Shown on
    [-2π, 2π] so the runaway away from 0 is visible."""
    degrees = [1, 3, 5, 7]
    x = np.linspace(-2 * PI, 2 * PI, 1201)
    return {
        "degrees": degrees,
        "plot": {
            "x": curve(x),
            "xlim": [-2 * PI, 2 * PI],
            "ylim": [-1.6, 1.6],
            "xticks": PI_TICKS,
            "yticks": [-1, 0, 1],
            "series": [{"label": "\\sin x", "color": plots.TRUTH, "y": curve(np.sin(x))}],
            "color": plots.SERIES[0],
            "steps": [{"degree": n, "y": curve(sinpoly.eval_odd(x, sinpoly.taylor_coeffs(n), np.float64))}
                      for n in degrees],
        },
    }


@item("taylor-sin-7")
def taylor_sin_7():
    """Degree-7 Taylor sin as cppgen emits it: the worked implementation
    example, small enough to match the Horner steps line for line."""
    name, code = cppgen.emit_taylor(7)
    # The Horner steps in u, each coefficient as cppgen writes it, and as a
    # factorial in a comment: c_k = (-1)^k / (2k+1)!, top term first.
    c = sinpoly.taylor_coeffs(7)
    lines = []
    for k in reversed(range(len(c))):
        frac = "1" if k == 0 else f"{'-' if k % 2 else ' '}1/{2 * k + 1}!"
        code_line = f"auto p = {cppgen.lit(c[k])};" if k == len(c) - 1 else f"p = p * u + {cppgen.lit(c[k])};"
        lines.append((code_line, frac if k else " 1"))
    width = max(len(l) for l, _ in lines)
    horner = "\n".join(f"{l:<{width}}  // {f}" for l, f in lines) + "\n"
    # The emitted function with each coefficient's factorial, and u = x^2,
    # as comments.
    code_lines = code.rstrip().splitlines()
    coef_lines = [i for i, line in enumerate(code_lines) if re.search(r"T\(-?[\d.]", line)]
    notes = {i: frac for i, (_, frac) in zip(coef_lines, lines)}
    notes[next(i for i, line in enumerate(code_lines) if "x2 = x * x" in line)] = "u = x^2"
    width = max(len(code_lines[i]) for i in notes)
    for i, note in notes.items():
        code_lines[i] = f"{code_lines[i]:<{width}}  // {note}"
    return {"name": name, "cpp": code.rstrip() + "\n", "horner": horner,
            "cpp_commented": "\n".join(code_lines) + "\n"}


@item("taylor-sin-13")
def taylor_sin_13():
    """The degree-13 Taylor sin as the bench builds it: the C++ that
    bench/generate.py emits, and its error on [-π, π] in f32 (example 00).
    The error figure morphs absolute -> relative -> ULP, each log10 of the
    worst error in each x bin. Maxima: absolute and relative from the
    sample, ULP exhaustive from the bench."""
    degree = 13
    name, code = cppgen.emit_taylor(degree)
    ctx = harness.make_context("radians")
    c = sinpoly.taylor_coeffs(degree)
    err = metrics.pointwise_errors(sinpoly.eval_odd(ctx.x, c, np.float32), ctx.truth)
    xb, abs_log = envelope(ctx.x.astype(np.float64), err["abs"])
    _, rel_log = envelope(ctx.x.astype(np.float64), err["rel"])
    _, ulp_log = envelope(ctx.x.astype(np.float64), err["ulp"])
    max_abs, max_rel = err["abs"].max(), np.nanmax(err["rel"])
    bits = max(0.0, -math.log2(max_rel))
    ulp = bench_ulp()
    return {
        "name": name,
        "cpp": code.rstrip() + "\n",
        "max_abs": sig(max_abs),
        "max_abs_db": sig(20 * np.log10(max_abs), 3),
        "max_rel": sig(max_rel),
        "tex_abs": f"\\max = {sci(max_abs)} \\;\\; ({20 * np.log10(max_abs):.0f}\\,\\text{{dB}})",
        "tex_rel": f"\\max = {max_rel:.0f} \\;\\; ({bits:.0f}\\text{{ bits}})",
        # Exhaustive (C++ bench), not the sampled max: the real number.
        "tex_ulp": sci(float(ulp[name]["max"])),
        "tex_goal": f"{float(ulp['cheb_split_sin_13']['max']):.1f}",
        "tex_ulp_max": f"\\max = {sci(float(ulp[name]['max']))}",
        "tex_samples": f"{ctx.x.size:,}".replace(",", "{,}"),
        "plot": {
            "x": curve(xb),
            "xlim": [-PI, PI],
            "ylim": [-10, 10],
            "xticks": [t for t in PI_TICKS if abs(t["v"]) <= PI + 1e-9],
            "yticks": log_ticks(-10, 10, 5),
            "series": [],
            "color": plots.SERIES[0],
            "steps": [{"y": curve(abs_log)}, {"y": curve(rel_log)}, {"y": curve(ulp_log)}],
        },
    }


@item("taylor-sin-13-spectrum")
def taylor_sin_13_spectrum():
    """The degree-13 Taylor sin's harmonics: one cycle of it in f32 at
    N_FFT phases, so bin k is harmonic k. Levels in dBc, stems for the
    fundamental and harmonics 2..40 (the rest carry on falling), the
    loudest harmonic (marked), and SINAD: the fundamental against all the
    rest together. float-rounded sin's SINAD for comparison."""
    ctx = harness.make_context("radians")
    c = sinpoly.taylor_coeffs(13)
    spec = metrics.summarize_spectrum(sinpoly.eval_odd(ctx.phases, c, np.float32))
    k = np.arange(1, 41)
    h, peak = spec["worst_h"], spec["worst_dbc"]
    return {
        "tex_peak": f"\\text{{loudest}} = {{{peak:.0f}}}\\,\\text{{dBc}} \\;\\; (h_{{{h}}})",
        "tex_sinad": f"\\text{{SINAD}} = {spec['sinad']:.0f}\\,\\text{{dB}}",
        "tex_ref": f"{ctx.ref['sinad']:.0f}\\,\\text{{dB}}",
        "plot": {
            "x": [int(v) for v in k],
            "xlim": [-0.5, 41],
            "ylim": [-160, 5],
            "xticks": [{"v": v, "label": str(v)} for v in (1, 10, 20, 30, 40)],
            "yticks": [{"v": v, "label": f"{v}".replace("-", "−")} for v in (0, -40, -80, -120, -160)],
            "series": [{"text": "Harmonic level (dBc)", "color": plots.SERIES[0], "stems": True, "r": 4,
                        "y": curve(spec["dbc"][1:41])}],
            "points": [{"x": h, "y": sig(peak, 4), "color": plots.SERIES[1]}],
        },
    }


def tex_int_poly(coeffs, var="x^{2}"):
    """Integer coefficients in powers of `var`: a - b\\,x^{2} + c\\,x^{4}."""
    out = []
    for k, c in enumerate(coeffs):
        mag = f"{abs(int(c)):,}".replace(",", "{,}")
        base = var.split("^")[0]
        term = mag if k == 0 else ("" if abs(c) == 1 else f"{mag}\\,") + f"{base}^{{{2 * k}}}"
        out.append(("-" if c < 0 else "+", term))
    first = ("-" if out[0][0] == "-" else "") + out[0][1]
    return " ".join([first] + [f"{sign} {t}" for sign, t in out[1:]])


def tex_frac(f, lead=False):
    """A Fraction as signed TeX: '- \\tfrac{1}{6}', or bare when lead."""
    mag = str(abs(f.numerator)) if f.denominator == 1 else f"\\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}"
    if lead:
        return ("-" if f < 0 else "") + mag
    return ("- " if f < 0 else "+ ") + mag


@item("pade-sin-3-2")
def pade_sin_3_2():
    """The [3/2] Padé of sin worked by hand, for the example slides: the
    Taylor terms through x^5, the product with the denominator multiplied
    out and grouped by power under the numerator, the coefficient matches,
    and the solution (checked against sinpoly.pade_sin). Coloured like the
    derivation slide: Taylor blue, numerator orange, denominator green."""
    _, c1, _, c3, _, c5 = sinpoly.sin_taylor_fractions(5)
    (p1, p3), (_, q2) = sinpoly.pade_sin(3, 2)
    assert c1 == 1 and c3 + q2 * c1 == p3 and c5 + q2 * c3 == 0
    T, N, D = plots.SERIES[0], plots.SERIES[1], plots.SERIES[2]
    MUTED = "#8a8985"
    col = lambda c, t: f"\\textcolor{{{c}}}{{{t}}}"
    q, P1, P3 = col(D, "q_2"), col(N, "p_1"), col(N, "p_3")
    # Taylor coefficient as a signed coloured term; c1 = 1 stays implicit.
    tc = lambda f, lead=False: (("" if f > 0 else "-") if lead else ("+ " if f > 0 else "- ")) + (
        "" if abs(f) == 1 else col(T, tex_frac(abs(f), True)) + "\\,")
    taylor = f"{tc(c1, True)}x {tc(c3)}x^3 {tc(c5)}x^5"
    den = col(D, "1 + q_2 x^2")
    expanded = (f"{tc(c1, True)}x {tc(c3)}x^3 {tc(c5)}x^5 "
                f"+ {q}\\,x^3 {tc(c3)}{q}\\,x^5 {tc(c5)}{q}\\,x^7")
    lab_prod, lab_num = col(MUTED, "\\text{product}"), col(MUTED, "\\text{numerator}")
    g3 = f"({q} {tc(c3)[:-2]})"
    g5 = f"({tc(c5, True)[:-2]} {tc(c3)}{q})"
    grouped = ("\\begin{array}{rllll}"
               f"{lab_prod} & x & + \\,{g3}\\,x^3 & + \\,{g5}\\,x^5 & + \\;O(x^7) \\\\[6pt]"
               f"{lab_num} & {P1}\\,x & + \\,{P3}\\,x^3 & + \\,0\\,x^5 & + \\;O(x^7)"
               "\\end{array}")
    scale = math.lcm(p3.denominator, q2.denominator)
    num, dn = [p1 * scale, p3 * scale], [scale, q2 * scale]
    return {
        "tex_setup": f"{taylor} \\approx \\frac{{{P1} x + {P3} x^3}}{{{den}}}",
        "tex_multiply": f"({den})\\,({taylor}) = {P1} x + {P3} x^3 + O(x^7)",
        "tex_expand": f"= {expanded}",
        "tex_grouped": grouped,
        "tex_match": ("\\begin{alignedat}{2}"
                      f"x^1&:\\quad & {col(T, tex_frac(c1, True))} &= {P1} \\\\"
                      f"x^3&:\\quad & {q} {tc(c3)[:-2]} &= {P3} \\\\"
                      f"x^5&:\\quad & {tc(c5, True)[:-2]} {tc(c3)}{q} &= 0"
                      "\\end{alignedat}"),
        "tex_solve": (f"{q} = {tex_frac(q2, True)}, \\quad {P1} = {tex_frac(p1, True)}, \\quad {P3} = {tex_frac(p3, True)}"
                      f"\\quad\\Rightarrow\\quad \\sin x \\approx x\\,\\frac{{{tex_int_poly(num)}}}{{{tex_int_poly(dn)}}}"),
    }


@item("pade-vs-taylor-5")
def pade_vs_taylor_5():
    """Taylor degree 5 against Padé [3/2], which uses the same Taylor terms
    (m + n = 5; sin is odd, so the numerator is odd and the denominator
    even). f64, on [-2π, 2π] like taylor-sin, so the behaviour past ±π
    shows; worst error on [-π, π] for each."""
    x = np.linspace(-2 * PI, 2 * PI, 1201)
    taylor = sinpoly.eval_odd(x, sinpoly.taylor_coeffs(5), np.float64)
    pade = sinpoly.eval_pade(x, *sinpoly.pade_sin(3, 2), np.float64)
    inside = np.abs(x) <= PI
    worst = lambda y: float(np.abs(y - np.sin(x))[inside].max())
    return {
        "tex_taylor": sci(worst(taylor)),
        "tex_pade": sci(worst(pade)),
        "plot": {
            "x": curve(x),
            "xlim": [-2 * PI, 2 * PI],
            "ylim": [-1.6, 1.6],
            "xticks": PI_TICKS,
            "yticks": [-1, 0, 1],
            "band": [-PI, PI],
            "series": [
                {"label": "\\sin x", "color": plots.TRUTH, "y": curve(np.sin(x))},
                {"text": "Taylor, degree 5", "color": plots.SERIES[0], "y": curve(taylor)},
                {"text": "Padé [3/2]", "color": plots.SERIES[1], "y": curve(pade)},
            ],
        },
    }


@item("pade-sin-7-6")
def pade_sin_7_6():
    """Example 02's Padé [7/6] (JUCE's FastMathApproximations::sin): the exact
    rational in integers, and its error on [-π, π] in f32 against Taylor
    degree 13 (same 14 Taylor terms matched). log10 of the worst absolute
    error in each x bin."""
    pe, qe = sinpoly.pade_sin(7, 6)
    scale = math.lcm(*(f.denominator for f in pe + qe))
    num, den = [f * scale for f in pe], [f * scale for f in qe]
    ctx = harness.make_context("radians")
    xf = ctx.x.astype(np.float64)
    taylor = metrics.abs_error(sinpoly.eval_odd(ctx.x, sinpoly.taylor_coeffs(13), np.float32), ctx.truth)
    pade = metrics.abs_error(sinpoly.eval_pade(ctx.x, pe, qe, np.float32), ctx.truth)
    xb, taylor_log = envelope(xf, taylor)
    _, pade_log = envelope(xf, pade)
    return {
        "tex": f"\\sin x \\approx x\\,\\frac{{{tex_int_poly(num)}}}{{{tex_int_poly(den)}}}",
        "max_abs_db": sig(20 * np.log10(pade.max()), 3),
        "plot": {
            "x": curve(xb),
            "xlim": [-PI, PI],
            "ylim": [-10, -3],
            "xticks": [t for t in PI_TICKS if abs(t["v"]) <= PI + 1e-9],
            "yticks": log_ticks(-10, -4, 2),
            "series": [
                {"text": "Taylor, degree 13", "color": plots.SERIES[0], "y": curve(taylor_log)},
                {"text": "Padé [7/6]", "color": plots.SERIES[1], "y": curve(pade_log)},
            ],
        },
    }


COMPARE = "#c4c3bf"  # a faint neutral, for a comparison curve behind the subject


@item("pade-tanh-7-6")
def pade_tanh_7_6():
    """JUCE's FastMathApproximations::tanh is Padé [7/6] too: the exact
    rational in integers (12-tanh's pade_tanh)."""
    pe, qe = tanhpoly.pade_tanh(7, 6)
    scale = math.lcm(*(f.denominator for f in list(pe) + list(qe)))
    num, den = [f * scale for f in pe], [f * scale for f in qe]
    return {"tex": f"\\tanh x \\approx x\\,\\frac{{{tex_int_poly(num)}}}{{{tex_int_poly(den)}}}"}


def juce_plot(x, truth, approx, label, xlim, xticks, band):
    """A small function-and-approximation plot for Padé in the wild."""
    return {
        "x": curve(x),
        "xlim": xlim,
        "ylim": [-1.6, 1.6],
        "xticks": xticks,
        "yticks": [-1, 0, 1],
        "band": band,
        "series": [
            {"label": label, "color": plots.TRUTH, "y": curve(truth)},
            {"text": "Padé [7/6]", "color": plots.SERIES[0], "dashed": True, "y": curve(approx)},
        ],
    }


@item("juce-sin")
def juce_sin():
    """JUCE's Padé [7/6] sin against sin on [-3π, 3π]; [-π, π] (what it's
    for) shaded. It tracks to about ±5, then runs away."""
    x = np.linspace(-3 * PI, 3 * PI, 900)
    pe, qe = sinpoly.pade_sin(7, 6)
    ticks = [{"v": k * PI, "label": lab} for k, lab in [(-3, "−3π"), (-1, "−π"), (1, "π"), (3, "3π")]]
    return {"plot": juce_plot(x, np.sin(x), sinpoly.eval_pade(x, pe, qe, np.float64), "\\sin x",
                              [-3 * PI, 3 * PI], ticks, [-PI, PI])}


@item("juce-tanh")
def juce_tanh():
    """JUCE's Padé [7/6] tanh, unclamped, against tanh on [-24, 24]; [-5, 5]
    shaded (roughly where JUCE says to use it). It passes 1 just before 5
    and keeps creeping up: 1.009 at 10, 1.13 at 20."""
    x = np.linspace(-24, 24, 900)
    pe, qe = tanhpoly.pade_tanh(7, 6)
    ticks = [{"v": v, "label": str(v).replace("-", "−")} for v in (-20, -5, 5, 20)]
    return {"plot": juce_plot(x, np.tanh(x), tanhpoly.eval_pade(x, pe, qe, np.float64, clamp=False), "\\tanh x",
                              [-24, 24], ticks, [-5, 5])}


def ulp_text(v):
    """A max ULP for TeX: 4.44 as is, 1.8e7 in scientific."""
    return f"{v:.3g}" if v < 100 else sci(v)


def taylor_13_f32(x):
    return sinpoly.eval_odd(x, sinpoly.taylor_coeffs(13), np.float32)


def plot_position(ctx, domain):
    """A context's inputs placed on a plot in `domain`, and the order that
    sorts them: as they are when the domains match; radians x drawn at
    x / 2π, wrapped to [0, 1), on a turns plot, so a radians baseline lines
    up with a turns method."""
    pos = ctx.x.astype(np.float64)
    if ctx.domain == domain:
        return pos, slice(None)
    pos = np.where(pos < 0, pos + 2 * PI, pos) / (2 * PI)
    return pos, np.argsort(pos, kind="stable")


TURNS_TICKS = [{"v": v, "label": lab} for v, lab in [(0, "0"), (0.25, "¼"), (0.5, "½"), (0.75, "¾"), (1, "1")]]


def vs_taylor_errors(label, approx, bench_name, base=("Taylor, degree 13", taylor_13_f32), domain="radians"):
    """How did we do, for `approx` (x -> f32 sin x) on [-π, π]: absolute,
    relative and ULP error as morph steps, log10 of the worst error per
    bin, with a baseline (`base`: label and function; Taylor 13 unless
    given) moving alongside in a faint neutral for comparison. Readouts
    are approx's: absolute and relative from the sample, ULP exhaustive
    from the bench (`bench_name`). With domain="turns", approx takes turns
    and the plot is on [0, 1]; the baseline stays in radians, drawn at the
    same angle (plot_position)."""
    ctx = harness.make_context("radians")
    mine = harness.make_context(domain)
    errs = {"taylor": (ctx, metrics.pointwise_errors(base[1](ctx.x), ctx.truth)),
            "approx": (mine, metrics.pointwise_errors(approx(mine.x), mine.truth))}
    steps = {}
    for key, (c, err) in errs.items():
        theta, order = plot_position(c, domain)
        steps[key] = []
        for k in ("abs", "rel", "ulp"):
            xb, y = envelope(theta[order], err[k][order])
            steps[key].append({"y": curve(y)})
    err = errs["approx"][1]
    max_abs, max_rel = err["abs"].max(), np.nanmax(err["rel"])
    bits = max(0.0, -math.log2(max_rel))
    return {
        "tex_abs": f"\\max = {sci(max_abs)} \\;\\; ({20 * np.log10(max_abs):.0f}\\,\\text{{dB}})",
        "tex_rel": (f"\\max = {max_rel:.0f}" if max_rel >= 1 else f"\\max = {sci(max_rel)}")
                   + f" \\;\\; ({bits:.0f}\\text{{ bits}})",
        "tex_ulp_max": f"\\max = {ulp_text(float(bench_ulp(domain)[bench_name]['max']))}",
        "plot": {
            "x": curve(xb),
            "xlim": [0, 1] if domain == "turns" else [-PI, PI],
            "ylim": [-10, 10],
            "xticks": TURNS_TICKS if domain == "turns" else [t for t in PI_TICKS if abs(t["v"]) <= PI + 1e-9],
            "yticks": log_ticks(-10, 10, 5),
            "series": [],
            "curves": [
                {"text": base[0], "color": COMPARE, "steps": steps["taylor"]},
                {"text": label, "color": plots.SERIES[0], "steps": steps["approx"]},
            ],
        },
    }


def vs_taylor_spectrum(label, approx, floor=-160, base=("Taylor, degree 13", taylor_13_f32), domain="radians"):
    """`approx`'s harmonics, one cycle in f32 (bin k is harmonic k), as
    stems for harmonics 1..40 with the loudest of 2..40 marked, and a baseline's
    (`base`: label and function; Taylor 13 unless given) beside them in a
    faint neutral for comparison; loudest and SINAD for approx. `floor` is
    the bottom of the dBc axis. With domain="turns", approx takes turns:
    the same cycle, so harmonic k is bin k for both."""
    ctx = harness.make_context("radians")
    taylor = metrics.summarize_spectrum(base[1](ctx.phases))
    mine = metrics.summarize_spectrum(approx(metrics.fft_phases(domain)))
    k = np.arange(1, 41)
    # Loudest of the plotted harmonics: turns' rounding noise peaks at
    # Nyquist and N/4 (inputs are exactly k / 2^16), which isn't a harmonic.
    h = int(np.argmax(mine["dbc"][2:41])) + 2
    peak = float(mine["dbc"][h])
    return {
        "tex_peak": f"\\text{{loudest}} = {{{peak:.0f}}}\\,\\text{{dBc}} \\;\\; (h_{{{h}}})",
        "tex_sinad": f"\\text{{SINAD}} = {mine['sinad']:.0f}\\,\\text{{dB}}",
        "tex_sinad_taylor": f"{taylor['sinad']:.0f}\\,\\text{{dB}}",
        "plot": {
            "x": [int(v) for v in k],
            "xlim": [-0.5, 41],
            "ylim": [floor, 5],
            "xticks": [{"v": v, "label": str(v)} for v in (1, 10, 20, 30, 40)],
            "yticks": [{"v": v, "label": f"{v}".replace("-", "−")} for v in range(0, floor - 1, -40)],
            "series": [
                {"text": base[0], "color": COMPARE, "stems": True, "r": 3.5, "dx": -0.2,
                 "y": curve(taylor["dbc"][1:41])},
                {"text": label, "color": plots.SERIES[0], "stems": True, "r": 3.5, "dx": 0.2,
                 "y": curve(mine["dbc"][1:41])},
            ],
            "points": [{"x": h + 0.2, "y": sig(peak, 4), "color": plots.SERIES[1]}],
        },
    }


@item("pade-vs-taylor-13")
def pade_vs_taylor_13():
    """How did we do, for Padé [7/6] (JUCE's), against Taylor 13 (the same
    14 Taylor terms); see vs_taylor_errors."""
    pe, qe = sinpoly.pade_sin(7, 6)
    return vs_taylor_errors("Padé [7/6]", lambda x: sinpoly.eval_pade(x, pe, qe, np.float32), "pade_sin_7_6")


@item("pade-vs-taylor-spectrum")
def pade_vs_taylor_spectrum():
    """Padé [7/6]'s harmonics beside Taylor 13's; see vs_taylor_spectrum."""
    pe, qe = sinpoly.pade_sin(7, 6)
    return vs_taylor_spectrum("Padé [7/6]", lambda x: sinpoly.eval_pade(x, pe, qe, np.float32))


@item("cheb-vs-taylor-13")
def cheb_vs_taylor_13():
    """How did we do, for Chebyshev 13 (example 03, projected), against
    Taylor 13; see vs_taylor_errors."""
    c = sinpoly.fit_cheb_vanilla(DEGREE)
    return vs_taylor_errors("Chebyshev, degree 13", lambda x: sinpoly.eval_odd(x, c, np.float32), "cheb_sin_13")


@item("cheb-factored-vs-vanilla")
def cheb_factored_vs_vanilla():
    """How did we do, for factored Chebyshev 13 (example 04) against
    vanilla Chebyshev 13 (example 03); see vs_taylor_errors."""
    v = sinpoly.fit_cheb_vanilla(DEGREE)
    f = sinpoly.fit_cheb_factored(DEGREE)
    return vs_taylor_errors("Factored zeros", lambda x: sinpoly.eval_factored(x, f, np.float32),
                            "cheb_factored_sin_13",
                            base=("Chebyshev, degree 13", lambda x: sinpoly.eval_odd(x, v, np.float32)))


@item("cheb-split-13")
def cheb_split_13():
    """Example 05 as the bench builds it: factored zeros with π carried in
    two constants, (x - π_hi) - π_lo, guarded so it survives -ffast-math and
    /fp:fast and stays inlined (github.com/rygrob/adc-msvc). Comments
    stripped (doc lines and trailing //). Split π in C++ shows the guarded
    function in full, under the plain name (it's the only one on the slides)
    and with pi_lo on three lines, so the macros fit beside it. The macros
    drop the #else (GCC: unguarded) branch and the indent under #if. The
    retired Keeping the split under fast-math slide keeps cpp_guarded, the
    polynomial lines elided."""
    def strip(code):
        lines = [re.sub(r"\s+//.*$", "", ln) for ln in code.splitlines() if not ln.startswith("///")]
        return "\n".join(lines).strip() + "\n"
    name, _ = cppgen.emit_cheb_split(DEGREE)
    guarded_name, guarded = cppgen.emit_cheb_split_guarded(DEGREE)
    macros, fn = guarded.split("\n\nFP_PRECISE_BEGIN\n", 1)
    macros = re.sub(r"#else\n.*?(?=#endif)", "", macros, flags=re.S)
    fn = "FP_PRECISE_BEGIN\n" + fn
    full = re.sub(r"(    constexpr auto pi_lo = .*?) (\? .*?) (: .*?;)\n", r"\1\n        \2\n        \3\n",
                  fn.replace(guarded_name, name))
    assert full.count("\n        ? ") == 1
    macros = macros.replace("\n    #define", "\n#define")
    elided = re.sub(r"    const auto x2 = .*?\n(    p = .*?\n)+", "    const auto p = /* P(x * x), as before */;\n", fn, flags=re.S)
    return {"name": name, "cpp": strip(full),
            "cpp_macros": strip(macros), "cpp_guarded": strip(elided)}


@item("cheb-split-vs-factored")
def cheb_split_vs_factored():
    """How did we do, for split π (example 05) against factored zeros
    (example 04); see vs_taylor_errors."""
    f = sinpoly.fit_cheb_factored(DEGREE)
    return vs_taylor_errors("Split π", lambda x: sinpoly.eval_split(x, f, np.float32), "cheb_split_sin_13",
                            base=("Factored zeros", lambda x: sinpoly.eval_factored(x, f, np.float32)))


@item("cheb-factored-vs-vanilla-spectrum")
def cheb_factored_vs_vanilla_spectrum():
    """Factored zeros' harmonics beside vanilla Chebyshev 13's; see
    vs_taylor_spectrum."""
    v = sinpoly.fit_cheb_vanilla(DEGREE)
    f = sinpoly.fit_cheb_factored(DEGREE)
    return vs_taylor_spectrum("Factored zeros", lambda x: sinpoly.eval_factored(x, f, np.float32), floor=-180,
                              base=("Chebyshev, degree 13", lambda x: sinpoly.eval_odd(x, v, np.float32)))


@item("cheb-split-vs-factored-spectrum")
def cheb_split_vs_factored_spectrum():
    """Split π's harmonics beside factored zeros'; see vs_taylor_spectrum."""
    f = sinpoly.fit_cheb_factored(DEGREE)
    return vs_taylor_spectrum("Split π", lambda x: sinpoly.eval_split(x, f, np.float32), floor=-180,
                              base=("Factored zeros", lambda x: sinpoly.eval_factored(x, f, np.float32)))


@item("cheb-turns-vs-split")
def cheb_turns_vs_split():
    """How did we do, for turns (example 06) against split π (example 05),
    turns drawn at the same angle; see vs_taylor_errors."""
    f = sinpoly.fit_cheb_factored(DEGREE)
    t = sinpoly.fit_cheb_turns(DEGREE)
    return vs_taylor_errors("Turns", lambda x: sinpoly.eval_turns(x, t, np.float32), "cheb_turns_sin_13",
                            base=("Split π", lambda x: sinpoly.eval_split(x, f, np.float32)), domain="turns")


@item("cheb-turns-vs-split-spectrum")
def cheb_turns_vs_split_spectrum():
    """Turns' harmonics beside split π's; see vs_taylor_spectrum."""
    f = sinpoly.fit_cheb_factored(DEGREE)
    t = sinpoly.fit_cheb_turns(DEGREE)
    return vs_taylor_spectrum("Turns", lambda x: sinpoly.eval_turns(x, t, np.float32), floor=-180,
                              base=("Split π", lambda x: sinpoly.eval_split(x, f, np.float32)), domain="turns")


@item("cheb-vs-taylor-spectrum")
def cheb_vs_taylor_spectrum():
    """Chebyshev 13's harmonics beside Taylor 13's; see vs_taylor_spectrum."""
    c = sinpoly.fit_cheb_vanilla(DEGREE)
    return vs_taylor_spectrum("Chebyshev, degree 13", lambda x: sinpoly.eval_odd(x, c, np.float32), floor=-180)


@item("lut-sin-16")
def lut_sin_16():
    """Example 01 with a tiny 16-entry table so the steps show: nearest, then
    linear, for the morph figure over sin. Evaluated in f64; this is about
    the shape, not rounding."""
    size = 16
    x = np.linspace(-PI, PI, 2401)
    table = sinpoly.make_table(size, np.float64)
    return {
        "size": size,
        "plot": {
            "x": curve(x),
            "xlim": [-PI, PI],
            "ylim": [-1.2, 1.2],
            "xticks": [t for t in PI_TICKS if abs(t["v"]) <= PI + 1e-9],
            "yticks": [-1, 0, 1],
            "series": [{"label": "\\sin x", "color": plots.TRUTH, "y": curve(np.sin(x))}],
            "color": plots.SERIES[0],
            "steps": [{"y": curve(sinpoly.eval_lut(x, table, interp, np.float64))}
                      for interp in ("nearest", "linear")],
        },
    }


@item("lut-sin-size")
def lut_sin_size():
    """Max absolute error (dB) against table size, nearest and linear, in f32
    on example 01's sample grid (a lower bound at large sizes), with Taylor 13
    and Padé [7/6] as horizontal references. x is log2 of the size."""
    ctx = harness.make_context("radians")
    exps = list(range(4, 21))
    def max_db(interp, n):
        y = sinpoly.eval_lut(ctx.x, sinpoly.make_table(2**n, np.float32), interp, np.float32)
        return 20 * np.log10(metrics.abs_error(y, ctx.truth).max())
    nearest = [max_db("nearest", n) for n in exps]
    linear = [max_db("linear", n) for n in exps]
    taylor = 20 * np.log10(metrics.abs_error(
        sinpoly.eval_odd(ctx.x, sinpoly.taylor_coeffs(13), np.float32), ctx.truth).max())
    flat = lambda v: [sig(v, 4)] * len(exps)
    labels = {4: "16", 8: "256", 10: "1K", 12: "4K", 16: "64K", 20: "1M"}
    # Table size where each curve first reaches Taylor's error.
    reach = lambda ys: next((2**n for n, v in zip(exps, ys) if v <= taylor), None)
    return {
        "taylor_db": sig(taylor, 3),
        "nearest_match": reach(nearest),
        "linear_match": reach(linear),
        "plot": {
            "x": exps,
            "xlim": [exps[0], exps[-1]],
            "ylim": [-160, 0],
            "xticks": [{"v": n, "label": lab} for n, lab in labels.items()],
            "yticks": [{"v": v, "label": f"{v} dB" if v else "0 dB"} for v in range(-160, 1, 40)],
            "series": [
                {"text": "Nearest", "color": plots.SERIES[0], "y": curve(np.array(nearest))},
                {"text": "Linear", "color": plots.SERIES[1], "y": curve(np.array(linear))},
                {"text": "Taylor, degree 13", "color": plots.TRUTH, "dashed": True, "y": flat(taylor)},
            ],
        },
    }


def f32_fields(v):
    """Sign, biased exponent, fraction and the 32-bit string of float32(v)."""
    u = int(np.float32(v).view(np.uint32))
    return u >> 31, (u >> 23) & 0xFF, u & 0x7FFFFF, f"{u:032b}"


def f32_decimal(v, digits=9):
    """The stored value as a short exact decimal if it has one, else
    `digits` significant digits and an ellipsis."""
    x = float(np.float32(v))
    short = str(int(x)) if x.is_integer() else repr(x)
    return short if len(short.lstrip("-").replace(".", "")) <= digits else f"{x:.{digits}g}\\ldots"


def float_step(typed, v):
    """One value field by field: its bits, each field decoded, the decoded
    (-1)^s 2^e 1.f, the ULP in its binade, and pos = e + (m - 1)."""
    s, E, F, bits = f32_fields(v)
    e = E - 127
    m = 1 + F / 2**23
    m_tex = f32_decimal(m)  # m is exact in f32 too: 24 bits
    value = f32_decimal(v)
    eq = "=" if repr(float(np.float32(v))) == repr(v) else "\\ne"
    return {
        "typed": typed,
        "bits": bits,
        "sign_tex": "+" if s == 0 else "-",
        "exp_tex": f"{E} - 127 = {e}",
        "frac_tex": f"1.f = {m_tex}",
        "value_tex": (f"(-1)^{{{s}}} \\times 2^{{{e}}} \\times {m_tex} = {value}"
                      + ("" if eq == "=" else f" \\;{eq}\\; {typed}")),
        "ulp_tex": f"\\mathrm{{ulp}} = 2^{{{e - 23}}}",
        "pos": e + (m - 1),
    }


@item("float-anatomy")
def float_anatomy():
    """IEEE single precision, field by field, for a few values: their bits,
    the decoded (-1)^s 2^e 1.f, the ULP in their binade, and a position on an
    axis where every binade has the same width (pos = e + f, so moving one
    binade is one unit). 0.75, 1.5 and 3 share their fraction bits."""
    lo_e, hi_e = -4, 3
    steps = [float_step(typed, v) for typed, v in
             [("1", 1.0), ("1.5", 1.5), ("3", 3.0), ("0.75", 0.75), ("-0.75", -0.75), ("0.1", 0.1)]]
    return {
        "steps": steps,
        "binades": [{"e": e, "label": (f"1/{2**-e}" if e < 0 else str(2**e))} for e in range(lo_e, hi_e + 1)],
    }


@item("float-pi-e")
def float_pi_e():
    """π and e between their neighbouring floats: the two floats either side,
    where the constant falls between them (0..1 of the gap), which way it
    rounds and by how many ULPs, and the bits of the float it rounds to."""
    steps = []
    for name, tex_name, c in [("pi", "\\pi", math.pi), ("e", "e", math.e)]:
        f = np.float32(c)
        lo = f if float(f) < c else np.nextafter(f, np.float32(-np.inf))
        hi = np.nextafter(lo, np.float32(np.inf))
        gap = float(hi) - float(lo)
        nearest = float(f)
        err = (nearest - c) / gap
        s, E, F, bits = f32_fields(c)
        steps.append({
            "name": name,
            "bits": bits,
            "sign_tex": "+",
            "exp_tex": f"{E} - 127 = {E - 127}",
            "frac_tex": f"1.f = {f32_decimal(1 + F / 2**23)}",
            "ticks": [f"{float(v):.9g}" for v in (np.nextafter(lo, np.float32(-np.inf)), lo, hi,
                                                   np.nextafter(hi, np.float32(np.inf)))],
            "at": (c - float(lo)) / gap,
            "rounds": "up" if nearest > c else "down",
            "true_tex": f"{tex_name} = {c:.12g}\\ldots",
            "f32_tex": f"\\mathrm{{f32}}({tex_name}) = {nearest:.12g}\\ldots",
            "err_tex": (f"\\mathrm{{f32}}({tex_name}) - {tex_name} = {'+' if err > 0 else '-'}{abs(err):.2f}"
                        f"\\,\\mathrm{{ulp}}"),
            "tex_name": tex_name,
        })
    return {"steps": steps, "sin_f32_pi": sig(float(np.sin(np.float64(np.float32(np.pi)))), 4)}


def interp_on(nodes, f, x):
    """Example 15's barycentric interpolant of f at `nodes`, on `x`; exact
    where x lands on a node (the barycentric form divides by zero there)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = equispaced.interpolate(nodes, f(nodes), x)
    hit = ~np.isfinite(p)
    p[hit] = f(x[hit])
    return p


@item("cheb-nodes-rows")
def cheb_nodes_rows():
    """Chebyshev nodes of the first kind for N = 1 .. 20, one row each (N = 1
    at the top): x_k = cos((k + 1/2) π / N). Stacked, the rows trace out
    the cosines' arcs."""
    rows = 20
    pts = []
    for n in range(1, rows + 1):
        for k in range(n):
            pts.append({"x": sig(math.cos((k + 0.5) * PI / n)), "y": -n, "color": plots.SERIES[1], "r": 4.5})
    return {
        "plot": {
            "x": [],
            "xlim": [-1.06, 1.06],
            "ylim": [-rows - 0.8, 0.2],
            "xticks": [{"v": v, "label": str(v).replace("-", "−")} for v in (-1, 0, 1)],
            "yticks": [{"v": -n, "label": f"N = {n}" if n in (1, 5, 10, 15, 20) else ""} for n in (1, 5, 10, 15, 20)],
            "series": [],
            "points": pts,
        },
    }


@item("cheb-polys")
def cheb_polys():
    """T_0 .. T_5 on [-1, 1], for the definition slide: each between -1 and
    1, with n + 1 extremes, all of them ±1. Each label is the polynomial in
    powers of x (highest first), from numpy's cheb2poly; and the other way,
    x^5 as a sum of T_k (poly2cheb), for the basis bullet."""
    t = np.linspace(-1, 1, 601)

    def tex_t(n):
        c = np.polynomial.chebyshev.cheb2poly([0] * n + [1])
        terms = [(int(c[k]), k) for k in range(n, -1, -1) if c[k]]
        out = []
        for i, (a, k) in enumerate(terms):
            mag = "" if abs(a) == 1 and k else str(abs(a))
            x = "" if k == 0 else ("x" if k == 1 else f"x^{k}")
            sign = "-" if a < 0 else ("+" if i else "")
            out.append(f"{sign} {mag}{x}".strip() if i else f"{sign}{mag}{x}")
        return f"T_{n} = " + " ".join(out)

    # The other way round: x^5 in the T_k, over the common denominator 2^4.
    m = 5
    den = 2 ** (m - 1)
    a = np.polynomial.chebyshev.poly2cheb([0] * m + [1]) * den
    coef = lambda c: "" if c == 1 else f"{c}\\,"
    terms = " + ".join(f"{coef(int(round(a[k])))}T_{k}" for k in range(m + 1) if round(a[k]))
    tex_x5 = f"x^{m} = \\tfrac{{1}}{{{den}}}\\left({terms}\\right)"

    return {
        "tex_x5": tex_x5,
        "plot": {
            "x": curve(t),
            "xlim": [-1.04, 1.04],
            "ylim": [-1.15, 1.15],
            "xticks": [{"v": v, "label": str(v).replace("-", "−")} for v in (-1, 0, 1)],
            "yticks": [-1, 0, 1],
            "series": [{"label": tex_t(n), "color": plots.SERIES[n], "y": curve(np.cos(n * np.arccos(t)))}
                       for n in range(6)],
        },
    }


@item("interp-sin-5")
def interp_sin_5():
    """Interpolation step by step: sin on [-π, π], 5 equally spaced points,
    the polynomial through them (degree 4; the x^4 term is zero, sin being
    odd) and its error p - sin: zero at every point, oscillating between,
    max 0.18. f64, example 15's barycentric interpolation."""
    n = 5
    x = np.linspace(-PI, PI, 601)
    nodes = np.linspace(-PI, PI, n)
    p = interp_on(nodes, np.sin, x)
    err = p - np.sin(x)
    return {
        "x": curve(x),
        "f": curve(np.sin(x)),
        "p": curve(p),
        "err": curve(err),
        "nodes": [{"x": sig(v), "y": sig(np.sin(v))} for v in nodes],
        "max": sig(np.abs(err).max(), 2),
        "ylim": [-1.3, 1.3],
        "err_lim": 0.25,
        "xticks": [t for t in PI_TICKS if abs(t["v"]) <= PI + 1e-9],
        "yticks": [-1, 0, 1],
        "label": "\\sin x \\text{ and } p(x)",
        "err_label": "p(x) - \\sin x",
        "color": plots.SERIES[0],
        "err_color": plots.SERIES[1],
    }


@item("nodes-sin")
def nodes_sin():
    """Example 15's interpolation with 11 points: sin on [-π, π] as the
    points slide from equally spaced (q = 0) to Chebyshev (q = 1). 11, not
    10: sin is odd, so with an odd point count the interpolant is odd
    (degree 9, x^10 term zero) and nothing aliases onto its top term; the
    Chebyshev error comes out equiripple (smallest bump 0.94 of the
    largest, against 0.06 with 10 points). Frames along q, each with the
    points, the interpolant and its signed error; the figure blends
    neighbouring frames. f64, as in example 15."""
    degree, n_frames = 10, 25
    x = np.linspace(-PI, PI, 401)
    eq, ch = equispaced.equispaced_nodes(degree), equispaced.chebyshev_nodes(degree)
    frames = []
    for q in np.linspace(0, 1, n_frames):
        nodes = (1 - q) * eq + q * ch
        p = interp_on(nodes, np.sin, x)
        err = p - np.sin(x)
        frames.append({"nodes": curve(nodes), "p": curve(p), "err": [sig(v, 3) for v in err],
                       "max": sig(np.abs(err).max(), 4)})
    cheb_max = frames[-1]["max"]
    return {
        "degree": degree,
        "x": curve(x),
        "sin": curve(np.sin(x)),
        "frames": frames,
        "err_lim": sig(1.6 * cheb_max, 3),
        "max_equi": frames[0]["max"],
        "max_cheb": cheb_max,
        "xticks": [t for t in PI_TICKS if abs(t["v"]) <= PI + 1e-9],
    }


def interp_envelope_steps():
    """sin on [-π, π] interpolated at N = n + 2 Chebyshev nodes for n = 1, 3,
    5, 7 (odd N: the interpolant is odd, degree n, and nothing aliases onto
    its top term), with the error split as the formula splits it:

        sin x - p(x) = g(x) ω(x),   g = f^(N)(ξ) / N!,   ω = Π (x - x_k)

    At the nodes ω = 2^(1-N) π^N T_N(x/π), the even ripple we chose; g is
    the part f decides, computed exactly as (sin - p) / ω (a divided
    difference, smooth through the nodes). The envelope ±|g| max|ω| bends
    with f and flattens as N grows. Each step is scaled to its envelope's
    peak, since the error falls ~3000x from n = 1 to 7."""
    x = np.linspace(-PI, PI, 801)
    steps = []
    for n in (1, 3, 5, 7):
        N = n + 2
        nodes = PI * np.cos((np.arange(N) + 0.5) * PI / N)
        p = interp_on(nodes, np.sin, x)
        err = np.sin(x) - p
        w = np.prod([x - v for v in nodes], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = err / w
        bad = np.abs(w) < 1e-9 * np.abs(w).max()  # at a node: 0/0, fill from neighbours
        g[bad] = np.interp(x[bad], x[~bad], g[~bad])
        env = np.abs(g) * 2.0 ** (1 - N) * PI ** N
        scale = env.max()
        steps.append({"n": n, "N": N, "nodes": nodes, "p": p, "err": err / scale, "env": env / scale,
                      "max": np.abs(err).max(), "ratio": env.min() / env.max()})
    return x, steps


@item("interp-env-fit")
def interp_env_fit():
    """Left pane: sin and its interpolant p_n through N = n + 2 Chebyshev
    nodes, n = 1 .. 7 (see interp_envelope_steps)."""
    x, steps = interp_envelope_steps()
    return {
        "plot": {
            "x": curve(x),
            "xlim": [-PI, PI],
            "ylim": [-1.3, 1.3],
            "xticks": [t for t in PI_TICKS if abs(t["v"]) <= PI + 1e-9],
            "yticks": [-1, 0, 1],
            "series": [{"label": "\\sin x", "color": plots.TRUTH, "y": curve(np.sin(x))}],
            "color": plots.SERIES[0],
            "steps": [{"y": curve(st["p"])} for st in steps],
            "stepPoints": [[{"x": sig(v), "y": sig(np.sin(v)), "color": plots.SERIES[1]} for v in st["nodes"]]
                           for st in steps],
        },
    }


@item("interp-env-err")
def interp_env_err():
    """Right pane: the signed error sin - p_n inside its envelope ±|g| max|ω|,
    each step scaled to the envelope's peak; zeros at the nodes. Captions
    carry each step's real max error and how flat the envelope is."""
    x, steps = interp_envelope_steps()
    return {
        **{f"tex_{st['n']}": f"p_{{{st['n']}}},\\ N = {st['N']}\\text{{ nodes: max error }} {sci(st['max'])}"
           for st in steps},
        "ratios": {str(st["n"]): sig(st["ratio"], 2) for st in steps},
        "plot": {
            "x": curve(x),
            "xlim": [-PI, PI],
            "ylim": [-1.2, 1.2],
            "xticks": [t for t in PI_TICKS if abs(t["v"]) <= PI + 1e-9],
            "yticks": [{"v": -1, "label": "−peak"}, {"v": 0, "label": "0"}, {"v": 1, "label": "peak"}],
            "series": [],
            "curves": [
                {"text": "envelope, set by f", "color": plots.SERIES[2], "dashed": True,
                 "steps": [{"y": curve(st["env"])} for st in steps]},
                {"text": "", "color": plots.SERIES[2], "dashed": True, "nolegend": True,
                 "steps": [{"y": curve(-st["env"])} for st in steps]},
                {"text": "error", "color": plots.SERIES[0], "steps": [{"y": curve(st["err"])} for st in steps]},
            ],
            "stepPoints": [[{"x": sig(v), "y": 0, "color": plots.SERIES[1], "r": 5} for v in st["nodes"]]
                           for st in steps],
        },
    }


@item("float-line")
def float_line():
    """A toy float with a 3-bit mantissa: 8 values per binade, so the gap
    doubles at every power of 2. For the ULP slide."""
    binades = [-2, -1, 0, 1]
    values = [m / 8 * 2.0**e for e in binades for m in range(8, 16)] + [2.0 ** (binades[-1] + 1)]
    return {
        "values": values,
        "binades": [{"lo": 2.0**e, "hi": 2.0 ** (e + 1), "ulp": f"2^{{{e - 3}}}"} for e in binades],
    }


@item("error-line")
def error_line():
    """The three error metrics on one number line, for Measuring error: the
    same toy float (3-bit mantissa), a true value f = π/2 between floats and
    a computed p on a float. Absolute is the gap, relative the gap against
    the distance to 0, ULP the gap in units of the float spacing at f."""
    values = [0.0] + [m / 8 * 2.0**e for e in range(-4, 2) for m in range(8, 16)] + [4.0]
    f, p = math.pi / 2, 15 / 8
    e = math.floor(math.log2(f))
    ulp = 2.0 ** (e - 3)
    err = abs(p - f)
    return {
        "values": values,
        "f": f,
        "p": p,
        "ulp": ulp,
        "ulp_lo": math.floor(f / ulp) * ulp,  # the floats either side of f
        "n_ulp": err / ulp,
        "near": [1.3, 2.3],
        "wide": [0.0, 2.3],
        "tex": {
            "f": "f(x)",
            "p": "p(x)",
            "abs": f"{err:.2f}",
            "rel": f"{err:.2f} \\,/\\, {f:.2f} \\approx {err / f:.2f}",
            "ulp": f"\\approx {err / ulp:.1f} \\text{{ ULP}}",
            "mag": f"|f(x)| \\approx {f:.2f}",
            "gap": f"\\operatorname{{ulp}}(f(x)) = 2^{{{e - 3}}}",
        },
    }


def bench_rows(path, pattern, section=None):
    """Rows of a saved bench table, by method name. Names may hold single
    spaces (std::sin(2pi x)); columns are two or more apart. With section,
    only the block from the line starting with it to the next blank line
    (the f32 and f64 timing tables share method names)."""
    lines = (BENCH_DIR / path).read_text().splitlines()
    if section:
        start = next(i for i, line in enumerate(lines) if line.startswith(section))
        end = next((i for i in range(start, len(lines)) if not lines[i].strip()), len(lines))
        lines = lines[start:end]
    rows = {}
    for line in lines:
        m = re.match(pattern, line)
        if m:
            rows[m["name"]] = m.groupdict()
    return rows


def bench_ulp(domain="radians", fast=False):
    return bench_rows(f"sin-ulp-f32-{domain}{'-fast' if fast else ''}.txt",
                      r"\s+(?P<name>.+?)\s{2,}(?P<max>[\d.e+-]+)\s+(?P<at>[\d.e+-]+)\s+\S+\s+\S+\s+\S+\s+(?P<db>-?[\d.]+)$")


TIMING = (r"\s+(?P<name>.+?)\s{2,}(?:\[\w+\]\s+)?(?P<ns>[\d.]+)\s+(?P<speedup>[\d.]+)x\s+(?P<vec>[\d.]+)\s+"
          r"(?P<vec_x>[\d.]+)x\s+(?P<cold>[\d.]+)\s+(?P<cold_x>[\d.]+)x")


def bench_timing(fn="sin"):
    """f32 timing rows of the saved launches of `adc_bench [fn] timing f32`,
    by method name: each column's median over the launches. ns and speedup
    are per sample, independent (the method on the slides); vec the same
    loop vectorized; cold a 256-sample block after evicting the caches."""
    runs = {}
    for line in (BENCH_DIR / f"{fn}-timing-f32.txt").read_text().splitlines():
        m = re.match(TIMING, line)
        if m:
            runs.setdefault(m["name"], []).append({k: float(v) for k, v in m.groupdict().items() if k != "name"})
    return {name: {k: float(np.median([r[k] for r in rs])) for k in rs[0]} for name, rs in runs.items()}


def loudest_harmonic(fn, ctx):
    """Level (dBc) of the loudest harmonic of one cycle of fn in f32, up to
    but not including Nyquist: turns' rounding noise peaks there (inputs
    are exactly k / 2^16), which isn't a harmonic. Matches the How did we
    do? readouts."""
    dbc = metrics.summarize_spectrum(fn(ctx.phases))["dbc"]
    return float(np.max(dbc[2:-1]))


@item("lut-phase")
def lut_phase():
    """Lookup tables against the polynomial, the oscillator's way: phase in
    (every 32-bit phase, exhaustive), nearest and linear at several sizes,
    beside cheb_phase_sin_13. Max abs error and ULP from the phase ULP
    bench; per sample independent, vectorized, and cold (a 256-sample
    oscillator block after evicting 64 MB) from the timing bench. Size is the table's N + 1 floats."""
    ulp = bench_ulp("phase")
    timing = bench_timing()
    def kb(n):
        b = (n + 1) * 4
        return f"{b / 1024:.0f} KB"
    rows = [("cheb_phase_sin_13", "Polynomial, degree 13", "14 coefficients")]
    for interp in ("nearest", "linear"):
        for n in ((4096, 65536) if interp == "nearest" else (1024, 4096, 65536)):
            rows.append((f"lut_phase_{interp}_{n}", f"{interp.capitalize()}, {n:,}", kb(n)))
    out = []
    for key, label, size in rows:
        u_, t = ulp[key], timing[key]
        out.append({"key": key, "label": label, "size": size, "db": float(u_["db"]),
                    "max_ulp": sig(float(u_["max"]), 3), "ns": sig(t["ns"], 3), "vec": sig(t["vec"], 3),
                    "cold": sig(t["cold"], 3)})
    # The lookup-tables slide quotes the table that matches the polynomial,
    # and both timings.
    by = {r["key"]: r for r in out}
    ns = lambda r: {**r, **{k: f"{r[k]:.2f}" for k in ("ns", "vec", "cold")}}
    return {"rows": out, "match_entries": f"{4096:,}",
            "match": ns(by["lut_phase_linear_4096"]), "poly": ns(by["cheb_phase_sin_13"])}

@item("bench-simple")
def bench_simple():
    """The benchmark on the slides, taken from the source: the bench itself
    (constants, escape, and bench, which times whichever loop it's given)
    and the three inner loops it times (bench/simple/loops.cpp); main from
    bench/simple/main.cpp (retired from the slides); and the cold version's evict and
    bench_cold (bench/simple/cold.cpp)."""
    loops = (ROOT / "bench/simple/loops.cpp").read_text()
    i, j, k = (loops.index("constexpr auto SAMPLES"), loops.index("// Buffer:"),
               loops.index("auto row("))
    bench, loops = "// loops.cpp\n" + loops[i:j].rstrip() + "\n", loops[j:k].rstrip() + "\n"
    assert "        loop();" in bench and loops.count("-> void") == 3
    src = (ROOT / "bench/simple/main.cpp").read_text()
    main = src[src.index("auto main() -> int"):]
    assert main.rstrip().endswith("}")
    cold = (ROOT / "bench/simple/cold.cpp").read_text()
    i, j = cold.index("// Read one word per cache line"), cold.index("auto main() -> int")
    cold = cold[i:j].rstrip() + "\n"
    assert "evict();" in cold
    return {"bench": bench, "main": main, "cold": cold, "loops": loops}


@item("bench-loops")
def bench_loops():
    """Taylor 13's speedup over std::sin in each of the three loops
    (bench/simple/loops.cpp: buffer, sample, feedback), on sorted and on
    random inputs: the ratio of the median ns over the saved launches
    (bench/simple-loops.txt), and the setup under the table: machine and
    compiler from that file's header. Plus what Estrin's scheme takes off
    Taylor's Horner time in the feedback loop (bench/simple-estrin.txt,
    bench/simple/estrin.cpp): the median over its saved launches."""
    ns, head = {}, {}
    order = None
    for line in (BENCH_DIR / "simple-loops.txt").read_text().splitlines():
        if m := re.match(r"(machine|compiler): (.*)", line):
            head[m[1]] = m[2]
        elif m := re.match(r"(Sorted|Random) inputs", line):
            order = m[1].lower()
        elif m := re.match(r"(std::sin|Taylor)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", line):
            for loop, v in zip(("buffer", "sample", "feedback"), m.groups()[1:]):
                ns.setdefault((order, m[1], loop), []).append(float(v))
    med = {k: float(np.median(v)) for k, v in ns.items()}
    launches = len(ns[("sorted", "Taylor", "buffer")])
    assert all(len(v) == launches for v in ns.values())
    chain = [[float(v) for v in m.groups()] for m in re.finditer(
        r"^  chain\s+[\d.]+\s+([\d.]+)\s+([\d.]+)", (BENCH_DIR / "simple-estrin.txt").read_text(), re.M)]
    horner, estrin = np.median(chain, axis=0)
    return {
        "machine": head["machine"],
        "compiler": head["compiler"],
        "estrin_feedback": f"{100 * (1 - estrin / horner):.0f}%",
        "rows": [{
            "key": loop,
            "label": loop.capitalize(),
            **{order: sig(med[(order, "std::sin", loop)] / med[(order, "Taylor", loop)], 3)
               for order in ("sorted", "random")},
        } for loop in ("buffer", "sample", "feedback")],
    }


DEGREE_RESULTS = ROOT / "examples/18-degree-effects/results"


def degree_rows():
    """Example 18's saved results: {(family, degree): row}, timings the median
    over its launches. Degree 0 is the library."""
    rows = {}
    for line in (DEGREE_RESULTS / "errors.txt").read_text().splitlines():
        _, family, d, _, *v = line.split()
        rows[(family, int(d))] = dict(zip(("abs", "rel", "ulp", "at", "abs64", "rel64"), map(float, v)))
    launches = {}
    for line in (DEGREE_RESULTS / "timing.txt").read_text().splitlines():
        _, family, d, _, *v = line.split()
        launches.setdefault((family, int(d)), []).append([float(t) for t in v])
    for key, runs in launches.items():
        rows.setdefault(key, {}).update(zip(("buffer", "sample", "feedback"), np.median(runs, axis=0)))
    return rows


DEGREE_TICKS = [{"v": d, "label": str(d)} for d in (5, 9, 13, 17)]
DEGREE_PLOT = {"xlim": [4.5, 17.5], "xticks": DEGREE_TICKS, "vlines": [13]}
DEGREES = [5, 7, 9, 11, 13, 15, 17]
DEGREE_FORM, DEGREE_LABEL = "sin_split", "split π"  # or "sin_turns", "turns"


@item("degree-sin-abs")
def degree_sin_abs():
    """What a degree buys, absolute error: DEGREE_FORM, every float in f32 against
    the fit alone in f64. The f64 fit error falls without limit; f32 stops."""
    r = degree_rows()
    abs32 = [r[(DEGREE_FORM, d)]["abs"] for d in DEGREES]
    abs64 = [r[(DEGREE_FORM, d)]["abs64"] for d in DEGREES]
    return {
        "plot": {
            **DEGREE_PLOT,
            "x": DEGREES,
            "ylim": [-14, 0],
            "yticks": log_ticks(-14, 0, 4) + [{"v": 0, "label": "1"}],
            "series": [
                {"text": "double", "color": COMPARE, "dashed": True, "y": curve(np.log10(abs64))},
                {"text": "float", "color": plots.SERIES[0], "y": curve(np.log10(abs32))},
            ],
        },
    }


@item("degree-sin-ulp")
def degree_sin_ulp():
    """What a degree buys, max ULP over every float: DEGREE_FORM against the
    plain fit, whose zeros aren't built in, so no degree helps."""
    r = degree_rows()
    form = [r[(DEGREE_FORM, d)]["ulp"] for d in DEGREES]
    plain = [r[("sin_radians", d)]["ulp"] if ("sin_radians", d) in r else np.nan for d in DEGREES]
    return {
        "plot": {
            **DEGREE_PLOT,
            "x": DEGREES,
            "ylim": [0, 12],
            "yticks": log_ticks(0, 12, 4),
            "series": [
                {"text": "plain fit", "color": COMPARE, "gaps": True, "y": curve(np.log10(plain))},
                {"text": DEGREE_LABEL, "color": plots.SERIES[0], "y": curve(np.log10(form))},
            ],
        },
    }


@item("degree-sin-speed")
def degree_sin_speed():
    """What a degree buys, speed: DEGREE_FORM's speedup over std::sin (as on
    the scoreboards) in the three loops."""
    r = degree_rows()
    ref = r[("sin_radians", 0)]
    series = []
    for color, loop in zip(plots.SERIES, ("buffer", "sample", "feedback")):
        ns = np.array([r[(DEGREE_FORM, d)][loop] for d in DEGREES])
        series.append({"text": loop.capitalize(), "color": color, "y": curve(np.log10(ref[loop] / ns))})
    return {
        "plot": {
            **DEGREE_PLOT,
            "x": DEGREES,
            "ylim": [math.log10(0.8), math.log10(16)],
            "yticks": [{"v": math.log10(v), "label": f"{v}×"} for v in (1, 2, 4, 8, 16)],
            "series": series,
        },
    }


@item("sin-scoreboard")
def sin_scoreboard():
    """Max ULP and max abs error over every f32 in [-π, π] and ns per
    sample, independent (and its speedup over std::sin), from the saved
    bench output; the vectorized loop is kept in the data. Loudest harmonic of one cycle in f32
    from Python (std::sin's row is float-rounded sin). One row per method;
    each scoreboard slide picks rows by key. Turns methods are measured
    over every f32 in [0, 1] instead."""
    ulp = {**bench_ulp(), **bench_ulp("turns")}
    rad, turns = harness.make_context("radians"), harness.make_context("turns")
    f32 = np.float32
    table = sinpoly.make_table(1024, f32)
    pe, qe = sinpoly.pade_sin(7, 6)
    spectra = {
        "std::sin(x)": float(rad.ref["worst_dbc"]),
        "taylor_sin_13": loudest_harmonic(lambda x: sinpoly.eval_odd(x, sinpoly.taylor_coeffs(13), f32), rad),
        "pade_sin_7_6": loudest_harmonic(lambda x: sinpoly.eval_pade(x, pe, qe, f32), rad),
        "lut_sin_nearest_1024": loudest_harmonic(lambda x: sinpoly.eval_lut(x, table, "nearest", f32), rad),
        "lut_sin_linear_1024": loudest_harmonic(lambda x: sinpoly.eval_lut(x, table, "linear", f32), rad),
        "cheb_sin_13": loudest_harmonic(sinpoly.method("vanilla", 13, f32).f, rad),
        "cheb_factored_sin_13": loudest_harmonic(sinpoly.method("factored", 13, f32).f, rad),
        "cheb_split_sin_13": loudest_harmonic(sinpoly.method("split", 13, f32).f, rad),
        "cheb_turns_sin_13": loudest_harmonic(sinpoly.method("turns", 13, f32).f, turns),
    }
    timing = bench_timing()
    build = (BENCH_DIR / "sin-ulp-f32-radians.txt").read_text().splitlines()[0]
    methods = [
        ("std::sin(x)", "std::sin"),
        ("taylor_sin_13", "Taylor, degree 13"),
        ("pade_sin_7_6", "Padé [7/6]"),
        ("lut_sin_nearest_1024", "Table, 1024, nearest"),
        ("lut_sin_linear_1024", "Table, 1024, linear"),
        ("cheb_sin_13", "Chebyshev, degree 13"),
        ("cheb_factored_sin_13", "Chebyshev, factored"),
        ("cheb_split_sin_13", "Chebyshev, split π"),
        ("cheb_turns_sin_13", "Chebyshev, turns"),
    ]
    return {
        "build": build[build.index("(") + 1:build.rindex(")")],
        "rows": [{
            "key": name,
            "label": label,
            "max_ulp": sig(float(ulp[name]["max"]), 4),
            "max_abs_db": round(float(ulp[name]["db"]), 1),
            "loudest_dbc": round(spectra[name], 1),
            "ns": sig(timing[name]["ns"], 3),
            "speedup": sig(timing[name]["speedup"], 3),
            "ns_vec": sig(timing[name]["vec"], 3),
            "speedup_vec": sig(timing[name]["vec_x"], 3),
        } for name, label in methods],
    }


# ---------------------------------------------------------------------------
# Chebyshev, worked: 03 vanilla -> 06 turns (projection fits)
# ---------------------------------------------------------------------------

RADIAN_TICKS = [t for t in PI_TICKS if abs(t["v"]) <= PI + 1e-9]
DEGREE = 13  # the sin section's degree for code, error and timing


def sci_label(v):
    """1e-05 -> "1e−5" for tick labels (exact powers of ten), "5e−6" else."""
    m, e = f"{v:.0e}".split("e")
    return f"{m}e{int(e)}".replace("-", "−")


SPECTRUM_KMAX = 21  # both spectra drawn to k = 21, so their slopes compare


def cheb_spectrum(fn, odd=False):
    """|a_k| of fn on [-1, 1], k up to SPECTRUM_KMAX (odd k only for an odd
    fn, whose even terms are 0), as stems on a log axis; the band covers
    the terms down to float resolution 2^-24 (the degree float needs), and
    dashed lines mark float's and double's resolution. The coefficients are
    computed in f64, so they fall until double's 2^-53 and then flatten
    into rounding noise: shown as is, the precision floor made visible."""
    a = fit.series(fn, SPECTRUM_KMAX).coef
    k = np.arange(1 if odd else 0, SPECTRUM_KMAX + 1, 2 if odd else 1)
    floor = 2.0 ** -24
    need = int(max(v for v in k if abs(a[v]) >= floor))
    return need, {
        "x": [int(v) for v in k],
        "xlim": [-1, SPECTRUM_KMAX + 1],
        "ylim": [-18, 1],
        "band": [-1, need + 0.5],
        "xticks": [{"v": v, "label": str(v)} for v in range(1, SPECTRUM_KMAX + 1, 4)],
        "yticks": log_ticks(-16, 0, 4),
        "series": [{"label": "|a_k|", "color": plots.SERIES[0], "stems": True, "r": 5,
                    "y": curve(np.log10(np.abs(a[k]) + 1e-300))},
                   {"label": "\\text{float, } 2^{-24}", "color": plots.TRUTH, "dashed": True,
                    "y": [sig(math.log10(floor), 4)] * len(k)},
                   {"label": "\\text{double, } 2^{-53}", "color": COMPARE, "dashed": True,
                    "y": [sig(-53 * math.log10(2), 4)] * len(k)}],
    }


@item("cheb-fold")
def cheb_fold():
    """Aliasing in the spectrum of sin on [-π, π], for the fold figure:
    interpolating at N = 10 nodes (degree 9, as cheb-proj-9) folds the
    dropped terms a_11, a_13, ... onto a_9, a_7, ... (T_(N+j) = -T_(N-j) at
    the nodes). The stems are the true |a_k|; the images are the first fold
    only (N < k < 2N; the second, from k > 2N, is below 1e-15). Oversampling
    moves the fold line off the chart, leaving the projection.
    Only a_9 moves visibly: the real interpolant's a_9 is 2.3% larger than
    the true one (a_7 moves 1.4e-5 relative, the rest far less), so an
    inset zooms on a_9's tip."""
    N, kmax = 10, 19
    a = cheb_sin_projection(kmax)
    k = np.arange(1, kmax + 1, 2)
    interp = Chebyshev.interpolate(lambda t: np.sin(PI * t), N - 1).coef
    top = N - 1
    change = (abs(interp[top]) - abs(a[top])) / abs(a[top])
    return {
        "N": N,
        "zoom": {"k": top, "from": sig(abs(a[top]), 6), "to": sig(abs(interp[top]), 6),
                 "y": sig(math.log10(abs(a[top])), 4), "text": f"{change:+.1%}"},
        "dx": 0.45,
        "imageColor": plots.SERIES[1],
        "tail": [{"k": int(v), "y": sig(math.log10(abs(a[v])), 4)} for v in k if N < v < 2 * N],
        "plot": {
            "x": [int(v) for v in k],
            "xlim": [-1, kmax + 1],
            "ylim": [-17, 1],
            "band": [-1, N - 0.5],
            "xticks": [{"v": v, "label": str(v)} for v in range(1, kmax + 1, 4)],
            "yticks": log_ticks(-16, 0, 4),
            "series": [{"label": "|a_k|", "color": plots.SERIES[0],
                        "y": curve(np.where(np.abs(a[k]) >= 1e-16, np.log10(np.abs(a[k]) + 1e-300), np.nan))}],
        },
    }


@item("cheb-coeffs")
def cheb_coeffs():
    """|a_k| of sin(π t) on [-1, 1]: the Chebyshev "spectrum" of sin on
    [-π, π]. Odd terms only (the even ones are exactly 0). sin is entire,
    so they fall faster than geometrically, to f64 noise by k ≈ 21; float
    needs degree 13 (the band). The fold (as cheb-fold without the
    merge): interpolating at N = 10 nodes, the dropped terms N < k < 2N
    land beside their mirror images 2N - k; then oversample and truncate
    at the same point, keeping k < N (keep), so the terms kept are the
    ones the fold landed on. No double-precision line: the slide drops it."""
    need, plot = cheb_spectrum(lambda t: np.sin(PI * t), odd=True)
    plot["series"] = plot["series"][:2]
    N = 10
    plot["fold"] = {"N": N, "keep": N - 1, "color": plots.SERIES[1], "dx": 0.45, "text": "folded images",
                    "tail": [{"k": k, "y": y} for k, y in zip(plot["x"], plot["series"][0]["y"])
                             if N < k < 2 * N]}
    return {"degree": need, "plot": plot}


@item("cheb-coeffs-log2")
def cheb_coeffs_log2():
    """The same for log2 on [1, 2]: every term (it isn't odd or even), and a
    steady geometric fall, about 7x per degree (the log's singularity at
    0 sets the rate); float needs degree 8."""
    need, plot = cheb_spectrum(lambda t: np.log2(1.5 + 0.5 * t))
    return {"degree": need, "plot": plot}


@item("cheb-alias")
def cheb_alias():
    """Aliasing at the nodes: interpolating at the 6 zeros of T_6 can't tell
    T_7 from -T_5 (cos 7θ = -cos 5θ when 6θ is an odd multiple of π/2). The
    morph turns one into the other; the curve moves everywhere except at
    the nodes."""
    n = 6
    t = np.linspace(-1, 1, 801)
    nodes = np.cos((np.arange(n) + 0.5) * PI / n)
    T = lambda k, v: np.cos(k * np.arccos(np.clip(v, -1, 1)))
    return {
        "nodes": n,
        "plot": {
            "x": curve(t),
            "xlim": [-1.04, 1.04],
            "ylim": [-1.15, 1.15],
            "xticks": [{"v": v, "label": str(v)} for v in (-1, 0, 1)],
            "yticks": [-1, 0, 1],
            "series": [],
            "color": plots.SERIES[0],
            "steps": [{"y": curve(T(7, t))}, {"y": curve(-T(5, t))}],
            "points": [{"x": sig(v), "y": sig(T(7, v)), "color": plots.SERIES[1]} for v in nodes],
        },
    }


@item("cheb-fit-code")
def cheb_fit_code():
    """fit.series as it is: numpy's Chebyshev.interpolate, and the projection
    as a degree-128 interpolation truncated to the first terms."""
    return {"py": inspect.getsource(fit.series).rstrip() + "\n",
            "tex_oversample": str(fit.PROJECTION_DEGREE)}


def cheb_sin_projection(degree):
    """a_0 .. a_degree of sin on [-π, π] (sin(π t), t on [-1, 1])."""
    return fit.series(lambda t: np.sin(PI * t), degree).coef


@item("cheb-sin-terms")
def cheb_sin_terms():
    """sin on [-π, π] built up from its Chebyshev terms, for the morph figure,
    as taylor-sin builds the Maclaurin polynomials: a_1 T_1, then + a_3 T_3,
    + a_5 T_5, + a_7 T_7 (T_k of x/π; the even a_k are 0). The projection's
    coefficients don't change as terms are added, so each partial sum is
    itself the projection of that degree. Shown on [-2π, 2π] like Taylor,
    the fit interval shaded. Taylor's polynomial of the same degree moves
    with it, faint, for comparison."""
    degrees = [1, 3, 5, 7]
    a = cheb_sin_projection(degrees[-1])
    x = np.linspace(-2 * PI, 2 * PI, 1201)
    tex_terms = [f"{'-' if a[k] < 0 else '+'} {abs(a[k]):#.4g}\\,T_{k}" for k in degrees]
    tex_terms[0] = tex_terms[0].lstrip("+ ")
    cheb, taylor = [], []
    for k in degrees:
        c = np.where(np.arange(len(a)) <= k, a, 0.0)
        cheb.append({"degree": k, "y": curve(np.polynomial.chebyshev.chebval(x / PI, c))})
        taylor.append({"degree": k, "y": curve(sinpoly.eval_odd(x, sinpoly.taylor_coeffs(k), np.float64))})
    return {
        "coeffs": {str(k): sig(a[k], 4) for k in degrees},
        **{f"tex_{k}": f"p_{k}(x) = " + " ".join(tex_terms[:i + 1]) for i, k in enumerate(degrees)},
        "plot": {
            "x": curve(x),
            "xlim": [-2 * PI, 2 * PI],
            "ylim": [-1.6, 1.6],
            "xticks": PI_TICKS,
            "yticks": [-1, 0, 1],
            "band": [-PI, PI],
            "series": [{"label": "\\sin x", "color": plots.TRUTH, "y": curve(np.sin(x))}],
            "curves": [{"text": "Taylor", "color": COMPARE, "steps": taylor},
                       {"text": "Chebyshev", "color": plots.SERIES[0], "steps": cheb}],
        },
    }


@item("cheb-unwrap-f")
def cheb_unwrap_f():
    """sin on [-π, π]: the function before x = π cos θ unwraps it."""
    x = np.linspace(-PI, PI, 601)
    return {
        "plot": {
            "x": curve(x),
            "xlim": [-PI, PI],
            "ylim": [-1.2, 1.2],
            "xticks": [t for t in PI_TICKS if abs(t["v"]) <= PI + 1e-9],
            "yticks": [-1, 0, 1],
            "series": [{"label": "\\sin x", "color": plots.TRUTH, "y": curve(np.sin(x))}],
        },
    }


@item("cheb-unwrap-g")
def cheb_unwrap_g():
    """sin(π cos θ) for θ in [0, 4π]: x = π cos θ sweeps the interval back
    and forth, so f(x) becomes an even, periodic waveform in θ. Its cosine
    series is the Chebyshev series of f."""
    th = np.linspace(0, 4 * PI, 1201)
    labels = ["0", "π", "2π", "3π", "4π"]
    return {
        "plot": {
            "x": curve(th),
            "xlim": [0, 4 * PI],
            "ylim": [-1.2, 1.2],
            "xticks": [{"v": k * PI, "label": lab} for k, lab in enumerate(labels)],
            "yticks": [-1, 0, 1],
            "series": [{"label": "\\sin(\\pi \\cos\\theta)", "color": plots.SERIES[0],
                        "y": curve(np.sin(PI * np.cos(th)))}],
        },
    }


@item("cheb-dct")
def cheb_dct():
    """fit.dct_coeffs as it is (docstring dropped), and how close it comes to numpy's
    Chebyshev.interpolate for sin on [-π, π] at 14 nodes (rounding only)."""
    n = 14
    fn = lambda t: np.sin(PI * t)
    diff = np.abs(fit.dct_coeffs(fn, n) - fit.Chebyshev.interpolate(fn, n - 1).coef).max()
    src = re.sub(r'\n    """.*?"""', "", inspect.getsource(fit.dct_coeffs), flags=re.S)
    return {"py": src.rstrip() + "\n",
            "n": n,
            "tex_diff": sci(max(diff, 1e-300), 2)}


@item("cheb-proj-9")
def cheb_proj_9():
    """Signed error of the vanilla degree-9 fit of sin on [-π, π], by
    interpolation at 10 nodes and by projection, in f64 (the fit alone).
    The aliased a_11 doubles interpolation's error at the ends."""
    degree = 9
    x = np.linspace(-PI, PI, 1201)
    errs = {}
    for m in fit.METHODS:
        fit.set_method(m)
        errs[m] = sinpoly.eval_odd(x, sinpoly.fit_cheb_vanilla(degree), np.float64) - np.sin(x)
    fit.set_method("projection")
    lim = 1.3e-5
    ticks = [-1e-5, -5e-6, 0, 5e-6, 1e-5]
    a11 = abs(sinpoly.cheb_vanilla_series(degree + 2)[degree + 2])
    return {
        "degree": degree,
        "tex_interp": sci(np.abs(errs["interpolation"]).max()),
        "tex_proj": sci(np.abs(errs["projection"]).max()),
        "tex_a11": sci(a11),
        "plot": {
            "x": curve(x),
            "xlim": [-PI, PI],
            "ylim": [-lim, lim],
            "xticks": RADIAN_TICKS,
            "yticks": [{"v": v, "label": "0" if v == 0 else sci_label(v)} for v in ticks],
            "series": [
                {"text": "Interpolation, 10 nodes", "color": plots.SERIES[1], "y": curve(errs["interpolation"])},
                {"text": "Projection", "color": plots.SERIES[0], "y": curve(errs["projection"])},
            ],
        },
    }


def error_steps(fn, ctx, keys=("abs", "rel")):
    """log10 envelopes of fn's f32 errors on ctx, one morph step per key."""
    err = metrics.pointwise_errors(fn(ctx.x), ctx.truth)
    out = []
    for k in keys:
        xb, y = envelope(ctx.x.astype(np.float64), err[k])
        out.append(y)
    return xb, out, err


@item("cheb-sin-13")
def cheb_sin_13():
    """Example 03 at degree 13: the C++ the bench builds; the two steps
    that get there (fit on [-π, π], convert) as short snippets, executed
    here and checked against sinpoly's coefficients; and the error on
    [-π, π] in f32, morphing absolute -> relative like taylor-sin-13."""
    name, code = cppgen.emit_cheb_vanilla(DEGREE)
    ctx = harness.make_context("radians")
    c = sinpoly.fit_cheb_vanilla(DEGREE)
    xb, (abs_log, rel_log), err = error_steps(lambda x: sinpoly.eval_odd(x, c, np.float32), ctx)
    max_abs, max_rel = err["abs"].max(), np.nanmax(err["rel"])
    ulp = bench_ulp()
    # The implementation slide's two steps, as snippets; run here and
    # checked against the example's own coefficients. The domain does the
    # remap (the series is in t = x/π); convert undoes it.
    steps = {
        "py_fit": ("n = 13\n"
                   "p = Chebyshev.interpolate(np.sin, 128, domain=[-np.pi, np.pi])\n"
                   "p = p.truncate(n + 1)   # keep a_0 ... a_n\n"),
        "py_convert": "c = p.convert(kind=Polynomial).coef   # powers of x, on [-π, π]\n",
    }
    scope = {"np": np, "Chebyshev": Chebyshev, "Polynomial": np.polynomial.Polynomial}
    exec("".join(steps.values()), scope)
    assert np.allclose(scope["c"][1::2], c, rtol=1e-12, atol=1e-18), "implementation snippets drifted"
    return {
        "name": name,
        "cpp": code.rstrip() + "\n",
        **steps,
        "tex_abs": f"\\max = {sci(max_abs)} \;\; ({20 * np.log10(max_abs):.0f}\\,\\text{{dB}})",
        "tex_rel": f"\\max = {max_rel:.2g}",
        "tex_ulp": sci(float(ulp[name]["max"])),
        "plot": {
            "x": curve(xb),
            "xlim": [-PI, PI],
            "ylim": [-10, 3],
            "xticks": RADIAN_TICKS,
            "yticks": log_ticks(-10, 2, 2),
            "series": [],
            "color": plots.SERIES[0],
            "steps": [{"y": curve(abs_log)}, {"y": curve(rel_log)}],
        },
    }


@item("cheb-factored-13")
def cheb_factored_13():
    """Example 04: divide sin by its zeros, x (x - π)(x + π), and fit the
    smooth, even quotient. The quotient's plot, and the C++."""
    name, code = cppgen.emit_cheb_factored(DEGREE)
    x = np.linspace(-PI, PI, 801)
    q = sinpoly.factored_quotient(x[1:-1])
    # The ends and the middle are removable: fill them from just inside.
    q = np.concatenate([[q[0]], q, [q[-1]]])
    return {
        "name": name,
        "cpp": code.rstrip() + "\n",
        "terms": (DEGREE - 1) // 2,
        "plot": {
            "x": curve(x),
            "xlim": [-PI, PI],
            "ylim": [-0.11, 0.0],
            "xticks": RADIAN_TICKS,
            "yticks": [{"v": v, "label": f"{v:g}".replace("-", "−")} for v in (-0.1, -0.05, 0)],
            "series": [{"label": "\\sin x \\,/\\, x\\,(x - \\pi)(x + \\pi)", "color": plots.SERIES[0],
                        "y": curve(q)}],
        },
    }


@item("sin-factor")
def sin_factor():
    """What the polynomial has to approximate on [-π, π], step by step
    (morph, like tan-factor): sin x itself; sin x / x once the zero at 0 is
    factored out (still zero at ±π); sin x / x (x - π)(x + π) once those are
    too (sinpoly.factored_quotient: smooth, even, -1/π² to -1/2π², never
    near zero). The last step zooms the y axis in on it."""
    x = np.linspace(-PI, PI, 801)
    q = sinpoly.factored_quotient(x[1:-1])
    # The ends are removable: fill them from just inside.
    q = np.concatenate([[q[0]], q, [q[-1]]])
    unit = {"ylim": [-1.1, 1.1], "yticks": [-1, 0, 1]}
    return {
        "plot": {
            "x": curve(x),
            "xlim": [-PI, PI],
            "ylim": unit["ylim"],
            "xticks": RADIAN_TICKS,
            "yticks": unit["yticks"],
            "stepYlims": [unit, unit, {"ylim": [-0.12, 0.02],
                                       "yticks": [{"v": v, "label": f"{v:g}"} for v in (-0.1, -0.05, 0)]}],
            "series": [],
            "color": plots.SERIES[0],
            "steps": [{"y": curve(y)} for y in (np.sin(x), np.sinc(x / PI), q)],
        },
    }


@item("cheb-near-pi")
def cheb_near_pi():
    """The zoom figure: all of sin x on [-π, π], zooming into its zero at π
    down to the last few floats, where f32(π) sits 0.37 of a step past π.
    There, each step's f32 output as dots: vanilla misses everywhere;
    factored puts its zero on f32(π), not π; split π lands on sin. Each
    step's polynomial is also drawn as a curve, evaluated exactly (f64)
    but with the code's own π constants, so the factored curve crosses
    zero at f32(π) and the split one at π.
    x is stored as d = x - π so the deep end keeps its digits; sin is
    sampled log-spaced toward π so every zoom level has points."""
    gap = float(np.spacing(np.float32(2.0)))  # f32 spacing in [2, 4)
    xs = np.float32(PI) + np.arange(-6, 1).astype(np.float32) * np.float32(gap)
    reach = 2 * PI + 0.4
    d = np.unique(np.concatenate([
        -np.logspace(-9, np.log10(reach), 700), np.logspace(-9, np.log10(0.7), 300), [0.0],
        np.linspace(-reach, 0.7, 600)]))
    p_van = sinpoly.fit_cheb_vanilla(DEGREE)
    p_fac = sinpoly.fit_cheb_factored(DEGREE)
    steps = [sinpoly.eval_odd(xs, p_van, np.float32), sinpoly.eval_factored(xs, p_fac, np.float32),
             sinpoly.eval_split(xs, p_fac, np.float32)]
    ulp = bench_ulp()
    hi, lo = sinpoly.split_pi(np.float32)
    x = PI + d
    P = lambda c: sinpoly.horner(c, x * x)
    pi32, hi64, lo64 = float(np.float32(PI)), float(hi), float(lo)
    approx = [x * P(p_van),
              (x - pi32) * (x + pi32) * P(p_fac) * x,
              ((x - hi64) - lo64) * ((x + hi64) + lo64) * P(p_fac) * x]
    return {
        "tex_pi_hi": f"\\pi_{{hi}} = {float(hi):.9g}",
        "tex_pi_lo": f"\\pi_{{lo}} = {sci(float(lo), 3)}",
        "tex_ulp_vanilla": sci(float(ulp["cheb_sin_13"]["max"])),
        "tex_ulp_factored": sci(float(ulp["cheb_factored_sin_13"]["max"])),
        "tex_ulp_split": f"{float(ulp['cheb_split_sin_13']['max']):.2f}",
        "tex_ulp_split_fast": sci(float(bench_ulp(fast=True)["cheb_split_sin_13"]["max"])),
        "sin": {"d": [sig(v, 6) for v in d], "y": curve(-np.sin(d))},  # sin(π + d) = -sin d
        "approx": [{"y": curve(y)} for y in approx],
        "approxColor": plots.SERIES[0],
        "gap": sig(gap, 10),
        "f32_off": sig(float(np.float32(PI)) - PI, 10),
        "floats": [sig(v, 10) for v in xs.astype(np.float64) - PI],
        "full": {"x": [-2 * PI - 0.3, 0.6], "y": [-1.2, 1.2]},
        "near": {"x": [-6.2 * gap, 0.9 * gap], "y": [-3e-7, 1.9e-6]},
        "box": {"x": [-0.45, 0.45], "y": [-0.35, 0.35]},
        "color": plots.SERIES[1],
        "steps": [{"y": [sig(float(v), 6) for v in y]} for y in steps],
    }


@item("split-pi-line")
def split_pi_line():
    """Split π on two number lines at the same scale. Top: the floats near
    π, π between two of them, π_hi = f32(π) the one above, and x the float
    just below π. Bottom: near 0, where x − π_hi lands exactly (Sterbenz)
    and the floats are so dense the line looks solid; π lands at π − π_hi,
    which π_lo carries; subtracting it gives x − π. Positions are offsets
    from π in f64; the JS scales them by the gap."""
    hi, lo = sinpoly.split_pi(np.float32)
    x = np.nextafter(hi, np.float32(0))  # the float just below π
    gap = float(hi) - float(x)
    computed = x - hi  # f32
    assert float(computed) == float(x) - float(hi)  # Sterbenz: exact
    result = computed - lo  # f32
    true = float(x) - PI
    log2_ulp = lambda v: int(np.log2(float(np.spacing(np.float32(abs(v))))))
    return {
        "gap": sig(gap, 10),
        "hi": sig(float(hi) - PI, 10),
        "x": sig(true, 10),
        "lo": sig(float(lo), 10),
        "computed": sig(float(computed), 10),
        "result": sig(float(result), 10),
        "tex_ulp_top": f"2^{{{log2_ulp(PI)}}}",
        "tex_ulp_bottom": f"\\le 2^{{{log2_ulp(computed)}}}",
        "tex_computed": f"{sci(float(computed), 4)}",
        "tex_off": f"{abs(float(computed) - true) / abs(true) * 100:.0f}\\%",
        "tex_err": sci(abs(float(result) - true)),
    }


@item("cheb-turns-13")
def cheb_turns_13():
    """Example 06: sin(2π x) in turns, x in [0, 1]. The C++ (without the
    doc line on the zeros: the slide says it) and its exhaustive max ULP."""
    name, code = cppgen.emit_cheb_turns(DEGREE)
    lines = [ln for ln in code.rstrip().splitlines() if not ln.startswith("/// Zeros")]
    assert len(lines) == len(code.rstrip().splitlines()) - 1
    ulp = bench_ulp("turns")
    return {
        "name": name,
        "cpp": "\n".join(lines) + "\n",
        "tex_ulp": f"{float(ulp[name]['max']):.2f}",
        "tex_ulp_split": f"{float(bench_ulp()['cheb_split_sin_13']['max']):.2f}",
    }


@item("cheb-ulp-steps")
def cheb_ulp_steps():
    """Every step's ULP error over one period, in f32, as morph steps:
    vanilla, factored, split π, turns. Turns inputs x in [0, 1] are drawn at
    the same angle, 2π x wrapped to [-π, π], so the four line up. Sampled:
    the dips and spikes are real, the maxima are lower bounds (the bench's
    exhaustive numbers are on the slide)."""
    rad, turns = harness.make_context("radians"), harness.make_context("turns")
    p_fac = sinpoly.fit_cheb_factored(DEGREE)
    runs = [
        (rad, lambda x: sinpoly.eval_odd(x, sinpoly.fit_cheb_vanilla(DEGREE), np.float32)),
        (rad, lambda x: sinpoly.eval_factored(x, p_fac, np.float32)),
        (rad, lambda x: sinpoly.eval_split(x, p_fac, np.float32)),
        (turns, lambda x: sinpoly.eval_turns(x, sinpoly.fit_cheb_turns(DEGREE), np.float32)),
    ]
    steps = []
    for ctx, fn in runs:
        ulp = metrics.pointwise_errors(fn(ctx.x), ctx.truth)["ulp"]
        theta = ctx.x.astype(np.float64)
        if ctx.domain == "turns":
            theta = 2 * PI * np.where(theta > 0.5, theta - 1, theta)
        order = np.argsort(theta, kind="stable")
        xb, y = envelope(theta[order], ulp[order], bins=400)
        steps.append({"y": curve(y)})
    ulp_r, ulp_t = bench_ulp(), bench_ulp("turns")
    big = lambda v: sci(float(v["max"]))
    small = lambda v: f"{float(v['max']):.2f}"
    return {
        "tex_vanilla": big(ulp_r["cheb_sin_13"]),
        "tex_factored": big(ulp_r["cheb_factored_sin_13"]),
        "tex_split": small(ulp_r["cheb_split_sin_13"]),
        "tex_turns": small(ulp_t["cheb_turns_sin_13"]),
        "plot": {
            "x": curve(xb),
            "xlim": [-PI, PI],
            "ylim": [-2, 8],
            "xticks": RADIAN_TICKS,
            "yticks": log_ticks(-2, 8, 2),
            "series": [],
            "color": plots.SERIES[0],
            "steps": steps,
        },
    }


@item("sin-input-trap")
def sin_input_trap():
    """Turns in, three ways, max ULP over every input (bench): std::sin on
    2π x, the turns polynomial on float(phase) / 2^32, and the polynomial
    with each factor taken from the 32-bit phase."""
    t, ph = bench_ulp("turns"), bench_ulp("phase")
    rows = [
        ("std::sin(2pi x)", "std::sin(2 * pi * x)", t),
        ("cheb_turns_sin_13", "cheb_turns_sin_13(x)", t),
        ("std::sin(2pi phase/2^32)", "std::sin(2 * pi * phase / 0x1p32)", ph),
        ("cheb_turns_phase_sin_13", "cheb_turns_sin_13(phase * 0x1p-32f)", ph),
        ("cheb_phase_sin_13", "cheb_phase_sin_13(phase)", ph),
    ]
    return {"rows": [{"key": k, "label": label, "max_ulp": sig(float(src[k]["max"]), 4),
                      "at": src[k]["at"]} for k, label, src in rows]}


# ---------------------------------------------------------------------------
# tan(πx), exp2 / log2, tanh: one lesson each
# ---------------------------------------------------------------------------

FS = 48_000


def code_body(code):
    """Generated C++ without its /// doc lines, for a code panel beside
    maths that already says it."""
    return "\n".join(line for line in code.rstrip().splitlines() if not line.startswith("///")) + "\n"
def unary_timing(fn):
    """f32 timing rows of `adc_bench <fn> timing f32`, median of the launches."""
    return bench_timing(fn)


def timing_cols(row):
    return {"ns": sig(row["ns"], 3), "speedup": sig(row["speedup"], 3)}


def khz_ticks(lo=0.0):
    """fc/fs ticks labelled in kHz at 48 kHz."""
    return [{"v": v, "label": "0" if v == 0 else f"{v * FS / 1000:g} kHz"}
            for v in (0, 0.125, 0.25, 0.375, 0.5) if v >= lo]


HALF_PI_TICKS = [{"v": k * PI / 2, "label": lab} for k, lab in
                 [(-2, "−π"), (-1, "−π/2"), (0, "0"), (1, "π/2"), (2, "π")]]


@item("tan-x")
def tan_x():
    """tan x on [-π, π]: the zero at 0, the poles at ±π/2 (dashed), a marker
    at 20 kHz for fs = 48 kHz (a click), one branch per period; the samples nearest each pole are left out so the
    curve breaks there instead of jumping across."""
    x = np.linspace(-PI, PI, 1201)
    y = np.tan(x)
    near_pole = np.abs(np.abs(x) - PI / 2) < 0.01
    return {
        "plot": {
            "x": curve(x),
            "xlim": [-PI, PI],
            "ylim": [-4, 4],
            "xticks": HALF_PI_TICKS,
            "yticks": [-4, -2, 0, 2, 4],
            "vlines": [sig(-PI / 2), sig(PI / 2)],
            # 20 kHz at 48 kHz: x = π fc / fs, the argument g = tan(x) takes
            "markers": [{"x": sig(PI * 20_000 / FS), "label": ["20 kHz", "at 48 kHz"]}],
            "series": [{"label": "\\tan x", "color": plots.TRUTH, "gaps": True,
                        "y": [None if p else v for p, v in zip(near_pole, curve(y))]}],
        },
    }


@item("tan-factor")
def tan_factor():
    """What the polynomial has to approximate on (-π/2, π/2), step by step
    (morph): tan x itself; tan x / x once the zero is factored out (still
    the poles); tan x (π²/4 - x²) / x once the poles are divided out
    (tanpoly.radians_quotient: smooth, 2 to π²/4); then normalized to
    turns, Q (tanpoly.turns_quotient: the same shape over π), with the x
    labels switching to ±½. Values are clamped well outside the plot so
    the morph has numbers to blend."""
    x = np.linspace(-PI / 2, PI / 2, 801)[1:-1]
    steps = [np.tan(x), np.sinc(x / PI) / np.cos(x), tanpoly.radians_quotient(x), tanpoly.turns_quotient(x / PI)]
    radians = [t for t in HALF_PI_TICKS if abs(t["v"]) <= PI / 2 + 1e-9]
    turns = [t["label"].replace("π/2", "½").replace("π", "1") for t in radians]
    return {
        "plot": {
            "x": curve(x),
            "xlim": [-PI / 2, PI / 2],
            "ylim": [-4, 6],
            "xticks": radians,
            "stepXlabels": [[t["label"] for t in radians]] * 3 + [turns],
            "yticks": [-4, -2, 0, 2, 4, 6],
            "vlines": [sig(-PI / 2), sig(PI / 2)],
            "series": [],
            "color": plots.SERIES[0],
            "steps": [{"y": curve(np.clip(y, -40, 40))} for y in steps],
        },
    }


@item("tan-pi")
def tan_pi():
    """The prewarp g = tan(π fc/fs) on [0, ½), against g = πx (no prewarp):
    the pole at Nyquist."""
    x = np.linspace(0, 0.4995, 800)
    return {
        "plot": {
            "x": curve(x),
            "xlim": [0, 0.5],
            "ylim": [0, 6],
            "xticks": khz_ticks(),
            "yticks": [0, 2, 4, 6],
            "series": [
                {"label": "\\tan \\pi x", "color": plots.TRUTH, "y": curve(metrics.tan_pi(x))},
                {"text": "No prewarp, g = πx", "color": plots.SERIES[1], "dashed": True, "y": curve(PI * x)},
            ],
        },
    }


@item("tan-pade")
def tan_pade():
    """Padé approximants of tan have poles of their own: where each lands,
    as fc/fs and as Hz past Nyquist at 48 kHz. [3/2] written out in integers."""
    pe, qe = tanpoly.pade_tan(3, 2)
    scale = math.lcm(*[f.denominator for f in list(pe) + list(qe)])
    num = tex_int_poly([f * scale for f in pe], "t^{2}")
    den = tex_int_poly([f * scale for f in qe], "t^{2}")
    poles = {}
    for m, n in [(3, 2), (5, 4), (7, 6)]:
        x = tanpoly.pade_pole(m, n) / PI
        poles[f"tex_{m}_{n}"] = (f"[{m}/{n}]: \; {x:.10g} \\quad (\\text{{Nyquist}} + "
                                 f"{sci((x - 0.5) * FS, 2) if (x - 0.5) * FS < 0.01 else f'{(x - 0.5) * FS:.3g}'}\\,\\text{{Hz}})")
    return {"tex": f"\\tan t \\approx \\frac{{t\\,({num})}}{{{den}}}", **poles}


def tan_turns(degree):
    """09 at `degree`: the turns form's C++, its max ULP to Nyquist and
    cents up to 20 kHz (bench, exhaustive), and the radians forms' max ULP
    to Nyquist from example 09's input set (sampled: a lower bound)."""
    name, code = cppgen.emit_tan_turns(degree)
    ctx = tan_example.make_context(FS)
    radians = {s: tan_example.evaluate(s, tanpoly.method(s, degree, np.float32), ctx).stats["ulp_all"]
               for s in ("radians", "split")}
    err = bench_rows("tan-f32.txt", r"\s+(?P<name>\S+)\s{2,}(?P<audio>[\d.e+-]+)\s+(?P<nyq>[\d.e+-]+)\s+\S+"
                                    r"\s+(?P<cents>\S+)\s+\S+$")
    nyq = float(err[name]["nyq"])
    # The implementation slide's Python: the quotient handed to
    # Chebyshev.interpolate, then the powers of x². tan(πx)/x is 0/0 at
    # x = 0, where a node lands; π sinc(x) / cos(πx) is the same function,
    # safe there (the slide's bullet says so). Run here and checked against
    # the example's own coefficients.
    py = ("q = lambda x: (0.5 - x) * (0.5 + x) * np.pi * np.sinc(x) / np.cos(np.pi * x)\n"
          "Q = Chebyshev.interpolate(q, 128, domain=[-0.5, 0.5])\n"
          f"Q = Q.truncate({degree})\n"
          "c = Q.convert(kind=Polynomial).coef[0::2]\n")
    scope = {"np": np, "Chebyshev": Chebyshev, "Polynomial": np.polynomial.Polynomial}
    exec(py, scope)
    assert np.allclose(scope["c"], tanpoly.fit_cheb_turns(degree), rtol=1e-7, atol=1e-9), "tan snippet drifted"  # same function; rounds differently, ~2e-9
    return {
        "name": name,
        "py": py,
        "cpp": code_body(code),
        "tex_radians": sci(radians["radians"]),
        "tex_split": sci(radians["split"]),
        "tex_turns": f"{nyq:.2f}" if nyq < 10 else f"{nyq:.0f}",
        "tex_cents": f"{float(err[name]['cents']):.2g}",
    }


@item("tan-turns-9")
def tan_turns_9():
    """Turns at degree 9, the implementation slide: see tan_turns."""
    return tan_turns(9)


@item("tan-turns-5")
def tan_turns_5():
    """Turns at degree 5, the bonus after the scoreboard: see tan_turns."""
    return tan_turns(5)


@item("tan-scoreboard")
def tan_scoreboard():
    """Max ULP up to Nyquist and max cutoff error in cents (≤ 20 kHz and
    above), every normal f32 fc/fs below ½, from the bench; buffer timing."""
    err = bench_rows("tan-f32.txt", r"\s+(?P<name>.+?)\s{2,}(?P<audio>[\d.e+-]+)\s+(?P<nyq>[\d.e+-]+)\s+\S+\s+(?P<ca>[\d.e+-]+)\s+(?P<ct>[\d.e+-]+)$")
    timing = unary_timing("tan")
    rows = []
    for name, label in [("std::tan(pi x)", "std::tan(π·x)"), ("pade_tan_pi_3_2", "Padé [3/2]"),
                        ("pade_tan_pi_7_6", "Padé [7/6]"), ("cheb_turns_tan_pi_3", "Turns, degree 3"),
                        ("cheb_turns_tan_pi_5", "Turns, degree 5"), ("cheb_turns_tan_pi_9", "Turns, degree 9")]:
        e = err[name]
        rows.append({"key": name, "label": label, "max_ulp": sig(float(e["nyq"]), 3),
                     "cents_audio": sig(float(e["ca"]), 2), "cents_top": sig(float(e["ct"]), 2),
                     **timing_cols(timing[name])})
    worst = lambda r: max(r["cents_audio"], r["cents_top"])
    by = {r["key"]: r for r in rows}
    small = lambda v: sci(v) if v < 0.01 else f"{v:g}"
    return {
        "rows": rows,
        "tex_pade76_ulp": sci(by["pade_tan_pi_7_6"]["max_ulp"]),
        "tex_pade76_cents": small(worst(by["pade_tan_pi_7_6"])),
        "tex_pade32_cents": small(worst(by["pade_tan_pi_3_2"])),
        "tex_turns3_cents": small(worst(by["cheb_turns_tan_pi_3"])),
    }


@item("log2-exp2")
def log2_exp2():
    """log₂ x and its inverse 2^x on one square-ish frame, with y = x dashed:
    log₂ x is 0 at 1 and climbs 1 per octave; 2^x is its mirror image."""
    x = np.linspace(-3, 4, 1401)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(x > 0.005, np.log2(np.maximum(x, 1e-300)), np.nan)
    ticks = [{"v": v, "label": str(v).replace("-", "−")} for v in range(-2, 5)]
    return {
        "plot": {
            "x": curve(x),
            "xlim": [-3, 4],
            "ylim": [-3, 4],
            "xticks": ticks,
            "yticks": ticks,
            "series": [
                {"label": "\\log_2 x", "color": plots.SERIES[0], "y": curve(lg)},
                {"label": "2^x", "color": plots.SERIES[1], "y": curve(np.exp2(x))},
                {"label": "y = x", "color": plots.TRUTH, "dashed": True, "y": curve(x)},
            ],
        },
    }


@item("log2-x")
def log2_x():
    """log2 x alone on (0, 16]: no symmetry, no period, and an input domain
    and output range as wide as the floats."""
    x = np.linspace(0, 16, 1601)
    with np.errstate(divide="ignore"):
        lg = np.where(x > 0.01, np.log2(np.maximum(x, 1e-300)), np.nan)
    return {
        "plot": {
            "x": curve(x),
            "xlim": [0, 16],
            "ylim": [-4, 4.5],
            "xticks": [{"v": v, "label": str(v)} for v in (0, 1, 2, 4, 8, 16)],
            "yticks": [{"v": v, "label": str(v).replace("-", "−")} for v in range(-3, 5)],
            "series": [{"label": "\\log_2 x", "color": plots.SERIES[0], "y": curve(lg)}],
        },
    }

@item("float-440")
def float_440():
    """440 in single precision, for the log2 section: 2^8 × 1.71875, exact.
    Its bits and fields (as float-anatomy's steps), and the chord: the bits
    read as a number, / 2^23 - 127, give e + (m - 1) against log2 440."""
    step = float_step("440", 440.0)
    e, m = 8, 440.0 / 2**8
    assert step["pos"] == e + (m - 1)
    return {
        **step,
        "tex_log": f"\\log_2 440 = {math.log2(440):.4f}\\ldots",
        "tex_chord": f"e + (m - 1) = {e + (m - 1):g}",
    }


@item("log2-octave")
def log2_octave():
    """log2 x on (0, 8], one octave [1, 2] shaded and drawn in blue over it:
    every other octave is that piece, stretched by 2^e and lifted by e. Then
    what the polynomial fits once the zero at m = 1 is built in: P(u) =
    log2 m / u, u = m - 1, smooth, from 1/ln 2 down to 1, nowhere near 0.
    And the fit of P as a snippet (the lambda: P is the quotient), run here
    and checked against log2poly's coefficients. The nodes are
    interior, so u = 0 is never evaluated. log1p, not log2(1 + u): 1 + u
    rounds, and near u = 0 the quotient loses ~1e-12, 1.5e-8 in the
    monomial coefficients."""
    py_fit = ("f = lambda u: np.log1p(u) / (u * np.log(2))\n"
              "P = Chebyshev.interpolate(f, 128, domain=[0, 1])\n"
              f"P = P.truncate({LOG2_DEGREE})   # u P(u): degree {LOG2_DEGREE}\n")
    scope = {"np": np, "Chebyshev": Chebyshev}
    exec(py_fit, scope)
    c = scope["P"].convert(kind=np.polynomial.Polynomial, domain=[-1, 1], window=[-1, 1]).coef
    assert np.allclose(c, log2poly.fit_mantissa(LOG2_DEGREE), rtol=1e-12, atol=1e-15), "log2 fit snippet drifted"
    x = np.linspace(0, 8, 1601)
    with np.errstate(divide="ignore"):
        lg = np.where(x > 0.01, np.log2(np.maximum(x, 1e-300)), np.nan)
    piece = np.where((x >= 1) & (x <= 2), lg, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = np.where(x > 1, lg / (x - 1), 1 / math.log(2))  # P(u) = log2(1 + u) / u
    quotient = np.where((x >= 1) & (x <= 2), quotient, np.nan)
    return {
        "py_fit": py_fit,
        "plot": {
            "x": curve(x),
            "xlim": [0, 8],
            "ylim": [-3, 3.5],
            "xticks": [{"v": v, "label": str(v)} for v in (0, 1, 2, 4, 8)],
            "yticks": [{"v": v, "label": str(v).replace("-", "−")} for v in range(-2, 4)],
            "band": [1, 2],
            "series": [
                {"label": "\\log_2 x", "color": plots.TRUTH, "y": curve(lg)},
                {"label": "\\log_2 m, \\; m \\in [1, 2)", "color": plots.SERIES[0], "gaps": True, "y": curve(piece)},
                {"label": "P(u) = \\log_2 m \\,/\\, u", "color": plots.SERIES[1], "gaps": True, "y": curve(quotient)},
            ],
        },
    }


@item("split-exponent")
def split_exponent_code():
    """The generated bit helpers (cppgen.bit_helpers) for the slide: first
    split_exponent for float alone, everything written out; then the
    ieee<T> traits for float and double, and split_exponent written once
    over them with ieee_int, the signed integer as wide as T (doc lines
    stripped): e from the exponent field, m with its exponent field set to
    2^0, so m in [1, 2)."""
    code = cppgen.bit_helpers()
    def block(start, end):
        i = code.index(start)
        return code[i:code.index(end, i) + len(end)]
    traits = block("template<>\nstruct ieee<float>", "\n};") + "\n\n" + block("template<>\nstruct ieee<double>", "\n};")
    split = (block("template<typename T>\nusing ieee_int", ";") + "\n\n"
             + block("template<typename T>\ninline auto split_exponent", "\n}"))
    L = bits.layout(np.float32)
    exp_bits = int(L.exp_mask).bit_length()
    one = (L.bias << L.sig_bits)
    assert one == int(np.float32(1).view(np.uint32))
    constants = [
        {"code": f"{L.sig_bits}", "tex": f"\\text{{fraction bits stored (one more is implicit)}}"},
        {"code": f"0x{L.exp_mask:X}u", "tex": f"2^{{{exp_bits}}} - 1\\text{{: the {exp_bits} exponent bits}}"},
        {"code": f"0x{(1 << L.sig_bits) - 1:X}u", "tex": f"2^{{{L.sig_bits}}} - 1\\text{{: the {L.sig_bits} fraction bits}}"},
        {"code": f"{L.bias}", "tex": f"\\text{{the bias: }} e = \\text{{field}} - {L.bias}"},
        {"code": f"{L.bias}u << {L.sig_bits}", "tex": f"\\text{{the exponent field of }} 2^0 \\text{{: }} \\texttt{{1.0f}} = \\texttt{{0x{one:08X}}}"},
    ]
    return {"float": cppgen.split_exponent_float(), "constants": constants,
            "traits": traits + "\n", "split": split + "\n"}


@item("bits-chords")
def bits_chords():
    """Schraudolph's 2^x and Mitchell's log2 x, offsets tuned, in f32: their
    error in cents over four octaves. Both are a chord per octave."""
    u = np.linspace(-2, 2, 1601)
    x = u.astype(np.float32)
    up = exp2poly.eval_bits(x, np.float32, exp2poly.bits_offset()).astype(np.float64)
    exp_cents = 1200 * np.log2(up / np.exp2(x.astype(np.float64)))
    xs = np.exp2(u).astype(np.float32)
    lg = log2poly.eval_bits(xs, np.float32, log2poly.bits_offset()).astype(np.float64)
    log_cents = 1200 * (lg - np.log2(xs.astype(np.float64)))
    return {
        "tex_max": f"{max(np.abs(exp_cents).max(), np.abs(log_cents).max()):.0f}",
        "plot": {
            "x": curve(u),
            "xlim": [-2, 2],
            "ylim": [-70, 70],
            "xticks": [{"v": v, "label": str(v).replace("-", "−")} for v in (-2, -1, 0, 1, 2)],
            "yticks": [{"v": v, "label": f"{v:+d}".replace("-", "−").replace("+0", "0")} for v in (-50, 0, 50)],
            "series": [
                {"label": "2^x \\text{ (Schraudolph)}", "color": plots.SERIES[0], "y": curve(exp_cents)},
                {"label": "\\log_2 x \\text{ (Mitchell)}", "color": plots.SERIES[1], "y": curve(log_cents)},
            ],
        },
    }


@item("bits-code")
def bits_code():
    """The two bit tricks as the bench builds them."""
    return {"exp2": cppgen.emit_exp2_bits(True)[1].rstrip() + "\n",
            "log2": cppgen.emit_log2_bits(True)[1].rstrip() + "\n"}


# The unary benches' error table, every column: max ULP (at x), max rel, max abs (at x), cents, dB.
UNARY_ERR_FULL = (r"\s+(?P<name>.+?)\s{2,}(?P<ulp>[\d.e+-]+)\s+\S+\s+(?P<rel>[\d.e+-]+)\s+(?P<abs>[\d.e+-]+)"
                  r"\s+\S+\s+(?P<cents>[\d.e+-]+)\s+(?P<db>[\d.e+-]+)$")
# The same, with the absolute columns possibly "—" (exp2, where they mean nothing).
UNARY_ERR_ROW = (r"\s+(?P<name>.+?)\s{2,}(?P<ulp>[\d.e+-]+)\s+\S+\s+(?P<rel>[\d.e+-]+)\s+(?:[\d.e+-]+\s+\S+|—)"
                 r"\s+(?P<cents>[\d.e+-]+)\s+(?P<db>[\d.e+-]+)$")
UNARY_ERR = r"\s+(?P<name>.+?)\s{2,}(?P<max>[\d.e+-]+)\s+.*\s(?P<cents>[\d.e+-]+)\s+(?P<db>[\d.e+-]+)$"


LOG2_DEGREE = 7   # u P(u) on the slides
EXP2_DEGREE = 5   # 1 + f Q(f) on the slides
EXP2_NAME = f"round_unit_exp2_{EXP2_DEGREE}"


@item("exp2-cpp")
def exp2_cpp():
    """10 at the slides' degree, 1 + f Q(f): the C++ and its exhaustive max ULP."""
    name, code = cppgen.emit_exp2_unit(EXP2_DEGREE)
    err = bench_rows("exp2-f32.txt", UNARY_ERR, section="Error")
    return {"name": name, "cpp": code_body(code), "tex_ulp": f"{float(err[name]['max']):.2f}"}


@item("exp2-pow2i")
def exp2_pow2i():
    """The generic pow2i for the slide, from cppgen.bit_helpers (doc lines
    stripped), beside the 2^x it serves."""
    code = cppgen.bit_helpers()
    i = code.index("template<typename T>\ninline auto pow2i")
    return {"generic": code[i:code.index("\n}", i) + 2] + "\n"}


@item("exp2-x")
def exp2_x():
    """2^x on [-3, 3], [-1/2, 1/2] shaded and drawn in blue over it: all the
    polynomial sees; every other unit is that piece times 2^n."""
    x = np.linspace(-3, 3, 1201)
    y = np.exp2(x)
    piece = np.where(np.abs(x) <= 0.5, y, np.nan)
    return {
        "plot": {
            "x": curve(x),
            "xlim": [-3, 3],
            "ylim": [0, 8.5],
            "xticks": [{"v": v, "label": str(v).replace("-", "−")} for v in range(-3, 4)],
            "yticks": [0, 2, 4, 6, 8],
            "band": [-0.5, 0.5],
            "series": [
                {"label": "2^x", "color": plots.TRUTH, "y": curve(y)},
                {"label": "2^f, \\; f \\in [-\\tfrac{1}{2}, \\tfrac{1}{2}]", "color": plots.SERIES[0], "gaps": True, "y": curve(piece)},
            ],
        },
    }

@item("exp2-errors")
def exp2_errors():
    """How did we do for 10's 2^x on the slides: relative and ULP error as morph
    steps over x in [-2, 2] (every unit interval looks the same: f is all
    the polynomial sees). Readouts from the bench, every float in
    [-125, 126]; no absolute error, which scales with 2^x."""
    x = np.linspace(-2, 2, 400001).astype(np.float32)
    truth = np.exp2(x.astype(np.float64))
    y = exp2poly.eval_unit(x, exp2poly.fit_exp2_unit(EXP2_DEGREE), np.float32)
    err = metrics.pointwise_errors(y, truth)
    steps = []
    for k in ("rel", "ulp"):
        xb, yb = envelope(x.astype(np.float64), err[k], bins=400)
        steps.append({"y": curve(yb)})
    row = bench_rows("exp2-f32.txt", UNARY_ERR_ROW, section="Error")[EXP2_NAME]
    max_rel = float(row["rel"])
    return {
        "tex_rel": f"\\max = {sci(max_rel)} \\;\\; ({-math.log2(max_rel):.0f}\\text{{ bits}})",
        "tex_cents": f"{sig(float(row['cents']), 2):g}\\,\\text{{cents}}",
        "tex_ulp_max": f"\\max = {ulp_text(float(row['ulp']))}",
        "plot": {
            "x": curve(xb),
            "xlim": [-2, 2],
            "ylim": [-10, 2],
            "xticks": [{"v": v, "label": str(v).replace("-", "−")} for v in (-2, -1, 0, 1, 2)],
            "yticks": log_ticks(-10, 2, 2),
            "series": [],
            "color": plots.SERIES[0],
            "steps": steps,
        },
    }

@item("log2-fold")
def log2_fold():
    """11: log2 near x = 1, ULP error over [½, 2] (log scale in x), for the
    plain mantissa form and the fold at 1.5, both LOG2_DEGREE. The
    floats just below 1 are all included: that's where the plain form fails.
    The C++ is the fold."""
    name, code = cppgen.emit_log2(LOG2_DEGREE, 1.5)
    below = np.float32(1).view(np.uint32) - np.arange(1, 4097, dtype=np.uint32)
    x = np.unique(np.concatenate([np.exp2(np.linspace(-1, 1, 200001)).astype(np.float32), below.view(np.float32)]))
    x = x[(x >= 0.5) & (x < 2)]
    truth = np.log2(x.astype(np.float64))
    steps = []
    for fold in (None, 1.5):
        y = log2poly.eval_mantissa(x, log2poly.fit_mantissa(LOG2_DEGREE, fold), fold, np.float32)
        ulp = metrics.ulp_error_f32(y, truth)
        xb, yb = envelope(np.log2(x.astype(np.float64)), np.where(np.isfinite(ulp), ulp, 0), bins=400)
        steps.append({"y": curve(yb)})
    err = bench_rows("log2-f32.txt", UNARY_ERR, section="Error")
    return {
        "name": name,
        "cpp": code.rstrip() + "\n",
        "cpp_body": code_body(code),
        "tex_mantissa": sci(float(err[f"mantissa_log2_{LOG2_DEGREE}"]["max"])),
        "tex_fold": f"{float(err[name]['max']):.2f}",
        "tex_sqrt2": f"{float(err[f'foldsqrt2_log2_{LOG2_DEGREE}']['max']):.2f}",
        "plot": {
            "x": curve(xb),
            "xlim": [-1, 1],
            "ylim": [-2, 9],
            "xticks": [{"v": -1, "label": "½"}, {"v": 0, "label": "1"}, {"v": 1, "label": "2"}],
            "yticks": log_ticks(-2, 9, 2),
            "series": [],
            "color": plots.SERIES[0],
            "steps": steps,
        },
    }


def log2_errors(fold, base_fold="none"):
    """How did we do for 11's log2 at LOG2_DEGREE with `fold` (None: m in [1, 2)):
    the C++, and absolute, relative and ULP error as morph steps over
    x in [½, 2] (log scale in x, the floats just below 1 included, as
    log2-fold), with the plain form faint behind it when base_fold isn't
    "none". Readouts from the bench, over every positive normal float;
    cents is the absolute error in octaves × 1200."""
    name, code = cppgen.emit_log2(LOG2_DEGREE, fold)
    below = np.float32(1).view(np.uint32) - np.arange(1, 4097, dtype=np.uint32)
    x = np.unique(np.concatenate([np.exp2(np.linspace(-1, 1, 200001)).astype(np.float32), below.view(np.float32)]))
    x = x[(x >= 0.5) & (x < 2)]
    truth = np.log2(x.astype(np.float64))

    def steps_for(f):
        y = log2poly.eval_mantissa(x, log2poly.fit_mantissa(LOG2_DEGREE, f), f, np.float32)
        err = metrics.pointwise_errors(y, truth)
        out = []
        for k in ("abs", "rel", "ulp"):
            xb, yb = envelope(np.log2(x.astype(np.float64)), np.where(np.isfinite(err[k]), err[k], 0), bins=400)
            out.append({"y": curve(yb)})
        return xb, out

    xb, steps = steps_for(fold)
    row = bench_rows("log2-f32.txt", UNARY_ERR_FULL, section="Error")[name]
    max_abs, max_rel = float(row["abs"]), float(row["rel"])
    plot = {
        "x": curve(xb),
        "xlim": [-1, 1],
        "ylim": [-10, 9],
        "xticks": [{"v": -1, "label": "½"}, {"v": 0, "label": "1"}, {"v": 1, "label": "2"}],
        "yticks": log_ticks(-10, 9, 2),
        "series": [],
    }
    if base_fold == "none":
        plot.update({"color": plots.SERIES[0], "steps": steps})
    else:
        plot["curves"] = [
            {"text": f"[1, 2) + degree {LOG2_DEGREE}", "color": COMPARE, "steps": steps_for(base_fold)[1]},
            {"text": f"Fold at 1.5 + degree {LOG2_DEGREE}", "color": plots.SERIES[0], "steps": steps},
        ]
    return {
        "name": name,
        "cpp_body": code_body(code),
        "tex_abs": f"\\max = {sci(max_abs)} \\;\\; ({20 * np.log10(max_abs):.0f}\\,\\text{{dB}})",
        "tex_cents": f"{sig(float(row['cents']), 2):g}\\,\\text{{cents}}",
        "tex_rel": (f"\\max = {max_rel:.0f}" if max_rel >= 1 else f"\\max = {sci(max_rel)}")
                   + f" \\;\\; ({max(0.0, -math.log2(max_rel)):.0f}\\text{{ bits}})",
        "tex_ulp_max": f"\\max = {ulp_text(float(row['ulp']))}",
        "plot": plot,
    }


@item("log2-mantissa")
def log2_mantissa():
    """How did we do, m in [1, 2); see log2_errors."""
    return log2_errors(None)


@item("log2-fold-errors")
def log2_fold_errors():
    """How did we do, folded at 1.5, the [1, 2) form faint behind; see
    log2_errors."""
    return log2_errors(1.5, base_fold=None)


@item("log2-fold-line")
def log2_fold_line():
    """Move the octave: m on [½, 2] and log2 m over it, for the foldline
    figure. The octave [1, 2); its top, [1.5, 2), halves onto [0.75, 1),
    and log2 of it drops by exactly 1 (so e + 1)."""
    m = np.linspace(0.5, 2, 601)
    return {
        "m": curve(m),
        "y": curve(np.log2(m)),
        "lo": 1, "fold": 1.5, "hi": 2,
        "xlim": [0.5, 2],
        "ylim": [-1.15, 1.15],
        "xticks": [{"v": v, "label": lab} for v, lab in [(0.5, "0.5"), (0.75, "0.75"), (1, "1"), (1.5, "1.5"), (2, "2")]],
        "yticks": [-1, 0, 1],
    }

@item("bits-scoreboard")
def bits_scoreboard():
    """exp2 and log2 rows from the bench: max ULP, cents, and max absolute
    error, also in dB (log2) or relative error (exp2: its output spans 2^-125 to 2^126,
    so absolute error means nothing) over every f32 (normal x in
    [-125, 126] for exp2, every normal x > 0 for log2), and buffer timing."""
    rows = []
    for fn, methods in [("log2", [("std::log2", "std::log2"), ("bits_log2_tuned", "Bits (Mitchell)"),
                                  (f"mantissa_log2_{LOG2_DEGREE}", f"[1, 2) + degree {LOG2_DEGREE}"),
                                  (f"fold15_log2_{LOG2_DEGREE}", f"Fold at 1.5 + degree {LOG2_DEGREE}")]),
                        ("exp2", [("std::exp2", "std::exp2"), ("bits_exp2_tuned", "Bits (Schraudolph)"),
                                  (EXP2_NAME, f"Round + degree {EXP2_DEGREE}"),
                                  ("table_exp2_3_16", "Table of 16 + degree 3")])]:
        err = bench_rows(f"{fn}-f32.txt", UNARY_ERR_FULL if fn == "log2" else UNARY_ERR_ROW, section="Error")
        timing = unary_timing(fn)
        for name, label in methods:
            extra = ({"abs": sig(float(err[name]["abs"]), 3), "abs_db": round(20 * math.log10(float(err[name]["abs"])), 1)}
                     if fn == "log2" else {"rel": sig(float(err[name]["rel"]), 3)})
            rows.append({"key": name, "label": label, "max_ulp": sig(float(err[name]["ulp"]), 3),
                         "cents": sig(float(err[name]["cents"]), 2), **extra, **timing_cols(timing[name])})
    return {"rows": rows}


TANH_ERR = (r"\s+(?P<name>.+?)\s{2,}(?P<ulp>[\d.e+-]+|inf)\s+(?P<ulp_at>\S+)\s+(?P<rel>\S+)\s+(?P<abs>[\d.e+-]+|inf)"
            r"\s+(?P<at>\S+)\s+(?P<db>-?[\d.]+|inf)\s+(?P<maxy>[\d.]+|inf)$")
TANH_ONE = float(np.arctanh(1 - 2.0**-25))  # f32(tanh x) = 1 from here
TANH_CLAMP = 9.0                            # the generated clamp


def tanh_err_rows():
    """The tanh bench's error table (every normal float): a row whose max is
    inf went non-finite, at x = `at`."""
    return bench_rows("tanh-f32.txt", TANH_ERR, section="Error")


@item("tanh-x")
def tanh_x():
    """tanh on [-12, 12] with [-9, 9] shaded (where we approximate) and
    dashed lines where f32(tanh x) becomes 1: past them no float can tell
    tanh from 1, so clamping x there costs nothing."""
    x = np.linspace(-12, 12, 1201)
    return {
        "tex_one": f"{TANH_ONE:.2f}",
        "tex_clamp": f"{TANH_CLAMP:g}",
        "plot": {
            "x": curve(x),
            "xlim": [-12, 12],
            "ylim": [-1.3, 1.3],
            "xticks": [{"v": v, "label": str(v).replace("-", "−")} for v in (-12, -9, -6, -3, 0, 3, 6, 9, 12)],
            "yticks": [-1, 0, 1],
            "band": [-TANH_CLAMP, TANH_CLAMP],
            "vlines": [sig(-TANH_ONE), sig(TANH_ONE)],
            "series": [{"label": "\\tanh x", "color": plots.TRUTH, "y": curve(np.tanh(x))}],
        },
    }


@item("tanh-sigmoid-shape")
def tanh_sigmoid_shape():
    """The algebraic sigmoid y = s / sqrt(1 + s^2) beside tanh x, on [-5, 5],
    a morph in two steps (Build in the asymptote, a pop-over from the second
    line's click): s = x, the plain sigmoid, already bounded by ±1; then
    s = x R(x^2) with the slide's fitted R (fit_sigmoid_cheb(9), in f64), on
    top of tanh. tanh is a moving curve that doesn't move, so the legend
    lists both."""
    x = np.linspace(-5, 5, 501)
    r = tanhpoly.fit_sigmoid_cheb(9)
    return {
        "plot": {
            "x": curve(x),
            "xlim": [-5, 5],
            "ylim": [-1.25, 1.25],
            "xticks": [{"v": v, "label": str(v).replace("-", "−")} for v in (-5, 0, 5)],
            "yticks": [-1, 0, 1],
            "series": [],
            "curves": [
                {"label": "\\tanh x", "color": plots.TRUTH, "steps": [{"y": curve(np.tanh(x))}] * 2},
                {"label": "s / \\sqrt{1 + s^2}", "color": plots.SERIES[0], "steps": [
                    {"y": curve(x / np.sqrt(1 + x * x))},
                    {"y": curve(tanhpoly.eval_sigmoid(x, r, np.float64))},
                ]},
            ],
        },
    }


def tanh_runoff_fns():
    """12's degree-9 polynomial (fit on ±L) and JUCE's Padé [7/6], neither
    clamped, in f32."""
    limit = tanhpoly.best_clamp_limit(9)
    p = tanhpoly.fit_clamp(9, limit)
    pe, qe = tanhpoly.pade_tanh(7, 6)
    return limit, [
        ("Polynomial, degree 9", plots.SERIES[1], lambda x: tanhpoly.eval_clamp(x, p, 1e30, np.float32)),
        ("Padé [7/6]", plots.SERIES[2], lambda x: tanhpoly.eval_pade(x, pe, qe, np.float32, clamp=False)),
    ]


@item("tanh-runoff")
def tanh_runoff():
    """tanh on [0, 30] (it's odd) with the polynomial and Padé, unclamped:
    the polynomial leaves at its fit edge; Padé looks right out to 9 and
    then climbs away from 1. Clicks reveal one series each (plot.js
    `shown`), then stretch the x axis from [0, 4] to [0, 30] (`view`)."""
    limit, fns = tanh_runoff_fns()
    x = np.linspace(0, RUNOFF_VIEWS[-1]["xlim"][1], 3001)
    pe, qe = tanhpoly.pade_tanh(7, 6)
    xs = np.linspace(0, 10, 100001)
    cross = xs[np.argmax(tanhpoly.eval_pade(xs, pe, qe, np.float64, clamp=False) > 1)]
    return {
        "tex_limit": f"{limit:.2f}",
        "tex_cross": f"{cross:.2f}",
        "plot": {
            "x": curve(x),
            "views": RUNOFF_VIEWS,
            "ylim": [0, 1.4],
            "xticks": runoff_xticks(),
            "yticks": [0, 1],
            "band": [0, TANH_CLAMP],
            "series": [{"label": "\\tanh x", "color": plots.TRUTH, "y": curve(np.tanh(x))}]
                      + [{"text": name, "color": c, "y": curve(f(x.astype(np.float32)))} for name, c, f in fns],
        },
    }


def tanh_err_series(fns, x, bins=400):
    """log10 |error| per bin on x, for each (name, color, f)."""
    t = np.tanh(x.astype(np.float64))
    out = []
    for name, color, f in fns:
        with np.errstate(over="ignore", invalid="ignore"):
            e = np.abs(f(x).astype(np.float64) - t)
        xb, yb = envelope(x.astype(np.float64), np.where(np.isfinite(e), e, 1.0), bins=bins)
        out.append({"text": name, "color": color, "y": curve(yb)})
    return xb, out


def tanh_cost_series(xb):
    """1 - tanh x: what clamping at x costs, dashed."""
    with np.errstate(divide="ignore"):  # 1 - tanh x is exactly 0 in f64 past ~19: a gap
        cost = np.log10(1 - np.tanh(xb))
    return {"text": "1 − tanh x: clamping here", "color": plots.MUTED, "dashed": True, "y": curve(cost)}


def tanh_err_plot(xb, series, views=None):
    plot = {
        "x": curve(xb),
        "xlim": [0, 10],
        "ylim": [-9, 0],
        "xticks": [{"v": v, "label": str(v)} for v in (0, 2, 4, 6, 8, 10)],
        "yticks": log_ticks(-9, 0, 3),
        "vlines": [TANH_CLAMP],
        "series": series,
    }
    if views:
        plot.update({"views": views, "xticks": runoff_xticks()})
        del plot["xlim"]
    return plot


# The run-off slide's x views: the polynomial's reach, then far enough out
# that Padé visibly leaves 1 (1.37 at 30).
RUNOFF_VIEWS = [{"xlim": [0, 4], "xticks": [0, 1, 2, 3, 4]},
                {"xlim": [0, 30], "xticks": [0, 5, 10, 15, 20, 25, 30]}]


def runoff_xticks():
    return [{"v": v, "label": str(v)} for v in sorted({v for view in RUNOFF_VIEWS for v in view["xticks"]})]


@item("tanh-runoff-err")
def tanh_runoff_err():
    """|error| of the unclamped polynomial and Padé on [0, 30], log scale,
    under 1 - tanh x (the cost of clamping at x): the polynomial goes at
    its fit edge; Padé is superb near 0, climbs from ~3, and keeps going.
    The same x views as tanh-runoff."""
    limit, fns = tanh_runoff_fns()
    x = np.linspace(0, RUNOFF_VIEWS[-1]["xlim"][1], 600001).astype(np.float32)
    xb, series = tanh_err_series(fns, x, bins=2400)
    pade = fns[1][2]
    at9 = abs(float(pade(np.array([TANH_CLAMP], np.float32))[0]) - np.tanh(TANH_CLAMP))
    at30 = float(pade(np.array([30], np.float32))[0])
    return {"tex_pade9": sci(at9), "tex_pade30": f"{at30:.2f}",
            "plot": tanh_err_plot(xb, [tanh_cost_series(xb)] + series, views=RUNOFF_VIEWS)}


@item("tanh-reach")
def tanh_reach():
    """|error| on [0, 10] of what ships: the polynomial, Padé [7/6] and the
    sigmoid (plain Chebyshev R, tanhpoly.fit_sigmoid_cheb), each with its
    input clamped (2.80, 4.79, 9), under 1 - tanh x. Readouts are the
    bench's max over every float."""
    limit, (poly, pade) = tanh_runoff_fns()
    p = tanhpoly.fit_clamp(9, limit)
    pe, qe = tanhpoly.pade_tanh(7, 6)
    pade_limit = tanhpoly.best_pade_clamp(7, 6)
    r = tanhpoly.fit_sigmoid_cheb(9)
    fns = [
        (f"Polynomial, clamp {limit:.2f}", plots.SERIES[1], lambda x: tanhpoly.eval_clamp(x, p, limit, np.float32)),
        (f"Padé [7/6], clamp {pade_limit:.2f}", plots.SERIES[2],
         lambda x: tanhpoly.eval_pade(x, pe, qe, np.float32, limit=pade_limit)),
        (f"Sigmoid, clamp {TANH_CLAMP:g}", plots.SERIES[0], lambda x: tanhpoly.eval_sigmoid(x, r, np.float32, TANH_CLAMP)),
    ]
    x = np.linspace(0, 10, 200001).astype(np.float32)
    xb, series = tanh_err_series(fns, x)
    err = tanh_err_rows()
    db = lambda k: f"{float(err[k]['db']):.0f}\\,\\text{{dB}}"
    return {
        "tex_clamp": db("clamp_tanh_9"),
        "tex_pade_limit": f"{pade_limit:.2f}",
        "tex_pade": db("pade_clamp_tanh_7_6"),
        "tex_sigmoid": db("sigmoid_cheb_clamp_tanh_9"),
        "plot": tanh_err_plot(xb, [tanh_cost_series(xb)] + series),
    }


@item("tanh-clamp")
def tanh_clamp():
    """12's polynomial: degree 9 fit to tanh on [-L, L] (L chosen by
    tanhpoly.best_clamp_limit), then the same with x clamped to ±L. A
    polynomial runs off; the clamp makes the asymptote, with a kink."""
    degree = 9
    limit = tanhpoly.best_clamp_limit(degree)
    p = tanhpoly.fit_clamp(degree, limit)
    x = np.linspace(-6, 6, 1201)
    free = x * np.polynomial.polynomial.polyval(x * x, p)
    return {
        "tex_limit": f"L = {limit:.2f}",
        "plot": {
            "x": curve(x),
            "xlim": [-6, 6],
            "ylim": [-1.6, 1.6],
            "xticks": [{"v": v, "label": str(v).replace("-", "−")} for v in range(-6, 7, 2)],
            "yticks": [-1, 0, 1],
            "band": [-limit, limit],
            "series": [{"label": "\\tanh x", "color": plots.TRUTH, "y": curve(np.tanh(x))}],
            "color": plots.SERIES[0],
            "steps": [{"y": curve(free)}, {"y": curve(tanhpoly.eval_clamp(x, p, limit, np.float64))}],
        },
    }


@item("tanh-pade")
def tanh_pade():
    """Padé [7/6] of tanh (JUCE's fastTanh): where it passes 1, and what it
    tends to (an odd rational over an even one grows like c x)."""
    pe, qe = tanhpoly.pade_tanh(7, 6)
    x = np.linspace(0, 10, 100001)
    y = tanhpoly.eval_pade(x, pe, qe, np.float64, clamp=False)
    cross = x[np.argmax(y > 1)]
    return {"tex_cross": f"x = {cross:.2f}", "tex_slope": f"{float(pe[-1] / qe[-1]):.3g}\\,x"}


@item("tanh-sigmoid-9")
def tanh_sigmoid_9():
    """The algebraic sigmoid, R by a plain Chebyshev fit on [0, 3.4] with
    R(0) = 1 (tanhpoly.fit_sigmoid_cheb), x clamped at 9: its C++, and the
    fit as a snippet, run here and checked against tanhpoly (q is never
    evaluated at v = 0: the nodes are interior)."""
    r = tanhpoly.fit_sigmoid_cheb(9)
    name, code = cppgen.emit_tanh_sigmoid(9, r, "sigmoid_cheb_clamp_tanh_9", TANH_CLAMP)
    py_fit = ("def q(v):\n"
              "    x = np.sqrt(v)\n"
              "    return (np.sinh(x) / x - 1) / v\n"
              "\n"
              "Q = Chebyshev.interpolate(q, 128, domain=[0, 3.4**2])\n"
              "Q = Q.truncate(4)\n")
    scope = {"np": np, "Chebyshev": Chebyshev}
    exec(py_fit, scope)
    c = scope["Q"].convert(kind=np.polynomial.Polynomial, domain=[-1, 1], window=[-1, 1]).coef
    assert np.allclose(np.concatenate([[1.0], c]), r, rtol=1e-9, atol=0), "tanh fit snippet drifted"
    return {"name": name, "cpp": code_body(code), "py_fit": py_fit}


@item("tanh-errors")
def tanh_errors():
    """Absolute error of each tanh form in f32 on [0, 8], as morph steps
    (log10 of the worst error per bin): poly + clamp 9, Padé [7/6] + clamp,
    sigmoid with Remez s, degree 9. Exhaustive maxima from the bench."""
    x = np.linspace(0, 8, 200001).astype(np.float32)
    truth = np.tanh(x.astype(np.float64))
    steps = []
    for step, size in [("clamp", 9), ("pade", (7, 6)), ("sigmoid", 9)]:
        y = tanhpoly.method(step, size, np.float32).f(x).astype(np.float64)
        xb, yb = envelope(x.astype(np.float64), np.abs(y - truth), bins=400)
        steps.append({"y": curve(yb)})
    err = tanh_err_rows()
    db = lambda k: f"{float(err[k]['db']):.0f}\\,\\text{{dB}}"
    return {
        "tex_clamp": db("clamp_tanh_9"),
        "tex_pade": db("pade_clamp_tanh_7_6"),
        "tex_sigmoid": db("sigmoid_clamp_tanh_9"),
        "plot": {
            "x": curve(xb),
            "xlim": [0, 8],
            "ylim": [-9, -1],
            "xticks": [{"v": v, "label": str(v)} for v in range(0, 9, 2)],
            "yticks": log_ticks(-9, -1, 2),
            "series": [],
            "color": plots.SERIES[0],
            "steps": steps,
        },
    }


@item("tanh-rsqrt")
def tanh_rsqrt():
    """tanh's square root in SIMD (bench/src/rsqrt.cpp): sigmoid_tanh_9 four
    wide with five square roots, every float in [-20, 20]. NEON measured
    here (M-series); SSE under Rosetta, so its accuracy is Rosetta's
    rsqrtps and its timing is left out. The NEON rsqrt + Newton code is
    taken from the source."""
    pat = r"\s+(?P<name>.+?)\s{2,}(?P<db>-?[\d.]+)\s+(?P<y>[\d.]+)\s+(?P<ns>[\d.]+)$"
    neon = bench_rows("tanh-rsqrt-neon.txt", pat)
    sse = bench_rows("tanh-rsqrt-sse-rosetta.txt", pat)
    def max_y(v):
        v = float(v)
        return "1" if v == 1 else (f"{v:.4f}" if v - 1 > 1e-4 else f"{v:.8f}")
    rows = [{"key": name, "label": name, "neon_db": sig(float(r["db"]), 4), "neon_y": max_y(r["y"]),
             "neon_ns": sig(float(r["ns"]), 3), "sse_db": sig(float(sse[name]["db"]), 4)}
            for name, r in neon.items()]
    src = (ROOT / "bench/src/rsqrt.cpp").read_text()
    i = src.index("template<int Steps>\ninline auto rsqrt(V a) -> V")
    code = src[i:src.index("\n}\n", i) + 3]
    assert "vrsqrteq_f32" in code
    return {"rows": rows, "cpp": code}


def first_wrong_tanh(f):
    """Smallest f32 x >= 100 where f(x) is no longer near 1 (|y - 1| >= 1/2
    or non-finite), by bisection on the bit patterns: past the overflow the
    failure is monotone (17-tanh-revisited)."""
    as_x = lambda u: np.array([u], dtype=np.uint32).view(np.float32)
    wrong = lambda u: not (abs(float(f(as_x(u))[0]) - 1) < 0.5)
    a, b = int(np.float32(100).view(np.uint32)), int(np.float32(np.finfo(np.float32).max).view(np.uint32))
    assert wrong(b)
    with np.errstate(over="ignore", invalid="ignore"):
        while a < b:
            m = (a + b) // 2
            a, b = (a, m) if wrong(m) else (m + 1, b)
    return float(as_x(a)[0])


def sigmoid_clamp_f32(x, degree=9):
    """The generated sigmoid_clamp_tanh_<degree> in f32, as clang builds it on
    arm64: x clamped to ±9, R by Horner with each step fused (r * x2 + c),
    s * s + 1 fused, then s / sqrt. A fused step is exact in f64 and rounded
    once. Matches the bench's worst cases exactly (abs at 0.7908, ULP at
    0.01561); tanhpoly.eval_sigmoid computes s * (1 / sqrt) unfused."""
    f32, f64 = np.float32, np.float64
    fma = lambda a, b, c: (a.astype(f64) * b.astype(f64) + f64(c)).astype(f32)
    r = tanhpoly.fit_sigmoid(degree).astype(f32)
    xc = np.clip(np.asarray(x, f32), f32(-TANH_CLAMP), f32(TANH_CLAMP))
    x2 = xc * xc
    p = np.full_like(xc, r[-1])
    for c in r[-2::-1]:
        p = fma(p, x2, c)
    s = xc * p
    return s / np.sqrt(fma(s, s, f32(1)))


@item("tanh-sigmoid-errors")
def tanh_sigmoid_errors():
    """How did we do, for the clamped sigmoid (degree 9, clamp at 9) on
    [-9, 9]: absolute, relative and ULP error as morph steps (the floats
    near 0 included, where the ULP max is). Readouts from the bench, over
    every normal float: outside ±9 the clamp returns 1, which is tanh in
    f32 from 9.01."""
    x = np.linspace(-TANH_CLAMP, TANH_CLAMP, 1800001).astype(np.float32)
    tiny = np.float32(2) ** -np.arange(1, 40, dtype=np.float32)
    x = np.unique(np.concatenate([x, tiny, -tiny]))
    err = metrics.pointwise_errors(sigmoid_clamp_f32(x), np.tanh(x.astype(np.float64)))
    steps = []
    for k in ("abs", "rel", "ulp"):
        xb, yb = envelope(x.astype(np.float64), np.where(np.isfinite(err[k]), err[k], 0), bins=600)
        steps.append({"y": curve(yb)})
    row = tanh_err_rows()["sigmoid_clamp_tanh_9"]
    max_abs, max_rel = float(row["abs"]), float(row["rel"])
    return {
        "tex_abs": f"\\max = {sci(max_abs)} \\;\\; ({float(row['db']):.0f}\\,\\text{{dB}})",
        "tex_rel": f"\\max = {sci(max_rel)} \\;\\; ({-math.log2(max_rel):.0f}\\text{{ bits}})",
        "tex_ulp_max": f"\\max = {ulp_text(float(row['ulp']))} \\;\\; (x = {float(row['ulp_at']):.3f})",
        "plot": {
            "x": curve(xb),
            "xlim": [-TANH_CLAMP, TANH_CLAMP],
            "ylim": [-10, 4],
            "xticks": [{"v": v, "label": str(v).replace("-", "−")} for v in (-9, -6, -3, 0, 3, 6, 9)],
            "yticks": log_ticks(-10, 4, 2),
            "series": [],
            "color": plots.SERIES[0],
            "steps": steps,
        },
    }

@item("tanh-improve")
def tanh_improve():
    """Improvements, after the tanh scoreboard: the clamped sigmoid fitted for
    absolute error (as shipped), for relative error, and at degree 11 for
    relative error with R(0) = 1 (tanhpoly.fit_sigmoid): exhaustive max ULP,
    absolute error and buffer timing from the bench."""
    err = tanh_err_rows()
    timing = unary_timing("tanh")
    methods = [("sigmoid_clamp_tanh_9", "Degree 9", "absolute"),
               ("sigmoid_rel_clamp_tanh_9", "Degree 9", "relative"),
               ("sigmoid_relpin_clamp_tanh_11", "Degree 11", "relative, R(0) = 1")]
    return {"rows": [{"key": k, "label": label, "fit": fit, "max_ulp": sig(float(err[k]["ulp"]), 3),
                      "db": f"{float(err[k]['db']):.1f}", **timing_cols(timing[k])}
                     for k, label, fit in methods]}

@item("tanh-scoreboard")
def tanh_scoreboard():
    """Max ULP and max absolute error over every normal f32 (as dB, or
    where the output goes non-finite), max |y|, and buffer timing, from
    the bench."""
    err = tanh_err_rows()
    timing = unary_timing("tanh")
    methods = [("std::tanh", "std::tanh"), ("clamp_tanh_9", "Poly + clamp, degree 9"),
               ("pade_clamp_tanh_7_6", f"Padé [7/6], clamp at {tanhpoly.best_pade_clamp(7, 6):.2f}"),
               ("sigmoid_cheb_clamp_tanh_9", "Sigmoid 9, clamp at 9")]
    pe, qe = tanhpoly.pade_tanh(7, 6)
    python = {"pade_tanh_7_6": lambda x: tanhpoly.eval_pade(x, pe, qe, np.float32)}
    def abs_text(k, r):
        if r["db"] != "inf":
            return f"{float(r['db']):.1f} dB".replace("-", "−")
        # Non-finite somewhere: say where it first goes wrong (|y - 1| >= 1/2).
        x = first_wrong_tanh(python[k])
        return f"wrong past {x:.3g}".replace("e+0", "e").replace("e+", "e")
    def max_y(r):
        return "—" if r["maxy"] == "inf" else f"{sig(float(r['maxy']), 5):g}"
    return {"rows": [{"key": k, "label": label, "max_ulp": sig(float(err[k]["ulp"]), 3),
                      "max_abs": abs_text(k, err[k]), "max_y": max_y(err[k]),
                      **timing_cols(timing[k])} for k, label in methods]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(names):
    unknown = [n for n in names if n not in ITEMS]
    if unknown:
        sys.exit(f"unknown item(s): {', '.join(unknown)}; have {', '.join(ITEMS)}")
    OUT_DIR.mkdir(exist_ok=True)
    for name in names or ITEMS:
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(ITEMS[name](), separators=(",", ":")) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
