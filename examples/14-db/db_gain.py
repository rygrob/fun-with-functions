"""Fast dB <-> linear gain: exp2 and log2 with a scale, sized for a dB budget.

    gain = 10^(dB/20) = 2^(dB K)       dB = 20 log10(x) = log2(x) / K,   K = 0.1661

So these are 10-exp2 and 11-log2 with one extra multiply. What's specific to
gain is the budget and what matters inside it:

  budget       a level error of 0.01 dB is far below audibility, so the right
               question is the cheapest form for 0.1 / 0.01 / 0.001 dB, not
               the most accurate. Degree 3 is already ~0.001 dB
  continuity   floor reduction steps at every 6.02 dB unless P(1) = 2 P(0).
               A free fit misses by up to 0.03 dB at degree 2 -- a tiny step
               on every slow fade, and a gain that isn't monotonic. Pinning
               the fit's ends (P(0) = 1, P(1) = 2, fit.minimax_endpoints)
               removes the step and makes 0 dB exactly unity, for free
  unity        0 dB must give a gain of exactly 1 (a bypassed compressor, a
               fader at 0). The tuned-offset bit trick gives 0.978
  ramps        a fade that's linear in dB is geometric in gain:
               g[n+1] = g[n] * r, one multiply per sample. f32 drift is
               measured below, with re-anchoring once per block
  compressor   level -> dB -> gain computer -> gain, per sample, both
               conversions in one loop: how their errors combine, and which
               pairs keep the gain exactly 1 below threshold
  extended     what happens at the extremes of the float range: silence
               (-inf dB, gain 0), subnormal gains, overflow, huge dB

Mitchell's (1962) and Schraudolph's (1999) bit tricks are the degree-1 ends of
the same line: a pinned degree-1 fit *is* Schraudolph's chord.

Conversions live in ../common/dbpoly.py.

Usage:
    python db_gain.py                    # DEGREE below
    python db_gain.py -d 2               # another degree for ramps / compressor
    python db_gain.py --compare          # full degree sweep, cheapest per budget
    python db_gain.py --eval f64         # evaluate in f64
    python db_gain.py --emit-cpp         # C++ for the pinned and safe forms
    python db_gain.py --no-show          # just write the PNGs
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, dbpoly, fit, harness, plots, report
from common.metrics import EVAL_DTYPES

DEGREE = 3
COMPARE_DEGREES = [1, 2, 3, 4, 5, 6]
BUDGETS = [0.1, 0.01, 0.001]          # dB

DB_RANGE = (-120.0, 24.0)             # the audio range: noise floor to +24 dB of boost
GAIN_RANGE = (1e-6, 16.0)             # the same, as gains
FS = 48_000
RAMPS = [(-60.0, 0.0, 1.0), (0.0, -60.0, 10.0), (-96.0, 0.0, 60.0)]   # (from dB, to dB, seconds)
ANCHOR_BLOCK = 64
THRESHOLD, RATIO = -24.0, 4.0
EXT_DB = [-np.inf, -1000.0, -900.0, -800.0, -760.0, -750.0, -400.0, 0.0, 400.0, 770.0, 771.0, 800.0, 1000.0,
          1600.0]
EXT_GAIN = [0.0, 1e-45, 1e-40, 1.1754944e-38, 1e-30, 1.0, 1e30, 3.4028235e38, np.inf]

OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Accuracy and continuity
# ---------------------------------------------------------------------------

def gain_inputs():
    return np.unique(np.linspace(*DB_RANGE, 2_000_001).astype(np.float32))


def db_inputs():
    return np.unique(np.geomspace(*GAIN_RANGE, 2_000_001).astype(np.float32))


def neighbours(centres, dtype, n=48):
    """Every float within n steps of each centre: windows straddling a
    boundary, so a step there shows as a jump between neighbours."""
    out = []
    for c in centres:
        c = dtype(c)
        below, above = [c], [c]
        for _ in range(n):
            below.append(np.nextafter(below[-1], dtype(-np.inf)))
            above.append(np.nextafter(above[-1], dtype(np.inf)))
        out.append(np.array(below[::-1] + above[1:], dtype=dtype))
    return out


def max_step(windows, f, truth):
    """Largest jump in the error between neighbouring inputs: a step in the
    approximation that the truth doesn't take."""
    worst = 0.0
    for w in windows:
        e = f(w) - truth(w)
        worst = max(worst, np.abs(np.diff(e)).max())
    return worst


def gain_stats(m, db, truth):
    g = m.f(db).astype(np.float64)
    err = 20 * np.log10(g / truth)
    level = 20 * np.log10(g)
    windows = neighbours([n / dbpoly.K for n in range(-19, 4)], np.float32)
    to_db = lambda w: 20 * np.log10(m.f(w).astype(np.float64))
    return err, {
        "max": np.abs(err).max(),
        "unity": 20 * np.log10(float(m.f(np.array([0.0], dtype=np.float32))[0])),
        "step": max_step(windows, to_db, lambda w: w.astype(np.float64)),
        "drop": max(0.0, -np.diff(level).min()),
    }


def db_stats(m, x, truth):
    y = m.f(x).astype(np.float64)
    err = y - truth
    centres = [1.5 * 2.0**k for k in range(-19, 4)] + [2.0**k for k in range(-19, 4)]
    windows = neighbours(centres, np.float32)
    return err, {
        "max": np.abs(err).max(),
        "unity": float(m.f(np.array([1.0], dtype=np.float32))[0]),
        "step": max_step(windows, lambda w: m.f(w).astype(np.float64), lambda w: 20 * np.log10(w.astype(np.float64))),
        "drop": max(0.0, -np.diff(y).min()),
    }


def print_accuracy(rows, title, unity_label):
    print(title)
    print(f"  {'method':<24} {'degree':>6} {'mul':>4} {'add':>4}  {'max err dB':>10}  {unity_label:>12}"
          f"  {'max step dB':>11}  {'max drop dB':>11}")
    for m, _, s in rows:
        print(f"  {m.name:<24} {m.degree:>6} {m.mul:>4} {m.add:>4}  {s['max']:>10.2e}  {s['unity']:>+12.3e}"
              f"  {s['step']:>11.1e}  {s['drop']:>11.1e}")
    print()


def print_budget(gain_rows, db_rows):
    """The cheapest polynomial in each direction for each budget."""
    print("Cheapest for a dB budget (pinned fits; mul + add per conversion):")
    for b in BUDGETS:
        cells = []
        for rows in (gain_rows, db_rows):
            ok = [(m.mul + m.add, m) for m, _, s in rows if s["max"] <= b and m.degree != "—" and "safe" not in m.name
                  and "pinned" in m.name]
            cells.append(f"degree {min(ok, key=lambda t: t[0])[1].degree} ({min(ok, key=lambda t: t[0])[0]} ops)"
                         if ok else "—")
        print(f"  {b:>6g} dB:  dB -> gain {cells[0]:<20}  gain -> dB {cells[1]}")
    print("  (the bit tricks are 0.26-0.52 dB: fine for a meter, not for a gain)")
    print()


# ---------------------------------------------------------------------------
# Extended range
# ---------------------------------------------------------------------------

def print_extended(degree, dtype):
    g_methods = [dbpoly.gain_method(s, degree, dtype) for s in dbpoly.GAIN_STEPS if s != "free"]
    d_methods = [dbpoly.db_method(s, degree + 1, dtype) for s in dbpoly.DB_STEPS if s != "fold"]
    db = np.array(EXT_DB, dtype=np.float32)     # inputs are f32 even in f64 runs
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        truth = np.power(10.0, db.astype(np.float64) / 20)
        print(f"Extended range, dB -> gain ({np.dtype(dtype).name}; f32 gains run from -897 dB, the smallest"
              " subnormal, to +770.6 dB):")
        print(f"  {'dB':>8}  {'true gain':>11}" + "".join(f"  {m.name[:19]:>19}" for m in g_methods))
        for i, d in enumerate(db):
            vals = [float(m.f(db[i:i + 1])[0]) for m in g_methods]
            print(f"  {float(d):>8g}  {truth[i]:>11.4g}" + "".join(f"  {v:>19.6g}" for v in vals))
    print("  bits: dB K 2^23 overflows int32 past ~1540 dB; between, the exponent field wraps like 10-exp2's")
    print("  plain polynomials: right only while 2^n is a normal float (-758 .. +770 dB)")
    print()

    x = np.array(EXT_GAIN, dtype=np.float32)
    with np.errstate(divide="ignore"):
        truth = 20 * np.log10(x.astype(np.float64))
        print(f"Extended range, gain -> dB ({np.dtype(dtype).name}):")
        print(f"  {'gain':>12}  {'true dB':>10}" + "".join(f"  {m.name[:23]:>23}" for m in d_methods))
        for i, xi in enumerate(x):
            vals = [float(m.f(x[i:i + 1])[0]) for m in d_methods]
            print(f"  {float(xi):>12.5g}  {truth[i]:>10.4g}" + "".join(f"  {v:>23.6g}" for v in vals))
    print("  subnormal gains read e = -127: 1e-40 comes out as -764 dB instead of -800; silence (0) as -765 dB")
    print()

    # Error over the whole normal range, not just the audio one.
    ext_db = np.unique(np.linspace(-758.0, 770.0, 2_000_001).astype(np.float32))
    tg = np.power(10.0, ext_db.astype(np.float64) / 20)
    ext_x = np.unique(np.exp2(np.linspace(-149, 127.99, 2_000_001)).astype(np.float32))
    ext_x = ext_x[ext_x > 0]
    td = 20 * np.log10(ext_x.astype(np.float64))
    print("Max error over the extended ranges (dB):")
    print(f"  {'method':<28} {'dB in [-758, 770]':>18}   {'method':<28} {'normal gains':>13} {'subnormal':>11}")
    sub = ext_x < np.finfo(np.float32).tiny     # subnormal as f32 inputs
    for gm, dm in zip(g_methods, d_methods):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ge = np.abs(20 * np.log10(gm.f(ext_db).astype(np.float64) / tg))
            de = np.abs(dm.f(ext_x).astype(np.float64) - td)
        ge = np.where(np.isfinite(ge), ge, np.inf)
        print(f"  {gm.name:<28} {ge.max():>18.3g}   {dm.name:<28} {de[~sub].max():>13.3g} {de[sub].max():>11.3g}")
    print()


# ---------------------------------------------------------------------------
# Ramps
# ---------------------------------------------------------------------------

def ramp_variants(degree, dtype):
    gm = dbpoly.gain_method("pinned", degree, dtype)

    def per_sample(db, r, g0):
        return gm.f(db.astype(dtype))

    def recurrence(db, r, g0, state=dtype):
        steps = np.full(len(db), state(r), dtype=state)
        steps[0] = state(g0)
        return np.cumprod(steps, dtype=state).astype(dtype)

    def anchored(db, r, g0):
        out = np.empty(len(db), dtype=dtype)
        for start in range(0, len(db), ANCHOR_BLOCK):
            blk = db[start:start + ANCHOR_BLOCK]
            steps = np.full(len(blk), dtype(r), dtype=dtype)
            steps[0] = gm.f(np.array([blk[0]], dtype=dtype))[0]
            out[start:start + len(blk)] = np.cumprod(steps, dtype=dtype)
        return out

    return [
        (f"exp2 pinned {degree}, every sample", f"{gm.mul} mul, {gm.add} add", per_sample),
        ("g *= r, f32", "1 mul", recurrence),
        ("g *= r, f64 state", "1 mul (f64)", lambda db, r, g0: recurrence(db, r, g0, np.float64)),
        (f"g *= r, f32, re-anchored / {ANCHOR_BLOCK}", f"1 mul + 1/{ANCHOR_BLOCK} exp2", anchored),
    ]


def run_ramps(degree, dtype):
    print(f"Ramps linear in dB at {FS / 1000:g} kHz ({np.dtype(dtype).name}): error vs exact 10^(dB/20), in dB")
    variants = ramp_variants(degree, dtype)
    print(f"  {'method':<36} {'cost / sample':>18}" + "".join(f"  {f'{a:g} -> {b:g} dB, {s:g} s':>22}"
                                                             for a, b, s in RAMPS))
    traces = {}
    rows = {name: [] for name, _, _ in variants}
    for a, b, secs in RAMPS:
        n = int(secs * FS)
        db = a + (b - a) * np.arange(n) / n
        exact = np.power(10.0, db / 20)
        r = 10.0 ** ((b - a) / (20 * n))
        for name, _, fn in variants:
            err = 20 * np.log10(fn(db, r, 10.0 ** (a / 20)).astype(np.float64) / exact)
            rows[name].append(np.abs(err).max())
            if (a, b, secs) == RAMPS[1]:
                traces[name] = err
    for name, cost, _ in variants:
        print(f"  {name:<36} {cost:>18}" + "".join(f"  {e:>22.2e}" for e in rows[name]))
    print("  f32 drift is mostly r's own rounding, compounded n times; f64 state or re-anchoring bounds it")
    print()
    return traces


def plot_ramp(traces, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    a, b, secs = RAMPS[1]
    t = np.arange(int(secs * FS)) / FS
    for color, (name, err) in zip(plots.SERIES, traces.items()):
        ax.plot(t[::16], err[::16], color=color, lw=1.2, label=name)
    ax.axhline(0, color=plots.MUTED, lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("error (dB)")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    plots.clean_axes(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

def compressor_pairs(degree, dtype):
    g = lambda s, d=degree: dbpoly.gain_method(s, d, dtype)
    d = lambda s, deg=degree + 1: dbpoly.db_method(s, deg, dtype)
    return [
        ("std (log10 + pow)", d("log10"), g("pow")),
        ("Mitchell + Schraudolph", d("bits"), g("bits")),
        ("both tuned offsets", d("bits-tuned"), g("bits-tuned")),
        ("Mitchell + exact pow", d("bits"), g("pow")),
        ("exact log10 + Schraudolph", d("log10"), g("bits")),
        (f"free fits ({degree + 1}, {degree})", d("fold"), g("free")),
        (f"pinned fits ({degree + 1}, {degree})", d("fold-pinned"), g("pinned")),
    ]


def run_compressor(degree, dtype):
    """Static curve: level -> dB -> gain computer -> gain, against exact."""
    levels = np.linspace(-90.0, 0.0, 90_001)
    x = np.power(10.0, levels / 20).astype(dtype)
    lx = 20 * np.log10(x.astype(np.float64))
    slope = dtype(1 - 1 / RATIO)
    exact = np.power(10.0, np.minimum(0.0, (THRESHOLD - lx) * (1 - 1 / RATIO)) / 20)
    below = lx < THRESHOLD - 1
    above = lx > THRESHOLD + 1
    print(f"Compressor static curve ({np.dtype(dtype).name}): threshold {THRESHOLD:g} dB, ratio {RATIO:g}:1, hard knee;"
          " output gain error vs exact, dB")
    print(f"  {'detector + gain':<28} {'below threshold':>16} {'gain == 1 below':>16} {'above threshold':>16}")
    traces = {}
    for name, dm, gm in compressor_pairs(degree, dtype):
        level = dm.f(x)
        gr = np.where(level > dtype(THRESHOLD), (dtype(THRESHOLD) - level) * slope, dtype(0))
        gain = gm.f(gr.astype(dtype)).astype(np.float64)
        err = 20 * np.log10(gain / exact)
        traces[name] = err
        print(f"  {name:<28} {np.abs(err[below]).max():>16.2e} {str(bool(np.all(gain[below] == 1))):>16}"
              f" {np.abs(err[above]).max():>16.2e}")
    print(f"  above threshold: output error ~ (1 - 1/R) x detector error + gain error;"
          f" a ratio of {RATIO:g} keeps {1 - 1 / RATIO:.0%} of the detector's")
    print()
    return levels, traces


def plot_compressor(levels, traces, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    for color, (name, err) in zip(plots.SERIES, traces.items()):
        ax.plot(levels, err, color=color, lw=1.2, label=name)
    ax.axvline(THRESHOLD, color=plots.MUTED, lw=0.8, ls=":")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("input level (dB)")
    ax.set_ylabel("gain error (dB)")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
    plots.clean_axes(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plots: accuracy
# ---------------------------------------------------------------------------

def plot_gain_error(db, rows, title, dtype):
    """Signed error across a few 6.02 dB steps, and a zoom on one step at
    degree 2, where the free fit's jump is big enough to see."""
    fig, (ax, ax_z) = plt.subplots(2, 1, figsize=(10, 8.5))
    sel = (db >= -20) & (db <= 8)
    for color, (m, err, _) in zip(plots.SERIES, rows):
        ax.plot(db[sel], err[sel], color=color, lw=1.2, label=m.name if m.degree == "—" else f"{m.name} {m.degree}")
    for k in range(-3, 2):
        ax.axvline(k / dbpoly.K, color=plots.MUTED, lw=0.5, ls=":")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("dB  (dotted: 6.02 dB steps, where the exponent field changes)")
    ax.set_ylabel("gain error (dB)")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)

    edge = -1 / dbpoly.K
    z = np.linspace(edge - 0.05, edge + 0.05, 2001).astype(dtype)
    for color, step in zip(plots.SERIES[4:], ("free", "pinned")):
        m = dbpoly.gain_method(step, 2, dtype)
        level = 20 * np.log10(m.f(z).astype(np.float64))
        ax_z.plot(z, level - z.astype(np.float64), ".-", color=color, ms=2, lw=1, label=f"{m.name} 2")
    ax_z.axvline(edge, color=plots.MUTED, lw=0.8, ls=":")
    ax_z.set_xlabel(f"dB, either side of the step at {edge:.3f} dB")
    ax_z.set_ylabel("output level − dB (dB)")
    ax_z.set_title("Degree 2 at one step: the free fit jumps, the pinned one doesn't", fontsize=10)
    ax_z.legend(loc="upper left", frameon=False, fontsize=9)
    for a in (ax, ax_z):
        plots.clean_axes(a)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-d", "--degree", type=int, default=DEGREE,
                      help=f"exp2 degree for the side-by-side, ramps and compressor (log2 gets +1; default {DEGREE})")
    mode.add_argument("--compare", action="store_true", help=f"sweep degrees {COMPARE_DEGREES}")
    parser.add_argument("--eval", choices=EVAL_DTYPES, default="f32", help="precision to evaluate in")
    fit.add_argument(parser)
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--emit-cpp", action="store_true", help="print the C++ for this configuration and exit")
    args = parser.parse_args()
    fit.set_method(args.fit)
    if args.degree < 1:
        parser.error("degree must be at least 1")
    dtype = EVAL_DTYPES[args.eval]
    deg = args.degree

    if args.emit_cpp:
        if args.compare:
            parser.error("--emit-cpp emits one configuration; drop --compare")
        code = "\n".join([cppgen.bit_helpers()] + [e[1] for e in (
            cppgen.emit_db_to_gain(deg, False), cppgen.emit_db_to_gain(deg, True),
            cppgen.emit_gain_to_db(deg + 1, False), cppgen.emit_gain_to_db(deg + 1, True))])
        cppgen.print_emitted(code)
        print("// (also needs <algorithm>, <bit> and <limits>)")
        return

    report.print_header(f"dB <-> gain ({'compare' if args.compare else f'degree {deg}'})", [
        ("functions", "gain = 2^(dB K), dB = log2(x) / K, K = log2(10) / 20"),
        ("range", f"dB in [{DB_RANGE[0]:g}, {DB_RANGE[1]:g}], gains in [{GAIN_RANGE[0]:g}, {GAIN_RANGE[1]:g}],"
                  " 2M f32 inputs each; extended ranges below"),
        ("eval", args.eval),
        ("truth", "f64 10^(dB/20), 20 log10 x"),
    ])

    db, x = gain_inputs(), db_inputs()
    tg, td = np.power(10.0, db.astype(np.float64) / 20), 20 * np.log10(x.astype(np.float64))
    degrees = COMPARE_DEGREES if args.compare else [deg]
    fixed = lambda s: s in ("pow", "bits", "bits-tuned", "log10")
    gain_rows, db_rows = [], []
    for s in dbpoly.GAIN_STEPS:
        for d in ([None] if fixed(s) else degrees):
            m = dbpoly.gain_method(s, d, dtype)
            gain_rows.append((m, *gain_stats(m, db, tg)))
    for s in dbpoly.DB_STEPS:
        for d in ([None] if fixed(s) else [d + 1 for d in degrees]):
            m = dbpoly.db_method(s, d, dtype)
            db_rows.append((m, *db_stats(m, x, td)))

    print_accuracy(gain_rows, f"dB -> gain, dB in [{DB_RANGE[0]:g}, {DB_RANGE[1]:g}]:", "at 0 dB")
    print_accuracy(db_rows, f"gain -> dB, gain in [{GAIN_RANGE[0]:g}, {GAIN_RANGE[1]:g}] (log2 degree counts u P(u)):",
                   "at gain 1")
    print("  Steps: dB -> gain at each 6.02 dB (exponent field change); gain -> dB at the fold, 1.5 * 2^k.")
    print("  Power domain: RMS in dB is 10 log10(mean square) = 3.0103 log2(ms) -- no sqrt, and the same log2")
    print("  error costs half the dB of 20 log10(rms).")
    print()

    if args.compare:
        print_budget(gain_rows, db_rows)
        plotted = [r for r in gain_rows if r[0].degree in ("2", "3") and "safe" not in r[0].name or fixed_name(r[0])]
        figs = {"gain_error": plot_gain_error(db, plotted, f"dB -> gain error, evaluated in {args.eval}", dtype)}
        harness.save_figures(OUT_DIR, f"db_gain_compare_{args.eval}", figs, show=not args.no_show)
        return

    print_extended(deg, dtype)
    traces = run_ramps(deg, dtype)
    levels, comp = run_compressor(deg, dtype)

    plotted = [r for r in gain_rows if "safe" not in r[0].name and r[0].name != "pow(10, dB/20)"]
    harness.save_figures(OUT_DIR, f"db_gain_{deg}_{args.eval}", {
        "gain_error": plot_gain_error(db, plotted, f"dB -> gain error, degree {deg}, evaluated in {args.eval}", dtype),
        "ramp": plot_ramp(traces, f"{RAMPS[1][0]:g} -> {RAMPS[1][1]:g} dB over {RAMPS[1][2]:g} s: error vs exact"),
        "compressor": plot_compressor(levels, comp, f"Compressor gain error, {THRESHOLD:g} dB, {RATIO:g}:1,"
                                                    f" degree {deg} / {deg + 1}"),
    }, show=not args.no_show)


def fixed_name(m):
    return m.name.startswith("bits")


if __name__ == "__main__":
    main()
