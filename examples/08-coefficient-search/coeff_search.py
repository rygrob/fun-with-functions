"""Brute-force f32 coefficient search for the turns form of sin(2 pi x).

Remez (07) gives the best *real-valued* coefficients. But the shipped code
stores them as f32 and rounds at every step, so the best *f32* coefficients
are usually a few f32 steps away -- the "20 hour search" in
https://mooooo.ooo/chebyshev-sine-approximation/ (4.90 -> 4.58 ULP).

This is a hand-rolled version of that search:

  1. start   Remez (or Chebyshev) coefficients rounded to f32
  2. model   evaluate exactly as the compiled code does, in f32, with or
             without FMA per Horner step (clang on arm64 fuses by default;
             see notes/fp-contraction.md) -- the model reproduces the C++
             bench's exhaustive worst case to the last digit
  3. objective  max ULP over a *hard set* rather than all 1e9 floats:
             random floats (log-uniform, like the exhaustive check), uniform
             x, and the worst inputs found so far. Ties broken by the mean of
             the 64 worst, so the search can move along plateaus.
  4. search  coordinate descent: nudge one coefficient by ±1..R f32 steps,
             keep improvements; then random two-coefficient kicks
  5. refresh after each sweep, sample a fresh batch, add its worst inputs
             to the hard set (a discrete cousin of Remez's exchange step)

The hard-set max is a lower bound on the true max. With --bench-rounds K,
each round closes the loop with the C++ bench: write the coefficients,
regenerate + rebuild the bench, run the exhaustive turns check, and pull the
worst inputs it found (plus a dense neighbourhood of floats around each)
into the hard set. The coefficients with the best *exhaustive* max win.
Needs a configured bench (cmake -S bench -B bench/build).

Usage:
    python coeff_search.py                   # degree 13, FMA, ~1 minute
    python coeff_search.py -d 11             # another degree
    python coeff_search.py --no-fma          # model separate mul + add
    python coeff_search.py --bench-rounds 5  # exhaustive C++ check between rounds
    python coeff_search.py --bench-rounds 20 --minutes 3   # deep: 20 rounds, 3 min each
    python coeff_search.py --start cheb      # start from Chebyshev instead
    python coeff_search.py --emit-cpp        # print the tuned C++ and exit
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for ../common
from common import cppgen, fit, harness, plots, report, sinpoly
from common.metrics import sin_turns, ulp_f32

DEGREE = 13
SWEEPS = 12
RADIUS = 3
HARD_RANDOM = 1_000_000   # log-uniform random floats in the hard set
HARD_UNIFORM = 1_000_000  # uniform x in [0, 1)
REFRESH_SAMPLES = 2_000_000
REFRESH_KEEP = 4_000
KICKS = 40                # random two-coefficient moves per sweep

NEIGHBOURS = 32           # floats either side of each exhaustive worst input

DOMAIN = "turns"
OUT_DIR = Path(__file__).parent / "out"
TUNED_DIR = Path(__file__).parent / "tuned"
REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench" / "build" / "adc_bench"

F32, F64 = np.float32, np.float64
FLT_MIN_BITS = np.float32(np.finfo(np.float32).tiny).view(np.uint32)
ONE_BITS = np.float32(1.0).view(np.uint32)


# ---------------------------------------------------------------------------
# The f32 model of the shipped code
# ---------------------------------------------------------------------------

def eval_turns_f32(x, c, fma):
    """-u * x * (x - 1) * P(u^2), u = x - 1/2, every op in f32 exactly as the
    emitted C++ (cppgen.turns) compiles. fma=True fuses each Horner step:
    the f32 x f32 product is exact in f64, then one rounding of the sum (an
    f64 add then f32 cast double-rounds only in rare halfway cases)."""
    x = np.asarray(x, F32)
    u = x - F32(0.5)
    u2 = u * u
    p = np.full_like(x, c[-1])
    for ck in c[-2::-1]:
        if fma:
            p = (p.astype(F64) * u2.astype(F64) + F64(ck)).astype(F32)
        else:
            p = p * u2 + ck
    return ((-u * x) * (x - F32(1))) * p


def random_floats(rng, n):
    """Log-uniform over normal floats in [FLT_MIN, 1]: uniform random bit
    patterns, so the mix matches the exhaustive check's."""
    bits = rng.integers(FLT_MIN_BITS, ONE_BITS, size=n, endpoint=True, dtype=np.uint32)
    return bits.view(F32)


class HardSet:
    """Inputs the objective is measured on, with truth and ULP size cached."""

    def __init__(self, rng):
        x = np.concatenate([random_floats(rng, HARD_RANDOM), rng.random(HARD_UNIFORM).astype(F32)])
        self.set(np.unique(x[x > 0]))

    def set(self, x):
        self.x = x
        self.truth = sin_turns(x)
        self.ulp = ulp_f32(self.truth)

    def errors(self, c, fma):
        return np.abs(eval_turns_f32(self.x, c, fma).astype(F64) - self.truth) / self.ulp

    def objective(self, c, fma):
        """(max ULP, mean of the 64 worst) -- compared as a tuple."""
        e = self.errors(c, fma)
        top = np.partition(e, -64)[-64:]
        return (float(top.max()), float(top.mean()))

    def add(self, x):
        self.set(np.unique(np.concatenate([self.x, np.asarray(x, F32)])))

    def refresh(self, rng, c, fma):
        """Add the worst inputs from a fresh random batch."""
        x = np.concatenate([random_floats(rng, REFRESH_SAMPLES // 2), rng.random(REFRESH_SAMPLES // 2).astype(F32)])
        x = x[x > 0]
        t = sin_turns(x)
        e = np.abs(eval_turns_f32(x, c, fma).astype(F64) - t) / ulp_f32(t)
        worst = x[np.argpartition(e, -REFRESH_KEEP)[-REFRESH_KEEP:]]
        self.set(np.unique(np.concatenate([self.x, worst])))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def nudge(v, steps):
    """v moved by `steps` f32 values (through zero if need be)."""
    i = np.array([v], F32).view(np.int32)[0]
    ordered = np.int64(i) if i >= 0 else -np.int64(i & 0x7FFFFFFF)
    ordered += steps
    bits = ordered if ordered >= 0 else (-ordered) | 0x80000000
    return np.array([bits], np.int64).astype(np.uint32).view(F32)[0]


def search(c0, fma, hard, rng, sweeps, radius, deadline):
    c = np.array(c0, F32)
    best = hard.objective(c, fma)
    history = [(0, best[0], "start")]
    evals = 0
    r = radius
    sweep = 0
    while sweep < sweeps or (deadline and time.time() < deadline):
        sweep += 1
        improved = False
        for k in rng.permutation(len(c)):
            for step in [s for d in range(1, r + 1) for s in (d, -d)]:
                cand = c.copy()
                cand[k] = nudge(c[k], step)
                o = hard.objective(cand, fma)
                evals += 1
                if o < best:
                    best, c, improved = o, cand, True
        for _ in range(KICKS):
            i, j = rng.choice(len(c), 2, replace=False)
            cand = c.copy()
            cand[i] = nudge(c[i], int(rng.integers(-r, r + 1)))
            cand[j] = nudge(c[j], int(rng.integers(-r, r + 1)))
            o = hard.objective(cand, fma)
            evals += 1
            if o < best:
                best, c, improved = o, cand, True

        n_before = len(hard.x)
        hard.refresh(rng, c, fma)
        best = hard.objective(c, fma)  # harder set: re-measure
        history.append((evals, best[0], f"sweep {sweep}"))
        print(f"  sweep {sweep:3d}: max {best[0]:.4f} ULP  top-64 mean {best[1]:.4f}  "
              f"radius {r}  hard set {n_before:,} -> {len(hard.x):,}  ({evals} evals)")
        if not improved:
            if deadline and time.time() < deadline:
                r += 1  # stuck: look further out
            elif not deadline:
                break
    return c, history


# ---------------------------------------------------------------------------
# Exhaustive check through the C++ bench
# ---------------------------------------------------------------------------

def tuned_path(degree, fma):
    """Interpolation-started searches save under tuned/interpolation/, so they
    never clobber the default results the bench picks up from tuned/*.json."""
    d = TUNED_DIR if fit.get_method() == "projection" else TUNED_DIR / fit.get_method()
    return d / f"turns_{degree}_{'fma' if fma else 'nofma'}.json"


def save(path, deg, fma, start, c, c0, note):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "degree": deg,
        "form": "turns",
        "fma": fma,
        "start": start,
        "coefficients": [float(v) for v in c],
        "start_coefficients": [float(v) for v in c0],
        "note": note,
    }, indent=2) + "\n")


def bench_check(deg, fma, dump_n=2000):
    """Regenerate + rebuild the bench with the saved coefficients, run the
    exhaustive turns check, return (max ULP, worst inputs)."""
    if not BENCH.exists():
        sys.exit("bench not built: cmake -S bench -B bench/build && cmake --build bench/build")
    dump_dir = OUT_DIR / "worst"
    dump_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(REPO / "bench" / "generate.py")], check=True, capture_output=True)
    subprocess.run(["cmake", "--build", str(BENCH.parent)], check=True, capture_output=True)
    subprocess.run([str(BENCH), "ulp", "f32", "--domain", "turns", "--dump-worst", str(dump_n),
                    "--dump-dir", str(dump_dir)], check=True, capture_output=True)
    rows = np.loadtxt(dump_dir / f"worst_tuned_turns_sin_{deg}_{'fma' if fma else 'nofma'}.txt", ndmin=2)
    return float(rows[0, 1]), rows[:, 0].astype(F32)


def neighbourhood(x, n=NEIGHBOURS):
    """x and the n floats either side of each -- worst cases cluster."""
    bits = np.unique(np.asarray(x, F32)).view(np.int32)
    offsets = np.arange(-n, n + 1, dtype=np.int32)
    out = (bits[:, None] + offsets[None, :]).ravel().view(F32)
    return out[(out > 0) & (out <= 1)]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def plot_history(history, start_label):
    fig, ax = plt.subplots(figsize=(10, 4))
    evals = [h[0] for h in history]
    ulps = [h[1] for h in history]
    ax.plot(evals, ulps, "o-", color=plots.SERIES[0], lw=1.8, ms=4)
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("max ULP on hard set")
    ax.set_title(f"f32 coefficient search from {start_label} (hard set grows each sweep)")
    plots.clean_axes(ax)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-d", "--degree", type=int, default=DEGREE, help=f"total polynomial degree (default {DEGREE})")
    parser.add_argument("--start", choices=["remez", "cheb"], default="remez", help="starting coefficients")
    parser.add_argument("--no-fma", action="store_true", help="model separate multiply and add (default: FMA Horner)")
    parser.add_argument("--sweeps", type=int, default=SWEEPS, help=f"coordinate-descent sweeps (default {SWEEPS})")
    parser.add_argument("--radius", type=int, default=RADIUS, help=f"f32 steps tried per coefficient (default {RADIUS})")
    parser.add_argument("--minutes", type=float, default=0, help="per round: keep going for this long, widening the radius when stuck")
    parser.add_argument("--bench-rounds", type=int, default=0, help="rounds of search + exhaustive C++ check (needs bench/build)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--emit-cpp", action="store_true", help="print the C++ for the saved tuned coefficients and exit")
    fit.add_argument(parser)
    args = parser.parse_args()
    fit.set_method(args.fit)

    deg = args.degree if args.degree % 2 else args.degree - 1
    fma = not args.no_fma
    path = tuned_path(deg, fma)

    if args.emit_cpp:
        if not path.exists():
            parser.error(f"no tuned coefficients at {path}; run the search first")
        cppgen.print_emitted(cppgen.emit_tuned_turns(path)[1])
        return

    rng = np.random.default_rng(args.seed)
    c0 = (sinpoly.fit_remez_turns(deg) if args.start == "remez" else sinpoly.fit_cheb_turns(deg)).astype(F32)
    start_label = "Remez (relative)" if args.start == "remez" else "Chebyshev"

    report.print_header(f"f32 coefficient search, degree {deg}", [
        ("function", "sin(2πx)"),
        ("range", "[0, 1] turns"),
        ("form", "-u * x * (x - 1) * P(u^2), u = x - 1/2"),
        ("model", "f32, FMA Horner" if fma else "f32, separate mul + add"),
        ("start", f"{start_label}, rounded to f32"),
        ("budget", (f"{args.minutes:g} min" if args.minutes else f"up to {args.sweeps} sweeps, radius {args.radius}")
                   + (f" per round, {args.bench_rounds} rounds with exhaustive C++ check" if args.bench_rounds else "")),
    ])

    hard = HardSet(rng)
    t0 = time.time()
    if not args.bench_rounds:
        print(f"Hard set: {len(hard.x):,} inputs. Searching:")
        deadline = t0 + 60 * args.minutes if args.minutes else None
        c, history = search(c0, fma, hard, rng, args.sweeps, args.radius, deadline)
    else:
        # Exhaustive max of the start, and its worst inputs, before searching.
        save(path, deg, fma, args.start, c0, c0, "start")
        best_ulp, worst = bench_check(deg, fma)
        print(f"Round 0 (start): exhaustive max {best_ulp:.4f} ULP")
        hard.add(neighbourhood(worst))
        c, best_c, history = c0.copy(), c0.copy(), []
        for rnd in range(1, args.bench_rounds + 1):
            print(f"Round {rnd}: hard set {len(hard.x):,} inputs")
            deadline = time.time() + 60 * args.minutes if args.minutes else None
            c, h = search(c, fma, hard, rng, args.sweeps, args.radius, deadline)
            history += h
            save(path, deg, fma, args.start, c, c0, f"round {rnd}")
            ulp, worst = bench_check(deg, fma)
            mark = "  (best so far)" if ulp < best_ulp else ""
            print(f"Round {rnd}: exhaustive max {ulp:.4f} ULP{mark}")
            if ulp < best_ulp:
                best_ulp, best_c = ulp, c.copy()
            hard.add(neighbourhood(worst))
        c = best_c
        print(f"Best exhaustive max: {best_ulp:.4f} ULP")
    print(f"  ({time.time() - t0:.0f} s)")
    print()

    print(f"Coefficients (f32), {start_label} -> tuned, in f32 steps:")
    def ordered(v):  # f32 bits as an integer that increases with the value
        i = int(np.array([v], F32).view(np.int32)[0])
        return i if i >= 0 else -(i & 0x7FFFFFFF)

    for k, (a, b) in enumerate(zip(c0, c)):
        steps = ordered(b) - ordered(a)
        print(f"  P u^{2 * k:<2d}  {float(a): .8e}  ->  {float(b): .8e}   ({steps:+d})")
    print()

    # Fresh check, independent of the hard set the search saw.
    check = HardSet(np.random.default_rng(args.seed + 1000))
    o0, o1 = check.objective(c0, fma), check.objective(c, fma)
    print(f"On a fresh {len(check.x):,}-input set: max {o0[0]:.4f} -> {o1[0]:.4f} ULP")
    print("Exhaustive check: python bench/generate.py && cmake --build bench/build && bench/build/adc_bench ulp f32")
    print()

    save(path, deg, fma, args.start, c, c0,
         f"best of {args.bench_rounds} bench rounds" if args.bench_rounds else f"hard-set max {history[-1][1]:.4f} ULP")
    print(f"Wrote {path.relative_to(REPO)}")
    if args.bench_rounds:
        bench_check(deg, fma)  # leave the bench built with the winner

    ctx = harness.make_context(DOMAIN)
    results = [
        harness.evaluate(lambda x: eval_turns_f32(x, c0, fma), "start", start_label, ctx),
        harness.evaluate(lambda x: eval_turns_f32(x, c, fma), "tuned", "tuned", ctx),
    ]
    model = "FMA" if fma else "no FMA"
    x = ctx.x[(ctx.x > 0) & (ctx.truth != 0)]
    truth = sin_turns(x)
    fit_rows = [(lab, col, (sinpoly.eval_turns(x, cc.astype(F64), F64) - truth) / truth)
                for lab, col, cc in ((start_label, plots.MUTED, c0), ("tuned", plots.SERIES[0], c))]
    f32_rows = [(lab, col, (eval_turns_f32(x, cc, fma).astype(F64) - truth) / ulp_f32(truth))
                for lab, col, cc in ((start_label, plots.MUTED, c0), ("tuned", plots.SERIES[0], c))]
    signed = plots.plot_signed_error(
        x, fit_rows, f32_rows,
        f"Signed error, degree {deg}: f32 coefficients in f64 (fit) vs. f32 ({model})", DOMAIN)
    harness.save_figures(OUT_DIR, f"coeff_search_{deg}_{'fma' if fma else 'nofma'}", {
        "signed": signed,
        "history": plot_history(history, start_label),
        "error": plots.plot_errors(ctx.x, results, f"Turns degree {deg}, f32 ({model}): {start_label} vs. tuned", DOMAIN),
    }, show=not args.no_show)


if __name__ == "__main__":
    main()
