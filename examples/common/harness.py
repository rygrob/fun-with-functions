"""Run an approximation through the full error suite.

An example builds a Context once, then calls evaluate() for each variant
(degree, table size, ...) with a function f(x) -> y that takes f32 inputs.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import metrics
from .metrics import PI, N_SAMPLES


@dataclass
class Context:
    domain: str             # "radians" or "turns", see metrics.DOMAINS
    x: np.ndarray           # f32 sample points over the domain's range
    truth: np.ndarray       # f64 sin(x) or sin(2 pi x)
    phases: np.ndarray      # one-cycle FFT phases
    osc_phases: np.ndarray  # oscillator phases
    ref: dict               # one-cycle spectrum of f32(sin)
    ref_osc: dict           # oscillator spectrum of f32(sin)


@dataclass
class Result:
    key: object       # the swept parameter (degree, table size, ...)
    label: str        # legend text, e.g. "degree 7" or "N = 1024"
    approx: np.ndarray
    err: dict
    stats: dict
    spec: dict
    osc: dict


def make_context(domain="radians"):
    # Inputs are always f32 values so f32 and f64 runs see identical x.
    lo, hi = (0.0, 1.0) if domain == "turns" else (-PI, PI)
    x = np.linspace(lo, hi, N_SAMPLES, dtype=np.float32)
    phases = metrics.fft_phases(domain)
    osc_phases = metrics.oscillator_phases(domain)
    f32_sin = lambda p: metrics.true_sin(p, domain).astype(np.float32)
    return Context(
        domain=domain,
        x=x,
        truth=metrics.true_sin(x, domain),
        phases=phases,
        osc_phases=osc_phases,
        ref=metrics.summarize_spectrum(f32_sin(phases)),
        ref_osc=metrics.summarize_oscillator(f32_sin(osc_phases)),
    )


def evaluate(fn, key, label, ctx):
    approx = fn(ctx.x)
    err = metrics.pointwise_errors(approx, ctx.truth)
    return Result(
        key=key,
        label=label,
        approx=approx,
        err=err,
        stats=metrics.summarize(err),
        spec=metrics.summarize_spectrum(fn(ctx.phases)),
        osc=metrics.summarize_oscillator(fn(ctx.osc_phases)),
    )


def save_figures(out_dir, prefix, figs, show):
    """figs maps a filename suffix ("" for the main plot) to a figure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for suffix, fig in figs.items():
        fig.savefig(out_dir / f"{prefix}{'_' + suffix if suffix else ''}.png", dpi=130)
    print(f"Saved plots to {out_dir}/")
    if show:
        plt.show()
