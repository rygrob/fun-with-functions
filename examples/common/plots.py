"""Plots shared by every example: error panels, harmonic spectrum, oscillator.

Each takes a list of harness.Result and a title; the per-example function
plot lives with the example since what's worth showing differs.
"""

import matplotlib.pyplot as plt
import numpy as np

from .metrics import N_FFT, OSC_CYCLES, PI, SAMPLE_RATE

N_BINS = 400
SPECTRUM_FLOOR_DB = -220

# Categorical series colors, assigned to results in this fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TRUTH = "#0b0b0b"
MUTED = "#8a8985"


def binned_max(x, y, n_bins=N_BINS):
    """Max of y in n_bins equal-count bins of x (x is sorted). NaNs ignored."""
    starts = np.linspace(0, len(x), n_bins, endpoint=False).astype(int)
    centers = np.add.reduceat(x.astype(np.float64), starts) / np.diff(np.append(starts, len(x)))
    return centers, np.fmax.reduceat(y, starts)


def log_binned_max(dbc, n_bins=N_BINS):
    """Max dBc in log-spaced bins of harmonic number, from harmonic 2 up.
    Low harmonics get a bin each, so they're exact."""
    edges = np.unique(np.geomspace(2, len(dbc), n_bins).astype(int))
    edges = edges[edges < len(dbc)]
    return edges, np.maximum.reduceat(dbc[2:], edges - 2)


def clean_axes(ax, which="major"):
    ax.grid(alpha=0.25, lw=0.6, which=which)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def style_axes(ax, domain="radians"):
    """Grid, spines, and an x-axis over the domain: [-pi, pi] in multiples of
    pi/2, or [0, 1] turns in quarters."""
    clean_axes(ax)
    if domain == "turns":
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "¼", "½", "¾", "1"])
    else:
        ax.set_xlim(-PI, PI)
        ax.set_xticks([-PI, -PI / 2, 0, PI / 2, PI], ["-π", "-π/2", "0", "π/2", "π"])


def _label_line(ax, y, text):
    ax.axhline(y, color=MUTED, ls=":", lw=1)
    ax.annotate(text, (0.99, y), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                color=MUTED, fontsize=9, zorder=5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))


def plot_errors(x, results, title, domain="radians"):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax_abs, ax_rel, ax_ulp = axes
    panels = [(ax_abs, "abs"), (ax_rel, "rel"), (ax_ulp, "ulp")]

    # Faint pointwise samples underneath, binned-max envelope on top.
    for color, r in zip(SERIES, results):
        for ax, key in panels:
            ax.plot(x, r.err[key], ".", color=color, ms=1, alpha=0.15, rasterized=True)
    for color, r in zip(SERIES, results):
        for ax, key in panels:
            ax.plot(*binned_max(x, r.err[key]), color=color, lw=1.8, label=r.label)

    _label_line(ax_rel, 2.0**-24, "f32 rounding (2⁻²⁴)")
    _label_line(ax_ulp, 0.5, "0.5 ULP (correctly rounded)")

    ax_abs.set_yscale("log")
    ax_rel.set_yscale("log")
    # Linear in [0, 1] ULP so the 0-0.5 region stays readable, log above.
    ax_ulp.set_yscale("symlog", linthresh=1)
    ax_ulp.set_ylim(bottom=0)

    ax_abs.set_ylabel("absolute error")
    ax_rel.set_ylabel("relative error")
    ax_ulp.set_ylabel("error (f32 ULPs)")
    ax_ulp.set_xlabel("x (turns)" if domain == "turns" else "x")
    fig.suptitle(title)
    for ax in axes:
        style_axes(ax, domain)
    fig.tight_layout()
    if len(results) > 1:
        fig.subplots_adjust(top=0.92)
        handles, labels = ax_abs.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
                   frameon=False, ncol=len(results))
    return fig


def plot_spectrum(results, ref, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    h = np.arange(len(ref["dbc"]))
    ax.plot(h[2:], ref["dbc"][2:], ".", color=MUTED, ms=1, alpha=0.3, rasterized=True)
    ax.plot(*log_binned_max(ref["dbc"]), color=MUTED, lw=1.5,
            label=f"f32(sin) reference, SINAD {ref['sinad']:.1f} dB")
    for color, r in zip(SERIES, results):
        ax.plot(h[2:], r.spec["dbc"][2:], ".", color=color, ms=1, alpha=0.15, rasterized=True)
    for color, r in zip(SERIES, results):
        ax.plot(*log_binned_max(r.spec["dbc"]), color=color, lw=1.8,
                label=f"{r.label}, SINAD {r.spec['sinad']:.1f} dB")
    ax.set_xscale("log")
    ax.set_xlim(2, len(ref["dbc"]))
    ax.set_ylim(SPECTRUM_FLOOR_DB, 0)
    ax.set_xlabel("harmonic number")
    ax.set_ylabel("level (dBc)")
    ax.set_title(f"{title} ({N_FFT:,}-point FFT)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    clean_axes(ax, which="both")
    fig.tight_layout()
    return fig


def plot_oscillator(results, ref_osc, title):
    """Waveform and spectrum of an actual ~1 kHz oscillator at 48 kHz. Same
    energies as plot_spectrum, but harmonics past Nyquist fold back down.
    One spectrum panel per result -- overlaid line spectra are unreadable."""
    f0 = OSC_CYCLES * SAMPLE_RATE / N_FFT
    n = len(results)
    fig, axes = plt.subplots(1 + n, 1, figsize=(10, 3.2 + 2.6 * n),
                             gridspec_kw={"height_ratios": [1.2] + [1] * n}, layout="constrained")
    ax_t, spec_axes = axes[0], axes[1:]

    n_show = int(3 * N_FFT / OSC_CYCLES)
    t_ms = np.arange(n_show) / SAMPLE_RATE * 1e3
    freqs = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)

    for ax, color, r in zip(spec_axes, SERIES, results):
        ax_t.plot(t_ms, r.osc["y"][:n_show], color=color, lw=1.5, label=r.label)
        ax.plot(freqs[1:], ref_osc["dbc"][1:], color=MUTED, lw=0.6, alpha=0.6)
        ax.plot(freqs[1:], r.osc["dbc"][1:], color=color, lw=0.6)
        ax.annotate(f"{r.label}   SINAD {r.osc['sinad']:.1f} dB", (0.99, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9, zorder=5,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
        ax.set_xscale("log")
        ax.set_xlim(20, SAMPLE_RATE / 2)
        ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                      ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
        ax.set_ylim(SPECTRUM_FLOOR_DB, 10)
        ax.set_ylabel("dBc")

    spec_axes[0].set_title(f"{N_FFT:,}-point FFT, gray = f32(sin) reference "
                           f"(SINAD {ref_osc['sinad']:.1f} dB); harmonics above "
                           f"{SAMPLE_RATE // 2:,} Hz alias back down", fontsize=10)
    spec_axes[-1].set_xlabel("frequency (Hz)")
    for ax in spec_axes[:-1]:
        ax.tick_params(labelbottom=False)

    ax_t.set_xlim(0, t_ms[-1])
    ax_t.set_xlabel("time (ms)")
    ax_t.set_ylabel("output")
    ax_t.set_title(f"{title}, f0 = {f0:.1f} Hz at {SAMPLE_RATE / 1000:g} kHz")
    if n > 1:
        ax_t.legend(loc="upper right", frameon=False, fontsize=9, ncol=n)

    for ax in axes:
        clean_axes(ax)
    return fig


def near_pi_inputs(n_ulps=24):
    """Every f32 from n_ulps below f32(pi) up to f32(pi), the top of the range."""
    xs = [np.float32(PI)]
    for _ in range(n_ulps):
        xs.append(np.nextafter(xs[-1], np.float32(0)))
    return np.array(xs[::-1], dtype=np.float32)


def plot_near_pi(ax, series):
    """sin(x) and approximations at the last few f32 inputs below pi.

    series: list of (label, color, f). Values are shown in units of 1e-7 so
    the f32(pi) vs. true-pi offset (~8.7e-8) is visible.
    """
    x = near_pi_inputs()
    k = np.arange(len(x)) - (len(x) - 1)  # ULPs relative to f32(pi)
    ax.plot(k, np.sin(x.astype(np.float64)) * 1e7, "o-", color=TRUTH, lw=1.5, ms=4, label="sin(x)")
    for label, color, f in series:
        ax.plot(k, f(x).astype(np.float64) * 1e7, "o--", color=color, lw=1.5, ms=4, label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("x, in f32 steps below f32(π) = 3.14159274")
    ax.set_ylabel("y  (× 10⁻⁷)")
    ax.set_title("Last f32 inputs below π: where does the zero land?", fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    clean_axes(ax)


def binned_minmax(x, y, n_bins=N_BINS):
    """(centers, min, max) of y in n_bins equal-count bins of sorted x."""
    starts = np.linspace(0, len(x), n_bins, endpoint=False).astype(int)
    centers = np.add.reduceat(x.astype(np.float64), starts) / np.diff(np.append(starts, len(x)))
    return centers, np.fmin.reduceat(y, starts), np.fmax.reduceat(y, starts)


def plot_signed_error(x, fit_series, shipped_series, title, domain="radians", levels=None):
    """Signed error on a linear axis, so equioscillation is visible.

    fit_series:     [(label, color, signed relative error)] from f64 evaluation
                    -- the fit alone; smooth, drawn as lines
    shipped_series: [(label, color, signed error in ULPs)] from f32 evaluation
                    -- rounding makes these noise, so each is a min/max band
                    per bin with the bin mean as a line
    levels:         optional ±E to mark on the fit panel
    """
    fig, (ax_fit, ax_f32) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for label, color, e in fit_series:
        ax_fit.plot(x, e, color=color, lw=1.8, label=label)
    if levels is not None:
        for lv in (levels, -levels):
            ax_fit.axhline(lv, color=TRUTH, ls=":", lw=1)
        ax_fit.annotate(f"±E = {abs(levels):.2e}", (0.99, abs(levels)), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=9, color=TRUTH, zorder=5,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax_fit.axhline(0, color=MUTED, lw=0.8)
    ax_fit.set_ylabel("relative error, f64 eval (fit)")
    ax_fit.legend(loc="lower center", frameon=False, fontsize=9, ncol=len(fit_series))

    for label, color, e in shipped_series:
        c, lo, hi = binned_minmax(x, e)
        ax_f32.fill_between(c, lo, hi, color=color, alpha=0.25, lw=0)
        ax_f32.plot(c, lo, color=color, lw=1.0)
        ax_f32.plot(c, hi, color=color, lw=1.0, label=f"{label} (max {np.nanmax(np.abs(e)):.2f})")
    ax_f32.axhline(0, color=MUTED, lw=0.8)
    ax_f32.set_ylabel("error in f32 ULPs, f32 eval (shipped)")
    ax_f32.set_xlabel("x (turns)" if domain == "turns" else "x")
    ax_f32.legend(loc="lower center", frameon=False, fontsize=9, ncol=len(shipped_series))

    fig.suptitle(title)
    for ax in (ax_fit, ax_f32):
        style_axes(ax, domain)
    fig.tight_layout()
    return fig
