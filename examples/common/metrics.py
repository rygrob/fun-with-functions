"""Error metrics and spectrum analysis shared by every sin(x) example.

Error views (truth is f64):

  absolute  |approx - sin(x)|
  relative  |approx - sin(x)| / |sin(x)|
  ULP       |approx - sin(x)| / ulp_f32(sin(x))   (correctly rounded <= 0.5)

Spectrum: one cycle sampled at N_FFT uniform phases, FFT'd so bin h is
harmonic h. Reported as SINAD (fundamental vs. everything else) and
ENOB = (SINAD - 1.76) / 6.02.
"""

import numpy as np

PI = np.pi
N_SAMPLES = 100_001
N_FFT = 65_536
SAMPLE_RATE = 48_000
OSC_CYCLES = 1361  # prime, so every phase lands on a distinct sample; ~996.8 Hz
EVAL_DTYPES = {"f32": np.float32, "f64": np.float64}


# ---------------------------------------------------------------------------
# Pointwise error
# ---------------------------------------------------------------------------

def abs_error(approx, truth):
    return np.abs(approx.astype(np.float64) - truth)


def rel_error(approx, truth):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(approx.astype(np.float64) - truth) / np.abs(truth)


def ulp_f32(t):
    """Gap between adjacent f32 values in the binade containing t.

    |t| = m * 2^e with m in [0.5, 1), so t lives in [2^(e-1), 2^e) where f32
    spacing is 2^(e-1-23). Subnormals bottom out at 2^-149.
    """
    _, e = np.frexp(np.abs(t))
    return np.ldexp(1.0, np.maximum(e - 24, -149))


def ulp_error_f32(approx, truth):
    """Error in units of f32 ULPs of the true value. Fractional: a correctly
    rounded f32 result scores <= 0.5."""
    return np.abs(approx.astype(np.float64) - truth) / ulp_f32(truth)


def pointwise_errors(approx, truth):
    return {
        "abs": abs_error(approx, truth),
        "rel": rel_error(approx, truth),
        "ulp": ulp_error_f32(approx, truth),
    }


def bits(err):
    """Bits of precision implied by an error: -log2(err), floored at 0."""
    return max(0.0, -np.log2(err)) if err > 0 else np.inf


def summarize(err):
    max_abs = err["abs"].max()
    max_rel = np.nanmax(err["rel"])
    return {
        "max_abs": max_abs,
        "abs_bits": bits(max_abs),
        "max_rel": max_rel,
        "rel_bits": bits(max_rel),
        "max_ulp": err["ulp"].max(),
        "mean_ulp": err["ulp"].mean(),
    }


# ---------------------------------------------------------------------------
# Spectrum
#
# Sampling one cycle at N uniform phases is the same as running an oscillator:
# the output is a function of phase only, so with K cycles in N samples
# (K coprime to N) every harmonic h lands on bin h*K mod N -- same energies,
# just shuffled. With K = 1, bin h *is* harmonic h. All the error -- including
# f32 rounding -- is periodic in phase, so it all shows up as harmonics.
# ---------------------------------------------------------------------------

# Inputs come in one of two domains:
#   radians  x in [-pi, pi],  approximating sin(x)
#   turns    x in [0, 1],     approximating sin(2 pi x) -- phase as an
#                             oscillator's accumulator holds it
DOMAINS = ("radians", "turns")


def cycles_to_input(cycles, domain):
    """Fractional cycles in [0, 1) -> f32 inputs for the domain."""
    if domain == "turns":
        return cycles.astype(np.float32)
    return (2 * PI * cycles - PI).astype(np.float32)


def sin_turns(x):
    """sin(2 pi x) in f64, exact at the zeros.

    np.sin(2 * np.pi * x) isn't: at x = 1/2 it computes sin(fl(pi)) = 1.2e-16
    rather than 0. Reduce by quarter turns first -- n = round(4x) and
    r = x - n/4 are exact for f32 inputs -- then evaluate sin or cos of the
    small remainder 2 pi r.
    """
    x = np.asarray(x, dtype=np.float64)
    n = np.rint(4 * x)
    r = 2 * PI * (x - n / 4)
    q = n.astype(np.int64) % 4
    return np.select([q == 0, q == 1, q == 2], [np.sin(r), np.cos(r), -np.sin(r)], -np.cos(r))


def true_sin(x, domain):
    """f64 truth for f32 inputs x."""
    return sin_turns(x) if domain == "turns" else np.sin(x.astype(np.float64))


def fft_phases(domain="radians"):
    """N_FFT uniform phases over one cycle, as f32 inputs. In turns these
    are k / 2^16, exactly representable; in radians they're rounded."""
    return cycles_to_input(np.arange(N_FFT) / N_FFT, domain)


def oscillator_phases(domain="radians"):
    """Phase of a naive oscillator at OSC_CYCLES cycles per N_FFT samples,
    wrapped to one period. Accumulated in f64, fed to the approximation as f32."""
    cycles = np.arange(N_FFT) * OSC_CYCLES / N_FFT
    return cycles_to_input(cycles % 1.0, domain)


def spectrum(y, fund_bin=1):
    """Per-bin levels in dBc and SINAD in dB. With fund_bin=1 (one cycle in
    the buffer), the bin index is the harmonic number."""
    Y = np.fft.rfft(np.asarray(y, dtype=np.float64))
    power = np.abs(Y) ** 2
    # rfft folds negative frequencies in, so every bin except DC and Nyquist
    # carries twice the power it shows.
    power[1:-1] *= 2
    fund = power[fund_bin]
    # Sum the other bins directly: total - fund cancels once the noise is
    # ~1e-15 of the fundamental, which is right where the f32 ceiling sits.
    noise = power[:fund_bin].sum() + power[fund_bin + 1:].sum()
    sinad = 10 * np.log10(fund / max(noise, 1e-300))
    dbc = 10 * np.log10(np.maximum(power / fund, 1e-300))
    return dbc, sinad


def enob(sinad):
    return (sinad - 1.76) / 6.02


def summarize_spectrum(y):
    """One-cycle spectrum: harmonic levels, SINAD, ENOB and the worst harmonic."""
    dbc, sinad = spectrum(y)
    worst = 2 + int(np.argmax(dbc[2:]))
    return {"dbc": dbc, "sinad": sinad, "enob": enob(sinad), "worst_h": worst, "worst_dbc": dbc[worst]}


def summarize_oscillator(y):
    """Oscillator spectrum (fundamental at bin OSC_CYCLES), keeping the waveform."""
    dbc, sinad = spectrum(y, OSC_CYCLES)
    return {"dbc": dbc, "sinad": sinad, "y": y}


# ---------------------------------------------------------------------------
# Filter cutoff (tan)
#
# A bilinear-transform filter (TPT one-pole, SVF, ...) takes its cutoff as
# g = tan(pi x), x = fc / fs in [0, 1/2). Its response at the cutoff is exact
# for any g: the filter behaves as if the cutoff were x' = atan(g) / pi. So
# an approximate g moves the cutoff, and the honest error is that move in
# cents. Near Nyquist tan's slope soaks it up: d(cents) ~ sinc(2x) * rel(g),
# so the huge relative errors next to the pole barely move the cutoff.
# ---------------------------------------------------------------------------

def tan_pi(x):
    """tan(pi x) in f64, exact at the pole.

    np.tan(np.pi * x) loses the pole: pi * x near 1/2 is off by an ulp of
    pi/2, which is the whole distance to the pole for the last f32 inputs.
    For |x| > 1/4 use tan(pi x) = 1 / tan(pi (1/2 - x)) instead -- 1/2 - |x|
    is exact for f32 inputs, so the small angle loses nothing.
    """
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    with np.errstate(divide="ignore"):
        far = np.sign(x) / np.tan(PI * (0.5 - a))
    return np.where(a <= 0.25, np.tan(PI * x), far)


def cutoff_cents(g, x):
    """Cutoff error in cents of a filter given g ~ tan(pi x). A g that isn't
    positive and finite puts the cutoff nowhere: reported as inf."""
    g = np.asarray(g, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(g) & (g > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200 * np.log2(np.arctan(np.where(ok, g, 1.0)) / (PI * x))
    return np.where(ok, np.abs(cents), np.inf)
