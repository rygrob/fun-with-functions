"""Piecewise tanh for a feedback loop: segments of width h on [0, 9], each a
degree-d polynomial in t = a - center, fit by Chebyshev interpolation (near
minimax). Reports the max abs error (f64 evaluation) per (h, d)."""
import numpy as np

def fit(h, d):
    edges = np.arange(0.0, 9.0 + 1e-12, h)
    segs = []
    worst = 0.0
    for lo in edges[:-1]:
        hi = lo + h
        c = (lo + hi) / 2
        # Chebyshev nodes on the segment, fit in t = a - c
        k = np.arange(4 * (d + 1))
        a = c + (h / 2) * np.cos(np.pi * (k + 0.5) / len(k))
        coef = np.polynomial.chebyshev.chebfit((a - c) / (h / 2), np.tanh(a), d)
        mono = np.polynomial.chebyshev.cheb2poly(coef)            # in u = t / (h/2)
        mono = mono / (h / 2) ** np.arange(d + 1)                 # in t
        aa = np.linspace(lo, hi, 2001)
        err = np.max(np.abs(np.polyval(mono[::-1], aa - c) - np.tanh(aa)))
        worst = max(worst, err)
        segs.append((c, mono))
    return segs, worst

for h in (1.0, 0.5, 0.25):
    row = []
    for d in range(3, 9):
        _, w = fit(h, d)
        row.append(f"d{d} {20*np.log10(w):7.1f}")
    print(f"h={h:<5} " + "  ".join(row))

# The C++ table: h = 0.5, d = 5, one row per segment: center, c0..c5.
segs, w = fit(0.5, 5)
rows = [f"    {{{', '.join(f'{v:.9g}f' for v in (c, *m))}}}," for c, m in segs]
open("pw_tanh_table.inc", "w").write("\n".join(rows) + "\n")
print(f"wrote {len(rows)} segments, f64 error {20*np.log10(w):.1f} dB")
