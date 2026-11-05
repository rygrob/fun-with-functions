"""Hand-rolled Remez exchange: the minimax polynomial for a weighted error.

Find P of degree n - 1 (n coefficients) on [a, b] minimizing

    max |w(v) (f(v) - P(v))|

The optimum *equioscillates*: the weighted error reaches ±E at n + 1 points,
alternating in sign (Chebyshev's alternation theorem). Remez finds those
points by iterating:

  1. reference: n + 1 points, initially Chebyshev extrema on [a, b]
  2. solve      P(v_i) + (-1)^i E / w(v_i) = f(v_i)  for P and the level E
                -- forces the error to alternate ±E on the reference
  3. extrema    find where the actual weighted error peaks (dense grid,
                one peak per sign-run, alternating)
  4. exchange   those peaks become the new reference; stop when the largest
                peak is within tol of |E| (the grid locates peaks only to
                about its spacing, so tol much below 1e-6 isn't reachable)

Everything is f64. The polynomial is solved in s = (v - a) / (b - a) on
[0, 1] to keep the system well conditioned, then converted back to
monomials in v.
"""

import numpy as np


def _extrema(err, grid, n_ref):
    """n_ref alternating peaks of err: the largest |err| in each run of one
    sign, then trimmed from the ends (keeping the global max) to n_ref."""
    sign = np.sign(err)
    sign[sign == 0] = 1
    runs = np.flatnonzero(np.diff(sign)) + 1
    starts = np.concatenate([[0], runs])
    ends = np.concatenate([runs, [len(err)]])
    peaks = [s + int(np.argmax(np.abs(err[s:e]))) for s, e in zip(starts, ends)]
    while len(peaks) > n_ref:
        # Drop whichever end has the smaller peak; never drop the global max.
        if abs(err[peaks[0]]) < abs(err[peaks[-1]]):
            peaks.pop(0)
        else:
            peaks.pop()
    return np.array(peaks)


def remez(f, w, a, b, n, grid_size=20001, tol=1e-6, max_iter=50):
    """Minimax polynomial coefficients (monomials in v, low order first) for
    max |w (f - P)| on [a, b] with n coefficients.

    Returns (coeffs, E, history): E is the levelled error, history the
    (|E|, max |err|) pair from each iteration -- the gap closing is the
    convergence to show on a slide.
    """
    s = np.linspace(0.0, 1.0, grid_size)
    v = a + (b - a) * s
    fv, wv = f(v), w(v)

    ref = np.round((1 - np.cos(np.pi * np.arange(n + 1) / n)) / 2 * (grid_size - 1)).astype(int)
    history = []
    for _ in range(max_iter):
        A = np.zeros((n + 1, n + 1))
        A[:, :n] = s[ref, None] ** np.arange(n)
        A[:, n] = (-1.0) ** np.arange(n + 1) / wv[ref]
        sol = np.linalg.solve(A, fv[ref])
        c, E = sol[:n], sol[n]

        err = wv * (fv - np.polynomial.polynomial.polyval(s, c))
        history.append((abs(E), np.abs(err).max()))
        if np.abs(err).max() - abs(E) <= tol * abs(E):
            break
        # At high degree the grid can make the reference cycle between
        # near-identical sets; once the level stops moving we're done.
        if len(history) > 1 and abs(history[-1][0] - history[-2][0]) <= tol * abs(E):
            break
        new_ref = _extrema(err, s, n + 1)
        if len(new_ref) < n + 1 or np.array_equal(new_ref, ref):
            break
        ref = new_ref

    # s = (v - a) / (b - a)  ->  monomials in v.
    poly_s = np.polynomial.Polynomial(c)
    poly_v = poly_s(np.polynomial.Polynomial([-a / (b - a), 1 / (b - a)]))
    return poly_v.coef[:n], E, history
