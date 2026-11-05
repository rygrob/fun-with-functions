"""Rational Remez exchange for tanh on [0, 9]: tanh x ~ x P(s) / Q(s), s = x^2,
deg P = m, deg Q = n, Q(0) = 1, minimax absolute error. Hand-rolled.

Each step solves, at the m + n + 2 reference points x_i,
    x_i P(s_i) - f_i Q(s_i) - (-1)^i E Qprev(s_i) = 0
(linear: the error term's Q is the previous iterate's), iterates that to a
fixed point, then moves the reference to the error's alternating extrema.
Powers are in u = s / 81 for conditioning; the result is converted to s.
Run from this directory: writes rational_coeffs.inc.
"""
import numpy as np

LO, HI = 1e-6, 9.0
SCALE = HI * HI


def evaluate(p, q, x):
    u = x * x / SCALE
    return x * np.polyval(p[::-1], u) / np.polyval(q[::-1], u)


def solve(m, n, x, qprev):
    """One linear solve: coefficients p (m+1), q (n+1, q0 = 1), and E."""
    u = x * x / SCALE
    f = np.tanh(x)
    sign = (-1.0) ** np.arange(len(x))
    cols = [x * u**k for k in range(m + 1)]          # P terms
    cols += [-f * u**k for k in range(1, n + 1)]     # Q terms past q0
    cols += [-sign * np.polyval(qprev[::-1], u)]     # E
    A = np.stack(cols, axis=1)
    b = f                                            # q0 = 1 moved across
    sol = np.linalg.solve(A, b)
    p = sol[: m + 1]
    q = np.concatenate([[1.0], sol[m + 1 : m + 1 + n]])
    return p, q, sol[-1]


def extrema(err, grid, k):
    """k alternating extrema of err on grid: local extremes, merged by sign,
    then the largest k in a row (dropping from the ends)."""
    idx = [0] + [i for i in range(1, len(err) - 1)
                 if (err[i] - err[i - 1]) * (err[i + 1] - err[i]) <= 0] + [len(err) - 1]
    pts = []
    for i in idx:                                    # one point per sign run: the biggest
        if pts and np.sign(err[i]) == np.sign(err[pts[-1]]):
            if abs(err[i]) > abs(err[pts[-1]]):
                pts[-1] = i
        else:
            pts.append(i)
    while len(pts) > k:                              # drop the smaller end
        if abs(err[pts[0]]) < abs(err[pts[-1]]):
            pts.pop(0)
        else:
            pts.pop()
    return grid[pts]


def floats(c):
    """C++ float literals: 1 -> 1.0f, 0.12 -> 0.12f."""
    text = [f"{v:.9g}" for v in c]
    return ", ".join(t + ("" if "." in t or "e" in t else ".0") + "f" for t in text)


def remez(m, n, iters=40):
    k = m + n + 2
    x = (LO + HI) / 2 - (HI - LO) / 2 * np.cos(np.pi * np.arange(k) / (k - 1))
    q = np.ones(n + 1)
    grid = np.concatenate([np.geomspace(LO, 0.5, 20000), np.linspace(0.5, HI, 200000)])
    for _ in range(iters):
        for _ in range(20):                          # fixed point on the error's Q
            p, q_new, E = solve(m, n, x, q)
            if np.max(np.abs(q_new - q)) < 1e-14:
                q = q_new
                break
            q = q_new
        err = evaluate(p, q, grid) - np.tanh(grid)
        if len(extrema(err, grid, k)) < k:
            break
        x = extrema(err, grid, k)
    worst = np.max(np.abs(err))
    # back to coefficients in s: u^k = s^k / 81^k
    return p / SCALE ** np.arange(m + 1), q / SCALE ** np.arange(n + 1), abs(E), worst


if __name__ == "__main__":
    for m, n in [(3, 3), (4, 3), (4, 4), (5, 4), (5, 5)]:
        p, q, E, worst = remez(m, n)
        print(f"[{2*m+1}/{2*n}]  levelled |E| {20*np.log10(E):7.1f} dB   max on grid {20*np.log10(worst):7.1f} dB")

    # The C++ coefficients (in s) for the three fits tanh_feedback.cpp uses.
    out = []
    for m, n in [(3, 3), (4, 3), (4, 4)]:
        p, q, E, worst = remez(m, n)
        assert np.all(np.polyval(q[::-1], np.linspace(0, SCALE, 100001)) > 0), "pole in [0, 9]"
        out.append(f"// [{2*m+1}/{2*n}] minimax on [0, 9]: {20*np.log10(worst):.1f} dB (f64)")
        out.append(f"inline constexpr float rat_p_{2*m+1}_{2*n}[] = {{{floats(p)}}};")
        out.append(f"inline constexpr float rat_q_{2*m+1}_{2*n}[] = {{{floats(q)}}};")
    open("rational_coeffs.inc", "w").write("\n".join(out) + "\n")
    print("wrote rational_coeffs.inc")
