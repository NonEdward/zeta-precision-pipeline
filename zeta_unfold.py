#!/usr/bin/env python3
"""
Riemann zero unfolding + GUE comparison.
Computes the first N non-trivial zeros on the critical line, unfolds the
spacing using the Riemann-von Mangoldt density, and compares against the
GUE Wigner surmise via a KS test.

This is the CPU reference / analysis side of a WebGL2 Hardy-Z renderer.
"""
import numpy as np
from scipy.stats import kstest
import mpmath as mp

mp.mp.dps = 30


def zeta_zeros(n):
    """First n non-trivial zeros t_k (imaginary parts) via mpmath zetazero."""
    return np.array([float(mp.im(mp.zetazero(k))) for k in range(1, n + 1)])


def smooth_counting(t):
    """
    Riemann-von Mangoldt smooth zero count N(T):
        N(T) = T/(2pi) log(T/(2pi)) - T/(2pi) + 7/8 + 1/(48 T) + ...
    Used to unfold: u_k = N(t_k), so mean spacing becomes 1.
    """
    t = np.asarray(t, dtype=float)
    x = t / (2 * np.pi)
    return x * np.log(x) - x + 7.0 / 8.0 + 1.0 / (48.0 * t)


def unfold(t):
    u = smooth_counting(t)
    return np.diff(u)


def wigner_surmise(s, beta=2):
    """GUE nearest-neighbor spacing distribution (beta=2)."""
    if beta == 2:
        return (32.0 / np.pi**2) * s**2 * np.exp(-4.0 * s**2 / np.pi)
    raise ValueError("only beta=2 here")


def main(n=500):
    print(f"[zeta_unfold] computing first {n} zeros ...")
    t = zeta_zeros(n)
    print(f"  t[0]={t[0]:.9f}  t[-1]={t[-1]:.6f}")

    s = unfold(t)
    print(f"[unfold] {len(s)} spacings, mean={s.mean():.6f}, std={s.std():.6f}")

    # KS test against GUE Wigner surmise (CDF from numerical integration)
    grid = np.linspace(0, 6, 20000)
    pdf = wigner_surmise(grid)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    def gue_cdf(x):
        idx = np.clip((x / grid[-1] * (len(grid) - 1)).astype(int), 0, len(grid) - 1)
        return cdf[idx]

    ks = kstest(s, lambda x: np.interp(x, grid, cdf))
    print(f"[KS] statistic={ks.statistic:.6f}  p-value={ks.pvalue:.6f}")

    # bin histogram vs surmise for eyeballing
    bins = np.linspace(0, 3, 13)
    hist, _ = np.histogram(s, bins=bins, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    print("[hist] spacing-density vs GUE:")
    for c, h in zip(centers, hist):
        print(f"  s={c:.2f}  data={h:.4f}  gue={wigner_surmise(c):.4f}")

    verdict = "GUE-consistent" if ks.pvalue > 0.01 else "deviation (p<=0.01)"
    print(f"[result] {verdict}")


if __name__ == "__main__":
    main()
