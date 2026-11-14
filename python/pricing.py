"""Python baselines (steps 2-4): Black-76, GBM Monte Carlo in NumPy and Numba, and a
Heston path simulator in NumPy. Same conventions as the Rust crate: forward F, zero rate,
T in years, volatilities as decimals, prices in the currency of F."""

from __future__ import annotations

import numba
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black76(F, K, T, sigma, call=True):
    """Black-76 price; works on scalars or NumPy arrays."""
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    c = F * norm.cdf(d1) - K * norm.cdf(d2)
    return np.where(call, c, c - (F - K))  # put by put-call parity


def implied_vol(price, F, K, T, call=True):
    """Black-76 implied volatility by root finding."""
    return brentq(lambda s: float(black76(F, K, T, s, call)) - price, 1e-6, 10.0, xtol=1e-12)


def mc_numpy(F, K, T, sigma, call=True, n=1_000_000, seed=42):
    """GBM European option, vectorised NumPy. Returns (price, std_err)."""
    z = np.random.default_rng(seed).standard_normal(n)
    FT = F * np.exp(-0.5 * sigma**2 * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(FT - K, 0.0) if call else np.maximum(K - FT, 0.0)
    return payoff.mean(), payoff.std(ddof=1) / np.sqrt(n)


@numba.njit(cache=True)
def _mc_numba_seq(F, K, T, sigma, call, n, seed):
    np.random.seed(seed)
    drift, vol = -0.5 * sigma * sigma * T, sigma * np.sqrt(T)
    s = s2 = 0.0
    for _ in range(n):
        x = F * np.exp(drift + vol * np.random.standard_normal())
        p = max(x - K, 0.0) if call else max(K - x, 0.0)
        s += p
        s2 += p * p
    mean = s / n
    return mean, np.sqrt((s2 / n - mean * mean) / (n - 1))


@numba.njit(cache=True, parallel=True)
def _mc_numba_par(F, K, T, sigma, call, n, seed):
    # 256 chunks like the Rust engine, but Numba's RNG state is per thread and the seed
    # only reaches the calling thread: results are statistically valid, not reproducible
    # from one run to the next.
    chunks = 256
    np.random.seed(seed)
    drift, vol = -0.5 * sigma * sigma * T, sigma * np.sqrt(T)
    sums = np.zeros(chunks)
    sums2 = np.zeros(chunks)
    per, extra = divmod(n, chunks)
    for c in numba.prange(chunks):
        s = s2 = 0.0
        for _ in range(per + (c < extra)):
            x = F * np.exp(drift + vol * np.random.standard_normal())
            p = max(x - K, 0.0) if call else max(K - x, 0.0)
            s += p
            s2 += p * p
        sums[c] = s
        sums2[c] = s2
    mean = sums.sum() / n
    return mean, np.sqrt((sums2.sum() / n - mean * mean) / (n - 1))


def mc_numba(F, K, T, sigma, call=True, n=1_000_000, seed=42, parallel=False):
    """GBM European option, Numba-compiled loop. Returns (price, std_err)."""
    f = _mc_numba_par if parallel else _mc_numba_seq
    return f(float(F), float(K), float(T), float(sigma), bool(call), int(n), int(seed))


def heston_paths_numpy(F, T, steps, v0, kappa, theta, xi, rho, n=100_000, seed=42):
    """Full-truncation Euler paths under Heston, vectorised over paths.
    Returns (terminal F, running max, running mean)."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    x = np.full(n, float(F))
    v = np.full(n, float(v0))
    mx = x.copy()
    total = np.zeros(n)
    rho_c = np.sqrt(1 - rho * rho)
    for _ in range(steps):
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + rho_c * rng.standard_normal(n)
        vp = np.maximum(v, 0.0)
        sd = np.sqrt(vp * dt)
        x *= np.exp(-0.5 * vp * dt + sd * z1)
        v += kappa * (theta - vp) * dt + xi * sd * z2
        np.maximum(mx, x, out=mx)
        total += x
    return x, mx, total / steps


def heston_up_and_out_numpy(F, K, barrier, T, steps, heston: dict, n=100_000, seed=42):
    """Up-and-out call under Heston in NumPy. Returns (price, std_err)."""
    x, mx, _ = heston_paths_numpy(F, T, steps, n=n, seed=seed, **heston)
    payoff = np.where(mx >= barrier, 0.0, np.maximum(x - K, 0.0))
    return payoff.mean(), payoff.std(ddof=1) / np.sqrt(n)
