"""One-day VaR of the sample book in NumPy (step 6), to compare with the Rust version.

Same model as `carlo::var`: correlated lognormal one-day shocks to the forward and to implied
vol, vanillas revalued with Black-76, the knock-out with a nested Monte Carlo on common
random numbers: one matrix of daily normals for the days after the horizon, used by every
scenario (valued a day later) and by today's valuation, which is the expectation of
tomorrow's value over the first day's move (Gauss-Hermite quadrature on the same paths).

Common random numbers are also what makes a fast NumPy version possible: the nested
normals are the same matrix for every scenario, so a whole batch of scenarios is one
broadcast instead of a Python loop. The barrier is checked in log space, so only the last
step of each path needs an exp.
"""

from __future__ import annotations

import numpy as np

from pricing import black76

DAY = 1 / 365


def sample_book(F: float) -> list[dict]:
    """Same book as `carlo::var::sample_book`."""
    T = 76 / 365

    def vanilla(qty, k_rel, kind, iv):
        return {"qty": qty, "kind": kind, "k": round(F * k_rel), "barrier": None, "t": T, "iv": iv}

    return [
        vanilla(-10.0, 0.85, "put", 0.42),
        vanilla(-10.0, 1.15, "call", 0.39),
        vanilla(5.0, 1.05, "call", 0.38),
        vanilla(-5.0, 1.20, "call", 0.40),
        {"qty": 20.0, "kind": "up_and_out_call", "k": round(F * 1.05), "barrier": round(F * 1.30), "t": T, "iv": 0.38},
    ]


def _days(t: float) -> int:
    return max(1, round(t / DAY))


# Probabilists' Gauss-Hermite rule, 7 points: E[g(Z)] ~ sum(w * g(x)) for Z ~ N(0, 1).
_HERMITE_X, _HERMITE_W = np.polynomial.hermite_e.hermegauss(7)
_HERMITE_W = _HERMITE_W / _HERMITE_W.sum()


def _knock_out(F, K, barrier, sigma, cum, steps):
    """Up-and-out call for a batch of scenarios. F, sigma: shape (s,); cum: running sums of
    daily normals, shape (paths, days). Monitors the first `steps` days of every path."""
    walk = cum[None, :, :steps]  # (1, paths, steps)
    F, sigma = F[:, None, None], sigma[:, None, None]
    j = np.arange(1, steps + 1)
    y = j * (-0.5 * sigma**2 * DAY) + sigma * np.sqrt(DAY) * walk  # log(F_t / F)
    y = np.concatenate([np.zeros(y.shape[:2] + (1,)), y], axis=2)  # day 0, so steps may be 0
    alive = (y.max(axis=2) < np.log(barrier / F[:, :, 0])) & (F[:, :, 0] < barrier)
    payoff = np.where(alive, np.maximum(F[:, :, 0] * np.exp(y[:, :, -1]) - K, 0.0), 0.0)
    return payoff.mean(axis=1)


def _batched_knock_out(F, p, iv, cum, batch=64):
    steps = _days(p["t"]) - 1
    return np.concatenate(
        [
            _knock_out(F[i : i + batch], p["k"], p["barrier"], iv[i : i + batch], cum, steps)
            for i in range(0, len(F), batch)
        ]
    )


def value_today(book, F, cum):
    """Book value today; knock-outs as the expectation of tomorrow's value over day one."""
    total = 0.0
    for p in book:
        if p["kind"] in ("call", "put"):
            total += p["qty"] * black76(F, p["k"], p["t"], p["iv"], call=p["kind"] == "call")
        else:
            tomorrow = F * np.exp(-0.5 * p["iv"] ** 2 * DAY + p["iv"] * np.sqrt(DAY) * _HERMITE_X)
            iv = np.full_like(tomorrow, p["iv"])
            total += p["qty"] * _HERMITE_W @ _batched_knock_out(tomorrow, p, iv, cum)
    return total


def value_tomorrow(book, F, vol_scale, cum):
    """Book value one day later, for arrays of forwards and vol scales (one per scenario)."""
    F = np.atleast_1d(F).astype(float)
    vol_scale = np.broadcast_to(vol_scale, F.shape)
    total = np.zeros_like(F)
    for p in book:
        iv = p["iv"] * vol_scale
        if p["kind"] in ("call", "put"):
            total += p["qty"] * black76(F, p["k"], p["t"] - DAY, iv, call=p["kind"] == "call")
        else:
            total += p["qty"] * _batched_knock_out(F, p, iv, cum)
    return total


def one_day_var(
    F, scenarios=10_000, nested_paths=1_000, confidence=0.99, spot_vol=0.45, vol_of_vol=1.0, rho=-0.4, seed=1
):
    """Returns a dict with base_value, var and expected_shortfall, like the Rust version."""
    book = sample_book(F)
    rng = np.random.default_rng(seed)
    longest = max((_days(p["t"]) - 1 for p in book if p["kind"] == "up_and_out_call"), default=0)
    cum = np.cumsum(rng.standard_normal((nested_paths, longest)), axis=1)

    base = value_today(book, F, cum)
    sd_f, sd_v = spot_vol * np.sqrt(DAY), vol_of_vol * np.sqrt(DAY)
    z1 = rng.standard_normal(scenarios)
    z2 = rho * z1 + np.sqrt(1 - rho * rho) * rng.standard_normal(scenarios)
    F1 = F * np.exp(-0.5 * sd_f**2 + sd_f * z1)
    vol_scale = np.exp(-0.5 * sd_v**2 + sd_v * z2)
    pnl = np.sort(value_tomorrow(book, F1, vol_scale, cum) - base)
    tail = max(1, int(np.ceil((1 - confidence) * scenarios)))
    return {"base_value": base, "var": -pnl[tail - 1], "expected_shortfall": -pnl[:tail].mean()}
