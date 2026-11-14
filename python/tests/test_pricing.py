"""Offline checks of the Python baselines and of the Rust module against them.
Tests that need the Rust module are skipped until `just py-build` has run."""

import numpy as np
import pytest

from pricing import black76, heston_up_and_out_numpy, implied_vol, mc_numba, mc_numpy
from var_numpy import one_day_var, sample_book

F, K, T, SIGMA = 67_931.42, 72_000.0, 0.21, 0.3833

try:
    import carlo_py
except ImportError:  # pragma: no cover
    carlo_py = None
needs_rust = pytest.mark.skipif(carlo_py is None, reason="Rust module not built: run `just py-build`")


def within(est, exact, z=3.0):
    price, se = est
    return abs(price - exact) < z * se


def test_black76_matches_deribit_mark():
    # BTC-29JAN27-72000-C, 2026-11-14: mark 0.04571703 BTC with F = 67,931.42.
    assert black76(F, K, 75.93 / 365, SIGMA) / F == pytest.approx(0.04571703, abs=2e-4)


def test_put_call_parity_and_implied_vol_round_trip():
    c, p = black76(F, K, T, SIGMA, True), black76(F, K, T, SIGMA, False)
    assert c - p == pytest.approx(F - K)
    assert implied_vol(float(c), F, K, T) == pytest.approx(SIGMA, abs=1e-8)


@pytest.mark.parametrize("call", [True, False])
def test_numpy_and_numba_converge_to_black76(call):
    exact = black76(F, K, T, SIGMA, call)
    assert within(mc_numpy(F, K, T, SIGMA, call), exact)
    assert within(mc_numba(F, K, T, SIGMA, call), exact)
    assert within(mc_numba(F, K, T, SIGMA, call, parallel=True), exact)


def test_heston_knock_out_is_below_the_vanilla():
    h = {"v0": 0.19, "kappa": 10.7, "theta": 0.225, "xi": 2.0, "rho": -0.35}
    ko, _ = heston_up_and_out_numpy(F, K, 90_000.0, T, 76, h, n=50_000)
    assert 0 < ko < black76(F, K, T, np.sqrt(h["theta"]))


@needs_rust
def test_rust_matches_black76_and_rejects_bad_input():
    assert within(carlo_py.gbm_european(F, K, T, SIGMA), black76(F, K, T, SIGMA))
    with pytest.raises(ValueError, match="forward"):
        carlo_py.gbm_european(-1.0, K, T, SIGMA)


@needs_rust
def test_rust_book_call_equals_per_option_calls():
    f = np.full(3, F)
    k = np.array([60_000.0, 72_000.0, 80_000.0])
    t = np.full(3, T)
    s = np.full(3, SIGMA)
    call = np.array([False, True, True])
    prices, _ = carlo_py.gbm_european_book(f, k, t, s, call, paths=100_000)
    singles = [carlo_py.gbm_european(*a, paths=100_000)[0] for a in zip(f, k, t, s, call, strict=True)]
    assert list(prices) == singles  # bit-identical: same seed, same chunks


@needs_rust
def test_rust_heston_matches_numpy():
    h = {"v0": 0.19, "kappa": 10.7, "theta": 0.225, "xi": 2.0, "rho": -0.35}
    rust = carlo_py.heston_price(F, T, 76, *h.values(), "up_and_out_call", K, barrier=90_000.0, paths=200_000)
    py = heston_up_and_out_numpy(F, K, 90_000.0, T, 76, h, n=200_000)
    assert abs(rust[0] - py[0]) < 3 * np.hypot(rust[1], py[1])


@needs_rust
def test_rust_and_numpy_var_agree():
    assert [p for p in carlo_py.sample_book(F)] == sample_book(F)
    # Different random numbers, so compare averages over several seeds.
    seeds = range(1, 6)
    rust = np.mean([carlo_py.sample_book_var(F, scenarios=5_000, seed=s)["var"] for s in seeds])
    py = np.mean([one_day_var(F, scenarios=5_000, seed=s)["var"] for s in seeds])
    assert rust == pytest.approx(py, rel=0.1)
