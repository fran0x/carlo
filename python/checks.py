"""Correctness checks against the frozen snapshot (steps 2-5). Needs `just py-build`
(the Rust module) and `just calibrate` (data/heston.json).

    uv run python python/checks.py
"""

from __future__ import annotations

import json

import carlo_py
import numpy as np
import pandas as pd
import QuantLib as ql

from calibrate import MIN_DAYS, helpers_for, load_snapshot, setup
from deribit import DATA, latest
from pricing import black76, mc_numpy

PATHS = 200_000


def step2_gbm_vs_black76_and_market(df: pd.DataFrame) -> None:
    call = (df["type"] == "C").to_numpy()
    exact = black76(df["F"], df["strike"], df["T"], df["iv"], call)
    # A different seed per option: with one shared seed every error moves together and the
    # whole snapshot is effectively a single draw, so "95% within 2 SE" cannot be checked.
    mc, se = np.array(
        [mc_numpy(r.F, r.strike, r.T, r.iv, r.type == "C", n=PATHS, seed=i) for i, r in enumerate(df.itertuples())]
    ).T
    z = (mc - exact) / se
    gap = (exact - df["price_usd"]).abs()
    print("Step 2 (NumPy GBM vs Black-76 vs Deribit mark)")
    print(f"  {np.mean(np.abs(z) < 2):.0%} of {len(df)} options within 2 standard errors (expect ~95%)")
    print(f"  Black-76 vs Deribit mark: median gap {gap.median():.2f} USD, max {gap.max():.2f} USD")


def step3_rust_vs_black76(df: pd.DataFrame) -> None:
    call = (df["type"] == "C").to_numpy()
    # The book call shares one seed (see step 2), so price each option with its own seed here.
    prices, ses = np.array(
        [
            carlo_py.gbm_european(r.F, r.strike, r.T, r.iv, call=r.type == "C", paths=PATHS, seed=i)
            for i, r in enumerate(df.itertuples())
        ]
    ).T
    exact = black76(df["F"], df["strike"], df["T"], df["iv"], call)
    z = (prices - exact) / ses
    print("Step 3 (Rust GBM vs Black-76)")
    print(f"  {np.mean(np.abs(z) < 2):.0%} of {len(df)} options within 2 standard errors (expect ~95%)")


def step4_heston_mc_vs_quantlib(df: pd.DataFrame, snapshot, h: dict) -> None:
    """Rust Heston Monte Carlo against QuantLib's semi-analytic price, same parameters."""
    _, curve = setup(snapshot)
    spot = ql.QuoteHandle(ql.SimpleQuote(1.0))
    model = ql.HestonModel(ql.HestonProcess(curve, curve, spot, h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"]))
    engine = ql.AnalyticHestonEngine(model)
    # Three strikes per expiry: a put wing, near the money, a call wing.
    eligible = df[df["days"] >= MIN_DAYS].sort_values(["expiry", "strike"])
    picks = pd.concat(g.iloc[sorted({0, len(g) // 2, len(g) - 1})] for _, g in eligible.groupby("expiry"))
    print("Step 4 (Rust Heston Monte Carlo vs QuantLib analytic, calibrated parameters)")
    exacts = [
        helper.modelValue() * r.F
        for r, helper in zip(picks.itertuples(), helpers_for(picks, curve, engine), strict=True)
    ]
    for label, steps_of in (
        ("weekly", lambda d: max(1, round(d / 7))),
        ("1/day", lambda d: d),
        ("8/day", lambda d: 8 * d),
    ):
        for scheme in ("euler", "qe"):
            zs = []
            for i, (r, exact) in enumerate(zip(picks.itertuples(), exacts, strict=True)):
                price, se = carlo_py.heston_price(
                    r.F,
                    r.days / 365,
                    steps_of(r.days),
                    h["v0"],
                    h["kappa"],
                    h["theta"],
                    h["xi"],
                    h["rho"],
                    "call" if r.type == "C" else "put",
                    r.strike,
                    paths=PATHS,
                    seed=i,
                    scheme=scheme,
                )
                zs.append((price - exact) / se)
            zs = np.array(zs)
            print(
                f"  {label:6} {scheme:5}: {np.mean(np.abs(zs) < 2):4.0%} of {len(zs)} within 2 SE, "
                f"mean z {zs.mean():+6.2f} (a consistent sign is discretisation bias)"
            )


def step5_exotics(df: pd.DataFrame, h: dict) -> None:
    g = df[df["days"] >= MIN_DAYS]
    expiry = sorted(g["expiry"].unique())[1]
    e = g[g["expiry"] == expiry].iloc[0]
    F, days = e["F"], int(e["days"])
    K, barrier = round(F * 1.05, -3), round(F * 1.30, -3)
    args = (h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"])

    def price(payoff, steps, **kw):
        return carlo_py.heston_price(F, days / 365, steps, *args, payoff, K, paths=PATHS, **kw)

    vanilla = price("call", days * 4)
    asian = price("asian_call", days)
    ko_daily = price("up_and_out_call", days, barrier=barrier)
    ko_hourly = price("up_and_out_call", days * 24, barrier=barrier)
    print(f"Step 5 (exotics, Heston, F={F:,.0f}, K={K:,.0f}, barrier={barrier:,.0f}, {days} days)")
    print(f"  vanilla call           {vanilla[0]:9.2f} ± {vanilla[1]:.2f}")
    print(f"  asian call (daily avg) {asian[0]:9.2f} ± {asian[1]:.2f}   < vanilla: {asian[0] < vanilla[0]}")
    print(f"  knock-out, daily       {ko_daily[0]:9.2f} ± {ko_daily[1]:.2f}   < vanilla: {ko_daily[0] < vanilla[0]}")
    print(f"  knock-out, hourly      {ko_hourly[0]:9.2f} ± {ko_hourly[1]:.2f}   < daily: {ko_hourly[0] < ko_daily[0]}")


def realized_vs_implied(df: pd.DataFrame) -> None:
    close = pd.read_parquet(latest("history"))["close"]
    rv = np.log(close).diff().std() * np.sqrt(365)
    atm = df.loc[(df["strike"] / df["F"] - 1).abs().idxmin()]
    print(f"Realised vol (1y daily): {rv:.1%}; implied near the money ({atm['instrument_name']}): {atm['iv']:.1%}")


def main() -> None:
    df, snapshot, src = load_snapshot()
    h = json.loads((DATA / "heston.json").read_text())
    print(f"snapshot {src.name}, {len(df)} options, {PATHS:,} paths per price\n")
    realized_vs_implied(df)
    step2_gbm_vs_black76_and_market(df)
    step3_rust_vs_black76(df)
    step4_heston_mc_vs_quantlib(df, snapshot, h)
    step5_exotics(df, h)


if __name__ == "__main__":
    main()
