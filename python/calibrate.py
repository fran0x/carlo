"""Calibrate Heston to the frozen Deribit snapshot with QuantLib (step 4).

Calibration uses Heston's semi-analytic formula (QuantLib's AnalyticHestonEngine), as desks
do; Monte Carlo is then used for what has no formula. The fit is a bounded least squares on
implied-vol errors (scipy), over options with at least MIN_DAYS to expiry: one set of Heston
parameters cannot match both the very short and the long smiles, and unbounded fits drift to
degenerate values (kappa ~ 40, xi ~ 14) that make the Monte Carlo scheme unreliable. Each expiry has its own forward, so
options are normalised to a forward of 1: with a zero rate, C(F, K) = F * C(1, K/F), and
implied vols are unchanged.

    uv run python python/calibrate.py   # writes data/heston.json and out/smile.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.optimize import least_squares

from deribit import DATA, latest
from pricing import implied_vol

OUT = DATA.parent / "out"
MIN_DAYS = 30
PARAMS = ("v0", "kappa", "theta", "xi", "rho")
START = (0.15, 2.0, 0.2, 1.0, -0.1)
LOWER = (0.005, 0.1, 0.005, 0.05, -0.99)
UPPER = (2.0, 20.0, 2.0, 5.0, 0.99)


def setup(snapshot: pd.Timestamp):
    today = ql.Date(snapshot.day, snapshot.month, snapshot.year)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    curve = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    return today, curve


def helpers_for(df: pd.DataFrame, curve, engine) -> list:
    out = []
    for row in df.itertuples():
        h = ql.HestonModelHelper(
            ql.Period(row.days, ql.Days),
            ql.NullCalendar(),
            1.0,
            row.strike / row.F,
            ql.QuoteHandle(ql.SimpleQuote(row.iv)),
            curve,
            curve,
            ql.BlackCalibrationHelper.ImpliedVolError,
        )
        h.setPricingEngine(engine)
        out.append(h)
    return out


def calibrate(df: pd.DataFrame, snapshot: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    """Fit (v0, kappa, theta, xi, rho). Returns the parameters and the per-option fit."""
    _, curve = setup(snapshot)
    spot = ql.QuoteHandle(ql.SimpleQuote(1.0))
    model = ql.HestonModel(ql.HestonProcess(curve, curve, spot, *START))
    helpers = helpers_for(df, curve, ql.AnalyticHestonEngine(model))

    def residuals(x):
        v0, kappa, theta, xi, rho = x
        model.setParams(ql.Array([theta, kappa, xi, rho, v0]))  # QuantLib's order
        return np.array([h.calibrationError() for h in helpers])

    result = least_squares(residuals, START, bounds=(LOWER, UPPER))
    if not result.success:
        raise RuntimeError(f"calibration did not converge: {result.message}")
    residuals(result.x)  # leave the model at the optimum
    params = dict(zip(PARAMS, map(float, result.x), strict=True))

    fit = df.copy()
    fit["T_ql"] = fit["days"] / 365
    fit["model_price"] = [h.modelValue() * row.F for h, row in zip(helpers, df.itertuples(), strict=True)]
    fit["model_iv"] = [
        implied_vol(p, r.F, r.strike, r.T_ql, call=r.type == "C")
        for p, r in zip(fit["model_price"], fit.itertuples(), strict=True)
    ]
    fit["err_vol_pts"] = (fit["model_iv"] - fit["iv"]) * 100
    return params, fit


def load_snapshot() -> tuple[pd.DataFrame, pd.Timestamp, Path]:
    path = latest("options")
    df = pd.read_parquet(path)
    snapshot = df["snapshot_utc"].iloc[0]
    # QuantLib works in whole days; use the same day count everywhere we compare.
    df["days"] = (df["expiry"].dt.normalize() - snapshot.normalize()).dt.days
    return df, snapshot, path


def plot(fit: pd.DataFrame, path: Path) -> None:
    expiries = sorted(fit["expiry"].unique())
    cols = 4
    rows = -(-len(expiries) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, e in zip(axes.flat, expiries, strict=False):
        g = fit[fit["expiry"] == e].sort_values("strike")
        m = np.log(g["strike"] / g["F"])
        ax.plot(m, g["iv"] * 100, "o", ms=3, label="market")
        ax.plot(m, g["model_iv"] * 100, "-", label="Heston")
        ax.axhline(g["iv"].mean() * 100, ls=":", c="grey", label="flat (GBM)")
        ax.set_title(f"{pd.Timestamp(e):%d %b %Y}", fontsize=9)
        ax.set_xlabel("log(K/F)")
        ax.set_ylabel("implied vol (%)")
    for ax in list(axes.flat)[len(expiries) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    df, snapshot, src = load_snapshot()
    df = df[df["days"] >= MIN_DAYS]
    params, fit = calibrate(df, snapshot)
    at_bound = [
        k
        for k, v, lo, hi in zip(PARAMS, params.values(), LOWER, UPPER, strict=True)
        if np.isclose(v, lo) or np.isclose(v, hi)
    ]
    rmse = float(np.sqrt((fit["err_vol_pts"] ** 2).mean()))
    feller = 2 * params["kappa"] * params["theta"] >= params["xi"] ** 2
    print(f"snapshot {src.name}: {len(fit)} options, {fit['expiry'].nunique()} expiries")
    print("Heston: " + ", ".join(f"{k}={v:.4f}" for k, v in params.items()))
    print(f"fit RMSE {rmse:.2f} vol points; Feller condition {'holds' if feller else 'fails'}")
    if at_bound:
        print(f"at a bound: {', '.join(at_bound)} (the model is straining to fit this surface)")
    print(
        fit.groupby(fit["expiry"].dt.strftime("%Y-%m-%d"))["err_vol_pts"]
        .agg(rmse=lambda e: float(np.sqrt((e**2).mean())), worst=lambda e: float(e.abs().max()))
        .round(2)
        .to_string()
    )
    OUT.mkdir(exist_ok=True)
    plot(fit, OUT / "smile.png")
    out = {
        **params,
        "rmse_vol_points": rmse,
        "feller": bool(feller),
        "at_bound": at_bound,
        "min_days": MIN_DAYS,
        "snapshot": src.name,
    }
    (DATA / "heston.json").write_text(json.dumps(out, indent=2) + "\n")
    print("saved data/heston.json and out/smile.png")


if __name__ == "__main__":
    main()
