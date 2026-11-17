"""Regenerate the charts embedded in docs/reading-results.md.

The inputs are fixed and offline. The smile uses synthetic observations generated around a
Heston curve; it does not contain downloaded Deribit market data.

    uv run python python/docs_charts.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import QuantLib as ql

from pricing import implied_vol, mc_numpy
from var_numpy import DAY, _days, sample_book, value_today, value_tomorrow

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"

BLUE = "#2563eb"
LIGHT_BLUE = "#93c5fd"
RED = "#dc2626"
PURPLE = "#7c3aed"
GREY = "#64748b"
BLACK = "#111827"

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 10,
        "axes.edgecolor": "#94a3b8",
        "axes.labelcolor": BLACK,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "text.color": BLACK,
        "svg.hashsalt": "carlo-docs-v1",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    """Save a deterministic SVG with a white background."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "carlo python/docs_charts.py", "Date": "2026-11-17"},
    )
    # Matplotlib leaves spaces at the end of multiline SVG paths. Normalize the
    # generated file so repository whitespace checks stay clean.
    svg = path.read_text()
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    if preview_root := os.environ.get("CARLO_CHART_PREVIEW_DIR"):
        preview = Path(preview_root)
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / f"{Path(name).stem}.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def convergence_chart() -> None:
    """Show the inverse-square-root relationship between paths and standard error."""
    f, k, t, sigma = 67_931.42, 72_000.0, 0.21, 0.3833
    paths = np.array([1_000, 4_000, 16_000, 64_000, 256_000, 1_024_000])
    errors = np.array([mc_numpy(f, k, t, sigma, n=int(n), seed=42)[1] for n in paths])
    reference = errors[0] * np.sqrt(paths[0] / paths)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.loglog(paths, errors, "o-", color=BLUE, lw=2, ms=6, label="Measured standard error")
    ax.loglog(paths, reference, "--", color=GREY, lw=1.5, label=r"Reference: $1/\sqrt{paths}$")
    ax.set_title("More paths reduce Monte Carlo noise")
    ax.set_xlabel("Simulated paths (log scale)")
    ax.set_ylabel("Standard error in USD (log scale)")
    ax.grid(True, which="both", color="#e2e8f0", lw=0.8)
    ax.legend(frameon=False)
    ax.annotate(
        "4× paths → about ½ the error",
        xy=(64_000, errors[3]),
        xytext=(7_000, errors[3] * 0.58),
        arrowprops={"arrowstyle": "->", "color": GREY},
        color=BLACK,
    )
    fig.text(
        0.5,
        0.01,
        "Fixed GBM option and seed; the curve shows numerical uncertainty, not market risk.",
        ha="center",
        color=GREY,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "monte-carlo-convergence.svg")


def heston_curve(strikes: np.ndarray) -> np.ndarray:
    """Return implied vols from a fixed, illustrative Heston model."""
    today = ql.Date(14, 11, 2026)
    days = 90
    maturity = today + days
    day_count = ql.Actual365Fixed()
    curve = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
    spot = ql.QuoteHandle(ql.SimpleQuote(1.0))
    process = ql.HestonProcess(curve, curve, spot, 0.19, 10.7, 0.225, 2.0, -0.35)
    engine = ql.AnalyticHestonEngine(ql.HestonModel(process))
    t = days / 365
    vols = []
    for strike in strikes:
        call = strike >= 1.0
        option_type = ql.Option.Call if call else ql.Option.Put
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(option_type, float(strike)),
            ql.EuropeanExercise(maturity),
        )
        option.setPricingEngine(engine)
        vols.append(implied_vol(option.NPV(), 1.0, float(strike), t, call=call))
    return np.array(vols)


def smile_chart() -> None:
    """Explain a smile without embedding downloaded market observations."""
    strikes = np.linspace(0.75, 1.25, 31)
    model = heston_curve(strikes)
    observed_idx = np.arange(0, len(strikes), 3)
    observed = model[observed_idx] + 0.006 * np.sin(np.linspace(0, 3 * np.pi, len(observed_idx)))
    atm = model[np.abs(strikes - 1.0).argmin()]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(np.log(strikes), model * 100, color=BLUE, lw=2.2, label="Heston curve")
    ax.scatter(
        np.log(strikes[observed_idx]),
        observed * 100,
        color=BLACK,
        s=28,
        zorder=3,
        label="Synthetic observations",
    )
    ax.axhline(atm * 100, color=GREY, ls="--", lw=1.5, label="Flat GBM volatility")
    ax.set_title("A volatility smile cannot be captured by one flat volatility")
    ax.set_xlabel("Log-moneyness: log(strike / forward)")
    ax.set_ylabel("Implied volatility (%)")
    ax.grid(True, color="#e2e8f0", lw=0.8)
    ax.legend(frameon=False)
    fig.text(
        0.5,
        0.01,
        "Illustrative synthetic observations around a fixed Heston model; no Deribit data.",
        ha="center",
        color=GREY,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "synthetic-volatility-smile.svg")


def sample_losses() -> tuple[np.ndarray, float, float]:
    """Reproduce the benchmark's one-day scenarios and return positive losses."""
    f = 67_931.42
    scenarios, nested_paths, confidence, seed = 10_000, 1_000, 0.99, 1
    spot_vol, vol_of_vol, rho = 0.45, 1.0, -0.4
    book = sample_book(f)
    rng = np.random.default_rng(seed)
    longest = max((_days(p["t"]) - 1 for p in book if p["kind"] == "up_and_out_call"), default=0)
    cumulative = np.cumsum(rng.standard_normal((nested_paths, longest)), axis=1)
    base = value_today(book, f, cumulative)

    sd_f, sd_v = spot_vol * np.sqrt(DAY), vol_of_vol * np.sqrt(DAY)
    z1 = rng.standard_normal(scenarios)
    z2 = rho * z1 + np.sqrt(1 - rho * rho) * rng.standard_normal(scenarios)
    forwards = f * np.exp(-0.5 * sd_f**2 + sd_f * z1)
    vol_scale = np.exp(-0.5 * sd_v**2 + sd_v * z2)
    pnl = np.sort(value_tomorrow(book, forwards, vol_scale, cumulative) - base)
    tail_count = max(1, int(np.ceil((1 - confidence) * scenarios)))
    worst = pnl[:tail_count]
    return -pnl, float(-worst[-1]), float(-worst.mean())


def var_chart() -> None:
    """Show where VaR ends and expected shortfall begins in the loss distribution."""
    losses, var, expected_shortfall = sample_losses()
    counts, bins = np.histogram(losses, bins=72)
    centres = 0.5 * (bins[:-1] + bins[1:])
    colours = [RED if centre >= var else LIGHT_BLUE for centre in centres]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(centres, counts, width=np.diff(bins), color=colours, edgecolor="white", linewidth=0.3)
    ax.axvline(var, color=RED, lw=2, label=f"99% VaR: {var:,.0f}")
    ax.axvline(
        expected_shortfall,
        color=PURPLE,
        lw=2,
        ls="--",
        label=f"Expected shortfall: {expected_shortfall:,.0f}",
    )
    ax.set_title("VaR marks the start of the loss tail")
    ax.set_xlabel("One-day loss in USD (negative values are gains)")
    ax.set_ylabel("Scenarios")
    ax.grid(True, axis="y", color="#e2e8f0", lw=0.8)
    ax.legend(frameon=False)
    fig.text(
        0.5,
        0.01,
        "NumPy baseline • 10,000 scenarios • built-in sample book • red bars are the worst 1%.",
        ha="center",
        color=GREY,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "var-loss-distribution.svg")


def main() -> None:
    convergence_chart()
    smile_chart()
    var_chart()


if __name__ == "__main__":
    main()
