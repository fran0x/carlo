"""Download a frozen snapshot of Deribit BTC options and a year of daily prices (step 1).

Read-only calls to Deribit's public API: no account, no orders.

    uv run python python/deribit.py            # writes data/options_<timestamp>.parquet
                                               # and data/history_<timestamp>.parquet
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.deribit.com/api/v2/public"
DATA = Path(__file__).resolve().parent.parent / "data"
YEAR_SECONDS = 365 * 24 * 3600


def _get(method: str, **params) -> dict | list:
    r = requests.get(f"{BASE}/{method}", params=params, timeout=10)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"Deribit {method} failed: {body['error']}")
    return body["result"]


def fetch_options(now: datetime) -> pd.DataFrame:
    """All live BTC options, cleaned to liquid out-of-the-money quotes."""
    df = pd.DataFrame(_get("get_book_summary_by_currency", currency="BTC", kind="option"))
    parts = df["instrument_name"].str.split("-", expand=True)
    df["expiry"] = pd.to_datetime(parts[1], format="%d%b%y").dt.tz_localize("UTC") + pd.Timedelta(hours=8)
    df["strike"] = parts[2].astype(float)
    df["type"] = parts[3]
    df["T"] = (df["expiry"] - now).dt.total_seconds() / YEAR_SECONDS
    df["iv"] = df["mark_iv"] / 100
    df["F"] = df["underlying_price"]
    df["price_usd"] = df["mark_price"] * df["F"]

    raw = len(df)
    df = df[(df.bid_price > 0) & (df.ask_price > 0) & (df["T"] > 7 / 365)]
    otm = ((df.type == "C") & (df.strike >= df.F)) | ((df.type == "P") & (df.strike <= df.F))
    df = df[otm].copy()
    df["snapshot_utc"] = now
    print(f"options: {raw} listed, {len(df)} kept (two-sided quote, >7 days, out of the money)")
    return df.reset_index(drop=True)


def fetch_history(now: datetime, days: int = 365) -> pd.DataFrame:
    """Daily closes of the BTC perpetual."""
    end = int(now.timestamp() * 1000)
    start = end - days * 24 * 3600 * 1000
    h = _get(
        "get_tradingview_chart_data",
        instrument_name="BTC-PERPETUAL",
        resolution="1D",
        start_timestamp=start,
        end_timestamp=end,
    )
    if h.get("status") != "ok":
        raise RuntimeError(f"history request returned status {h.get('status')!r}")
    df = pd.DataFrame({"date": pd.to_datetime(h["ticks"], unit="ms", utc=True), "close": h["close"]})
    print(f"history: {len(df)} daily closes")
    return df


def latest(prefix: str) -> Path:
    """Most recent snapshot file with this prefix in data/."""
    files = sorted(DATA.glob(f"{prefix}_*.parquet"))
    if not files:
        sys.exit(f"no data/{prefix}_*.parquet found: run `just fetch` first")
    return files[-1]


def main() -> None:
    DATA.mkdir(exist_ok=True)
    now = datetime.now(UTC)
    stamp = f"{now:%Y%m%d_%H%M}"
    options = fetch_options(now)
    time.sleep(0.2)  # be polite to the public API
    history = fetch_history(now)
    options.to_parquet(DATA / f"options_{stamp}.parquet")
    history.to_parquet(DATA / f"history_{stamp}.parquet")
    print(f"saved data/options_{stamp}.parquet and data/history_{stamp}.parquet")


if __name__ == "__main__":
    main()
