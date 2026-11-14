"""Python vs Rust timings on the same problems (steps 3, 6, 7). Needs `just py-build`.

    uv run python python/bench.py     # prints a table and writes out/bench.md

Each timing is the median of several runs after one warm-up (which also triggers Numba's
compilation). Rust is called through the PyO3 module, so its numbers include the cost of
crossing the boundary.
"""

from __future__ import annotations

import os
import platform
import statistics
import time
from pathlib import Path

import carlo_py
import numba
import numpy as np
import pandas as pd

from deribit import latest
from pricing import heston_up_and_out_numpy, mc_numba, mc_numpy
from var_numpy import one_day_var

OUT = Path(__file__).resolve().parent.parent / "out"
F, K, T, SIGMA = 67_931.42, 72_000.0, 0.21, 0.3833
HESTON = {"v0": 0.19, "kappa": 10.7, "theta": 0.225, "xi": 2.0, "rho": -0.35}


def timed(fn, repeats=5) -> tuple[float, object]:
    result = fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def main() -> None:
    rows: list[tuple[str, str, float, str]] = []

    def add(problem, impl, fn, repeats=5, show=lambda r: f"{r[0]:.2f} ± {r[1]:.2f}"):
        secs, result = timed(fn, repeats)
        rows.append((problem, impl, secs, show(result)))
        print(f"{problem:<38} {impl:<34} {secs * 1000:10.1f} ms   {show(result)}")

    p1 = "GBM European, 1M paths"
    add(p1, "NumPy (vectorised)", lambda: mc_numpy(F, K, T, SIGMA))
    add(p1, "Numba, 1 thread", lambda: mc_numba(F, K, T, SIGMA))
    add(p1, "Numba, parallel", lambda: mc_numba(F, K, T, SIGMA, parallel=True))
    add(p1, "Rust, 1 thread", lambda: carlo_py.gbm_european(F, K, T, SIGMA, parallel=False))
    add(p1, "Rust, parallel", lambda: carlo_py.gbm_european(F, K, T, SIGMA))

    df = pd.read_parquet(latest("options"))
    f, k, t, s = (df[c].to_numpy() for c in ("F", "strike", "T", "iv"))
    call = (df["type"] == "C").to_numpy()
    p2 = f"Book of {len(df)} options, 100k paths each"

    def numpy_loop():
        return [mc_numpy(*a, n=100_000) for a in zip(f, k, t, s, call, strict=True)]

    def rust_loop():
        return [carlo_py.gbm_european(*a, paths=100_000) for a in zip(f, k, t, s, call, strict=True)]

    def total(r):
        prices = [x[0] for x in r] if isinstance(r, list) else r[0]
        return f"sum {np.sum(prices):,.0f}"

    add(p2, "NumPy, loop over options", numpy_loop, 3, total)
    add(p2, "Rust, one call per option", rust_loop, 3, total)

    def rust_book():
        return carlo_py.gbm_european_book(f, k, t, s, call, paths=100_000)

    add(p2, "Rust, one call for the book", rust_book, 3, total)

    p3 = "Heston knock-out, 100k paths x 76 steps"
    args = (F, 0.21, 76, *HESTON.values(), "up_and_out_call", K)
    add(p3, "NumPy (vectorised over paths)", lambda: heston_up_and_out_numpy(F, K, 90_000.0, 0.21, 76, HESTON), 3)
    add(p3, "Rust, 1 thread", lambda: carlo_py.heston_price(*args, barrier=90_000.0, paths=100_000, parallel=False), 3)
    add(p3, "Rust, parallel", lambda: carlo_py.heston_price(*args, barrier=90_000.0, paths=100_000), 3)
    qe = {"barrier": 90_000.0, "paths": 100_000, "scheme": "qe"}
    add(p3, "Rust QE, 1 thread", lambda: carlo_py.heston_price(*args, **qe, parallel=False), 3)
    add(p3, "Rust QE, parallel", lambda: carlo_py.heston_price(*args, **qe), 3)

    p4 = "1-day 99% VaR, 10k scen. x 1k nested"

    def show_var(r):
        return f"VaR {r['var']:,.0f}, ES {r['expected_shortfall']:,.0f}"

    add(p4, "NumPy (vectorised, common numbers)", lambda: one_day_var(F), 1, show_var)
    add(p4, "Rust, 1 thread", lambda: carlo_py.sample_book_var(F, parallel=False), 1, show_var)
    add(p4, "Rust, parallel", lambda: carlo_py.sample_book_var(F), 1, show_var)

    env = (
        f"{platform.machine()}, {os.cpu_count()} cores, {platform.system()} {platform.release()}, "
        f"Python {platform.python_version()}, NumPy {np.__version__}, Numba {numba.__version__}"
    )
    table = ["| Problem | Implementation | Time (ms) | Result |", "|---|---|---:|---|"]
    table += [f"| {p} | {i} | {secs * 1000:,.1f} | {r} |" for p, i, secs, r in rows]
    OUT.mkdir(exist_ok=True)
    (OUT / "bench.md").write_text(f"Environment: {env}\n\n" + "\n".join(table) + "\n")
    print(f"\n{env}\nsaved out/bench.md")


if __name__ == "__main__":
    main()
