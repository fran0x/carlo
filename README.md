# carlo

**Price Bitcoin options by simulation, in Python and in Rust, and find out where Rust is faster
and why.**

An option pays off depending on where a price ends up, for example "the amount by which BTC is
above 72,000 USD on 29 January". Its fair value is the average payoff over the futures the market
thinks possible. **Monte Carlo** estimates that average by simulating many price paths: it works
for any product, even ones with no formula, but 10× more precision needs 100× more paths, so
speed matters. Every step here is built in both languages, checked against an exact answer or the
market, and timed, on real [Deribit](https://www.deribit.com) data.

## What a trader can use it for

Mostly risk and pricing, not trading signals. Deribit already publishes fair prices and Greeks
for the options it lists; the value is in what it does not give you.

| Your question | What carlo answers | Usefulness |
|---|---|---|
| How much could my book lose tomorrow? | One-day VaR and expected shortfall for the whole book, under moves in price and implied volatility | High: Deribit shows Greeks per option, not the book's worst loss |
| What should I charge for a barrier or Asian option? | Price and Greeks for products Deribit does not list, from Heston fitted to Deribit's smile | High if you quote over-the-counter or structured products |
| Is volatility cheap or expensive? | Realised vs implied volatility (42.3% vs 39.0% on 14 Nov 2026) | Medium: an input to a view, not a strategy |
| Which strikes look rich or cheap? | Market prices next to the calibrated model | Low: Heston misses BTC by 2.3 vol points, so gaps are mostly model error |
| Does my position still look right? | Repricing on every live Deribit update, in about 5 ms | Medium: a building block for monitoring |

For example: a **morning risk check** (the book's loss in the worst 1% of tomorrows, including a
price drop with a volatility jump); **quoting a structured product** (a 76-day call that dies at
+30%: fit Heston to today's smile, price it, hedge with its Greeks, add margin); **tracking the
volatility gap** (realised above implied makes owning options cheap; the reverse favours selling).

Limits:

- **Sample book only.** VaR runs on a built-in book (`var::sample_book`); reading your positions
  needs Deribit's private API, which is not implemented.
- **Assumed shocks.** The VaR's market moves (45% price vol, 100% vol of vol, −0.4 correlation)
  are fixed defaults, not estimated.
- **No trading.** No orders, fees, funding, interest rates or backtest.
- **The vanilla match is not an edge.** Black-76 matches Deribit's mark to 0.19 USD because the
  mark comes from Deribit's own implied volatility: it proves the pricer, not a mispricing.

## Get set up

You need [Rust](https://rustup.rs) (stable), [uv](https://docs.astral.sh/uv/) and
[just](https://github.com/casey/just). uv installs Python 3.13 and every package.

```bash
just tools    # check prerequisites, create .venv
just ci       # all checks and tests, offline: if this passes, you are ready
just fetch    # download a Deribit snapshot into data/ (network, read-only)
```

`just ci` needs no market data. Every step after 1 except 8 reads the snapshot that `just fetch`
saves.

## The journey

Eight steps, each answering one question. [The journey](docs/journey.md) has the details,
results and lessons.

| Step | Question | Run | Headline result |
|---|---|---|---|
| [1](docs/journey.md#1-get-real-market-data) | What do BTC options cost now? | `just fetch` | 322 liquid options and a year of prices |
| [2](docs/journey.md#2-price-with-the-simplest-model-in-python) | Can a simulation reproduce the market? | `just checks` | 93% of GBM prices within 2 standard errors of Black-76 |
| [3](docs/journey.md#3-the-same-in-rust-then-in-parallel) | Is Rust faster, and why? | `just bench-py` | Slower than NumPy on 1 thread, 4× faster in parallel |
| [4](docs/journey.md#4-a-realistic-model-fitted-to-the-market) | Why doesn't one volatility fit? | `just calibrate` | Heston fits to 2.3 vol points; QE ~28× faster than Euler |
| [5](docs/journey.md#5-products-only-simulation-can-price) | Options that depend on the whole path? | `just checks` | Knock-out 1,713 USD vs vanilla 3,280 |
| [6](docs/journey.md#6-sensitivities-and-risk-against-the-clock) | How much could the book lose tomorrow? | `just bench-py` | Nested VaR in 32 ms (NumPy 719 ms) |
| [7](docs/journey.md#7-python-and-rust-together) | Python's convenience with Rust's speed? | `just bench-py` | 322 options in 66 ms (NumPy 353 ms) |
| [8](docs/journey.md#8-live-data) | How fast can we reprice live? | `just live --instrument BTC-29JAN27-72000-C` | About 5 ms per update |

## Code map

| Path | Steps | What it is |
|---|---|---|
| `python/deribit.py` | 1 | Download the snapshot |
| `python/pricing.py` | 2, 4 | Black-76, GBM in NumPy and Numba, Heston in NumPy |
| `python/checks.py` | 2–5 | Correctness checks against the snapshot |
| `python/calibrate.py` | 4 | Heston calibration |
| `python/var_numpy.py` | 6 | VaR in NumPy |
| `python/bench.py` | 3, 6, 7 | Python vs Rust timings |
| `crates/carlo` | 3–6 | Rust library: pricing, exotics, Greeks, VaR, benchmarks |
| `crates/carlo-py` | 7 | Python bindings |
| `crates/carlo-live` | 8 | Live feed and latency |

Run `just` to list every command. Only `fetch` and `live` use the network, and only Deribit's
public, read-only market data: no account, no orders.

## Status

- **CI:** `.github/workflows/ci.yml` runs `just ci` on Rust 1.97.1 with locked dependencies.
  `.github/workflows/audit.yml` checks Rust and Python dependencies for known vulnerabilities
  (`cargo audit`, `pip-audit`) every Monday and when they change.
- **Data:** market data is not committed (`data/*.parquet` is git-ignored) because Deribit's
  terms on redistribution are unchecked. Only the Heston parameters fitted to the 14 Nov 2026
  snapshot (`data/heston.json`) are included, since the quoted numbers come from them.
- **Checks and timings:** the figures above come from the 14 Nov 2026 snapshot on the machine
  described in the journey. `just checks` needs downloaded data, so CI does not run it.

## More

- [The journey](docs/journey.md): the eight steps in full, with results and lessons.
- [Glossary](docs/glossary.md): every term used here, from call and put to nested Monte Carlo.
- Licence: MIT (`LICENSE`). It does not cover Deribit data you download.
