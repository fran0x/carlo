# carlo

Price Bitcoin options by simulation, in Python and in Rust, and find out where Rust is faster and why.

Every step is implemented in both languages, checked against an exact answer or the market, and
timed.

## The idea in one minute

An **option** pays off depending on where a price ends up, for example "the amount by which BTC
is above 72,000 USD on 29 January". Its fair value today is the average payoff over all the
futures the market thinks are possible.

**Monte Carlo** estimates that average by simulating many possible futures (paths) and averaging
the payoffs. It works for any product, even ones with no formula, but it is slow: 10 times more
precision needs 100 times more paths. That is why speed matters, and why this repository compares
Python (NumPy, Numba) with Rust.

The data is real. [Deribit](https://www.deribit.com), the main crypto options exchange, publishes
prices, implied volatilities and Greeks in a free public API. `just fetch` downloads a snapshot
into `data/`; after that everything runs offline. The repository ships no market data, only the
Heston parameters fitted to the 14 Nov 2026 snapshot (`data/heston.json`) that the numbers below
come from.

## What a trader can use it for

Mostly risk and pricing, not trading signals. Deribit already publishes fair prices and Greeks
for the options it lists, so the value is in what it does not give you.

| Your question | What carlo answers | Step | Usefulness |
|---|---|---|---|
| How much could my book lose tomorrow? | One-day VaR and expected shortfall for a whole portfolio, under moves in both price and implied volatility | 6 | High: Deribit shows Greeks per option, not the worst loss for the book |
| What should I charge for a barrier or Asian option? | A price and Greeks for products Deribit does not list, from a model fitted to Deribit's smile | 4, 5 | High if you quote OTC or structured products |
| Is volatility cheap or expensive? | Realised vs implied volatility (42.3% vs 39.0% in the snapshot) | 2 | Medium: an input to a buy-vol or sell-vol view, not a strategy |
| Which strikes look rich or cheap? | Market prices next to a calibrated Heston model | 4 | Low: Heston misses BTC by 2.3 vol points on average, so gaps are mostly model error |
| Does my position still look right? | Repricing on every live Deribit update, in about 5 ms | 8 | Medium: a building block for monitoring, not a dashboard |

Three ways to use it:

1. **Morning risk check.** Value the book, then see the one-day loss in the worst 1% of
   scenarios, including a price drop combined with a jump in implied volatility.
2. **Quote a structured product.** A client wants a 76-day BTC call that dies at +30%. Fit Heston
   to today's smile, price the knock-out, take its Greeks to hedge, add your margin.
3. **Track the volatility gap.** If realised volatility stays above implied, owning options is
   cheap relative to how much BTC actually moves; the reverse favours selling them.

Limits:

- **Sample book only.** The VaR runs on a built-in book (`var::sample_book`). Your positions need
  Deribit's private API with a read-only key, which is not implemented.
- **Assumed shocks.** The VaR's market moves (45% price volatility, 100% vol of vol, −0.4
  correlation) are fixed defaults, not estimated from data.
- **No trading.** No orders, fees, funding or interest rates (prices assume a zero rate), and no
  backtest showing any signal would have made money.
- **The vanilla check is not an edge.** Black-76 matches Deribit's mark to 0.19 USD because the
  mark is built from Deribit's own implied volatility: it proves the pricer is right, not that the
  market is wrong.

## Key terms

| Term | Meaning |
|---|---|
| **Call / put** | The right to buy (call) or sell (put) at a fixed price, the **strike**, on a fixed date, the **expiry**. Deribit's options are **European**: exercised only at expiry. |
| **Forward (F)** | Today's market price for delivery at expiry. Pricing from F with a zero interest rate sidesteps crypto interest rates. |
| **Volatility (σ)** | How much returns swing, as an annual standard deviation: 40% means a typical yearly move of ±40%. |
| **Implied / realised volatility** | Implied: the σ that makes a formula match the market price, the market's forecast. Realised: measured from past prices. |
| **Volatility smile** | Implied volatility differs by strike and expiry. If the market believed in one constant σ, it would be flat. |
| **GBM** | Geometric Brownian motion: the price moves randomly in percentage terms with a constant σ. The simplest price model. |
| **Black-76** | The exact price of a European option under GBM, written in terms of the forward. The answer key for the simulation. |
| **Heston** | GBM where the variance (σ²) is random too: it pulls back to a long-run level θ at speed κ, jitters with size ξ (vol of vol) and moves with correlation ρ to the price. Five parameters, with v₀ the starting variance. It produces a smile. |
| **Euler / QE scheme** | Two ways to step a Heston path through time. Euler moves the variance by its average drift plus noise and clips it at zero; QE (Andersen, 2008) draws the next variance from a distribution matched to the exact one, so it stays accurate with large steps. |
| **Calibration** | Tuning a model's parameters until its prices match the market's. |
| **Standard error** | The uncertainty of a Monte Carlo estimate: payoff standard deviation / √paths. Two prices agree if they are within about 2 standard errors. |
| **Exotic option** | One whose payoff depends on the whole price path. A **knock-out** dies if the price touches a barrier; an **Asian** pays on the average price. Usually no formula: simulate. |
| **Greeks** | Sensitivities of the price: **delta** to the forward, **gamma** of delta, **vega** to volatility (per vol point), **theta** to one day passing. |
| **VaR / expected shortfall** | Value at Risk: a loss exceeded only with small probability, e.g. the worst 1% of tomorrows. Expected shortfall: the average loss in that worst 1%. |
| **Nested Monte Carlo** | A simulation inside each scenario of another simulation, needed when a product has no formula. Expensive. |
| **Common random numbers** | Using the same random draws for two valuations you compare, so their noise cancels in the difference. |

## Get set up

You need [Rust](https://rustup.rs) (stable), [uv](https://docs.astral.sh/uv/) and
[just](https://github.com/casey/just). uv installs Python 3.13 and every package.

```bash
just tools    # check prerequisites, create .venv
just ci       # all checks and tests, offline: if this passes, you are ready
```

## The journey

Each step answers one question, builds on the one before, and has a check that proves it works.
Read the step, run the command, then look at the code.

### 1. Get real market data

*What do Bitcoin options cost right now?*

`python/deribit.py` downloads every BTC option from Deribit (about 1,000), keeps the 322 liquid
ones, and saves them with a year of daily prices in `data/`. Steps 2, 4, 6 and 7 read it. A new
snapshot gives different prices, so your numbers will differ from the ones quoted here, which
come from 14 Nov 2026.

```bash
just fetch    # download a snapshot (network, read-only)
```

### 2. Price with the simplest model, in Python

*Can a simulation reproduce the market price?*

The simplest model (**GBM**) moves the price randomly with constant volatility. For this model
an exact formula exists (**Black-76**), so the simulation can be checked against it.
Code: `python/pricing.py`.

```bash
just checks
```

- Black-76 matches Deribit's price with a median gap of 0.19 USD.
- The simulation lands within 2 standard errors of the formula for 93% of options, as expected
  for a 95% band.
- BTC actually moved with 42.3% volatility over the past year (realised), against 39.0% implied
  near the money.

**Lesson:** give each option its own random seed. With one shared seed, all the errors move
together and this check means nothing (it showed 63%).

### 3. The same in Rust, then in parallel

*Is Rust faster? By how much, and why?*

`crates/carlo` is the Rust library. Its engine splits the paths into 256 blocks, each with its own
random stream, so the result is identical on 1 core or 14.

```bash
just bench-py    # Python vs Rust → out/bench.md
just bench-rs    # Rust micro-benchmarks (criterion)
```

| 1M paths | NumPy | Numba, 1 thread | Numba, parallel | Rust, 1 thread | Rust, parallel |
|---|---:|---:|---:|---:|---:|
| Time | 11.1 ms | 40.5 ms | 5.1 ms | 17.0 ms | 2.6 ms |

**Lesson:** on one thread, Rust *loses* to NumPy. NumPy runs optimised C, and the Rust code uses
ChaCha8, a cryptographic-quality random generator (5.9 ms per million numbers vs 1.2 ms for the
simpler `SmallRng`). Rust wins with parallelism, about 10× on 14 cores.

Compiler settings matter too (single thread):

| | Fast build (`lto=false`, 16 units) | Release (`lto=true`, 1 unit) | + `target-cpu=native` |
|---|---:|---:|---:|
| Heston knock-out | 310 ms | 217 ms | 222 ms |
| GBM, 1M paths | 15.6 ms | 16.4 ms | 16.2 ms |

### 4. A realistic model, fitted to the market

*Why doesn't one volatility fit every option?*

Plot implied volatility against strike and it is not flat: the **volatility smile**. GBM can't
produce it; **Heston**, where volatility itself moves randomly, can. `python/calibrate.py` fits
Heston's five parameters to the market with QuantLib's fast formula, as desks do, and saves
them to `data/heston.json` and the plot to `out/smile.png`.

```bash
just calibrate
just checks
```

- The fit misses the market by 2.3 volatility points on average (expiries of 30 days or more).
- The Rust simulation is then checked against QuantLib's exact Heston prices, with two ways of
  stepping through time: simple **Euler** steps and Andersen's **QE** scheme.

| Time steps | Euler: within 2 SE | QE: within 2 SE | Euler, time | QE, time |
|---|---:|---:|---:|---:|
| Weekly | 27% | 93% | | 5.5 ms |
| 1 per day | 80% | 93% | 19.5 ms | 35.7 ms |
| 8 per day | 100% | 100% | 154 ms | |

Share of 15 calibrated options within 2 standard errors of QuantLib (expect ~95%). Time: one
76-day call, 100k paths, parallel.

**Lessons:** time steps that are too coarse bias the price, and a better scheme beats smaller
steps. QE costs 1.8× more per step but is accurate with weekly steps, so it reaches the same
accuracy as Euler about 28× faster. That only helps when the payoff allows coarse steps: a
knock-out checked daily still needs daily steps. And Heston is a poor fit for BTC:
without limits the fit drifts to absurd values, and even bounded, one parameter sits at its
limit. Desks use richer models for this reason.

### 5. Products only simulation can price

*What about options that depend on the whole path?*

With the fitted Heston model, `crates/carlo/src/paths.rs` prices two **exotics**: a knock-out call
(dies if BTC ever rises 30% above today) and an Asian call (pays on the average price). 76 days,
strike 5% above today:

| Vanilla call | Knock-out, checked daily | Knock-out, checked hourly | Asian |
|---:|---:|---:|---:|
| 3,280 USD | 1,713 | 1,651 | 1,515 |

**Lesson:** checking the barrier less often misses touches in between, so it overprices the
knock-out.

### 6. Sensitivities and risk, against the clock

*How much could this book lose tomorrow, and can we compute it in under a second?*

- **Greeks** (`greeks.rs`): move one input a little, reprice, take the difference.
- **VaR** (`var.rs`, and `python/var_numpy.py` in NumPy): simulate 10,000 possible tomorrows,
  reprice a small book in each one, and read off the loss at the worst 1%. The knock-out has no
  formula, so every scenario runs its own simulation inside: 10,000 × 1,000 nested paths.

| 1-day 99% VaR | NumPy | Rust, 1 thread | Rust, parallel |
|---|---:|---:|---:|
| Time | 719 ms | 326 ms | 32 ms |

**Lesson: reuse the same random numbers when comparing two prices.**

- For Greeks, pricing with different random numbers buries the tiny difference in noise and
  makes gamma useless.
- For VaR, today's value and every scenario must share their random numbers too, or noise shifts
  every result. The first version had this bug (NumPy said 17,959, Rust 13,164). The fix values
  today as the average of tomorrow's value over the first day's move, computed on the
  scenarios' own paths (Gauss–Hermite quadrature). The two now agree to within ordinary noise
  (16,632 vs 16,138), and the shared paths also made the VaR 6–26× faster.

### 7. Python and Rust together

*Can Python keep the convenience and get Rust's speed?*

`crates/carlo-py` exposes the Rust library to Python as the `carlo_py` module (PyO3 + maturin), so
Python handles data and plots and Rust does the maths. Pricing all 322 options:

| NumPy, loop over options | Rust, one call per option | Rust, one call for the book |
|---:|---:|---:|
| 353 ms | 108 ms | 66 ms |

**Lesson:** each call across the language boundary has a cost, so send work in batches.

### 8. Live data

*How fast can we reprice when the market moves?*

`crates/carlo-live` subscribes to Deribit's live feed and reprices an option on every update, measuring
the time from message to new price. On Linux, `--pin` ties it to one CPU core to reduce latency
jitter; macOS does not allow this, so the program says so and continues.

```bash
just live --instrument BTC-29JAN27-72000-C --max-messages 50 --verbose
```

At 100,000 paths per repricing, the median latency was about 5 ms.

## Reference

**Where things are**

| Path | Step | What it is |
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

**Commands:** run `just` to list them all. Only `fetch` and `live` use the network, and only
Deribit's public, read-only market data: no account, no orders.

**Machine for the numbers above:** Apple Silicon, 14 cores, macOS; Python 3.13, NumPy 2.5,
Numba 0.68; Rust 1.97. Times are medians of several runs; yours will differ.

**Library versions:** `rand` 0.10 moved `sample` to the `RngExt` trait, and PyO3 0.26+ renamed
`allow_threads` to `detach`. Most examples online no longer compile as written.

**CI:** `.github/workflows/ci.yml` runs `just ci` on Rust 1.97.1 with the locked dependencies.
`.github/workflows/audit.yml` checks the Rust and Python dependencies for known vulnerabilities
(`cargo audit`, `pip-audit`) every Monday and when they change.

**Data:** market data is not committed (`data/*.parquet` is git-ignored) because Deribit's terms
on redistribution are unchecked. Download it yourself with `just fetch`.

**Licence:** MIT (`LICENSE`). It does not cover Deribit data you download.
