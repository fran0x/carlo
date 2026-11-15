# The journey

Eight steps from market data to live repricing. Each answers one question, builds on the one
before, and has a check that proves it works: read the step, run the command, then look at the
code. Terms in **bold** are defined in the [glossary](glossary.md).

Set up first (see the [README](../README.md#get-set-up)), then download data: every number below
comes from the 14 Nov 2026 snapshot, so yours will differ.

## 1. Get real market data

*What do Bitcoin options cost right now?*

`python/deribit.py` downloads every BTC option from Deribit (about 1,000), keeps the 322 liquid
ones, and saves them with a year of daily prices in `data/`. Steps 2 to 7 read it.

```bash
just fetch    # download a snapshot (network, read-only)
```

## 2. Price with the simplest model, in Python

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

## 3. The same in Rust, then in parallel

*Is Rust faster? By how much, and why?*

`crates/carlo` is the Rust library. Its engine splits the paths into 256 blocks, each with its own
random stream, so the result is identical on 1 core or 14. `just checks` confirms it against
Black-76: 95% of options within 2 standard errors.

```bash
just bench-py    # Python vs Rust → out/bench.md
just bench-rs    # Rust micro-benchmarks (criterion)
```

| 1M paths | NumPy | Numba, 1 thread | Numba, parallel | Rust, 1 thread | Rust, parallel |
|---|---:|---:|---:|---:|---:|
| Time | 11.1 ms | 40.5 ms | 5.1 ms | 17.0 ms | 2.6 ms |

**Lesson:** on one thread, Rust *loses* to NumPy. NumPy runs optimised C, and the Rust code uses
ChaCha8, a cryptographic-quality random generator (5.9 ms per million numbers vs 1.2 ms for the
simpler `SmallRng`). Rust wins with parallelism: 6–10× its own single-thread time on 14 cores and 4–6× faster than
NumPy, depending on the run (parallel times ranged from 1.6 to 2.6 ms).

Compiler settings matter too (single thread; `just bench-rs --profile release-fast-build` for
the fast build, `RUSTFLAGS="-C target-cpu=native" just bench-rs` for the last column):

| | Fast build (`lto=false`, 16 units) | Release (`lto=true`, 1 unit) | + `target-cpu=native` |
|---|---:|---:|---:|
| Heston knock-out | 310 ms | 217 ms | 222 ms |
| GBM, 1M paths | 15.6 ms | 16.4 ms | 16.2 ms |

## 4. A realistic model, fitted to the market

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

| Time steps | Euler: within 2 standard errors | QE: within 2 standard errors | Euler, time | QE, time |
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

## 5. Products only simulation can price

*What about options that depend on the whole path?*

With the fitted Heston model, `crates/carlo/src/paths.rs` prices two **exotics**: a knock-out call
(dies if BTC ever rises 30% above today) and an Asian call (pays on the average price). 76 days,
forward 67,978 USD, strike 71,000, barrier 88,000:

```bash
just checks
```

| Vanilla call | Knock-out, checked daily | Knock-out, checked hourly | Asian |
|---:|---:|---:|---:|
| 3,280 USD | 1,713 | 1,651 | 1,515 |

**Lesson:** checking the barrier less often misses touches in between, so it overprices the
knock-out.

## 6. Sensitivities and risk, against the clock

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

## 7. Python and Rust together

*Can Python keep the convenience and get Rust's speed?*

`crates/carlo-py` exposes the Rust library to Python as the `carlo_py` module (PyO3 + maturin), so
Python handles data and plots and Rust does the maths. Pricing all 322 options:

| NumPy, loop over options | Rust, one call per option | Rust, one call for the book |
|---:|---:|---:|
| 353 ms | 108 ms | 66 ms |

**Lesson:** each call across the language boundary has a cost, so send work in batches.

## 8. Live data

*How fast can we reprice when the market moves?*

`crates/carlo-live` subscribes to Deribit's live feed and reprices an option on every update, measuring
the time from message to new price. On Linux, `--pin` ties it to one CPU core to reduce latency
jitter; macOS does not allow this, so the program says so and continues.

```bash
just live --instrument BTC-29JAN27-72000-C --max-messages 50 --verbose
```

At 100,000 paths per repricing, the median latency was about 5 ms.

## About the numbers

**Machine:** Apple Silicon, 14 cores, macOS; Python 3.13, NumPy 2.5, Numba 0.68; Rust 1.97.
Times are medians of several runs; yours will differ.

**Library versions:** `rand` 0.10 moved `sample` to the `RngExt` trait, and PyO3 0.26+ renamed
`allow_threads` to `detach`. Most examples online no longer compile as written.
