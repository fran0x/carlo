# Carlo's concepts in plain English

Start here if options, Monte Carlo or Value at Risk are new to you. This page explains what
carlo is trying to calculate before the [results guide](reading-results.md) explains its actual
output.

At its core, carlo asks four questions:

1. What might an option be worth under a chosen model?
2. How noisy is that calculation?
3. How much could a group of positions lose over one day?
4. Can the calculation run fast enough to be useful?

## The basic objects

An **option** is a contract linked to an asset such as Bitcoin.

- A **call** benefits when the price finishes above an agreed level.
- A **put** benefits when the price finishes below an agreed level.
- A **book** is a group of positions valued and managed together.
- **Volatility** describes how widely returns tend to move. Higher volatility means a wider
  range of plausible prices.

## Monte Carlo: try many possible futures

Monte Carlo simulation generates thousands or millions of possible price journeys, usually
called **paths**. For each path, carlo calculates what the option would pay. It averages those
payoffs to estimate the option's value.

Think of it like estimating the average result of a die by rolling it many times. More rolls make
the average steadier. They do not make the die itself more predictable.

A result such as:

```text
price: 3,128.95 ± 6.38
```

means:

- `3,128.95` is the estimated price under the model.
- `6.38` is the **standard error**, or remaining simulation noise.

It does not mean that the market price will stay within `6.38` of the estimate. Four times as
many paths should reduce this numerical noise by about half.

## Three ways to describe the market

**Geometric Brownian motion (GBM)** is the simplest model here. The price moves randomly, while
volatility stays fixed.

**Black-76** gives an exact option price in that simple world. Carlo uses it as an answer key: if
the Monte Carlo result is close to Black-76, the simulation is probably doing what it should.

**Heston** is a richer model where volatility also changes. This lets it reproduce the broad
shape seen in option markets, where one flat volatility does not fit every strike. More
flexibility does not make Heston true; it remains a simplified description of the market.

## Calibration: fit the model to a snapshot

Calibration adjusts the Heston model's parameters until its prices resemble observed market
prices.

A good fit means that the model describes that market snapshot reasonably well. It does not mean
that the parameters are permanent facts or that the future will follow the model. A parameter
stuck at its allowed limit is a warning that the model may be straining to fit the data.

## Exotic options: the journey matters

Some options depend on the whole price path rather than only the final price.

- A **barrier option** can become worthless if the price touches a particular level.
- An **Asian option** uses an average price over time.

Monte Carlo is useful for these products because it produces the complete path. The same path can
tell us whether a barrier was touched, what the average was and what the final payoff became.

## Greeks: small-change gauges

Greeks describe how an option price responds to a small change in one input.

| Greek | Plain-English question |
|---|---|
| Delta | What happens if the underlying price moves a little? |
| Gamma | How quickly will delta itself change? |
| Vega | What happens if implied volatility moves? |
| Theta | What happens when one day passes? |

Traders use Greeks to plan hedges. They are local approximations, not estimates of the worst
possible loss. Large moves, barrier events and changes across the volatility surface need full
repricing or stress scenarios.

## VaR: where the bad tail begins

**Value at Risk (VaR)** starts with a book of positions. Carlo creates many possible one-day
market moves, recalculates the book in every scenario, and sorts the gains and losses.

If the output says:

```text
99% VaR: 16,138 USD
```

then, under the model and its inputs, about 99% of the scenarios lose no more than `16,138 USD`.
The worst 1% lose more. VaR is not the maximum possible loss; it only marks where the modelled
loss tail begins.

## Expected shortfall: how bad is the tail?

Expected shortfall asks what VaR leaves unanswered: once a loss crosses that line, how large is
it on average?

```text
99% VaR:             16,138 USD
Expected shortfall:  18,735 USD
```

Here the worst 1% starts near `16,138 USD`, and the average loss inside that tail is
`18,735 USD`. Expected shortfall should therefore be at least as large as VaR.

## Nested Monte Carlo: a simulation inside a simulation

The outer simulation creates possible markets tomorrow. For each one, an inner simulation values
a complicated path-dependent option. This is called **nested Monte Carlo**.

It is useful when an option has no simple pricing formula, but it is expensive: every outer
scenario starts another collection of simulated paths.

## What people use these numbers for

Model prices and risk estimates can help people:

- compare quotes and implementations;
- size hedges;
- set portfolio risk limits;
- allocate capital;
- test whether faster code still produces the right answer; and
- estimate the computing capacity needed for live pricing.

They should not be used alone as trading signals. Real decisions also need current market data,
actual positions, spreads, fees, liquidity assumptions, historical backtests, stress tests and
an allowance for model error.

The shortest mental model is:

- **Monte Carlo** tries many possible futures.
- **Greeks** describe small changes around today.
- **VaR** marks where the bad tail begins.
- **Expected shortfall** describes how painful that tail is.

Next, [read the results](reading-results.md) to see how these ideas appear in carlo's files and
terminal output.
