# Glossary

The short version of the terms used in the [README](../README.md), the
[journey](journey.md) and [Reading the results](reading-results.md). You do not need to
memorise these; come back when a result uses a term you do not recognise.

| Term | Meaning |
|---|---|
| **Call / put** | A call pays when the market finishes above its **strike**; a put pays when it finishes below. The **expiry** is the date when that comparison is made. Deribit's options are European, so they exercise only at expiry. |
| **Book** | A group of positions valued and managed together. |
| **Mark price** | Deribit's reference price for margin and profit and loss (P&L). It is built from the option's implied volatility. |
| **Forward (F)** | The market price agreed today for delivery at the option's expiry. This project prices from the forward and assumes a zero interest rate. |
| **Volatility (σ)** | A measure of how widely returns move, quoted as an annual percentage. Higher volatility means a wider range of plausible prices. |
| **Vol point** | One percentage point of volatility: 39% to 40% is one vol point. |
| **Implied / realised volatility** | Implied volatility is the value that makes a pricing formula match today's option price. Realised volatility is measured from past market moves. |
| **Volatility smile / skew** | A plot showing that implied volatility changes with strike and expiry instead of staying flat. A smile bends at both ends; a skew mainly slopes in one direction. |
| **GBM** | Geometric Brownian motion: a simple model where the price moves randomly and volatility stays constant. |
| **Black-76** | The exact price of a European option under GBM, written in terms of the forward. Here it acts as the answer key for the simplest simulations. |
| **Heston** | A model where both price and variance can move. Five parameters describe the starting variance, long-run level, mean reversion, vol of vol and correlation. Unlike GBM, it can produce a volatility smile. |
| **Euler / QE scheme** | Two ways to move a Heston simulation forward in time. Euler is simpler; Andersen's QE scheme costs more per step but can stay accurate with larger steps. |
| **Calibration** | Choosing model parameters that make its prices fit observed market prices as closely as possible. |
| **Standard error** | An estimate of the simulation noise around a Monte Carlo price. It shrinks as more paths are added. |
| **Exotic option** | An option whose payoff depends on the path, not only the final price. A **knock-out** expires worthless after touching a barrier; an **Asian** uses an average price. |
| **Greeks** | Small-change sensitivities: **delta** to the forward, **gamma** to the change in delta, **vega** to volatility and **theta** to one day passing. |
| **VaR / expected shortfall** | Value at Risk is a loss threshold exceeded only in a chosen fraction of scenarios. Expected shortfall is the average loss beyond that threshold. |
| **Nested Monte Carlo** | A simulation inside each scenario of another simulation. It is useful when the inner product has no formula, but it is expensive. |
| **Common random numbers** | Reusing the same random draws for two valuations so that simulation noise cancels when you compare them. |
