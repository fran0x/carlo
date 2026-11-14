# Glossary

The terms used in the [README](../README.md) and the [journey](journey.md).

| Term | Meaning |
|---|---|
| **Call / put** | The right to buy (call) or sell (put) at a fixed price, the **strike**, on a fixed date, the **expiry**. Deribit's options are **European**: exercised only at expiry. |
| **Book** | All the positions a trader holds, valued and risk-managed together. |
| **Mark price** | The exchange's own fair price for an option, used for margin and P&L. Deribit computes it from its implied volatility. |
| **Forward (F)** | Today's market price for delivery at expiry. Pricing from F with a zero interest rate sidesteps crypto interest rates. |
| **Volatility (σ)** | How much returns swing, as an annual standard deviation: 40% means a typical yearly move of ±40%. |
| **Vol point** | One percentage point of volatility: 39% to 40% is one vol point. |
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
