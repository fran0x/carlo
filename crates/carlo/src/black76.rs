//! Black-76: closed-form price and Greeks of a European option on a forward, with a
//! zero interest rate. This is the answer key the simulations are checked against.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use crate::{McError, Right, Vanilla, positive};

/// Standard normal cumulative distribution function.
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x * FRAC_1_SQRT_2)
}

/// Standard normal density.
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

fn d1_d2(f: f64, k: f64, t: f64, sigma: f64) -> (f64, f64) {
    let sd = sigma * t.sqrt();
    let d1 = ((f / k).ln() + 0.5 * sd * sd) / sd;
    (d1, d1 - sd)
}

fn validate(f: f64, opt: &Vanilla, sigma: f64) -> Result<(), McError> {
    positive("forward", f)?;
    positive("volatility", sigma)?;
    opt.validate()
}

/// Option price.
pub fn price(f: f64, opt: &Vanilla, sigma: f64) -> Result<f64, McError> {
    validate(f, opt, sigma)?;
    let (d1, d2) = d1_d2(f, opt.k, opt.t, sigma);
    Ok(match opt.right {
        Right::Call => f * norm_cdf(d1) - opt.k * norm_cdf(d2),
        Right::Put => opt.k * norm_cdf(-d2) - f * norm_cdf(-d1),
    })
}

/// Analytic sensitivities, in the units Deribit reports them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
    /// Change in price per 1 unit move in the forward.
    pub delta: f64,
    /// Change in delta per 1 unit move in the forward.
    pub gamma: f64,
    /// Change in price per 1 volatility point (0.01).
    pub vega: f64,
    /// Change in price per calendar day.
    pub theta: f64,
}

/// Analytic Greeks.
pub fn greeks(f: f64, opt: &Vanilla, sigma: f64) -> Result<Greeks, McError> {
    validate(f, opt, sigma)?;
    let (d1, _) = d1_d2(f, opt.k, opt.t, sigma);
    let sqrt_t = opt.t.sqrt();
    let pdf = norm_pdf(d1);
    let delta = match opt.right {
        Right::Call => norm_cdf(d1),
        Right::Put => norm_cdf(d1) - 1.0,
    };
    Ok(Greeks {
        delta,
        gamma: pdf / (f * sigma * sqrt_t),
        vega: f * pdf * sqrt_t * 0.01,
        theta: -f * pdf * sigma / (2.0 * sqrt_t) / 365.0,
    })
}

/// Implied volatility from a price, by bisection. Returns an error when the price is
/// outside the no-arbitrage bounds.
pub fn implied_vol(f: f64, opt: &Vanilla, target: f64) -> Result<f64, McError> {
    positive("forward", f)?;
    opt.validate()?;
    let intrinsic = opt.payoff(f);
    let upper_bound = match opt.right {
        Right::Call => f,
        Right::Put => opt.k,
    };
    crate::check(
        "price",
        "between intrinsic value and the no-arbitrage upper bound",
        target,
        target > intrinsic && target < upper_bound,
    )?;
    let (mut lo, mut hi) = (1e-6, 20.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if price(f, opt, mid)? < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    // BTC-29JAN27-72000-C on Deribit, 2026-11-14: F = 67,931.42, IV = 38.33%,
    // 76 days to expiry. Deribit reported mark 0.04571703 BTC, delta 0.40307,
    // vega 119.94 USD per vol point, theta -30.27 USD per day.
    const F: f64 = 67_931.42;
    const SIGMA: f64 = 0.3833;
    fn deribit_call() -> Vanilla {
        Vanilla {
            k: 72_000.0,
            t: 75.93 / 365.0,
            right: Right::Call,
        }
    }

    #[test]
    fn matches_deribit_mark_and_greeks() {
        let opt = deribit_call();
        let usd = price(F, &opt, SIGMA).unwrap();
        assert!((usd / F - 0.04571703).abs() < 2e-4, "price {}", usd / F);
        let g = greeks(F, &opt, SIGMA).unwrap();
        assert!((g.delta - 0.40307).abs() < 2e-3, "delta {}", g.delta);
        assert!((g.vega - 119.94).abs() < 1.0, "vega {}", g.vega);
        assert!((g.theta + 30.27).abs() < 0.5, "theta {}", g.theta);
    }

    #[test]
    fn put_call_parity_holds() {
        let call = deribit_call();
        let put = Vanilla {
            right: Right::Put,
            ..call
        };
        let c = price(F, &call, SIGMA).unwrap();
        let p = price(F, &put, SIGMA).unwrap();
        assert!((c - p - (F - call.k)).abs() < 1e-8);
    }

    #[test]
    fn implied_vol_round_trips() {
        let opt = deribit_call();
        let p = price(F, &opt, SIGMA).unwrap();
        let iv = implied_vol(F, &opt, p).unwrap();
        assert!((iv - SIGMA).abs() < 1e-9);
    }

    #[test]
    fn rejects_bad_inputs() {
        let opt = deribit_call();
        assert!(price(-1.0, &opt, SIGMA).is_err());
        assert!(price(F, &opt, 0.0).is_err());
        assert!(implied_vol(F, &opt, 1e9).is_err());
    }
}
