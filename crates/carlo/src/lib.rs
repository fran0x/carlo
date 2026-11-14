//! Monte Carlo pricing of European-style BTC options.
//!
//! Everything is priced off the forward `F` with a zero interest rate, which is how
//! Deribit quotes its options (`underlying_price` is the forward for each expiry and
//! `interest_rate` is 0). Times are in years, volatilities are annualised decimals
//! (0.40 = 40%), and prices are in the same currency as `F`.
//!
//! - [`black76`]: closed-form prices and Greeks, the answer key for the simulations.
//! - [`gbm`]: one-step GBM simulation of a European option (steps 2–3).
//! - [`paths`]: path simulation under GBM or Heston, for exotics (steps 4–5).
//! - [`greeks`]: bump-and-reprice sensitivities with common random numbers (step 6).
//! - [`var`]: one-day Value at Risk of a small book by full revaluation (step 6).

pub mod black76;
mod engine;
pub mod gbm;
pub mod greeks;
pub mod paths;
pub mod var;

pub use engine::{CHUNKS, Estimate, Exec, Sim};

/// Call or put.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Right {
    Call,
    Put,
}

/// A European option on the forward: strike `k`, time to expiry `t` in years.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vanilla {
    pub k: f64,
    pub t: f64,
    pub right: Right,
}

impl Vanilla {
    /// Payoff at expiry given the terminal forward.
    #[inline]
    pub fn payoff(&self, f_t: f64) -> f64 {
        match self.right {
            Right::Call => (f_t - self.k).max(0.0),
            Right::Put => (self.k - f_t).max(0.0),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), McError> {
        positive("strike", self.k)?;
        positive("expiry", self.t)
    }
}

/// A rejected input. Pricing functions validate their arguments instead of
/// silently returning NaN.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum McError {
    #[error("{name} must be {rule}, got {value}")]
    InvalidInput {
        name: &'static str,
        rule: &'static str,
        value: f64,
    },
}

pub(crate) fn positive(name: &'static str, value: f64) -> Result<(), McError> {
    check(
        name,
        "positive and finite",
        value,
        value > 0.0 && value.is_finite(),
    )
}

pub(crate) fn non_negative(name: &'static str, value: f64) -> Result<(), McError> {
    check(
        name,
        "non-negative and finite",
        value,
        value >= 0.0 && value.is_finite(),
    )
}

pub(crate) fn check(
    name: &'static str,
    rule: &'static str,
    value: f64,
    ok: bool,
) -> Result<(), McError> {
    if ok {
        Ok(())
    } else {
        Err(McError::InvalidInput { name, rule, value })
    }
}
