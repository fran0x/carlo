//! Greeks by bump-and-reprice (step 6).
//!
//! Each bumped price must reuse the same random numbers as the base price (same seed,
//! same path count), otherwise the Monte Carlo noise swamps the tiny difference being
//! measured. The pricers in this crate are deterministic for a given [`Sim`](crate::Sim),
//! so passing the same `Sim` to every call is enough.

use crate::McError;

/// Relative bump of the forward used for delta and gamma.
pub const SPOT_BUMP: f64 = 0.01;
/// Absolute bump of volatility used for vega: one vol point.
pub const VOL_BUMP: f64 = 0.01;

/// Greeks that bump-and-reprice in forward and volatility can measure. Theta needs a
/// time bump of the pricer's expiry, which `price(forward, sigma)` does not expose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BumpGreeks {
    pub delta: f64,
    pub gamma: f64,
    /// Per vol point.
    pub vega: f64,
}

/// Delta, gamma and vega of any pricer `price(forward, sigma)`, by central differences.
pub fn bump_and_reprice<P>(price: P, f: f64, sigma: f64) -> Result<BumpGreeks, McError>
where
    P: Fn(f64, f64) -> Result<f64, McError>,
{
    let h = f * SPOT_BUMP;
    let base = price(f, sigma)?;
    let up = price(f + h, sigma)?;
    let down = price(f - h, sigma)?;
    let vol_up = price(f, sigma + VOL_BUMP)?;
    let vol_down = price(f, sigma - VOL_BUMP)?;
    Ok(BumpGreeks {
        delta: (up - down) / (2.0 * h),
        gamma: (up - 2.0 * base + down) / (h * h),
        vega: (vol_up - vol_down) / 2.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Exec, Right, Sim, Vanilla, black76, gbm};

    const F: f64 = 67_931.42;
    const SIGMA: f64 = 0.3833;

    fn opt() -> Vanilla {
        Vanilla {
            k: 72_000.0,
            t: 0.21,
            right: Right::Call,
        }
    }

    /// `seed_base = None` reuses seed 42 for every price (common random numbers);
    /// `Some(b)` gives each of the five prices its own seed starting at `b`.
    fn mc_greeks(seed_base: Option<u64>) -> BumpGreeks {
        let opt = opt();
        let counter = std::sync::atomic::AtomicU64::new(seed_base.unwrap_or(0));
        bump_and_reprice(
            |f, s| {
                let seed = match seed_base {
                    Some(_) => counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    None => 42,
                };
                let sim = Sim {
                    paths: 1_000_000,
                    seed,
                    exec: Exec::Parallel,
                };
                gbm::european(f, &opt, s, &sim).map(|e| e.price)
            },
            F,
            SIGMA,
        )
        .unwrap()
    }

    #[test]
    fn common_random_numbers_recover_the_analytic_greeks() {
        let exact = black76::greeks(F, &opt(), SIGMA).unwrap();
        let mc = mc_greeks(None);
        assert!(
            (mc.delta - exact.delta).abs() < 0.005,
            "delta {} vs {}",
            mc.delta,
            exact.delta
        );
        assert!(
            (mc.vega - exact.vega).abs() / exact.vega < 0.02,
            "vega {} vs {}",
            mc.vega,
            exact.vega
        );
        assert!(
            (mc.gamma - exact.gamma).abs() / exact.gamma < 0.1,
            "gamma {} vs {}",
            mc.gamma,
            exact.gamma
        );
    }

    #[test]
    fn independent_seeds_ruin_gamma() {
        // The point of the exercise: without common random numbers, the noise in each
        // price (a few USD) divided by h² (~460,000) dwarfs the true gamma (~3e-5).
        let exact = black76::greeks(F, &opt(), SIGMA).unwrap();
        let rel_err = |g: BumpGreeks| (g.gamma - exact.gamma).abs() / exact.gamma;
        let crn = rel_err(mc_greeks(None));
        let independent: Vec<f64> = (0..8).map(|i| rel_err(mc_greeks(Some(100 * i)))).collect();
        let rms = (independent.iter().map(|e| e * e).sum::<f64>() / 8.0).sqrt();
        assert!(
            rms > 10.0 * crn,
            "independent rms {rms:.3} vs common {crn:.4}"
        );
    }
}
