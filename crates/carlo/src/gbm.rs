//! European options under GBM in a single step (steps 2–3).
//!
//! A European payoff depends only on the forward at expiry, and under GBM that has an
//! exact distribution, so one normal draw per path is enough:
//! `F_T = F · exp(−½σ²T + σ√T·Z)`.

use rand::RngExt;
use rand_distr::StandardNormal;

use crate::engine::run;
use crate::{Estimate, McError, Sim, Vanilla, positive};

/// Prices a European option on the forward `f` with volatility `sigma`.
pub fn european(f: f64, opt: &Vanilla, sigma: f64, sim: &Sim) -> Result<Estimate, McError> {
    positive("forward", f)?;
    positive("volatility", sigma)?;
    opt.validate()?;
    sim.validate()?;
    let drift = -0.5 * sigma * sigma * opt.t;
    let vol = sigma * opt.t.sqrt();
    Ok(run(sim, |rng| {
        let z: f64 = rng.sample(StandardNormal);
        opt.payoff(f * (drift + vol * z).exp())
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Exec, Right, black76};

    const F: f64 = 67_931.42;
    const SIGMA: f64 = 0.3833;

    fn sim(exec: Exec) -> Sim {
        Sim {
            paths: 400_000,
            seed: 42,
            exec,
        }
    }

    #[test]
    fn converges_to_black76_for_calls_and_puts() {
        for (k, right) in [
            (72_000.0, Right::Call),
            (60_000.0, Right::Put),
            (68_000.0, Right::Call),
        ] {
            let opt = Vanilla { k, t: 0.25, right };
            let mc = european(F, &opt, SIGMA, &sim(Exec::Parallel)).unwrap();
            let exact = black76::price(F, &opt, SIGMA).unwrap();
            let z = (mc.price - exact) / mc.std_err;
            assert!(z.abs() < 3.0, "{right:?} {k}: z = {z:.2}");
        }
    }

    #[test]
    fn sequential_and_parallel_are_bit_identical() {
        let opt = Vanilla {
            k: 72_000.0,
            t: 0.25,
            right: Right::Call,
        };
        let seq = european(F, &opt, SIGMA, &sim(Exec::Sequential)).unwrap();
        let par = european(F, &opt, SIGMA, &sim(Exec::Parallel)).unwrap();
        assert_eq!(seq.price.to_bits(), par.price.to_bits());
        assert_eq!(seq.std_err.to_bits(), par.std_err.to_bits());
    }

    #[test]
    fn standard_error_halves_with_four_times_the_paths() {
        let opt = Vanilla {
            k: 72_000.0,
            t: 0.25,
            right: Right::Call,
        };
        let small = Sim {
            paths: 100_000,
            ..sim(Exec::Parallel)
        };
        let big = Sim {
            paths: 400_000,
            ..sim(Exec::Parallel)
        };
        let ratio = european(F, &opt, SIGMA, &small).unwrap().std_err
            / european(F, &opt, SIGMA, &big).unwrap().std_err;
        assert!((ratio - 2.0).abs() < 0.1, "ratio {ratio}");
    }
}
