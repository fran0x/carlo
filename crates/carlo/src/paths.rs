//! Path simulation under GBM or Heston, for payoffs that depend on the whole path
//! (steps 4–5).
//!
//! Heston has two discretisation schemes ([`Scheme`]):
//!
//! - **Euler** with full truncation, `v⁺ = max(v, 0)`:
//!
//!   ```text
//!   z2 = ρ·z1 + √(1−ρ²)·w
//!   F ← F · exp(−½ v⁺ dt + √(v⁺ dt) · z1)
//!   v ← v + κ(θ − v⁺) dt + ξ √(v⁺ dt) · z2
//!   ```
//!
//! - **QE**, Andersen's quadratic-exponential scheme (2008). It draws the next variance
//!   from a distribution matched to the exact mean and variance of the square-root
//!   process (a scaled non-central χ² when variance is high, a point mass at zero plus an
//!   exponential when it is low), so it stays accurate with large steps and when the
//!   Feller condition fails. The log-forward step uses trapezoidal weights and Andersen's
//!   martingale correction, so the simulated forward has the right mean on every step.
//!
//! Barriers and averages are monitored at the end of each step only, so the step count
//! is also the monitoring frequency (daily steps = daily monitoring).

use rand::RngExt;
use rand_distr::StandardNormal;

use crate::engine::run;
use crate::{Estimate, McError, Right, Sim, black76, check, non_negative, positive};

/// Heston parameters: initial variance `v0`, mean-reversion speed `kappa`, long-run
/// variance `theta`, vol of vol `xi`, and price–variance correlation `rho`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Heston {
    pub v0: f64,
    pub kappa: f64,
    pub theta: f64,
    pub xi: f64,
    pub rho: f64,
}

impl Heston {
    fn validate(&self) -> Result<(), McError> {
        non_negative("v0", self.v0)?;
        non_negative("kappa", self.kappa)?;
        non_negative("theta", self.theta)?;
        non_negative("xi", self.xi)?;
        check(
            "rho",
            "between -1 and 1",
            self.rho,
            (-1.0..=1.0).contains(&self.rho),
        )
    }

    /// Feller condition `2κθ ≥ ξ²`: when it fails, the variance process can hit zero and
    /// the truncation in the scheme matters more. Common for calibrated crypto parameters.
    pub fn feller(&self) -> bool {
        2.0 * self.kappa * self.theta >= self.xi * self.xi
    }
}

/// How a Heston path is stepped through time (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// Full-truncation Euler: simple, but biased unless steps are small.
    Euler,
    /// Andersen's quadratic-exponential scheme with martingale correction.
    Qe,
}

/// The price dynamics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Model {
    /// Constant volatility.
    Gbm { sigma: f64 },
    /// Stochastic variance.
    Heston { params: Heston, scheme: Scheme },
}

/// What the option pays, evaluated on a simulated path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Payoff {
    /// Pays on the terminal forward only.
    European { k: f64, right: Right },
    /// A call that is worth zero if the forward ever reaches `barrier`.
    UpAndOutCall { k: f64, barrier: f64 },
    /// A call on the arithmetic average of the forward over the monitoring dates.
    AsianCall { k: f64 },
}

impl Payoff {
    #[inline]
    fn eval(&self, last: f64, max: f64, mean: f64) -> f64 {
        match *self {
            Payoff::European {
                k,
                right: Right::Call,
            } => (last - k).max(0.0),
            Payoff::European {
                k,
                right: Right::Put,
            } => (k - last).max(0.0),
            Payoff::UpAndOutCall { k, barrier } => {
                if max >= barrier {
                    0.0
                } else {
                    (last - k).max(0.0)
                }
            }
            Payoff::AsianCall { k } => (mean - k).max(0.0),
        }
    }

    fn validate(&self) -> Result<(), McError> {
        match *self {
            Payoff::European { k, .. } | Payoff::AsianCall { k } => positive("strike", k),
            Payoff::UpAndOutCall { k, barrier } => {
                positive("strike", k)?;
                check("barrier", "above the strike", barrier, barrier > k)
            }
        }
    }
}

/// Prices `payoff` on paths of `steps` equal time steps from `f` to expiry `t`.
pub fn price(
    model: &Model,
    f: f64,
    t: f64,
    steps: u32,
    payoff: &Payoff,
    sim: &Sim,
) -> Result<Estimate, McError> {
    positive("forward", f)?;
    positive("expiry", t)?;
    check("steps", "at least 1", f64::from(steps), steps > 0)?;
    payoff.validate()?;
    sim.validate()?;
    let dt = t / f64::from(steps);
    let sqrt_dt = dt.sqrt();
    let n = f64::from(steps);

    // The match sits outside the path loop so each model gets its own tight loop.
    Ok(match *model {
        Model::Gbm { sigma } => {
            positive("volatility", sigma)?;
            let drift = -0.5 * sigma * sigma * dt;
            let vol = sigma * sqrt_dt;
            run(sim, |rng| {
                let (mut x, mut max, mut sum) = (f, f, 0.0);
                for _ in 0..steps {
                    let z: f64 = rng.sample(StandardNormal);
                    x *= (drift + vol * z).exp();
                    max = max.max(x);
                    sum += x;
                }
                payoff.eval(x, max, sum / n)
            })
        }
        Model::Heston {
            params: h,
            scheme: Scheme::Euler,
        } => {
            h.validate()?;
            let rho_c = (1.0 - h.rho * h.rho).sqrt();
            run(sim, |rng| {
                let (mut x, mut v, mut max, mut sum) = (f, h.v0, f, 0.0);
                for _ in 0..steps {
                    let z1: f64 = rng.sample(StandardNormal);
                    let w: f64 = rng.sample(StandardNormal);
                    let z2 = h.rho * z1 + rho_c * w;
                    let vp = v.max(0.0);
                    let sd = (vp * dt).sqrt();
                    x *= (-0.5 * vp * dt + sd * z1).exp();
                    v += h.kappa * (h.theta - vp) * dt + h.xi * sd * z2;
                    max = max.max(x);
                    sum += x;
                }
                payoff.eval(x, max, sum / n)
            })
        }
        Model::Heston {
            params: h,
            scheme: Scheme::Qe,
        } => {
            h.validate()?;
            positive("kappa", h.kappa)?;
            positive("theta", h.theta)?;
            positive("xi", h.xi)?;
            let qe = Qe::new(&h, dt);
            run(sim, |rng| {
                let (mut x, mut v, mut max, mut sum) = (f, h.v0, f, 0.0);
                for _ in 0..steps {
                    let zv: f64 = rng.sample(StandardNormal);
                    let z: f64 = rng.sample(StandardNormal);
                    let (next, log_m) = qe.variance(v, zv);
                    x *= qe.log_step(v, next, log_m, z).exp();
                    v = next;
                    max = max.max(x);
                    sum += x;
                }
                payoff.eval(x, max, sum / n)
            })
        }
    })
}

/// Constants of Andersen's QE step for one step size (γ₁ = γ₂ = ½).
struct Qe {
    /// e^(−κΔ)
    decay: f64,
    theta: f64,
    /// ξ²e^(−κΔ)(1 − e^(−κΔ))/κ and θξ²(1 − e^(−κΔ))²/(2κ): the conditional variance of
    /// the next variance is `c1·v + c2`.
    c1: f64,
    c2: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
    /// K₂ + ½K₄: the exponent whose moment the martingale correction cancels.
    a: f64,
    /// Drift without the correction, −ρκθΔ/ξ, for the rare steps where the moment is
    /// infinite (Andersen's fallback).
    k0: f64,
}

impl Qe {
    /// Switch between the quadratic and the exponential branch, as in Andersen (2008).
    const PSI_C: f64 = 1.5;

    fn new(h: &Heston, dt: f64) -> Self {
        let decay = (-h.kappa * dt).exp();
        let one_minus = -(-h.kappa * dt).exp_m1();
        let xi2 = h.xi * h.xi;
        let k = h.kappa * h.rho / h.xi - 0.5;
        let (k1, k2) = (0.5 * dt * k - h.rho / h.xi, 0.5 * dt * k + h.rho / h.xi);
        let (k3, k4) = (
            0.5 * dt * (1.0 - h.rho * h.rho),
            0.5 * dt * (1.0 - h.rho * h.rho),
        );
        Self {
            decay,
            theta: h.theta,
            c1: xi2 * decay * one_minus / h.kappa,
            c2: h.theta * xi2 * one_minus * one_minus / (2.0 * h.kappa),
            k1,
            k2,
            k3,
            k4,
            a: k2 + 0.5 * k4,
            k0: -h.rho * h.kappa * h.theta * dt / h.xi,
        }
    }

    /// Next variance from `v` and a standard normal `zv`, with `ln E[e^(A·v_next) | v]`
    /// for the martingale correction.
    #[inline]
    fn variance(&self, v: f64, zv: f64) -> (f64, f64) {
        let m = self.theta + (v - self.theta) * self.decay;
        let s2 = self.c1 * v + self.c2;
        let psi = s2 / (m * m);
        if psi <= Self::PSI_C {
            let r = 2.0 / psi;
            let b2 = r - 1.0 + (r * (r - 1.0)).sqrt();
            let a = m / (1.0 + b2);
            let b = b2.sqrt();
            let next = a * (b + zv) * (b + zv);
            let d = 1.0 - 2.0 * self.a * a;
            let log_m = if d > 0.0 {
                self.a * b2 * a / d - 0.5 * d.ln()
            } else {
                self.uncorrected(v)
            };
            (next, log_m)
        } else {
            let p = (psi - 1.0) / (psi + 1.0);
            let beta = (1.0 - p) / m;
            let u = black76::norm_cdf(zv);
            let next = if u <= p {
                0.0
            } else {
                ((1.0 - p) / (1.0 - u)).ln() / beta
            };
            let log_m = if self.a < beta {
                (p + beta * (1.0 - p) / (beta - self.a)).ln()
            } else {
                self.uncorrected(v)
            };
            (next, log_m)
        }
    }

    /// The `log_m` that makes [`Self::log_step`] use the uncorrected drift `k0`.
    fn uncorrected(&self, v: f64) -> f64 {
        -self.k0 - (self.k1 + 0.5 * self.k3) * v
    }

    /// Log-forward increment over the step, martingale-corrected.
    #[inline]
    fn log_step(&self, v: f64, next: f64, log_m: f64, z: f64) -> f64 {
        let k0 = -log_m - (self.k1 + 0.5 * self.k3) * v;
        k0 + self.k1 * v + self.k2 * next + (self.k3 * v + self.k4 * next).sqrt() * z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Exec, Vanilla, black76};

    const F: f64 = 67_931.42;
    const SIGMA: f64 = 0.40;
    const T: f64 = 0.25;

    fn sim(paths: u64) -> Sim {
        Sim {
            paths,
            seed: 7,
            exec: Exec::Parallel,
        }
    }

    fn call(k: f64) -> Payoff {
        Payoff::European {
            k,
            right: Right::Call,
        }
    }

    fn exact_call(k: f64) -> f64 {
        let opt = Vanilla {
            k,
            t: T,
            right: Right::Call,
        };
        black76::price(F, &opt, SIGMA).unwrap()
    }

    fn assert_close(mc: Estimate, exact: f64) {
        let z = (mc.price - exact) / mc.std_err;
        assert!(z.abs() < 3.0, "mc {} vs {exact}: z = {z:.2}", mc.price);
    }

    #[test]
    fn gbm_paths_match_black76() {
        let mc = price(
            &Model::Gbm { sigma: SIGMA },
            F,
            T,
            30,
            &call(72_000.0),
            &sim(200_000),
        )
        .unwrap();
        assert_close(mc, exact_call(72_000.0));
    }

    #[test]
    fn heston_without_vol_of_vol_is_gbm() {
        // xi = 0 and v0 = theta: the variance never moves, so Heston collapses to GBM
        // with sigma = sqrt(v0). The Euler scheme is then exact.
        let h = Heston {
            v0: SIGMA * SIGMA,
            kappa: 2.0,
            theta: SIGMA * SIGMA,
            xi: 0.0,
            rho: -0.5,
        };
        let mc = price(
            &Model::Heston {
                params: h,
                scheme: Scheme::Euler,
            },
            F,
            T,
            30,
            &call(72_000.0),
            &sim(200_000),
        )
        .unwrap();
        assert_close(mc, exact_call(72_000.0));
    }

    #[test]
    fn heston_negative_rho_produces_a_skew() {
        // With rho < 0 the left tail is fatter: a far put is worth more than under GBM
        // with the same average variance, a far call less.
        let h = Heston {
            v0: SIGMA * SIGMA,
            kappa: 1.5,
            theta: SIGMA * SIGMA,
            xi: 1.0,
            rho: -0.7,
        };
        let put = Payoff::European {
            k: 50_000.0,
            right: Right::Put,
        };
        let opt = Vanilla {
            k: 50_000.0,
            t: T,
            right: Right::Put,
        };
        let heston = price(
            &Model::Heston {
                params: h,
                scheme: Scheme::Euler,
            },
            F,
            T,
            90,
            &put,
            &sim(200_000),
        )
        .unwrap();
        let gbm = black76::price(F, &opt, SIGMA).unwrap();
        assert!(heston.price > gbm + 3.0 * heston.std_err);
    }

    /// Calibrated to the BTC snapshot (`data/heston.json`): high vol of vol, Feller fails.
    const BTC: Heston = Heston {
        v0: 0.191,
        kappa: 10.7,
        theta: 0.225,
        xi: 5.0,
        rho: -0.35,
    };
    const T_76: f64 = 76.0 / 365.0;
    /// QuantLib `AnalyticHestonEngine` prices for `BTC`, F = 67,931.42, 76 days, zero rate.
    const QL_PUT_60K: f64 = 2_113.940_884_367_128;
    const QL_CALL_72K: f64 = 2_884.595_521_878_191;

    fn heston(scheme: Scheme) -> Model {
        Model::Heston {
            params: BTC,
            scheme,
        }
    }

    fn z_vs_quantlib(scheme: Scheme, steps: u32) -> [f64; 2] {
        let put = Payoff::European {
            k: 60_000.0,
            right: Right::Put,
        };
        let z = |payoff: &Payoff, exact: f64| {
            let e = price(&heston(scheme), F, T_76, steps, payoff, &sim(200_000)).unwrap();
            (e.price - exact) / e.std_err
        };
        [z(&put, QL_PUT_60K), z(&call(72_000.0), QL_CALL_72K)]
    }

    #[test]
    fn qe_matches_quantlib_with_weekly_steps_where_euler_is_biased() {
        // 11 steps over 76 days. Euler overprices the call by about 13%; QE is within noise.
        for z in z_vs_quantlib(Scheme::Qe, 11) {
            assert!(z.abs() < 3.0, "qe z = {z:.2}");
        }
        let [_, euler_call] = z_vs_quantlib(Scheme::Euler, 11);
        assert!(euler_call > 10.0, "euler z = {euler_call:.2}");
    }

    #[test]
    fn qe_matches_quantlib_with_daily_steps() {
        for z in z_vs_quantlib(Scheme::Qe, 76) {
            assert!(z.abs() < 3.0, "qe z = {z:.2}");
        }
    }

    #[test]
    fn qe_keeps_the_forward_a_martingale() {
        // A call struck at 1 USD is worth E[F_T] − 1 = F − 1 when the forward has the
        // right mean. One year in quarterly steps: without the martingale correction the
        // mean forward drifts by about 470 USD (z ≈ 7 at these paths).
        let e = price(&heston(Scheme::Qe), F, 1.0, 4, &call(1.0), &sim(200_000)).unwrap();
        let z = (e.price - (F - 1.0)) / e.std_err;
        assert!(
            z.abs() < 3.0,
            "mean forward {} vs {F}: z = {z:.2}",
            e.price + 1.0
        );
    }

    #[test]
    fn qe_rejects_zero_vol_of_vol() {
        let flat = Model::Heston {
            params: Heston { xi: 0.0, ..BTC },
            scheme: Scheme::Qe,
        };
        assert!(price(&flat, F, T, 10, &call(72_000.0), &sim(10)).is_err());
    }

    #[test]
    fn knock_out_is_cheaper_than_the_vanilla_and_converges_to_it() {
        let gbm = Model::Gbm { sigma: SIGMA };
        let vanilla = price(&gbm, F, T, 90, &call(72_000.0), &sim(100_000)).unwrap();
        let ko = price(
            &gbm,
            F,
            T,
            90,
            &Payoff::UpAndOutCall {
                k: 72_000.0,
                barrier: 90_000.0,
            },
            &sim(100_000),
        )
        .unwrap();
        let far = price(
            &gbm,
            F,
            T,
            90,
            &Payoff::UpAndOutCall {
                k: 72_000.0,
                barrier: 1e9,
            },
            &sim(100_000),
        )
        .unwrap();
        assert!(ko.price < vanilla.price);
        // Same seed and steps: an unreachable barrier gives exactly the vanilla.
        assert_eq!(far.price, vanilla.price);
    }

    #[test]
    fn finer_monitoring_lowers_the_knock_out_price() {
        // Checking the barrier less often misses touches between dates.
        let gbm = Model::Gbm { sigma: SIGMA };
        let ko = Payoff::UpAndOutCall {
            k: 72_000.0,
            barrier: 85_000.0,
        };
        let daily = price(&gbm, F, T, 91, &ko, &sim(200_000)).unwrap();
        let hourly = price(&gbm, F, T, 91 * 24, &ko, &sim(200_000)).unwrap();
        assert!(hourly.price < daily.price - 3.0 * daily.std_err);
    }

    #[test]
    fn asian_call_is_cheaper_than_the_vanilla() {
        // Averaging lowers the effective volatility. Same seed for both, so the gap is
        // not noise: an Asian payoff that used the last price would match exactly.
        let gbm = Model::Gbm { sigma: SIGMA };
        let k = 68_000.0;
        let asian = price(&gbm, F, T, 90, &Payoff::AsianCall { k }, &sim(100_000)).unwrap();
        let vanilla = price(&gbm, F, T, 90, &call(k), &sim(100_000)).unwrap();
        assert!(
            asian.price < 0.8 * vanilla.price,
            "asian {} vs vanilla {}",
            asian.price,
            vanilla.price
        );
    }

    #[test]
    fn rejects_bad_inputs() {
        let gbm = Model::Gbm { sigma: SIGMA };
        assert!(price(&gbm, F, T, 0, &call(1.0), &sim(10)).is_err());
        let bad_ko = Payoff::UpAndOutCall {
            k: 72_000.0,
            barrier: 60_000.0,
        };
        assert!(price(&gbm, F, T, 10, &bad_ko, &sim(10)).is_err());
        let bad_rho = Model::Heston {
            params: Heston {
                v0: 0.1,
                kappa: 1.0,
                theta: 0.1,
                xi: 0.5,
                rho: -1.5,
            },
            scheme: Scheme::Euler,
        };
        assert!(price(&bad_rho, F, T, 10, &call(1.0), &sim(10)).is_err());
    }
}
