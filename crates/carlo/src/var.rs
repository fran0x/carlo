//! One-day Value at Risk of a small options book by full revaluation (step 6).
//!
//! 1. Simulate `scenarios` one-day moves of the forward and of implied volatility
//!    (correlated lognormal shocks).
//! 2. Revalue every position in every scenario: vanillas with Black-76, knock-outs with a
//!    nested Monte Carlo (the expensive part).
//! 3. VaR is the loss at the chosen confidence level; expected shortfall is the average
//!    loss beyond it.
//!
//! The nested pricer uses common random numbers: one matrix of daily normals for the days
//! after the horizon is drawn up front and every valuation uses it. Each scenario values
//! the knock-out one day later on those paths. Today's value is the expectation of
//! tomorrow's over the first day's move, `V(F, t) = E[V(F·e^x, t − 1d)]`, computed by
//! Gauss–Hermite quadrature on the same paths. Base and scenarios then share all their
//! randomness, so the noise cancels in the profit and loss. Valuing today on its own paths
//! instead (even days 1..n of the same matrix) leaves a constant noise term, several
//! hundred USD at 500 nested paths for the sample book, that shifts the whole VaR.

use rand::RngExt;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::engine::chunk_rng;
use crate::{Exec, McError, Right, Vanilla, black76, check, positive};

const DAY: f64 = 1.0 / 365.0;

/// What a position holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instrument {
    Vanilla(Vanilla),
    /// Up-and-out call with strike `k`, barrier and expiry `t`, monitored daily.
    UpAndOutCall {
        k: f64,
        barrier: f64,
        t: f64,
    },
}

/// A position: signed quantity (negative = sold) of an instrument at implied vol `iv`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub qty: f64,
    pub instrument: Instrument,
    pub iv: f64,
}

/// How the market moves over one day.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shocks {
    /// Annualised volatility of the forward.
    pub spot_vol: f64,
    /// Annualised volatility of implied volatility (log changes).
    pub vol_of_vol: f64,
    /// Correlation between forward returns and implied-vol changes (usually negative).
    pub rho: f64,
}

/// VaR settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarConfig {
    pub scenarios: u64,
    /// For example 0.99 for 99% VaR.
    pub confidence: f64,
    /// Paths per nested knock-out valuation.
    pub nested_paths: u64,
    pub seed: u64,
    /// Parallel runs the scenarios on all cores.
    pub exec: Exec,
}

/// VaR result. Losses are positive numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarReport {
    pub base_value: f64,
    pub var: f64,
    pub expected_shortfall: f64,
}

/// Daily standard normals for the nested knock-out valuations (the days after the horizon),
/// stored as running sums per path so that a barrier check needs no `exp` per step (it is
/// done in log space).
#[derive(Debug, Clone)]
struct NestedPaths {
    paths: usize,
    days: usize,
    /// `cum[i * days + j]` = z[i][0] + … + z[i][j].
    cum: Vec<f64>,
}

impl NestedPaths {
    /// Draws `paths` paths of `days` daily normals, one random stream per path.
    fn new(paths: u64, days: usize, seed: u64) -> Self {
        let mut cum = Vec::with_capacity(paths as usize * days);
        for i in 0..paths {
            let mut rng = chunk_rng(seed, i);
            let mut s = 0.0;
            for _ in 0..days {
                let z: f64 = rng.sample(StandardNormal);
                s += z;
                cum.push(s);
            }
        }
        Self {
            paths: paths as usize,
            days,
            cum,
        }
    }

    /// Up-and-out call under GBM with daily monitoring over the first `steps` days of
    /// every path. Zero if `f` is already at or above the barrier.
    fn knock_out(&self, f: f64, k: f64, barrier: f64, sigma: f64, steps: usize) -> f64 {
        assert!(steps <= self.days, "nested paths are too short");
        if f >= barrier {
            return 0.0;
        }
        let drift = -0.5 * sigma * sigma * DAY;
        let vol = sigma * DAY.sqrt();
        let log_barrier = (barrier / f).ln();
        let mut sum = 0.0;
        for path in self.cum.chunks_exact(self.days.max(1)) {
            let mut y = 0.0;
            let mut alive = true;
            for (j, &c) in path[..steps].iter().enumerate() {
                y = (j + 1) as f64 * drift + vol * c;
                if y >= log_barrier {
                    alive = false;
                    break;
                }
            }
            if alive {
                sum += (f * y.exp() - k).max(0.0);
            }
        }
        sum / self.paths as f64
    }
}

/// Monitoring days of a knock-out expiring at `t`; at least 1 for `t` past the horizon.
fn days(t: f64) -> usize {
    (t / DAY).round().max(1.0) as usize
}

/// Probabilists' Gauss–Hermite rule, 7 points: `E[g(Z)] ≈ Σ w·g(x)` for `Z ~ N(0, 1)`.
const HERMITE: [(f64, f64); 7] = [
    (-2.651_961_356_835_233, 0.000_548_268_855_972_218_4),
    (-1.673_551_628_767_471_4, 0.030_757_123_967_586_5),
    (-0.816_287_882_858_964_7, 0.240_123_178_605_012_75),
    (0.0, 0.457_142_857_142_857_1),
    (0.816_287_882_858_964_7, 0.240_123_178_605_012_75),
    (1.673_551_628_767_471_4, 0.030_757_123_967_586_5),
    (2.651_961_356_835_233, 0.000_548_268_855_972_218_4),
];

/// Value of the book today at forward `f`. Knock-outs are valued as the expectation of
/// tomorrow's value over the first day's move (see the module docs).
fn value_today(book: &[Position], f: f64, nested: &NestedPaths) -> Result<f64, McError> {
    book.iter().try_fold(0.0, |acc, p| {
        let value = match p.instrument {
            Instrument::Vanilla(v) => black76::price(f, &v, p.iv)?,
            Instrument::UpAndOutCall { k, barrier, t } => {
                let (drift, vol) = (-0.5 * p.iv * p.iv * DAY, p.iv * DAY.sqrt());
                let steps = days(t) - 1;
                HERMITE
                    .iter()
                    .map(|&(x, w)| {
                        let tomorrow = f * (drift + vol * x).exp();
                        w * nested.knock_out(tomorrow, k, barrier, p.iv, steps)
                    })
                    .sum()
            }
        };
        Ok(acc + p.qty * value)
    })
}

/// Value of the book one day from today at forward `f`, with every implied vol multiplied
/// by `vol_scale`.
fn value_tomorrow(
    book: &[Position],
    f: f64,
    vol_scale: f64,
    nested: &NestedPaths,
) -> Result<f64, McError> {
    book.iter().try_fold(0.0, |acc, p| {
        let iv = p.iv * vol_scale;
        let value = match p.instrument {
            Instrument::Vanilla(v) => {
                let v = Vanilla { t: v.t - DAY, ..v };
                black76::price(f, &v, iv)?
            }
            Instrument::UpAndOutCall { k, barrier, t } => {
                nested.knock_out(f, k, barrier, iv, days(t) - 1)
            }
        };
        Ok(acc + p.qty * value)
    })
}

/// One-day VaR and expected shortfall of `book` at forward `f`.
pub fn one_day(
    book: &[Position],
    f: f64,
    shocks: &Shocks,
    cfg: &VarConfig,
) -> Result<VarReport, McError> {
    positive("forward", f)?;
    check(
        "confidence",
        "strictly between 0 and 1",
        cfg.confidence,
        cfg.confidence > 0.0 && cfg.confidence < 1.0,
    )?;
    check(
        "scenarios",
        "at least 1",
        cfg.scenarios as f64,
        cfg.scenarios > 0,
    )?;
    check(
        "rho",
        "between -1 and 1",
        shocks.rho,
        (-1.0..=1.0).contains(&shocks.rho),
    )?;
    for p in book {
        let t = match p.instrument {
            Instrument::Vanilla(v) => v.t,
            Instrument::UpAndOutCall { t, .. } => t,
        };
        check("expiry", "longer than the one-day horizon", t, t > DAY)?;
    }

    check(
        "nested_paths",
        "at least 1",
        cfg.nested_paths as f64,
        cfg.nested_paths > 0,
    )?;
    let longest = book
        .iter()
        .filter_map(|p| match p.instrument {
            Instrument::UpAndOutCall { t, .. } => Some(days(t) - 1),
            Instrument::Vanilla(_) => None,
        })
        .max()
        .unwrap_or(0);
    let nested = NestedPaths::new(cfg.nested_paths, longest, cfg.seed ^ 0x9E37_79B9_7F4A_7C15);
    let base = value_today(book, f, &nested)?;

    let sd_f = shocks.spot_vol * DAY.sqrt();
    let sd_v = shocks.vol_of_vol * DAY.sqrt();
    let rho_c = (1.0 - shocks.rho * shocks.rho).sqrt();
    let scenario = |i: u64| -> Result<f64, McError> {
        let mut rng = chunk_rng(cfg.seed, i);
        let z1: f64 = rng.sample(StandardNormal);
        let w: f64 = rng.sample(StandardNormal);
        let z2 = shocks.rho * z1 + rho_c * w;
        let f1 = f * (-0.5 * sd_f * sd_f + sd_f * z1).exp();
        let vol_scale = (-0.5 * sd_v * sd_v + sd_v * z2).exp();
        Ok(value_tomorrow(book, f1, vol_scale, &nested)? - base)
    };
    let mut pnl: Vec<f64> = match cfg.exec {
        Exec::Sequential => (0..cfg.scenarios).map(scenario).collect::<Result<_, _>>()?,
        Exec::Parallel => (0..cfg.scenarios)
            .into_par_iter()
            .map(scenario)
            .collect::<Result<_, _>>()?,
    };

    pnl.sort_by(f64::total_cmp);
    let tail = (((1.0 - cfg.confidence) * pnl.len() as f64).ceil() as usize).max(1);
    let worst = &pnl[..tail];
    Ok(VarReport {
        base_value: base,
        var: -worst[tail - 1],
        expected_shortfall: -worst.iter().sum::<f64>() / tail as f64,
    })
}

/// A sample book used by the benchmarks and the Python comparison: a short strangle, a
/// long call spread and a long knock-out call on BTC at `f`, expiring in about 76 days.
pub fn sample_book(f: f64) -> Vec<Position> {
    let t = 76.0 / 365.0;
    let vanilla = |qty: f64, k_rel: f64, right: Right, iv: f64| Position {
        qty,
        instrument: Instrument::Vanilla(Vanilla {
            k: (f * k_rel).round(),
            t,
            right,
        }),
        iv,
    };
    vec![
        vanilla(-10.0, 0.85, Right::Put, 0.42),
        vanilla(-10.0, 1.15, Right::Call, 0.39),
        vanilla(5.0, 1.05, Right::Call, 0.38),
        vanilla(-5.0, 1.20, Right::Call, 0.40),
        Position {
            qty: 20.0,
            instrument: Instrument::UpAndOutCall {
                k: (f * 1.05).round(),
                barrier: (f * 1.30).round(),
                t,
            },
            iv: 0.38,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const F: f64 = 67_931.42;
    const SHOCKS: Shocks = Shocks {
        spot_vol: 0.45,
        vol_of_vol: 1.0,
        rho: -0.4,
    };

    fn cfg(exec: Exec) -> VarConfig {
        VarConfig {
            scenarios: 2_000,
            confidence: 0.99,
            nested_paths: 500,
            seed: 1,
            exec,
        }
    }

    #[test]
    fn sequential_and_parallel_agree() {
        let book = sample_book(F);
        let seq = one_day(&book, F, &SHOCKS, &cfg(Exec::Sequential)).unwrap();
        let par = one_day(&book, F, &SHOCKS, &cfg(Exec::Parallel)).unwrap();
        assert_eq!(seq, par);
    }

    #[test]
    fn expected_shortfall_is_at_least_var_and_var_is_a_loss() {
        let r = one_day(&sample_book(F), F, &SHOCKS, &cfg(Exec::Parallel)).unwrap();
        assert!(r.var > 0.0);
        assert!(r.expected_shortfall >= r.var);
    }

    #[test]
    fn a_single_long_call_matches_the_delta_gamma_approximation() {
        // For a small move, P&L ≈ delta·dF + ½·gamma·dF² + vega·dσ + theta. With no vol
        // shocks, the 99% loss of a long call is close to delta × the 1st-percentile move.
        let opt = Vanilla {
            k: 72_000.0,
            t: 0.21,
            right: Right::Call,
        };
        let book = [Position {
            qty: 1.0,
            instrument: Instrument::Vanilla(opt),
            iv: 0.40,
        }];
        let shocks = Shocks {
            spot_vol: 0.45,
            vol_of_vol: 0.0,
            rho: 0.0,
        };
        let cfg = VarConfig {
            scenarios: 200_000,
            ..cfg(Exec::Parallel)
        };
        let r = one_day(&book, F, &shocks, &cfg).unwrap();
        let g = black76::greeks(F, &opt, 0.40).unwrap();
        let move_99 = F * (1.0 - (-2.326 * 0.45 * DAY.sqrt()).exp());
        let approx = g.delta * move_99 - 0.5 * g.gamma * move_99 * move_99 - g.theta;
        assert!(
            (r.var - approx).abs() / approx < 0.03,
            "var {} vs {approx}",
            r.var
        );
    }

    #[test]
    fn nested_knock_out_matches_the_path_pricer() {
        use crate::paths::{self, Model, Payoff};
        use crate::{Exec, Sim};
        let (k, barrier, sigma) = (72_000.0, 90_000.0, 0.40);
        let nested = NestedPaths::new(200_000, 76, 3).knock_out(F, k, barrier, sigma, 76);
        let sim = Sim {
            paths: 200_000,
            seed: 4,
            exec: Exec::Parallel,
        };
        let payoff = Payoff::UpAndOutCall { k, barrier };
        let reference =
            paths::price(&Model::Gbm { sigma }, F, 76.0 * DAY, 76, &payoff, &sim).unwrap();
        let z = (nested - reference.price) / (reference.std_err * 2f64.sqrt());
        assert!(
            z.abs() < 3.0,
            "nested {nested} vs {}: z = {z:.2}",
            reference.price
        );
    }

    #[test]
    fn without_market_moves_the_loss_is_one_day_of_decay() {
        // With no shocks every scenario revalues identically and the P&L must be the
        // book's one-day decay, computed here independently: Black-76 for the vanillas,
        // the path pricer (2M paths) for the knock-out. Over 8 seeds the mean must match
        // and the spread must be small: valuing today on paths of its own leaves the full
        // nested-MC noise in every P&L (a spread of several hundred USD at 20k paths).
        use crate::Sim;
        use crate::paths::{self, Model, Payoff};
        let calm = Shocks {
            spot_vol: 0.0,
            vol_of_vol: 0.0,
            rho: 0.0,
        };
        let book = sample_book(F);
        let pnl: Vec<f64> = (0..8)
            .map(|seed| {
                let cfg = VarConfig {
                    scenarios: 10,
                    nested_paths: 20_000,
                    seed,
                    ..cfg(Exec::Parallel)
                };
                let r = one_day(&book, F, &calm, &cfg).unwrap();
                assert_eq!(r.var, r.expected_shortfall);
                -r.var
            })
            .collect();
        let mean = pnl.iter().sum::<f64>() / 8.0;
        let spread = (pnl.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / 7.0).sqrt();

        let sim = Sim {
            paths: 2_000_000,
            seed: 99,
            exec: Exec::Parallel,
        };
        let decay: f64 = book
            .iter()
            .map(|p| {
                let change = match p.instrument {
                    Instrument::Vanilla(v) => {
                        let later = Vanilla { t: v.t - DAY, ..v };
                        black76::price(F, &later, p.iv).unwrap()
                            - black76::price(F, &v, p.iv).unwrap()
                    }
                    Instrument::UpAndOutCall { k, barrier, t } => {
                        let n = days(t) as u32;
                        let gbm = Model::Gbm { sigma: p.iv };
                        let ko = Payoff::UpAndOutCall { k, barrier };
                        let value = |steps: u32| {
                            paths::price(&gbm, F, f64::from(steps) * DAY, steps, &ko, &sim)
                                .unwrap()
                                .price
                        };
                        value(n - 1) - value(n)
                    }
                };
                p.qty * change
            })
            .sum();
        assert!(
            (mean - decay).abs() < 150.0,
            "mean one-day P&L {mean:.0} vs independent decay {decay:.0}"
        );
        assert!(spread < 200.0, "P&L spread across seeds {spread:.0}");
    }

    #[test]
    fn rejects_bad_config() {
        let book = sample_book(F);
        let bad = VarConfig {
            confidence: 1.0,
            ..cfg(Exec::Parallel)
        };
        assert!(one_day(&book, F, &SHOCKS, &bad).is_err());
    }
}
