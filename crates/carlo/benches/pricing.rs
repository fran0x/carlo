//! Criterion benchmarks: `cargo bench -p carlo` (or `just bench-rs`).
//!
//! Thread count is controlled with `RAYON_NUM_THREADS`, compiler settings with
//! `RUSTFLAGS="-C target-cpu=native"` and `--profile release-fast-build`.

use std::hint::black_box;

use carlo::paths::{self, Heston, Model, Payoff, Scheme};
use carlo::var::{self, Shocks, VarConfig};
use carlo::{Exec, Right, Sim, Vanilla, gbm};
use criterion::{Criterion, criterion_group, criterion_main};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

const F: f64 = 67_931.42;

fn european(c: &mut Criterion) {
    let opt = Vanilla {
        k: 72_000.0,
        t: 0.21,
        right: Right::Call,
    };
    let mut g = c.benchmark_group("gbm_european_1m_paths");
    for (name, exec) in [
        ("sequential", Exec::Sequential),
        ("parallel", Exec::Parallel),
    ] {
        let sim = Sim {
            paths: 1_000_000,
            seed: 42,
            exec,
        };
        g.bench_function(name, |b| {
            b.iter(|| gbm::european(black_box(F), &opt, 0.3833, &sim).unwrap())
        });
    }
    // The sequential loop with SmallRng instead of ChaCha8: isolates the generator's cost.
    // Not in the library, which needs ChaCha8's independent streams for bit-identical
    // sequential and parallel results.
    g.bench_function("sequential_small_rng", |b| {
        b.iter(|| {
            let (drift, vol) = (-0.5 * 0.3833_f64.powi(2) * opt.t, 0.3833 * opt.t.sqrt());
            let mut rng = SmallRng::seed_from_u64(black_box(42));
            let sum: f64 = (0..1_000_000)
                .map(|_| {
                    let z: f64 = rng.sample(StandardNormal);
                    opt.payoff(black_box(F) * (drift + vol * z).exp())
                })
                .sum();
            sum / 1e6
        })
    });
    g.finish();
}

fn heston_knock_out(c: &mut Criterion) {
    let params = Heston {
        v0: 0.16,
        kappa: 2.0,
        theta: 0.2,
        xi: 1.2,
        rho: -0.3,
    };
    let payoff = Payoff::UpAndOutCall {
        k: 72_000.0,
        barrier: 90_000.0,
    };
    let mut g = c.benchmark_group("heston_knock_out_100k_paths_76_steps");
    g.sample_size(20);
    for (scheme_name, scheme) in [("euler", Scheme::Euler), ("qe", Scheme::Qe)] {
        let model = Model::Heston { params, scheme };
        for (exec_name, exec) in [
            ("sequential", Exec::Sequential),
            ("parallel", Exec::Parallel),
        ] {
            let sim = Sim {
                paths: 100_000,
                seed: 42,
                exec,
            };
            g.bench_function(format!("{scheme_name}_{exec_name}"), |b| {
                b.iter(|| paths::price(&model, black_box(F), 0.21, 76, &payoff, &sim).unwrap())
            });
        }
    }
    g.finish();
}

fn book_var(c: &mut Criterion) {
    let book = var::sample_book(F);
    let shocks = Shocks {
        spot_vol: 0.45,
        vol_of_vol: 1.0,
        rho: -0.4,
    };
    let mut g = c.benchmark_group("var_10k_scenarios_1k_nested_paths");
    g.sample_size(10);
    let cfg = VarConfig {
        scenarios: 10_000,
        confidence: 0.99,
        nested_paths: 1_000,
        seed: 1,
        exec: Exec::Parallel,
    };
    g.bench_function("parallel", |b| {
        b.iter(|| var::one_day(&book, black_box(F), &shocks, &cfg).unwrap())
    });
    g.finish();
}

/// Where single-threaded time goes: drawing normals. ChaCha8 (what the pricers use) is a
/// cryptographically strong generator; SmallRng (xoshiro256++) is much cheaper.
fn normals(c: &mut Criterion) {
    fn draw(rng: &mut impl Rng) -> f64 {
        (0..1_000_000)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .sum()
    }
    let mut g = c.benchmark_group("normals_1m");
    g.bench_function("chacha8", |b| {
        b.iter(|| draw(&mut ChaCha8Rng::seed_from_u64(black_box(42))))
    });
    g.bench_function("small_rng_xoshiro256pp", |b| {
        b.iter(|| draw(&mut SmallRng::seed_from_u64(black_box(42))))
    });
    g.finish();
}

criterion_group!(benches, european, heston_knock_out, book_var, normals);
criterion_main!(benches);
