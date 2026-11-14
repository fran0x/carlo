//! The Monte Carlo loop shared by every pricer.
//!
//! Paths are split into a fixed number of chunks, each with its own ChaCha8 stream.
//! Because the split does not depend on the number of threads, sequential and parallel
//! runs draw exactly the same numbers and return bit-identical results.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::{McError, check};

/// Number of independent random streams. Fixed so results never depend on the
/// thread count; large enough to keep every core busy.
pub const CHUNKS: u64 = 256;

/// How to run the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exec {
    Sequential,
    Parallel,
}

/// Simulation settings: number of paths, random seed and execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sim {
    pub paths: u64,
    pub seed: u64,
    pub exec: Exec,
}

impl Sim {
    pub(crate) fn validate(&self) -> Result<(), McError> {
        check("paths", "at least 1", self.paths as f64, self.paths > 0)
    }
}

/// A Monte Carlo estimate and its standard error. Two estimates agree when they are
/// within about two standard errors of each other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    pub price: f64,
    pub std_err: f64,
}

/// The random stream for one chunk: same seed, different stream per chunk.
pub(crate) fn chunk_rng(seed: u64, chunk: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(chunk);
    rng
}

/// Runs `sample` once per path and returns the mean and its standard error.
pub(crate) fn run<F>(sim: &Sim, sample: F) -> Estimate
where
    F: Fn(&mut ChaCha8Rng) -> f64 + Sync,
{
    let chunk = |c: u64| {
        let mut rng = chunk_rng(sim.seed, c);
        let mut m = Moments::default();
        for _ in 0..paths_in_chunk(sim.paths, c) {
            m.push(sample(&mut rng));
        }
        m
    };
    let parts: Vec<Moments> = match sim.exec {
        Exec::Sequential => (0..CHUNKS).map(chunk).collect(),
        Exec::Parallel => (0..CHUNKS).into_par_iter().map(chunk).collect(),
    };
    // Merge in chunk order so the floating-point result is reproducible.
    parts
        .into_iter()
        .fold(Moments::default(), Moments::merge)
        .estimate()
}

fn paths_in_chunk(total: u64, chunk: u64) -> u64 {
    total / CHUNKS + u64::from(chunk < total % CHUNKS)
}

/// Running mean and variance (Welford), mergeable across chunks (Chan et al.).
#[derive(Debug, Clone, Copy, Default)]
struct Moments {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Moments {
    #[inline]
    fn push(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    fn merge(self, other: Self) -> Self {
        if self.n == 0 {
            return other;
        }
        if other.n == 0 {
            return self;
        }
        let n = self.n + other.n;
        let d = other.mean - self.mean;
        let w = other.n as f64 / n as f64;
        Self {
            n,
            mean: self.mean + d * w,
            m2: self.m2 + other.m2 + d * d * self.n as f64 * w,
        }
    }

    fn estimate(self) -> Estimate {
        let var = if self.n > 1 {
            self.m2 / (self.n - 1) as f64
        } else {
            0.0
        };
        Estimate {
            price: self.mean,
            std_err: (var / self.n as f64).sqrt(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_every_path_exactly_once() {
        for total in [1, 255, 256, 257, 1_000_003] {
            let sum: u64 = (0..CHUNKS).map(|c| paths_in_chunk(total, c)).sum();
            assert_eq!(sum, total);
        }
    }

    #[test]
    fn merged_moments_match_a_single_pass() {
        let xs: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.37).sin() * 10.0).collect();
        let mut whole = Moments::default();
        xs.iter().for_each(|&x| whole.push(x));
        let (a, b) = xs.split_at(337);
        let mut ma = Moments::default();
        let mut mb = Moments::default();
        a.iter().for_each(|&x| ma.push(x));
        b.iter().for_each(|&x| mb.push(x));
        let merged = ma.merge(mb);
        assert!((merged.mean - whole.mean).abs() < 1e-12);
        assert!((merged.m2 - whole.m2).abs() < 1e-9);
    }
}
