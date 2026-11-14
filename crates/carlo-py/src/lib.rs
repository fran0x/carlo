//! Python bindings for the `carlo` crate (step 7).
//!
//! Build into the active environment with `just py-build`
//! (`maturin develop --release`). Every function releases the GIL while it computes, so
//! Python threads keep running and the Rust side can use every core.

use carlo::paths::{self, Heston, Model, Payoff, Scheme};
use carlo::var::{self, Instrument, Position, Shocks, VarConfig};
use carlo::{Exec, McError, Right, Sim, Vanilla, gbm};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

fn py_err(e: McError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn exec(parallel: bool) -> Exec {
    if parallel {
        Exec::Parallel
    } else {
        Exec::Sequential
    }
}

fn right(call: bool) -> Right {
    if call { Right::Call } else { Right::Put }
}

/// European option under GBM. Returns `(price, std_err)`.
#[pyfunction]
#[pyo3(signature = (f, k, t, sigma, call=true, paths=1_000_000, seed=42, parallel=true))]
#[allow(clippy::too_many_arguments)]
fn gbm_european(
    py: Python<'_>,
    f: f64,
    k: f64,
    t: f64,
    sigma: f64,
    call: bool,
    paths: u64,
    seed: u64,
    parallel: bool,
) -> PyResult<(f64, f64)> {
    let opt = Vanilla {
        k,
        t,
        right: right(call),
    };
    let sim = Sim {
        paths,
        seed,
        exec: exec(parallel),
    };
    let e = py
        .detach(|| gbm::european(f, &opt, sigma, &sim))
        .map_err(py_err)?;
    Ok((e.price, e.std_err))
}

/// A whole book of European options in one call: one crossing of the Python–Rust
/// boundary instead of one per option. Inputs are NumPy arrays of equal length.
/// Returns `(prices, std_errs)` as arrays.
#[pyfunction]
#[pyo3(signature = (f, k, t, sigma, call, paths=1_000_000, seed=42))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn gbm_european_book<'py>(
    py: Python<'py>,
    f: PyReadonlyArray1<'py, f64>,
    k: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    call: PyReadonlyArray1<'py, bool>,
    paths: u64,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (f, k, t, sigma, call) = (
        f.as_slice()?,
        k.as_slice()?,
        t.as_slice()?,
        sigma.as_slice()?,
        call.as_slice()?,
    );
    let n = f.len();
    if [k.len(), t.len(), sigma.len(), call.len()]
        .iter()
        .any(|&len| len != n)
    {
        return Err(PyValueError::new_err(
            "all arrays must have the same length",
        ));
    }
    // Options run in parallel, each one sequentially: same numbers, no nested pools.
    let sim = Sim {
        paths,
        seed,
        exec: Exec::Sequential,
    };
    let results: Result<Vec<_>, McError> = py.detach(|| {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let opt = Vanilla {
                    k: k[i],
                    t: t[i],
                    right: right(call[i]),
                };
                gbm::european(f[i], &opt, sigma[i], &sim)
            })
            .collect()
    });
    let results = results.map_err(py_err)?;
    let prices: Vec<f64> = results.iter().map(|e| e.price).collect();
    let errs: Vec<f64> = results.iter().map(|e| e.std_err).collect();
    Ok((PyArray1::from_vec(py, prices), PyArray1::from_vec(py, errs)))
}

/// Path-dependent payoff under Heston. `payoff` is one of `"call"`, `"put"`,
/// `"up_and_out_call"` (needs `barrier`) or `"asian_call"`; `scheme` is `"euler"`
/// (full truncation) or `"qe"` (Andersen). Returns `(price, std_err)`.
#[pyfunction]
#[pyo3(signature = (f, t, steps, v0, kappa, theta, xi, rho, payoff, k, barrier=None, paths=200_000, seed=42, parallel=true, scheme="euler"))]
#[allow(clippy::too_many_arguments)]
fn heston_price(
    py: Python<'_>,
    f: f64,
    t: f64,
    steps: u32,
    v0: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    payoff: &str,
    k: f64,
    barrier: Option<f64>,
    paths: u64,
    seed: u64,
    parallel: bool,
    scheme: &str,
) -> PyResult<(f64, f64)> {
    let scheme = match scheme {
        "euler" => Scheme::Euler,
        "qe" => Scheme::Qe,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown scheme {other:?}; use \"euler\" or \"qe\""
            )));
        }
    };
    let payoff = match (payoff, barrier) {
        ("call", None) => Payoff::European {
            k,
            right: Right::Call,
        },
        ("put", None) => Payoff::European {
            k,
            right: Right::Put,
        },
        ("asian_call", None) => Payoff::AsianCall { k },
        ("up_and_out_call", Some(barrier)) => Payoff::UpAndOutCall { k, barrier },
        ("up_and_out_call", None) => {
            return Err(PyValueError::new_err("up_and_out_call needs a barrier"));
        }
        ("call" | "put" | "asian_call", Some(_)) => {
            return Err(PyValueError::new_err(format!(
                "{payoff} takes no barrier; use up_and_out_call"
            )));
        }
        (other, _) => {
            return Err(PyValueError::new_err(format!("unknown payoff {other:?}")));
        }
    };
    let model = Model::Heston {
        params: Heston {
            v0,
            kappa,
            theta,
            xi,
            rho,
        },
        scheme,
    };
    let sim = Sim {
        paths,
        seed,
        exec: exec(parallel),
    };
    let e = py
        .detach(|| paths::price(&model, f, t, steps, &payoff, &sim))
        .map_err(py_err)?;
    Ok((e.price, e.std_err))
}

/// The sample book used by the VaR benchmark, as a list of dicts with keys `qty`,
/// `kind` (`"call"`, `"put"` or `"up_and_out_call"`), `k`, `barrier` (or None), `t`, `iv`.
#[pyfunction]
fn sample_book<'py>(py: Python<'py>, f: f64) -> PyResult<Vec<Bound<'py, PyDict>>> {
    var::sample_book(f)
        .iter()
        .map(|p| {
            let d = PyDict::new(py);
            d.set_item("qty", p.qty)?;
            d.set_item("iv", p.iv)?;
            match p.instrument {
                Instrument::Vanilla(v) => {
                    let kind = if v.right == Right::Call {
                        "call"
                    } else {
                        "put"
                    };
                    d.set_item("kind", kind)?;
                    d.set_item("k", v.k)?;
                    d.set_item("barrier", py.None())?;
                    d.set_item("t", v.t)?;
                }
                Instrument::UpAndOutCall { k, barrier, t } => {
                    d.set_item("kind", "up_and_out_call")?;
                    d.set_item("k", k)?;
                    d.set_item("barrier", barrier)?;
                    d.set_item("t", t)?;
                }
            }
            Ok(d)
        })
        .collect()
}

/// One-day VaR of the sample book at forward `f`. Returns a dict with `base_value`,
/// `var` and `expected_shortfall` (losses as positive numbers).
#[pyfunction]
#[pyo3(signature = (f, scenarios=10_000, nested_paths=1_000, confidence=0.99, spot_vol=0.45, vol_of_vol=1.0, rho=-0.4, seed=1, parallel=true))]
#[allow(clippy::too_many_arguments)]
fn sample_book_var<'py>(
    py: Python<'py>,
    f: f64,
    scenarios: u64,
    nested_paths: u64,
    confidence: f64,
    spot_vol: f64,
    vol_of_vol: f64,
    rho: f64,
    seed: u64,
    parallel: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let book: Vec<Position> = var::sample_book(f);
    let shocks = Shocks {
        spot_vol,
        vol_of_vol,
        rho,
    };
    let cfg = VarConfig {
        scenarios,
        confidence,
        nested_paths,
        seed,
        exec: exec(parallel),
    };
    let r = py
        .detach(|| var::one_day(&book, f, &shocks, &cfg))
        .map_err(py_err)?;
    let d = PyDict::new(py);
    d.set_item("base_value", r.base_value)?;
    d.set_item("var", r.var)?;
    d.set_item("expected_shortfall", r.expected_shortfall)?;
    Ok(d)
}

#[pymodule]
fn carlo_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gbm_european, m)?)?;
    m.add_function(wrap_pyfunction!(gbm_european_book, m)?)?;
    m.add_function(wrap_pyfunction!(heston_price, m)?)?;
    m.add_function(wrap_pyfunction!(sample_book, m)?)?;
    m.add_function(wrap_pyfunction!(sample_book_var, m)?)?;
    Ok(())
}
