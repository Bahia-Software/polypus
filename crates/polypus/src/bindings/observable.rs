//! Python bindings for the declarative cost observables.
//!
//! [`Qubo`] and [`Ising`] wrap the pure-Rust
//! [`QuboObservable`](polypus_observable::QuboObservable) /
//! [`IsingObservable`](polypus_observable::IsingObservable). The user builds one
//! of these as *data* and passes it as `expectation_function=` to
//! [`train`](crate::bindings::train); evaluation then runs natively (rayon, no
//! GIL) instead of through a per-bitstring Python callback.
//!
//! Both hold an `Arc<...>` so the entry point can hand a cheap, GIL-free clone to
//! the oracle. Construction validation surfaces as `ValueError` before any
//! optimizer or QPU work starts.

use std::sync::Arc;

use polypus_observable::{IsingObservable, QuboObservable};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A QUBO cost `f(x) = Σ_i linear[i]·x_i + Σ_(i,j) w·x_i·x_j + constant`, times
/// `scale`, over bits `x_i ∈ {0, 1}`.
///
/// Optimizers **maximise** fitness — pass `scale=-1.0` for a minimisation
/// problem. Bit ordering matches the simulator/Qiskit read-out (MSB left; the
/// right-most bit is variable 0).
#[pyclass(module = "polypus", name = "Qubo", frozen)]
pub struct Qubo {
    pub(crate) inner: Arc<QuboObservable>,
}

#[pymethods]
impl Qubo {
    /// Build from sparse coefficients.
    ///
    /// `linear`: list of `(index, coefficient)`. `quadratic`: list of
    /// `(i, j, weight)` with `i != j`. Raises `ValueError` on an out-of-range
    /// index, a self-quadratic term, or a non-finite coefficient.
    #[new]
    #[pyo3(signature = (num_vars, linear=None, quadratic=None, constant=0.0, scale=1.0))]
    fn new(
        num_vars: usize,
        linear: Option<Vec<(usize, f64)>>,
        quadratic: Option<Vec<(usize, usize, f64)>>,
        constant: f64,
        scale: f64,
    ) -> PyResult<Self> {
        let inner = QuboObservable::new(
            num_vars,
            linear.unwrap_or_default(),
            quadratic.unwrap_or_default(),
            constant,
            scale,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Build from a dense square matrix `Q`, evaluating `f(x) = xᵀ·Q·x`.
    #[staticmethod]
    #[pyo3(signature = (matrix, scale=1.0))]
    fn from_matrix(matrix: Vec<Vec<f64>>, scale: f64) -> PyResult<Self> {
        let inner = QuboObservable::from_matrix(&matrix, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Number of variables the observable is defined over.
    #[getter]
    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    fn __repr__(&self) -> String {
        format!("Qubo(num_vars={})", self.inner.num_vars())
    }
}

/// An Ising cost `f(s) = Σ_i fields[i]·z_i + Σ_(i,j) J·z_i·z_j + constant`, times
/// `scale`, over spins `z_i = 1 - 2·x_i ∈ {+1, -1}` (bit `0 → +1`, `1 → -1`).
///
/// Same sign convention and bit ordering as [`Qubo`].
#[pyclass(module = "polypus", name = "Ising", frozen)]
pub struct Ising {
    pub(crate) inner: Arc<IsingObservable>,
}

#[pymethods]
impl Ising {
    /// Build from sparse fields and couplings.
    ///
    /// `fields`: list of `(index, h)`. `couplings`: list of `(i, j, J)`. Raises
    /// `ValueError` on an out-of-range index or a non-finite coefficient.
    #[new]
    #[pyo3(signature = (num_vars, fields=None, couplings=None, constant=0.0, scale=1.0))]
    fn new(
        num_vars: usize,
        fields: Option<Vec<(usize, f64)>>,
        couplings: Option<Vec<(usize, usize, f64)>>,
        constant: f64,
        scale: f64,
    ) -> PyResult<Self> {
        let inner = IsingObservable::new(
            num_vars,
            fields.unwrap_or_default(),
            couplings.unwrap_or_default(),
            constant,
            scale,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Number of variables the observable is defined over.
    #[getter]
    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    fn __repr__(&self) -> String {
        format!("Ising(num_vars={})", self.inner.num_vars())
    }
}

/// Opt-in wrapper that enables **cross-generation** memoisation of a Python cost
/// callable.
///
/// Passing `expectation_function=polypus.CachedCost(fn)` evaluates each distinct
/// bitstring at most once across the *whole* optimization, not just once per
/// batch — worth it when the population concentrates on a shrinking set of
/// bitstrings near convergence, driving the number of Python calls per
/// generation toward zero. A bare callable (`expectation_function=fn`) still
/// deduplicates within each batch but re-evaluates across generations.
///
/// Soundness: `fn` **must be pure** — a given bitstring must always map to the
/// same value, since a cached value is reused forever. Memory grows with the
/// number of distinct bitstrings ever seen (bounded by `2**num_qubits`).
#[pyclass(module = "polypus", name = "CachedCost", frozen)]
pub struct CachedCost {
    pub(crate) cost_fn: Py<PyAny>,
}

#[pymethods]
impl CachedCost {
    /// Wrap a `bitstring -> float` callable. Raises `TypeError` if not callable.
    #[new]
    fn new(cost_fn: Bound<'_, PyAny>) -> PyResult<Self> {
        if !cost_fn.is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "CachedCost expects a callable (bitstring -> float)",
            ));
        }
        Ok(Self {
            cost_fn: cost_fn.unbind(),
        })
    }

    fn __repr__(&self) -> String {
        "CachedCost(<callable>)".to_string()
    }
}
