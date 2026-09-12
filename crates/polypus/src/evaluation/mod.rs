pub mod error;
pub mod py_callback_observable;
pub mod qml_oracle;
pub mod vqc_oracle;

pub use error::EvaluationError;
pub use py_callback_observable::PyCallbackObservable;
pub use qml_oracle::QmlOracle;
pub use vqc_oracle::VqcOracle;

/// Re-export the native cost-observable seam so `crate::evaluation::CostObservable`
/// resolves alongside the oracles that consume it.
pub use polypus_observable::CostObservable;

use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use polypus_circuit::ParameterizedCircuit;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

/// Re-export the type-erased error slot from the pure `polypus-scheduler` crate,
/// where it now lives — it is shared with `dispatch_optimizer`, and the scheduler
/// crate cannot depend on this pyo3-touching one. The oracles below box their
/// [`EvaluationError`] into it (`slot.record(Box::new(err), id)`); the FFI edge
/// downcasts it back to re-raise the original exception. Re-exported here so the
/// existing `crate::evaluation::OracleErrorSlot` path keeps resolving.
pub use polypus_scheduler::OracleErrorSlot;

/// A parameterised circuit template, in one of the representations Polypus
/// supports as optimisation targets.
///
/// The variant determines *where* per-candidate parameter binding happens:
///
/// - [`Qiskit`](CircuitSource::Qiskit): `assign_parameters` is called on the
///   Python object — requires the GIL for every candidate.
/// - [`Native`](CircuitSource::Native): binding + OpenQASM 2.0 generation run
///   in pure Rust — **no GIL**, so candidates can be bound without holding the
///   interpreter lock (binding itself is still sequential today, not
///   parallel — see `VqcOracle::try_evaluate`) and the only remaining Python
///   touchpoint is the simulator call itself.
#[derive(Debug)]
pub enum CircuitSource {
    /// A Qiskit `QuantumCircuit` with unbound `Parameter`s.
    Qiskit(Py<PyAny>),
    /// A native Rust circuit from `polypus-circuit`.
    Native(ParameterizedCircuit),
}

impl CircuitSource {
    /// Bind one candidate parameter vector, producing an executable circuit.
    ///
    /// Returns an [`EvaluationError`] on failure (wrong parameter count, a
    /// Python error binding a Qiskit circuit) rather than panicking, so the
    /// failure can cross the FFI as a typed exception. Entry points still
    /// validate `dimensions` up front, so a failure here is normally
    /// unreachable — but it is reported, never a panic.
    pub fn bind(&self, params: &[f64]) -> Result<BoundCircuit, EvaluationError> {
        match self {
            CircuitSource::Qiskit(circuit) => Ok(BoundCircuit::Qiskit(assign_parameters_qiskit(
                circuit, params,
            )?)),
            // Pure Rust: no GIL anywhere on this path. The bound circuit keeps
            // its native structure so the statevector backend can simulate it
            // directly; Python backends serialise it to OpenQASM 2.0 on demand.
            CircuitSource::Native(circuit) => Ok(BoundCircuit::Native(
                circuit
                    .assign_parameters(params)
                    .map_err(EvaluationError::Binding)?,
            )),
        }
    }

    /// Number of free parameters, when it can be known without Python
    /// (`None` for Qiskit circuits — querying them needs the GIL and is done
    /// at the entry points instead).
    pub fn num_params(&self) -> Option<usize> {
        match self {
            CircuitSource::Qiskit(_) => None,
            CircuitSource::Native(c) => Some(c.num_params),
        }
    }
}

/// Bind `params` to a copy of a Qiskit `circuit` and return the bound circuit.
///
/// Any Python error (constructing the kwargs, calling `assign_parameters`) is
/// returned as [`EvaluationError::Python`] — carried verbatim so the caller can
/// re-raise it with its original type across the FFI.
pub(crate) fn assign_parameters_qiskit(
    circuit: &Py<PyAny>,
    params: &[f64],
) -> Result<Py<PyAny>, EvaluationError> {
    Python::with_gil(|py| {
        let qc = circuit
            .clone_ref(py)
            .into_pyobject(py)
            .map_err(|e| EvaluationError::Python(e.into()))?;
        let kwargs = [("inplace", false)]
            .into_py_dict(py)
            .map_err(EvaluationError::Python)?;
        Ok(qc
            .call_method("assign_parameters", (params.to_vec(),), Some(&kwargs))
            .map_err(EvaluationError::Python)?
            .unbind())
    })
}

/// Contract between optimization algorithms and quantum circuit evaluation.
///
/// Re-exported from the pure-Rust [`polypus_optimizers`] crate, where the trait
/// now lives (it is the optimizers' input contract). Re-exporting here keeps the
/// `crate::evaluation::EvaluationOracle` path — used by [`VqcOracle`] and
/// [`QmlOracle`] — resolving unchanged.
///
/// An oracle encapsulates everything needed to translate a parameter vector
/// into a scalar fitness value: the circuit template (or training circuits),
/// the backend, and the expectation function.
///
/// Algorithms only call [`EvaluationOracle::evaluate_batch`] and have no
/// knowledge of circuits, QPUs, infrastructure, or training modes.
///
/// To add a new evaluation strategy (e.g. noisy readout mitigation, hardware
/// native gates, …) implement this trait without touching any algorithm.
pub use polypus_optimizers::EvaluationOracle;

/// Execute a batch of bound circuits through `backend` and reduce the resulting
/// counts to expectation values via `observable`.
///
/// This is the **single place** in the codebase that turns measurement counts
/// into fitness values, shared by [`VqcOracle`] and [`QmlOracle`]. The
/// expectation is computed natively (rayon, no GIL) for the declarative
/// observables, or via a single deduplicated GIL section for the Python-callback
/// fallback — replacing the former per-bitstring `polypus_python.expectation_values`
/// round-trip (which also required serialising the counts into a `list[dict]`).
///
/// Returns an [`EvaluationError`] on any failure: a backend error is wrapped; a
/// native-evaluation error (bad bitstring width/char, invalid construction) maps
/// to a typed exception via [`EvaluationError::Observable`]; and a Python
/// callback error is carried verbatim (the callback observable boxes its `PyErr`
/// in [`ObservableError::External`], recovered on the way out). The resulting
/// batch is finally checked against contract C-5 — exactly one finite `f64` per
/// submitted circuit — surfacing [`EvaluationError::WrongLength`] or
/// [`EvaluationError::NonFinite`] instead of letting a short or non-finite
/// result poison the pure-Rust optimizer. Never a panic.
pub(crate) fn run_and_evaluate(
    backend: &dyn QuantumBackend,
    qcs: &[BoundCircuit],
    config: &ExecutionConfig,
    observable: &dyn CostObservable,
) -> Result<Vec<f64>, EvaluationError> {
    let counts = backend.run_circuits(qcs, config)?;
    // Central result validation (contract C-3 + empty-map guard): one map per
    // circuit, each non-empty and conserving the requested shots. An empty map
    // would otherwise reduce to a silent 0.0 fitness with no error at all.
    crate::infrastructure::validate_run_results(&counts, qcs.len(), config.shots)
        .map_err(EvaluationError::Backend)?;
    // Turn a pending SIGINT (Ctrl+C) into a `KeyboardInterrupt` at this safe
    // per-batch boundary. The optimizer entry points release the GIL around
    // `optimize()`, which lets other Python threads run but does NOT by itself
    // process signals: CPython only acts on a pending signal while the main
    // thread runs Python bytecode or when `PyErr_CheckSignals` is called
    // explicitly. This is that explicit call, so a long native-backend run stays
    // interruptible (see docs/ENGINEERING.md §3). It is the *only* GIL touch on
    // the native path; the aggregation below runs GIL-free (the callback
    // observable re-acquires the GIL internally for one deduplicated section).
    Python::with_gil(|py| py.check_signals()).map_err(EvaluationError::Python)?;
    let values = observable
        .expectation_batch(&counts)
        .map_err(EvaluationError::from)?;

    // Contract C-5: the oracle must return exactly one finite f64 per submitted
    // circuit. This is the single choke point that reduces counts to fitness, so
    // validating here protects every oracle: a short batch would otherwise index
    // out of bounds inside the pure-Rust optimizer (an uncatchable
    // `PanicException` across the FFI), and a NaN/inf would silently poison the
    // optimizer and yield a bogus result with no error at all. The native
    // observables guarantee this structurally; the Python-callback fallback does
    // not (a user cost function may return a non-finite value).
    if values.len() != qcs.len() {
        return Err(EvaluationError::WrongLength {
            expected: qcs.len(),
            got: values.len(),
        });
    }
    if let Some((index, &value)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(EvaluationError::NonFinite { index, value });
    }
    Ok(values)
}
