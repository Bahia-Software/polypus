pub mod qml_oracle;
pub mod vqc_oracle;

pub use qml_oracle::QmlOracle;
pub use vqc_oracle::VqcOracle;

use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use polypus_circuit::ParameterizedCircuit;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyModule};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// A parameterised circuit template, in one of the representations Polypus
/// supports as optimisation targets.
///
/// The variant determines *where* per-candidate parameter binding happens:
///
/// - [`Qiskit`](CircuitSource::Qiskit): `assign_parameters` is called on the
///   Python object — requires the GIL for every candidate.
/// - [`Native`](CircuitSource::Native): binding + OpenQASM 2.0 generation run
///   in pure Rust — **no GIL**, so candidates can be bound truly in parallel
///   and the only remaining Python touchpoint is the simulator call itself.
pub enum CircuitSource {
    /// A Qiskit `QuantumCircuit` with unbound `Parameter`s.
    Qiskit(Py<PyAny>),
    /// A native Rust circuit from `polypus-circuit`.
    Native(ParameterizedCircuit),
}

impl CircuitSource {
    /// Bind one candidate parameter vector, producing an executable circuit.
    ///
    /// # Panics
    ///
    /// Panics if binding fails (wrong number of parameters, Python error).
    /// Entry points validate `dimensions` against the template up front, so a
    /// failure here is a programming error, consistent with the rest of the
    /// evaluation layer.
    pub fn bind(&self, params: &[f64]) -> BoundCircuit {
        match self {
            CircuitSource::Qiskit(circuit) => {
                BoundCircuit::Qiskit(assign_parameters_qiskit(circuit, params))
            }
            // Pure Rust: no GIL anywhere on this path. The bound circuit keeps
            // its native structure so the statevector backend can simulate it
            // directly; Python backends serialise it to OpenQASM 2.0 on demand.
            CircuitSource::Native(circuit) => BoundCircuit::Native(
                circuit
                    .assign_parameters(params)
                    .unwrap_or_else(|e| panic!("Error binding native circuit: {e}")),
            ),
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
pub(crate) fn assign_parameters_qiskit(circuit: &Py<PyAny>, params: &[f64]) -> Py<PyAny> {
    Python::with_gil(|py| {
        let qc = circuit
            .clone_ref(py)
            .into_pyobject(py)
            .expect("Failed to get circuit as PyObject");
        let kwargs = [("inplace", false)].into_py_dict(py).unwrap();
        qc.call_method("assign_parameters", (params.to_vec(),), Some(&kwargs))
            .expect("Error assigning parameters to circuit")
            .unbind()
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

/// Cooperative-cancellation state shared between the `train` / `qml_train` entry
/// points and the oracle(s) they drive.
///
/// It lets a `KeyboardInterrupt` — or any other exception raised while an oracle
/// is inside Python (the user `expectation_function`, or a QNG variance
/// callback) — travel back out to the entry point **as the original
/// [`PyErr`]**, instead of being swallowed into a panic by an `.expect()` and
/// re-surfaced to Python as an opaque `PanicException`.
///
/// Why a side channel rather than the oracle's return type: the
/// [`EvaluationOracle::evaluate_batch`] contract (C-5) returns a bare
/// `Vec<f64>`, and the pure-Rust `OptimizerError` cannot carry a `PyErr` without
/// dragging PyO3 into `polypus-optimizers` (forbidden by `docs/ENGINEERING.md`
/// §2). Capturing the error here — inside the `polypus` crate, which already
/// owns `Py<...>` values — keeps the optimizer crate Python-free while still
/// delivering the exact exception to Python.
///
/// Once an error is captured the oracle short-circuits every later evaluation to
/// a cheap, correctly-sized placeholder, so the optimizer finishes its remaining
/// generations without doing any more (GIL-free) simulation; the entry point
/// discards that outcome and re-raises the captured error via [`take`](Self::take).
#[derive(Default)]
pub struct InterruptState {
    /// The first error observed during evaluation, if any.
    error: Mutex<Option<PyErr>>,
}

impl InterruptState {
    /// Record the first error observed during evaluation. Later errors are
    /// ignored so the root cause is the one propagated to Python (and so a
    /// worker seeing the flag already set does not overwrite it in a race).
    pub fn capture(&self, err: PyErr) {
        let mut slot = self.lock();
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    /// Whether an error has already been captured — the signal for an oracle to
    /// stop doing real work and return a placeholder.
    pub fn is_interrupted(&self) -> bool {
        self.lock().is_some()
    }

    /// Remove and return the captured error, if any, so the entry point can
    /// re-raise it once the optimizer has returned.
    pub fn take(&self) -> Option<PyErr> {
        self.lock().take()
    }

    /// Lock the inner slot, recovering the guard if a previous holder panicked.
    /// The critical sections here are a trivial `Option` swap that cannot panic,
    /// so poisoning is not expected; recovering the guard simply avoids an
    /// `.unwrap()` that would itself panic across the FFI boundary.
    fn lock(&self) -> MutexGuard<'_, Option<PyErr>> {
        self.error.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Execute a batch of bound circuits through `backend` and extract expectation
/// values using the Python `expectation_fn`.
///
/// This is the **single place** in the codebase that calls
/// `polypus_python.expectation_values`, eliminating the duplication that
/// previously existed across DE, PSO, QNG, and the orchestration layer.
///
/// Returns a `Result` rather than a bare `Vec<f64>`: the two Python touchpoints
/// (the pending-signal check and the user `expectation_function`) can raise, and
/// their `PyErr` must reach Python **as the original exception**. Swallowing it
/// with `.expect()` would turn a real `KeyboardInterrupt` (or any user error)
/// into an opaque `PanicException`, violating the FFI error-handling rule
/// (`docs/ENGINEERING.md` §9). The caller (an oracle) captures the error into
/// its [`InterruptState`] and bails out.
pub(crate) fn run_and_expect(
    backend: &dyn QuantumBackend,
    qcs: &[BoundCircuit],
    config: &ExecutionConfig,
    expectation_fn: &Py<PyAny>,
) -> Result<Vec<f64>, PyErr> {
    let counts = backend.run_circuits(qcs, config);
    Python::with_gil(|py| {
        // Turn a pending SIGINT (Ctrl+C) into a `KeyboardInterrupt` at this safe
        // per-batch boundary. Releasing the GIL around the optimizer (see the
        // entry points) lets other Python threads run, but it does NOT by itself
        // process signals: CPython only acts on a pending signal while the main
        // thread runs Python bytecode, or when `PyErr_CheckSignals` is called
        // explicitly. This is that explicit call, so a long native-backend run
        // stays interruptible instead of ignoring Ctrl+C until it finishes (see
        // `docs/ENGINEERING.md` §3).
        py.check_signals()?;
        // Convert the native counts into a Python `list[dict]` for the Python
        // `expectation_values` function. This is pure serialization of our own
        // data and cannot raise a user exception, so a failure here is a genuine
        // invariant violation rather than something to propagate. Once
        // expectation computation is also native this round-trip disappears.
        let py_counts = counts
            .into_pyobject(py)
            .expect("Failed to convert counts to a Python object");
        // The user's `expectation_function` runs here; propagate whatever it
        // raises (a `ValueError`, a `KeyboardInterrupt`, …) as the original
        // `PyErr` rather than swallowing it into a panic.
        PyModule::import(py, "polypus_python")?
            .call_method("expectation_values", (py_counts, expectation_fn), None)?
            .extract::<Vec<f64>>()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::{PyKeyboardInterrupt, PyValueError};

    /// The side channel keeps the *first* error (the root cause), reports the
    /// interrupted state, and hands the original exception — with its Python
    /// type intact — to `take()` exactly once. This is what lets `train()`
    /// re-raise a genuine `KeyboardInterrupt` (not a `PanicException`).
    #[test]
    fn interrupt_state_captures_first_error_and_takes_it_once() {
        pyo3::prepare_freethreaded_python();

        let state = InterruptState::default();
        assert!(!state.is_interrupted());
        assert!(state.take().is_none());

        Python::with_gil(|_py| {
            state.capture(PyKeyboardInterrupt::new_err("interrupted"));
            // A later error must not clobber the root cause.
            state.capture(PyValueError::new_err("second"));
        });
        assert!(state.is_interrupted());

        Python::with_gil(|py| {
            let err = state.take().expect("an error was captured");
            assert!(
                err.is_instance_of::<PyKeyboardInterrupt>(py),
                "the first (KeyboardInterrupt) error must survive, not the later ValueError"
            );
        });

        // `take` consumed it: the state is clear again.
        assert!(!state.is_interrupted());
        assert!(state.take().is_none());
    }
}
