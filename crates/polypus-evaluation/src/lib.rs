//! # polypus-evaluation
//!
//! **How a candidate is evaluated** — the oracles that turn a parameter vector
//! into a scalar fitness. Holds [`VqcOracle`], [`QmlOracle`],
//! [`PyVarianceOracle`], the Python-callback observable [`PyCallbackObservable`],
//! the supervised QML objectives ([`SupervisedObjective`]: [`PyLabelledCost`],
//! [`PySampleCost`]), the [`CircuitSource`] binding boundary and the evaluation
//! error type.
//!
//! It also holds the [`OracleFactory`](polypus_orchestration::OracleFactory)
//! implementations — [`VqcOracleFactory`] and [`QmlOracleFactory`] — that assemble
//! those oracles for a training flow. The training *order* itself
//! ([`TrainFlow`](polypus_orchestration::TrainFlow)) lives in `polypus-orchestration`
//! with the scheduler that runs it; only the `pyo3`-touching *assembly* (binding
//! circuit templates, wiring observables) lives here.
//!
//! This crate touches `pyo3` (Qiskit binding under the GIL, Python callbacks) but
//! — like the rest of the workspace below the edge — defines no `#[pyclass]` and
//! no `From<_> for PyErr`: turning an [`EvaluationError`] into a typed `polypus.*`
//! exception is the `polypus` edge's job
//! (`polypus::exceptions::evaluation_error_to_pyerr`).

pub mod error;
pub mod py_callback_observable;
pub mod qml_oracle;
pub mod supervised;
pub mod variance_oracle;
pub mod vqc_oracle;

pub use error::EvaluationError;
pub use py_callback_observable::PyCallbackObservable;
pub use qml_oracle::{QmlObjective, QmlOracle, QmlOracleFactory};
pub use supervised::{Label, PyLabelledCost, PySampleCost, SupervisedObjective};
pub use variance_oracle::PyVarianceOracle;
pub use vqc_oracle::{VqcOracle, VqcOracleFactory};

/// Re-export the native cost-observable seam so `crate::CostObservable`
/// resolves alongside the oracles that consume it.
pub use polypus_observable::CostObservable;

use polypus_circuit::ParameterizedCircuit;
use polypus_infrastructure::{attach_or, BoundCircuit, InterpreterUnavailable, QiskitCircuit};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

/// Re-export the type-erased error slot from the pure `polypus-orchestration`
/// crate, where it now lives — it is shared with `dispatch_optimizer`, and the
/// orchestration crate cannot depend on this pyo3-touching one. The oracles below
/// box their [`EvaluationError`] into it (`slot.record(Box::new(err), id)`); the
/// FFI edge downcasts it back to re-raise the original exception. Re-exported here
/// so the existing `crate::OracleErrorSlot` path keeps resolving.
pub use polypus_orchestration::OracleErrorSlot;

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
            // The bound Qiskit object rides through the pyo3-free `BoundCircuit`
            // enum in its `Foreign` escape hatch (a `QiskitCircuit`), so the Aer/
            // CUNQA backends still receive the native Qiskit object unchanged.
            CircuitSource::Qiskit(circuit) => Ok(QiskitCircuit::into_bound(
                assign_parameters_qiskit(circuit, params)?,
            )),
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
/// A failure of Qiskit's `assign_parameters` call itself is
/// [`EvaluationError::Qiskit`], which the FFI edge raises as
/// `polypus.EvaluationError` when Qiskit's own exception class raised it (issue
/// #218). A failure converting the arguments is [`EvaluationError::Python`],
/// carried verbatim.
///
/// Runs while the run is detached (the planner binds each candidate), so it
/// attaches with [`attach_or`]: an unreachable interpreter is
/// [`InterpreterUnavailable`], never a panic or a hang (docs/ENGINEERING.md §9).
pub(crate) fn assign_parameters_qiskit(
    circuit: &Py<PyAny>,
    params: &[f64],
) -> Result<Py<PyAny>, EvaluationError> {
    attach_or(
        || Err(InterpreterUnavailable.into()),
        |py| {
            let qc = circuit
                .clone_ref(py)
                .into_pyobject(py)
                .map_err(|e| EvaluationError::Python(PyErr::from(e).into()))?;
            let kwargs = [("inplace", false)]
                .into_py_dict(py)
                .map_err(|e| EvaluationError::Python(e.into()))?;
            Ok(qc
                .call_method("assign_parameters", (params.to_vec(),), Some(&kwargs))
                .map_err(|e| EvaluationError::Qiskit(e.into()))?
                .unbind())
        },
    )
}

/// Contract between optimization algorithms and quantum circuit evaluation.
///
/// Re-exported from the pure-Rust [`polypus_optimizers`] crate, where the trait
/// now lives (it is the optimizers' input contract). Re-exporting here keeps the
/// `crate::EvaluationOracle` path — used by [`VqcOracle`] and
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

/// Test support shared by this crate's modules.
#[cfg(test)]
pub(crate) mod test_support {
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
    use std::sync::{Arc, Mutex};

    use pyo3::prelude::*;
    use pyo3::{PyTraverseError, PyVisit};

    type Probe = Box<dyn FnOnce() + Send>;

    /// A Python object whose `__traverse__` runs a probe once.
    #[pyclass]
    struct TraverseProbe {
        probe: Mutex<Option<Probe>>,
    }

    #[pymethods]
    impl TraverseProbe {
        fn __traverse__(&self, _visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
            let probe = self.probe.lock().unwrap_or_else(|p| p.into_inner()).take();
            if let Some(probe) = probe {
                probe();
            }
            Ok(())
        }
    }

    /// Run `f` from inside a GC traversal, where PyO3 forbids attaching to the
    /// interpreter, and return what it returned.
    ///
    /// This is how a test reaches the "interpreter unavailable" branch of code
    /// that needs a live Python object (a user callback, a Qiskit circuit): the
    /// fresh-process route cannot build one without initializing the
    /// interpreter, which cannot be undone. Inside `__traverse__`,
    /// `Python::try_attach` returns `None` deterministically, while
    /// `Python::attach` panics. That panic is caught here and re-raised on the
    /// test thread: one escaping `__traverse__` would abort the process.
    pub(crate) fn run_while_attaching_is_forbidden<R: Send + 'static>(
        f: impl FnOnce() -> R + Send + 'static,
    ) -> R {
        let outcome = Arc::new(Mutex::new(None));
        let slot = Arc::clone(&outcome);
        let probe: Probe = Box::new(move || {
            let forbidden = Python::try_attach(|_py| ()).is_none();
            let result = catch_unwind(AssertUnwindSafe(f));
            *slot.lock().unwrap_or_else(|p| p.into_inner()) = Some((forbidden, result));
        });
        Python::initialize();
        Python::attach(|py| {
            let object = Bound::new(
                py,
                TraverseProbe {
                    probe: Mutex::new(Some(probe)),
                },
            )
            .expect("the probe object is created");
            py.import("gc")
                .and_then(|gc| gc.call_method0("collect"))
                .expect("gc.collect() runs");
            drop(object);
        });
        let (forbidden, result) = outcome
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
            .expect("gc.collect() traversed the probe object");
        assert!(forbidden, "the probe must run where attaching is forbidden");
        result.unwrap_or_else(|panic| resume_unwind(panic))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in circuit whose `assign_parameters` raises `RuntimeError`, run in a
    /// bare interpreter (no Qiskit needed, ENGINEERING §3).
    fn failing_circuit(py: Python<'_>) -> Py<PyAny> {
        pyo3::types::PyModule::from_code(
            py,
            c"class Circuit:\n    def assign_parameters(self, *args, **kwargs):\n        raise RuntimeError('binding refused')\n",
            c"stand_in.py",
            c"stand_in",
        )
        .and_then(|module| module.getattr("Circuit")?.call0())
        .expect("the stand-in circuit instantiates")
        .unbind()
    }

    #[test]
    fn a_failing_binding_call_is_the_qiskit_variant() {
        // Issue #218: Qiskit's `assign_parameters` failing is told apart from a
        // user callback (`Python`), so the FFI edge can wrap a Qiskit exception
        // without retyping a callback's. The exception itself is kept as raised.
        pyo3::Python::initialize();
        let circuit = Python::attach(failing_circuit);
        match assign_parameters_qiskit(&circuit, &[0.5]) {
            Err(EvaluationError::Qiskit(err)) => Python::attach(|py| {
                assert!(err
                    .as_py_err()
                    .is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
                assert!(err.to_string().contains("binding refused"));
            }),
            other => panic!("expected EvaluationError::Qiskit, got {other:?}"),
        }
    }

    /// Binding a Qiskit circuit is reached while the run is detached (the
    /// planner binds each candidate per wave). When the interpreter cannot be
    /// attached to, it fails with `InterpreterUnavailable` boxed in
    /// `BackendError::External`, without touching the circuit: its
    /// `assign_parameters` records every call, and none is made.
    #[test]
    fn binding_fails_typed_without_touching_the_circuit_when_attaching_is_forbidden() {
        use polypus_infrastructure::{BackendError, InterpreterUnavailable};
        pyo3::Python::initialize();
        let module = Python::attach(|py| {
            pyo3::types::PyModule::from_code(
                py,
                c"calls = []\nclass Circuit:\n    def assign_parameters(self, *args, **kwargs):\n        calls.append(args)\n        return self\n",
                c"binding_recorder.py",
                c"binding_recorder",
            )
            .expect("the recorder compiles")
            .unbind()
        });
        let circuit = Python::attach(|py| {
            module
                .bind(py)
                .getattr("Circuit")
                .and_then(|class| class.call0())
                .expect("the stand-in circuit instantiates")
                .unbind()
        });
        let result = test_support::run_while_attaching_is_forbidden(move || {
            assign_parameters_qiskit(&circuit, &[0.5]).map(drop)
        });
        match result {
            Err(EvaluationError::Backend(BackendError::External(boxed))) => {
                assert!(boxed.is::<InterpreterUnavailable>(), "{boxed}")
            }
            other => panic!("expected Backend(External(InterpreterUnavailable)), got {other:?}"),
        }
        Python::attach(|py| {
            let calls = module.bind(py).getattr("calls").unwrap();
            assert_eq!(calls.len().unwrap(), 0, "the circuit must not be called");
        });
    }

    /// Building an Aer backend with a noise model runs inside `detach` for
    /// `run_quantum_circuit`, and the noise model is the one Python object it
    /// needs. When the interpreter cannot be attached to, it fails with
    /// `InterpreterUnavailable` boxed in `BackendError::External`. Lives here,
    /// not in `polypus-infrastructure`, because it needs a live object and so
    /// the `__traverse__` probe of this crate's `test_support`.
    #[test]
    fn create_backend_with_a_noise_model_fails_typed_when_attaching_is_forbidden() {
        use polypus_infrastructure::{
            BackendConfig, BackendError, ExecutionConfig, Infrastructure, OptLevel,
        };
        pyo3::Python::initialize();
        let noise_model = Python::attach(|py| pyo3::types::PyDict::new(py).into_any().unbind());
        let config = ExecutionConfig {
            id: "no-interpreter".to_string(),
            shots: 8,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: BackendConfig::Local {
                backend: "AerSimulator".to_string(),
                sim_method: "statevector".to_string(),
                noise_model: Some(noise_model),
            },
            opt_level: OptLevel::default(),
            seed: Some(1),
        };
        let result = test_support::run_while_attaching_is_forbidden(move || {
            Infrastructure::create_backend(&config).map(drop)
        });
        match result {
            Err(BackendError::External(boxed)) => {
                assert!(boxed.is::<InterpreterUnavailable>(), "{boxed}")
            }
            other => panic!("expected External(InterpreterUnavailable), got {other:?}"),
        }
    }

    #[test]
    fn qiskit_variant_display_names_the_binding() {
        pyo3::Python::initialize();
        let err = EvaluationError::Qiskit(pyo3::exceptions::PyRuntimeError::new_err("nope").into());
        let message = err.to_string();
        assert!(
            message.contains("Qiskit parameter binding failed") && message.contains("nope"),
            "{message}"
        );
    }
}
