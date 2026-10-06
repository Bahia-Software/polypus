use crate::error::BackendError;
use crate::transpiler::{IdentityTranspiler, TranspileOptions, Transpiler};
use crate::{
    max_statevector_concurrency, BackendCapabilities, BoundCircuit, BudgetSource, CircuitTask,
    InfrastructureError, MemBudget, QuantumBackend, RunParams,
};
use polypus_backend::mem_budget::{active_budget, check_fits};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// LocalBackend: runs quantum circuits on the local machine using Qiskit AerSimulator.
pub struct LocalBackend {
    /// Backend/device class name forwarded to Python (e.g. `"AerSimulator"`).
    backend: String,
    /// Aer simulation method.
    sim_method: String,
    /// Optional Qiskit `NoiseModel`.
    noise_model: Option<Py<PyAny>>,
    /// Native-circuit transpiler applied before submission. Aer transpiles
    /// internally, so this defaults to the no-op [`IdentityTranspiler`] and the
    /// observable behavior is unchanged.
    transpiler: Box<dyn Transpiler>,
}

impl LocalBackend {
    pub fn new(backend: String, sim_method: String, noise_model: Option<Py<PyAny>>) -> Self {
        LocalBackend {
            backend,
            sim_method,
            noise_model,
            transpiler: Box::new(IdentityTranspiler),
        }
    }
}

impl QuantumBackend for LocalBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        config: &RunParams,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
        Python::attach(|py| {
            // Refuse up front a batch whose widest statevector cannot fit in the
            // memory budget (issue #215), before anything is converted or handed
            // to Aer — see `statevector_memory_is_known`. The other methods are
            // bounded by Aer itself through `max_memory_mb` below.
            let widest = widest_qubits(qcs, py);
            let budget = active_budget();
            if statevector_memory_is_known(&self.sim_method) {
                check_fits(widest, budget).map_err(|e| {
                    log::error!("local (Aer) backend refused a circuit: {e}");
                    BackendError::from(e)
                })?;
            }

            // Native circuits are transpiled in pure Rust before submission;
            // Qiskit circuits pass through untouched (Aer transpiles them) and
            // every native circuit travels to Python as OpenQASM 2.0.
            let opts = TranspileOptions {
                level: config.opt_level,
            };
            let qcs_pylist = PyList::empty(py);
            for qc in qcs {
                let qc = qc.transpiled(self.transpiler.as_ref(), &opts);
                qcs_pylist
                    .append(crate::to_py_object(&qc, py)?)
                    .map_err(|e| BackendError::Conversion(e.to_string()))?;
            }

            let module = PyModule::import(py, "polypus_python").map_err(crate::seam_error)?;
            let connection = module
                .call_method("connect_to_infrastructure", ("local",), None)
                .map_err(|e| {
                    // Surface the failure before it crosses the FFI as a Python
                    // exception (mirrors the QMIO/CUNQA error paths).
                    log::error!("local infrastructure connection failed: {e}");
                    crate::seam_error(e)
                })?;
            // The call above succeeded; a wrong-shaped return value is our
            // Rust-side conversion failure, not a seam exception (contract C-1).
            let connection_str = connection.extract::<String>().map_err(|e| {
                BackendError::Conversion(format!(
                    "expected connect_to_infrastructure(\"local\") to return str: {e}"
                ))
            })?;

            let kwargs = PyDict::new(py);
            let conv = |e: PyErr| BackendError::Conversion(e.to_string());
            kwargs.set_item("id", &config.id).map_err(conv)?;
            kwargs.set_item("backend", &self.backend).map_err(conv)?;
            kwargs.set_item("qcs", qcs_pylist).map_err(conv)?;
            kwargs.set_item("shots", config.shots).map_err(conv)?;
            kwargs
                .set_item("sim_method", &self.sim_method)
                .map_err(conv)?;
            // Bound Aer's concurrent experiments to the statevector memory budget
            // (plan §4.5, P1-memory): Aer's `max_parallel_experiments=0` default
            // ("auto") spawns one process per experiment and OOMs at high qubit
            // counts. This is a pure resource knob — Aer seeds each experiment
            // deterministically, so the counts are unchanged for any bound.
            let cores = std::thread::available_parallelism()
                .map(|c| c.get())
                .unwrap_or(1);
            let max_parallel_experiments = max_statevector_concurrency(widest, cores);
            kwargs
                .set_item("max_parallel_experiments", max_parallel_experiments)
                .map_err(conv)?;
            // Hand a *known* budget to Aer's own per-experiment memory validation
            // (issue #215), for every method: it covers what the dense model above
            // cannot — `automatic` picking a statevector for a non-Clifford circuit,
            // `density_matrix`'s `4^n` — while a Clifford circuit that Aer runs on
            // `stabilizer` still fits. Aer's refusal comes back as an unsuccessful
            // experiment, which the Python side raises as
            // `polypus.InsufficientMemoryError`. Omitted for the fallback budget,
            // so Aer keeps its own default (the host's RAM).
            if let Some(max_memory_mb) = aer_max_memory_mb(budget) {
                kwargs
                    .set_item("max_memory_mb", max_memory_mb)
                    .map_err(conv)?;
            }
            if let Some(nm) = &self.noise_model {
                kwargs
                    .set_item("noise_model", nm.clone_ref(py))
                    .map_err(conv)?;
            }
            // Forwarded to Aer's `seed_simulator` on the Python side (contract
            // C-7); `None` is simply omitted rather than sent as `seed=None`,
            // so Aer's own unseeded default behavior is unchanged.
            if let Some(seed) = config.seed {
                kwargs.set_item("seed", seed).map_err(conv)?;
            }

            let result = module
                .call_method("run_qcs", (connection_str,), Some(&kwargs))
                .map_err(|e| {
                    log::error!("local circuit execution failed: {e}");
                    crate::seam_error(e)
                })?;
            // As above: `run_qcs` returned successfully, so a wrong-shaped
            // value is a Rust-side conversion failure, not a seam exception.
            result.extract::<Vec<HashMap<String, u64>>>().map_err(|e| {
                BackendError::Conversion(format!(
                    "expected run_qcs() to return list[dict[str, int]] (contract C-1): {e}"
                ))
            })
        })
    }

    /// Report the wave size the planner should use for this batch, derived from
    /// the same statevector memory budget [`run_circuits`](Self::run_circuits)
    /// hands Aer as `max_parallel_experiments` — the batch's widest circuit fed to
    /// [`max_statevector_concurrency`] against `available_parallelism()`, read off
    /// the task slice without cloning any circuit. `Native`/`Qasm2` widths are read
    /// GIL-free; a `Qiskit` circuit's `num_qubits` is read via `getattr` under the
    /// GIL, exactly as `run_circuits` does — so a high-qubit Qiskit population
    /// (a Qiskit-templated ansatz on `backend="aer"`, the common QML case) is still
    /// wave-split for interruptibility, not treated as one uninterruptible batch.
    ///
    /// **Signal safety (issue #147 follow-up).** That `getattr` runs Python
    /// bytecode, which CPython may abort with a `KeyboardInterrupt` for a pending
    /// Ctrl+C. This runs at the top of every [`Planner::execute`](crate::Planner::execute)
    /// — for training, once per generation on the optimizer thread — so a swallowed
    /// interrupt (the earlier `.ok()` mapping *any* failure to "width unknown")
    /// cleared the pending signal before the planner's between-wave
    /// `py.check_signals()` could see it, leaving `qml.train` unresponsive to
    /// Ctrl+C. Here the interrupt is instead **propagated verbatim** as
    /// [`BackendError::External`]; only a genuine non-interrupt failure (e.g. a
    /// missing attribute, which a real `QuantumCircuit` never has) falls back to
    /// "width unknown". The Qiskit path's memory bound is separately enforced in
    /// `run_circuits`.
    ///
    /// The wave size is capped **only when the whole batch cannot be held in the
    /// memory budget at once** (the high-qubit regime): there splitting into waves
    /// lets the planner run a `py.check_signals()` between them (issue #147). When
    /// the batch fits, the cap is reported as unbounded so the whole population
    /// reaches Aer as a **single** call (Aer parallelises the experiments
    /// internally; see `tests/python/test_qml_concurrency.py`). See
    /// `wave_concurrency` for why the fit test uses the
    /// pure memory limit rather than a core-count gate (which would degenerate on a
    /// single-core host, issue #147's 1-thread case).
    fn capabilities_for(
        &self,
        tasks: &[CircuitTask<'_>],
    ) -> Result<BackendCapabilities, InfrastructureError> {
        let cores = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1);
        let widest = Python::attach(|py| -> Result<usize, InfrastructureError> {
            let mut widest = 0usize;
            for task in tasks {
                if let Some(n) = circuit_qubits_checked(task.circuit, py)? {
                    widest = widest.max(n);
                }
            }
            Ok(widest)
        })?;
        Ok(BackendCapabilities {
            max_concurrency: crate::wave_concurrency(widest, cores, tasks.len()),
            supports_shot_distribution: true,
        })
    }
}

/// Whether Aer's memory for `sim_method` is the dense `16 · 2^n`-byte
/// statevector the budget models, so that a circuit which does not fit can be
/// refused up front, in Rust (issue #215).
///
/// Only `"statevector"` qualifies. `"automatic"` (the default) lets Aer pick the
/// method per circuit — `stabilizer` for a Clifford circuit, whose memory is
/// polynomial in `n` — so refusing on `2^n` there would reject large circuits
/// that run fine today; `matrix_product_state`, `stabilizer` and
/// `extended_stabilizer` are not dense, and `density_matrix` needs `16 · 4^n`,
/// which this `2^n` model would underestimate. Those methods are not left
/// unguarded: the same budget reaches Aer as `max_memory_mb`
/// ([`aer_max_memory_mb`]), whose own per-experiment validation knows the method
/// it chose, and its refusal surfaces as the same
/// `polypus.InsufficientMemoryError`. The budget also throttles their concurrency.
fn statevector_memory_is_known(sim_method: &str) -> bool {
    sim_method == "statevector"
}

/// The `max_memory_mb` to hand Aer for `budget` (issue #215): the budget in whole
/// MiB, at least 1, or `None` for a [`BudgetSource::Fallback`] budget, which is a
/// guess and must not become a hard limit. Never `Some(0)`: Aer reads 0 as
/// "no limit".
///
/// Aer applies the limit **per experiment**, not across the experiments it runs
/// in parallel; that matches the budget, whose concurrency throttle
/// (`max_parallel_experiments`) already keeps `concurrency · size ≤ budget`.
fn aer_max_memory_mb(budget: MemBudget) -> Option<u64> {
    const MIB: u64 = 1024 * 1024;
    match budget.source {
        BudgetSource::Fallback => None,
        BudgetSource::Explicit | BudgetSource::Detected => Some((budget.bytes / MIB).max(1)),
    }
}

/// Qubit width of a single circuit for wave sizing, propagating a
/// `KeyboardInterrupt` instead of swallowing it (issue #147 follow-up). The
/// `Native` and `Qasm2` arms come from the shared, GIL-free
/// [`BoundCircuit::native_qubit_width`]; a `Qiskit` circuit's `num_qubits` is read
/// via `getattr`, and if that `getattr` fails **because CPython raised a
/// `KeyboardInterrupt`** for a pending Ctrl+C, the error is returned verbatim so
/// the planner re-raises it — rather than being mistaken for a missing attribute
/// and cleared. Any other failure (a genuine missing/incompatible attribute,
/// which a real `QuantumCircuit` never has) means the width is simply unknown
/// (`None`). Contrast [`circuit_qubits`], the swallowing variant `run_circuits`
/// uses immediately before its GIL-releasing Aer call.
fn circuit_qubits_checked(
    qc: &BoundCircuit,
    py: Python<'_>,
) -> Result<Option<usize>, InfrastructureError> {
    match crate::as_qiskit(qc) {
        Some(obj) => match obj.bind(py).getattr("num_qubits") {
            Ok(attr) => Ok(attr.extract::<usize>().ok()),
            // Carry the KeyboardInterrupt verbatim across the pyo3-free contract in
            // `BackendError::External`; the FFI edge downcasts it back and re-raises
            // it as the `KeyboardInterrupt` the planner's interrupt guard expects.
            Err(e) if e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) => {
                Err(InfrastructureError::Backend(crate::seam_error(e)))
            }
            Err(_) => Ok(None),
        },
        // Native/Qasm2 (and any non-Qiskit foreign) use the shared GIL-free width
        // rule; only the Qiskit read can raise, so only it needs the
        // KeyboardInterrupt-propagating handling.
        None => Ok(qc.native_qubit_width()),
    }
}

/// The widest circuit in the batch, used to size Aer's `max_parallel_experiments`
/// against the statevector memory budget (plan §4.5). Each variant's width is
/// read where it is cheapest: `Native` exposes it directly, `Qasm2` is parsed for
/// it, and a `Qiskit` circuit is read through the GIL (`num_qubits`), which the
/// caller already holds. An empty or unreadable batch yields 0 — a one-amplitude
/// budget that leaves Aer's parallelism at the core count.
fn widest_qubits(qcs: &[BoundCircuit], py: Python<'_>) -> usize {
    qcs.iter()
        .filter_map(|qc| circuit_qubits(qc, py))
        .max()
        .unwrap_or(0)
}

/// Qubit width of a single circuit, read where it is cheapest: the `Native` and
/// `Qasm2` arms come from the shared, GIL-free
/// [`BoundCircuit::native_qubit_width`], while a `Qiskit` circuit is read through
/// the GIL (`num_qubits`), which the caller already holds. This is the swallowing
/// variant: **any** failure reading the Qiskit width (including a
/// `KeyboardInterrupt`) becomes `None`, which is deliberate at its sole call site
/// (via [`widest_qubits`] inside [`LocalBackend::run_circuits`], immediately before
/// a GIL-releasing Aer call). Contrast [`circuit_qubits_checked`], the hardened
/// variant [`LocalBackend::capabilities_for`] uses, which propagates a
/// `KeyboardInterrupt` verbatim.
fn circuit_qubits(qc: &BoundCircuit, py: Python<'_>) -> Option<usize> {
    match crate::as_qiskit(qc) {
        Some(obj) => obj.bind(py).getattr("num_qubits").ok()?.extract().ok(),
        // Native/Qasm2 (and any non-Qiskit foreign) use the shared GIL-free rule.
        None => qc.native_qubit_width(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polypus_circuit::ParameterizedCircuit;

    /// `widest_qubits` budgets for the largest statevector in a mixed batch: a
    /// too-small `n` would under-bound Aer and reintroduce the OOM. The `Native`
    /// and `Qasm2` arms are GIL-free; the GIL is only held to satisfy the
    /// signature (the `Qiskit` arm is exercised end-to-end by the Python suite).
    #[test]
    fn widest_qubits_picks_the_largest_circuit() {
        pyo3::Python::initialize();
        let small = ParameterizedCircuit::new(2)
            .h(0)
            .measure_all()
            .assign_parameters(&[])
            .unwrap();
        let big = ParameterizedCircuit::new(7)
            .h(0)
            .measure_all()
            .assign_parameters(&[])
            .unwrap();
        let batch = vec![
            BoundCircuit::Native(small),
            BoundCircuit::Qasm2(big.to_qasm2()),
        ];
        Python::attach(|py| {
            assert_eq!(widest_qubits(&batch, py), 7);
            assert_eq!(widest_qubits(&[], py), 0);
        });
    }

    /// Issue #215: the known budget reaches Aer in whole MiB with a floor of 1,
    /// because Aer reads `max_memory_mb=0` as "no limit"; the `Fallback` budget is
    /// a guess and is never imposed (`None`: the kwarg is not sent).
    #[test]
    fn aer_max_memory_mb_is_the_known_budget_in_mib_and_never_zero() {
        const MIB: u64 = 1024 * 1024;
        for source in [BudgetSource::Explicit, BudgetSource::Detected] {
            let mb = |bytes| aer_max_memory_mb(MemBudget { bytes, source });
            // Aer reads 0 as "no limit": a sub-MiB budget must floor to 1, not 0.
            assert_eq!(mb(0), Some(1));
            assert_eq!(mb(1), Some(1));
            assert_eq!(mb(MIB - 1), Some(1));
            assert_eq!(mb(MIB), Some(1));
            assert_eq!(mb(100 * MIB), Some(100));
            assert_eq!(mb(100 * MIB + MIB - 1), Some(100)); // rounded down
            assert_eq!(mb(32 * 1024 * MIB), Some(32768));
        }
        // The blind fallback is never imposed on Aer.
        assert_eq!(
            aer_max_memory_mb(MemBudget {
                bytes: 16 * 1024 * MIB,
                source: BudgetSource::Fallback,
            }),
            None
        );
    }

    /// Issue #215: only the dense `statevector` method is refused up front, in
    /// Rust, on the `16 · 2^n` model. The methods whose memory is not that (Aer's
    /// `automatic` default may pick `stabilizer`; `density_matrix` is `16 · 4^n`)
    /// are not refused here: Aer refuses them itself against the same budget,
    /// passed as `max_memory_mb` (see `aer_max_memory_mb`).
    #[test]
    fn only_the_statevector_method_is_refused_on_the_dense_model() {
        assert!(statevector_memory_is_known("statevector"));
        for method in [
            "automatic",
            "stabilizer",
            "extended_stabilizer",
            "matrix_product_state",
            "density_matrix",
            "unitary",
            "superop",
            "",
        ] {
            assert!(!statevector_memory_is_known(method), "{method}");
        }
    }

    /// Issue #147: a high-qubit batch that cannot fit in the memory budget is
    /// reported with the memory cap `run_circuits` hands Aer as
    /// `max_parallel_experiments`, so the planner can split it into memory-safe
    /// waves. A 40-qubit statevector is 16 TiB — beyond any budget this test can
    /// meet, detected or explicit (issue #215 made the default host-dependent) — so
    /// only one fits at a time, and a 4-circuit batch cannot be held at once ⇒
    /// cap 1 — **regardless of the host core count** (the single-core regression
    /// from issue #147, so the test deliberately does not guard on the core
    /// count). The circuit is zero-gate and never submitted to Aer, so this costs
    /// nothing. The batch-agnostic `capabilities()` is left at its unbounded
    /// default (purely additive).
    #[test]
    fn capabilities_for_exposes_the_memory_cap_leaving_capabilities_unchanged() {
        pyo3::Python::initialize();
        let wide = BoundCircuit::Native(
            ParameterizedCircuit::new(40)
                .assign_parameters(&[])
                .unwrap(),
        );
        let tasks: Vec<CircuitTask> = (0..4)
            .map(|_| CircuitTask {
                circuit: &wide,
                shots: 8,
            })
            .collect();
        let backend =
            LocalBackend::new("AerSimulator".to_string(), "statevector".to_string(), None);

        let cap = backend.capabilities_for(&tasks).unwrap().max_concurrency;
        assert_eq!(
            cap, 1,
            "a 40-qubit statevector (16 TiB) admits one at a time under any budget"
        );
        assert!(
            cap < tasks.len(),
            "the cap must force more than one wave for this batch"
        );

        // The frozen, batch-agnostic seam is untouched.
        assert_eq!(backend.capabilities().max_concurrency, usize::MAX);
        assert!(backend.capabilities().supports_shot_distribution);
    }

    /// A low-qubit batch is reported as a **single unbounded wave**, so the whole
    /// population reaches Aer in one call (issue #147 must not regress the
    /// single-batch contract of `tests/python/test_qml_concurrency.py`).
    #[test]
    fn capabilities_for_keeps_a_single_wave_at_low_qubits() {
        pyo3::Python::initialize();
        let small =
            BoundCircuit::Native(ParameterizedCircuit::new(2).assign_parameters(&[]).unwrap());
        let tasks: Vec<CircuitTask> = (0..64)
            .map(|_| CircuitTask {
                circuit: &small,
                shots: 8,
            })
            .collect();
        let backend =
            LocalBackend::new("AerSimulator".to_string(), "statevector".to_string(), None);
        assert_eq!(
            backend.capabilities_for(&tasks).unwrap().max_concurrency,
            usize::MAX,
            "a low-qubit population must stay a single wave"
        );
    }

    /// Issue #147 follow-up: a high-qubit **Qiskit** batch (a Qiskit-templated
    /// ansatz on `backend="aer"`, the common QML case) must still be wave-split —
    /// `capabilities_for` reads the Qiskit `num_qubits` through the GIL, just like
    /// `run_circuits`. Every task is a `Qiskit` variant whose object exposes
    /// `num_qubits == 40` (16 TiB per statevector), so the 64-circuit batch cannot
    /// fit any budget and is capped to 1 (not left unbounded). This guards against the regression from
    /// `588a138`, which stopped reading Qiskit widths and made this batch one
    /// uninterruptible wave.
    #[test]
    fn capabilities_for_wave_splits_a_high_qubit_qiskit_batch() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            // A lightweight stand-in for a wide Qiskit circuit: any object exposing
            // `num_qubits`. `types.SimpleNamespace(num_qubits=40)` avoids a qiskit
            // dependency in this unit test while exercising the getattr path.
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("num_qubits", 40usize).unwrap();
            let wide_obj: Py<PyAny> = py
                .import("types")
                .unwrap()
                .getattr("SimpleNamespace")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap()
                .unbind();

            let circuits: Vec<BoundCircuit> = (0..64)
                .map(|_| crate::QiskitCircuit::into_bound(wide_obj.clone_ref(py)))
                .collect();
            let tasks: Vec<CircuitTask> = circuits
                .iter()
                .map(|c| CircuitTask {
                    circuit: c,
                    shots: 8,
                })
                .collect();
            let backend =
                LocalBackend::new("AerSimulator".to_string(), "statevector".to_string(), None);
            assert_eq!(
                backend.capabilities_for(&tasks).unwrap().max_concurrency,
                1,
                "a 40-qubit Qiskit population must be wave-split (its widths ARE read)"
            );
        });
    }

    /// Issue #147 follow-up: a `KeyboardInterrupt` CPython raises while reading a
    /// Qiskit `num_qubits` (a pending Ctrl+C landing during the `getattr`) must be
    /// **propagated** as [`BackendError::External`], not swallowed as "width
    /// unknown" — otherwise the pending signal is cleared and the planner's
    /// between-wave `check_signals` never fires. Simulated with a Python object
    /// whose `num_qubits` property raises `KeyboardInterrupt`.
    #[test]
    fn capabilities_for_propagates_keyboard_interrupt_from_qiskit_width_read() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            // A stand-in whose `num_qubits` getattr raises KeyboardInterrupt, exactly
            // as CPython would for a pending Ctrl+C mid-bytecode.
            let module = PyModule::from_code(
                py,
                std::ffi::CString::new(
                    "class Boom:\n    @property\n    def num_qubits(self):\n        raise KeyboardInterrupt()\n",
                )
                .unwrap()
                .as_c_str(),
                std::ffi::CString::new("boom.py").unwrap().as_c_str(),
                std::ffi::CString::new("boom").unwrap().as_c_str(),
            )
            .unwrap();
            let boom: Py<PyAny> = module.getattr("Boom").unwrap().call0().unwrap().unbind();

            let circuit = crate::QiskitCircuit::into_bound(boom);
            let tasks = vec![CircuitTask {
                circuit: &circuit,
                shots: 8,
            }];
            let backend =
                LocalBackend::new("AerSimulator".to_string(), "statevector".to_string(), None);

            let err = backend.capabilities_for(&tasks).unwrap_err();
            // The KeyboardInterrupt is carried verbatim, type-erased in
            // `BackendError::External`, so the FFI edge can downcast it back and
            // re-raise it with its original class (see `polypus::exceptions`).
            match err {
                InfrastructureError::Backend(BackendError::External(boxed)) => {
                    let py_err = boxed
                        .downcast_ref::<crate::DisplaySafePyErr>()
                        .expect("the boxed error must carry the original PyErr")
                        .as_py_err();
                    assert!(
                        py_err.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py),
                        "expected the KeyboardInterrupt to be carried verbatim"
                    );
                }
                other => {
                    panic!("expected InfrastructureError::Backend(External), got {other:?}")
                }
            }
        });
    }
}
