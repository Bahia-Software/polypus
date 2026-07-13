use crate::infrastructure::error::BackendError;
use crate::infrastructure::transpiler::{IdentityTranspiler, TranspileOptions, Transpiler};
use crate::infrastructure::{
    record_cleanup_failure, BoundCircuit, ExecutionConfig, QuantumBackend,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

/// The QPU-release operation, boxed so it can capture the CUNQA family handle
/// obtained at construction and — crucially — so the panic-safety of `Drop` can
/// be exercised in tests by injecting a failing closure that never touches the
/// Python interpreter (see [`CunqaBackend::with_releaser`]).
type ReleaseFn = Box<dyn Fn() -> Result<(), BackendError> + Send + Sync>;

/// CunqaBackend: runs quantum circuits on the CUNQA distributed QPU platform.
pub struct CunqaBackend {
    /// Guards against double-release of the QPU allocation (explicit `close()`
    /// followed by `Drop`, or vice versa).
    closed: AtomicBool,
    /// Backend/device class name forwarded to CUNQA.
    backend: String,
    /// Simulation method for CUNQA's simulated QPUs.
    sim_method: String,
    /// Number of physical QPUs in the allocation. Bounds how many circuits can
    /// be dispatched per `run_circuits` call (one circuit per QPU).
    n_qpus: u32,
    /// Native-circuit transpiler applied before submission. CUNQA's simulated
    /// QPUs transpile internally, so this defaults to the no-op
    /// [`IdentityTranspiler`] and the observable behavior is unchanged.
    transpiler: Box<dyn Transpiler>,
    /// Releases the QPU allocation on `close`/`Drop`. Captured at construction
    /// (holding the family handle) so `Drop` needs nothing but this field, and
    /// so panic-safety is testable without a Python interpreter.
    release: ReleaseFn,
}

impl QuantumBackend for CunqaBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        config: &ExecutionConfig,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
        Python::with_gil(|py| {
            // Native circuits are transpiled in pure Rust before submission;
            // Qiskit circuits pass through untouched and every native circuit
            // travels to Python as OpenQASM 2.0.
            let opts = TranspileOptions {
                level: config.opt_level,
            };
            let qcs_pylist = PyList::empty(py);
            for qc in qcs {
                let qc = qc.transpiled(self.transpiler.as_ref(), &opts);
                qcs_pylist
                    .append(qc.to_py_object(py)?)
                    .map_err(|e| BackendError::Conversion(e.to_string()))?;
            }

            let module = PyModule::import(py, "polypus_python").map_err(BackendError::Seam)?;
            let kwargs = PyDict::new(py);
            let conv = |e: PyErr| BackendError::Conversion(e.to_string());
            kwargs.set_item("family_id", &config.id).map_err(conv)?;
            kwargs.set_item("backend", &self.backend).map_err(conv)?;
            kwargs.set_item("qcs", qcs_pylist).map_err(conv)?;
            kwargs.set_item("shots", config.shots).map_err(conv)?;
            kwargs
                .set_item("sim_method", &self.sim_method)
                .map_err(conv)?;
            // Forwarded to each QPU's `run(..., seed=...)` on the Python side
            // (contract C-7); `None` is simply omitted, so CUNQA's own
            // unseeded default behavior is unchanged. Unlike the identical
            // arrangement for Aer (`local.rs`), this seam is unverified — see
            // `polypus_python/cunqa.py` for the caveats.
            if let Some(seed) = config.seed {
                kwargs.set_item("seed", seed).map_err(conv)?;
            }

            let result = module
                .call_method("run_qcs", ("cunqa",), Some(&kwargs))
                .map_err(|e| {
                    // Surface the failure at error level (mirrors the QMIO/local
                    // error paths) before it crosses the FFI as an exception.
                    log::error!("CUNQA circuit execution failed: {e}");
                    BackendError::Seam(e)
                })?;
            result
                .extract::<Vec<HashMap<String, u64>>>()
                .map_err(BackendError::Seam)
        })
    }

    fn max_batch_size(&self, _total: usize) -> usize {
        // CUNQA dispatches one circuit per QPU, so a single call can carry at
        // most `n_qpus` circuits.
        self.n_qpus as usize
    }

    fn close(&self) {
        // Idempotent: only the first call actually releases the allocation.
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        log::info!("Dropping QPUs");
        // Panic-free by construction: `release` returns a `Result`, so a failure
        // is logged and recorded in the process-wide counter instead of
        // propagated. This is what makes `Drop` safe even mid-unwind, and it is
        // reached identically from the explicit `close()` calls in the
        // orchestration algorithms.
        match (self.release)() {
            Ok(()) => log::info!("QPUs dropped successfully"),
            Err(e) => {
                log::error!("CUNQA QPU release failed: {e}");
                record_cleanup_failure();
            }
        }
    }
}

/// RAII guarantee: QPUs are released even if the algorithm panics or returns
/// early. This is essential for the HPC use case, where leaking a SLURM
/// allocation would keep nodes reserved for the full requested walltime.
///
/// `close` can never panic (it only logs and counts a failed release), so this
/// `Drop` is safe to run while another panic is already unwinding — the double
/// panic that would abort the process cannot occur.
impl Drop for CunqaBackend {
    fn drop(&mut self) {
        self.close();
    }
}

impl CunqaBackend {
    pub fn new(
        n_qpus: u32,
        nodes: u32,
        id: &str,
        cores_per_qpu: u32,
        backend: String,
        sim_method: String,
    ) -> Result<Self, BackendError> {
        // Allocating QPUs reserves an HPC (SLURM) allocation — a rare, coarse
        // lifecycle event an operator wants to see at the default level.
        log::info!(
            "Raising QPUs in CUNQA: n_qpus={n_qpus}, nodes={nodes}, id={id}, cores_per_qpu={cores_per_qpu}"
        );
        let family = raise_qpus(n_qpus, nodes, id, cores_per_qpu)?;
        // Capture the family handle in the release closure; it is the only thing
        // `drop_qpus` needs, and keeping it here (rather than as a struct field)
        // keeps the release operation self-contained and injectable.
        let release: ReleaseFn = Box::new(move || drop_qpus(&family));
        Ok(CunqaBackend {
            closed: AtomicBool::new(false),
            backend,
            sim_method,
            n_qpus,
            transpiler: Box::new(IdentityTranspiler),
            release,
        })
    }

    /// Construct a backend with an injected release operation, bypassing the
    /// real CUNQA allocation. Test-only hook that lets the `Drop` panic-safety
    /// test force a cleanup failure without a Python interpreter or SLURM.
    #[cfg(test)]
    fn with_releaser(release: ReleaseFn) -> Self {
        CunqaBackend {
            closed: AtomicBool::new(false),
            backend: String::new(),
            sim_method: String::new(),
            n_qpus: 1,
            transpiler: Box::new(IdentityTranspiler),
            release,
        }
    }
}

/// Raise the SLURM QPU allocation via the `polypus_python` seam, returning the
/// opaque CUNQA family handle.
fn raise_qpus(
    n_qpus: u32,
    nodes: u32,
    id: &str,
    cores_per_qpu: u32,
) -> Result<Py<PyAny>, BackendError> {
    Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        let conv = |e: PyErr| BackendError::Conversion(e.to_string());
        kwargs.set_item("n", n_qpus).map_err(conv)?;
        kwargs.set_item("t", "10:00:00").map_err(conv)?;
        kwargs.set_item("n_nodes", nodes).map_err(conv)?;
        kwargs.set_item("family_name", id).map_err(conv)?;
        kwargs
            .set_item("cores_per_qpu", cores_per_qpu)
            .map_err(conv)?;

        let module = PyModule::import(py, "polypus_python").map_err(BackendError::Seam)?;
        let connection = module
            .call_method("connect_to_infrastructure", ("cunqa",), Some(&kwargs))
            .map_err(|e| {
                log::error!("CUNQA QPU allocation failed: {e}");
                BackendError::Seam(e)
            })?;
        let family: Py<PyAny> = connection.extract().map_err(|e| {
            BackendError::Cunqa(format!("could not extract the CUNQA family handle: {e}"))
        })?;
        log::info!("QPUs raised successfully");
        Ok(family)
    })
}

/// Release the QPU allocation identified by `family` through the
/// `polypus_python` seam.
///
/// Panic-free: every failure is returned as a [`BackendError`] so the caller
/// ([`CunqaBackend::close`], reached from `Drop`) can log-and-continue.
/// Acquiring the GIL here is safe from within `Drop`: the extension module is
/// only dropped while the interpreter is alive, and PyO3 0.24's
/// [`Python::with_gil`] is re-entrant, so this cannot deadlock or propagate a
/// panic through an in-progress unwind.
fn drop_qpus(family: &Py<PyAny>) -> Result<(), BackendError> {
    Python::with_gil(|py| {
        let module = PyModule::import(py, "polypus_python").map_err(BackendError::Seam)?;
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("family", family.clone_ref(py))
            .map_err(|e| BackendError::Conversion(e.to_string()))?;
        module
            .call_method("disconnect_from_infrastructure", ("cunqa",), Some(&kwargs))
            .map_err(BackendError::Seam)?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::cleanup_failure_count;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    /// Failure-injection test (issue acceptance criterion 2): a `CunqaBackend`
    /// whose release always fails is dropped *while another panic is already
    /// unwinding*. A panic in `Drop` mid-unwind would abort the process; this
    /// test proves it does not, and that the failure is recorded.
    ///
    /// No Python interpreter is involved — the injected releaser is pure Rust —
    /// so this honours ENGINEERING.md §3 (Python-runtime-free Rust test suite).
    #[test]
    fn drop_during_unwind_is_panic_free_and_records_failure() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let before = cleanup_failure_count();
        let attempts_in = Arc::clone(&attempts);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _backend = CunqaBackend::with_releaser(Box::new(move || {
                attempts_in.fetch_add(1, Ordering::SeqCst);
                Err(BackendError::Cunqa("injected release failure".to_string()))
            }));
            // Drop runs while THIS panic unwinds out of the closure.
            panic!("forced unwind with a live CunqaBackend");
        }));

        assert!(
            result.is_err(),
            "the forced panic must propagate — a double panic would have aborted the process"
        );
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "Drop must attempt the release exactly once during the unwind"
        );
        assert!(
            cleanup_failure_count() > before,
            "the failed cleanup must be recorded in the process-wide counter"
        );
    }

    /// `close()` is idempotent and never releases more than once, whether it is
    /// called explicitly (orchestration path) or via `Drop`.
    #[test]
    fn close_is_idempotent_across_explicit_close_and_drop() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_in = Arc::clone(&calls);
        let backend = CunqaBackend::with_releaser(Box::new(move || {
            calls_in.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }));
        backend.close();
        backend.close();
        drop(backend);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "release must run exactly once across close/close/drop"
        );
    }
}
