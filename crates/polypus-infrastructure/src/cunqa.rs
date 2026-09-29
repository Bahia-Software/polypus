use crate::attach::attach_for_cleanup;
use crate::error::BackendError;
use crate::transpiler::{IdentityTranspiler, TranspileOptions, Transpiler};
use crate::{record_cleanup_failure, BoundCircuit, QuantumBackend, RunParams};
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
        config: &RunParams,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
        Python::attach(|py| {
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
                    .append(crate::to_py_object(&qc, py)?)
                    .map_err(|e| BackendError::Conversion(e.to_string()))?;
            }

            let module = PyModule::import(py, "polypus_python").map_err(crate::seam_error)?;
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
                    crate::seam_error(e)
                })?;
            // `run_qcs` returned successfully; a wrong-shaped value is a
            // Rust-side conversion failure, not a seam exception (contract C-1).
            result.extract::<Vec<HashMap<String, u64>>>().map_err(|e| {
                BackendError::Conversion(format!(
                    "expected run_qcs() to return list[dict[str, int]] (contract C-1): {e}"
                ))
            })
        })
    }

    fn capabilities(&self) -> super::BackendCapabilities {
        // One circuit per QPU per call, so a wave is at most `n_qpus` circuits.
        super::BackendCapabilities {
            max_concurrency: self.n_qpus as usize,
            supports_shot_distribution: true,
        }
    }

    /// CUNQA allocated exactly `n_qpus` QPUs, so a shot-distributing planner paired
    /// with it must split into the same number of replicas (validated in
    /// `Resources::new`).
    fn replica_count(&self) -> Option<u32> {
        Some(self.n_qpus)
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
        let release = release_via(family, drop_qpus);
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
    Python::attach(|py| {
        let kwargs = PyDict::new(py);
        let conv = |e: PyErr| BackendError::Conversion(e.to_string());
        kwargs.set_item("n", n_qpus).map_err(conv)?;
        kwargs.set_item("t", "10:00:00").map_err(conv)?;
        kwargs.set_item("n_nodes", nodes).map_err(conv)?;
        kwargs.set_item("family_name", id).map_err(conv)?;
        kwargs
            .set_item("cores_per_qpu", cores_per_qpu)
            .map_err(conv)?;

        let module = PyModule::import(py, "polypus_python").map_err(crate::seam_error)?;
        let connection = module
            .call_method("connect_to_infrastructure", ("cunqa",), Some(&kwargs))
            .map_err(|e| {
                log::error!("CUNQA QPU allocation failed: {e}");
                crate::seam_error(e)
            })?;
        let family: Py<PyAny> = connection.extract().map_err(|e| {
            BackendError::Cunqa(format!("could not extract the CUNQA family handle: {e}"))
        })?;
        log::info!("QPUs raised successfully");
        Ok(family)
    })
}

/// Build the release operation `new` installs: `op` run attached to the
/// interpreter, with `state` (the CUNQA family handle) as its argument.
///
/// Attaching happens here, through [`attach_for_cleanup`], and nowhere else on
/// the release path: `op` receives the `Python` token instead of attaching on
/// its own. That is what keeps [`CunqaBackend::close`] — and so `Drop` —
/// panic-free at interpreter shutdown, and it is also why no error that comes
/// out of here holds a `PyErr` (whose `Display` would attach again when `close`
/// logs it). The tests build their releaser with this same function.
fn release_via<T: Send + Sync + 'static>(
    state: T,
    op: fn(Python<'_>, &T) -> PyResult<()>,
) -> ReleaseFn {
    Box::new(move || {
        attach_for_cleanup(|py| op(py, &state)).map_err(|e| BackendError::Cunqa(e.to_string()))
    })
}

/// Release the QPU allocation identified by `family` through the
/// `polypus_python` seam. Called only through [`release_via`], which attaches
/// and turns any exception into an owned message.
fn drop_qpus(py: Python<'_>, family: &Py<PyAny>) -> PyResult<()> {
    let module = PyModule::import(py, "polypus_python")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("family", family.clone_ref(py))?;
    module.call_method("disconnect_from_infrastructure", ("cunqa",), Some(&kwargs))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cleanup_failure_count;
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

    /// Set by [`recording_op`] if the release operation body ever runs.
    static OP_RAN: AtomicBool = AtomicBool::new(false);

    fn recording_op(_py: Python<'_>, _state: &()) -> PyResult<()> {
        OP_RAN.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// A backend whose interpreter is unavailable when it is dropped — even
    /// while another panic unwinds — records a cleanup failure instead of
    /// panicking. The releaser is built by [`release_via`], the same function
    /// `CunqaBackend::new` wires `drop_qpus` through, so this covers the real
    /// path: `Drop` → `close` → `release_via` → `attach_for_cleanup` → logged,
    /// counted error. Only the `release_via(family, drop_qpus)` call in `new`
    /// is not executed, as it needs a live CUNQA family handle.
    ///
    /// The unavailable state exercised is "not initialized", the only one a
    /// Rust test can produce deterministically. It already made `with_gil`
    /// panic in PyO3 0.25; the state that motivated `attach_for_cleanup` —
    /// "finalizing", which makes `Python::attach` panic since PyO3 0.26 on
    /// Python ≥ 3.13 — takes the same `try_attach` branch but cannot be
    /// reproduced here. The test runs in a fresh process because other tests
    /// in this binary initialize the interpreter, which cannot be undone.
    #[test]
    fn drop_without_interpreter_records_a_failure_instead_of_panicking() {
        if !crate::attach::is_fresh_process() {
            crate::attach::run_in_fresh_process(
                "cunqa::tests::drop_without_interpreter_records_a_failure_instead_of_panicking",
            );
            return;
        }
        let err = release_via((), recording_op)()
            .expect_err("the release cannot succeed without an interpreter");
        assert!(
            matches!(&err, BackendError::Cunqa(msg) if msg.contains("unavailable")),
            "an unavailable interpreter must surface as a CUNQA release failure, got {err:?}"
        );

        let before = cleanup_failure_count();
        let result = std::panic::catch_unwind(|| {
            let _backend = CunqaBackend::with_releaser(release_via((), recording_op));
            // Drop runs while THIS panic unwinds out of the closure.
            panic!("forced unwind with a live CunqaBackend");
        });
        assert!(
            result.is_err(),
            "the forced panic must propagate — a double panic would have aborted the process"
        );
        assert!(
            cleanup_failure_count() > before,
            "the failed release must be recorded in the process-wide counter"
        );
        assert!(
            !OP_RAN.load(Ordering::SeqCst),
            "the release body must not run without an interpreter"
        );
    }
}
