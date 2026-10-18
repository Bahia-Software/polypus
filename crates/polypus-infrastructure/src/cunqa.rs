use crate::attach::{attach_for_cleanup, attach_or, InterpreterUnavailable};
use crate::error::BackendError;
use crate::transpiler::{IdentityTranspiler, TranspileOptions, Transpiler};
use crate::{record_cleanup_failure, BoundCircuit, QuantumBackend, RunParams};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// The QPU-release operation, boxed so it can capture the CUNQA family handle
/// obtained at construction, and injectable so the panic-safety of `Drop` can be
/// tested (see [`CunqaBackend::with_releaser`]).
///
/// It receives the `Python` token rather than attaching itself: attaching
/// happens in exactly one place, [`CunqaBackend::close`], through
/// [`attach_for_cleanup`], and no error a release operation raises can leave
/// `close` as a `PyErr`. An `attach` written inside a release operation reuses
/// the attachment it was handed, without re-checking the interpreter's state,
/// as long as it runs on the same thread and outside a `detach`. A release
/// operation that detaches (`py.detach(|| Python::attach(..))`) makes that inner
/// attach a fresh one again, which can panic or hang at shutdown; don't.
type ReleaseFn = Box<dyn Fn(Python<'_>) -> PyResult<()> + Send + Sync>;

/// SLURM walltime of the virtual QPUs when the caller does not set one.
pub const DEFAULT_WALLTIME: &str = "10:00:00";

/// A walltime string that is not `HH:MM:SS` (see [`validate_walltime`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidWalltime {
    /// The rejected value.
    pub value: String,
    reason: &'static str,
}

impl std::fmt::Display for InvalidWalltime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid walltime {:?}: {}; expected \"HH:MM:SS\" (hours: one or more digits, \
             minutes and seconds: two digits below 60, total above zero), e.g. \"48:00:00\"",
            self.value, self.reason
        )
    }
}

impl std::error::Error for InvalidWalltime {}

/// Check that `walltime` is a SLURM walltime in the `HH:MM:SS` form.
///
/// Grammar: `HH ":" MM ":" SS`, where `HH` is one or more ASCII digits (it may
/// exceed 24, so `48:00:00` is valid), `MM` and `SS` are exactly two digits
/// below 60, and the total is greater than zero. SLURM's `D-HH:MM:SS` and the
/// shorter `MM:SS` / `MM` forms are rejected.
pub fn validate_walltime(walltime: &str) -> Result<(), InvalidWalltime> {
    parse_walltime(walltime).map(|_| ())
}

/// Parse a walltime accepted by [`validate_walltime`] into its duration.
pub fn parse_walltime(walltime: &str) -> Result<Duration, InvalidWalltime> {
    let err = |reason| InvalidWalltime {
        value: walltime.to_string(),
        reason,
    };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let mut parts = walltime.split(':');
    let (Some(h), Some(m), Some(s), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(err("it must have exactly three ':'-separated fields"));
    };
    if !digits(h) {
        return Err(err("hours must be one or more digits"));
    }
    if m.len() != 2 || s.len() != 2 || !digits(m) || !digits(s) {
        return Err(err("minutes and seconds must be two digits"));
    }
    // Two ASCII digits always parse.
    if m.parse::<u8>().unwrap_or(99) >= 60 || s.parse::<u8>().unwrap_or(99) >= 60 {
        return Err(err("minutes and seconds must be below 60"));
    }
    if [h, m, s].iter().all(|f| f.bytes().all(|b| b == b'0')) {
        return Err(err("the walltime must be greater than zero"));
    }
    // Fields are validated digits; an hour count beyond `u64` saturates.
    let field = |f: &str| f.parse::<u64>().unwrap_or(u64::MAX);
    let seconds = field(h)
        .saturating_mul(3600)
        .saturating_add(field(m) * 60)
        .saturating_add(field(s));
    Ok(Duration::from_secs(seconds))
}

/// CunqaBackend: runs quantum circuits on the CUNQA distributed QPU platform.
pub struct CunqaBackend {
    /// Guards against double-release of the QPU allocation (explicit `close()`
    /// followed by `Drop`, or vice versa).
    closed: AtomicBool,
    /// Simulation method for CUNQA's simulated QPUs.
    sim_method: String,
    /// Number of physical QPUs in the allocation. Bounds how many circuits can
    /// be dispatched per `run_circuits` call (one circuit per QPU).
    n_qpus: u32,
    /// SLURM walltime the QPUs were raised with (see [`QuantumBackend::walltime`]).
    walltime: Option<Duration>,
    /// Native-circuit transpiler applied before submission. CUNQA's simulated
    /// QPUs transpile internally, so this defaults to the no-op
    /// [`IdentityTranspiler`] and the observable behavior is unchanged.
    transpiler: Box<dyn Transpiler>,
    /// Releases the QPU allocation on `close`/`Drop`. Captured at construction
    /// (holding the family handle) so `Drop` needs nothing but this field, and
    /// so panic-safety is testable with an injected operation.
    release: ReleaseFn,
}

impl QuantumBackend for CunqaBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        config: &RunParams,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
        // An empty batch is a valid request with no results, as on the native
        // backend; do not reach the seal with it.
        if qcs.is_empty() {
            return Ok(Vec::new());
        }
        // Called while the run is detached, so this attach is a fresh one: an
        // unreachable interpreter is a typed error, never a panic or a hang
        // (docs/ENGINEERING.md §9).
        attach_or(
            || Err(InterpreterUnavailable.into()),
            |py| {
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
                kwargs.set_item("qcs", qcs_pylist).map_err(conv)?;
                kwargs.set_item("shots", config.shots).map_err(conv)?;
                kwargs
                    .set_item("sim_method", &self.sim_method)
                    .map_err(conv)?;
                // Forwarded to each QPU's `run(..., seed=...)` on the Python side
                // (contract C-7); `None` is simply omitted, so CUNQA's own
                // unseeded default behavior is unchanged. Unlike the identical
                // arrangement for Aer (`local.rs`), this seam is unverified — see
                // `python/polypus_python/cunqa.py` for the caveats.
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
            },
        )
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

    /// The QPUs are released when their SLURM walltime runs out.
    fn walltime(&self) -> Option<Duration> {
        self.walltime
    }

    fn close(&self) {
        // Idempotent: only the first call actually releases the allocation.
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        log::info!("Dropping QPUs");
        // Panic-free by construction: `attach_for_cleanup` never panics (an
        // unreachable interpreter, e.g. at shutdown, is an error), and it turns
        // any Python exception into an owned message while still attached, so
        // logging `e` below cannot attach again. A failure is logged and
        // recorded in the process-wide counter instead of propagated. This is
        // what makes `Drop` safe even mid-unwind, and it is reached identically
        // from the explicit `close()` calls in the orchestration algorithms.
        match attach_for_cleanup(|py| (self.release)(py)) {
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
        walltime: &str,
        sim_method: String,
    ) -> Result<Self, BackendError> {
        // Allocating QPUs reserves an HPC (SLURM) allocation — a rare, coarse
        // lifecycle event an operator wants to see at the default level.
        log::info!(
            "Raising QPUs in CUNQA: n_qpus={n_qpus}, nodes={nodes}, id={id}, cores_per_qpu={cores_per_qpu}, walltime={walltime}"
        );
        let family = raise_qpus(n_qpus, nodes, id, cores_per_qpu, walltime)?;
        // The edge validated the string; one that does not parse here only loses
        // the overrun warning, never the run.
        let walltime = parse_walltime(walltime).ok();
        // Capture the family handle in the release closure; it is the only thing
        // `drop_qpus` needs, and keeping it here (rather than as a struct field)
        // keeps the release operation self-contained and injectable.
        let release: ReleaseFn = Box::new(move |py| drop_qpus(py, &family));
        Ok(CunqaBackend {
            closed: AtomicBool::new(false),
            sim_method,
            n_qpus,
            walltime,
            transpiler: Box::new(IdentityTranspiler),
            release,
        })
    }

    /// Construct a backend with an injected release operation, bypassing the
    /// real CUNQA allocation. Test-only hook that lets the `Drop` panic-safety
    /// tests force a cleanup failure without SLURM or the `polypus_python`
    /// seam.
    #[cfg(test)]
    fn with_releaser(release: ReleaseFn) -> Self {
        CunqaBackend {
            closed: AtomicBool::new(false),
            sim_method: String::new(),
            n_qpus: 1,
            walltime: None,
            transpiler: Box::new(IdentityTranspiler),
            release,
        }
    }
}

/// Raise the SLURM QPU allocation via the `polypus_python` seam, returning the
/// opaque CUNQA family handle.
///
/// Reached while detached (`create_backend` runs inside `detach` for
/// `run_quantum_circuit` and `qml.predict`), so it attaches with [`attach_or`]:
/// an unreachable interpreter fails before anything is asked of SLURM.
fn raise_qpus(
    n_qpus: u32,
    nodes: u32,
    id: &str,
    cores_per_qpu: u32,
    walltime: &str,
) -> Result<Py<PyAny>, BackendError> {
    attach_or(
        || Err(InterpreterUnavailable.into()),
        |py| {
            let kwargs = PyDict::new(py);
            let conv = |e: PyErr| BackendError::Conversion(e.to_string());
            kwargs.set_item("n", n_qpus).map_err(conv)?;
            kwargs.set_item("t", walltime).map_err(conv)?;
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
        },
    )
}

/// Release the QPU allocation identified by `family` through the
/// `polypus_python` seam. Installed as the backend's [`ReleaseFn`], so it runs
/// only from [`CunqaBackend::close`], already attached.
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
    use pyo3::exceptions::PyValueError;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    #[test]
    fn walltime_accepts_hh_mm_ss() {
        for ok in [
            DEFAULT_WALLTIME,
            "48:00:00",
            "1:02:03",
            "100:59:59",
            "00:00:01",
            "000:01:00",
        ] {
            assert_eq!(validate_walltime(ok), Ok(()), "{ok:?} must be accepted");
        }
    }

    #[test]
    fn parse_walltime_returns_the_duration() {
        for (text, secs) in [
            (DEFAULT_WALLTIME, 36_000),
            ("48:00:00", 172_800),
            ("1:02:03", 3_723),
            ("00:00:01", 1),
            ("99999999999999999999:00:00", u64::MAX),
        ] {
            assert_eq!(
                parse_walltime(text),
                Ok(Duration::from_secs(secs)),
                "{text:?}"
            );
        }
        assert!(parse_walltime("00:00:00").is_err());
    }

    #[test]
    fn walltime_rejects_everything_else() {
        for bad in [
            "00:00:00",
            "000:00:00",
            "1:2:3",
            "10:0:00",
            "10:00:5",
            "10:60:00",
            "10:00:60",
            "1-12:00:00",
            "10:00",
            "10",
            "10:00:00:00",
            "",
            " 10:00:00",
            "10:00:00 ",
            "-1:00:00",
            "+1:00:00",
            "1e1:00:00",
            "aa:00:00",
            ":00:00",
            "1\u{0663}:00:00",
        ] {
            let err = validate_walltime(bad).expect_err(bad);
            assert_eq!(err.value, bad);
            assert!(err.to_string().contains("HH:MM:SS"), "{err}");
        }
    }

    /// Failure-injection test (issue acceptance criterion 2): a `CunqaBackend`
    /// whose release raises a Python exception is dropped *while another panic
    /// is already unwinding*. A panic in `Drop` mid-unwind would abort the
    /// process; this test proves it does not, and that the failure is recorded.
    ///
    /// The release operation receives a `Python` token, so this needs a bare
    /// interpreter (no package imported), like the other tests in this crate
    /// that call `Python::initialize()`.
    #[test]
    fn drop_during_unwind_is_panic_free_and_records_failure() {
        Python::initialize();
        let attempts = Arc::new(AtomicUsize::new(0));
        let before = cleanup_failure_count();
        let attempts_in = Arc::clone(&attempts);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _backend = CunqaBackend::with_releaser(Box::new(move |_py| {
                attempts_in.fetch_add(1, Ordering::SeqCst);
                Err(PyValueError::new_err("injected release failure"))
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
        Python::initialize();
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_in = Arc::clone(&calls);
        let backend = CunqaBackend::with_releaser(Box::new(move |_py| {
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

    /// Raising the QPUs and running circuits, both reached while the run is
    /// detached (`create_backend` runs inside `detach` for `run_quantum_circuit`
    /// and `qml.predict`), fail with a typed `InterpreterUnavailable` when the
    /// interpreter cannot be attached to: no panic, nothing asked of SLURM, and
    /// the interpreter is never initialized. Runs in a fresh process that never
    /// initializes Python; see the drop test below for why that state.
    #[test]
    fn seams_fail_typed_without_an_interpreter() {
        use crate::{InterpreterUnavailable, OptLevel};
        if !crate::attach::is_fresh_process() {
            crate::attach::run_in_fresh_process(
                "cunqa::tests::seams_fail_typed_without_an_interpreter",
            );
            return;
        }
        let unavailable = |result: Result<(), BackendError>| match result {
            Err(BackendError::External(boxed)) => boxed.is::<InterpreterUnavailable>(),
            other => panic!("expected External(InterpreterUnavailable), got {other:?}"),
        };
        let raised = CunqaBackend::new(2, 1, "no-interpreter", 1, DEFAULT_WALLTIME, String::new());
        assert!(unavailable(raised.map(|_backend| ())));

        let backend = CunqaBackend::with_releaser(Box::new(|_py| Ok(())));
        let circuit = BoundCircuit::Native(
            polypus_circuit::ParameterizedCircuit::new(1)
                .x(0)
                .measure_all()
                .assign_parameters(&[])
                .unwrap(),
        );
        let params = RunParams {
            id: "no-interpreter".to_string(),
            shots: 8,
            seed: None,
            opt_level: OptLevel::default(),
        };
        assert!(unavailable(
            backend.run_circuits(&[circuit], &params).map(|_counts| ())
        ));
        assert!(
            Python::try_attach(|_py| ()).is_none(),
            "a seam must never initialize the interpreter"
        );
    }

    /// Set if the release operation of the test below ever runs.
    static RELEASE_RAN: AtomicBool = AtomicBool::new(false);

    /// A backend dropped when the interpreter cannot be attached to — even
    /// while another panic unwinds — records a cleanup failure instead of
    /// panicking, and never runs its release operation. This covers the one
    /// place the release path attaches (`close`, through `attach_for_cleanup`):
    /// a release operation, `new`'s included, only receives a `Python` token and
    /// cannot attach on its own.
    ///
    /// The unavailable state exercised is "not initialized", the only one a
    /// Rust test can produce deterministically. It is also how a shutting-down
    /// interpreter looks to PyO3 on every Python version (CPython clears its
    /// `initialized` flag right after marking itself finalizing), but a real
    /// shutdown cannot be reproduced here. The test runs in a fresh process
    /// because other tests in this binary initialize the interpreter, which
    /// cannot be undone.
    #[test]
    fn drop_without_interpreter_records_a_failure_instead_of_panicking() {
        if !crate::attach::is_fresh_process() {
            crate::attach::run_in_fresh_process(
                "cunqa::tests::drop_without_interpreter_records_a_failure_instead_of_panicking",
            );
            return;
        }
        let before = cleanup_failure_count();
        let result = std::panic::catch_unwind(|| {
            let _backend = CunqaBackend::with_releaser(Box::new(|_py| {
                RELEASE_RAN.store(true, Ordering::SeqCst);
                Ok(())
            }));
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
            !RELEASE_RAN.load(Ordering::SeqCst),
            "the release operation must not run without an interpreter"
        );
    }
}
