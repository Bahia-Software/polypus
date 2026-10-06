//! Error type for the optimizer-oracle / expectation-evaluation path.
//!
//! See [`polypus_infrastructure::error`] for the crate-wide granularity
//! decision. This enum wraps a [`BackendError`] (the underlying execution
//! failure), a [`CircuitError`] (native parameter binding) or a Python exception
//! (a Python callback/conversion, or a Qiskit binding call), all reachable while
//! an optimizer drives an oracle across the FFI. A Python exception is carried as
//! a [`DisplaySafePyErr`], never a bare `PyErr`: the pyo3-free `OracleErrorSlot`
//! logs this error with `{e}` while the thread is detached, where formatting a
//! bare `PyErr` could panic at interpreter shutdown.

use std::fmt;

use polypus_circuit::CircuitError;
use polypus_infrastructure::{BackendError, DisplaySafePyErr, InfrastructureError};
use polypus_observable::ObservableError;

/// A failure encountered while evaluating a candidate parameter vector.
///
/// The optimizer traits ([`EvaluationOracle`](polypus_optimizers::EvaluationOracle),
/// [`VarianceOracle`](polypus_optimizers::VarianceOracle)) return plain
/// `f64`/`Vec<f64>` and cannot carry a `Result` across the FFI, so an oracle
/// records its first failure of this type in an
/// [`OracleErrorSlot`](crate::OracleErrorSlot) and the entry point
/// surfaces it after `optimize` returns.
///
/// `Clone`/`Eq` are omitted: the [`EvaluationError::Python`] variant carries a
/// Python exception. `Display` and the derived `Debug` never panic, even without
/// an interpreter: the `Python`/`Qiskit` payloads are [`DisplaySafePyErr`]s.
#[derive(Debug)]
pub enum EvaluationError {
    /// The underlying execution backend failed.
    Backend(BackendError),
    /// Native parameter binding failed (wrong count, non-finite value, …).
    Binding(CircuitError),
    /// Native cost-observable evaluation failed (bad bitstring width/char, or a
    /// callback observable's error carried in [`ObservableError::External`]).
    Observable(ObservableError),
    /// A Python callback or conversion on the evaluation path raised. Carried
    /// verbatim so the original exception type is preserved across the FFI.
    Python(DisplaySafePyErr),
    /// Qiskit raised while binding a Qiskit circuit's parameters (the only
    /// Qiskit call on the evaluation path). Kept apart from [`Python`](Self::Python),
    /// which also carries the user's callback exceptions, so that the FFI edge
    /// can raise a Qiskit exception as `polypus.EvaluationError` (contract C-1,
    /// issue #218) while a callback's exception still re-raises verbatim. The
    /// edge applies the same rule as on the execution seam: only an exception
    /// whose class comes from Qiskit is wrapped; a `ValueError`/`TypeError`
    /// raised there keeps its class.
    Qiskit(DisplaySafePyErr),
    /// A Rust-originated infrastructure failure on the QML evaluation path
    /// (Tokio runtime construction, or a worker task panic surfaced as a
    /// `JoinError`). Never a Python exception, so unlike `Python` it must not be
    /// re-raised verbatim.
    Runtime(String),
    /// Converting data across the Rust↔Python boundary on the evaluation path
    /// failed (e.g. `expectation_values`'s return value isn't `list[float]`).
    /// Unlike `Python`, this never originated in a raised Python exception, so
    /// it must not be re-raised verbatim.
    Conversion(String),
    /// The Python-backed oracle returned a different number of expectation
    /// values than circuits were submitted in this call (contract C-5).
    WrongLength { expected: usize, got: usize },
    /// The Python-backed oracle returned a non-finite expectation value
    /// (contract C-5 requires every output to be a finite f64).
    NonFinite { index: usize, value: f64 },
    /// A supervised objective scored a training sample as `NaN`/infinite. Reported
    /// per sample so the message names the `x_train` row (typically a `log(0)`).
    NonFiniteScore {
        /// The 0-based `x_train` row.
        sample: usize,
        value: f64,
    },
    /// A supervised [`QmlOracle`](crate::QmlOracle) has a different number of labels
    /// than training circuits. The `qml.train` edge rejects this up front (contract
    /// C-8), so only a direct Rust caller can reach it.
    LabelCount { labels: usize, samples: usize },
    /// The Python `variance_function` (QNG) returned an invalid QFIM diagonal
    /// element: a `NaN`/infinite value or a negative one. A variance must be a
    /// finite, non-negative number — zero is allowed (Tikhonov regularisation
    /// keeps the QNG division well-posed). Rejected at the callback boundary so a
    /// bad value cannot silently corrupt the natural-gradient update.
    InvalidVariance {
        /// The parameter index the callback was evaluated at.
        param_index: usize,
        /// The offending value returned by `variance_function`.
        value: f64,
    },
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationError::Backend(err) => write!(f, "{err}"),
            EvaluationError::Binding(err) => write!(f, "circuit binding failed: {err}"),
            EvaluationError::Observable(err) => write!(f, "expectation evaluation failed: {err}"),
            EvaluationError::Python(err) => write!(f, "Python evaluation error: {err}"),
            EvaluationError::Qiskit(err) => write!(f, "Qiskit parameter binding failed: {err}"),
            EvaluationError::Runtime(m) => write!(f, "QML evaluation runtime error: {m}"),
            EvaluationError::Conversion(m) => {
                write!(f, "data conversion across the Python boundary failed: {m}")
            }
            EvaluationError::WrongLength { expected, got } => write!(
                f,
                "oracle returned the wrong number of expectation values: expected {expected} (one per submitted circuit) but got {got} (contract C-5)"
            ),
            EvaluationError::NonFinite { index, value } => write!(
                f,
                "oracle returned a non-finite expectation value {value} at index {index}; contract C-5 requires every output to be a finite f64"
            ),
            EvaluationError::NonFiniteScore { sample, value } => write!(
                f,
                "the supervised objective scored x_train row {sample} as {value}; every per-sample score must be finite (clip probabilities before taking a log)"
            ),
            EvaluationError::LabelCount { labels, samples } => write!(
                f,
                "{labels} labels were given for {samples} training samples; supervised QML needs exactly one label per training sample (contract C-8)"
            ),
            EvaluationError::InvalidVariance { param_index, value } => write!(
                f,
                "variance_function returned an invalid value {value} for parameter index {param_index}; a QFIM diagonal element must be a finite, non-negative number"
            ),
        }
    }
}

impl std::error::Error for EvaluationError {}

impl From<BackendError> for EvaluationError {
    fn from(err: BackendError) -> Self {
        EvaluationError::Backend(err)
    }
}

impl From<ObservableError> for EvaluationError {
    fn from(err: ObservableError) -> Self {
        EvaluationError::Observable(err)
    }
}

impl From<InfrastructureError> for EvaluationError {
    fn from(err: InfrastructureError) -> Self {
        match err {
            // A backend failure (including a between-wave interrupt, which now
            // arrives as `Backend(BackendError::External(boxed DisplaySafePyErr))`) and an
            // observable failure map straight through, exactly as the former
            // `run_and_evaluate` returned them.
            InfrastructureError::Backend(e) => EvaluationError::Backend(e),
            InfrastructureError::Observable(e) => EvaluationError::Observable(e),
            // A cooperative cancel surfaces as a KeyboardInterrupt, the same class
            // a SIGINT would (unreachable while nothing sets the token).
            InfrastructureError::Cancelled => EvaluationError::Python(
                pyo3::exceptions::PyKeyboardInterrupt::new_err("the run was cancelled").into(),
            ),
            // A planner/backend mismatch is a construction-time check, not reached
            // through the oracle; surface it as the typed evaluation error.
            InfrastructureError::IncompatiblePlanner(m) => EvaluationError::Runtime(m),
        }
    }
}

// `EvaluationError` deliberately implements no `From<_> for PyErr`: mapping it to
// the typed `polypus.*` exception hierarchy is the `polypus` FFI edge's job
// (`polypus::exceptions::evaluation_error_to_pyerr`), which owns those
// `#[pyclass]` types. The `Python`/`Observable(External)` variants still carry the
// original exception (as a `DisplaySafePyErr`) so the edge can re-raise it
// unchanged; `Qiskit` carries one too, for the edge to wrap when Qiskit raised it.

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};

    // The FFI mapping of these variants to the typed `polypus.*` exception
    // classes now lives at the `polypus` edge (`exceptions::evaluation_error_to_pyerr`)
    // and is tested there; this crate owns only the pure-Rust `Display`, which is
    // what those Python messages are built from — so it is pinned here.

    #[test]
    fn wrong_length_display_names_both_lengths() {
        let msg = EvaluationError::WrongLength {
            expected: 4,
            got: 2,
        }
        .to_string();
        assert!(msg.contains('4'), "expected length missing from: {msg}");
        assert!(msg.contains('2'), "got length missing from: {msg}");
    }

    #[test]
    fn non_finite_display_names_index_and_value() {
        let msg = EvaluationError::NonFinite {
            index: 3,
            value: f64::NAN,
        }
        .to_string();
        assert!(msg.contains('3'), "offending index missing from: {msg}");
        assert!(msg.contains("NaN"), "offending value missing from: {msg}");
    }

    #[test]
    fn non_finite_score_display_names_the_x_train_row_and_value() {
        let msg = EvaluationError::NonFiniteScore {
            sample: 7,
            value: f64::NEG_INFINITY,
        }
        .to_string();
        assert!(
            msg.contains("x_train row 7"),
            "offending row missing from: {msg}"
        );
        assert!(msg.contains("-inf"), "offending value missing from: {msg}");
    }

    #[test]
    fn label_count_display_names_both_counts() {
        let msg = EvaluationError::LabelCount {
            labels: 9,
            samples: 10,
        }
        .to_string();
        assert!(
            msg.contains("9 labels") && msg.contains("10 training samples"),
            "both counts must be named: {msg}"
        );
    }

    /// With a live interpreter the Python-carrying variants show the original
    /// exception's class and message, as before the `DisplaySafePyErr` carrier.
    #[test]
    fn python_variants_display_the_original_exception() {
        pyo3::Python::initialize();
        let python = EvaluationError::Python(PyValueError::new_err("bad row").into());
        assert_eq!(
            python.to_string(),
            "Python evaluation error: ValueError: bad row"
        );
        let qiskit = EvaluationError::Qiskit(PyRuntimeError::new_err("nope").into());
        assert_eq!(
            qiskit.to_string(),
            "Qiskit parameter binding failed: RuntimeError: nope"
        );
    }

    /// Without an interpreter every variant that can hold a Python exception
    /// formats (`Display` and `Debug`) to the fixed fallback instead of panicking,
    /// including through the `Box<dyn Error + Send>` that `OracleErrorSlot::record`
    /// logs while the thread is detached. With a bare `PyErr` inside, each of these
    /// panics: its `Display`/`Debug` call `Python::attach`. Runs in a fresh process
    /// that never initializes Python (a lazily built `PyErr` needs none).
    #[test]
    fn python_carrying_errors_format_without_an_interpreter() {
        if !is_fresh_process() {
            run_in_fresh_process(
                "error::tests::python_carrying_errors_format_without_an_interpreter",
            );
            return;
        }
        let fallback = DisplaySafePyErr::UNAVAILABLE;
        let py_err = || DisplaySafePyErr::from(PyValueError::new_err("bad row"));
        let errors = [
            EvaluationError::Python(py_err()),
            EvaluationError::Qiskit(py_err()),
            EvaluationError::Backend(BackendError::External(Box::new(py_err()))),
            EvaluationError::Observable(ObservableError::External(Box::new(py_err()))),
            EvaluationError::from(InfrastructureError::Cancelled),
        ];
        for err in errors {
            assert!(err.to_string().contains(fallback), "{err}");
            assert!(format!("{err:?}").contains(fallback), "{err:?}");
            let slot_payload: Box<dyn std::error::Error + Send> = Box::new(err);
            assert!(format!("{slot_payload}").contains(fallback));
        }
        let observable = ObservableError::External(Box::new(py_err()));
        assert_eq!(observable.to_string(), fallback);
        assert!(format!("{observable:?}").contains(fallback));
    }

    /// Run the test `name` alone in a fresh copy of this test binary, in which
    /// Python is never initialized, and fail if it fails there. A local copy of
    /// `polypus-infrastructure`'s test-only helper, which is not public.
    fn run_in_fresh_process(name: &str) {
        let exe = std::env::current_exe().expect("the test binary path is available");
        let output = std::process::Command::new(exe)
            .args(["--exact", name, "--test-threads=1", "--nocapture"])
            .env(FRESH_PROCESS_ENV, "1")
            .output()
            .expect("the test binary can be re-executed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "child run failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            stdout.contains("1 passed"),
            "the child must actually run the test, not filter it out:\n{stdout}"
        );
    }

    /// Set in the child process started by [`run_in_fresh_process`].
    const FRESH_PROCESS_ENV: &str = "POLYPUS_FRESH_PROCESS_CHILD";

    /// Whether this process is a child started by [`run_in_fresh_process`].
    fn is_fresh_process() -> bool {
        std::env::var_os(FRESH_PROCESS_ENV).is_some()
    }
}
