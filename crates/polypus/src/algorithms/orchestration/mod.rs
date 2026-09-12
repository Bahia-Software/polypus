pub mod single_run;
pub use single_run::AlgorithmSingleRun;
pub mod distribute_by_shots;
pub use distribute_by_shots::DistributeByShotsRun;

use crate::infrastructure::InfrastructureError;
use pyo3::PyErr;

/// Convert a planner's [`InfrastructureError`] into a `PyErr` at the FFI edge.
///
/// `InfrastructureError` deliberately implements no `From<_> for PyErr`; this is
/// the one place the `run_quantum_circuit` orchestration performs that
/// conversion, mapping each variant to the class it always surfaced as: a backend
/// error keeps its own class, a Python exception (a SIGINT from the planner's
/// `check_signals`) re-raises verbatim.
pub(crate) fn infrastructure_error_to_pyerr(err: InfrastructureError) -> PyErr {
    match err {
        InfrastructureError::Backend(e) => crate::exceptions::backend_error_to_pyerr(e),
        InfrastructureError::Observable(e) => {
            crate::exceptions::EvaluationError::new_err(e.to_string())
        }
        InfrastructureError::Python(e) => e,
        InfrastructureError::Cancelled => {
            pyo3::exceptions::PyKeyboardInterrupt::new_err("the run was cancelled")
        }
        InfrastructureError::IncompatiblePlanner(m) => pyo3::exceptions::PyValueError::new_err(m),
    }
}
