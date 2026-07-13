//! Custom Python exception hierarchy for the Polypus bindings.
//!
//! The orchestration/FFI crate turns its internal `Result` error enums
//! ([`BackendError`](crate::infrastructure::BackendError),
//! [`EvaluationError`](crate::evaluation::EvaluationError)) into typed Python
//! exceptions so that a failure reaching Python is a catchable, documented
//! class instead of a `pyo3_runtime.PanicException` (or a process abort).
//!
//! The hierarchy is rooted at `PolypusError` so user code can catch every
//! Polypus-originated failure with a single `except polypus.PolypusError`.
//! Domain subclasses mirror the layers that raise them:
//!
//! ```text
//! Exception
//! └── PolypusError
//!     ├── BackendError            # execution/orchestration layer
//!     │   ├── CunqaError          # CUNQA distributed-QPU backend
//!     │   ├── QmioError           # QMIO real-QPU network path
//!     │   └── NativeCircuitError  # pure-Rust circuit / simulator path
//!     └── EvaluationError         # optimizer oracle / expectation evaluation
//! ```
//!
//! Contract C-1 (see `docs/CONTRACTS.md`) keeps its documented failure modes:
//! an *unknown infrastructure* still surfaces as `ValueError` and a *bad kwarg*
//! at the `polypus_python` seam still surfaces as `TypeError`, because
//! [`BackendError::Seam`](crate::infrastructure::BackendError::Seam) re-raises
//! the original Python exception verbatim. The classes below are raised for the
//! Rust-originated runtime failures that previously *panicked*.

use pyo3::create_exception;
use pyo3::exceptions::PyException;

create_exception!(
    polypus,
    PolypusError,
    PyException,
    "Base class for every Polypus runtime error."
);
create_exception!(
    polypus,
    BackendError,
    PolypusError,
    "A quantum-execution backend failed at runtime."
);
create_exception!(
    polypus,
    CunqaError,
    BackendError,
    "A failure originating in the CUNQA distributed-QPU backend."
);
create_exception!(
    polypus,
    QmioError,
    BackendError,
    "A failure on the QMIO real-QPU network/serialisation path."
);
create_exception!(
    polypus,
    NativeCircuitError,
    BackendError,
    "The native (pure-Rust) circuit or statevector-simulator path failed."
);
create_exception!(
    polypus,
    EvaluationError,
    PolypusError,
    "An oracle/expectation-evaluation failure during training."
);

/// Register the exception hierarchy on the extension module so Python can both
/// see (`polypus.BackendError`) and catch these classes.
pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    let py = m.py();
    m.add("PolypusError", py.get_type::<PolypusError>())?;
    m.add("BackendError", py.get_type::<BackendError>())?;
    m.add("CunqaError", py.get_type::<CunqaError>())?;
    m.add("QmioError", py.get_type::<QmioError>())?;
    m.add("NativeCircuitError", py.get_type::<NativeCircuitError>())?;
    m.add("EvaluationError", py.get_type::<EvaluationError>())?;
    Ok(())
}
