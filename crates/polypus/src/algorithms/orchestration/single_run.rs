use crate::algorithms::{AlgorithmArgs, AlgorithmTrait};
use crate::infrastructure::Infrastructure;
use pyo3::prelude::*;

/// Runs a single quantum circuit using the specified infrastructure.
pub struct AlgorithmSingleRun;

impl AlgorithmTrait for AlgorithmSingleRun {
    type Args = AlgorithmArgs;
    type AlgorithmReturnType = PyResult<pyo3::PyObject>;

    fn run(&self, args: AlgorithmArgs) -> PyResult<pyo3::PyObject> {
        // Backend creation and execution surface any failure as a `PyErr`; the
        // backend's `Drop` still releases resources if we return early.
        let backend = Infrastructure::create_backend(&args.config)?;
        let counts = backend.run_circuits(&args.qcs, &args.config)?;
        backend.close();
        // Convert native counts to a Python `list[dict]` at the FFI boundary.
        Python::with_gil(|py| {
            // The run above executes with the GIL released (see
            // `run_quantum_circuit`); this reacquire is the first Python
            // touchpoint, so honor a pending Ctrl+C here before building any
            // Python object and propagate it verbatim. See docs/ENGINEERING.md §3.
            py.check_signals()?;
            Ok(counts.into_pyobject(py)?.into_any().unbind())
        })
    }

    fn name(&self) -> String {
        String::from("Single Run Algorithm")
    }

    fn description(&self) -> String {
        String::from("Runs a single quantum circuit using the specified infrastructure.")
    }
}
