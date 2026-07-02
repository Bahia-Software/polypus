use crate::algorithms::{AlgorithmArgs, AlgorithmTrait};
use crate::infrastructure::Infrastructure;
use pyo3::prelude::*;

/// Runs a single quantum circuit using the specified infrastructure.
pub struct AlgorithmSingleRun;

impl AlgorithmTrait for AlgorithmSingleRun {
    type Args = AlgorithmArgs;
    type AlgorithmReturnType = pyo3::PyObject;

    fn run(&self, args: AlgorithmArgs) -> pyo3::PyObject {
        let backend = Infrastructure::create_backend(&args.config);
        let counts = backend.run_circuits(&args.qcs, &args.config);
        backend.close();
        // Convert native counts to a Python `list[dict]` at the FFI boundary.
        Python::with_gil(|py| {
            counts
                .into_pyobject(py)
                .expect("Failed to convert counts to a Python object")
                .into_any()
                .unbind()
        })
    }

    fn name(&self) -> String {
        String::from("Single Run Algorithm")
    }

    fn description(&self) -> String {
        String::from("Runs a single quantum circuit using the specified infrastructure.")
    }
}
