use crate::algorithms::{AlgorithmArgs, AlgorithmTrait};
use crate::infrastructure::{CancelToken, CircuitTask, Infrastructure};
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
        // The default planner (SequentialPlanner) owns the waves, the concurrency
        // cap, the C-3 result validation and the between-wave `check_signals`
        // (ENGINEERING §3) that used to live inline here.
        let planner = backend.default_planner();
        let cancel = CancelToken::default();
        let tasks: Vec<CircuitTask> = args
            .qcs
            .iter()
            .map(|circuit| CircuitTask {
                circuit,
                shots: args.config.shots,
            })
            .collect();
        let counts = planner
            .execute(backend.as_ref(), &tasks, &args.config, &cancel)
            .map_err(super::infrastructure_error_to_pyerr)?;
        backend.close();
        // Convert native counts to a Python `list[dict]` at the FFI boundary.
        Python::with_gil(|py| Ok(counts.into_pyobject(py)?.into_any().unbind()))
    }

    fn name(&self) -> String {
        String::from("Single Run Algorithm")
    }

    fn description(&self) -> String {
        String::from("Runs a single quantum circuit using the specified infrastructure.")
    }
}
