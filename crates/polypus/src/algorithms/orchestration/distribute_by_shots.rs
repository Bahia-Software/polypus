use crate::algorithms::{AlgorithmArgs, AlgorithmTrait};
use crate::infrastructure::{
    BackendError, CancelToken, CircuitTask, Infrastructure, Planner, ShotDistributingPlanner,
};

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Distributes shots across multiple QPUs and merges the result counts.
pub struct DistributeByShotsRun;

impl AlgorithmTrait for DistributeByShotsRun {
    type Args = AlgorithmArgs;
    type AlgorithmReturnType = PyResult<pyo3::PyObject>;

    fn run(&self, args: AlgorithmArgs) -> PyResult<pyo3::PyObject> {
        // This algorithm operates on exactly one circuit. Reject any other count
        // up front, before a backend is even built (the planner guards it again as
        // defense in depth). `n_qpus >= 1` and `shots >= 1` come from the boundary.
        if args.qcs.len() != 1 {
            return Err(BackendError::InvalidCircuitCount {
                expected: 1,
                got: args.qcs.len(),
            }
            .into());
        }

        let backend = Infrastructure::create_backend(&args.config)?;
        // The ShotDistributingPlanner owns the whole distribution: it apportions
        // this circuit's shots across `n_qpus` (base + one extra on the first
        // `remainder`, conserving the total per C-3), runs `run_shots_distributed`,
        // merges the replicas, validates and honours Ctrl+C.
        let planner = ShotDistributingPlanner;
        let cancel = CancelToken::default();
        let tasks = [CircuitTask {
            circuit: &args.qcs[0],
            shots: args.config.shots,
        }];
        let mut merged = planner
            .execute(backend.as_ref(), &tasks, &args.config, &cancel)
            .map_err(super::infrastructure_error_to_pyerr)?;
        backend.close();

        // `execute` returns exactly one merged `Counts` for the single circuit.
        let total = merged.pop().unwrap_or_default();
        Python::with_gil(|py| -> PyResult<pyo3::PyObject> {
            let py_dict = PyDict::new(py);
            for (k, v) in total {
                py_dict.set_item(k, v)?;
            }
            Ok(py_dict.into())
        })
    }

    fn name(&self) -> String {
        String::from("Distribute By Shots Run Algorithm")
    }

    fn description(&self) -> String {
        String::from("Algorithm to run quantum circuits distributed by shots")
    }
}
