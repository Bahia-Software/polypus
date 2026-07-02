use crate::algorithms::{AlgorithmArgs, AlgorithmTrait};
use crate::infrastructure::Infrastructure;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Distributes shots across multiple QPUs and merges the result counts.
pub struct DistributeByShotsRun;

impl AlgorithmTrait for DistributeByShotsRun {
    type Args = AlgorithmArgs;
    type AlgorithmReturnType = pyo3::PyObject;

    fn run(&self, mut args: AlgorithmArgs) -> pyo3::PyObject {
        let backend = Infrastructure::create_backend(&args.config);

        // Divide shots across QPUs
        args.config.shots /= args.config.n_qpus;

        // Replicate the circuit once per QPU (cheap: refcount bump or string clone)
        let qcs: Vec<_> = (0..args.config.n_qpus)
            .map(|_| args.qcs[0].duplicate())
            .collect();

        // Run — native counts, one dict per QPU
        let counts_vec = backend.run_circuits(&qcs, &args.config);

        // Merge counts from all QPUs into a single dict
        let mut total: HashMap<String, u64> = HashMap::new();
        for counts in counts_vec {
            for (k, v) in counts {
                *total.entry(k).or_insert(0) += v;
            }
        }
        let merged_pyobj = Python::with_gil(|py| {
            let py_dict = PyDict::new(py);
            for (k, v) in total {
                py_dict.set_item(k, v).unwrap();
            }
            py_dict.into()
        });

        backend.close();
        merged_pyobj
    }

    fn name(&self) -> String {
        String::from("Distribute By Shots Run Algorithm")
    }

    fn description(&self) -> String {
        String::from("Algorithm to run quantum circuits distributed by shots")
    }
}
