use crate::algorithms::{AlgorithmTrait, AlgorithmArgs};
use crate::infrastructure::{Infrastructure, QuantumRunner, LocalRunner, CunqaRunner};

use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// AlgorithmDistributeByShotsRun struct to run quantum circuits distributed by shots using the specified infrastructure.
pub struct DistributeByShotsRun;

impl AlgorithmTrait for DistributeByShotsRun {
	type Args = AlgorithmArgs;
	type AlgorithmReturnType = pyo3::PyObject;

    fn run(&self, args: AlgorithmArgs) -> Self::AlgorithmReturnType {
        let infra = Infrastructure::from_str(&args.infrastructure);
        let runner: Box<dyn QuantumRunner> = match infra {
            Infrastructure::Local => Box::new(LocalRunner),
            Infrastructure::Cunqa => Box::new(CunqaRunner::new(args.n_qpus, args.nodes, &args.id)),
        };

        // Update number of shots based on number of QPUs
        let mut args = args;
        args.shots = args.shots / args.n_qpus;

        // Copy the qc for each QPU
        let mut qcs: Vec<pyo3::PyObject> = Vec::new();
        Python::with_gil(|py| {
            let qc_any = match args.qcs[0].clone_ref(py).into_pyobject(py) {
                Ok(obj) => obj,
                Err(e) => {
                    panic!("Error converting QC to PyObject: {e}");
                }
            };
            for _ in 0..args.n_qpus {
                qcs.push(qc_any.clone().into());
            }
        });
        args.qcs = qcs;

        // Run
        let running_result = runner.run(&args);

        let merged_pyobj = {
            // 1) Extract Python -> Rust Vec<HashMap<_,_>>
            let counts_vec: Vec<HashMap<String, u64>> = Python::with_gil(|py| {
                running_result
                    .extract(py)
                    .expect("Expected running_result to be a list[dict[str,int]]")
            });

            // 2) Merge into a single HashMap
            let mut total: HashMap<String, u64> = HashMap::new();
            for counts in counts_vec {
                for (k, v) in counts {
                    *total.entry(k).or_insert(0) += v;
                }
            }

            // 3) Convert merged HashMap -> PyDict(PyObject)
            Python::with_gil(|py| {
                let py_dict = PyDict::new(py);
                for (k, v) in total {
                    py_dict.set_item(k, v).unwrap();
                }
                py_dict.into()
            })
        };


        // Close
		runner.close();

        merged_pyobj
    }

	fn name(&self) -> String {
		String::from("Distribute By Shots Run Algorithm")
	}

	fn description(&self) -> String {
		String::from("Algorithm to run quantum circuits distributed by shots")
	}
}