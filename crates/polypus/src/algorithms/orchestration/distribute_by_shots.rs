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

        // Distribute shots across QPUs, conserving the total (contract C-3): the
        // remainder `shots % n_qpus` is spread one extra shot per QPU over the
        // first `remainder` QPUs, never dropped. `n_qpus >= 1` and `shots >= 1`
        // are guaranteed by the Python-facing boundary validation, so no guard is
        // duplicated here.
        //
        // `ExecutionConfig::shots` applies uniformly to a whole `run_circuits`
        // batch, so we submit at most two uniform-shots batches: `remainder`
        // circuits at `base + 1` shots, and the remaining `n_qpus - remainder`
        // circuits at `base` shots. When `shots < n_qpus` the base is `0`; that
        // group is skipped so no zero-shot circuits are ever submitted.
        let shots = args.config.shots;
        let n_qpus = args.config.n_qpus;
        let base = shots / n_qpus;
        let remainder = shots % n_qpus;
        log::debug!(
            "distributing {} shots across {} QPUs: {} QPU(s) at {} shots, {} QPU(s) at {} shots",
            shots,
            n_qpus,
            remainder,
            base + 1,
            n_qpus - remainder,
            base
        );

        // Run — native counts, one dict per QPU. Replicating the circuit is cheap
        // (refcount bump or string/struct clone).
        let mut counts_vec: Vec<HashMap<String, u64>> = Vec::new();
        if remainder > 0 {
            let qcs: Vec<_> = (0..remainder).map(|_| args.qcs[0].duplicate()).collect();
            args.config.shots = base + 1;
            counts_vec.extend(backend.run_circuits(&qcs, &args.config));
        }
        if base > 0 {
            let qcs: Vec<_> = (0..n_qpus - remainder)
                .map(|_| args.qcs[0].duplicate())
                .collect();
            args.config.shots = base;
            counts_vec.extend(backend.run_circuits(&qcs, &args.config));
        }

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
