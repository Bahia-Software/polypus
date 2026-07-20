use crate::algorithms::{AlgorithmArgs, AlgorithmTrait};
use crate::infrastructure::{BackendError, Infrastructure};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Distributes shots across multiple QPUs and merges the result counts.
pub struct DistributeByShotsRun;

impl AlgorithmTrait for DistributeByShotsRun {
    type Args = AlgorithmArgs;
    type AlgorithmReturnType = PyResult<pyo3::PyObject>;

    fn run(&self, args: AlgorithmArgs) -> PyResult<pyo3::PyObject> {
        // This algorithm operates on exactly one circuit: it replicates that
        // single circuit across `n_qpus` and splits the shots between the
        // replicas. Reject any other count with a typed error instead of
        // panicking on `args.qcs[0]` (empty `qcs`) or silently dropping the
        // extras (`qcs[1..]`). `n_qpus >= 1` and `shots >= 1` are guaranteed by
        // the Python-facing boundary; the circuit count is not, so this is the
        // sole place that contract is enforced — before any backend is built.
        if args.qcs.len() != 1 {
            return Err(BackendError::InvalidCircuitCount {
                expected: 1,
                got: args.qcs.len(),
            }
            .into());
        }

        let backend = Infrastructure::create_backend(&args.config)?;

        // Distribute shots across QPUs, conserving the total (contract C-3): the
        // remainder `shots % n_qpus` is spread one extra shot per QPU over the
        // first `remainder` QPUs, never dropped. `n_qpus >= 1` and `shots >= 1`
        // are guaranteed by the Python-facing boundary validation, so no guard is
        // duplicated here.
        //
        // `shot_batches[i]` is replica `i`'s shot count (`base + 1` for the first
        // `remainder` replicas, `base` for the rest). The backend runs the single
        // circuit once per batch via `run_shots_distributed`; the native backend
        // overrides that to evolve the statevector once and sample each batch,
        // while other backends fall back to replicating + `run_circuits`.
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

        let shot_batches: Vec<u32> = (0..n_qpus)
            .map(|i| if i < remainder { base + 1 } else { base })
            .collect();
        let counts_vec =
            backend.run_shots_distributed(&args.qcs[0], &shot_batches, &args.config)?;

        // Merge counts from all QPUs into a single dict
        let mut total: HashMap<String, u64> = HashMap::new();
        for counts in counts_vec {
            for (k, v) in counts {
                *total.entry(k).or_insert(0) += v;
            }
        }
        let merged_pyobj = Python::with_gil(|py| -> PyResult<pyo3::PyObject> {
            let py_dict = PyDict::new(py);
            for (k, v) in total {
                py_dict.set_item(k, v)?;
            }
            Ok(py_dict.into())
        })?;

        backend.close();
        Ok(merged_pyobj)
    }

    fn name(&self) -> String {
        String::from("Distribute By Shots Run Algorithm")
    }

    fn description(&self) -> String {
        String::from("Algorithm to run quantum circuits distributed by shots")
    }
}
