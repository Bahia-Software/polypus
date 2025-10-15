use std::process::Command;
use std::{thread, time};

use crate::infrastructure::{QuantumRunner};
use crate::algorithms::AlgorithmArgs;

use pyo3::prelude::*;

pub struct CunqaRunner;

impl QuantumRunner for CunqaRunner {

	/// Run the quantum circuit on QMIO cloud infrastructure using the provided arguments.
	fn run<'py>(&self, args: &AlgorithmArgs<'py>) -> pyo3::PyObject {
		let qc = args.qc.clone();
		let shots = match args.shots.as_ref() {
			Some(val) => val,
			None => {
				// error!("shots is required");
				panic!("shots is required");
			}
		};
		let shots = shots.extract::<u32>().unwrap_or(1);
		// debug!("Raising QPUs for cunqa infrastructure id family: {}", args.id.clone());
		let _raise_qpus = Command::new("qraise")
			.arg("-n")
			.arg(format!("{}", args.n_qpus.unwrap_or(1)))
			.arg("-N")
			.arg(format!("{}", args.nodes))
			.arg("-t")
			.arg("10:00:00")
			.arg("--fam")
			.arg(format!("{}", args.id))
			.arg("--cloud")
			.spawn()
			.expect("Failed to start qraise command");
		thread::sleep(time::Duration::from_secs(30));
		// debug!("qraise command executed");
		// debug!("Running {} shots per qpu", shots_per_qpu);
		Python::with_gil(|py| {
			let module = PyModule::import(py, "polypus_python").unwrap();
			let running_result = module.call_method("run_qc_in_qpu", (args.id.clone(), qc, shots), None);
			// debug!("CALLING PYTHON!");
			match running_result {
				Ok(results) => results.unbind(),
				Err(e) => {
					// error!("Error running run_qc: {e}");
					panic!("{e}, Error running run_qc");
				},
			}
		})
	}
}