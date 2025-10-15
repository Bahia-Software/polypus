use crate::infrastructure::{QuantumRunner};
use crate::algorithms::AlgorithmArgs;

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
pub struct LocalRunner;

impl QuantumRunner for LocalRunner {

	/// Run the quantum circuit locally using the provided arguments.
	fn run<'py>(&self, args: &AlgorithmArgs<'py>) -> pyo3::PyObject {
		let id = args.id.clone();
		let qc = args.qc.clone();
		let backend = String::from("AerSimulator");
		let shots = args.shots.clone();

		Python::with_gil(|py| {
			let module = PyModule::import(py, "polypus_python").unwrap();
			let connection = module.call_method("connect_to_infrastructure", ("local", ), None);
			let connection_str = match connection {
				Ok(conn) => conn.extract::<String>().unwrap(),
				Err(e) => {
					panic!("{e}, Error connecting to infrastructure");
				},
			};
			
			let kwargs = PyDict::new(py);
        	kwargs.set_item("id", id);
        	kwargs.set_item("backend", backend);
			kwargs.set_item("qc", qc);
			kwargs.set_item("shots", shots);


			let running_result = module.call_method("run_qc", (connection_str,), Some(&kwargs));
			match running_result {
				Ok(result) => result.unbind(),
				Err(e) => {
					// error!("Error running run_qc: {e}");
					panic!("{e}, Error running run_qc");
				},
			}
		})
	}
}