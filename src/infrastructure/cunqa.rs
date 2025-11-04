use crate::infrastructure::{QuantumRunner};
use crate::algorithms::AlgorithmArgs;
use pyo3::prelude::*;
use pyo3::types::{PyList,PyDict};

/// CunqaRunner struct to run quantum circuits on CUNQA infrastructure.
pub struct CunqaRunner{
	slurm_job_id: String,
}

impl QuantumRunner for CunqaRunner {
	fn run(&self, args: &AlgorithmArgs) -> pyo3::PyObject {
		let shots = args.shots.clone();
		Python::with_gil(|py| {
            let qcs_pylist = PyList::empty(py);
			for qc in &args.qcs {
				qcs_pylist.append(qc.clone_ref(py)).unwrap();
			}
			let module = PyModule::import(py, "polypus_python").unwrap();
			let kwargs = PyDict::new(py);
        	kwargs.set_item("family_id", args.id.clone());
        	kwargs.set_item("backend", args.backend.clone());
			kwargs.set_item("qcs", qcs_pylist);
			kwargs.set_item("shots", shots);

			let running_result = module.call_method("run_qcs", ("cunqa",), Some(&kwargs));
			match running_result {
				Ok(results) => results.unbind(),
				Err(e) => {
					// error!("Error running run_qc: {e}");
					panic!("{e}, Error running run_qc");
				},
			}
		})
	}

	fn close(&self) {
		self.drop_qpus();
	}
}

impl CunqaRunner {

	pub fn new(n_qpus: u32, nodes: u32, id: &str) -> Self {
		let mut runner = CunqaRunner{
			slurm_job_id: String::new(),
		};
		runner.raise_qpus(n_qpus, nodes, id);
		runner
	}

	pub fn raise_qpus(&mut self, n_qpus: u32, nodes: u32, id: &str) {
		println!("Calling python cunqa qraise");
		Python::with_gil(|py| {
			let kwargs = PyDict::new(py);
        	kwargs.set_item("n", n_qpus);
        	kwargs.set_item("t", "10:00:00");
			kwargs.set_item("n_nodes", nodes);
			kwargs.set_item("family_name", id);

			let module = PyModule::import(py, "polypus_python").unwrap();
			let connection = module.call_method("connect_to_infrastructure", ("cunqa", ), Some(&kwargs));
			match connection {
				Ok(obj) => {
					let (_family, slurm_job_id): (Py<PyAny>, String) = obj.extract().expect("Error extracting return values from cunqa qraise");
					println!("QPUs raised successfully with Slurm Job ID: {}", slurm_job_id);
					self.slurm_job_id = slurm_job_id;
					println!("Cunqa slurm job id: {}", self.slurm_job_id);
				},
				Err(e) => {
					panic!("{e}, Error raising QPUs in cunqa");
				},
			}
		});
	}

	pub fn drop_qpus(&self) {
		println!("Dropping QPUs for Slurm Job ID: {}", self.slurm_job_id);
		Python::with_gil(|py| {
			let module = PyModule::import(py, "polypus_python").unwrap();
			let kwargs = PyDict::new(py);
			kwargs.set_item("slurm_job_id", &self.slurm_job_id);
			let drop_result = module.call_method("disconnect_from_infrastructure", ("cunqa", ), Some(&kwargs));
			match drop_result {
				Ok(_) => {
					println!("QPUs dropped successfully for Slurm Job ID: {}", self.slurm_job_id);
				},
				Err(e) => {
					panic!("{e}, Error dropping QPUs in cunqa");
				},
			}
		});
	}	
}
