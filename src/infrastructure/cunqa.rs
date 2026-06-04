use crate::infrastructure::{QuantumRunner};
use crate::algorithms::AlgorithmArgs;
use pyo3::prelude::*;
use pyo3::types::{PyList,PyDict};

/// CunqaRunner struct to run quantum circuits on CUNQA infrastructure.
pub struct CunqaRunner{
	family: Option<Py<PyAny>>,
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
        	let _ = kwargs.set_item("family_id", args.id.clone());
        	let _ = kwargs.set_item("backend", args.backend.clone());
			let _ = kwargs.set_item("qcs", qcs_pylist);
			let _ = kwargs.set_item("shots", shots);
			let _ = kwargs.set_item("sim_method", args.sim_method.clone());

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

	pub fn new(n_qpus: u32, nodes: u32, id: &str, cores_per_qpu: u32) -> Self {
		let mut runner = CunqaRunner{
			family: None,
		};
		runner.raise_qpus(n_qpus, nodes, id, cores_per_qpu);
		runner
	}

	pub fn raise_qpus(&mut self, n_qpus: u32, nodes: u32, id: &str, cores_per_qpu: u32) {
		println!("Calling python cunqa qraise");
		Python::with_gil(|py| {
			let kwargs = PyDict::new(py);
        	let _ = kwargs.set_item("n", n_qpus);
        	let _ = kwargs.set_item("t", "10:00:00");
			let _ = kwargs.set_item("n_nodes", nodes);
			let _ = kwargs.set_item("family_name", id);
			let _ = kwargs.set_item("cores_per_qpu", cores_per_qpu);

			let module = PyModule::import(py, "polypus_python").unwrap();
			println!("Raising QPUs in cunqa with n_qpus: {n_qpus}, nodes: {nodes}, id: {id}, cores_per_qpu: {cores_per_qpu}");
			let connection = module.call_method("connect_to_infrastructure", ("cunqa", ), Some(&kwargs));
			match connection {
				Ok(obj) => {
						let family: Py<PyAny> = obj.extract().expect("Error extracting family from cunqa qraise");
						println!("QPUs raised successfully");
						self.family = Some(family);
				},
				Err(e) => {
					panic!("{e}, Error raising QPUs in cunqa");
				},
			}
		});
	}

	pub fn drop_qpus(&self) {
		println!("Dropping QPUs");
		Python::with_gil(|py| {
			let module = PyModule::import(py, "polypus_python").unwrap();
			let kwargs = PyDict::new(py);
			let _ = kwargs.set_item("family", self.family.as_ref().expect("family not initialised").clone_ref(py));
			let drop_result = module.call_method("disconnect_from_infrastructure", ("cunqa", ), Some(&kwargs));
			match drop_result {
				Ok(_) => {
					println!("QPUs dropped successfully");
				},
				Err(e) => {
					panic!("{e}, Error dropping QPUs in cunqa");
				},
			}
		});
	}	
}
