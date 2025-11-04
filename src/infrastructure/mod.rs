use crate::algorithms::AlgorithmArgs;

/// Infrastructures supported by Polypus
/// - Local: Runs on local infrastructure (e.g. a local server).
/// - Cunqa: Platform to simulate distributed quantum computing using multiple emulated QPUs. This infraestructure is desgined by CESGA (<https://github.com/CESGA-Quantum-Spain/cunqa>).
pub enum Infrastructure {
	Local,
	Cunqa,
}

impl Infrastructure {
	pub fn from_str(s: &str) -> Self {
		match s {
			"local" => Infrastructure::Local,
			"cunqa" => Infrastructure::Cunqa,
			_ => panic!("Unknown infrastructure: {}", s),
		}
	}
}

/// Trait for quantum circuit runners
/// - run: Run using the given arguments.
/// - close: Close the runner. 
pub trait QuantumRunner {
	fn run(&self, args: &AlgorithmArgs) -> pyo3::PyObject;
	fn close(&self) {}
}

pub mod local;
pub mod cunqa;
pub use local::LocalRunner;
pub use cunqa::CunqaRunner;