use crate::algorithms::AlgorithmArgs;

pub enum Infrastructure {
	Local,
	Cunqa,
}

impl Infrastructure {
	pub fn from_str(s: &str) -> Self {
		match s {
			"local" => Infrastructure::Local,
			"Cunqa" => Infrastructure::Cunqa,
			_ => panic!("Unknown infrastructure: {}", s),
		}
	}
}

pub trait QuantumRunner {
	fn run<'py>(&self, args: &AlgorithmArgs<'py>) -> pyo3::PyObject;
}

pub mod local;
pub mod cunqa;
pub use local::LocalRunner;
pub use cunqa::CunqaRunner;