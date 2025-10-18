// Additional shared helpers/utilities can be added here
pub mod orchestration;
pub mod vcq;

pub use crate::algorithms::orchestration::*;
pub use crate::algorithms::vcq::*;

use pyo3::prelude::*;
use pyo3::types::PyInt;
use std::fmt;

/// Trait for all quantum algorithms in Polypus.
/// Each algorithm should implement this trait for its argument and output types.
pub trait AlgorithmTrait{

    type Args;
    type AlgorithmReturnType: fmt::Display;

	/// Run the algorithm with the given arguments.
	fn run(&self, args: Self::Args) -> Self::AlgorithmReturnType;

	/// Get the algorithm's name.
	fn name(&self) -> String;
	
    /// Get a description of the algorithm.
	fn description(&self) -> String;
}

/// Arguments required to run any quantum algorithm.
#[derive(Debug)]
pub struct AlgorithmArgs {
	pub id: String,
	pub qcs: Vec<Py<PyAny>>,
	pub shots: u32,
	pub n_qpus: u32,
	pub infrastructure: String,
    pub backend: String,
	pub nodes: u32,
}
