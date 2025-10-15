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

    type Args<'py>;
    type AlgorithmReturnType: fmt::Display;

	/// Run the algorithm with the given arguments.
	fn run<'py>(&self, args: Self::Args<'py>) -> Self::AlgorithmReturnType;

	/// Get the algorithm's name.
	fn name(&self) -> String;
	
    /// Get a description of the algorithm.
	fn description(&self) -> String;
}

/// Arguments required to run any quantum algorithm.
#[derive(Debug, Clone)]
pub struct AlgorithmArgs<'py> {
	pub id: String,
	pub qc: Bound<'py, PyAny>,
	pub shots: Option<Bound<'py, PyInt>>,
	pub n_qpus: Option<u32>,
	pub infrastructure: String,
    pub backend: String,
	pub nodes: u32,
}
