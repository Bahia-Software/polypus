pub mod orchestration;
pub mod vqc;
pub use crate::algorithms::orchestration::*;
pub use crate::algorithms::vqc::*;

use pyo3::prelude::*;
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

/// Discriminates between plain VQC training and QML training with feature-map encoding.
///
/// - `Vqc`: the optimiser assigns its candidate parameter vector directly to a
///   single circuit template (`base.qcs[0]`).
/// - `Qml`: `base.qcs` contains N pre-bound training circuits (one per sample).
///   For each candidate θ the optimiser binds θ to every circuit, runs them,
///   and returns the mean expectation value as the fitness.
#[derive(Debug, Clone)]
pub enum TrainMode {
    Vqc,
    Qml,
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
	pub cores_per_qpu: u32,
	/// Aer simulation method (e.g. "automatic", "statevector",
	/// "matrix_product_state", "density_matrix").
	pub sim_method: String,
	/// Optional Qiskit NoiseModel passed through to the Aer backend.
	pub noise_model: Option<Py<PyAny>>,
}
