pub mod orchestration;
pub mod vqc;
pub use crate::algorithms::orchestration::*;
pub use crate::algorithms::vqc::*;

use std::fmt;
use crate::infrastructure::{BoundCircuit, ExecutionConfig};

/// Trait for all quantum algorithms in Polypus.
/// Each algorithm should implement this trait for its argument and output types.
pub trait AlgorithmTrait {
    type Args;
    type AlgorithmReturnType: fmt::Display;

    /// Run the algorithm with the given arguments.
    fn run(&self, args: Self::Args) -> Self::AlgorithmReturnType;

    /// Get the algorithm's name.
    fn name(&self) -> String;

    /// Get a description of the algorithm.
    fn description(&self) -> String;
}

/// Arguments for orchestration algorithms (single run, distribute-by-shots).
///
/// Orchestration algorithms operate directly on circuits + infrastructure and
/// do not need an optimisation loop. Circuits arrive as [`BoundCircuit`]s
/// (fully bound Qiskit objects or native OpenQASM 2.0), so orchestration is
/// representation-agnostic. VQC optimisation algorithms (DE, PSO, QNG)
/// use their own `Args` structs that contain an
/// [`crate::evaluation::EvaluationOracle`] instead.
pub struct AlgorithmArgs {
    pub config: ExecutionConfig,
    pub qcs: Vec<BoundCircuit>,
}
