pub mod orchestration;
pub use crate::algorithms::orchestration::*;

// The VQC optimizers (DE, PSO, QNG) now live in the pure-Rust `polypus-optimizers`
// crate. Re-export them so the `polypus::algorithms::*` and `polypus::*` paths keep
// resolving for existing consumers.
pub use polypus_optimizers::{
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, AlgorithmPSO,
    AlgorithmPSOArgs, AlgorithmQNG, AlgorithmQNGArgs, EvaluationOracle, OptimizationOutcome,
    Optimizer, VarianceOracle,
};

use crate::infrastructure::{BoundCircuit, ExecutionConfig};

/// Trait for all quantum algorithms in Polypus.
/// Each algorithm should implement this trait for its argument and output types.
pub trait AlgorithmTrait {
    type Args;
    type AlgorithmReturnType;

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
/// [`EvaluationOracle`] instead.
pub struct AlgorithmArgs {
    pub config: ExecutionConfig,
    pub qcs: Vec<BoundCircuit>,
}
