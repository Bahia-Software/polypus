// The VQC optimizers (DE, PSO, QNG) live in the pure-Rust `polypus-optimizers`
// crate. Re-export them so the `polypus::algorithms::*` and `polypus::*` paths keep
// resolving for existing consumers.
//
// Orchestration (`run_quantum_circuit`, training) is no longer a local
// `AlgorithmTrait`: it runs through the `polypus-orchestration` `Flow`/`Scheduler`
// at the `polypus` edge (`bindings`), so the former `AlgorithmSingleRun` /
// `DistributeByShotsRun` / `AlgorithmArgs` types are gone.
pub use polypus_optimizers::{
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, AlgorithmPSO,
    AlgorithmPSOArgs, AlgorithmQNG, AlgorithmQNGArgs, EvaluationOracle, OptimizationOutcome,
    Optimizer, VarianceOracle,
};
