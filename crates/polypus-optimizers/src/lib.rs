//! # polypus-optimizers
//!
//! Pure-Rust variational optimizers for Polypus. No Python, no PyO3.
//!
//! This crate holds the parameter-optimization loops used to train variational
//! quantum circuits:
//!
//! - [`AlgorithmDifferentialEvolution`] — Differential Evolution (DE)
//! - [`AlgorithmPSO`] — Particle Swarm Optimization (PSO)
//! - [`AlgorithmQNG`] — Quantum Natural Gradient (QNG)
//!
//! The optimizers are completely decoupled from circuits, backends, and the
//! Python interpreter. They only depend on two input contracts:
//!
//! - [`EvaluationOracle`] maps a batch of parameter vectors to fitness values
//!   (higher is better — the optimizers *maximise*). This is the single seam
//!   between an optimizer and *how* a candidate is scored (statevector sim,
//!   QPU, analytic function, …).
//! - [`VarianceOracle`] (QNG only) returns the diagonal elements of the
//!   Fubini–Study metric (QFIM). It abstracts the one algorithm-specific
//!   callback QNG needs, so no runtime detail leaks into this crate.
//!
//! Because the crate is Python-free, it can be reused by any Rust project
//! (a future `polypus-vqe`, `polypus-qml`, …) without pulling in PyO3.
//!
//! ## Determinism
//!
//! Every optimizer accepts an optional `seed`. With `None` (the default used
//! by the Polypus Python bindings) it draws from [`rand::thread_rng`], exactly
//! as before. With `Some(seed)` the run is fully reproducible, which is what
//! the unit tests rely on.
//!
//! ## Example — optimize a trivial analytic function without Python
//!
//! ```
//! use polypus_optimizers::{
//!     AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs,
//!     EvaluationOracle, Optimizer,
//! };
//!
//! // Fitness = -Σ(xᵢ - 1)² is maximised (value 0) at xᵢ = 1.
//! struct Sphere;
//! impl EvaluationOracle for Sphere {
//!     fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
//!         candidates
//!             .iter()
//!             .map(|c| -c.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>())
//!             .collect()
//!     }
//! }
//!
//! let outcome = AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
//!     oracle: Box::new(Sphere),
//!     population_size: 40,
//!     generations: 200,
//!     dimensions: 2,
//!     tolerance: 1e-6,
//!     seed: Some(7),
//! });
//!
//! // Converges close to the optimum (fitness 0).
//! assert!(outcome.best_fitness > -1e-2, "fitness = {}", outcome.best_fitness);
//! for x in &outcome.best_params {
//!     assert!((x - 1.0).abs() < 0.1);
//! }
//! ```

#![deny(clippy::all)]

pub mod differential_evolution;
pub mod objective;
pub mod outcome;
pub mod pso;
pub mod quantum_natural_gradient;

mod rng;
mod util;

pub use differential_evolution::{
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs,
};
pub use objective::{EvaluationOracle, VarianceOracle};
pub use outcome::{OptimizationOutcome, Optimizer};
pub use pso::{AlgorithmPSO, AlgorithmPSOArgs};
pub use quantum_natural_gradient::{AlgorithmQNG, AlgorithmQNGArgs};
