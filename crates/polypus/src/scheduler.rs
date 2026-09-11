//! Optimizer dispatch — the single place that turns a chosen optimization
//! [`Method`] plus an [`EvaluationOracle`] into an [`OptimizationOutcome`].
//!
//! `train` and `qml_train` used to carry *identical* DE/PSO/QNG dispatch blocks
//! (build the optimizer args, release the GIL, run `optimize`, then surface
//! either the oracle's recorded failure or the optimizer's own config error).
//! That triplication now lives here once:
//! [`dispatch_optimizer`] takes a pyo3-free [`Method`] (the binding parses the
//! `polypus.DE`/`PSO`/`QNG` object into it and builds the QNG variance oracle),
//! runs the matching optimizer, and returns a typed [`OracleError`] the binding
//! converts back into a `PyErr`.
//!
//! This module is deliberately Python-free — the GIL is released by the caller
//! (`py.allow_threads(|| dispatch_optimizer(...))`) and re-acquired only inside
//! the oracle. It is the seed of the future `polypus-scheduler` crate.

use polypus_optimizers::{
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, AlgorithmPSO,
    AlgorithmPSOArgs, AlgorithmQNG, AlgorithmQNGArgs, EvaluationOracle, OptimizationOutcome,
    Optimizer, OptimizerError, VarianceOracle,
};

use crate::evaluation::{EvaluationError, OracleErrorSlot};

/// DE configuration, mirroring the `polypus.DE` object minus the shared
/// execution parameters (`oracle`, `dimensions`, `seed`) dispatch supplies.
pub struct DeConfig {
    pub generations: u32,
    pub population_size: u32,
    pub tolerance: f64,
    pub patience: u32,
}

/// PSO configuration, mirroring the `polypus.PSO` object (see [`DeConfig`]).
pub struct PsoConfig {
    pub generations: u32,
    pub population_size: u32,
    pub bounds: (f64, f64),
    pub inertia_weight: f64,
    pub cognitive_weight: f64,
    pub social_weight: f64,
    pub tolerance: f64,
}

/// QNG configuration, mirroring the `polypus.QNG` object (see [`DeConfig`]). The
/// Python `variance_function` is adapted into a [`VarianceOracle`] by the binding
/// and travels alongside this config in [`Method::Qng`].
pub struct QngConfig {
    pub max_iters: u32,
    pub learning_rate: f64,
    pub finite_difference_step: f64,
    pub bounds: (f64, f64),
    pub tikhonov_reg: f64,
}

/// The chosen optimizer and its configuration. `Qng` additionally carries its
/// variance oracle (built from the Python callback at the binding boundary), so
/// dispatch stays Python-free.
pub enum Method {
    De(DeConfig),
    Pso(PsoConfig),
    Qng(QngConfig, Box<dyn VarianceOracle>),
}

/// A training run's failure, in the order the binding must surface it.
///
/// `Evaluation` (an oracle failure recorded mid-run in the shared slot) takes
/// precedence over `Config`, exactly as the previous inline dispatch did: an
/// oracle error is the *cause*, and the optimizer's own error is often just its
/// downstream symptom.
pub enum OracleError {
    /// The optimizer rejected its configuration before any evaluation
    /// (population too small, empty/non-finite bounds, `max_iters == 0`, ...).
    /// Surfaces as `ValueError`.
    Config(OptimizerError),
    /// An oracle evaluation failed and was recorded in the shared slot during
    /// the run (backend, binding, observable, a Python callback, or a C-5
    /// violation). Surfaces with its original class preserved.
    Evaluation(EvaluationError),
}

/// Run `method` over `oracle` and return the outcome, or the run's failure.
///
/// The caller releases the GIL around this call; the optimizer loop is pure Rust
/// and the oracle re-acquires the GIL internally where it must. An oracle failure
/// recorded in `errors` during the run is returned in preference to the
/// optimizer's own error (see [`OracleError`]).
pub fn dispatch_optimizer(
    method: Method,
    oracle: Box<dyn EvaluationOracle>,
    dimensions: u32,
    errors: &OracleErrorSlot,
    seed: u64,
) -> Result<OptimizationOutcome, OracleError> {
    let result = match method {
        Method::De(cfg) => {
            AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
                oracle,
                population_size: cfg.population_size,
                generations: cfg.generations,
                dimensions,
                tolerance: cfg.tolerance,
                patience: cfg.patience,
                seed: Some(seed),
            })
        }
        Method::Pso(cfg) => AlgorithmPSO.optimize(AlgorithmPSOArgs {
            oracle,
            population_size: cfg.population_size,
            generations: cfg.generations,
            dimensions,
            bounds: cfg.bounds,
            inertia_weight: cfg.inertia_weight,
            cognitive_weight: cfg.cognitive_weight,
            social_weight: cfg.social_weight,
            tolerance: cfg.tolerance,
            seed: Some(seed),
        }),
        Method::Qng(cfg, variance_oracle) => AlgorithmQNG.optimize(AlgorithmQNGArgs {
            oracle,
            max_iters: cfg.max_iters,
            learning_rate: cfg.learning_rate,
            finite_difference_step: cfg.finite_difference_step,
            bounds: cfg.bounds,
            dimensions,
            variance_oracle,
            tikhonov_reg: cfg.tikhonov_reg,
            seed: Some(seed),
        }),
    };

    // An oracle failure recorded during the run is the root cause; surface it
    // ahead of the optimizer's own (often downstream) error.
    if let Some(eval_err) = errors.take() {
        return Err(OracleError::Evaluation(eval_err));
    }
    result.map_err(OracleError::Config)
}
