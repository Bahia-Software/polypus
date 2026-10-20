//! Configuration-validation guards for the QNG optimizer.
//!
//! An out-of-domain configuration — zero iterations, a non-positive
//! finite-difference step, a negative or non-finite Tikhonov term, a non-finite
//! learning rate, an infinite bound — and a variance oracle that returns a
//! wrong-length diagonal must all surface a typed [`OptimizerError`] *before or
//! at the seam*, never a silent `-inf` result, a `NaN`, or an out-of-bounds
//! panic deep in the optimization loop.

use polypus_optimizers::{
    AlgorithmQNG, AlgorithmQNGArgs, EvaluationOracle, Optimizer, OptimizerError, VarianceOracle,
};

/// Concave objective maximised at the origin; deterministic.
struct Quadratic;
impl EvaluationOracle for Quadratic {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates
            .iter()
            .map(|c| -c.iter().map(|x| x * x).sum::<f64>())
            .collect()
    }
}

/// Constant diagonal QFIM.
struct ConstVariance(f64);
impl VarianceOracle for ConstVariance {
    fn variance(&self, _theta: &[f64], _param_index: usize) -> f64 {
        self.0
    }
}

/// Returns a diagonal one element short of `dims`. Only a custom Rust
/// `VarianceOracle` can do this (the Python `PyVarianceOracle` maps over
/// `0..dims`), so this is the defense-in-depth path the length check guards.
struct ShortVariance;
impl VarianceOracle for ShortVariance {
    fn variance(&self, _theta: &[f64], _param_index: usize) -> f64 {
        1.0
    }
    fn variance_diagonal(&self, _theta: &[f64], dims: usize) -> Vec<f64> {
        vec![1.0; dims.saturating_sub(1)]
    }
}

fn valid_args() -> AlgorithmQNGArgs {
    AlgorithmQNGArgs {
        oracle: Box::new(Quadratic),
        max_iters: 5,
        learning_rate: 0.1,
        finite_difference_step: 1e-3,
        bounds: (-1.0, 1.0),
        dimensions: 2,
        variance_oracle: Box::new(ConstVariance(1.0)),
        tikhonov_reg: 1e-3,
        seed: Some(42),
    }
}

#[test]
fn valid_config_still_runs() {
    let out = AlgorithmQNG
        .optimize(valid_args())
        .expect("a valid config runs");
    assert_eq!(out.iterations_run, 5);
    assert!(out.best_fitness.is_finite());
}

#[test]
fn rejects_zero_max_iters() {
    // 0 iterations never evaluates the objective and used to report -inf.
    let args = AlgorithmQNGArgs {
        max_iters: 0,
        ..valid_args()
    };
    match AlgorithmQNG.optimize(args) {
        Err(OptimizerError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "max_iters"),
        other => panic!("expected InvalidConfig(max_iters), got {other:?}"),
    }
}

#[test]
fn rejects_nonpositive_finite_difference_step() {
    for step in [0.0, -1e-3] {
        let args = AlgorithmQNGArgs {
            finite_difference_step: step,
            ..valid_args()
        };
        assert!(
            matches!(
                AlgorithmQNG.optimize(args),
                Err(OptimizerError::InvalidConfig {
                    parameter: "finite_difference_step",
                    ..
                })
            ),
            "step {step} must be rejected"
        );
    }
}

#[test]
fn rejects_negative_or_nonfinite_tikhonov() {
    for reg in [-1e-6, f64::NAN, f64::INFINITY] {
        let args = AlgorithmQNGArgs {
            tikhonov_reg: reg,
            ..valid_args()
        };
        assert!(
            matches!(
                AlgorithmQNG.optimize(args),
                Err(OptimizerError::InvalidConfig {
                    parameter: "tikhonov_reg",
                    ..
                })
            ),
            "tikhonov_reg {reg} must be rejected"
        );
    }
}

#[test]
fn rejects_nonfinite_learning_rate() {
    for lr in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let args = AlgorithmQNGArgs {
            learning_rate: lr,
            ..valid_args()
        };
        assert!(
            matches!(
                AlgorithmQNG.optimize(args),
                Err(OptimizerError::InvalidConfig {
                    parameter: "learning_rate",
                    ..
                })
            ),
            "learning_rate {lr} must be rejected"
        );
    }
}

#[test]
fn rejects_infinite_bounds() {
    // `lb < ub` holds for (0, +inf), so the old partial_cmp check let it through;
    // an infinite interval must be rejected before the sampler sees it.
    let args = AlgorithmQNGArgs {
        bounds: (0.0, f64::INFINITY),
        ..valid_args()
    };
    assert!(matches!(
        AlgorithmQNG.optimize(args),
        Err(OptimizerError::InvalidBounds { .. })
    ));
}

#[test]
fn rejects_short_variance_diagonal() {
    let args = AlgorithmQNGArgs {
        variance_oracle: Box::new(ShortVariance),
        ..valid_args()
    };
    assert!(matches!(
        AlgorithmQNG.optimize(args),
        Err(OptimizerError::OracleLengthMismatch {
            expected: 2,
            got: 1
        })
    ));
}
