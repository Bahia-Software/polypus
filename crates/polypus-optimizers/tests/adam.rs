//! Behavioural tests for [`AlgorithmAdam`] and [`FiniteDifferenceGradient`].
//!
//! Like `cmaes.rs`, these run without a Python interpreter. The oracles are
//! pure functions of the candidate; the ones that record what they are handed
//! let a test check what the optimizer actually sent, and that nothing was
//! sent at all when the configuration is rejected.

use polypus_optimizers::{
    AlgorithmAdam, AlgorithmAdamArgs, EvaluationOracle, FiniteDifferenceGradient, GradientOracle,
    OptimizationOutcome, Optimizer, OptimizerError,
};
use std::sync::{Arc, Mutex};

/// The maximum of every test landscape.
const TARGET: [f64; 3] = [0.7, -1.2, 2.1];

/// A concave quadratic with its maximum `0` at [`TARGET`].
fn bowl(x: &[f64]) -> f64 {
    -x.iter()
        .zip(TARGET)
        .map(|(xi, ti)| (xi - ti).powi(2))
        .sum::<f64>()
}

/// Its exact gradient, uphill.
fn bowl_gradient(x: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(TARGET)
        .map(|(xi, ti)| -2.0 * (xi - ti))
        .collect()
}

/// Scores candidates with a fitness function and logs every batch it is handed.
#[derive(Clone, Debug)]
struct Recorder {
    fitness: fn(&[f64]) -> f64,
    batches: Arc<Mutex<Vec<Vec<Vec<f64>>>>>,
}

impl Recorder {
    fn new(fitness: fn(&[f64]) -> f64) -> Self {
        Recorder {
            fitness,
            batches: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn batches(&self) -> Vec<Vec<Vec<f64>>> {
        self.batches.lock().unwrap().clone()
    }
}

impl EvaluationOracle for Recorder {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        self.batches.lock().unwrap().push(candidates.to_vec());
        candidates.iter().map(|c| (self.fitness)(c)).collect()
    }
}

/// An exact gradient that counts its calls.
#[derive(Clone)]
struct ExactGradient {
    gradient: fn(&[f64]) -> Vec<f64>,
    calls: Arc<Mutex<usize>>,
}

impl ExactGradient {
    fn new(gradient: fn(&[f64]) -> Vec<f64>) -> Self {
        ExactGradient {
            gradient,
            calls: Arc::new(Mutex::new(0)),
        }
    }

    fn calls(&self) -> usize {
        *self.calls.lock().unwrap()
    }
}

impl GradientOracle for ExactGradient {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        (self.gradient)(theta)[param_index]
    }

    fn gradient_batch(&self, theta: &[f64], _dims: usize) -> Vec<f64> {
        *self.calls.lock().unwrap() += 1;
        (self.gradient)(theta)
    }
}

fn args(
    oracle: impl EvaluationOracle + 'static,
    gradient: impl GradientOracle + 'static,
    seed: u64,
) -> AlgorithmAdamArgs {
    AlgorithmAdamArgs {
        oracle: Box::new(oracle),
        gradient_oracle: Box::new(gradient),
        max_iters: 400,
        learning_rate: 0.05,
        beta1: AlgorithmAdam::DEFAULT_BETA1,
        beta2: AlgorithmAdam::DEFAULT_BETA2,
        epsilon: AlgorithmAdam::DEFAULT_EPSILON,
        bounds: (-3.0, 3.0),
        dimensions: TARGET.len() as u32,
        tolerance: 1e-12,
        patience: 50,
        seed: Some(seed),
        initial_point: None,
    }
}

fn bowl_args(seed: u64) -> (AlgorithmAdamArgs, Recorder, ExactGradient) {
    let oracle = Recorder::new(bowl);
    let gradient = ExactGradient::new(bowl_gradient);
    (
        args(oracle.clone(), gradient.clone(), seed),
        oracle,
        gradient,
    )
}

/// Every C-5 postcondition, against the candidates the oracle actually scored.
fn assert_c5(outcome: &OptimizationOutcome, fitness: fn(&[f64]) -> f64, oracle: &Recorder) {
    assert_eq!(outcome.best_fitness, fitness(&outcome.best_params));
    assert_eq!(outcome.fitness_history.len(), outcome.iterations_run);
    assert_eq!(outcome.fitness_history.last(), Some(&outcome.best_fitness));
    assert!(outcome
        .fitness_history
        .windows(2)
        .all(|pair| pair[0] <= pair[1]));
    let scored = oracle.batches().concat();
    assert!(scored.contains(&outcome.best_params));
    assert!(scored
        .iter()
        .all(|candidate| fitness(candidate) <= outcome.best_fitness));
}

#[test]
fn converges_to_the_maximum_of_a_concave_quadratic() {
    let (args, oracle, _) = bowl_args(7);
    let outcome = AlgorithmAdam.optimize(args).unwrap();
    for (x, t) in outcome.best_params.iter().zip(TARGET) {
        assert!((x - t).abs() < 1e-3, "{:?}", outcome.best_params);
    }
    assert!(outcome.best_fitness > -1e-5, "{}", outcome.best_fitness);
    assert_c5(&outcome, bowl, &oracle);
}

#[test]
fn one_gradient_and_one_single_candidate_batch_per_iteration() {
    let (args, oracle, gradient) = bowl_args(3);
    let outcome = AlgorithmAdam.optimize(args).unwrap();
    assert_eq!(gradient.calls(), outcome.iterations_run);
    let batches = oracle.batches();
    assert_eq!(batches.len(), outcome.iterations_run);
    assert!(batches.iter().all(|batch| batch.len() == 1));
}

#[test]
fn best_params_fitness_invariant_holds_across_seeds() {
    for seed in 0..25 {
        let (mut args, oracle, _) = bowl_args(seed);
        // A rough step size, so iterates overshoot and the best is not the last.
        args.learning_rate = 0.8;
        args.max_iters = 60;
        let outcome = AlgorithmAdam.optimize(args).unwrap();
        assert_c5(&outcome, bowl, &oracle);
    }
}

#[test]
fn seeded_runs_are_reproducible() {
    let a = AlgorithmAdam.optimize(bowl_args(11).0).unwrap();
    let b = AlgorithmAdam.optimize(bowl_args(11).0).unwrap();
    assert_eq!(a, b);
    let c = AlgorithmAdam.optimize(bowl_args(12).0).unwrap();
    assert_ne!(a.best_params, c.best_params);
}

#[test]
fn the_initial_point_is_drawn_from_the_bounds() {
    let oracle = Recorder::new(bowl);
    let mut args = args(oracle.clone(), ExactGradient::new(|_| vec![0.0; 3]), 5);
    args.bounds = (1.0, 1.5);
    args.max_iters = 1;
    AlgorithmAdam.optimize(args).unwrap();
    // A zero gradient leaves θ where it was drawn.
    let first = &oracle.batches()[0][0];
    assert!(first.iter().all(|x| (1.0..1.5).contains(x)), "{first:?}");
}

fn flat(_: &[f64]) -> f64 {
    1.0
}

#[test]
fn stagnation_stops_like_de() {
    let oracle = Recorder::new(flat);
    let mut args = args(oracle.clone(), ExactGradient::new(|_| vec![0.0; 3]), 1);
    args.tolerance = 1e-9;
    args.patience = 7;
    let outcome = AlgorithmAdam.optimize(args).unwrap();
    // C-5's rule: the first iteration g >= patience whose best improved by less
    // than `tolerance` over the last `patience` iterations — here g = 7.
    assert!(outcome.converged);
    assert_eq!(outcome.iterations_run, 8);
    assert_c5(&outcome, flat, &oracle);
}

#[test]
fn a_large_patience_never_stops_early() {
    let oracle = Recorder::new(flat);
    let mut args = args(oracle, ExactGradient::new(|_| vec![0.0; 3]), 1);
    args.max_iters = 30;
    args.patience = 30;
    let outcome = AlgorithmAdam.optimize(args).unwrap();
    assert!(!outcome.converged);
    assert_eq!(outcome.iterations_run, 30);
}

/// Runs `edit`ed bowl arguments and returns the error, after checking that
/// neither oracle was called.
fn rejected(edit: impl FnOnce(&mut AlgorithmAdamArgs)) -> OptimizerError {
    let (mut args, oracle, gradient) = bowl_args(0);
    edit(&mut args);
    let err = AlgorithmAdam.optimize(args).unwrap_err();
    assert!(oracle.batches().is_empty(), "the oracle was called");
    assert_eq!(gradient.calls(), 0, "the gradient oracle was called");
    err
}

fn invalid_parameter(err: OptimizerError) -> &'static str {
    match err {
        OptimizerError::InvalidConfig { parameter, .. } => parameter,
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// An edit that breaks one precondition.
type Breaks = fn(&mut AlgorithmAdamArgs);

#[test]
fn invalid_configurations_are_rejected_before_any_oracle_call() {
    let cases: [(&str, Breaks); 11] = [
        ("max_iters", |a| a.max_iters = 0),
        ("dimensions", |a| a.dimensions = 0),
        ("learning_rate", |a| a.learning_rate = 0.0),
        ("learning_rate", |a| a.learning_rate = -0.1),
        ("learning_rate", |a| a.learning_rate = f64::NAN),
        ("beta1", |a| a.beta1 = 1.0),
        ("beta1", |a| a.beta1 = -0.1),
        ("beta2", |a| a.beta2 = 1.0),
        ("beta2", |a| a.beta2 = f64::NAN),
        ("epsilon", |a| a.epsilon = 0.0),
        ("epsilon", |a| a.epsilon = f64::INFINITY),
    ];
    for (parameter, edit) in cases {
        assert_eq!(invalid_parameter(rejected(edit)), parameter);
    }
}

#[test]
fn empty_or_non_finite_bounds_are_rejected() {
    for bounds in [
        (1.0, 1.0),
        (2.0, 1.0),
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
    ] {
        let err = rejected(|a| a.bounds = bounds);
        assert!(
            matches!(err, OptimizerError::InvalidBounds { .. }),
            "{bounds:?}: {err:?}"
        );
    }
}

#[test]
fn a_gradient_of_the_wrong_length_is_an_error() {
    let oracle = Recorder::new(bowl);
    let short = ExactGradient::new(|x| bowl_gradient(x)[..2].to_vec());
    let err = AlgorithmAdam.optimize(args(oracle, short, 0)).unwrap_err();
    assert_eq!(
        err,
        OptimizerError::OracleLengthMismatch {
            expected: 3,
            got: 2
        }
    );
}

/// The bowl's exact gradient with `self.0` in place of component 1.
struct Poisoned(f64);

impl GradientOracle for Poisoned {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        self.gradient_batch(theta, theta.len())[param_index]
    }

    fn gradient_batch(&self, theta: &[f64], _dims: usize) -> Vec<f64> {
        let mut gradient = bowl_gradient(theta);
        gradient[1] = self.0;
        gradient
    }
}

#[test]
fn a_non_finite_gradient_is_an_error() {
    // C-5: every component finite. A `NaN` would leave θ `NaN` and the run
    // would still end `Ok`.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = AlgorithmAdam
            .optimize(args(Recorder::new(bowl), Poisoned(bad), 0))
            .unwrap_err();
        assert!(
            matches!(
                err,
                OptimizerError::NonFiniteGradient { index: 1, value }
                    if value.to_bits() == bad.to_bits()
            ),
            "{err:?}"
        );
    }
}

// --- FiniteDifferenceGradient -------------------------------------------------

#[test]
fn finite_difference_matches_the_exact_gradient() {
    let fd = FiniteDifferenceGradient::new(Recorder::new(bowl), 1e-4).unwrap();
    let theta = [0.3, 0.9, -0.4];
    let exact = bowl_gradient(&theta);
    let batch = fd.gradient_batch(&theta, 3);
    for (k, (approx, exact)) in batch.iter().zip(&exact).enumerate() {
        // The central difference is exact on a quadratic up to rounding.
        assert!((approx - exact).abs() < 1e-8, "{batch:?} vs {exact}");
        assert!((fd.gradient(&theta, k) - exact).abs() < 1e-8);
    }
}

#[test]
fn finite_difference_scores_one_batch_of_the_whole_stencil() {
    let oracle = Recorder::new(bowl);
    let h = 0.25;
    let fd = FiniteDifferenceGradient::new(oracle.clone(), h).unwrap();
    let theta = vec![0.3, 0.9, -0.4];
    fd.gradient_batch(&theta, 3);
    let batches = oracle.batches();
    assert_eq!(batches.len(), 1);
    let stencil: Vec<Vec<f64>> = (0..3)
        .flat_map(|i| {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[i] += h;
            minus[i] -= h;
            [plus, minus]
        })
        .collect();
    assert_eq!(batches[0], stencil);
}

#[test]
fn finite_difference_rejects_a_bad_step() {
    for step in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
        let err = FiniteDifferenceGradient::new(Recorder::new(bowl), step).unwrap_err();
        assert_eq!(invalid_parameter(err), "finite_difference_step");
    }
}

/// Returns one value too few for every batch.
struct Short;

impl EvaluationOracle for Short {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        vec![0.0; candidates.len().saturating_sub(1)]
    }
}

#[test]
fn a_broken_oracle_behind_the_finite_difference_is_reported() {
    let fd = FiniteDifferenceGradient::new(Short, 1e-3).unwrap();
    assert!(fd.gradient_batch(&[0.1, 0.2], 2).is_empty());
    assert!(fd.gradient(&[0.1, 0.2], 0).is_nan());
    let err = AlgorithmAdam
        .optimize(args(
            Recorder::new(bowl),
            FiniteDifferenceGradient::new(Short, 1e-3).unwrap(),
            0,
        ))
        .unwrap_err();
    assert!(
        matches!(
            err,
            OptimizerError::OracleLengthMismatch {
                expected: 3,
                got: 0
            }
        ),
        "{err:?}"
    );
}

#[test]
fn finite_difference_never_panics_on_a_parameter_out_of_range() {
    // `dims` past the point's length, or a `param_index` past it, asks for
    // no candidate and reads as a broken oracle does.
    let oracle = Recorder::new(bowl);
    let fd = FiniteDifferenceGradient::new(oracle.clone(), 1e-3).unwrap();
    assert!(fd.gradient_batch(&[0.1, 0.2], 3).is_empty());
    assert!(fd.gradient(&[0.1, 0.2], 2).is_nan());
    assert!(fd.gradient(&[0.1, 0.2], usize::MAX).is_nan());
    assert!(oracle.batches().is_empty());
    assert_eq!(fd.gradient_batch(&[0.1, 0.2], 2).len(), 2);
}

/// Both an oracle and its gradient, like an oracle that computes its own exact
/// gradient.
struct BowlWithGradient;

impl EvaluationOracle for BowlWithGradient {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates.iter().map(|c| bowl(c)).collect()
    }
}

impl GradientOracle for BowlWithGradient {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        bowl_gradient(theta)[param_index]
    }
}

#[test]
fn one_shared_object_can_be_both_the_oracle_and_the_gradient() {
    let both = Arc::new(BowlWithGradient);
    let outcome = AlgorithmAdam
        .optimize(AlgorithmAdamArgs {
            oracle: Box::new(Arc::clone(&both)),
            gradient_oracle: Box::new(Arc::clone(&both)),
            ..bowl_args(2).0
        })
        .unwrap();
    for (x, t) in outcome.best_params.iter().zip(TARGET) {
        assert!((x - t).abs() < 1e-3, "{:?}", outcome.best_params);
    }
}

#[test]
fn adam_trains_with_a_finite_difference_over_the_same_shared_oracle() {
    let oracle = Recorder::new(bowl);
    let shared: Arc<dyn EvaluationOracle> = Arc::new(oracle.clone());
    let gradient = FiniteDifferenceGradient::new(Arc::clone(&shared), 1e-4).unwrap();
    let outcome = AlgorithmAdam
        .optimize(AlgorithmAdamArgs {
            oracle: Box::new(shared),
            ..args(Recorder::new(bowl), gradient, 9)
        })
        .unwrap();
    for (x, t) in outcome.best_params.iter().zip(TARGET) {
        assert!((x - t).abs() < 1e-3, "{:?}", outcome.best_params);
    }
    // Both the stencils and the tracking batches reached the one shared oracle.
    let sizes: Vec<usize> = oracle.batches().iter().map(Vec::len).collect();
    assert_eq!(sizes.len(), 2 * outcome.iterations_run);
    assert!(sizes.chunks_exact(2).all(|pair| pair == [6, 1]));
    // The stencil points are not candidates: C-5 holds over the tracked ones.
    assert_eq!(outcome.best_fitness, bowl(&outcome.best_params));
    assert_eq!(outcome.fitness_history.last(), Some(&outcome.best_fitness));
}
