//! Behavioural tests for the optional starting point every optimizer takes,
//! `initial_point` (contract C-5).
//!
//! Like `optimizers.rs`, these run with no Python interpreter. The oracles
//! record every batch they are handed, so a test can check what an optimizer
//! actually evaluated: the point is the first candidate of the first batch
//! (wrapped into `[0, 2π)` for DE), the rest of that batch is what the same
//! seed gives without a point, CMA-ES centres its first generation on it, QNG
//! starts its gradient stencil at it, Adam measures it first, SPSA centres its
//! calibration on it and then measures it first, and a rejected point reaches
//! no oracle.

use polypus_optimizers::{
    AdamRule, AlgorithmAdam, AlgorithmAdamArgs, AlgorithmCmaEs, AlgorithmCmaEsArgs,
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, AlgorithmPSO,
    AlgorithmPSOArgs, AlgorithmQNG, AlgorithmQNGArgs, AlgorithmSPSA, AlgorithmSPSAArgs, DeStrategy,
    EvaluationOracle, FiniteDifferenceGradient, OptimizationOutcome, Optimizer, OptimizerError,
    VarianceOracle,
};
use std::f64::consts::{PI, TAU};
use std::sync::{Arc, Mutex};

const STRATEGIES: [DeStrategy; 3] = [DeStrategy::Rand1Bin, DeStrategy::Shade, DeStrategy::LShade];

/// A point with two coordinates outside `[0, 2π)`, so DE has to wrap it.
const POINT: [f64; 3] = [-0.5, 7.0, 2.25];

/// A point inside both `[0, 2π)` and PSO's bounds `(-π, π)`, which every
/// optimizer takes as it is.
const INSIDE: [f64; 3] = [0.5, 2.0, 1.25];

/// `-Σ (xᵢ − 1)²`, maximised (value 0) at `xᵢ = 1`. Plain products, so its
/// value is bit-identical on every IEEE-754 platform.
fn quadratic(x: &[f64]) -> f64 {
    -x.iter()
        .map(|v| {
            let d = v - 1.0;
            d * d
        })
        .sum::<f64>()
}

/// The quadratic bowl with a sinusoidal ripple: many local optima and
/// near-ties, where a stale best would break `best_fitness == f(best_params)`.
fn rough(x: &[f64]) -> f64 {
    x.iter().map(|v| -(v - 1.0).powi(2) + (5.0 * v).sin()).sum()
}

/// Every candidate scores the same.
fn flat(_: &[f64]) -> f64 {
    0.0
}

/// Every batch an oracle was handed, in order.
type Batches = Arc<Mutex<Vec<Vec<Vec<f64>>>>>;

/// Oracle scoring `fitness` and recording every batch in `batches`.
struct Recorder {
    fitness: fn(&[f64]) -> f64,
    batches: Batches,
}

impl EvaluationOracle for Recorder {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        self.batches.lock().unwrap().push(candidates.to_vec());
        candidates.iter().map(|c| (self.fitness)(c)).collect()
    }
}

/// A recording oracle over `fitness`, and the log it records into.
fn recorder(fitness: fn(&[f64]) -> f64) -> (Box<dyn EvaluationOracle>, Batches) {
    let batches = Batches::default();
    let oracle = Recorder {
        fitness,
        batches: Arc::clone(&batches),
    };
    (Box::new(oracle), batches)
}

fn first_batch(batches: &Batches) -> Vec<Vec<f64>> {
    batches
        .lock()
        .unwrap()
        .first()
        .cloned()
        .expect("the oracle was called")
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

/// Constant QFIM diagonal.
struct ConstVariance;

impl VarianceOracle for ConstVariance {
    fn variance(&self, _theta: &[f64], _param_index: usize) -> f64 {
        1.0
    }
}

fn de_args(
    oracle: Box<dyn EvaluationOracle>,
    initial_point: Option<Vec<f64>>,
    seed: u64,
) -> AlgorithmDifferentialEvolutionArgs {
    AlgorithmDifferentialEvolutionArgs {
        oracle,
        population_size: 8,
        generations: 5,
        dimensions: 3,
        tolerance: 1e-12,
        patience: 1_000,
        seed: Some(seed),
        initial_point,
    }
}

fn pso_args(
    oracle: Box<dyn EvaluationOracle>,
    initial_point: Option<Vec<f64>>,
    seed: u64,
) -> AlgorithmPSOArgs {
    AlgorithmPSOArgs {
        oracle,
        population_size: 8,
        generations: 5,
        dimensions: 3,
        bounds: (-PI, PI),
        inertia_weight: 0.5,
        cognitive_weight: 1.0,
        social_weight: 1.0,
        tolerance: 1e-9,
        seed: Some(seed),
        initial_point,
    }
}

const QNG_STEP: f64 = 0.1;

fn qng_args(
    oracle: Box<dyn EvaluationOracle>,
    initial_point: Option<Vec<f64>>,
    seed: u64,
) -> AlgorithmQNGArgs {
    AlgorithmQNGArgs {
        oracle,
        max_iters: 5,
        learning_rate: 0.1,
        finite_difference_step: QNG_STEP,
        bounds: (-PI, PI),
        dimensions: 3,
        variance_oracle: Box::new(ConstVariance),
        tikhonov_reg: 0.05,
        seed: Some(seed),
        initial_point,
    }
}

fn cmaes_args(
    oracle: Box<dyn EvaluationOracle>,
    initial_point: Option<Vec<f64>>,
    seed: u64,
) -> AlgorithmCmaEsArgs {
    AlgorithmCmaEsArgs {
        oracle,
        population_size: None,
        generations: 5,
        dimensions: 3,
        sigma0: AlgorithmCmaEs::DEFAULT_SIGMA0,
        tolerance: 1e-12,
        patience: 30,
        restarts: 0,
        seed: Some(seed),
        initial_point,
    }
}

const ADAM_STEP: f64 = 0.1;

/// Adam as a training run without an exact gradient dispatches it: a central
/// finite difference of the same oracle that tracks the best solution, so each
/// iteration's one batch, θ and its stencil, reaches the recorder.
fn adam_args(
    oracle: Box<dyn EvaluationOracle>,
    initial_point: Option<Vec<f64>>,
    seed: u64,
) -> AlgorithmAdamArgs {
    let oracle: Arc<dyn EvaluationOracle> = Arc::from(oracle);
    let gradient = FiniteDifferenceGradient::new(Arc::clone(&oracle), ADAM_STEP)
        .expect("a finite, positive step is valid");
    AlgorithmAdamArgs {
        oracle: Box::new(oracle),
        gradient_oracle: Box::new(gradient),
        max_iters: 5,
        learning_rate: 0.1,
        beta1: AlgorithmAdam::DEFAULT_BETA1,
        beta2: AlgorithmAdam::DEFAULT_BETA2,
        epsilon: AlgorithmAdam::DEFAULT_EPSILON,
        bounds: (-PI, PI),
        dimensions: 3,
        tolerance: 1e-12,
        patience: 1_000,
        rule: AdamRule::Adam,
        seed: Some(seed),
        initial_point,
    }
}

const SPSA_STEP: f64 = 0.1;

fn spsa_args(
    oracle: Box<dyn EvaluationOracle>,
    initial_point: Option<Vec<f64>>,
    seed: u64,
) -> AlgorithmSPSAArgs {
    AlgorithmSPSAArgs {
        oracle,
        max_iters: 5,
        resamplings: 1,
        perturbation: SPSA_STEP,
        target_step: 0.63,
        alpha: AlgorithmSPSA::DEFAULT_ALPHA,
        gamma: AlgorithmSPSA::DEFAULT_GAMMA,
        eval_interval: 2,
        dimensions: 3,
        seed: Some(seed),
        initial_point,
    }
}

/// The central-difference stencil around `theta`:
/// `[θ + h·e₀, θ − h·e₀, θ + h·e₁, …]`.
fn stencil(theta: &[f64], h: f64) -> Vec<Vec<f64>> {
    (0..theta.len())
        .flat_map(|i| {
            let mut plus = theta.to_vec();
            let mut minus = theta.to_vec();
            plus[i] += h;
            minus[i] -= h;
            [plus, minus]
        })
        .collect()
}

/// Runs one optimizer over an oracle, from an optional point, with a seed.
type Runner = Box<
    dyn Fn(
        Box<dyn EvaluationOracle>,
        Option<Vec<f64>>,
        u64,
    ) -> Result<OptimizationOutcome, OptimizerError>,
>;

/// Every optimizer, by name, with the arguments above.
fn optimizers() -> Vec<(String, Runner)> {
    let mut all: Vec<(String, Runner)> = STRATEGIES
        .into_iter()
        .map(|strategy| {
            let run: Runner = Box::new(move |oracle, point, seed| {
                AlgorithmDifferentialEvolution
                    .optimize_with_strategy(strategy, de_args(oracle, point, seed))
            });
            (format!("DE {strategy}"), run)
        })
        .collect();
    let pso: Runner =
        Box::new(|oracle, point, seed| AlgorithmPSO.optimize(pso_args(oracle, point, seed)));
    let qng: Runner =
        Box::new(|oracle, point, seed| AlgorithmQNG.optimize(qng_args(oracle, point, seed)));
    let cmaes: Runner =
        Box::new(|oracle, point, seed| AlgorithmCmaEs.optimize(cmaes_args(oracle, point, seed)));
    let adam: Runner =
        Box::new(|oracle, point, seed| AlgorithmAdam.optimize(adam_args(oracle, point, seed)));
    let spsa: Runner =
        Box::new(|oracle, point, seed| AlgorithmSPSA.optimize(spsa_args(oracle, point, seed)));
    all.push(("PSO".to_string(), pso));
    all.push(("QNG".to_string(), qng));
    all.push(("CMA-ES".to_string(), cmaes));
    all.push(("Adam".to_string(), adam));
    all.push(("SPSA".to_string(), spsa));
    all
}

// ─────────────────────────────────────────────────────────────────────────────
// The point is the first candidate; the rest is drawn as without it
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn de_evaluates_the_wrapped_point_first_and_draws_the_rest_as_without_it() {
    let wrapped = POINT.map(|x| x.rem_euclid(TAU));
    assert!(wrapped.iter().all(|x| (0.0..TAU).contains(x)));
    assert_ne!(wrapped, POINT);
    for strategy in STRATEGIES {
        let run = |point| {
            let (oracle, batches) = recorder(quadratic);
            AlgorithmDifferentialEvolution
                .optimize_with_strategy(strategy, de_args(oracle, point, 17))
                .expect("valid DE args optimize successfully");
            first_batch(&batches)
        };
        let with = run(Some(POINT.to_vec()));
        let without = run(None);
        assert_eq!(with.len(), 8, "{strategy}");
        assert_eq!(bits(&with[0]), bits(&wrapped), "{strategy}");
        assert_eq!(
            with[1..],
            without[1..],
            "{strategy}: the other members are the ones drawn without a point"
        );
        assert_ne!(with[0], without[0], "{strategy}");
    }
}

#[test]
fn pso_evaluates_the_point_first_and_draws_the_rest_as_without_it() {
    let run = |point| {
        let (oracle, batches) = recorder(quadratic);
        AlgorithmPSO
            .optimize(pso_args(oracle, point, 17))
            .expect("valid PSO args optimize successfully");
        first_batch(&batches)
    };
    let with = run(Some(INSIDE.to_vec()));
    let without = run(None);
    assert_eq!(with.len(), 8);
    assert_eq!(bits(&with[0]), bits(&INSIDE));
    assert_eq!(
        with[1..],
        without[1..],
        "the other particles are the ones drawn without a point"
    );
    assert_ne!(with[0], without[0]);
}

#[test]
fn qng_starts_its_first_gradient_stencil_at_the_point_even_outside_its_bounds() {
    // QNG's bounds (-π, π) only shape the random draw of θ₀; a point outside
    // them is taken as it is.
    let point = vec![-5.0, 0.5, 9.0];
    let (oracle, batches) = recorder(quadratic);
    AlgorithmQNG
        .optimize(qng_args(oracle, Some(point.clone()), 17))
        .expect("a finite point outside the bounds is accepted");
    // The first batch is the central-difference stencil around θ₀:
    // [θ₀ + h·e₀, θ₀ − h·e₀, θ₀ + h·e₁, …].
    let stencil: Vec<Vec<f64>> = (0..point.len())
        .flat_map(|i| {
            let mut plus = point.clone();
            let mut minus = point.clone();
            plus[i] += QNG_STEP;
            minus[i] -= QNG_STEP;
            [plus, minus]
        })
        .collect();
    assert_eq!(first_batch(&batches), stencil);
}

#[test]
fn adam_evaluates_the_point_first_even_outside_its_bounds() {
    // Like QNG's, Adam's bounds (-π, π) only shape the random draw of θ₀.
    let point = vec![-5.0, 0.5, 9.0];
    let (oracle, batches) = recorder(quadratic);
    AlgorithmAdam
        .optimize(adam_args(oracle, Some(point.clone()), 17))
        .expect("a finite point outside the bounds is accepted");
    // The first batch is θ₀ itself, then the stencil of its first gradient.
    let first = first_batch(&batches);
    assert_eq!(first[0], point);
    assert_eq!(first[1..], stencil(&point, ADAM_STEP)[..]);
}

/// Adam draws θ₀ with or without a point and the point only replaces it, so a
/// run from the very θ₀ its seed draws is the run without a point, batch for
/// batch and bit for bit. (Adam's own seeded outcome is pinned bit for bit in
/// `adam.rs`.)
#[test]
fn adam_from_the_point_its_seed_draws_is_the_run_without_one() {
    let run = |point| {
        let (oracle, batches) = recorder(rough);
        let outcome = AlgorithmAdam
            .optimize(adam_args(oracle, point, 17))
            .expect("valid Adam args optimize successfully");
        let batches = batches.lock().unwrap().clone();
        (outcome, batches)
    };
    let (without, without_batches) = run(None);
    // The first batch is the drawn θ₀, then the stencil around it.
    let first = &without_batches[0];
    let drawn = first[0].clone();
    assert_eq!(first[1..], stencil(&drawn, ADAM_STEP)[..]);
    let (with, with_batches) = run(Some(drawn));
    assert_eq!(with_batches, without_batches);
    assert_eq!(with, without);
}

#[test]
fn spsa_centres_its_calibration_on_the_point_and_measures_it_first() {
    // SPSA takes no bounds and does not wrap: a point outside [0, 2π) is θ₀
    // as it is.
    let point = vec![-5.0, 0.5, 9.0];
    let (oracle, batches) = recorder(quadratic);
    AlgorithmSPSA
        .optimize(spsa_args(oracle, Some(point.clone()), 17))
        .expect("a finite point outside [0, 2π) is accepted");
    let batches = batches.lock().unwrap().clone();
    // The calibration: 25 pairs θ₀ ± c·Δ, Δ of ±1 components.
    assert_eq!(batches[0].len(), 50);
    for pair in batches[0].chunks_exact(2) {
        for j in 0..3 {
            let d = if pair[0][j] > point[j] { 1.0 } else { -1.0 };
            assert_eq!(pair[0][j], point[j] + SPSA_STEP * d, "{pair:?}");
            assert_eq!(pair[1][j], point[j] - SPSA_STEP * d, "{pair:?}");
        }
    }
    // The first iteration measures θ₀ first, then its one pair.
    assert_eq!(batches[1].len(), 3);
    assert_eq!(batches[1][0], point);
}

/// SPSA draws θ₀ with or without a point and the point only replaces it, so a
/// run from the very θ₀ its seed draws is the run without a point, batch for
/// batch and bit for bit.
#[test]
fn spsa_from_the_point_its_seed_draws_is_the_run_without_one() {
    let run = |point| {
        let (oracle, batches) = recorder(rough);
        let outcome = AlgorithmSPSA
            .optimize(spsa_args(oracle, point, 17))
            .expect("valid SPSA args optimize successfully");
        let batches = batches.lock().unwrap().clone();
        (outcome, batches)
    };
    let (without, without_batches) = run(None);
    let drawn = without_batches[1][0].clone();
    assert!(drawn.iter().all(|x| (0.0..TAU).contains(x)), "{drawn:?}");
    let (with, with_batches) = run(Some(drawn));
    assert_eq!(with_batches, without_batches);
    assert_eq!(with, without);
}

#[test]
fn cmaes_centres_its_first_generation_on_the_point() {
    // With a tiny step size every candidate of the first generation is the
    // mean, which is the point wrapped into [0, 2π): two of its coordinates
    // lie outside the circle.
    let (oracle, batches) = recorder(quadratic);
    AlgorithmCmaEs
        .optimize(AlgorithmCmaEsArgs {
            sigma0: 1e-9,
            generations: 1,
            ..cmaes_args(oracle, Some(POINT.to_vec()), 17)
        })
        .expect("valid CMA-ES args optimize successfully");
    let first = first_batch(&batches);
    // 4 + ⌊3 ln 3⌋ = 7 candidates.
    assert_eq!(first.len(), 7);
    for candidate in &first {
        for (x, p) in candidate.iter().zip(POINT) {
            let wrapped = p.rem_euclid(TAU);
            assert!((x - wrapped).abs() < 1e-6, "{candidate:?} vs {wrapped}");
        }
    }
}

#[test]
fn cmaes_wraps_a_far_point_so_its_samples_still_move() {
    // Unwrapped, a mean of 1e17 would absorb every σ·y step (m + σ·y == m), so
    // that coordinate would never move.
    let (oracle, batches) = recorder(quadratic);
    AlgorithmCmaEs
        .optimize(AlgorithmCmaEsArgs {
            sigma0: 0.3,
            generations: 1,
            ..cmaes_args(oracle, Some(vec![1e17, 0.5, -1e17]), 17)
        })
        .expect("valid CMA-ES args optimize successfully");
    let first = first_batch(&batches);
    for coordinate in [0, 2] {
        let mut values: Vec<f64> = first.iter().map(|c| c[coordinate]).collect();
        values.sort_by(f64::total_cmp);
        values.dedup();
        assert_eq!(
            values.len(),
            first.len(),
            "coordinate {coordinate}: {first:?}"
        );
    }
}

#[test]
fn cmaes_restarts_sample_what_they_would_without_a_point() {
    // On a flat landscape the ranking is the sampling order and every run
    // stagnates after `patience + 1` generations, so the distribution evolves
    // from its draws alone. The point moves only the first run's mean; the RNG
    // stream is unchanged, so the restart draws the same fresh mean and samples
    // the very same candidates.
    let run = |point| {
        let (oracle, batches) = recorder(flat);
        let outcome = AlgorithmCmaEs
            .optimize(AlgorithmCmaEsArgs {
                generations: 6,
                tolerance: 1e-9,
                patience: 2,
                restarts: 1,
                ..cmaes_args(oracle, point, 17)
            })
            .expect("valid CMA-ES args optimize successfully");
        assert!(outcome.converged);
        let batches = batches.lock().unwrap().clone();
        batches
    };
    let with = run(Some(POINT.to_vec()));
    let without = run(None);
    assert_eq!(
        with.iter().map(Vec::len).collect::<Vec<_>>(),
        [7, 7, 7, 14, 14, 14]
    );
    assert_ne!(with[..3], without[..3], "the first run starts elsewhere");
    assert_eq!(
        with[3..],
        without[3..],
        "the restart draws as without a point"
    );
}

/// DE's selection and PSO's personal bests never give up a candidate for a
/// worse one, so a run that starts at the optimum returns it.
#[test]
fn de_and_pso_keep_a_point_at_the_optimum() {
    let optimum = vec![1.0, 1.0, 1.0];
    for (name, optimize) in optimizers() {
        if ["QNG", "CMA-ES", "Adam", "SPSA"].contains(&name.as_str()) {
            continue;
        }
        let (oracle, _) = recorder(quadratic);
        let outcome =
            optimize(oracle, Some(optimum.clone()), 3).expect("valid args optimize successfully");
        assert_eq!(outcome.best_params, optimum, "{name}");
        assert_eq!(outcome.best_fitness, 0.0, "{name}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preconditions: rejected before any RNG draw or oracle call
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn a_point_of_the_wrong_length_or_not_finite_is_rejected_before_any_oracle_call() {
    let cases: [(&[f64], &str); 6] = [
        (&[0.5, 1.0], "must have one value per dimension (3), got 2"),
        (
            &[0.5, 1.0, 1.5, 2.0],
            "must have one value per dimension (3), got 4",
        ),
        (&[], "must have one value per dimension (3), got 0"),
        (
            &[0.5, f64::NAN, 1.5],
            "every value must be finite, got NaN at index 1",
        ),
        (
            &[f64::INFINITY, 1.0, 1.5],
            "every value must be finite, got inf at index 0",
        ),
        (
            &[0.5, 1.0, f64::NEG_INFINITY],
            "every value must be finite, got -inf at index 2",
        ),
    ];
    for (name, optimize) in optimizers() {
        for (point, reason) in cases {
            let (oracle, batches) = recorder(quadratic);
            assert_eq!(
                optimize(oracle, Some(point.to_vec()), 1),
                Err(OptimizerError::InvalidConfig {
                    parameter: "initial_point",
                    reason: reason.to_string(),
                }),
                "{name} {point:?}"
            );
            assert!(
                batches.lock().unwrap().is_empty(),
                "{name} {point:?}: the oracle was called"
            );
        }
    }
}

#[test]
fn the_other_configuration_errors_are_reported_first() {
    let bad = || Some(vec![f64::NAN]);
    for strategy in STRATEGIES {
        let (oracle, _) = recorder(quadratic);
        let args = AlgorithmDifferentialEvolutionArgs {
            population_size: 3,
            ..de_args(oracle, bad(), 1)
        };
        assert_eq!(
            AlgorithmDifferentialEvolution.optimize_with_strategy(strategy, args),
            Err(OptimizerError::PopulationTooSmall { got: 3, min: 4 }),
            "{strategy}"
        );
    }
    let (oracle, _) = recorder(quadratic);
    assert_eq!(
        AlgorithmPSO.optimize(AlgorithmPSOArgs {
            bounds: (1.0, 0.0),
            ..pso_args(oracle, bad(), 1)
        }),
        Err(OptimizerError::InvalidBounds { lb: 1.0, ub: 0.0 })
    );
    let (oracle, _) = recorder(quadratic);
    assert!(matches!(
        AlgorithmQNG.optimize(AlgorithmQNGArgs {
            max_iters: 0,
            ..qng_args(oracle, bad(), 1)
        }),
        Err(OptimizerError::InvalidConfig {
            parameter: "max_iters",
            ..
        })
    ));
    let (oracle, _) = recorder(quadratic);
    assert!(matches!(
        AlgorithmCmaEs.optimize(AlgorithmCmaEsArgs {
            sigma0: 0.0,
            ..cmaes_args(oracle, bad(), 1)
        }),
        Err(OptimizerError::InvalidConfig {
            parameter: "sigma0",
            ..
        })
    ));
    let (oracle, _) = recorder(quadratic);
    assert!(matches!(
        AlgorithmAdam.optimize(AlgorithmAdamArgs {
            beta1: 1.0,
            ..adam_args(oracle, bad(), 1)
        }),
        Err(OptimizerError::InvalidConfig {
            parameter: "beta1",
            ..
        })
    ));
    let (oracle, _) = recorder(quadratic);
    assert_eq!(
        AlgorithmAdam.optimize(AlgorithmAdamArgs {
            bounds: (1.0, 0.0),
            ..adam_args(oracle, bad(), 1)
        }),
        Err(OptimizerError::InvalidBounds { lb: 1.0, ub: 0.0 })
    );
    let (oracle, _) = recorder(quadratic);
    assert!(matches!(
        AlgorithmSPSA.optimize(AlgorithmSPSAArgs {
            resamplings: 0,
            ..spsa_args(oracle, bad(), 1)
        }),
        Err(OptimizerError::InvalidConfig {
            parameter: "resamplings",
            ..
        })
    ));
}

#[test]
fn pso_rejects_a_point_outside_its_bounds_and_accepts_one_on_them() {
    // PSO clamps every position to [lb, ub]; a point outside would be moved.
    for (point, index, value) in [([0.0, 3.5, 0.0], 1, 3.5), ([-3.2, 0.0, 0.0], 0, -3.2)] {
        let (oracle, batches) = recorder(quadratic);
        assert_eq!(
            AlgorithmPSO.optimize(pso_args(oracle, Some(point.to_vec()), 1)),
            Err(OptimizerError::InvalidConfig {
                parameter: "initial_point",
                reason: format!(
                    "every value must lie within bounds [{}, {PI}], got {value} at index {index}",
                    -PI
                ),
            })
        );
        assert!(batches.lock().unwrap().is_empty(), "the oracle was called");
    }
    // The bounds themselves are reachable positions, so they are accepted.
    let on_bounds = vec![-PI, PI, 0.0];
    let (oracle, batches) = recorder(quadratic);
    AlgorithmPSO
        .optimize(pso_args(oracle, Some(on_bounds.clone()), 1))
        .expect("a point on the bounds is accepted");
    assert_eq!(first_batch(&batches)[0], on_bounds);
}

// ─────────────────────────────────────────────────────────────────────────────
// Postconditions
// ─────────────────────────────────────────────────────────────────────────────

/// The C-5 postconditions hold from a point as they do without one:
/// `best_fitness` is the oracle's value for `best_params`, a candidate it was
/// handed, and `fitness_history` has one entry per iteration ending at
/// `best_fitness`.
#[test]
fn the_c5_postconditions_hold_with_a_point() {
    for (name, optimize) in optimizers() {
        for seed in 0..10 {
            let (oracle, batches) = recorder(rough);
            let outcome = optimize(oracle, Some(INSIDE.to_vec()), seed)
                .expect("valid args optimize successfully");
            let label = format!("{name} seed {seed}");
            assert_eq!(
                rough(&outcome.best_params).to_bits(),
                outcome.best_fitness.to_bits(),
                "{label}"
            );
            assert!(
                batches
                    .lock()
                    .unwrap()
                    .iter()
                    .flatten()
                    .any(|c| *c == outcome.best_params),
                "{label}: best_params was never handed to the oracle"
            );
            assert_eq!(
                outcome.fitness_history.len(),
                outcome.iterations_run,
                "{label}"
            );
            assert_eq!(
                outcome.fitness_history.last().map(|x| x.to_bits()),
                Some(outcome.best_fitness.to_bits()),
                "{label}"
            );
        }
    }
}

/// Without a point, PSO and QNG return exactly what they returned before
/// `initial_point` existed: these bits were captured from
/// `AlgorithmPSO::optimize` and `AlgorithmQNG::optimize` on the commit before
/// it was added. Both use basic IEEE-754 arithmetic only, so the values hold
/// on every platform. DE/rand/1/bin has its own pinned outcome in
/// `de_strategies.rs`; SHADE, L-SHADE and CMA-ES draw through `ln`, `cos` and
/// `tan`, whose last bit is platform-dependent, so the batch comparisons above
/// cover them instead.
#[test]
fn without_a_point_pso_and_qng_reproduce_their_outcome_from_before_it() {
    struct Pinned {
        best_params: &'static [u64],
        best_fitness: u64,
        iterations_run: usize,
        fitness_history: &'static [u64],
    }
    let check = |name: &str, outcome: OptimizationOutcome, pinned: Pinned| {
        assert_eq!(bits(&outcome.best_params), pinned.best_params, "{name}");
        assert_eq!(
            outcome.best_fitness.to_bits(),
            pinned.best_fitness,
            "{name}"
        );
        assert_eq!(outcome.iterations_run, pinned.iterations_run, "{name}");
        assert_eq!(
            bits(&outcome.fitness_history),
            pinned.fitness_history,
            "{name}"
        );
    };
    let (oracle, _) = recorder(quadratic);
    let pso = AlgorithmPSO
        .optimize(pso_args(oracle, None, 17))
        .expect("valid PSO args optimize successfully");
    check(
        "PSO",
        pso,
        Pinned {
            best_params: &[
                4607496038848384891,
                4606100685759150051,
                4605678846940969397,
            ],
            best_fitness: 13810326187959673574,
            iterations_run: 5,
            fitness_history: &[
                13840257837255229018,
                13831245630386886782,
                13821939311950049051,
                13817587939122341166,
                13810326187959673574,
            ],
        },
    );
    let (oracle, _) = recorder(quadratic);
    let qng = AlgorithmQNG
        .optimize(qng_args(oracle, None, 17))
        .expect("valid QNG args optimize successfully");
    check(
        "QNG",
        qng,
        Pinned {
            best_params: &[
                4607245084161110611,
                4593685091182845813,
                4609210197058490192,
            ],
            best_fitness: 13830265288437403643,
            iterations_run: 5,
            fitness_history: &[
                13840966748754563274,
                13838298729639026657,
                13835629501417703568,
                13832702503308516105,
                13830265288437403643,
            ],
        },
    );
}
