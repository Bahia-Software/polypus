//! Behavioural tests for the pure-Rust optimizers.
//!
//! These run in a plain `cargo test` binary with **no Python interpreter**: the
//! crate has no PyO3 dependency at all, so any accidental attempt to touch a
//! GIL could not even compile. Passing tests are therefore structural proof of
//! Python-freedom (the same guarantee `polypus`'s `native_circuit_path.rs`
//! documents), and they exercise the optimizers in complete isolation from the
//! Python extension — evidence a future pure-Rust consumer can reuse them.

use polypus_optimizers::{
    linear_parameter_shift_gradient, AlgorithmAdam, AlgorithmAdamArgs,
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, AlgorithmPSO,
    AlgorithmPSOArgs, AlgorithmQNG, AlgorithmQNGArgs, EvaluationOracle, GradientOracle,
    OptimizationOutcome, Optimizer, OptimizerError, VarianceOracle,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Concave test objective: fitness `= -Σ(xᵢ - target)²`, maximised (value 0)
/// exactly at `xᵢ = target`. The optimizers maximise, so this has a unique,
/// analytically known optimum.
struct Quadratic {
    target: f64,
}

impl EvaluationOracle for Quadratic {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates
            .iter()
            .map(|c| -c.iter().map(|x| (x - self.target).powi(2)).sum::<f64>())
            .collect()
    }
}

/// Rough (non-convex) test objective: a quadratic bowl overlaid with a bounded
/// sinusoidal ripple, so the landscape has many local optima and frequent
/// near-ties. Unlike [`Quadratic`], which is smooth and unimodal — the easy
/// case where a stale champion pointer stays within floating-point noise of the
/// true best — this surface is where a desynced `best`/`best_fitness` diverges
/// arbitrarily, so it is the landscape the C-5 invariant most needs to hold on.
/// Deterministic (a pure function of the parameters), like every oracle here,
/// so a fresh instance re-evaluates any candidate bit-for-bit identically.
struct Multimodal {
    target: f64,
}

impl EvaluationOracle for Multimodal {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates
            .iter()
            .map(|c| {
                c.iter()
                    .map(|x| -(x - self.target).powi(2) + (5.0 * x).sin())
                    .sum::<f64>()
            })
            .collect()
    }
}

/// Oracle that always returns NaN — used to prove the NaN-safe `max_by`
/// comparator (`partial_cmp(..).unwrap_or(Equal)`) never panics.
struct NanOracle;

impl EvaluationOracle for NanOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates.iter().map(|_| f64::NAN).collect()
    }
}

/// Oracle that violates the length contract: it returns one *fewer* fitness
/// value than the candidates it was handed (mirroring a Python
/// `expectation_function` that returns a short list). Used to prove the
/// optimizers surface a typed [`OptimizerError::OracleLengthMismatch`] instead
/// of panicking with an out-of-bounds index deep in the loop.
struct ShortOracle;

impl EvaluationOracle for ShortOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let short = candidates.len().saturating_sub(1);
        candidates.iter().take(short).map(|_| 0.0).collect()
    }
}

/// Constant diagonal QFIM, the simplest [`VarianceOracle`] for exercising QNG.
struct ConstVariance(f64);

impl VarianceOracle for ConstVariance {
    fn variance(&self, _theta: &[f64], _param_index: usize) -> f64 {
        self.0
    }
}

/// Exact gradient of [`Quadratic`]: fitness `= −Σ(xᵢ − target)²`, so
/// `∂fitness/∂xᵢ = −2(xᵢ − target)` (ascent sign, matching the oracle).
struct QuadraticGradient {
    target: f64,
}

impl GradientOracle for QuadraticGradient {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        -2.0 * (theta[param_index] - self.target)
    }
}

/// Exact gradient of [`Multimodal`]: fitness `= Σ(−(xᵢ − target)² + sin(5xᵢ))`,
/// so `∂fitness/∂xᵢ = −2(xᵢ − target) + 5·cos(5xᵢ)`.
struct MultimodalGradient {
    target: f64,
}

impl GradientOracle for MultimodalGradient {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        let x = theta[param_index];
        -2.0 * (x - self.target) + 5.0 * (5.0 * x).cos()
    }
}

/// Gradient oracle that violates the length contract: it returns one *fewer*
/// component than `dims`. The [`GradientOracle`] analogue of [`ShortOracle`],
/// used to prove QNG surfaces [`OptimizerError::OracleLengthMismatch`] instead
/// of indexing the gradient out of bounds.
struct ShortGradientOracle;

impl GradientOracle for ShortGradientOracle {
    fn gradient(&self, _theta: &[f64], _param_index: usize) -> f64 {
        0.0
    }
    fn gradient_batch(&self, _theta: &[f64], dims: usize) -> Vec<f64> {
        vec![0.0; dims.saturating_sub(1)]
    }
}

/// Gradient oracle with a **scripted** norm sequence, the stateful counterpart of
/// the analytic oracles above: the `n`-th [`GradientOracle::gradient_batch`] call
/// returns `[norms[n], 0, …, 0]`, whose L2 norm is exactly `norms[n]`. Once the
/// script runs out the last entry repeats forever, so a run may outlive it.
///
/// This is what lets the `patience` tests drive the convergence check
/// iteration-by-iteration — small, large, small, … — instead of hoping a smooth
/// analytic landscape happens to produce the sequence under test. The counter is
/// an [`AtomicUsize`] because [`GradientOracle`] is `Send + Sync`; QNG and Adam
/// both call `gradient_batch` exactly once per iteration, so call index ==
/// iteration index. [`gradient`](GradientOracle::gradient) is provided for
/// completeness and does **not** advance the script.
struct ScriptedGradient {
    norms: Vec<f64>,
    calls: AtomicUsize,
}

impl ScriptedGradient {
    fn new(norms: &[f64]) -> Self {
        assert!(
            !norms.is_empty(),
            "a scripted gradient needs at least one norm"
        );
        ScriptedGradient {
            norms: norms.to_vec(),
            calls: AtomicUsize::new(0),
        }
    }

    /// The norm scheduled for call `index`, clamped to the last scripted entry.
    fn norm_at(&self, index: usize) -> f64 {
        self.norms[index.min(self.norms.len() - 1)]
    }
}

impl GradientOracle for ScriptedGradient {
    fn gradient(&self, _theta: &[f64], param_index: usize) -> f64 {
        if param_index == 0 {
            self.norm_at(self.calls.load(Ordering::SeqCst))
        } else {
            0.0
        }
    }

    fn gradient_batch(&self, _theta: &[f64], dims: usize) -> Vec<f64> {
        let index = self.calls.fetch_add(1, Ordering::SeqCst);
        let mut grad = vec![0.0; dims];
        if dims > 0 {
            grad[0] = self.norm_at(index);
        }
        grad
    }
}

/// Fitness `= Σᵢ cos(θᵢ)`: a separable, raw-expectation-like objective for which
/// the parameter-shift rule is *exact* in closed form (`∂/∂θᵢ = −sin(θᵢ)`, since
/// `[cos(θ+π/2) − cos(θ−π/2)]/2 = −sin(θ)`). Used to check
/// [`linear_parameter_shift_gradient`] against a known analytic gradient with no
/// shot noise involved.
struct CosSum;

impl EvaluationOracle for CosSum {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates
            .iter()
            .map(|c| c.iter().map(|x| x.cos()).sum())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convergence on a known analytic optimum (deterministic via a fixed seed)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn de_converges_to_known_optimum() {
    let outcome = AlgorithmDifferentialEvolution
        .optimize(AlgorithmDifferentialEvolutionArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            population_size: 40,
            generations: 300,
            dimensions: 3,
            tolerance: 1e-9,
            seed: Some(42),
        })
        .expect("valid DE args optimize successfully");

    assert!(
        outcome.best_fitness > -1e-3,
        "fitness = {}",
        outcome.best_fitness
    );
    for x in &outcome.best_params {
        assert!((x - 1.0).abs() < 0.05, "param off target: {x}");
    }
}

#[test]
fn pso_converges_to_known_optimum() {
    let outcome = AlgorithmPSO
        .optimize(AlgorithmPSOArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            population_size: 40,
            generations: 300,
            dimensions: 3,
            bounds: (-std::f64::consts::PI, std::f64::consts::PI),
            inertia_weight: 0.5,
            cognitive_weight: 1.0,
            social_weight: 1.0,
            tolerance: 1e-9,
            seed: Some(42),
        })
        .expect("valid PSO args optimize successfully");

    assert!(
        outcome.best_fitness > -1e-3,
        "fitness = {}",
        outcome.best_fitness
    );
    for x in &outcome.best_params {
        assert!((x - 1.0).abs() < 0.05, "param off target: {x}");
    }
}

#[test]
fn qng_converges_to_known_optimum() {
    let outcome = AlgorithmQNG
        .optimize(AlgorithmQNGArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            gradient_oracle: Box::new(QuadraticGradient { target: 1.0 }),
            max_iters: 200,
            learning_rate: 0.1,
            bounds: (0.0, 2.0),
            dimensions: 3,
            // A zero tolerance can never fire (‖∇‖ ≥ 0), so the run still
            // exhausts its full iteration budget — the behaviour this test asserts.
            tolerance: 0.0,
            patience: 3,
            variance_oracle: Box::new(ConstVariance(1.0)),
            tikhonov_reg: 0.05,
            seed: Some(42),
        })
        .expect("valid QNG args optimize successfully");

    assert!(
        outcome.best_fitness > -1e-3,
        "fitness = {}",
        outcome.best_fitness
    );
    for x in &outcome.best_params {
        assert!((x - 1.0).abs() < 0.05, "param off target: {x}");
    }
    // With tolerance 0.0 the gradient-norm early-stopping test never fires, so
    // QNG runs the full iteration budget.
    assert_eq!(outcome.iterations_run, 200);
    assert!(!outcome.converged);
}

#[test]
fn adam_converges_to_known_optimum() {
    let outcome = AlgorithmAdam
        .optimize(AlgorithmAdamArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            gradient_oracle: Box::new(QuadraticGradient { target: 1.0 }),
            max_iters: 400,
            learning_rate: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            bounds: (0.0, 2.0),
            dimensions: 3,
            // A zero tolerance can never fire (‖∇‖ ≥ 0), so the run still
            // exhausts its full iteration budget — the behaviour this test asserts.
            tolerance: 0.0,
            patience: 3,
            seed: Some(42),
        })
        .expect("valid Adam args optimize successfully");

    assert!(
        outcome.best_fitness > -1e-3,
        "fitness = {}",
        outcome.best_fitness
    );
    for x in &outcome.best_params {
        assert!((x - 1.0).abs() < 0.05, "param off target: {x}");
    }
    // With tolerance 0.0 the gradient-norm early-stopping test never fires, so
    // Adam runs the full iteration budget.
    assert_eq!(outcome.iterations_run, 400);
    assert!(!outcome.converged);
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism: a fixed seed reproduces the trajectory exactly
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn de_is_deterministic_for_a_fixed_seed() {
    let make = || {
        AlgorithmDifferentialEvolution
            .optimize(AlgorithmDifferentialEvolutionArgs {
                oracle: Box::new(Quadratic { target: 0.7 }),
                population_size: 20,
                generations: 50,
                dimensions: 4,
                tolerance: 1e-9,
                seed: Some(123),
            })
            .expect("valid DE args optimize successfully")
    };
    // OptimizationOutcome derives PartialEq — same seed ⇒ identical outcome.
    assert_eq!(make(), make());
}

#[test]
fn pso_is_deterministic_for_a_fixed_seed() {
    let make = || {
        AlgorithmPSO
            .optimize(AlgorithmPSOArgs {
                oracle: Box::new(Quadratic { target: 0.7 }),
                population_size: 20,
                generations: 50,
                dimensions: 4,
                bounds: (-1.0, 2.0),
                inertia_weight: 0.5,
                cognitive_weight: 1.0,
                social_weight: 1.0,
                tolerance: 1e-9,
                seed: Some(123),
            })
            .expect("valid PSO args optimize successfully")
    };
    assert_eq!(make(), make());
}

#[test]
fn adam_is_deterministic_for_a_fixed_seed() {
    let make = || {
        AlgorithmAdam
            .optimize(AlgorithmAdamArgs {
                oracle: Box::new(Quadratic { target: 0.7 }),
                gradient_oracle: Box::new(QuadraticGradient { target: 0.7 }),
                max_iters: 50,
                learning_rate: 0.05,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                bounds: (0.0, 2.0),
                dimensions: 4,
                tolerance: 0.0,
                patience: 3,
                seed: Some(123),
            })
            .expect("valid Adam args optimize successfully")
    };
    // OptimizationOutcome derives PartialEq — same seed ⇒ identical outcome.
    assert_eq!(make(), make());
}

// ─────────────────────────────────────────────────────────────────────────────
// Early-stopping bookkeeping (iterations_run / converged) is reproducible
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn de_early_stops_and_reports_it() {
    // A loose tolerance makes the per-dimension `std_i < tolerance` test fire
    // before the generation budget is exhausted.
    let make = || {
        AlgorithmDifferentialEvolution
            .optimize(AlgorithmDifferentialEvolutionArgs {
                oracle: Box::new(Quadratic { target: 1.0 }),
                population_size: 30,
                generations: 500,
                dimensions: 3,
                tolerance: 0.5,
                seed: Some(7),
            })
            .expect("valid DE args optimize successfully")
    };
    let outcome = make();
    assert!(outcome.converged, "expected early convergence");
    assert!(
        outcome.iterations_run < 500,
        "iterations_run = {}",
        outcome.iterations_run
    );
    // The recorded iteration count is identical across runs with the same seed.
    assert_eq!(outcome.iterations_run, make().iterations_run);
}

#[test]
fn pso_early_stops_on_collapse_with_symmetric_bounds() {
    // Regression test for the dimensionally-incoherent convergence criterion.
    // With PSO's *real* default symmetric bounds (-π, π) and an optimum at 0,
    // the swarm collapses around 0, so every per-dimension mean → 0. The old
    // criterion `std < tolerance * mean` then reduced to `std < 0`, which
    // (std ≥ 0) can *never* fire — early stopping was dead on arrival for the
    // most common configuration. The per-dimension absolute test `std_i <
    // tolerance` fires as designed. This is the acceptance-criterion test.
    let make = || {
        AlgorithmPSO
            .optimize(AlgorithmPSOArgs {
                oracle: Box::new(Quadratic { target: 0.0 }),
                population_size: 40,
                generations: 2000,
                dimensions: 3,
                bounds: (-std::f64::consts::PI, std::f64::consts::PI),
                inertia_weight: 0.5,
                cognitive_weight: 1.0,
                social_weight: 1.0,
                tolerance: 0.05,
                seed: Some(42),
            })
            .expect("valid PSO args optimize successfully")
    };
    let outcome = make();
    assert!(
        outcome.converged,
        "expected early convergence under symmetric bounds (-π, π)"
    );
    assert!(
        outcome.iterations_run < 2000,
        "iterations_run = {}",
        outcome.iterations_run
    );
    // Reproducible across runs with the same seed.
    assert_eq!(outcome.iterations_run, make().iterations_run);
}

#[test]
fn qng_early_stops_on_small_gradient() {
    // A generous tolerance makes the gradient-norm test ‖∇fitness(θ)‖ < tolerance
    // fire once QNG has descended close enough to the optimum, well before the
    // iteration budget is exhausted. QuadraticGradient is exact, so the norm
    // shrinks monotonically toward zero as θ → target — which also means the
    // default `patience = 3` costs only the two extra iterations it takes to make
    // the streak consecutive, never the stop itself.
    let make = || {
        AlgorithmQNG
            .optimize(AlgorithmQNGArgs {
                oracle: Box::new(Quadratic { target: 1.0 }),
                gradient_oracle: Box::new(QuadraticGradient { target: 1.0 }),
                max_iters: 500,
                learning_rate: 0.1,
                bounds: (0.0, 2.0),
                dimensions: 3,
                tolerance: 0.5,
                patience: 3,
                variance_oracle: Box::new(ConstVariance(1.0)),
                tikhonov_reg: 0.05,
                seed: Some(7),
            })
            .expect("valid QNG args optimize successfully")
    };
    let outcome = make();
    assert!(outcome.converged, "expected early convergence");
    assert!(
        outcome.iterations_run < 500,
        "iterations_run = {}",
        outcome.iterations_run
    );
    // The recorded iteration count is identical across runs with the same seed.
    assert_eq!(outcome.iterations_run, make().iterations_run);
}

#[test]
fn adam_early_stops_on_small_gradient() {
    // Mirror of `qng_early_stops_on_small_gradient`: a generous tolerance makes
    // Adam's gradient-norm test fire before the iteration budget is exhausted,
    // once the exact QuadraticGradient has shrunk near the optimum. The
    // monotonically shrinking norm satisfies the default `patience = 3` a couple
    // of iterations later, not never.
    let make = || {
        AlgorithmAdam
            .optimize(AlgorithmAdamArgs {
                oracle: Box::new(Quadratic { target: 1.0 }),
                gradient_oracle: Box::new(QuadraticGradient { target: 1.0 }),
                max_iters: 500,
                learning_rate: 0.05,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                bounds: (0.0, 2.0),
                dimensions: 3,
                tolerance: 0.5,
                patience: 3,
                seed: Some(7),
            })
            .expect("valid Adam args optimize successfully")
    };
    let outcome = make();
    assert!(outcome.converged, "expected early convergence");
    assert!(
        outcome.iterations_run < 500,
        "iterations_run = {}",
        outcome.iterations_run
    );
    // The recorded iteration count is identical across runs with the same seed.
    assert_eq!(outcome.iterations_run, make().iterations_run);
}

// ─────────────────────────────────────────────────────────────────────────────
// `patience`: consecutive sub-tolerance iterations, not just one
//
// A single sub-tolerance gradient norm cannot be trusted, because the norm may
// come from a minibatch that cancelled to exactly zero at a θ whose full-dataset
// gradient is far from zero (the minibatch note beside C-5/C-7 in
// `docs/CONTRACTS.md`). The optimizers cannot tell the two apart — `GradientOracle`
// deliberately hides it — so they require `patience` *consecutive* iterations
// below `tolerance`. These tests drive that rule directly with `ScriptedGradient`.
// ─────────────────────────────────────────────────────────────────────────────

/// Run QNG over a scripted gradient-norm sequence, returning the outcome. All
/// non-`patience` knobs are fixed so only the script and `patience` vary.
fn qng_over_script(norms: &[f64], patience: usize, max_iters: u32) -> OptimizationOutcome {
    AlgorithmQNG
        .optimize(AlgorithmQNGArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            gradient_oracle: Box::new(ScriptedGradient::new(norms)),
            max_iters,
            learning_rate: 0.1,
            bounds: (0.0, 2.0),
            dimensions: 3,
            tolerance: 0.01,
            patience,
            variance_oracle: Box::new(ConstVariance(1.0)),
            tikhonov_reg: 0.05,
            seed: Some(7),
        })
        .expect("valid QNG args optimize successfully")
}

/// Adam mirror of [`qng_over_script`].
fn adam_over_script(norms: &[f64], patience: usize, max_iters: u32) -> OptimizationOutcome {
    AlgorithmAdam
        .optimize(AlgorithmAdamArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            gradient_oracle: Box::new(ScriptedGradient::new(norms)),
            max_iters,
            learning_rate: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            bounds: (0.0, 2.0),
            dimensions: 3,
            tolerance: 0.01,
            patience,
            seed: Some(7),
        })
        .expect("valid Adam args optimize successfully")
}

#[test]
fn qng_patience_of_one_reproduces_the_single_iteration_rule() {
    // `patience = 1` must be *exactly* the pre-`patience` behaviour — a design
    // property of the field, not an accident of the counter's arithmetic. With a
    // norm already below `tolerance` on the very first iteration, the run stops
    // there, as the old `if grad_norm < tolerance { break }` did.
    let outcome = qng_over_script(&[0.001], 1, 500);
    assert!(outcome.converged, "expected convergence on iteration 1");
    assert_eq!(outcome.iterations_run, 1);
}

#[test]
fn qng_one_sub_tolerance_iteration_is_not_enough_beyond_patience_one() {
    // The whole point of the fix: iteration 1 dips below `tolerance` (the shape a
    // cancelling minibatch produces) and iteration 2 is back above it. With
    // `patience = 3` the run must *not* stop on iteration 1 — it exhausts its
    // budget, because the script never yields three consecutive small norms.
    let outcome = qng_over_script(&[0.001, 5.0], 3, 6);
    assert!(
        !outcome.converged,
        "a single sub-tolerance iteration must not report convergence"
    );
    assert_eq!(outcome.iterations_run, 6);
}

#[test]
fn qng_patience_counts_consecutive_not_cumulative_sub_tolerance_iterations() {
    // small, large, small, small → three sub-tolerance iterations in *total* but
    // never three in a row, so the streak reset must keep `converged` false.
    let interrupted = qng_over_script(&[0.001, 5.0, 0.001, 0.001], 3, 4);
    assert!(
        !interrupted.converged,
        "three non-consecutive sub-tolerance iterations must not converge"
    );
    assert_eq!(interrupted.iterations_run, 4);

    // The same prefix plus one more small iteration *does* reach three in a row,
    // on iteration 5 — proving the counter resumed from 0 after the large norm
    // rather than carrying the first small iteration over.
    let sustained = qng_over_script(&[0.001, 5.0, 0.001, 0.001, 0.001], 3, 10);
    assert!(
        sustained.converged,
        "expected convergence once 3 are consecutive"
    );
    assert_eq!(sustained.iterations_run, 5);
}

#[test]
fn qng_converges_after_patience_consecutive_sub_tolerance_iterations() {
    // Symmetric positive case: one large norm, then exactly `patience` small
    // ones. The stop lands on the last of the three, never earlier.
    let outcome = qng_over_script(&[5.0, 0.001, 0.001, 0.001], 3, 10);
    assert!(outcome.converged, "expected convergence on iteration 4");
    assert_eq!(outcome.iterations_run, 4);

    // With `patience = 2` the same script stops one iteration sooner, so the
    // field really is what sets the streak length.
    let impatient = qng_over_script(&[5.0, 0.001, 0.001, 0.001], 2, 10);
    assert!(impatient.converged);
    assert_eq!(impatient.iterations_run, 3);
}

#[test]
fn adam_patience_of_one_reproduces_the_single_iteration_rule() {
    // Adam mirror: the rule lives in the shared convergence check, so both
    // optimizers must honour `patience = 1` as the pre-`patience` behaviour.
    let outcome = adam_over_script(&[0.001], 1, 500);
    assert!(outcome.converged, "expected convergence on iteration 1");
    assert_eq!(outcome.iterations_run, 1);
}

#[test]
fn adam_converges_after_patience_consecutive_sub_tolerance_iterations() {
    // Adam mirror of the positive case, including the reset: a lone small norm
    // followed by a large one buys nothing.
    let outcome = adam_over_script(&[5.0, 0.001, 0.001, 0.001], 3, 10);
    assert!(outcome.converged, "expected convergence on iteration 4");
    assert_eq!(outcome.iterations_run, 4);

    let interrupted = adam_over_script(&[0.001, 5.0], 3, 6);
    assert!(
        !interrupted.converged,
        "a single sub-tolerance iteration must not report convergence"
    );
    assert_eq!(interrupted.iterations_run, 6);
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn de_handles_zero_dimensions() {
    let outcome = AlgorithmDifferentialEvolution
        .optimize(AlgorithmDifferentialEvolutionArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            population_size: 10,
            generations: 5,
            dimensions: 0,
            tolerance: 1e-9,
            seed: Some(1),
        })
        .expect("valid DE args optimize successfully");
    assert!(outcome.best_params.is_empty());
}

#[test]
fn de_handles_minimum_population() {
    // DE needs population_size >= 4 to sample 3 distinct other members.
    let outcome = AlgorithmDifferentialEvolution
        .optimize(AlgorithmDifferentialEvolutionArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            population_size: 4,
            generations: 10,
            dimensions: 2,
            tolerance: 1e-9,
            seed: Some(1),
        })
        .expect("valid DE args optimize successfully");
    assert_eq!(outcome.best_params.len(), 2);
}

#[test]
fn de_rejects_population_below_four() {
    // Precondition: sampling 3 distinct other members needs population_size >= 4.
    // Sizes 0..=3 previously panicked with an out-of-bounds `sel[2]` inside the
    // trial loop; they must now return a typed error before any RNG draw.
    for popsize in 0..4u32 {
        let result = AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            population_size: popsize,
            generations: 10,
            dimensions: 2,
            tolerance: 1e-9,
            seed: Some(1),
        });
        assert!(
            matches!(
                result,
                Err(OptimizerError::PopulationTooSmall { got, min: 4 }) if got == popsize as usize
            ),
            "population_size {popsize} should be rejected, got {result:?}"
        );
    }
}

#[test]
fn de_nan_fitness_does_not_panic() {
    let outcome = AlgorithmDifferentialEvolution
        .optimize(AlgorithmDifferentialEvolutionArgs {
            oracle: Box::new(NanOracle),
            population_size: 6,
            generations: 5,
            dimensions: 2,
            tolerance: 1e-9,
            seed: Some(1),
        })
        .expect("valid DE args optimize successfully");
    assert_eq!(outcome.best_params.len(), 2);
    assert!(outcome.best_fitness.is_nan());
}

#[test]
fn de_short_oracle_returns_error_not_panic() {
    // An oracle that returns fewer fitness values than candidates would make the
    // selection loop index out of bounds; it must surface a typed error instead.
    let result = AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
        oracle: Box::new(ShortOracle),
        population_size: 6,
        generations: 5,
        dimensions: 2,
        tolerance: 1e-9,
        seed: Some(1),
    });
    assert!(
        matches!(
            result,
            Err(OptimizerError::OracleLengthMismatch {
                expected: 6,
                got: 5
            })
        ),
        "expected OracleLengthMismatch, got {result:?}"
    );
}

#[test]
fn pso_nan_fitness_does_not_panic() {
    let outcome = AlgorithmPSO
        .optimize(AlgorithmPSOArgs {
            oracle: Box::new(NanOracle),
            population_size: 6,
            generations: 5,
            dimensions: 2,
            bounds: (-1.0, 1.0),
            inertia_weight: 0.5,
            cognitive_weight: 1.0,
            social_weight: 1.0,
            tolerance: 1e-9,
            seed: Some(1),
        })
        .expect("valid PSO args optimize successfully");
    assert_eq!(outcome.best_params.len(), 2);
}

#[test]
fn pso_short_oracle_returns_error_not_panic() {
    let result = AlgorithmPSO.optimize(AlgorithmPSOArgs {
        oracle: Box::new(ShortOracle),
        population_size: 6,
        generations: 5,
        dimensions: 2,
        bounds: (-1.0, 1.0),
        inertia_weight: 0.5,
        cognitive_weight: 1.0,
        social_weight: 1.0,
        tolerance: 1e-9,
        seed: Some(1),
    });
    assert!(
        matches!(
            result,
            Err(OptimizerError::OracleLengthMismatch {
                expected: 6,
                got: 5
            })
        ),
        "expected OracleLengthMismatch, got {result:?}"
    );
}

#[test]
fn qng_tikhonov_avoids_division_blowup_when_qfim_is_zero() {
    // A zero variance would make the raw QFIM diagonal 0; the Tikhonov term
    // keeps the denominator at 0.05 so the update stays finite instead of
    // producing inf/NaN from a divide-by-zero.
    let outcome = AlgorithmQNG
        .optimize(AlgorithmQNGArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            gradient_oracle: Box::new(QuadraticGradient { target: 1.0 }),
            max_iters: 1,
            learning_rate: 0.1,
            bounds: (0.0, 2.0),
            dimensions: 2,
            tolerance: 0.0,
            patience: 3,
            variance_oracle: Box::new(ConstVariance(0.0)),
            tikhonov_reg: 0.05,
            seed: Some(1),
        })
        .expect("valid QNG args optimize successfully");
    for x in &outcome.best_params {
        assert!(x.is_finite(), "non-finite parameter: {x}");
    }
}

#[test]
fn qng_short_gradient_oracle_returns_error_not_panic() {
    // QNG now assembles its update from `gradient_oracle.gradient_batch(θ, dims)`
    // and indexes it positionally per parameter. A gradient oracle that returns
    // fewer than `dims` components (here dims−1 = 1) would make that indexing
    // panic — it must surface a typed error instead. The fitness `oracle` is
    // irrelevant: the gradient step runs (and fails the length check) first.
    let result = AlgorithmQNG.optimize(AlgorithmQNGArgs {
        oracle: Box::new(Quadratic { target: 1.0 }),
        gradient_oracle: Box::new(ShortGradientOracle),
        max_iters: 5,
        learning_rate: 0.1,
        bounds: (0.0, 2.0),
        dimensions: 2,
        tolerance: 0.0,
        patience: 3,
        variance_oracle: Box::new(ConstVariance(1.0)),
        tikhonov_reg: 0.05,
        seed: Some(1),
    });
    assert!(
        matches!(
            result,
            Err(OptimizerError::OracleLengthMismatch {
                expected: 2,
                got: 1
            })
        ),
        "expected OracleLengthMismatch, got {result:?}"
    );
}

#[test]
fn pso_rejects_empty_bounds() {
    // Precondition: bounds must be a non-empty interval (lb < ub). An empty
    // (lb == ub) or inverted (lb > ub) range previously panicked inside the
    // uniform sampler; both must now return a typed error before any RNG draw.
    for bounds in [(1.0, 1.0), (2.0, 1.0)] {
        let result = AlgorithmPSO.optimize(AlgorithmPSOArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            population_size: 10,
            generations: 5,
            dimensions: 2,
            bounds,
            inertia_weight: 0.5,
            cognitive_weight: 1.0,
            social_weight: 1.0,
            tolerance: 1e-9,
            seed: Some(1),
        });
        assert!(
            matches!(result, Err(OptimizerError::InvalidBounds { .. })),
            "bounds {bounds:?} should be rejected, got {result:?}"
        );
    }
}

#[test]
fn qng_rejects_empty_bounds() {
    // QNG draws θ from [lb, ub) exactly like PSO, so an empty interval is the
    // same panic risk and must likewise return a typed error, not panic.
    let result = AlgorithmQNG.optimize(AlgorithmQNGArgs {
        oracle: Box::new(Quadratic { target: 1.0 }),
        gradient_oracle: Box::new(QuadraticGradient { target: 1.0 }),
        max_iters: 5,
        learning_rate: 0.1,
        bounds: (1.0, 1.0),
        dimensions: 2,
        tolerance: 0.0,
        patience: 3,
        variance_oracle: Box::new(ConstVariance(1.0)),
        tikhonov_reg: 0.05,
        seed: Some(1),
    });
    assert!(
        matches!(result, Err(OptimizerError::InvalidBounds { .. })),
        "empty QNG bounds should be rejected, got {result:?}"
    );
}

#[test]
fn adam_short_gradient_oracle_returns_error_not_panic() {
    // Adam assembles its update from `gradient_oracle.gradient_batch(θ, dims)`
    // and indexes it positionally per parameter, exactly like QNG. A gradient
    // oracle returning fewer than `dims` components (here dims−1 = 1) would make
    // that indexing panic — it must surface a typed error instead. The fitness
    // `oracle` is irrelevant: the gradient step runs (and fails the length
    // check) first.
    let result = AlgorithmAdam.optimize(AlgorithmAdamArgs {
        oracle: Box::new(Quadratic { target: 1.0 }),
        gradient_oracle: Box::new(ShortGradientOracle),
        max_iters: 5,
        learning_rate: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        bounds: (0.0, 2.0),
        dimensions: 2,
        tolerance: 0.0,
        patience: 3,
        seed: Some(1),
    });
    assert!(
        matches!(
            result,
            Err(OptimizerError::OracleLengthMismatch {
                expected: 2,
                got: 1
            })
        ),
        "expected OracleLengthMismatch, got {result:?}"
    );
}

#[test]
fn adam_rejects_empty_bounds() {
    // Adam draws θ from [lb, ub) exactly like QNG/PSO, so an empty interval is
    // the same panic risk and must likewise return a typed error, not panic.
    let result = AlgorithmAdam.optimize(AlgorithmAdamArgs {
        oracle: Box::new(Quadratic { target: 1.0 }),
        gradient_oracle: Box::new(QuadraticGradient { target: 1.0 }),
        max_iters: 5,
        learning_rate: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        bounds: (1.0, 1.0),
        dimensions: 2,
        tolerance: 0.0,
        patience: 3,
        seed: Some(1),
    });
    assert!(
        matches!(result, Err(OptimizerError::InvalidBounds { .. })),
        "empty Adam bounds should be rejected, got {result:?}"
    );
}

#[test]
fn linear_parameter_shift_matches_analytic_gradient() {
    // For CosSum (fitness = Σ cos θᵢ) the parameter-shift rule is exact in
    // closed form: ∂/∂θᵢ = −sin θᵢ, and [cos(θ+π/2) − cos(θ−π/2)]/2 = −sin θ.
    // No shot noise is involved (the oracle is analytic), so equality is tight.
    let theta = vec![0.3, -1.1, 2.0, 0.0];
    let grad = linear_parameter_shift_gradient(&CosSum, &theta, theta.len());
    assert_eq!(grad.len(), theta.len());
    for (i, &g) in grad.iter().enumerate() {
        let expected = -theta[i].sin();
        assert!(
            (g - expected).abs() < 1e-12,
            "param {i}: parameter-shift {g} vs analytic {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// C-5 invariant: the reported best_fitness must describe the reported
// best_params. In a correct run both come from the *same* `evaluate_batch`
// call, so re-evaluating the (deterministic) oracle at best_params reproduces
// best_fitness *bit-for-bit* — the contract is exact equality, not "close
// enough". These guard the DE champion-self-improvement ordering bug, where
// best_idx/best were left pointing at a stale, arbitrarily worse vector.
// ─────────────────────────────────────────────────────────────────────────────

/// Re-evaluate a fresh instance of the same oracle at `outcome.best_params` and
/// assert it equals `outcome.best_fitness` exactly. A fresh instance is used
/// (not the one moved into the optimizer) because every oracle here is a pure,
/// deterministic function of its input, so the recheck must return identical
/// bits — anything else means the optimizer reported a fitness that does not
/// belong to the parameters it returned.
fn assert_reported_fitness_matches_params(
    label: &str,
    seed: u64,
    outcome: &OptimizationOutcome,
    recheck: &dyn EvaluationOracle,
) {
    let recomputed = recheck.evaluate_batch(std::slice::from_ref(&outcome.best_params))[0];
    assert_eq!(
        recomputed, outcome.best_fitness,
        "{label} seed {seed}: f(best_params) = {recomputed} != best_fitness = {}",
        outcome.best_fitness,
    );
}

#[test]
fn de_best_params_fitness_invariant_holds_across_seeds() {
    for seed in 0..20 {
        // The smooth Quadratic is the easy case: even the buggy ordering left
        // only a tiny gap here, which is exactly why the bug survived. The
        // rough Multimodal surface — many local optima, frequent argmax
        // near-ties — is where a stale champion diverges, so both are checked.
        let quad = AlgorithmDifferentialEvolution
            .optimize(AlgorithmDifferentialEvolutionArgs {
                oracle: Box::new(Quadratic { target: 1.0 }),
                population_size: 30,
                generations: 120,
                dimensions: 4,
                tolerance: 1e-9,
                seed: Some(seed),
            })
            .expect("valid DE args optimize successfully");
        assert_reported_fitness_matches_params(
            "de/quadratic",
            seed,
            &quad,
            &Quadratic { target: 1.0 },
        );

        let multi = AlgorithmDifferentialEvolution
            .optimize(AlgorithmDifferentialEvolutionArgs {
                oracle: Box::new(Multimodal { target: 1.0 }),
                population_size: 30,
                generations: 120,
                dimensions: 4,
                tolerance: 1e-9,
                seed: Some(seed),
            })
            .expect("valid DE args optimize successfully");
        assert_reported_fitness_matches_params(
            "de/multimodal",
            seed,
            &multi,
            &Multimodal { target: 1.0 },
        );
    }
}

#[test]
fn pso_best_params_fitness_invariant_holds_across_seeds() {
    // PSO already recomputes its global best via argmax after updating every
    // personal best, so this should pass — but C-5 requires the invariant
    // proven for all three optimizers, not assumed. Multimodal exercises the
    // same argmax near-ties DE hits.
    for seed in 0..20 {
        let outcome = AlgorithmPSO
            .optimize(AlgorithmPSOArgs {
                oracle: Box::new(Multimodal { target: 1.0 }),
                population_size: 30,
                generations: 120,
                dimensions: 4,
                bounds: (-std::f64::consts::PI, std::f64::consts::PI),
                inertia_weight: 0.5,
                cognitive_weight: 1.0,
                social_weight: 1.0,
                tolerance: 1e-9,
                seed: Some(seed),
            })
            .expect("valid PSO args optimize successfully");
        assert_reported_fitness_matches_params(
            "pso/multimodal",
            seed,
            &outcome,
            &Multimodal { target: 1.0 },
        );
    }
}

#[test]
fn qng_best_params_fitness_invariant_holds_across_seeds() {
    // QNG updates best_params/best_fitness atomically from the same evaluation,
    // so this should pass — asserted here so all three optimizers are covered.
    for seed in 0..20 {
        let outcome = AlgorithmQNG
            .optimize(AlgorithmQNGArgs {
                oracle: Box::new(Multimodal { target: 1.0 }),
                gradient_oracle: Box::new(MultimodalGradient { target: 1.0 }),
                max_iters: 120,
                learning_rate: 0.1,
                bounds: (0.0, 2.0),
                dimensions: 4,
                tolerance: 0.0,
                patience: 3,
                variance_oracle: Box::new(ConstVariance(1.0)),
                tikhonov_reg: 0.05,
                seed: Some(seed),
            })
            .expect("valid QNG args optimize successfully");
        assert_reported_fitness_matches_params(
            "qng/multimodal",
            seed,
            &outcome,
            &Multimodal { target: 1.0 },
        );
    }
}

#[test]
fn adam_best_params_fitness_invariant_holds_across_seeds() {
    // Adam updates best_params/best_fitness atomically from the same evaluation,
    // so this should pass — asserted here so all four optimizers are covered on
    // the rough Multimodal surface where a desynced champion would diverge.
    for seed in 0..20 {
        let outcome = AlgorithmAdam
            .optimize(AlgorithmAdamArgs {
                oracle: Box::new(Multimodal { target: 1.0 }),
                gradient_oracle: Box::new(MultimodalGradient { target: 1.0 }),
                max_iters: 120,
                learning_rate: 0.05,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                bounds: (0.0, 2.0),
                dimensions: 4,
                tolerance: 0.0,
                patience: 3,
                seed: Some(seed),
            })
            .expect("valid Adam args optimize successfully");
        assert_reported_fitness_matches_params(
            "adam/multimodal",
            seed,
            &outcome,
            &Multimodal { target: 1.0 },
        );
    }
}
