//! Behavioural tests for the pure-Rust optimizers.
//!
//! These run in a plain `cargo test` binary with **no Python interpreter**: the
//! crate has no PyO3 dependency at all, so any accidental attempt to touch a
//! GIL could not even compile. Passing tests are therefore structural proof of
//! Python-freedom (the same guarantee `polypus`'s `native_circuit_path.rs`
//! documents), and they exercise the optimizers in complete isolation from the
//! Python extension — evidence a future pure-Rust consumer can reuse them.

use polypus_optimizers::{
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, AlgorithmPSO,
    AlgorithmPSOArgs, AlgorithmQNG, AlgorithmQNGArgs, EvaluationOracle, OptimizationOutcome,
    Optimizer, VarianceOracle,
};

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

/// Constant diagonal QFIM, the simplest [`VarianceOracle`] for exercising QNG.
struct ConstVariance(f64);

impl VarianceOracle for ConstVariance {
    fn variance(&self, _theta: &[f64], _param_index: usize) -> f64 {
        self.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convergence on a known analytic optimum (deterministic via a fixed seed)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn de_converges_to_known_optimum() {
    let outcome = AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
        oracle: Box::new(Quadratic { target: 1.0 }),
        population_size: 40,
        generations: 300,
        dimensions: 3,
        tolerance: 1e-9,
        seed: Some(42),
    });

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
    let outcome = AlgorithmPSO.optimize(AlgorithmPSOArgs {
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
    });

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
    let outcome = AlgorithmQNG.optimize(AlgorithmQNGArgs {
        oracle: Box::new(Quadratic { target: 1.0 }),
        max_iters: 200,
        learning_rate: 0.1,
        finite_difference_step: 0.1,
        bounds: (0.0, 2.0),
        dimensions: 3,
        variance_oracle: Box::new(ConstVariance(1.0)),
        tikhonov_reg: 0.05,
        seed: Some(42),
    });

    assert!(
        outcome.best_fitness > -1e-3,
        "fitness = {}",
        outcome.best_fitness
    );
    for x in &outcome.best_params {
        assert!((x - 1.0).abs() < 0.05, "param off target: {x}");
    }
    // QNG runs a fixed iteration budget with no early-stopping test.
    assert_eq!(outcome.iterations_run, 200);
    assert!(!outcome.converged);
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism: a fixed seed reproduces the trajectory exactly
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn de_is_deterministic_for_a_fixed_seed() {
    let make = || {
        AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
            oracle: Box::new(Quadratic { target: 0.7 }),
            population_size: 20,
            generations: 50,
            dimensions: 4,
            tolerance: 1e-9,
            seed: Some(123),
        })
    };
    // OptimizationOutcome derives PartialEq — same seed ⇒ identical outcome.
    assert_eq!(make(), make());
}

#[test]
fn pso_is_deterministic_for_a_fixed_seed() {
    let make = || {
        AlgorithmPSO.optimize(AlgorithmPSOArgs {
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
    };
    assert_eq!(make(), make());
}

// ─────────────────────────────────────────────────────────────────────────────
// Early-stopping bookkeeping (iterations_run / converged) is reproducible
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn de_early_stops_and_reports_it() {
    // A loose tolerance makes the `std < tolerance * mean` test fire before the
    // generation budget is exhausted.
    let make = || {
        AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            population_size: 30,
            generations: 500,
            dimensions: 3,
            tolerance: 0.5,
            seed: Some(7),
        })
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
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn de_handles_zero_dimensions() {
    let outcome = AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
        oracle: Box::new(Quadratic { target: 1.0 }),
        population_size: 10,
        generations: 5,
        dimensions: 0,
        tolerance: 1e-9,
        seed: Some(1),
    });
    assert!(outcome.best_params.is_empty());
}

#[test]
fn de_handles_minimum_population() {
    // DE needs population_size >= 4 to sample 3 distinct other members.
    let outcome = AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
        oracle: Box::new(Quadratic { target: 1.0 }),
        population_size: 4,
        generations: 10,
        dimensions: 2,
        tolerance: 1e-9,
        seed: Some(1),
    });
    assert_eq!(outcome.best_params.len(), 2);
}

#[test]
fn de_nan_fitness_does_not_panic() {
    let outcome = AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
        oracle: Box::new(NanOracle),
        population_size: 6,
        generations: 5,
        dimensions: 2,
        tolerance: 1e-9,
        seed: Some(1),
    });
    assert_eq!(outcome.best_params.len(), 2);
    assert!(outcome.best_fitness.is_nan());
}

#[test]
fn pso_nan_fitness_does_not_panic() {
    let outcome = AlgorithmPSO.optimize(AlgorithmPSOArgs {
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
    });
    assert_eq!(outcome.best_params.len(), 2);
}

#[test]
fn qng_tikhonov_avoids_division_blowup_when_qfim_is_zero() {
    // A zero variance would make the raw QFIM diagonal 0; the Tikhonov term
    // keeps the denominator at 0.05 so the update stays finite instead of
    // producing inf/NaN from a divide-by-zero.
    let outcome = AlgorithmQNG.optimize(AlgorithmQNGArgs {
        oracle: Box::new(Quadratic { target: 1.0 }),
        max_iters: 1,
        learning_rate: 0.1,
        finite_difference_step: 0.1,
        bounds: (0.0, 2.0),
        dimensions: 2,
        variance_oracle: Box::new(ConstVariance(0.0)),
        tikhonov_reg: 0.05,
        seed: Some(1),
    });
    for x in &outcome.best_params {
        assert!(x.is_finite(), "non-finite parameter: {x}");
    }
}

#[test]
#[should_panic]
fn pso_rejects_empty_bounds() {
    // Precondition: bounds must be a non-empty interval (lb < ub). An empty
    // range panics inside the uniform sampler — documented, not silently wrong.
    let _ = AlgorithmPSO.optimize(AlgorithmPSOArgs {
        oracle: Box::new(Quadratic { target: 1.0 }),
        population_size: 10,
        generations: 5,
        dimensions: 2,
        bounds: (1.0, 1.0),
        inertia_weight: 0.5,
        cognitive_weight: 1.0,
        social_weight: 1.0,
        tolerance: 1e-9,
        seed: Some(1),
    });
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
        let quad = AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            population_size: 30,
            generations: 120,
            dimensions: 4,
            tolerance: 1e-9,
            seed: Some(seed),
        });
        assert_reported_fitness_matches_params(
            "de/quadratic",
            seed,
            &quad,
            &Quadratic { target: 1.0 },
        );

        let multi = AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
            oracle: Box::new(Multimodal { target: 1.0 }),
            population_size: 30,
            generations: 120,
            dimensions: 4,
            tolerance: 1e-9,
            seed: Some(seed),
        });
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
        let outcome = AlgorithmPSO.optimize(AlgorithmPSOArgs {
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
        });
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
        let outcome = AlgorithmQNG.optimize(AlgorithmQNGArgs {
            oracle: Box::new(Multimodal { target: 1.0 }),
            max_iters: 120,
            learning_rate: 0.1,
            finite_difference_step: 0.1,
            bounds: (0.0, 2.0),
            dimensions: 4,
            variance_oracle: Box::new(ConstVariance(1.0)),
            tikhonov_reg: 0.05,
            seed: Some(seed),
        });
        assert_reported_fitness_matches_params(
            "qng/multimodal",
            seed,
            &outcome,
            &Multimodal { target: 1.0 },
        );
    }
}
