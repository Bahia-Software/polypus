//! Behavioural tests for [`AlgorithmCmaEs`], the pure-Rust CMA-ES.
//!
//! Like `optimizers.rs`, these run without a Python interpreter. Every oracle
//! here is a pure function of the candidate, and most record the batches they
//! are handed, so a test can check what the optimizer actually sent: the batch
//! sizes (IPOP doubling), the wrapping of every candidate into `[0, 2π)`, and
//! that the reported best is the best candidate the oracle ever scored.

use polypus_optimizers::{
    AlgorithmCmaEs, AlgorithmCmaEsArgs, CmaEsBatchSchedule, EvaluationOracle, OptimizationOutcome,
    Optimizer, OptimizerError,
};
use std::f64::consts::{PI, TAU};
use std::sync::{Arc, Mutex};

// ─────────────────────────────────────────────────────────────────────────────
// Objectives (all maximised) and a recording oracle
// ─────────────────────────────────────────────────────────────────────────────

type Fitness = fn(&[f64]) -> f64;

/// `−Σ (xᵢ − π)²`: unimodal, maximum 0 at `xᵢ = π`.
fn sphere(x: &[f64]) -> f64 {
    -x.iter().map(|xi| (xi - PI).powi(2)).sum::<f64>()
}

/// `−Σ 10^(6i/(d−1))·(xᵢ − π)²`: the ellipsoid, Hessian condition number 1e6.
fn ellipsoid(x: &[f64]) -> f64 {
    let d = x.len();
    -x.iter()
        .enumerate()
        .map(|(i, xi)| 10f64.powf(6.0 * i as f64 / (d - 1) as f64) * (xi - PI).powi(2))
        .sum::<f64>()
}

/// Rastrigin centred at π, `−Σ (yᵢ² − 10·cos(2π·yᵢ) + 10)` with `yᵢ = xᵢ − π`:
/// maximum 0 at `xᵢ = π`, and local maxima near every integer `yᵢ` in
/// `[−3, 3]`, i.e. 7^d of them on the circle.
fn rastrigin(x: &[f64]) -> f64 {
    -x.iter()
        .map(|xi| {
            let y = xi - PI;
            y * y - 10.0 * (TAU * y).cos() + 10.0
        })
        .sum::<f64>()
}

/// `Σ cos(xᵢ − 0.05)`: smooth and periodic, maximum `d` at `xᵢ = 0.05`, right
/// next to the `0`/`2π` seam.
fn cosines_near_seam(x: &[f64]) -> f64 {
    x.iter().map(|xi| (xi - 0.05).cos()).sum()
}

/// A rough periodic landscape with several local maxima per coordinate.
fn rough(x: &[f64]) -> f64 {
    x.iter()
        .map(|xi| (xi - 1.0).cos() + 0.6 * (3.0 * xi + 0.3).cos() + 0.3 * (7.0 * xi).sin())
        .sum()
}

/// One `evaluate_batch` call: the candidates it was handed and the values it
/// returned.
#[derive(Debug, Clone, PartialEq)]
struct Batch {
    candidates: Vec<Vec<f64>>,
    values: Vec<f64>,
}

type Log = Arc<Mutex<Vec<Batch>>>;

/// Oracle evaluating `f` and recording every batch in `log`.
struct Recorder {
    f: Fitness,
    log: Log,
}

impl EvaluationOracle for Recorder {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let values: Vec<f64> = candidates.iter().map(|c| (self.f)(c)).collect();
        self.log.lock().expect("log lock").push(Batch {
            candidates: candidates.to_vec(),
            values: values.clone(),
        });
        values
    }
}

/// Typical arguments for a `dimensions`-parameter run on `f`, recording into
/// a fresh log.
fn setup(f: Fitness, dimensions: u32, generations: u32, seed: u64) -> (AlgorithmCmaEsArgs, Log) {
    let log = Log::default();
    let args = AlgorithmCmaEsArgs {
        oracle: Box::new(Recorder {
            f,
            log: log.clone(),
        }),
        population_size: None,
        generations,
        dimensions,
        sigma0: AlgorithmCmaEs::DEFAULT_SIGMA0,
        tolerance: 1e-12,
        patience: 30,
        restarts: 0,
        seed: Some(seed),
    };
    (args, log)
}

fn run(args: AlgorithmCmaEsArgs) -> OptimizationOutcome {
    AlgorithmCmaEs
        .optimize(args)
        .expect("valid CMA-ES args optimize successfully")
}

fn batch_sizes(log: &Log) -> Vec<usize> {
    log.lock()
        .expect("log lock")
        .iter()
        .map(|b| b.candidates.len())
        .collect()
}

/// The batch sizes of consecutive runs: `[(size, generations), …]`.
fn runs(sizes: &[usize]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &size in sizes {
        match runs.last_mut() {
            Some((s, count)) if *s == size => *count += 1,
            _ => runs.push((size, 1)),
        }
    }
    runs
}

/// The C-5 postconditions every outcome must satisfy, checked against what
/// the oracle actually saw:
///
/// - `best_params` is a candidate the oracle scored (wrapped into `[0, 2π)`),
///   and re-evaluating it reproduces `best_fitness` bit for bit;
/// - no scored candidate beats it (it is the best over every restart);
/// - `fitness_history` has one entry per generation, never decreases, ends at
///   `best_fitness`, and entry `g` is the best value of the first `g + 1`
///   batches.
fn assert_c5(outcome: &OptimizationOutcome, f: Fitness, log: &Log) {
    let log = log.lock().expect("log lock");
    assert_eq!(
        outcome.iterations_run,
        log.len(),
        "one batch per generation"
    );
    assert_eq!(outcome.fitness_history.len(), outcome.iterations_run);
    assert_eq!(
        outcome.fitness_history.last().map(|x| x.to_bits()),
        Some(outcome.best_fitness.to_bits())
    );
    assert_eq!(
        f(&outcome.best_params).to_bits(),
        outcome.best_fitness.to_bits()
    );
    assert!(outcome.best_params.iter().all(|x| (0.0..TAU).contains(x)));
    assert!(
        log.iter()
            .any(|b| b.candidates.contains(&outcome.best_params)),
        "best_params was never handed to the oracle"
    );
    let mut best_so_far = f64::NEG_INFINITY;
    for (g, (batch, recorded)) in log.iter().zip(&outcome.fitness_history).enumerate() {
        for &v in &batch.values {
            best_so_far = best_so_far.max(v);
        }
        assert_eq!(*recorded, best_so_far, "fitness_history[{g}]");
    }
    assert_eq!(outcome.best_fitness, best_so_far);
}

// ─────────────────────────────────────────────────────────────────────────────
// Convergence
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn converges_on_the_sphere() {
    let (args, log) = setup(sphere, 5, 1000, 1);
    let outcome = run(args);
    assert!(
        outcome.best_fitness > -1e-12,
        "fitness = {:e}",
        outcome.best_fitness
    );
    for x in &outcome.best_params {
        assert!((x - PI).abs() < 1e-6, "param off target: {x}");
    }
    // It stops on its own, long before the budget.
    assert!(outcome.converged);
    assert!(outcome.iterations_run < 1000, "{}", outcome.iterations_run);
    // Default population for 5 parameters: 4 + ⌊3 ln 5⌋ = 8, every batch.
    assert!(batch_sizes(&log).iter().all(|&s| s == 8));
    assert_c5(&outcome, sphere, &log);
}

#[test]
fn solves_an_ill_conditioned_ellipsoid() {
    // Hessian condition 1e6: an isotropic search would need ~1000 times as many
    // generations as on the sphere. CMA-ES learns the covariance (the unit test
    // `covariance_adaptation_learns_the_ellipsoid` checks cond(C) ≈ 1e6) and
    // gets there in a few times the sphere's.
    let (args, log) = setup(ellipsoid, 8, 3000, 2);
    let ellipsoid_run = run(args);
    assert!(
        ellipsoid_run.best_fitness > -1e-10,
        "fitness = {:e}",
        ellipsoid_run.best_fitness
    );
    assert_c5(&ellipsoid_run, ellipsoid, &log);

    let first_below = |outcome: &OptimizationOutcome| {
        outcome
            .fitness_history
            .iter()
            .position(|&f| f > -1e-8)
            .expect("reaches 1e-8")
    };
    let (args, _) = setup(sphere, 8, 3000, 2);
    let sphere_run = run(args);
    let (e, s) = (first_below(&ellipsoid_run), first_below(&sphere_run));
    assert!(e < 10 * s, "ellipsoid {e} generations vs sphere {s}");
}

#[test]
fn ipop_restarts_find_the_global_optimum_of_rastrigin() {
    // Without restarts, the default population (7 for 3 parameters) converges
    // to a local optimum of Rastrigin; with IPOP the population doubles at each
    // restart until a run is large enough to find the global one (here the
    // run of 14 to 224 candidates, depending on the seed).
    for seed in 0..4 {
        let (single, _) = setup(rastrigin, 3, 3000, seed);
        let single = run(AlgorithmCmaEsArgs {
            tolerance: 1e-9,
            ..single
        });
        assert!(
            single.best_fitness < -0.5,
            "seed {seed}: {}",
            single.best_fitness
        );

        let (ipop, log) = setup(rastrigin, 3, 3000, seed);
        let ipop = run(AlgorithmCmaEsArgs {
            tolerance: 1e-9,
            restarts: 5,
            ..ipop
        });
        assert!(
            ipop.best_fitness > -1e-9,
            "seed {seed}: {}",
            ipop.best_fitness
        );
        for x in &ipop.best_params {
            assert!((x - PI).abs() < 1e-4, "seed {seed}: param off target: {x}");
        }
        assert!(ipop.converged, "seed {seed}: the last run ends on its own");
        assert_c5(&ipop, rastrigin, &log);

        // Six runs, whose batch sizes double: 7, 14, …, 224.
        let sizes: Vec<usize> = runs(&batch_sizes(&log)).iter().map(|r| r.0).collect();
        assert_eq!(sizes, vec![7, 14, 28, 56, 112, 224], "seed {seed}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Restarts and stopping
// ─────────────────────────────────────────────────────────────────────────────

/// A flat landscape: every candidate scores 0.
fn flat(_: &[f64]) -> f64 {
    0.0
}

#[test]
fn stagnation_restarts_double_the_population_then_stop() {
    // On a flat landscape each run stagnates as soon as it has a full window:
    // patience 2 → 3 generations per run. With 3 restarts: 4 runs of 6, 12,
    // 24 and 48 candidates (6 = 4 + ⌊3 ln 2⌋), then the stop.
    let (args, log) = setup(flat, 2, 100, 3);
    let outcome = run(AlgorithmCmaEsArgs {
        tolerance: 1e-9,
        patience: 2,
        restarts: 3,
        ..args
    });
    assert_eq!(
        runs(&batch_sizes(&log)),
        vec![(6, 3), (12, 3), (24, 3), (48, 3)]
    );
    assert!(outcome.converged);
    assert_eq!(outcome.iterations_run, 12);
    assert_c5(&outcome, flat, &log);
}

#[test]
fn without_restarts_stagnation_stops_like_de() {
    // C-5's rule: stop at the first generation g >= patience whose best
    // improved by less than `tolerance` since generation g - patience.
    let (args, log) = setup(flat, 3, 100, 4);
    let outcome = run(AlgorithmCmaEsArgs {
        tolerance: 1e-9,
        patience: 5,
        ..args
    });
    assert!(outcome.converged);
    assert_eq!(outcome.iterations_run, 6);
    assert_c5(&outcome, flat, &log);

    // A non-positive tolerance never stops on stagnation: the budget runs out.
    let (args, log) = setup(flat, 3, 40, 4);
    let outcome = run(AlgorithmCmaEsArgs {
        tolerance: 0.0,
        patience: 5,
        ..args
    });
    assert!(!outcome.converged);
    assert_eq!(outcome.iterations_run, 40);
    assert_eq!(batch_sizes(&log).len(), 40);
}

#[test]
fn a_collapsed_distribution_restarts_each_run() {
    // With the stagnation stop off (tolerance 0), each run ends when TolX
    // fires: σ·max√C_ii < 1e-12·σ₀ on the sphere's optimum. Two restarts give
    // three runs of 6, 12 and 24 candidates, each converging, and then the
    // stop, well within the budget.
    let (args, log) = setup(sphere, 2, 5000, 5);
    let outcome = run(AlgorithmCmaEsArgs {
        tolerance: 0.0,
        restarts: 2,
        ..args
    });
    assert!(outcome.converged);
    assert!(outcome.iterations_run < 5000, "{}", outcome.iterations_run);
    let runs = runs(&batch_sizes(&log));
    assert_eq!(
        runs.iter().map(|r| r.0).collect::<Vec<_>>(),
        vec![6, 12, 24]
    );
    // Collapsing by 12 orders of magnitude takes many generations per run.
    assert!(runs.iter().all(|r| r.1 > 50), "{runs:?}");
    assert!(outcome.best_fitness > -1e-20, "{:e}", outcome.best_fitness);
    // Each run converged on its own: its last batch is at the optimum.
    let log = log.lock().expect("log lock");
    let mut end = 0;
    for (_, generations) in &runs {
        end += generations;
        let last = &log[end - 1];
        assert!(
            last.values.iter().all(|&v| v > -1e-20),
            "run ending at {end}"
        );
    }
}

#[test]
fn batch_schedule_bounds_every_batch() {
    // The public schedule (what a walltime projection would use) bounds the
    // batches the optimizer actually sends, and is exact without restarts.
    for (f, d, restarts, patience) in [
        (flat as Fitness, 2, 3, 2),
        (rastrigin as Fitness, 4, 4, 30),
        (sphere as Fitness, 3, 0, 30),
    ] {
        let (args, log) = setup(f, d, 400, 6);
        run(AlgorithmCmaEsArgs {
            tolerance: 1e-9,
            patience,
            restarts,
            ..args
        });
        let schedule = CmaEsBatchSchedule::new(d, None, 400, restarts);
        let sizes = batch_sizes(&log);
        assert!(sizes.len() as u32 <= schedule.generations());
        assert_eq!(sizes[0] as u32, schedule.initial_population());
        for (g, &size) in sizes.iter().enumerate() {
            assert!(
                size as u64 <= schedule.max_batch_size(g as u32),
                "generation {g}"
            );
        }
        let total: u64 = sizes.iter().map(|&s| s as u64).sum();
        assert!(total <= schedule.max_evaluations());
        assert!(sizes.iter().all(|&s| s as u64 <= schedule.max_population()));
        if restarts == 0 {
            assert!(sizes
                .iter()
                .all(|&s| s as u32 == schedule.initial_population()));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism, invariance and the C-5 contract
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn seeded_runs_are_reproducible() {
    let make = |seed: u64| {
        let (args, log) = setup(rough, 4, 150, seed);
        let outcome = run(AlgorithmCmaEsArgs {
            tolerance: 1e-6,
            patience: 15,
            restarts: 2,
            ..args
        });
        let batches = log.lock().expect("log lock").clone();
        (outcome, batches)
    };
    let (a, a_batches) = make(42);
    let (b, b_batches) = make(42);
    assert_eq!(a, b);
    assert_eq!(a_batches, b_batches, "same candidates, batch by batch");
    let (c, c_batches) = make(43);
    assert_ne!(a_batches, c_batches);
    assert_ne!(a.best_params, c.best_params);
}

#[test]
fn best_params_fitness_invariant_holds_across_seeds() {
    // Contract C-5 on a rough landscape, with restarts (so the best can come
    // from any run) and without, over 25 seeds each.
    for seed in 0..25 {
        for restarts in [0, 2] {
            let (args, log) = setup(rough, 5, 200, seed);
            let outcome = run(AlgorithmCmaEsArgs {
                tolerance: 1e-6,
                patience: 15,
                restarts,
                ..args
            });
            assert_c5(&outcome, rough, &log);
        }
    }
}

/// Oracle scoring `transform(f(x))`, recording the candidates.
struct Transformed {
    f: Fitness,
    transform: fn(f64) -> f64,
    log: Log,
}

impl EvaluationOracle for Transformed {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let values: Vec<f64> = candidates
            .iter()
            .map(|c| (self.transform)((self.f)(c)))
            .collect();
        self.log.lock().expect("log lock").push(Batch {
            candidates: candidates.to_vec(),
            values: values.clone(),
        });
        values
    }
}

/// The candidates of a run on `transform(f)`.
fn trajectory(
    f: Fitness,
    transform: fn(f64) -> f64,
    generations: u32,
    restarts: u32,
) -> (OptimizationOutcome, Vec<Vec<Vec<f64>>>) {
    let log = Log::default();
    let outcome = run(AlgorithmCmaEsArgs {
        oracle: Box::new(Transformed {
            f,
            transform,
            log: log.clone(),
        }),
        population_size: None,
        generations,
        dimensions: 3,
        sigma0: AlgorithmCmaEs::DEFAULT_SIGMA0,
        // The stagnation stop compares fitness values, so it is off here; the
        // rest of CMA-ES sees only the ranking.
        tolerance: 0.0,
        patience: 10,
        restarts,
        seed: Some(9),
    });
    let candidates = log
        .lock()
        .expect("log lock")
        .iter()
        .map(|b| b.candidates.clone())
        .collect();
    (outcome, candidates)
}

#[test]
fn strictly_increasing_transformations_leave_the_trajectory_unchanged() {
    // exp(f): 80 generations keep the fitness differences far above exp's
    // rounding, so the ranking of every batch is the same as f's.
    let (plain, plain_candidates) = trajectory(rough, |v| v, 80, 0);
    let (exp, exp_candidates) = trajectory(rough, f64::exp, 80, 0);
    assert_eq!(plain_candidates, exp_candidates);
    assert_eq!(plain.best_params, exp.best_params);
    assert_eq!(exp.best_fitness, plain.best_fitness.exp());
    assert_eq!(plain.iterations_run, exp.iterations_run);

    // v³ keeps the order of values down to 1e-24 (the sphere at TolX), so the
    // runs match through two restarts triggered by the collapsed distribution.
    let (plain, plain_candidates) = trajectory(sphere, |v| v, 4000, 2);
    let (cubed, cubed_candidates) = trajectory(sphere, |v| v * v * v, 4000, 2);
    assert!(plain.converged, "the three runs end on TolX");
    assert_eq!(
        runs(&plain_candidates.iter().map(Vec::len).collect::<Vec<_>>()).len(),
        3
    );
    assert_eq!(plain_candidates, cubed_candidates);
    assert_eq!(plain.best_params, cubed.best_params);
    assert_eq!(plain.iterations_run, cubed.iterations_run);
}

#[test]
fn candidates_are_wrapped_into_the_circle() {
    // The optimum (0.05) sits next to the 0/2π seam, so the unwrapped search
    // distribution straddles it; the oracle must still only ever see angles in
    // [0, 2π), from both sides of the seam, and the run converges across it.
    for seed in 0..5 {
        let (args, log) = setup(cosines_near_seam, 3, 400, seed);
        let outcome = run(args);
        let log = log.lock().expect("log lock");
        let all: Vec<f64> = log
            .iter()
            .flat_map(|b| b.candidates.iter().flatten().copied())
            .collect();
        assert!(all.iter().all(|x| (0.0..TAU).contains(x)), "seed {seed}");
        assert!(all.iter().any(|&x| x < 0.05) && all.iter().any(|&x| x > TAU - 0.05));
        assert!(
            outcome.best_fitness > 3.0 - 1e-10,
            "seed {seed}: {}",
            outcome.best_fitness
        );
        for x in &outcome.best_params {
            let distance = (x - 0.05).abs().min(TAU - (x - 0.05).abs());
            assert!(distance < 1e-4, "seed {seed}: {x}");
        }
    }
}

#[test]
fn the_widest_step_still_hands_the_oracle_wrapped_angles() {
    // σ₀ = 2π spreads the first samples over several turns: every candidate
    // the oracle sees is still in [0, 2π), and the result is a scored one.
    for seed in 0..5 {
        let (args, log) = setup(rough, 6, 200, seed);
        let outcome = run(AlgorithmCmaEsArgs {
            sigma0: AlgorithmCmaEs::MAX_SIGMA0,
            restarts: 1,
            ..args
        });
        let guard = log.lock().expect("log lock");
        assert!(guard
            .iter()
            .flat_map(|b| b.candidates.iter().flatten())
            .all(|x| (0.0..TAU).contains(x)));
        drop(guard);
        assert_c5(&outcome, rough, &log);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Robustness: non-finite fitness, oracle length, configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Oracle returning a fixed value for every candidate.
struct Constant(f64);

impl EvaluationOracle for Constant {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        vec![self.0; candidates.len()]
    }
}

/// Oracle scoring the sphere but returning `NaN` for every third candidate and
/// `-∞` for every fifth.
struct Holes;

impl EvaluationOracle for Holes {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates
            .iter()
            .enumerate()
            .map(|(i, c)| match i {
                i if i % 3 == 0 => f64::NAN,
                i if i % 5 == 0 => f64::NEG_INFINITY,
                _ => sphere(c),
            })
            .collect()
    }
}

fn constant_args(value: f64, restarts: u32) -> AlgorithmCmaEsArgs {
    AlgorithmCmaEsArgs {
        oracle: Box::new(Constant(value)),
        population_size: Some(5),
        generations: 300,
        dimensions: 3,
        sigma0: 1.0,
        tolerance: 1e-9,
        patience: 10,
        restarts,
        seed: Some(11),
    }
}

#[test]
fn nan_fitness_does_not_panic() {
    // All NaN: the ranking falls back to the sampling order, the stagnation
    // test never fires (NaN − NaN is not < tolerance) and the run ends on the
    // budget or a degenerate distribution, returning an evaluated candidate.
    for restarts in [0, 3] {
        let outcome = run(constant_args(f64::NAN, restarts));
        assert!(outcome.best_fitness.is_nan());
        assert_eq!(outcome.best_params.len(), 3);
        assert!(outcome.best_params.iter().all(|x| (0.0..TAU).contains(x)));
        assert_eq!(outcome.fitness_history.len(), outcome.iterations_run);
        assert!(outcome.iterations_run >= 1 && outcome.iterations_run <= 300);
    }
}

#[test]
fn infinite_fitness_does_not_panic() {
    for value in [f64::INFINITY, f64::NEG_INFINITY] {
        let outcome = run(constant_args(value, 1));
        assert_eq!(outcome.best_fitness, value);
        assert_eq!(outcome.fitness_history.last(), Some(&value));
        assert!(outcome.best_params.iter().all(|x| (0.0..TAU).contains(x)));
    }
}

#[test]
fn nan_and_infinite_candidates_rank_below_finite_ones() {
    let outcome = run(AlgorithmCmaEsArgs {
        oracle: Box::new(Holes),
        population_size: Some(10),
        generations: 400,
        dimensions: 3,
        sigma0: 1.0,
        tolerance: 1e-12,
        patience: 30,
        restarts: 0,
        seed: Some(12),
    });
    // The NaN and -∞ candidates never become the best, and the finite ones
    // still drive the search to the sphere's optimum.
    assert!(outcome.best_fitness.is_finite());
    assert_eq!(sphere(&outcome.best_params), outcome.best_fitness);
    assert!(outcome.best_fitness > -1e-10, "{:e}", outcome.best_fitness);
}

/// Oracle that answers correctly for its first `good_batches` calls, then
/// returns one value fewer than it was handed.
struct ShortAfter {
    good_batches: usize,
    calls: Mutex<usize>,
}

impl EvaluationOracle for ShortAfter {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let mut calls = self.calls.lock().expect("calls lock");
        *calls += 1;
        let n = if *calls > self.good_batches {
            candidates.len() - 1
        } else {
            candidates.len()
        };
        vec![0.0; n]
    }
}

#[test]
fn short_oracle_returns_an_error_not_a_panic() {
    let short_args = |good_batches: usize| AlgorithmCmaEsArgs {
        oracle: Box::new(ShortAfter {
            good_batches,
            calls: Mutex::new(0),
        }),
        population_size: None,
        generations: 100,
        dimensions: 4,
        sigma0: 1.0,
        tolerance: 1e-9,
        patience: 2,
        restarts: 2,
        seed: Some(13),
    };
    // First batch: 4 + ⌊3 ln 4⌋ = 8 candidates.
    assert_eq!(
        AlgorithmCmaEs.optimize(short_args(0)),
        Err(OptimizerError::OracleLengthMismatch {
            expected: 8,
            got: 7
        })
    );
    // The constant answers stagnate each run after 3 generations (patience
    // 2), so the 4th batch is the first restart's, of 16 candidates: the
    // length is checked on every batch, the larger ones too.
    assert_eq!(
        AlgorithmCmaEs.optimize(short_args(3)),
        Err(OptimizerError::OracleLengthMismatch {
            expected: 16,
            got: 15
        })
    );
}

/// Oracle that counts its calls (and returns zeros).
struct Counting(Arc<Mutex<usize>>);

impl EvaluationOracle for Counting {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        *self.0.lock().expect("count lock") += 1;
        vec![0.0; candidates.len()]
    }
}

#[test]
fn invalid_configuration_is_rejected_before_any_oracle_call() {
    let valid = |calls: &Arc<Mutex<usize>>| AlgorithmCmaEsArgs {
        oracle: Box::new(Counting(calls.clone())),
        population_size: None,
        generations: 10,
        dimensions: 3,
        sigma0: 1.0,
        tolerance: 1e-9,
        patience: 5,
        restarts: 0,
        seed: Some(14),
    };
    type Tweak = fn(AlgorithmCmaEsArgs) -> AlgorithmCmaEsArgs;
    let cases: [(&str, Tweak); 12] = [
        ("dimensions", |a| AlgorithmCmaEsArgs { dimensions: 0, ..a }),
        ("generations", |a| AlgorithmCmaEsArgs {
            generations: 0,
            ..a
        }),
        ("sigma0", |a| AlgorithmCmaEsArgs { sigma0: 0.0, ..a }),
        ("sigma0", |a| AlgorithmCmaEsArgs { sigma0: -0.5, ..a }),
        ("sigma0", |a| AlgorithmCmaEsArgs {
            sigma0: f64::NAN,
            ..a
        }),
        ("sigma0", |a| AlgorithmCmaEsArgs {
            sigma0: f64::INFINITY,
            ..a
        }),
        ("sigma0", |a| AlgorithmCmaEsArgs {
            sigma0: TAU * 1.000_001,
            ..a
        }),
        ("population_size", |a| AlgorithmCmaEsArgs {
            population_size: Some(0),
            ..a
        }),
        ("population_size", |a| AlgorithmCmaEsArgs {
            population_size: Some(1),
            ..a
        }),
        // 7 · 2^30 > u32::MAX
        ("restarts", |a| AlgorithmCmaEsArgs { restarts: 30, ..a }),
        ("restarts", |a| AlgorithmCmaEsArgs {
            restarts: u32::MAX,
            ..a
        }),
        ("restarts", |a| AlgorithmCmaEsArgs {
            population_size: Some(u32::MAX),
            restarts: 1,
            ..a
        }),
    ];
    for (parameter, tweak) in cases {
        let calls = Arc::new(Mutex::new(0));
        let result = AlgorithmCmaEs.optimize(tweak(valid(&calls)));
        match result {
            Err(OptimizerError::InvalidConfig {
                parameter: p,
                reason,
            }) => {
                assert_eq!(p, parameter, "{reason}");
                assert!(!reason.is_empty());
            }
            other => panic!("{parameter}: expected InvalidConfig, got {other:?}"),
        }
        assert_eq!(
            *calls.lock().expect("count lock"),
            0,
            "{parameter}: oracle called"
        );
    }

    // The boundaries are valid: 2 candidates, one full turn, 1 dimension,
    // 1 generation, and the most restarts that keep λ within u32
    // (7 · 2^29 < 2^32).
    let ok: [Tweak; 5] = [
        |a| AlgorithmCmaEsArgs {
            population_size: Some(2),
            ..a
        },
        |a| AlgorithmCmaEsArgs { sigma0: TAU, ..a },
        |a| AlgorithmCmaEsArgs { dimensions: 1, ..a },
        |a| AlgorithmCmaEsArgs {
            generations: 1,
            ..a
        },
        |a| AlgorithmCmaEsArgs {
            restarts: 29,
            generations: 3,
            ..a
        },
    ];
    for tweak in ok {
        let calls = Arc::new(Mutex::new(0));
        let outcome = AlgorithmCmaEs
            .optimize(tweak(valid(&calls)))
            .expect("boundary configuration is valid");
        assert!(outcome.iterations_run >= 1);
        assert_eq!(*calls.lock().expect("count lock"), outcome.iterations_run);
    }
}

#[test]
fn error_messages_name_the_rule() {
    let message = |a: AlgorithmCmaEsArgs| match AlgorithmCmaEs.optimize(a) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected an error"),
    };
    let base = || AlgorithmCmaEsArgs {
        oracle: Box::new(Constant(0.0)),
        population_size: None,
        generations: 10,
        dimensions: 2,
        sigma0: 1.0,
        tolerance: 0.0,
        patience: 5,
        restarts: 0,
        seed: None,
    };
    assert_eq!(
        message(AlgorithmCmaEsArgs {
            sigma0: 7.0,
            ..base()
        }),
        "invalid sigma0: must be finite and in (0, 2π] (radians), got 7"
    );
    assert!(message(AlgorithmCmaEsArgs {
        population_size: Some(1),
        ..base()
    })
    .starts_with("invalid population_size: must be at least 2"));
    assert!(message(AlgorithmCmaEsArgs {
        dimensions: 0,
        ..base()
    })
    .starts_with("invalid dimensions: must be at least 1"));
}

#[test]
fn the_2d_sphere_converges_from_seeds_that_used_to_stall() {
    // #382: of seeds 0..3000, these grew σ past the 2π period, sampled the
    // circle uniformly and stopped on stagnation 32–65 generations in, far
    // from the optimum. The cap on the coordinate spread keeps them on track
    // (all 3000 converge; checked outside CI, where the full range is slow).
    const STALLED: [u64; 43] = [
        7, 427, 436, 437, 452, 489, 560, 591, 605, 650, 769, 785, 798, 839, 869, 899, 1052, 1139,
        1159, 1352, 1399, 1525, 1551, 1622, 1840, 1913, 1980, 2239, 2259, 2274, 2314, 2376, 2394,
        2438, 2452, 2592, 2656, 2659, 2830, 2885, 2959, 2962, 2998,
    ];
    for seed in STALLED {
        let outcome = run(setup(sphere, 2, 300, seed).0);
        assert!(
            outcome.best_fitness > -1e-8,
            "seed {seed}: {:e} after {} generations",
            outcome.best_fitness,
            outcome.iterations_run
        );
    }
}

#[test]
fn unseeded_runs_work() {
    // `seed: None` draws from the thread RNG, like the other optimizers.
    let outcome = run(AlgorithmCmaEsArgs {
        seed: None,
        ..setup(sphere, 2, 300, 0).0
    });
    assert!(outcome.best_fitness > -1e-8, "{:e}", outcome.best_fitness);
}
