//! Behavioural tests for [`AlgorithmRotosolve`].
//!
//! Like `spsa.rs`, these run without a Python interpreter. The oracles are
//! pure functions of the candidate; the ones that record what they are handed
//! let a test rebuild every step from what the optimizer actually sent (one
//! batch per step, then the final point alone), and check that nothing was
//! sent at all when the configuration is rejected.

use polypus_optimizers::{
    AlgorithmRotosolve, AlgorithmRotosolveArgs, EvaluationOracle, OptimizationOutcome, Optimizer,
    OptimizerError,
};
use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::sync::{Arc, Mutex};

/// The maximum of [`cosine`].
const TARGET: [f64; 3] = [0.7, -1.2, 2.1];

/// `Σ cos(xᵢ − tᵢ)`: a sinusoid of each coordinate, separable, with its
/// maximum 3 at [`TARGET`] + 2πk.
fn cosine(x: &[f64]) -> f64 {
    x.iter().zip(TARGET).map(|(xi, ti)| (xi - ti).cos()).sum()
}

/// A sinusoid of each coordinate when the others are fixed, as the
/// expectation of a circuit of single-Pauli rotations is, but not separable:
/// the products couple the coordinates, so each step's optimum depends on the
/// others.
fn coupled(x: &[f64]) -> f64 {
    let (a, b, c) = (x[0], x[1], x[2]);
    a.cos() * b.cos() + 0.6 * a.sin() * b.sin() * c.cos() + 0.3 * (c + 0.4).sin() + 0.2 * b.cos()
        - 0.45 * (a - 1.1).sin() * c.cos()
}

/// The quadratic bowl with a sinusoidal ripple: not a sinusoid of any
/// coordinate, so the steps miss and the best is often not the last point.
fn rough(x: &[f64]) -> f64 {
    x.iter().map(|v| -(v - 1.0).powi(2) + (5.0 * v).sin()).sum()
}

/// Every candidate scores the same.
fn flat(_: &[f64]) -> f64 {
    0.0
}

/// The distance from `x` to `t` on the circle.
fn angular_gap(x: f64, t: f64) -> f64 {
    let gap = (x - t).rem_euclid(TAU);
    gap.min(TAU - gap)
}

/// Every batch an oracle was handed, with the values it returned.
type Log = Arc<Mutex<Vec<(Vec<Vec<f64>>, Vec<f64>)>>>;

/// Scores candidates with `fitness`, after `edit` has had its say on the
/// values of batch `index` (0 is the first step's), and logs every batch.
#[derive(Clone)]
struct Recorder {
    fitness: fn(&[f64]) -> f64,
    edit: fn(usize, &mut [f64]),
    log: Log,
}

impl Recorder {
    fn new(fitness: fn(&[f64]) -> f64) -> Self {
        Self::editing(fitness, |_, _| {})
    }

    fn editing(fitness: fn(&[f64]) -> f64, edit: fn(usize, &mut [f64])) -> Self {
        Recorder {
            fitness,
            edit,
            log: Log::default(),
        }
    }

    fn batches(&self) -> Vec<Vec<Vec<f64>>> {
        self.log
            .lock()
            .unwrap()
            .iter()
            .map(|(b, _)| b.clone())
            .collect()
    }

    /// The batches of the steps, after checking that the run ended with the
    /// final point alone.
    fn steps(&self) -> Vec<Vec<Vec<f64>>> {
        let mut batches = self.batches();
        let last = batches.pop().expect("a run sends batches");
        assert_eq!(last.len(), 1, "the last batch is the final point alone");
        batches
    }

    /// The final point, the last batch's one candidate.
    fn final_point(&self) -> Vec<f64> {
        let batches = self.batches();
        let last = batches.last().expect("a run sends batches");
        assert_eq!(last.len(), 1, "the last batch is the final point alone");
        last[0].clone()
    }

    fn values(&self) -> Vec<Vec<f64>> {
        self.log
            .lock()
            .unwrap()
            .iter()
            .map(|(_, v)| v.clone())
            .collect()
    }
}

impl EvaluationOracle for Recorder {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let mut log = self.log.lock().unwrap();
        let mut values: Vec<f64> = candidates.iter().map(|c| (self.fitness)(c)).collect();
        (self.edit)(log.len(), &mut values);
        log.push((candidates.to_vec(), values.clone()));
        values
    }
}

fn rotosolve_args(oracle: &Recorder, seed: u64) -> AlgorithmRotosolveArgs {
    AlgorithmRotosolveArgs {
        oracle: Box::new(oracle.clone()),
        sweeps: 10,
        reevaluate_every: 5,
        tolerance: 0.0,
        patience: 2,
        dimensions: TARGET.len() as u32,
        seed: Some(seed),
        initial_point: None,
    }
}

fn run(args: AlgorithmRotosolveArgs) -> OptimizationOutcome {
    AlgorithmRotosolve
        .optimize(args)
        .expect("valid Rotosolve args optimize successfully")
}

/// The point step `k` moves from, read off its batch: the head of a batch of
/// three, or, in a batch of two, `x ± π/2·e_j` with `x_j` their midpoint (to
/// within the rounding of `± π/2`). The final point's batch is the point.
fn point_of(batch: &[Vec<f64>], j: usize) -> Vec<f64> {
    match batch {
        [x] | [x, _, _] => x.clone(),
        [plus, minus] => {
            let mut x = plus.clone();
            x[j] = 0.5 * (plus[j] + minus[j]);
            x
        }
        _ => panic!("a batch has 1, 2 or 3 candidates, got {}", batch.len()),
    }
}

/// The maximum of `t ↦ fitness(x with x_j = t)` over a fine grid, and where.
fn grid_maximum(fitness: fn(&[f64]) -> f64, x: &[f64], j: usize) -> (f64, f64) {
    const GRID: usize = 20_000;
    (0..GRID)
        .map(|k| {
            let mut y = x.to_vec();
            y[j] = TAU * k as f64 / GRID as f64;
            (y[j], fitness(&y))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// The step
// ─────────────────────────────────────────────────────────────────────────────

/// On a landscape that is a sinusoid of each coordinate, every step moves its
/// coordinate to the exact maximum along it, the others fixed, whether the
/// point was measured or its value predicted by the step before: a wrong
/// value would move the coordinate elsewhere. So the fitness of the point
/// never decreases. The final point is where the last step moved.
#[test]
fn each_step_lands_on_the_maximum_along_its_coordinate() {
    for reevaluate_every in [1, 2, 5, 1000] {
        for seed in 0..5 {
            let oracle = Recorder::new(coupled);
            let mut args = rotosolve_args(&oracle, seed);
            args.sweeps = 6;
            args.reevaluate_every = reevaluate_every;
            run(args);
            // 18 steps, then the final point.
            assert_eq!(oracle.steps().len(), 18);
            let batches = oracle.batches();
            let label = format!("every {reevaluate_every} seed {seed}");
            for k in 0..batches.len() - 1 {
                let (j, next_j) = (k % 3, (k + 1) % 3);
                let x = point_of(&batches[k], j);
                let next = point_of(&batches[k + 1], next_j);
                // Only coordinate j moved, into [0, 2π).
                for i in (0..3).filter(|&i| i != j && i != next_j) {
                    assert_eq!(next[i], x[i], "{label} step {k}");
                }
                if next_j != j {
                    assert!((next[next_j] - x[next_j]).abs() < 1e-12, "{label} step {k}");
                }
                assert!((0.0..TAU).contains(&next[j]), "{label} step {k}: {next:?}");
                let (at, grid_best) = grid_maximum(coupled, &x, j);
                let mut moved = x.clone();
                moved[j] = next[j];
                assert!(
                    coupled(&moved) >= grid_best - 1e-12,
                    "{label} step {k}: {} below the grid's {grid_best}",
                    coupled(&moved)
                );
                assert!(
                    angular_gap(next[j], at) <= 2.0 * TAU / 20_000.0,
                    "{label} step {k}: {} vs the grid's {at}",
                    next[j]
                );
                assert!(coupled(&moved) >= coupled(&x) - 1e-12, "{label} step {k}");
            }
        }
    }
}

/// On a separable sinusoid one sweep moves every coordinate to its maximum,
/// the next measures the point there, and the third finds nothing better.
#[test]
fn a_separable_sinusoid_is_solved_in_one_sweep() {
    for seed in [21, 22, 23] {
        let oracle = Recorder::new(cosine);
        let mut args = rotosolve_args(&oracle, seed);
        // Every sweep starts by measuring the point.
        args.reevaluate_every = 3;
        args.tolerance = 1e-9;
        args.patience = 1;
        let outcome = run(args);
        let batches = oracle.batches();
        let label = format!("seed {seed}");
        assert!(
            outcome.best_fitness > 3.0 - 1e-12,
            "{label}: {}",
            outcome.best_fitness
        );
        for (x, t) in outcome.best_params.iter().zip(TARGET) {
            assert!(
                angular_gap(*x, t) < 1e-9,
                "{label}: {:?}",
                outcome.best_params
            );
        }
        // The point after the first sweep is the optimum, measured at the
        // head of the second.
        assert_eq!(batches[3][0], outcome.best_params, "{label}");
        assert!(outcome.converged, "{label}");
        assert_eq!(outcome.iterations_run, 9, "{label}");
    }
}

#[test]
fn a_flat_landscape_never_moves_the_point() {
    let oracle = Recorder::new(flat);
    let mut args = rotosolve_args(&oracle, 4);
    args.sweeps = 3;
    let outcome = run(args);
    let batches = oracle.steps();
    let theta0 = batches[0][0].clone();
    for (k, batch) in batches.iter().enumerate() {
        let j = k % 3;
        let mut plus = theta0.clone();
        plus[j] += FRAC_PI_2;
        let mut minus = theta0.clone();
        minus[j] -= FRAC_PI_2;
        assert_eq!(batch[batch.len() - 2..], [plus, minus], "step {k}");
    }
    assert_eq!(oracle.final_point(), theta0);
    // Ties keep the first measurement.
    assert_eq!(outcome.best_params, theta0);
    assert_eq!(outcome.best_fitness, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// One batch per step
// ─────────────────────────────────────────────────────────────────────────────

/// Step `k` sends `[x + π/2·e_j, x − π/2·e_j]` with `j = k mod d`, headed by
/// `x` on the first step and every `reevaluate_every`: the circuits per step
/// are 3 and 2. The final point is one more batch, of one.
#[test]
fn each_step_is_one_batch_of_two_with_the_point_first_when_measured() {
    for (reevaluate_every, sizes) in [
        (5, vec![3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 1]),
        (1, [vec![3; 12], vec![1]].concat()),
        (4, vec![3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 1]),
        (100, [vec![3], vec![2; 11], vec![1]].concat()),
    ] {
        let oracle = Recorder::new(rough);
        let mut args = rotosolve_args(&oracle, 9);
        args.sweeps = 4;
        args.reevaluate_every = reevaluate_every;
        let outcome = run(args);
        let batches = oracle.batches();
        let label = format!("every {reevaluate_every}");
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            sizes,
            "{label}"
        );
        assert_eq!(outcome.iterations_run, 12, "{label}");
        assert!(!outcome.converged, "{label}");
        for (k, batch) in oracle.steps().iter().enumerate() {
            let j = k % 3;
            let x = point_of(batch, j);
            let (plus, minus) = (&batch[batch.len() - 2], &batch[batch.len() - 1]);
            for i in (0..3).filter(|&i| i != j) {
                assert_eq!((plus[i], minus[i]), (x[i], x[i]), "{label} step {k}");
            }
            if batch.len() == 3 {
                assert_eq!(plus[j], x[j] + FRAC_PI_2, "{label} step {k}");
                assert_eq!(minus[j], x[j] - FRAC_PI_2, "{label} step {k}");
            } else {
                assert!((plus[j] - minus[j] - PI).abs() < 1e-12, "{label} step {k}");
            }
        }
    }
}

/// A one-parameter, one-sweep run is one batch of three, then the point it
/// moved to, measured alone. Its one history entry holds both.
#[test]
fn one_step_measures_theta0_its_two_neighbours_and_where_it_lands() {
    let oracle = Recorder::new(rough);
    let mut args = rotosolve_args(&oracle, 7);
    args.sweeps = 1;
    args.dimensions = 1;
    let outcome = run(args);
    let batches = oracle.batches();
    assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [3, 1]);
    let measured: Vec<(Vec<f64>, f64)> = batches
        .concat()
        .into_iter()
        .zip(oracle.values().concat())
        .collect();
    let best = (1..4).fold(0, |b, i| if measured[i].1 > measured[b].1 { i } else { b });
    assert_eq!(outcome.best_params, measured[best].0);
    assert_eq!(outcome.best_fitness, measured[best].1);
    assert_eq!(outcome.fitness_history, vec![outcome.best_fitness]);
    assert_eq!(outcome.iterations_run, 1);
    // The step moved θ₀; the final point is not one of the three it measured.
    assert_ne!(batches[1][0], batches[0][0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// What is measured (C-5)
// ─────────────────────────────────────────────────────────────────────────────

/// C-5 against the batches the oracle saw, over 25 seeds: every candidate is
/// a measured point, `best_params` the first with the best value,
/// `best_fitness` the oracle's value for it, and `fitness_history[k]` the
/// best measured up to step `k`, the last entry with the final point too.
#[test]
fn best_params_fitness_invariant_holds_across_seeds() {
    let mut not_last = false;
    for seed in 0..25 {
        let oracle = Recorder::new(rough);
        let mut args = rotosolve_args(&oracle, seed);
        args.sweeps = 8;
        args.reevaluate_every = 4;
        let outcome = run(args);
        let batches = oracle.batches();
        let label = format!("seed {seed}");
        assert_eq!(oracle.steps().len(), 24, "{label}");
        assert_eq!(outcome.iterations_run, 24, "{label}");
        assert_eq!(outcome.fitness_history.len(), 24, "{label}");
        let mut running = f64::NEG_INFINITY;
        let mut first_best: Option<&Vec<f64>> = None;
        for (k, batch) in batches.iter().enumerate() {
            for candidate in batch {
                if rough(candidate) > running {
                    running = rough(candidate);
                    first_best = Some(candidate);
                }
            }
            // The last step's entry holds the final point's batch too.
            if k < 23 {
                assert_eq!(outcome.fitness_history[k], running, "{label} step {k}");
            }
        }
        assert_eq!(outcome.fitness_history.last(), Some(&running), "{label}");
        assert_eq!(outcome.fitness_history.last(), Some(&outcome.best_fitness));
        assert_eq!(outcome.best_fitness, rough(&outcome.best_params), "{label}");
        assert_eq!(first_best, Some(&outcome.best_params), "{label}");
        not_last |= !batches[23..].concat().contains(&outcome.best_params);
    }
    assert!(not_last, "no run's best was found before its last step");
}

/// A seed reproduces a run, and another seed runs differently. The step calls
/// `atan2` and `hypot`, whose last bit may differ between platforms (contract
/// C-7), so the outcome is compared between two runs rather than pinned bit
/// for bit.
#[test]
fn seeded_runs_are_reproducible() {
    let go = |seed| {
        let oracle = Recorder::new(coupled);
        let outcome = run(rotosolve_args(&oracle, seed));
        (outcome, oracle.batches())
    };
    let (a, a_batches) = go(11);
    let (b, b_batches) = go(11);
    assert_eq!(a, b);
    assert_eq!(a_batches, b_batches);
    let (_, c_batches) = go(12);
    assert_ne!(a_batches[0][0], c_batches[0][0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Stopping, in whole sweeps
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn patience_counts_whole_sweeps() {
    for patience in [1, 2, 3] {
        let oracle = Recorder::new(flat);
        let mut args = rotosolve_args(&oracle, 1);
        args.tolerance = 1e-6;
        args.patience = patience;
        let outcome = run(args);
        let steps = 3 * patience as usize;
        assert_eq!(oracle.steps().len(), steps, "patience {patience}");
        assert_eq!(outcome.iterations_run, steps, "patience {patience}");
        assert_eq!(outcome.fitness_history, vec![0.0; steps]);
        assert!(outcome.converged, "patience {patience}");
    }
}

#[test]
fn a_zero_tolerance_never_stops_early() {
    let oracle = Recorder::new(flat);
    let mut args = rotosolve_args(&oracle, 1);
    args.patience = 1;
    let outcome = run(args);
    assert_eq!(outcome.iterations_run, 30);
    assert!(!outcome.converged);
}

/// An improvement in the middle resets the count: sweeps 1, 3 and 4 improve
/// by nothing, sweep 2 by 1, so a patience of two stops after sweep 4, not 2.
#[test]
fn an_improving_sweep_resets_the_count() {
    fn one_in_sweep_two(index: usize, values: &mut [f64]) {
        // Batch 3 is the first step of sweep 2 (three steps per sweep).
        if index == 3 {
            values[0] = 1.0;
        }
    }
    let oracle = Recorder::editing(flat, one_in_sweep_two);
    let mut args = rotosolve_args(&oracle, 1);
    args.tolerance = 0.5;
    args.patience = 2;
    let outcome = run(args);
    assert_eq!(outcome.iterations_run, 12);
    assert!(outcome.converged);
    assert_eq!(
        outcome.fitness_history,
        [[0.0; 3], [1.0; 3], [1.0; 3], [1.0; 3]].concat()
    );
    assert_eq!(outcome.best_params, oracle.batches()[3][0]);
}

/// θ₀'s measurement is the reference for the first sweep: a `NaN` there makes
/// that sweep's improvement `NaN`, which resets the count.
#[test]
fn a_nan_improvement_resets_the_count() {
    fn nan_theta0(index: usize, values: &mut [f64]) {
        if index == 0 {
            values[0] = f64::NAN;
        }
    }
    let oracle = Recorder::editing(flat, nan_theta0);
    let mut args = rotosolve_args(&oracle, 1);
    args.tolerance = 0.5;
    args.patience = 1;
    let outcome = run(args);
    assert_eq!(outcome.iterations_run, 6);
    assert!(outcome.converged);
    assert_eq!(outcome.best_fitness, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-finite values
// ─────────────────────────────────────────────────────────────────────────────

/// A `NaN` for θ₀ leaves the point where it is until it is measured again:
/// the predicted values after it are unknown too.
#[test]
fn a_nan_for_the_point_holds_it_until_it_is_measured_again() {
    fn nan_theta0(index: usize, values: &mut [f64]) {
        if index == 0 {
            values[0] = f64::NAN;
        }
    }
    let oracle = Recorder::editing(coupled, nan_theta0);
    let mut args = rotosolve_args(&oracle, 5);
    args.sweeps = 3;
    args.reevaluate_every = 4;
    let outcome = run(args);
    let batches = oracle.batches();
    let theta0 = batches[0][0].clone();
    // Steps 1 to 3 move from θ₀; step 4 measures it again (the head of its
    // batch) and moves coordinate 1.
    for (k, batch) in batches.iter().enumerate().take(4).skip(1) {
        let j = k % 3;
        for i in (0..3).filter(|&i| i != j) {
            assert_eq!(batch[0][i], theta0[i], "step {k}");
        }
        assert_eq!(batch[0][j], theta0[j] + FRAC_PI_2, "step {k}");
    }
    assert_eq!(batches[4][0], theta0);
    assert_ne!(batches[5][0][1], theta0[1]);
    assert!(outcome.best_fitness.is_finite());
    assert_ne!(outcome.best_params, theta0);
    assert!(outcome.fitness_history.iter().all(|f| f.is_finite()));
}

/// A `NaN` beside the point skips only its coordinate: the point's value is
/// still known (predicted by the step before), so the next step moves to the
/// maximum along its coordinate as usual.
#[test]
fn a_nan_beside_the_point_skips_only_its_coordinate() {
    fn nan_minus_of_step_one(index: usize, values: &mut [f64]) {
        if index == 1 {
            values[1] = f64::NAN;
        }
    }
    let plain = Recorder::new(coupled);
    run(rotosolve_args(&plain, 5));
    let poisoned = Recorder::editing(coupled, nan_minus_of_step_one);
    let outcome = run(rotosolve_args(&poisoned, 5));
    let (plain, batches) = (plain.batches(), poisoned.batches());
    assert_eq!(batches[..2], plain[..2]);
    // Step 1 (coordinate 1) is skipped: step 2 moves from the point step 1
    // moved from.
    let x1 = point_of(&batches[1], 1);
    assert_eq!(batches[2][0][0], batches[1][0][0]);
    assert!((batches[2][0][1] - x1[1]).abs() < 1e-12);
    assert_eq!(batches[2][0][2], batches[1][0][2] + FRAC_PI_2);
    // Step 2 lands on the maximum along coordinate 2 from there.
    let x2 = point_of(&batches[2], 2);
    let (at, grid_best) = grid_maximum(coupled, &x2, 2);
    let after = point_of(&batches[3], 0);
    assert!(angular_gap(after[2], at) <= 2.0 * TAU / 20_000.0);
    let mut moved = x2.clone();
    moved[2] = after[2];
    assert!(coupled(&moved) >= grid_best - 1e-12);
    assert!(outcome.best_fitness.is_finite());
    assert_eq!(outcome.best_fitness, coupled(&outcome.best_params));
}

#[test]
fn a_nan_measurement_never_becomes_the_best() {
    fn nan_every_head(index: usize, values: &mut [f64]) {
        if index % 2 == 0 {
            values[0] = f64::NAN;
        }
    }
    for seed in 0..10 {
        let oracle = Recorder::editing(rough, nan_every_head);
        let mut args = rotosolve_args(&oracle, seed);
        args.reevaluate_every = 2;
        let outcome = run(args);
        assert!(outcome.best_fitness.is_finite(), "seed {seed}");
        assert_eq!(outcome.best_fitness, rough(&outcome.best_params));
        assert!(outcome.fitness_history.iter().all(|f| f.is_finite()));
    }
}

#[test]
fn an_oracle_that_only_returns_nan_does_not_panic() {
    fn always_nan(_: &[f64]) -> f64 {
        f64::NAN
    }
    let oracle = Recorder::new(always_nan);
    let mut args = rotosolve_args(&oracle, 3);
    args.sweeps = 4;
    let outcome = run(args);
    let batches = oracle.batches();
    // No step is ever finite, so the point never moves: the best is θ₀, the
    // first point measured, with the NaN it got.
    assert_eq!(outcome.iterations_run, 12);
    assert!(!outcome.converged);
    assert_eq!(outcome.best_params, batches[0][0]);
    assert!(outcome.best_fitness.is_nan());
    assert_eq!(outcome.fitness_history.len(), 12);
    for k in [5, 10] {
        assert_eq!(batches[k][0], batches[0][0]);
    }
}

/// An infinite value is a measurement like any other for C-5, but gives no
/// step.
#[test]
fn an_infinite_value_gives_no_step() {
    fn infinite_plus(index: usize, values: &mut [f64]) {
        if index == 0 {
            values[1] = f64::INFINITY;
        }
    }
    let oracle = Recorder::editing(coupled, infinite_plus);
    let mut args = rotosolve_args(&oracle, 2);
    args.sweeps = 1;
    let outcome = run(args);
    let batches = oracle.batches();
    assert_eq!(batches[1][0][0], batches[0][0][0]);
    assert_eq!(outcome.best_params, batches[0][1]);
    assert_eq!(outcome.best_fitness, f64::INFINITY);
}

// ─────────────────────────────────────────────────────────────────────────────
// Preconditions and the oracle's length
// ─────────────────────────────────────────────────────────────────────────────

/// Runs `edit`ed arguments and returns the error, after checking that the
/// oracle was not called.
fn rejected(edit: impl FnOnce(&mut AlgorithmRotosolveArgs)) -> OptimizerError {
    let oracle = Recorder::new(cosine);
    let mut args = rotosolve_args(&oracle, 0);
    edit(&mut args);
    let err = AlgorithmRotosolve.optimize(args).unwrap_err();
    assert!(oracle.batches().is_empty(), "the oracle was called");
    err
}

/// An edit that breaks one precondition.
type Breaks = fn(&mut AlgorithmRotosolveArgs);

#[test]
fn invalid_configurations_are_rejected_before_any_oracle_call() {
    let cases: [(&str, &str, Breaks); 13] = [
        (
            "sweeps",
            "must be at least 1 (a sweep updates every parameter once)",
            |a| a.sweeps = 0,
        ),
        (
            "dimensions",
            "must be at least 1 (there is nothing to optimise)",
            |a| a.dimensions = 0,
        ),
        (
            "reevaluate_every",
            "must be at least 1 (steps between measurements of the parameters)",
            |a| a.reevaluate_every = 0,
        ),
        ("tolerance", "must be finite and >= 0, got -0.1", |a| {
            a.tolerance = -0.1
        }),
        ("tolerance", "must be finite and >= 0, got NaN", |a| {
            a.tolerance = f64::NAN
        }),
        ("tolerance", "must be finite and >= 0, got inf", |a| {
            a.tolerance = f64::INFINITY
        }),
        ("tolerance", "must be finite and >= 0, got -inf", |a| {
            a.tolerance = f64::NEG_INFINITY
        }),
        (
            "patience",
            "must be at least 1 (sweeps that improve less than tolerance before the run stops)",
            |a| a.patience = 0,
        ),
        (
            "initial_point",
            "must have one value per dimension (3), got 2",
            |a| a.initial_point = Some(vec![0.0, 1.0]),
        ),
        (
            "initial_point",
            "every value must be finite, got NaN at index 1",
            |a| a.initial_point = Some(vec![0.0, f64::NAN, 1.0]),
        ),
        // In order: the configuration first, then the point.
        (
            "sweeps",
            "must be at least 1 (a sweep updates every parameter once)",
            |a| {
                a.sweeps = 0;
                a.patience = 0;
                a.initial_point = Some(vec![]);
            },
        ),
        ("tolerance", "must be finite and >= 0, got -1", |a| {
            a.tolerance = -1.0;
            a.patience = 0;
        }),
        (
            "patience",
            "must be at least 1 (sweeps that improve less than tolerance before the run stops)",
            |a| {
                a.patience = 0;
                a.initial_point = Some(vec![]);
            },
        ),
    ];
    for (parameter, reason, edit) in cases {
        assert_eq!(
            rejected(edit),
            OptimizerError::InvalidConfig {
                parameter,
                reason: reason.to_string()
            }
        );
    }
}

/// Returns one value too few from its `short_from`-th batch on.
struct Short {
    short_from: usize,
    calls: Mutex<usize>,
}

impl EvaluationOracle for Short {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let mut calls = self.calls.lock().unwrap();
        *calls += 1;
        let n = candidates.len() - usize::from(*calls > self.short_from);
        vec![0.0; n]
    }
}

#[test]
fn a_wrong_length_from_the_oracle_is_an_error() {
    // Batch 30 is the final point's, after the 30 steps of 10 sweeps.
    for (short_from, expected) in [(0, 3), (1, 2), (5, 3), (30, 1)] {
        let oracle = Recorder::new(cosine);
        let args = AlgorithmRotosolveArgs {
            oracle: Box::new(Short {
                short_from,
                calls: Mutex::new(0),
            }),
            ..rotosolve_args(&oracle, 0)
        };
        assert_eq!(
            AlgorithmRotosolve.optimize(args).unwrap_err(),
            OptimizerError::OracleLengthMismatch {
                expected,
                got: expected - 1
            },
            "short from batch {short_from}"
        );
    }
}
