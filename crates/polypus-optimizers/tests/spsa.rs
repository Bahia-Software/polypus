//! Behavioural tests for [`AlgorithmSPSA`].
//!
//! Like `adam.rs`, these run without a Python interpreter. The oracles are
//! pure functions of the candidate; the ones that record what they are handed
//! let a test check what the optimizer actually sent (the calibration batch,
//! then one batch per iteration), rebuild each step from it, and check that
//! nothing was sent at all when the configuration is rejected.

use polypus_optimizers::{
    AlgorithmSPSA, AlgorithmSPSAArgs, EvaluationOracle, OptimizationOutcome, Optimizer,
    OptimizerError,
};
use std::f64::consts::TAU;
use std::sync::{Arc, Mutex};

/// The maximum of the test landscapes.
const TARGET: [f64; 3] = [0.7, -1.2, 2.1];

/// The calibration batch: 25 pairs.
const CALIBRATION: usize = 50;

/// A concave quadratic with its maximum `0` at [`TARGET`].
fn bowl(x: &[f64]) -> f64 {
    -x.iter()
        .zip(TARGET)
        .map(|(xi, ti)| (xi - ti).powi(2))
        .sum::<f64>()
}

/// `Σ cos(xᵢ − tᵢ)`: periodic, with its maximum 3 at [`TARGET`] + 2πk.
fn cosine(x: &[f64]) -> f64 {
    x.iter().zip(TARGET).map(|(xi, ti)| (xi - ti).cos()).sum()
}

/// The quadratic bowl with a sinusoidal ripple: many local optima and
/// near-ties, where large steps bounce and the best θ is often not the last.
fn rough(x: &[f64]) -> f64 {
    x.iter().map(|v| -(v - 1.0).powi(2) + (5.0 * v).sin()).sum()
}

/// A plane, `w·x`: every estimate's slope is `w·Δ`, whatever the point.
fn plane(x: &[f64]) -> f64 {
    x.iter().zip([1.5, -0.5, 2.0]).map(|(xi, wi)| xi * wi).sum()
}

/// The distance from `x` to `t` on the circle.
fn angular_gap(x: f64, t: f64) -> f64 {
    let gap = (x - t).rem_euclid(TAU);
    gap.min(TAU - gap)
}

/// Every batch an oracle was handed, with the values it returned.
type Log = Arc<Mutex<Vec<(Vec<Vec<f64>>, Vec<f64>)>>>;

/// Scores candidates with `fitness`, after `edit` has had its say on the
/// values of batch `index` (0 is the calibration), and logs every batch.
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

fn spsa_args(oracle: &Recorder, seed: u64) -> AlgorithmSPSAArgs {
    AlgorithmSPSAArgs {
        oracle: Box::new(oracle.clone()),
        max_iters: 300,
        resamplings: 1,
        perturbation: 0.1,
        target_step: 0.63,
        alpha: AlgorithmSPSA::DEFAULT_ALPHA,
        gamma: AlgorithmSPSA::DEFAULT_GAMMA,
        stability_constant: 0.0,
        eval_interval: 20,
        dimensions: TARGET.len() as u32,
        seed: Some(seed),
        initial_point: None,
    }
}

/// Whether iteration `k` (from 1) measures θ: the first, and every
/// `interval` after it.
fn measured(k: usize, interval: usize) -> bool {
    (k - 1) % interval == 0
}

/// The ±1 perturbation of a pair around `centre`, read from its `+` side.
fn signs(plus: &[f64], centre: &[f64]) -> Vec<f64> {
    plus.iter()
        .zip(centre)
        .map(|(p, x)| if p > x { 1.0 } else { -1.0 })
        .collect()
}

/// Checks that `pairs` are `[x + c·Δ₁, x − c·Δ₁, …]` around `centre`, each Δ
/// of ±1 components, to `tolerance`, and returns the Δs.
fn assert_pairs(pairs: &[Vec<f64>], centre: &[f64], c: f64, tolerance: f64) -> Vec<Vec<f64>> {
    assert_eq!(pairs.len() % 2, 0);
    pairs
        .chunks_exact(2)
        .map(|pair| {
            let delta = signs(&pair[0], centre);
            for j in 0..centre.len() {
                let plus = centre[j] + c * delta[j];
                let minus = centre[j] - c * delta[j];
                assert!(
                    (pair[0][j] - plus).abs() <= tolerance,
                    "{pair:?} vs {centre:?}"
                );
                assert!(
                    (pair[1][j] - minus).abs() <= tolerance,
                    "{pair:?} vs {centre:?}"
                );
            }
            delta
        })
        .collect()
}

/// `a = target_step / mean|ĝ|` from a logged calibration batch.
fn calibrated_gain(values: &[f64], c: f64, target_step: f64) -> f64 {
    let magnitudes: Vec<f64> = values
        .chunks_exact(2)
        .map(|p| ((p[0] - p[1]) / (2.0 * c)).abs())
        .collect();
    target_step / (magnitudes.iter().sum::<f64>() / magnitudes.len() as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// The calibration
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_calibration_is_one_batch_of_25_pairs_around_theta0() {
    let point = vec![0.3, -2.0, 7.5];
    let oracle = Recorder::new(bowl);
    let mut args = spsa_args(&oracle, 5);
    args.initial_point = Some(point.clone());
    args.max_iters = 3;
    AlgorithmSPSA.optimize(args).unwrap();
    let batches = oracle.batches();
    assert_eq!(batches[0].len(), CALIBRATION);
    // [θ₀ + c·Δᵢ, θ₀ − c·Δᵢ] with c the perturbation itself, exactly.
    let deltas = assert_pairs(&batches[0], &point, 0.1, 0.0);
    assert!(deltas.windows(2).any(|pair| pair[0] != pair[1]));
    // The first iteration measures θ₀ itself, first in its batch.
    assert_eq!(batches[1][0], point);

    // Without a point, θ₀ is drawn from [0, 2π) and the calibration is
    // centred on it.
    let oracle = Recorder::new(bowl);
    let mut args = spsa_args(&oracle, 5);
    args.max_iters = 1;
    AlgorithmSPSA.optimize(args).unwrap();
    let batches = oracle.batches();
    let theta0 = &batches[1][0];
    assert!(theta0.iter().all(|x| (0.0..TAU).contains(x)), "{theta0:?}");
    assert_pairs(&batches[0], theta0, 0.1, 1e-15);
}

/// The calibration sets `a = target_step / mean|ĝ|`, with `mean|ĝ|` the mean
/// of `|f₊ − f₋| / (2c)` over its 25 pairs, and the first gain is `a` itself:
/// the first step is `a·ĝ₁`, and it scales with `target_step`.
#[test]
fn the_calibration_sets_the_first_step() {
    let first_step = |target_step: f64| -> Vec<f64> {
        let oracle = Recorder::new(plane);
        let mut args = spsa_args(&oracle, 9);
        args.max_iters = 2;
        args.eval_interval = 1;
        args.target_step = target_step;
        AlgorithmSPSA.optimize(args).unwrap();
        let (batches, values) = (oracle.batches(), oracle.values());
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            [CALIBRATION, 3, 3, 1]
        );
        let a = calibrated_gain(&values[0], 0.1, target_step);
        let theta1 = &batches[1][0];
        let delta = &assert_pairs(&batches[1][1..], theta1, 0.1, 1e-15)[0];
        let slope = (values[1][1] - values[1][2]) / (2.0 * 0.1);
        let theta2 = &batches[2][0];
        let step: Vec<f64> = theta2.iter().zip(theta1).map(|(b, a)| b - a).collect();
        for j in 0..3 {
            let expected = a * slope * delta[j];
            assert!((step[j] - expected).abs() < 1e-12, "{step:?}, a = {a}");
        }
        step
    };
    let step = first_step(0.63);
    // On a plane every |ĝ| is |w·Δ|, so the first step is about target_step.
    assert!(
        step.iter().all(|s| (0.1..3.0).contains(&s.abs())),
        "{step:?}"
    );
    let doubled = first_step(1.26);
    for (s, d) in step.iter().zip(&doubled) {
        assert!((d - 2.0 * s).abs() < 1e-12, "{step:?} vs {doubled:?}");
    }
}

/// The calibration scales `a` by `(1 + A)^α`, so the first gain, and so the
/// first step, is the same whatever the stability constant `A`; the second
/// step, from the same θ along the same perturbation, is longer by
/// `(2(1 + A) / (2 + A))^α`: the gain decays more slowly at first.
#[test]
fn the_stability_constant_keeps_the_first_step_and_slows_the_decay() {
    let steps = |stability: f64| -> (Vec<f64>, Vec<f64>) {
        let oracle = Recorder::new(plane);
        let mut args = spsa_args(&oracle, 9);
        args.max_iters = 2;
        args.eval_interval = 1;
        args.stability_constant = stability;
        AlgorithmSPSA.optimize(args).unwrap();
        let batches = oracle.batches();
        let theta: Vec<&Vec<f64>> = batches[1..].iter().map(|b| &b[0]).collect();
        let step = |k: usize| -> Vec<f64> {
            theta[k + 1]
                .iter()
                .zip(theta[k])
                .map(|(b, a)| b - a)
                .collect()
        };
        (step(0), step(1))
    };
    let (first, second) = steps(0.0);
    for stability in [0.5, 9.0, 30.0] {
        let (first_a, second_a) = steps(stability);
        let ratio =
            (2.0 * (1.0 + stability) / (2.0 + stability)).powf(AlgorithmSPSA::DEFAULT_ALPHA);
        for j in 0..3 {
            assert!((first_a[j] - first[j]).abs() < 1e-12, "A {stability}");
            // On a plane every estimate along Δ is the same, whatever θ.
            assert!(second[j].abs() > 1e-3);
            assert!(
                (second_a[j] / second[j] - ratio).abs() < 1e-12,
                "A {stability}: {second_a:?} vs {second:?}"
            );
        }
    }
}

/// A flat landscape around θ₀ gives `mean|ĝ| = 0`, and an oracle of only
/// `NaN` no magnitude at all: either way `a = target_step`, as if `mean|ĝ|`
/// were 1, and the run goes on without a panic.
#[test]
fn a_flat_or_non_finite_calibration_falls_back_to_the_target_step() {
    fn flat_calibration(index: usize, values: &mut [f64]) {
        if index == 0 {
            values.fill(1.0);
        }
    }
    fn nan_calibration(index: usize, values: &mut [f64]) {
        if index == 0 {
            values.fill(f64::NAN);
        }
    }
    for edit in [flat_calibration as fn(usize, &mut [f64]), nan_calibration] {
        let oracle = Recorder::editing(plane, edit);
        let mut args = spsa_args(&oracle, 4);
        args.max_iters = 2;
        args.eval_interval = 1;
        let outcome = AlgorithmSPSA.optimize(args).unwrap();
        let (batches, values) = (oracle.batches(), oracle.values());
        let theta1 = &batches[1][0];
        let delta = &assert_pairs(&batches[1][1..], theta1, 0.1, 1e-15)[0];
        let slope = (values[1][1] - values[1][2]) / (2.0 * 0.1);
        for j in 0..3 {
            let expected = theta1[j] + 0.63 * slope * delta[j];
            assert!((batches[2][0][j] - expected).abs() < 1e-12, "{batches:?}");
        }
        assert!(outcome.best_fitness.is_finite());
    }
}

/// A calibration pair with a non-finite value is left out of `mean|ĝ|`.
#[test]
fn a_non_finite_calibration_pair_is_left_out_of_the_mean() {
    fn poison_first_pair(index: usize, values: &mut [f64]) {
        if index == 0 {
            values[0] = f64::NAN;
        }
    }
    let oracle = Recorder::editing(plane, poison_first_pair);
    let mut args = spsa_args(&oracle, 4);
    args.max_iters = 2;
    args.eval_interval = 1;
    AlgorithmSPSA.optimize(args).unwrap();
    let (batches, values) = (oracle.batches(), oracle.values());
    let a = calibrated_gain(&values[0][2..], 0.1, 0.63);
    let theta1 = &batches[1][0];
    let delta = &assert_pairs(&batches[1][1..], theta1, 0.1, 1e-15)[0];
    let slope = (values[1][1] - values[1][2]) / (2.0 * 0.1);
    for j in 0..3 {
        let expected = theta1[j] + a * slope * delta[j];
        assert!((batches[2][0][j] - expected).abs() < 1e-12);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// One batch per iteration, with Spall's gains
// ─────────────────────────────────────────────────────────────────────────────

/// After the calibration, each iteration is one batch of `2K` candidates,
/// with θ_k first (`2K + 1`) on the first iteration and every
/// `eval_interval` after it; the pairs are centred on θ_k. After the last
/// step, a batch of one candidate measures the θ it reached.
#[test]
fn each_iteration_is_one_batch_of_2k_with_theta_first_when_measured() {
    for resamplings in [1, 3, 8] {
        let oracle = Recorder::new(bowl);
        let mut args = spsa_args(&oracle, 2);
        args.max_iters = 23;
        args.eval_interval = 5;
        args.resamplings = resamplings;
        let outcome = AlgorithmSPSA.optimize(args).unwrap();
        let k2 = 2 * resamplings as usize;
        let expected: Vec<usize> = std::iter::once(CALIBRATION)
            .chain((1..=23).map(|k| k2 + usize::from(measured(k, 5))))
            .chain(std::iter::once(1))
            .collect();
        let batches = oracle.batches();
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            expected,
            "K = {resamplings}"
        );
        assert_eq!(
            expected.iter().filter(|&&n| n == k2 + 1).count(),
            5,
            "k = 1, 6, 11, 16, 21"
        );
        let iterations = &batches[1..batches.len() - 1];
        for (k, batch) in (1..).zip(iterations) {
            let pairs = &batch[batch.len() - k2..];
            // Every pair of the batch shares one centre: θ_k.
            let centre: Vec<f64> = pairs[0]
                .iter()
                .zip(&pairs[1])
                .map(|(p, m)| (p + m) / 2.0)
                .collect();
            if measured(k, 5) {
                for (x, c) in batch[0].iter().zip(&centre) {
                    assert!((x - c).abs() < 1e-12, "K = {resamplings}, k = {k}");
                }
            }
            let c_k = 0.1 / (k as f64).powf(AlgorithmSPSA::DEFAULT_GAMMA);
            assert_pairs(pairs, &centre, c_k, 1e-12);
        }
        assert_eq!(outcome.iterations_run, 23);
        assert!(!outcome.converged);
    }
}

/// The θs follow Spall's recurrence: `θ_{k+1} = θ_k + a_k·ĝ_k` with
/// `a_k = a/(k + A)^α`, `a = a₁·(1 + A)^α` from the calibrated first gain
/// `a₁`, `c_k = c/k^γ` and `ĝ_k` the mean of the K estimates
/// `(f₊ − f₋)/(2c_k)·Δ`, rebuilt here from what the oracle saw over a dozen
/// iterations of a landscape whose gradient changes with θ, with non-default
/// exponents and stability constants. Pinned bits would only freeze a mistake
/// (and `powf` is not the same on every platform).
#[test]
fn every_step_follows_spalls_gains_and_the_mean_of_the_estimates() {
    for (alpha, gamma, resamplings, stability) in [
        (0.602, 0.101, 1, 0.0),
        (1.0, 1.0 / 6.0, 4, 0.0),
        (0.602, 0.101, 2, 7.5),
        (1.0, 1.0 / 6.0, 1, 1.0),
    ] {
        let oracle = Recorder::new(bowl);
        let mut args = spsa_args(&oracle, 13);
        args.max_iters = 12;
        args.eval_interval = 1;
        args.resamplings = resamplings;
        args.alpha = alpha;
        args.gamma = gamma;
        args.stability_constant = stability;
        args.perturbation = 0.2;
        args.target_step = 0.4;
        AlgorithmSPSA.optimize(args).unwrap();
        let (batches, values) = (oracle.batches(), oracle.values());
        let a = calibrated_gain(&values[0], 0.2, 0.4) * (1.0 + stability).powf(alpha);
        // Batch 13 is the θ the twelfth step reached, alone.
        assert_eq!(batches.len(), 14);
        assert_eq!(batches[13].len(), 1);
        for k in 1..=12 {
            let (batch, scored) = (&batches[k], &values[k]);
            let kf = k as f64;
            let (a_k, c_k) = (a / (kf + stability).powf(alpha), 0.2 / kf.powf(gamma));
            let theta = &batch[0];
            let deltas = assert_pairs(&batch[1..], theta, c_k, 1e-12);
            let mut gradient = [0.0; 3];
            for (delta, pair) in deltas.iter().zip(scored[1..].chunks_exact(2)) {
                let slope = (pair[0] - pair[1]) / (2.0 * c_k);
                for j in 0..3 {
                    gradient[j] += slope * delta[j] / resamplings as f64;
                }
            }
            for j in 0..3 {
                let expected = theta[j] + a_k * gradient[j];
                assert!(
                    (batches[k + 1][0][j] - expected).abs() < 1e-12,
                    "α {alpha}, γ {gamma}, K {resamplings}, A {stability}, k {k}"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Contract C-5
// ─────────────────────────────────────────────────────────────────────────────

/// C-5 against the batches the oracle saw, over 25 seeds and two values of K:
/// only the measured θs count (they head their batches), `best_params` is the
/// first measured θ with the best value, `best_fitness` the oracle's value for
/// it, and `fitness_history[k − 1]` the best measured up to iteration `k`,
/// the final θ counted in the last entry.
#[test]
fn best_params_fitness_invariant_holds_across_seeds() {
    for resamplings in [1, 4] {
        let mut overshot = false;
        for seed in 0..25 {
            let oracle = Recorder::new(rough);
            let mut args = spsa_args(&oracle, seed);
            // A large first step on a rippled landscape, so the iterates bounce
            // and the best is not always the last.
            args.target_step = 2.5;
            args.max_iters = 60;
            args.eval_interval = 3;
            args.resamplings = resamplings;
            let outcome = AlgorithmSPSA.optimize(args).unwrap();
            let label = format!("K {resamplings} seed {seed}");
            let batches = oracle.batches();
            assert_eq!(batches.len(), 62, "{label}");
            assert_eq!(outcome.iterations_run, 60);
            assert!(!outcome.converged);
            assert_eq!(outcome.fitness_history.len(), 60);
            let mut running = f64::NEG_INFINITY;
            let mut measured_thetas = Vec::new();
            let (last, iterations) = batches[1..].split_last().unwrap();
            for (k, batch) in (1..).zip(iterations) {
                if measured(k, 3) {
                    running = running.max(rough(&batch[0]));
                    measured_thetas.push(batch[0].clone());
                }
                if k < 60 {
                    assert_eq!(outcome.fitness_history[k - 1], running, "{label} k {k}");
                }
            }
            // The final θ, alone in the last batch, joins the last entry.
            assert_eq!(last.len(), 1, "{label}");
            running = running.max(rough(&last[0]));
            measured_thetas.push(last[0].clone());
            assert_eq!(outcome.fitness_history[59], running, "{label}");
            assert_eq!(outcome.fitness_history.last(), Some(&outcome.best_fitness));
            assert_eq!(outcome.best_fitness, rough(&outcome.best_params), "{label}");
            assert_eq!(
                measured_thetas
                    .iter()
                    .find(|theta| rough(theta) == outcome.best_fitness),
                Some(&outcome.best_params),
                "{label}"
            );
            overshot |= measured_thetas.last() != Some(&outcome.best_params);
        }
        assert!(
            overshot,
            "K {resamplings}: no run's best was before its last θ"
        );
    }
}

#[test]
fn converges_on_a_sphere_and_on_a_periodic_cosine() {
    type Fitness = fn(&[f64]) -> f64;
    let landscapes: [(&str, Fitness, f64); 2] = [("sphere", bowl, 0.0), ("cosine", cosine, 3.0)];
    for resamplings in [1, 4] {
        for (name, fitness, maximum) in landscapes {
            for seed in [21, 22, 23] {
                let oracle = Recorder::new(fitness);
                let mut args = spsa_args(&oracle, seed);
                // θ₀ is drawn in [0, 2π), up to 7 rad from the sphere's
                // maximum, where its gradient is large and the calibration
                // sizes `a` for it: a larger first step gets there in budget.
                args.target_step = 1.2;
                args.max_iters = 2000;
                args.resamplings = resamplings;
                let outcome = AlgorithmSPSA.optimize(args).unwrap();
                let label = format!("K {resamplings} {name} seed {seed}");
                assert!(
                    outcome.best_fitness > maximum - 1e-4,
                    "{label}: {}",
                    outcome.best_fitness
                );
                for (x, t) in outcome.best_params.iter().zip(TARGET) {
                    // θ is not wrapped, so compare on the circle.
                    assert!(
                        angular_gap(*x, t) < 1e-2,
                        "{label}: {:?}",
                        outcome.best_params
                    );
                }
            }
        }
    }
}

/// A seed reproduces a run, and another seed runs differently. The gains call
/// `powf`, whose last bit may differ between platforms (contract C-7), so the
/// outcome is compared between two runs rather than pinned bit for bit.
#[test]
fn seeded_runs_are_reproducible() {
    let run = |seed| {
        let oracle = Recorder::new(cosine);
        let mut args = spsa_args(&oracle, seed);
        args.max_iters = 80;
        args.resamplings = 3;
        let outcome = AlgorithmSPSA.optimize(args).unwrap();
        (outcome, oracle.batches())
    };
    let (a, a_batches) = run(11);
    let (b, b_batches) = run(11);
    assert_eq!(a, b);
    assert_eq!(a_batches, b_batches);
    let (c, _) = run(12);
    assert_ne!(a.best_params, c.best_params);
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-finite values
// ─────────────────────────────────────────────────────────────────────────────

/// `NaN` for the measured θ of every second iteration (eval_interval 1, so
/// every θ is measured), the first one included: a number replaces the first
/// `NaN`, and no `NaN` becomes the best afterwards.
#[test]
fn a_nan_measurement_never_becomes_the_best() {
    fn nan_every_second_theta(index: usize, values: &mut [f64]) {
        if index % 2 == 1 {
            values[0] = f64::NAN;
        }
    }
    let oracle = Recorder::editing(cosine, nan_every_second_theta);
    let mut args = spsa_args(&oracle, 8);
    args.eval_interval = 1;
    args.max_iters = 400;
    let outcome = AlgorithmSPSA.optimize(args).unwrap();
    let (batches, values) = (oracle.batches(), oracle.values());
    assert!(values[1][0].is_nan());
    assert!(outcome.fitness_history[0].is_nan());
    assert!(outcome.fitness_history[1..].iter().all(|f| f.is_finite()));
    // The best is a θ the oracle gave a number, and that number.
    assert!(outcome.best_fitness.is_finite());
    assert!(batches[1..]
        .iter()
        .zip(&values[1..])
        .any(
            |(batch, scored)| batch[0] == outcome.best_params && scored[0] == outcome.best_fitness
        ));
    assert_eq!(outcome.fitness_history.last(), Some(&outcome.best_fitness));
    // Half the measurements are lost, but the estimates are finite: it climbs.
    assert!(
        outcome.best_fitness > 3.0 - 1e-4,
        "{}",
        outcome.best_fitness
    );
}

/// An estimate with a non-finite value is left out of the iteration's mean;
/// with none left, θ does not move.
#[test]
fn a_non_finite_estimate_is_left_out_of_the_step() {
    fn poison(index: usize, values: &mut [f64]) {
        match index {
            // Iteration 2: both pairs lost.
            2 => values[1..].fill(f64::INFINITY),
            // Iteration 3: the first pair lost.
            3 => values[1] = f64::NAN,
            _ => {}
        }
    }
    let oracle = Recorder::editing(plane, poison);
    let mut args = spsa_args(&oracle, 6);
    args.max_iters = 4;
    args.eval_interval = 1;
    args.resamplings = 2;
    let outcome = AlgorithmSPSA.optimize(args).unwrap();
    let (batches, values) = (oracle.batches(), oracle.values());
    assert_eq!(batches[3][0], batches[2][0], "no estimate left, θ stays");
    let a = calibrated_gain(&values[0], 0.1, 0.63);
    let theta3 = &batches[3][0];
    let (a_3, c_3) = (a / 3f64.powf(0.602), 0.1 / 3f64.powf(0.101));
    let deltas = assert_pairs(&batches[3][1..], theta3, c_3, 1e-12);
    let slope = (values[3][3] - values[3][4]) / (2.0 * c_3);
    for j in 0..3 {
        let expected = theta3[j] + a_3 * slope * deltas[1][j];
        assert!((batches[4][0][j] - expected).abs() < 1e-12);
    }
    assert!(outcome.best_fitness.is_finite());
}

#[test]
fn an_oracle_that_only_returns_nan_does_not_panic() {
    fn always_nan(_: &[f64]) -> f64 {
        f64::NAN
    }
    let oracle = Recorder::new(always_nan);
    let mut args = spsa_args(&oracle, 3);
    args.max_iters = 12;
    args.eval_interval = 4;
    let outcome = AlgorithmSPSA.optimize(args).unwrap();
    let batches = oracle.batches();
    // No estimate is ever finite, so θ never moves: the best is θ₀, the first
    // θ measured, with the NaN it got.
    assert_eq!(outcome.iterations_run, 12);
    assert!(!outcome.converged);
    assert_eq!(outcome.best_params, batches[1][0]);
    assert!(outcome.best_fitness.is_nan());
    assert_eq!(outcome.fitness_history.len(), 12);
    // Iterations 1, 5 and 9, then the final θ in batch 13.
    for k in [1, 5, 9, 13] {
        assert_eq!(batches[k][0], batches[1][0]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preconditions and the oracle's length
// ─────────────────────────────────────────────────────────────────────────────

/// Runs `edit`ed arguments and returns the error, after checking that the
/// oracle was not called.
fn rejected(edit: impl FnOnce(&mut AlgorithmSPSAArgs)) -> OptimizerError {
    let oracle = Recorder::new(bowl);
    let mut args = spsa_args(&oracle, 0);
    edit(&mut args);
    let err = AlgorithmSPSA.optimize(args).unwrap_err();
    assert!(oracle.batches().is_empty(), "the oracle was called");
    err
}

fn invalid_parameter(err: OptimizerError) -> &'static str {
    match err {
        OptimizerError::InvalidConfig { parameter, .. } => parameter,
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// An edit that breaks one precondition.
type Breaks = fn(&mut AlgorithmSPSAArgs);

#[test]
fn invalid_configurations_are_rejected_before_any_oracle_call() {
    let cases: [(&str, Breaks); 27] = [
        ("max_iters", |a| a.max_iters = 0),
        ("dimensions", |a| a.dimensions = 0),
        ("resamplings", |a| a.resamplings = 0),
        // A batch of 2·resamplings + 1 candidates must fit in a u32.
        ("resamplings", |a| {
            a.resamplings = AlgorithmSPSA::MAX_RESAMPLINGS + 1
        }),
        ("resamplings", |a| a.resamplings = u32::MAX),
        ("perturbation", |a| a.perturbation = 0.0),
        ("perturbation", |a| a.perturbation = -0.1),
        ("perturbation", |a| a.perturbation = f64::NAN),
        ("perturbation", |a| a.perturbation = f64::INFINITY),
        ("target_step", |a| a.target_step = 0.0),
        ("target_step", |a| a.target_step = -1.0),
        ("target_step", |a| a.target_step = f64::NAN),
        // α in (0.5, 1]: Σ a_k diverges, and α − γ > 0.5 leaves room for γ.
        ("alpha", |a| a.alpha = 0.5),
        ("alpha", |a| a.alpha = 1.01),
        ("alpha", |a| a.alpha = f64::NAN),
        ("alpha", |a| a.alpha = 0.0),
        // γ > 0 and α − γ > 0.5.
        ("gamma", |a| a.gamma = 0.0),
        ("gamma", |a| a.gamma = -0.1),
        ("gamma", |a| a.gamma = 0.102),
        ("gamma", |a| a.gamma = f64::NAN),
        ("stability_constant", |a| a.stability_constant = -0.5),
        ("stability_constant", |a| a.stability_constant = f64::NAN),
        ("stability_constant", |a| {
            a.stability_constant = f64::INFINITY
        }),
        ("eval_interval", |a| a.eval_interval = 0),
        ("initial_point", |a| a.initial_point = Some(vec![0.0, 1.0])),
        ("initial_point", |a| {
            a.initial_point = Some(vec![0.0, f64::NAN, 1.0])
        }),
        // The configuration errors come before the point's.
        ("max_iters", |a| {
            a.max_iters = 0;
            a.initial_point = Some(vec![]);
        }),
    ];
    for (parameter, edit) in cases {
        assert_eq!(invalid_parameter(rejected(edit)), parameter);
    }
}

#[test]
fn the_edges_of_the_gain_exponents_are_accepted() {
    for (alpha, gamma) in [(1.0, 0.499), (0.602, 0.101), (1.0, 1.0 / 6.0), (0.51, 1e-3)] {
        let oracle = Recorder::new(bowl);
        let mut args = spsa_args(&oracle, 1);
        args.max_iters = 2;
        args.alpha = alpha;
        args.gamma = gamma;
        assert!(AlgorithmSPSA.optimize(args).is_ok(), "α {alpha}, γ {gamma}");
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
    // The calibration, the first iteration, the second, and the final θ.
    for (short_from, expected) in [(0, CALIBRATION), (1, 3), (2, 2), (301, 1)] {
        let oracle = Recorder::new(bowl);
        let args = AlgorithmSPSAArgs {
            oracle: Box::new(Short {
                short_from,
                calls: Mutex::new(0),
            }),
            ..spsa_args(&oracle, 0)
        };
        assert_eq!(
            AlgorithmSPSA.optimize(args).unwrap_err(),
            OptimizerError::OracleLengthMismatch {
                expected,
                got: expected - 1
            },
            "short from batch {short_from}"
        );
    }
}

fn outcome_of(oracle: &Recorder, seed: u64, max_iters: u32) -> OptimizationOutcome {
    let mut args = spsa_args(oracle, seed);
    args.max_iters = max_iters;
    AlgorithmSPSA.optimize(args).unwrap()
}

/// One iteration measures θ₀ with its pair, steps, and measures the θ it
/// reached: the best of the two is the outcome.
#[test]
fn one_iteration_measures_theta0_and_the_theta_it_reached() {
    let oracle = Recorder::new(bowl);
    let outcome = outcome_of(&oracle, 7, 1);
    let batches = oracle.batches();
    let sizes: Vec<usize> = batches.iter().map(Vec::len).collect();
    assert_eq!(sizes, [CALIBRATION, 3, 1]);
    let (start, end) = (&batches[1][0], &batches[2][0]);
    assert_ne!(start, end);
    let best = bowl(start).max(bowl(end));
    assert_eq!(outcome.best_fitness, best);
    assert!(outcome.best_params == *start || outcome.best_params == *end);
    assert_eq!(outcome.fitness_history, vec![best]);
}
