//! Behavioural tests for watching a run iteration by iteration and stopping
//! it: [`Optimizer::optimize_observed`] and the [`IterationObserver`] it is
//! shown each iteration (contract C-5).
//!
//! Every optimizer runs here, DE in its three strategies and CMA-ES with and
//! without restarts, with no Python interpreter. The oracles record every
//! batch they are handed, so a test can check that an observer that never
//! stops changes nothing, that it is shown each iteration once with the
//! entry the iteration leaves in `fitness_history`, and that a stop returns
//! an outcome that meets every postcondition of C-5 with no oracle call (and,
//! for External, no `ask`) after it.

use polypus_optimizers::{
    AdamRule, AlgorithmAdam, AlgorithmAdamArgs, AlgorithmCmaEs, AlgorithmCmaEsArgs,
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, AlgorithmExternal,
    AlgorithmExternalArgs, AlgorithmPSO, AlgorithmPSOArgs, AlgorithmQNG, AlgorithmQNGArgs,
    AlgorithmRotosolve, AlgorithmRotosolveArgs, AlgorithmSPSA, AlgorithmSPSAArgs, AskTell,
    AskTellError, DeStrategy, EvaluationOracle, FiniteDifferenceGradient, IterationObserver,
    IterationReport, OptimizationOutcome, Optimizer, Unobserved, VarianceOracle,
};
use std::f64::consts::{PI, TAU};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Every optimizer, as a training run can name it.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    De(DeStrategy),
    Pso,
    Qng,
    Cmaes,
    CmaesRestarts,
    Adam,
    Spsa,
    Rotosolve,
    External,
}

const KINDS: [Kind; 11] = [
    Kind::De(DeStrategy::Rand1Bin),
    Kind::De(DeStrategy::Shade),
    Kind::De(DeStrategy::LShade),
    Kind::Pso,
    Kind::Qng,
    Kind::Cmaes,
    Kind::CmaesRestarts,
    Kind::Adam,
    Kind::Spsa,
    Kind::Rotosolve,
    Kind::External,
];

const SEEDS: [u64; 4] = [0, 1, 7, 2026];

/// The quadratic bowl with a sinusoidal ripple: many local optima and
/// near-ties, where a stale best would break `best_fitness == f(best_params)`.
fn rough(x: &[f64]) -> f64 {
    x.iter().map(|v| -(v - 1.0).powi(2) + (5.0 * v).sin()).sum()
}

/// Every candidate scores the same, so every stopping test fires.
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

fn recorder(fitness: fn(&[f64]) -> f64) -> (Box<dyn EvaluationOracle>, Batches) {
    let batches = Batches::default();
    let oracle = Recorder {
        fitness,
        batches: Arc::clone(&batches),
    };
    (Box::new(oracle), batches)
}

/// Constant QFIM diagonal.
struct ConstVariance;

impl VarianceOracle for ConstVariance {
    fn variance(&self, _theta: &[f64], _param_index: usize) -> f64 {
        1.0
    }
}

/// A seeded random search in `[0, 2π)³`, four candidates a batch, that ends
/// the run with `None` after `batches` asks, and counts its asks and tells.
struct RandomSearch {
    state: u64,
    left: usize,
    asks: Arc<AtomicUsize>,
    tells: Arc<AtomicUsize>,
}

impl RandomSearch {
    fn uniform(&mut self) -> f64 {
        // SplitMix64, then the top 53 bits as a fraction of 2π.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64 * TAU
    }
}

impl AskTell for RandomSearch {
    fn ask(&mut self) -> Result<Option<Vec<Vec<f64>>>, AskTellError> {
        self.asks.fetch_add(1, Ordering::SeqCst);
        if self.left == 0 {
            return Ok(None);
        }
        self.left -= 1;
        Ok(Some(
            (0..4)
                .map(|_| (0..3).map(|_| self.uniform()).collect())
                .collect(),
        ))
    }

    fn tell(&mut self, _candidates: &[Vec<f64>], _values: &[f64]) -> Result<(), AskTellError> {
        self.tells.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// What a run returned and, for External, how often it was asked and told.
struct Run {
    outcome: OptimizationOutcome,
    asks: usize,
    tells: usize,
}

/// Run `kind` over `oracle` with `seed`, through `optimize` (or
/// `optimize_with_strategy`) when `observer` is `None`, through the observed
/// entry point otherwise. `converging` sets every stopping test to fire within
/// a few iterations on a flat landscape; otherwise none fires within the
/// budget on [`rough`].
fn run(
    kind: Kind,
    oracle: Box<dyn EvaluationOracle>,
    seed: u64,
    converging: bool,
    observer: Option<&mut dyn IterationObserver>,
) -> Run {
    let (tolerance, patience) = if converging {
        (1e-9, 2)
    } else {
        (1e-12, 1_000)
    };
    let asks = Arc::new(AtomicUsize::new(0));
    let tells = Arc::new(AtomicUsize::new(0));
    let outcome = match kind {
        Kind::De(strategy) => {
            let args = AlgorithmDifferentialEvolutionArgs {
                oracle,
                population_size: 8,
                generations: if converging { 20 } else { 6 },
                dimensions: 3,
                tolerance,
                patience,
                seed: Some(seed),
                initial_point: None,
                mutation_factor: None,
                crossover_rate: None,
            };
            match observer {
                Some(o) => AlgorithmDifferentialEvolution
                    .optimize_with_strategy_observed(strategy, args, o),
                None => AlgorithmDifferentialEvolution.optimize_with_strategy(strategy, args),
            }
        }
        Kind::Pso => {
            let args = AlgorithmPSOArgs {
                oracle,
                population_size: 8,
                generations: 6,
                dimensions: 3,
                bounds: (-PI, PI),
                inertia_weight: 0.5,
                cognitive_weight: 1.0,
                social_weight: 1.0,
                // Any spread is below 100, so it converges on its first test.
                tolerance: if converging { 100.0 } else { 1e-12 },
                seed: Some(seed),
                initial_point: None,
            };
            go(AlgorithmPSO, args, observer)
        }
        Kind::Qng => {
            let args = AlgorithmQNGArgs {
                oracle,
                max_iters: 6,
                learning_rate: 0.1,
                finite_difference_step: 0.1,
                bounds: (-PI, PI),
                dimensions: 3,
                variance_oracle: Box::new(ConstVariance),
                tikhonov_reg: 0.05,
                seed: Some(seed),
                initial_point: None,
            };
            go(AlgorithmQNG, args, observer)
        }
        Kind::Cmaes | Kind::CmaesRestarts => {
            let restarts = kind == Kind::CmaesRestarts;
            let args = AlgorithmCmaEsArgs {
                oracle,
                population_size: None,
                generations: if restarts || converging { 40 } else { 6 },
                dimensions: 3,
                sigma0: AlgorithmCmaEs::DEFAULT_SIGMA0,
                // With restarts, a run that improves by less than 0.05 over 3
                // generations restarts, so the 40 generations span several runs.
                tolerance: if restarts { 0.05 } else { tolerance },
                patience: if restarts { 3 } else { patience },
                restarts: if restarts { 2 } else { 0 },
                seed: Some(seed),
                initial_point: None,
            };
            go(AlgorithmCmaEs, args, observer)
        }
        Kind::Adam => {
            // As a training run without an exact gradient dispatches it.
            let oracle: Arc<dyn EvaluationOracle> = Arc::from(oracle);
            let gradient = FiniteDifferenceGradient::new(Arc::clone(&oracle), 0.1)
                .expect("a finite, positive step is valid");
            let args = AlgorithmAdamArgs {
                oracle: Box::new(oracle),
                gradient_oracle: Box::new(gradient),
                max_iters: if converging { 20 } else { 6 },
                learning_rate: 0.1,
                beta1: AlgorithmAdam::DEFAULT_BETA1,
                beta2: AlgorithmAdam::DEFAULT_BETA2,
                epsilon: AlgorithmAdam::DEFAULT_EPSILON,
                bounds: (-PI, PI),
                dimensions: 3,
                tolerance,
                patience,
                rule: AdamRule::Adam,
                seed: Some(seed),
                initial_point: None,
            };
            go(AlgorithmAdam, args, observer)
        }
        Kind::Spsa => {
            let args = AlgorithmSPSAArgs {
                oracle,
                max_iters: 6,
                resamplings: 1,
                perturbation: 0.1,
                target_step: 0.63,
                alpha: AlgorithmSPSA::DEFAULT_ALPHA,
                gamma: AlgorithmSPSA::DEFAULT_GAMMA,
                stability_constant: 0.0,
                eval_interval: 2,
                dimensions: 3,
                seed: Some(seed),
                initial_point: None,
            };
            go(AlgorithmSPSA, args, observer)
        }
        Kind::Rotosolve => {
            let args = AlgorithmRotosolveArgs {
                oracle,
                sweeps: if converging { 5 } else { 2 },
                reevaluate_every: 2,
                tolerance: if converging { 1e-9 } else { 0.0 },
                patience: 1,
                dimensions: 3,
                seed: Some(seed),
                initial_point: None,
            };
            go(AlgorithmRotosolve, args, observer)
        }
        Kind::External => {
            let args = AlgorithmExternalArgs {
                oracle,
                optimizer: Box::new(RandomSearch {
                    state: seed,
                    left: 6,
                    asks: Arc::clone(&asks),
                    tells: Arc::clone(&tells),
                }),
                max_evaluations: 1_000,
                dimensions: 3,
            };
            go(AlgorithmExternal, args, observer)
        }
    };
    Run {
        outcome: outcome.expect("valid arguments optimize successfully"),
        asks: asks.load(Ordering::SeqCst),
        tells: tells.load(Ordering::SeqCst),
    }
}

fn go<O: Optimizer>(
    optimizer: O,
    args: O::Args,
    observer: Option<&mut dyn IterationObserver>,
) -> Result<OptimizationOutcome, polypus_optimizers::OptimizerError> {
    match observer {
        Some(observer) => optimizer.optimize_observed(args, observer),
        None => optimizer.optimize(args),
    }
}

/// One report as the observer saw it, owned.
#[derive(Debug, Clone)]
struct Seen {
    iteration: usize,
    best_fitness: f64,
    best_params: Vec<f64>,
    /// Oracle batches handed out when the report came.
    batches: usize,
}

/// An observer that records every report, with the oracle batches so far,
/// and stops the run on iteration `stop_at`, if any.
struct Watcher {
    seen: Vec<Seen>,
    batches: Batches,
    stop_at: Option<usize>,
}

impl Watcher {
    fn new(batches: &Batches, stop_at: Option<usize>) -> Self {
        Self {
            seen: Vec::new(),
            batches: Arc::clone(batches),
            stop_at,
        }
    }
}

impl IterationObserver for Watcher {
    fn observe(&mut self, report: &IterationReport<'_>) -> ControlFlow<()> {
        self.seen.push(Seen {
            iteration: report.iteration,
            best_fitness: report.best_fitness,
            best_params: report.best_params.to_vec(),
            batches: self.batches.lock().unwrap().len(),
        });
        if Some(report.iteration) == self.stop_at {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

/// Every field of two outcomes, bit for bit.
fn assert_same(a: &OptimizationOutcome, b: &OptimizationOutcome, what: &str) {
    assert_eq!(
        bits(&a.best_params),
        bits(&b.best_params),
        "{what}: best_params"
    );
    assert_eq!(
        a.best_fitness.to_bits(),
        b.best_fitness.to_bits(),
        "{what}: best_fitness"
    );
    assert_eq!(a.iterations_run, b.iterations_run, "{what}: iterations_run");
    assert_eq!(a.converged, b.converged, "{what}: converged");
    assert_eq!(
        bits(&a.fitness_history),
        bits(&b.fitness_history),
        "{what}: fitness_history"
    );
}

/// The postconditions of contract C-5 for `outcome`, with `best_fitness`
/// recomputed from the fitness function the oracle scores with.
fn assert_postconditions(outcome: &OptimizationOutcome, fitness: fn(&[f64]) -> f64, what: &str) {
    assert_eq!(
        outcome.fitness_history.len(),
        outcome.iterations_run,
        "{what}: one history entry per iteration"
    );
    assert_eq!(
        outcome.best_fitness.to_bits(),
        fitness(&outcome.best_params).to_bits(),
        "{what}: best_fitness is the oracle's value for best_params"
    );
    assert_eq!(
        outcome.fitness_history.last().map(|x| x.to_bits()),
        Some(outcome.best_fitness.to_bits()),
        "{what}: the history ends at best_fitness"
    );
    assert!(
        outcome.fitness_history.windows(2).all(|w| w[1] >= w[0]),
        "{what}: the history never falls: {:?}",
        outcome.fitness_history
    );
}

#[test]
fn an_observer_that_never_stops_changes_nothing() {
    for kind in KINDS {
        for seed in SEEDS {
            let what = format!("{kind:?}, seed {seed}");
            let (oracle, plain_batches) = recorder(rough);
            let plain = run(kind, oracle, seed, false, None);

            let (oracle, unobserved_batches) = recorder(rough);
            let unobserved = run(kind, oracle, seed, false, Some(&mut Unobserved));
            assert_same(&plain.outcome, &unobserved.outcome, &what);

            let (oracle, watched_batches) = recorder(rough);
            let mut watcher = Watcher::new(&watched_batches, None);
            let watched = run(kind, oracle, seed, false, Some(&mut watcher));
            assert_same(&plain.outcome, &watched.outcome, &what);

            // The same batches, in the same order, whoever watches.
            let plain_batches = plain_batches.lock().unwrap().clone();
            assert_eq!(*unobserved_batches.lock().unwrap(), plain_batches, "{what}");
            assert_eq!(*watched_batches.lock().unwrap(), plain_batches, "{what}");
            assert_eq!((plain.asks, plain.tells), (watched.asks, watched.tells));
        }
    }
}

#[test]
fn a_closure_is_an_observer() {
    let (oracle, _) = recorder(rough);
    let mut iterations = Vec::new();
    let mut observer = |report: &IterationReport<'_>| {
        iterations.push(report.iteration);
        ControlFlow::Continue(())
    };
    let outcome = run(Kind::Qng, oracle, 3, false, Some(&mut observer)).outcome;
    assert_eq!(iterations, (1..=outcome.iterations_run).collect::<Vec<_>>());
}

#[test]
fn the_observer_sees_every_iteration_once_with_its_history_entry() {
    for kind in KINDS {
        for seed in SEEDS {
            let what = format!("{kind:?}, seed {seed}");
            let (oracle, batches) = recorder(rough);
            let mut watcher = Watcher::new(&batches, None);
            let outcome = run(kind, oracle, seed, false, Some(&mut watcher)).outcome;
            let seen = watcher.seen;

            assert!(outcome.iterations_run > 0, "{what}");
            assert_eq!(
                seen.iter().map(|s| s.iteration).collect::<Vec<_>>(),
                (1..=outcome.iterations_run).collect::<Vec<_>>(),
                "{what}: one report per iteration, numbered from 1"
            );
            // Each report shows the entry its iteration leaves in the history,
            // the last one included: the final point of Adam, SPSA and
            // Rotosolve is measured before the last report.
            assert_eq!(
                bits(&seen.iter().map(|s| s.best_fitness).collect::<Vec<_>>()),
                bits(&outcome.fitness_history),
                "{what}"
            );
            for s in &seen {
                assert_eq!(
                    s.best_fitness.to_bits(),
                    rough(&s.best_params).to_bits(),
                    "{what}: iteration {}: the best shown is measured",
                    s.iteration
                );
            }
            let last = seen.last().expect("at least one report");
            assert_eq!(
                bits(&last.best_params),
                bits(&outcome.best_params),
                "{what}"
            );
            // Nothing is scored after the last report.
            assert_eq!(last.batches, batches.lock().unwrap().len(), "{what}");
        }
    }
}

#[test]
fn a_stop_returns_what_the_run_has_reached_and_calls_nothing_after() {
    for kind in KINDS {
        for seed in SEEDS {
            let (oracle, _) = recorder(rough);
            let full = run(kind, oracle, seed, false, None).outcome;
            assert!(
                full.iterations_run >= 3,
                "{kind:?}: the budget leaves room to stop"
            );
            // The first iteration, one in the middle, and the one before the last.
            for stop_at in [1, 2, full.iterations_run - 1] {
                let what = format!("{kind:?}, seed {seed}, stop at {stop_at}");
                let (oracle, batches) = recorder(rough);
                let mut watcher = Watcher::new(&batches, Some(stop_at));
                let stopped = run(kind, oracle, seed, false, Some(&mut watcher));
                let outcome = &stopped.outcome;

                assert_eq!(outcome.iterations_run, stop_at, "{what}");
                assert!(!outcome.converged, "{what}");
                assert_postconditions(outcome, rough, &what);
                assert_eq!(
                    watcher.seen.len(),
                    stop_at,
                    "{what}: no report after the stop"
                );

                // The outcome is what the last report showed.
                let last = watcher.seen.last().expect("a report");
                assert_eq!(
                    last.best_fitness.to_bits(),
                    outcome.best_fitness.to_bits(),
                    "{what}"
                );
                assert_eq!(
                    bits(&last.best_params),
                    bits(&outcome.best_params),
                    "{what}"
                );
                // No oracle call after the stop.
                assert_eq!(last.batches, batches.lock().unwrap().len(), "{what}");
                // Up to the stop, the run is the unwatched one.
                assert_eq!(
                    bits(&outcome.fitness_history),
                    bits(&full.fitness_history[..stop_at]),
                    "{what}"
                );
                if kind == Kind::External {
                    assert_eq!(
                        (stopped.asks, stopped.tells),
                        (stop_at, stop_at),
                        "{what}: no ask after the stop"
                    );
                }
            }
        }
    }
}

#[test]
fn a_stop_wins_over_the_optimizers_own_stop_on_the_same_iteration() {
    // External cannot meet its own stop on an iteration: it learns that the
    // optimizer is done from the next ask. QNG and SPSA have no stopping test.
    let converging = KINDS
        .into_iter()
        .filter(|kind| !matches!(kind, Kind::External | Kind::Qng | Kind::Spsa));
    for kind in converging {
        for seed in SEEDS {
            let (oracle, _) = recorder(flat);
            let unwatched = run(kind, oracle, seed, true, None).outcome;
            let what = format!("{kind:?}, seed {seed}");
            assert!(unwatched.converged, "{what}: the stopping test fires");
            let at = unwatched.iterations_run;

            let (oracle, batches) = recorder(flat);
            let mut watcher = Watcher::new(&batches, Some(at));
            let stopped = run(kind, oracle, seed, true, Some(&mut watcher)).outcome;
            assert!(!stopped.converged, "{what}: the stop wins");
            assert_eq!(stopped.iterations_run, at, "{what}");
            assert_postconditions(&stopped, flat, &what);
            assert_eq!(
                bits(&stopped.fitness_history),
                bits(&unwatched.fitness_history),
                "{what}"
            );
            assert_eq!(
                bits(&stopped.best_params),
                bits(&unwatched.best_params),
                "{what}"
            );
        }
    }
}

#[test]
fn cmaes_with_restarts_is_watched_across_its_runs() {
    // IPOP doubles the population on each restart, so more than one batch
    // size means the reports spanned several runs.
    for seed in SEEDS {
        let (oracle, batches) = recorder(rough);
        let mut watcher = Watcher::new(&batches, None);
        let outcome = run(Kind::CmaesRestarts, oracle, seed, false, Some(&mut watcher)).outcome;
        let mut sizes: Vec<usize> = batches.lock().unwrap().iter().map(Vec::len).collect();
        sizes.dedup();
        assert!(
            sizes.len() > 1,
            "seed {seed}: no restart happened: {sizes:?}"
        );
        assert_eq!(watcher.seen.len(), outcome.iterations_run, "seed {seed}");
    }
}
