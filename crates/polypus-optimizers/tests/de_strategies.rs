//! Behavioural tests for the Differential Evolution strategies
//! ([`DeStrategy`]): the default DE/rand/1/bin is pinned to its pre-strategy
//! output, and the adaptive SHADE and L-SHADE are held to the same contract
//! (C-5) as the rest of the crate: validated preconditions, an oracle length
//! check after every batch, `best_fitness == f(best_params)`, one history entry
//! per generation ending at `best_fitness`, and the fitness-stagnation stop.
//!
//! Like `optimizers.rs`, these run with no Python interpreter.

use polypus_optimizers::{
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, DeStrategy,
    EvaluationOracle, OptimizationOutcome, Optimizer, OptimizerError,
};
use std::f64::consts::{PI, TAU};
use std::sync::{Arc, Mutex};

const ADAPTIVE: [DeStrategy; 2] = [DeStrategy::Shade, DeStrategy::LShade];

/// Concave objective `-Σ(xᵢ - target)²`, maximised (value 0) at `xᵢ = target`.
/// Written with plain products so its value is bit-identical on every
/// IEEE-754 platform, which the pinned DE/rand/1/bin outcome below relies on.
struct Quadratic {
    target: f64,
}

impl EvaluationOracle for Quadratic {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates
            .iter()
            .map(|c| {
                -c.iter()
                    .map(|x| {
                        let d = x - self.target;
                        d * d
                    })
                    .sum::<f64>()
            })
            .collect()
    }
}

/// The quadratic bowl overlaid with a sinusoidal ripple from `optimizers.rs`:
/// many local optima and frequent near-ties, where a stale champion would
/// break `best_fitness == f(best_params)`.
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

/// Negated Rastrigin centred at π, so the angles' range `[0, 2π)` holds six
/// local optima per coordinate around the global one: fitness 0 at `xᵢ = π`.
struct Rastrigin;

impl EvaluationOracle for Rastrigin {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates
            .iter()
            .map(|c| {
                -c.iter()
                    .map(|x| {
                        let z = x - PI;
                        10.0 + z * z - 10.0 * (TAU * z).cos()
                    })
                    .sum::<f64>()
            })
            .collect()
    }
}

/// Always `NaN`.
struct NanOracle;

impl EvaluationOracle for NanOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates.iter().map(|_| f64::NAN).collect()
    }
}

/// A mix of `NaN`, `±inf` and finite values, chosen by the first coordinate:
/// infinite gains, `NaN` in the ranking and ties at infinity all at once.
struct NonFiniteOracle;

impl EvaluationOracle for NonFiniteOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates
            .iter()
            .map(|c| match (c[0] * 4.0 / TAU) as usize {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                _ => -c[1],
            })
            .collect()
    }
}

/// Returns one value fewer than it was given, starting with call number
/// `from_call` (0 is the initial population's batch).
struct ShortOracle {
    from_call: usize,
    calls: Mutex<usize>,
}

impl ShortOracle {
    fn from_call(from_call: usize) -> Self {
        Self {
            from_call,
            calls: Mutex::new(0),
        }
    }
}

impl EvaluationOracle for ShortOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let mut calls = self.calls.lock().unwrap();
        let short = *calls >= self.from_call;
        *calls += 1;
        let len = if short {
            candidates.len() - 1
        } else {
            candidates.len()
        };
        vec![0.0; len]
    }
}

/// Wraps an oracle and records the size of every batch it is handed.
struct Recording<O> {
    inner: O,
    sizes: Arc<Mutex<Vec<usize>>>,
}

impl<O: EvaluationOracle> EvaluationOracle for Recording<O> {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        self.sizes.lock().unwrap().push(candidates.len());
        self.inner.evaluate_batch(candidates)
    }
}

fn args(
    oracle: Box<dyn EvaluationOracle>,
    population_size: u32,
    generations: u32,
    dimensions: u32,
    seed: u64,
) -> AlgorithmDifferentialEvolutionArgs {
    AlgorithmDifferentialEvolutionArgs {
        oracle,
        population_size,
        generations,
        dimensions,
        tolerance: 1e-12,
        patience: 1_000,
        seed: Some(seed),
    }
}

fn run(
    strategy: DeStrategy,
    args: AlgorithmDifferentialEvolutionArgs,
) -> Result<OptimizationOutcome, OptimizerError> {
    AlgorithmDifferentialEvolution.optimize_with_strategy(strategy, args)
}

/// Run `strategy` over a recording [`Quadratic`] and return the outcome and
/// the size of every batch, the initial population's first.
fn batch_sizes(
    strategy: DeStrategy,
    population_size: u32,
    generations: u32,
    patience: u32,
) -> (OptimizationOutcome, Vec<usize>) {
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let oracle = Recording {
        inner: Quadratic { target: 1.0 },
        sizes: Arc::clone(&sizes),
    };
    let outcome = run(
        strategy,
        AlgorithmDifferentialEvolutionArgs {
            patience,
            tolerance: 1e-3,
            ..args(Box::new(oracle), population_size, generations, 3, 11)
        },
    )
    .expect("valid arguments optimize successfully");
    let sizes = sizes.lock().unwrap().clone();
    (outcome, sizes)
}

/// The L-SHADE schedule as documented, computed independently of the crate:
/// after each generation the size is `round(N + (4 - N) · evaluations /
/// (N · (generations + 1)))`, floored at 4, and the population never grows.
fn documented_lshade_sizes(population_size: usize, generations: usize) -> Vec<usize> {
    let budget = (population_size * (generations + 1)) as f64;
    let mut sizes = vec![population_size];
    let mut current = population_size;
    let mut evaluations = population_size;
    for _ in 0..generations {
        sizes.push(current);
        evaluations += current;
        let spent = evaluations as f64 / budget;
        let target =
            (population_size as f64 + (4.0 - population_size as f64) * spent).round() as usize;
        current = current.min(target.max(4));
    }
    sizes
}

/// Re-evaluate a fresh instance of the (deterministic) oracle at
/// `best_params` and require exactly `best_fitness`, and check the history's
/// shape: one entry per generation, never decreasing, ending at
/// `best_fitness`.
fn assert_c5(label: &str, outcome: &OptimizationOutcome, recheck: &dyn EvaluationOracle) {
    let recomputed = recheck.evaluate_batch(std::slice::from_ref(&outcome.best_params))[0];
    assert_eq!(
        recomputed, outcome.best_fitness,
        "{label}: f(best_params) = {recomputed} != best_fitness = {}",
        outcome.best_fitness
    );
    assert_eq!(
        outcome.fitness_history.len(),
        outcome.iterations_run,
        "{label}"
    );
    match outcome.fitness_history.last() {
        Some(last) => assert_eq!(last.to_bits(), outcome.best_fitness.to_bits(), "{label}"),
        None => assert_eq!(outcome.iterations_run, 0, "{label}"),
    }
    assert!(
        outcome.fitness_history.windows(2).all(|w| w[0] <= w[1]),
        "{label}: the best fitness fell: {:?}",
        outcome.fitness_history
    );
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// The default strategy is the DE/rand/1/bin that predates `DeStrategy`
// ─────────────────────────────────────────────────────────────────────────────

/// Pins DE/rand/1/bin's seeded outcome bit for bit, through both entry points.
/// The values were captured from `AlgorithmDifferentialEvolution::optimize`
/// before the strategies were added, so this fails if the default strategy's
/// draws, arithmetic or stopping rule change at all. Every operation involved
/// (the uniform draws, `F = 0.8`, `rem_euclid`, the [`Quadratic`] oracle) is
/// IEEE-754 basic arithmetic, so the values hold on every platform; a `rand`
/// release that changed the `StdRng` stream would also fail here, as it does
/// in `rng.rs`'s pinned-stream test.
#[test]
fn rand1bin_default_matches_its_pinned_outcome() {
    struct Pinned {
        population_size: u32,
        generations: u32,
        dimensions: u32,
        tolerance: f64,
        patience: u32,
        seed: u64,
        best_params: &'static [u64],
        best_fitness: u64,
        iterations_run: usize,
        converged: bool,
        fitness_history: &'static [u64],
    }
    let cases = [
        // The full budget: no early stop.
        Pinned {
            population_size: 8,
            generations: 6,
            dimensions: 3,
            tolerance: 1e-12,
            patience: 20,
            seed: 42,
            best_params: &[
                4607612620868739072,
                4602059860356609501,
                4604351434418528315,
            ],
            best_fitness: 13824131431062134976,
            iterations_run: 6,
            converged: false,
            fitness_history: &[
                13838482420622740690,
                13838482420622740690,
                13836545766831140230,
                13836545766831140230,
                13824131431062134976,
                13824131431062134976,
            ],
        },
        // Stopped by fitness stagnation after 13 of 200 generations.
        Pinned {
            population_size: 10,
            generations: 200,
            dimensions: 2,
            tolerance: 1e-3,
            patience: 5,
            seed: 7,
            best_params: &[4607239762160668585, 4606961771196870020],
            best_fitness: 13783541473274576150,
            iterations_run: 13,
            converged: true,
            fitness_history: &[
                13824008821365558148,
                13817008167141364435,
                13817008167141364435,
                13817008167141364435,
                13817008167141364435,
                13807722849325017925,
                13807722849325017925,
                13783541473274576150,
                13783541473274576150,
                13783541473274576150,
                13783541473274576150,
                13783541473274576150,
                13783541473274576150,
            ],
        },
    ];
    for case in cases {
        let make = || AlgorithmDifferentialEvolutionArgs {
            oracle: Box::new(Quadratic { target: 1.0 }),
            population_size: case.population_size,
            generations: case.generations,
            dimensions: case.dimensions,
            tolerance: case.tolerance,
            patience: case.patience,
            seed: Some(case.seed),
        };
        let through_trait = AlgorithmDifferentialEvolution
            .optimize(make())
            .expect("valid DE args optimize successfully");
        let through_strategy = run(DeStrategy::default(), make()).expect("valid DE args");
        assert_eq!(DeStrategy::default(), DeStrategy::Rand1Bin);
        assert_eq!(through_trait, through_strategy);

        assert_eq!(
            bits(&through_trait.best_params),
            case.best_params,
            "seed {}",
            case.seed
        );
        assert_eq!(through_trait.best_fitness.to_bits(), case.best_fitness);
        assert_eq!(through_trait.iterations_run, case.iterations_run);
        assert_eq!(through_trait.converged, case.converged);
        assert_eq!(bits(&through_trait.fitness_history), case.fitness_history);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Strategy names
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn strategy_names_round_trip() {
    for (strategy, name) in [
        (DeStrategy::Rand1Bin, "rand1bin"),
        (DeStrategy::Shade, "shade"),
        (DeStrategy::LShade, "lshade"),
    ] {
        assert_eq!(strategy.as_str(), name);
        assert_eq!(strategy.to_string(), name);
        assert_eq!(name.parse::<DeStrategy>(), Ok(strategy));
    }
}

#[test]
fn unknown_strategy_names_are_invalid_config() {
    for name in ["", "LSHADE", "l-shade", "rand/1/bin", " shade", "jade"] {
        match name.parse::<DeStrategy>() {
            Err(err @ OptimizerError::InvalidConfig { parameter, .. }) => {
                assert_eq!(parameter, "strategy");
                assert_eq!(
                    err.to_string(),
                    format!(
                        "invalid strategy: must be \"rand1bin\", \"shade\" or \"lshade\", \
                         got {name:?}"
                    )
                );
            }
            other => panic!("{name:?} should be rejected, got {other:?}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convergence on known optima (deterministic via fixed seeds)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn adaptive_strategies_converge_on_the_sphere() {
    for strategy in ADAPTIVE {
        for seed in 0..5 {
            let outcome = run(
                strategy,
                args(Box::new(Quadratic { target: 1.0 }), 30, 200, 4, seed),
            )
            .expect("valid arguments optimize successfully");
            assert!(
                outcome.best_fitness > -1e-8,
                "{strategy} seed {seed}: fitness = {}",
                outcome.best_fitness
            );
            for x in &outcome.best_params {
                assert!((x - 1.0).abs() < 1e-3, "{strategy} seed {seed}: {x}");
            }
        }
    }
}

#[test]
fn adaptive_strategies_find_the_global_optimum_of_rastrigin() {
    // 6⁴ local optima in [0, 2π)⁴; every seed must reach the global one.
    for strategy in ADAPTIVE {
        for seed in 0..5 {
            let outcome = run(strategy, args(Box::new(Rastrigin), 40, 300, 4, seed))
                .expect("valid arguments optimize successfully");
            assert!(
                outcome.best_fitness > -1e-6,
                "{strategy} seed {seed}: fitness = {}",
                outcome.best_fitness
            );
            for x in &outcome.best_params {
                assert!((x - PI).abs() < 1e-3, "{strategy} seed {seed}: {x}");
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn adaptive_strategies_are_deterministic_for_a_fixed_seed() {
    for strategy in ADAPTIVE {
        let make = |seed| {
            run(
                strategy,
                args(Box::new(Multimodal { target: 0.7 }), 20, 50, 4, seed),
            )
            .expect("valid arguments optimize successfully")
        };
        assert_eq!(make(123), make(123), "{strategy}");
        assert_ne!(make(123), make(124), "{strategy}");
    }
}

#[test]
fn strategies_explore_differently_from_the_same_seed() {
    let make = |strategy| {
        run(
            strategy,
            args(Box::new(Multimodal { target: 0.7 }), 20, 30, 4, 5),
        )
        .expect("valid arguments optimize successfully")
    };
    let rand1bin = make(DeStrategy::Rand1Bin);
    let shade = make(DeStrategy::Shade);
    let lshade = make(DeStrategy::LShade);
    assert_ne!(rand1bin.best_params, shade.best_params);
    assert_ne!(shade.best_params, lshade.best_params);
}

// ─────────────────────────────────────────────────────────────────────────────
// C-5: best_fitness == f(best_params), and the history's shape
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn adaptive_strategies_best_params_fitness_invariant_holds_across_seeds() {
    for strategy in ADAPTIVE {
        for seed in 0..20 {
            let quad = run(
                strategy,
                args(Box::new(Quadratic { target: 1.0 }), 30, 120, 4, seed),
            )
            .expect("valid arguments optimize successfully");
            assert_c5(
                &format!("{strategy}/quadratic seed {seed}"),
                &quad,
                &Quadratic { target: 1.0 },
            );

            let multi = run(
                strategy,
                args(Box::new(Multimodal { target: 1.0 }), 30, 120, 4, seed),
            )
            .expect("valid arguments optimize successfully");
            assert_c5(
                &format!("{strategy}/multimodal seed {seed}"),
                &multi,
                &Multimodal { target: 1.0 },
            );
        }
    }
}

#[test]
fn adaptive_strategies_early_stop_on_fitness_stagnation() {
    let patience = 20u32;
    for strategy in ADAPTIVE {
        let outcome = run(
            strategy,
            AlgorithmDifferentialEvolutionArgs {
                tolerance: 1e-3,
                patience,
                ..args(Box::new(Quadratic { target: 1.0 }), 30, 500, 3, 7)
            },
        )
        .expect("valid arguments optimize successfully");
        assert!(outcome.converged, "{strategy}: expected a stagnation stop");
        assert!(
            outcome.iterations_run > patience as usize && outcome.iterations_run < 500,
            "{strategy}: iterations_run = {}",
            outcome.iterations_run
        );
        assert_c5(&strategy.to_string(), &outcome, &Quadratic { target: 1.0 });
    }
}

#[test]
fn adaptive_strategies_large_patience_never_stops_early() {
    for strategy in ADAPTIVE {
        let outcome = run(
            strategy,
            AlgorithmDifferentialEvolutionArgs {
                tolerance: 1e9,
                ..args(Box::new(Quadratic { target: 1.0 }), 30, 50, 3, 7)
            },
        )
        .expect("valid arguments optimize successfully");
        assert!(!outcome.converged, "{strategy}");
        assert_eq!(outcome.iterations_run, 50, "{strategy}");
        assert_eq!(outcome.fitness_history.len(), 50, "{strategy}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Population size: SHADE keeps it, L-SHADE shrinks it linearly towards 4
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn shade_and_rand1bin_keep_the_population_size() {
    for strategy in [DeStrategy::Rand1Bin, DeStrategy::Shade] {
        let (outcome, sizes) = batch_sizes(strategy, 12, 25, 1_000);
        assert_eq!(sizes.len(), outcome.iterations_run + 1, "{strategy}");
        assert!(sizes.iter().all(|&n| n == 12), "{strategy}: {sizes:?}");
    }
}

#[test]
fn lshade_shrinks_the_population_linearly() {
    let (population_size, generations) = (20, 40);
    let (outcome, sizes) = batch_sizes(DeStrategy::LShade, 20, 40, 1_000);
    assert_eq!(outcome.iterations_run, generations);
    // One batch per generation plus the initial population's, as for every
    // strategy: the batches shrink, their number does not.
    assert_eq!(sizes.len(), generations + 1);
    assert_eq!(sizes, documented_lshade_sizes(population_size, generations));
    assert!(sizes.windows(2).all(|w| w[1] <= w[0]), "{sizes:?}");
    assert!(*sizes.last().unwrap() < population_size, "{sizes:?}");
    // Never more than DE/rand/1/bin's budget.
    assert!(sizes.iter().sum::<usize>() <= population_size * (generations + 1));
}

#[test]
fn lshade_bottoms_out_at_four() {
    // With 6 members over 60 generations the schedule reaches 4 well before the
    // end, and stays there.
    let (_, sizes) = batch_sizes(DeStrategy::LShade, 6, 60, 1_000);
    assert_eq!(sizes, documented_lshade_sizes(6, 60));
    assert_eq!(*sizes.last().unwrap(), 4);
    assert!(sizes.iter().all(|&n| n >= 4), "{sizes:?}");
}

#[test]
fn lshade_with_the_minimum_population_never_shrinks() {
    let (_, sizes) = batch_sizes(DeStrategy::LShade, 4, 30, 1_000);
    assert_eq!(sizes, vec![4; 31]);
}

#[test]
fn lshade_early_stop_ends_with_one_batch_per_generation_run() {
    let (outcome, sizes) = batch_sizes(DeStrategy::LShade, 30, 500, 10);
    assert!(outcome.converged);
    assert_eq!(sizes.len(), outcome.iterations_run + 1);
    assert_eq!(sizes, documented_lshade_sizes(30, 500)[..sizes.len()]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Preconditions and degenerate inputs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn adaptive_strategies_reject_population_below_four_before_any_evaluation() {
    for strategy in ADAPTIVE {
        for population_size in 0..4u32 {
            let sizes = Arc::new(Mutex::new(Vec::new()));
            let oracle = Recording {
                inner: Quadratic { target: 1.0 },
                sizes: Arc::clone(&sizes),
            };
            let result = run(strategy, args(Box::new(oracle), population_size, 10, 2, 1));
            assert!(
                matches!(
                    result,
                    Err(OptimizerError::PopulationTooSmall { got, min: 4 })
                        if got == population_size as usize
                ),
                "{strategy} population_size {population_size}: {result:?}"
            );
            assert!(sizes.lock().unwrap().is_empty(), "{strategy}");
        }
    }
}

#[test]
fn adaptive_strategies_run_with_the_minimum_population() {
    for strategy in ADAPTIVE {
        let outcome = run(
            strategy,
            args(Box::new(Multimodal { target: 1.0 }), 4, 20, 2, 1),
        )
        .expect("population_size 4 is valid");
        assert_c5(&strategy.to_string(), &outcome, &Multimodal { target: 1.0 });
    }
}

#[test]
fn adaptive_strategies_handle_zero_dimensions() {
    for strategy in ADAPTIVE {
        let outcome = run(
            strategy,
            args(Box::new(Quadratic { target: 1.0 }), 10, 5, 0, 1),
        )
        .expect("valid arguments optimize successfully");
        assert!(outcome.best_params.is_empty(), "{strategy}");
        assert_eq!(outcome.iterations_run, 5, "{strategy}");
    }
}

#[test]
fn adaptive_strategies_handle_zero_generations() {
    for strategy in ADAPTIVE {
        let outcome = run(
            strategy,
            args(Box::new(Quadratic { target: 1.0 }), 10, 0, 3, 1),
        )
        .expect("valid arguments optimize successfully");
        assert_eq!(outcome.iterations_run, 0, "{strategy}");
        assert!(outcome.fitness_history.is_empty(), "{strategy}");
        assert_c5(&strategy.to_string(), &outcome, &Quadratic { target: 1.0 });
    }
}

#[test]
fn adaptive_strategies_nan_fitness_does_not_panic() {
    for strategy in ADAPTIVE {
        let outcome = run(strategy, args(Box::new(NanOracle), 6, 5, 2, 1))
            .expect("valid arguments optimize successfully");
        assert_eq!(outcome.best_params.len(), 2, "{strategy}");
        assert!(outcome.best_fitness.is_nan(), "{strategy}");
    }
}

#[test]
fn adaptive_strategies_survive_non_finite_fitness() {
    // Infinite gains must not poison the memory (an F that can never be drawn
    // positive would loop forever), and NaN must not break the ranking sort.
    for strategy in ADAPTIVE {
        for seed in 0..5 {
            let outcome = run(strategy, args(Box::new(NonFiniteOracle), 12, 30, 2, seed))
                .expect("valid arguments optimize successfully");
            assert_eq!(outcome.best_params.len(), 2, "{strategy}");
            assert_eq!(outcome.iterations_run, 30, "{strategy}");
        }
    }
}

#[test]
fn adaptive_strategies_short_oracle_returns_error_not_panic() {
    for strategy in ADAPTIVE {
        // Short on the initial population's batch…
        let result = run(
            strategy,
            args(Box::new(ShortOracle::from_call(0)), 6, 5, 2, 1),
        );
        assert!(
            matches!(
                result,
                Err(OptimizerError::OracleLengthMismatch {
                    expected: 6,
                    got: 5
                })
            ),
            "{strategy}: {result:?}"
        );
        // …and on a later generation's batch.
        let result = run(
            strategy,
            args(Box::new(ShortOracle::from_call(3)), 6, 5, 2, 1),
        );
        assert!(
            matches!(
                result,
                Err(OptimizerError::OracleLengthMismatch { got, expected }) if got + 1 == expected
            ),
            "{strategy}: {result:?}"
        );
    }
}

#[test]
fn lshade_checks_the_length_of_a_shrunken_batch() {
    // By call 30 L-SHADE has shrunk 20 members to 15: the check compares with
    // the batch actually sent, not the initial size.
    let call = 30;
    let shrunk = documented_lshade_sizes(20, 40)[call];
    assert!(shrunk < 20);
    let result = run(
        DeStrategy::LShade,
        args(Box::new(ShortOracle::from_call(call)), 20, 40, 3, 11),
    );
    assert_eq!(
        result,
        Err(OptimizerError::OracleLengthMismatch {
            expected: shrunk,
            got: shrunk - 1
        })
    );
}
