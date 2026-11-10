//! Why a run stopped: every optimizer reports the [`StopReason`] that ended
//! its loop (issue #431).
//!
//! Each optimizer runs three ways: with a budget that runs out before any
//! stopping test fires, with a stopping test that fires on a flat landscape
//! (where the optimizer has one), and with an observer that stops the first
//! iteration, which wins over a stopping test firing on the same iteration.

use polypus_optimizers::{
    AdamRule, AlgorithmAdam, AlgorithmAdamArgs, AlgorithmCmaEs, AlgorithmCmaEsArgs,
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, AlgorithmExternal,
    AlgorithmExternalArgs, AlgorithmPSO, AlgorithmPSOArgs, AlgorithmQNG, AlgorithmQNGArgs,
    AlgorithmRotosolve, AlgorithmRotosolveArgs, AlgorithmSPSA, AlgorithmSPSAArgs, AskTell,
    AskTellError, DeStrategy, EvaluationOracle, FiniteDifferenceGradient, IterationObserver,
    IterationReport, OptimizationOutcome, Optimizer, StopReason, VarianceOracle,
};
use std::f64::consts::{PI, TAU};
use std::ops::ControlFlow;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    De(DeStrategy),
    Pso,
    Qng,
    Cmaes,
    Adam,
    Spsa,
    Rotosolve,
    External,
}

const KINDS: [Kind; 10] = [
    Kind::De(DeStrategy::Rand1Bin),
    Kind::De(DeStrategy::Shade),
    Kind::De(DeStrategy::LShade),
    Kind::Pso,
    Kind::Qng,
    Kind::Cmaes,
    Kind::Adam,
    Kind::Spsa,
    Kind::Rotosolve,
    Kind::External,
];

/// How a run is set up.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    /// A rough landscape and stopping tests that cannot fire: the budget ends
    /// the run.
    Budget,
    /// A flat landscape and stopping tests that fire within a few iterations.
    Tolerance,
}

/// Many local optima, so no stopping test with a tiny tolerance fires.
fn rough(x: &[f64]) -> f64 {
    x.iter().map(|v| -(v - 1.0).powi(2) + (5.0 * v).sin()).sum()
}

fn flat(_: &[f64]) -> f64 {
    0.0
}

/// A smooth single peak at `xᵢ = 1`.
fn sphere(x: &[f64]) -> f64 {
    -x.iter().map(|v| (v - 1.0).powi(2)).sum::<f64>()
}

struct Scored(fn(&[f64]) -> f64);

impl EvaluationOracle for Scored {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates.iter().map(|c| (self.0)(c)).collect()
    }
}

struct ConstVariance;

impl VarianceOracle for ConstVariance {
    fn variance(&self, _theta: &[f64], _param_index: usize) -> f64 {
        1.0
    }
}

/// Batches of four points in `[0, 2π)³`; `None` after `left` asks.
struct RandomSearch {
    state: u64,
    left: usize,
}

impl AskTell for RandomSearch {
    fn ask(&mut self) -> Result<Option<Vec<Vec<f64>>>, AskTellError> {
        if self.left == 0 {
            return Ok(None);
        }
        self.left -= 1;
        let mut uniform = || {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (self.state >> 11) as f64 / (1u64 << 53) as f64 * TAU
        };
        Ok(Some(
            (0..4)
                .map(|_| (0..3).map(|_| uniform()).collect())
                .collect(),
        ))
    }

    fn tell(&mut self, _candidates: &[Vec<f64>], _values: &[f64]) -> Result<(), AskTellError> {
        Ok(())
    }
}

fn go<O: Optimizer>(
    optimizer: O,
    args: O::Args,
    observer: Option<&mut dyn IterationObserver>,
) -> OptimizationOutcome {
    match observer {
        Some(observer) => optimizer.optimize_observed(args, observer),
        None => optimizer.optimize(args),
    }
    .expect("valid arguments optimize successfully")
}

fn run(
    kind: Kind,
    mode: Mode,
    observer: Option<&mut dyn IterationObserver>,
) -> OptimizationOutcome {
    let converging = mode == Mode::Tolerance;
    let oracle: Box<dyn EvaluationOracle> = Box::new(Scored(if converging { flat } else { rough }));
    let (tolerance, patience) = if converging { (1e-9, 2) } else { (0.0, 1_000) };
    match kind {
        Kind::De(strategy) => {
            let args = AlgorithmDifferentialEvolutionArgs {
                oracle,
                population_size: 8,
                generations: if converging { 20 } else { 6 },
                dimensions: 3,
                tolerance,
                patience,
                seed: Some(7),
                initial_point: None,
                mutation_factor: None,
                crossover_rate: None,
            };
            match observer {
                Some(o) => AlgorithmDifferentialEvolution
                    .optimize_with_strategy_observed(strategy, args, o),
                None => AlgorithmDifferentialEvolution.optimize_with_strategy(strategy, args),
            }
            .expect("valid arguments optimize successfully")
        }
        Kind::Pso => go(
            AlgorithmPSO,
            AlgorithmPSOArgs {
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
                seed: Some(7),
                initial_point: None,
            },
            observer,
        ),
        Kind::Qng => go(
            AlgorithmQNG,
            AlgorithmQNGArgs {
                oracle,
                max_iters: 6,
                learning_rate: 0.1,
                finite_difference_step: 0.1,
                bounds: (-PI, PI),
                dimensions: 3,
                variance_oracle: Box::new(ConstVariance),
                tikhonov_reg: 0.05,
                seed: Some(7),
                initial_point: None,
            },
            observer,
        ),
        Kind::Cmaes => go(
            AlgorithmCmaEs,
            AlgorithmCmaEsArgs {
                oracle,
                population_size: None,
                generations: if converging { 40 } else { 6 },
                dimensions: 3,
                sigma0: AlgorithmCmaEs::DEFAULT_SIGMA0,
                tolerance,
                patience,
                restarts: 0,
                seed: Some(7),
                initial_point: None,
            },
            observer,
        ),
        Kind::Adam => {
            let oracle: Arc<dyn EvaluationOracle> = Arc::from(oracle);
            let gradient = FiniteDifferenceGradient::new(Arc::clone(&oracle), 0.1)
                .expect("a finite, positive step is valid");
            go(
                AlgorithmAdam,
                AlgorithmAdamArgs {
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
                    seed: Some(7),
                    initial_point: None,
                },
                observer,
            )
        }
        Kind::Spsa => go(
            AlgorithmSPSA,
            AlgorithmSPSAArgs {
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
                seed: Some(7),
                initial_point: None,
            },
            observer,
        ),
        Kind::Rotosolve => go(
            AlgorithmRotosolve,
            AlgorithmRotosolveArgs {
                oracle,
                sweeps: if converging { 5 } else { 2 },
                reevaluate_every: 2,
                tolerance: if converging { 1e-9 } else { 0.0 },
                patience: 1,
                dimensions: 3,
                seed: Some(7),
                initial_point: None,
            },
            observer,
        ),
        // Converging: the optimizer ends the run itself after three batches.
        // Budget: 12 evaluations, three batches of four, are spent first.
        Kind::External => go(
            AlgorithmExternal,
            AlgorithmExternalArgs {
                oracle,
                optimizer: Box::new(RandomSearch {
                    state: 7,
                    left: if converging { 3 } else { 1_000 },
                }),
                max_evaluations: if converging { 1_000 } else { 12 },
                dimensions: 3,
            },
            observer,
        ),
    }
}

/// The reason the budget gives each optimizer.
fn budget_reason(kind: Kind) -> StopReason {
    match kind {
        // L-SHADE's schedule is a budget of evaluations.
        Kind::De(DeStrategy::LShade) | Kind::External => StopReason::EvaluationBudget,
        _ => StopReason::IterationBudget,
    }
}

/// The reason the optimizer's own test gives on a flat landscape, or `None`
/// for one without a test.
fn tolerance_reason(kind: Kind) -> Option<StopReason> {
    match kind {
        Kind::Qng | Kind::Spsa => None,
        Kind::Pso => Some(StopReason::ParameterTolerance),
        Kind::External => Some(StopReason::OptimizerFinished),
        _ => Some(StopReason::FitnessTolerance),
    }
}

#[test]
fn an_exhausted_budget_is_reported() {
    for kind in KINDS {
        let outcome = run(kind, Mode::Budget, None);
        assert_eq!(outcome.stop_reason, budget_reason(kind), "{kind:?}");
        assert!(!outcome.converged, "{kind:?}");
    }
}

#[test]
fn an_own_stopping_test_is_reported() {
    for kind in KINDS {
        let Some(expected) = tolerance_reason(kind) else {
            continue;
        };
        let outcome = run(kind, Mode::Tolerance, None);
        assert_eq!(outcome.stop_reason, expected, "{kind:?}");
        assert!(outcome.converged, "{kind:?}");
    }
}

#[test]
fn an_observer_stop_is_reported_as_callback_and_wins_over_a_test() {
    for kind in KINDS {
        for mode in [Mode::Budget, Mode::Tolerance] {
            let mut stop = |_: &IterationReport<'_>| ControlFlow::Break(());
            let outcome = run(kind, mode, Some(&mut stop));
            assert_eq!(
                outcome.stop_reason,
                StopReason::Callback,
                "{kind:?} {mode:?}"
            );
            assert_eq!(outcome.iterations_run, 1, "{kind:?} {mode:?}");
        }
    }
}

#[test]
fn an_observer_stop_on_the_last_iteration_is_still_callback() {
    for kind in KINDS {
        let budget = run(kind, Mode::Budget, None).iterations_run;
        let mut stop_last = |report: &IterationReport<'_>| {
            if report.iteration == budget {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let outcome = run(kind, Mode::Budget, Some(&mut stop_last));
        assert_eq!(outcome.stop_reason, StopReason::Callback, "{kind:?}");
    }
}

#[test]
fn a_zero_generation_de_run_reports_its_budget() {
    let outcome = AlgorithmDifferentialEvolution
        .optimize(AlgorithmDifferentialEvolutionArgs {
            oracle: Box::new(Scored(rough)),
            population_size: 8,
            generations: 0,
            dimensions: 3,
            tolerance: 0.0,
            patience: 1,
            seed: Some(7),
            initial_point: None,
            mutation_factor: None,
            crossover_rate: None,
        })
        .expect("valid arguments optimize successfully");
    assert_eq!(outcome.iterations_run, 0);
    assert_eq!(outcome.stop_reason, StopReason::IterationBudget);
}

#[test]
fn an_external_batch_past_the_budget_is_an_evaluation_budget_stop() {
    // Two batches of four spend 8 of 10; the third would exceed it.
    let outcome = AlgorithmExternal
        .optimize(AlgorithmExternalArgs {
            oracle: Box::new(Scored(rough)),
            optimizer: Box::new(RandomSearch {
                state: 7,
                left: 1_000,
            }),
            max_evaluations: 10,
            dimensions: 3,
        })
        .expect("valid arguments optimize successfully");
    assert_eq!(outcome.iterations_run, 2);
    assert_eq!(outcome.stop_reason, StopReason::EvaluationBudget);
}

/// With no fitness tolerance, CMA-ES on a smooth peak runs until its search
/// distribution collapses (TolX), a degenerate stop.
#[test]
fn a_collapsed_cmaes_distribution_is_degenerate() {
    let outcome = AlgorithmCmaEs
        .optimize(AlgorithmCmaEsArgs {
            oracle: Box::new(Scored(sphere)),
            population_size: None,
            generations: 5_000,
            dimensions: 3,
            sigma0: AlgorithmCmaEs::DEFAULT_SIGMA0,
            tolerance: 0.0,
            patience: 1,
            restarts: 0,
            seed: Some(7),
            initial_point: Some(vec![1.0; 3]),
        })
        .expect("valid arguments optimize successfully");
    assert!(outcome.iterations_run < 5_000, "the budget ran out first");
    assert_eq!(outcome.stop_reason, StopReason::Degenerate);
    assert!(outcome.converged);
}

#[test]
fn stop_reasons_have_snake_case_names() {
    let names = [
        (StopReason::FitnessTolerance, "fitness_tolerance"),
        (StopReason::ParameterTolerance, "parameter_tolerance"),
        (StopReason::IterationBudget, "iteration_budget"),
        (StopReason::EvaluationBudget, "evaluation_budget"),
        (StopReason::Callback, "callback"),
        (StopReason::Degenerate, "degenerate"),
        (StopReason::OptimizerFinished, "optimizer_finished"),
        (StopReason::Walltime, "walltime"),
        (StopReason::Interrupted, "interrupted"),
        (StopReason::Failed, "failed"),
    ];
    for (reason, name) in names {
        assert_eq!(reason.as_str(), name);
        assert_eq!(reason.to_string(), name);
    }
}
