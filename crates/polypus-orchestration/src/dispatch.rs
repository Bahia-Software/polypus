//! Optimizer dispatch — the single place that turns a chosen optimization
//! [`Method`] plus an [`EvaluationOracle`] into an [`OptimizationOutcome`].
//!
//! `train` and `qml_train` used to carry *identical* DE/PSO/QNG dispatch blocks
//! (build the optimizer args, release the GIL, run `optimize`, then surface
//! either the oracle's recorded failure or the optimizer's own config error).
//! That triplication now lives here once:
//! [`dispatch_optimizer`] takes a pyo3-free [`Method`] (the binding parses the
//! `polypus.DE`/`PSO`/`QNG`/`CMAES`/`Adam`/`SPSA`/`Rotosolve`/`External` object
//! into it and builds the QNG variance oracle and the external optimizer's
//! [`AskTell`]), runs the matching optimizer, and returns an [`OracleError`] the
//! binding converts back into a `PyErr`.
//!
//! This crate is deliberately Python-free — the GIL is released by the caller
//! (`py.detach(|| dispatch_optimizer(...))`) and re-acquired only inside
//! the oracle. So a real oracle failure reaches here **type-erased** as a
//! [`BoxedError`] held in the [`OracleErrorSlot`]: the concrete error (today an
//! `EvaluationError`, whose Python exception travels as a `DisplaySafePyErr`) is
//! boxed by the oracle and recovered by the `polypus` edge via `downcast` to
//! `EvaluationError`, keeping this crate free of `pyo3`.

use polypus_optimizers::finite_difference;
use polypus_optimizers::{
    AdamRule, AlgorithmAdam, AlgorithmAdamArgs, FiniteDifferenceGradient, GradientOracle,
};
use polypus_optimizers::{AlgorithmCmaEs, AlgorithmCmaEsArgs};
use polypus_optimizers::{
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, AlgorithmPSO,
    AlgorithmPSOArgs, AlgorithmQNG, AlgorithmQNGArgs, DeStrategy, EvaluationOracle,
    OptimizationOutcome, Optimizer, OptimizerError, VarianceOracle,
};
use polypus_optimizers::{AlgorithmExternal, AlgorithmExternalArgs, AskTell};
use polypus_optimizers::{AlgorithmRotosolve, AlgorithmRotosolveArgs};
use polypus_optimizers::{AlgorithmSPSA, AlgorithmSPSAArgs};
use polypus_optimizers::{IterationObserver, Unobserved};
use std::error::Error;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use crate::progress::{Counted, CountedGradient, TrainObserver, Watch};

/// DE configuration, mirroring the `polypus.DE` object minus the shared
/// execution parameters (`oracle`, `dimensions`, `seed`) dispatch supplies.
pub struct DeConfig {
    pub generations: u32,
    pub population_size: u32,
    pub tolerance: f64,
    pub patience: u32,
    /// How trial vectors are built (`polypus.DE(strategy=...)`);
    /// [`DeStrategy::Rand1Bin`] is the classic DE.
    pub strategy: DeStrategy,
    /// DE/rand/1/bin's `F`; `None` is the default (see
    /// [`AlgorithmDifferentialEvolutionArgs::mutation_factor`]).
    pub mutation_factor: Option<f64>,
    /// DE/rand/1/bin's `CR`; `None` is the default (see
    /// [`AlgorithmDifferentialEvolutionArgs::crossover_rate`]).
    pub crossover_rate: Option<f64>,
}

/// PSO configuration, mirroring the `polypus.PSO` object (see [`DeConfig`]).
pub struct PsoConfig {
    pub generations: u32,
    pub population_size: u32,
    pub bounds: (f64, f64),
    pub inertia_weight: f64,
    pub cognitive_weight: f64,
    pub social_weight: f64,
    pub tolerance: f64,
}

/// QNG configuration, mirroring the `polypus.QNG` object (see [`DeConfig`]). The
/// Python `variance_function` is adapted into a [`VarianceOracle`] by the binding
/// and travels alongside this config in [`Method::Qng`].
pub struct QngConfig {
    pub max_iters: u32,
    pub learning_rate: f64,
    pub finite_difference_step: f64,
    pub bounds: (f64, f64),
    pub tikhonov_reg: f64,
}

/// CMA-ES configuration, mirroring the `polypus.CMAES` object (see
/// [`DeConfig`]). `population_size: None` is Hansen's default λ for the run's
/// `dimensions`.
pub struct CmaesConfig {
    pub generations: u32,
    pub population_size: Option<u32>,
    pub sigma0: f64,
    pub tolerance: f64,
    pub patience: u32,
    pub restarts: u32,
}

/// Adam configuration, mirroring the `polypus.Adam` object (see [`DeConfig`]).
/// The gradient comes from the run's [`Oracles`]; when they carry no exact one,
/// dispatch takes a central finite difference of the evaluation oracle with
/// `finite_difference_step`, which scores θ in the same batch as its stencil.
pub struct AdamConfig {
    pub max_iters: u32,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub bounds: (f64, f64),
    pub tolerance: f64,
    pub patience: u32,
    pub finite_difference_step: f64,
    /// How the gradient becomes a step (`polypus.Adam(rule=...)`);
    /// [`AdamRule::Adam`] is Kingma and Ba's Adam.
    pub rule: AdamRule,
}

/// SPSA configuration, mirroring the `polypus.SPSA` object (see [`DeConfig`]).
/// The run starts with one calibration batch of 50 circuits, then makes one
/// batch of `2·resamplings` circuits per iteration, plus the parameters
/// themselves every `eval_interval` iterations.
pub struct SpsaConfig {
    pub max_iters: u32,
    pub resamplings: u32,
    pub perturbation: f64,
    pub target_step: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub stability_constant: f64,
    pub eval_interval: u32,
}

/// Rotosolve configuration, mirroring the `polypus.Rotosolve` object (see
/// [`DeConfig`]). Each step is one batch of two circuits, plus the parameters
/// themselves on the first step and every `reevaluate_every` steps; a sweep is
/// one step per parameter. The caller checks that the circuit qualifies
/// (every trainable parameter one single-Pauli rotation angle).
pub struct RotosolveConfig {
    pub sweeps: u32,
    pub reevaluate_every: u32,
    pub tolerance: f64,
    pub patience: u32,
}

/// External configuration, mirroring the `polypus.External` object (see
/// [`DeConfig`]). The external optimizer itself travels alongside it in
/// [`Method::External`]. Each of its `ask()` batches is one oracle batch; the
/// run stops at an empty `ask()` or before the batch that would take the
/// evaluations past `max_evaluations`. The library owns its seed and its
/// start, so dispatch passes it neither (contract C-7).
pub struct ExternalConfig {
    pub max_evaluations: u64,
}

/// The chosen optimizer and its configuration. `Qng` additionally carries its
/// variance oracle (built from the Python callback at the binding boundary), and
/// `External` the ask/tell optimizer it bridges (a Python object adapted by the
/// binding), so dispatch stays Python-free.
///
/// `#[non_exhaustive]`: every new optimizer adds a variant, which is then not
/// a breaking change; a match outside this crate needs a `_` arm (inside it
/// the matches stay exhaustive, so a new variant fails to build until each
/// one handles it).
#[non_exhaustive]
pub enum Method {
    De(DeConfig),
    Pso(PsoConfig),
    Qng(QngConfig, Box<dyn VarianceOracle>),
    Cmaes(CmaesConfig),
    Adam(AdamConfig),
    Spsa(SpsaConfig),
    Rotosolve(RotosolveConfig),
    External(ExternalConfig, Box<dyn AskTell>),
}

/// What a training run evaluates through: built by an
/// [`OracleFactory`](crate::OracleFactory) and consumed by [`dispatch_optimizer`].
///
/// - `evaluation` scores candidates; it is what every optimizer searches with.
/// - `gradient` is an exact fitness gradient, when the factory can compute one
///   (a parameter shift). Only Adam reads it; without it, Adam differentiates
///   `evaluation` by finite differences.
/// - `full_evaluation` is set when `evaluation` scores a minibatch: the same
///   model on every training sample. Adam tracks its best solution with it (the
///   minibatch only drives its gradient), and the other optimizers' reported
///   `best_fitness` is re-scored with it once they finish (contract C-5).
pub struct Oracles {
    pub evaluation: Box<dyn EvaluationOracle>,
    pub gradient: Option<Box<dyn GradientOracle>>,
    pub full_evaluation: Option<Box<dyn EvaluationOracle>>,
}

impl From<Box<dyn EvaluationOracle>> for Oracles {
    /// An evaluation oracle alone: no exact gradient, no minibatch.
    fn from(evaluation: Box<dyn EvaluationOracle>) -> Self {
        Oracles {
            evaluation,
            gradient: None,
            full_evaluation: None,
        }
    }
}

impl<O: EvaluationOracle + 'static> From<Box<O>> for Oracles {
    fn from(evaluation: Box<O>) -> Self {
        let evaluation: Box<dyn EvaluationOracle> = evaluation;
        evaluation.into()
    }
}

impl Method {
    /// Check every precondition the optimizer checks before any RNG draw or
    /// oracle call, in its order, without running anything: its configuration
    /// first, then `initial_point` (contract C-5). Each arm calls the
    /// optimizer's own `check_preconditions`, so the error is the one
    /// [`dispatch_optimizer`] would return as [`OracleError::Config`] for the
    /// same arguments. `train` and `qml.train` call it before any backend, and
    /// so any CUNQA allocation, exists.
    pub fn check_preconditions(
        &self,
        dimensions: u32,
        initial_point: Option<&[f64]>,
    ) -> Result<(), OptimizerError> {
        match self {
            Method::De(cfg) => AlgorithmDifferentialEvolution::check_preconditions(
                cfg.strategy,
                cfg.population_size,
                dimensions,
                cfg.mutation_factor,
                cfg.crossover_rate,
                cfg.patience,
                initial_point,
            ),
            Method::Pso(cfg) => {
                AlgorithmPSO::check_preconditions(dimensions, cfg.bounds, initial_point)
            }
            Method::Qng(cfg, _) => AlgorithmQNG::check_preconditions(
                cfg.max_iters,
                cfg.learning_rate,
                cfg.finite_difference_step,
                cfg.bounds,
                cfg.tikhonov_reg,
                dimensions,
                initial_point,
            ),
            Method::Cmaes(cfg) => AlgorithmCmaEs::check_preconditions(
                dimensions,
                cfg.population_size,
                cfg.generations,
                cfg.sigma0,
                cfg.restarts,
                cfg.patience,
                initial_point,
            ),
            Method::Spsa(cfg) => AlgorithmSPSA::check_preconditions(
                cfg.max_iters,
                cfg.resamplings,
                cfg.perturbation,
                cfg.target_step,
                cfg.alpha,
                cfg.gamma,
                cfg.stability_constant,
                cfg.eval_interval,
                dimensions,
                initial_point,
            ),
            Method::Rotosolve(cfg) => AlgorithmRotosolve::check_preconditions(
                cfg.sweeps,
                cfg.reevaluate_every,
                cfg.tolerance,
                cfg.patience,
                dimensions,
                initial_point,
            ),
            // The step first, as dispatch checks it: before it builds the
            // finite difference that every `train` and `qml.train` run uses,
            // and with an exact gradient too, so the two always agree.
            Method::Adam(cfg) => {
                finite_difference::check_step(cfg.finite_difference_step)?;
                AlgorithmAdam::check_preconditions(
                    cfg.max_iters,
                    cfg.learning_rate,
                    cfg.beta1,
                    cfg.beta2,
                    cfg.epsilon,
                    cfg.patience,
                    cfg.bounds,
                    dimensions,
                    initial_point,
                )
            }
            // The library chooses its own start, so after the configuration
            // any point is refused.
            Method::External(cfg, _) => {
                AlgorithmExternal::check_preconditions(cfg.max_evaluations, dimensions)?;
                refuse_external_initial_point(initial_point)
            }
        }
    }
}

/// A training run's failure, in the order the binding must surface it.
///
/// `Evaluation` (an oracle failure recorded mid-run in the shared slot) takes
/// precedence over `Config`, exactly as the previous inline dispatch did: an
/// oracle error is the *cause*, and the optimizer's own error is often just its
/// downstream symptom.
pub enum OracleError {
    /// The optimizer returned an error: it rejected its configuration before
    /// any evaluation (population too small, empty/non-finite bounds,
    /// `max_iters == 0`, ...), which surfaces as `polypus.ConfigurationError`,
    /// or an oracle returned the wrong number of values or a non-finite
    /// gradient ([`OptimizerError::OracleLengthMismatch`],
    /// [`OptimizerError::NonFiniteGradient`]), which surfaces as
    /// `polypus.EvaluationError`.
    Config(OptimizerError),
    /// An oracle evaluation failed and was recorded in the shared slot during the
    /// run (backend, binding, observable, a Python callback, or a C-5 violation).
    /// Type-erased as a [`BoxedError`] to keep this crate `pyo3`-free; the
    /// `polypus` edge downcasts it back to the concrete `EvaluationError` and
    /// re-raises it with its original Python class preserved.
    Evaluation(BoxedError),
}

/// A type-erased oracle failure. The concrete error — in practice an
/// `EvaluationError`, whose Python exception travels as a `DisplaySafePyErr` —
/// is boxed by the oracle so this crate never names `pyo3`; the `polypus` edge
/// downcasts it back to `EvaluationError` to re-raise the original Python
/// exception with its class preserved (plan §10.1), except as contract C-1
/// provides for failures of the circuit SDK behind the Python backends.
pub type BoxedError = Box<dyn Error + Send + 'static>;

/// Thread-safe holder for the first error an oracle hits during `optimize`.
///
/// The optimizer traits ([`EvaluationOracle`] and [`VarianceOracle`]) return
/// plain `f64`/`Vec<f64>` — a pure-crate contract this crate cannot change — so a
/// failure mid-optimization cannot be returned through the trait. Instead the
/// oracle records it here (type-erased) and yields a finite sentinel;
/// [`dispatch_optimizer`] takes it after `optimize` returns, and the `polypus`
/// edge downcasts and re-raises it (contract C-5 keeps oracle outputs finite
/// regardless). `Clone` shares one slot across an oracle's worker threads (the
/// QML oracle hands a clone to each `spawn_blocking` task).
#[derive(Clone, Default)]
pub struct OracleErrorSlot(Arc<Mutex<Option<BoxedError>>>);

impl OracleErrorSlot {
    /// A fresh, empty slot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `err` as the failure, keeping the *first* one recorded.
    ///
    /// `run_id` is the effective id of the run whose oracle failed, threaded in
    /// from the call site because this slot holds no run metadata of its own. The
    /// failure is logged at `error!` here, as it is recorded: from this point on
    /// the oracle only yields sentinel values, so without this record the log
    /// would go quiet until `optimize()` returns and the entry point raises.
    ///
    /// The error is formatted for that line *before* the mutex is taken. A boxed
    /// Python exception formats itself by attaching to the interpreter, while an
    /// oracle thread that holds the interpreter may be calling [`failed`](Self::failed);
    /// formatting under the mutex would make each wait for the other.
    pub fn record(&self, err: BoxedError, run_id: &str) {
        self.record_as(err, run_id, "oracle evaluation failed");
    }

    /// [`record`](Self::record), with `what` as the log line's account of the
    /// failure, for one that is not an oracle's: an external optimizer's
    /// `ask()` or `tell()` that raised, say.
    pub fn record_as(&self, err: BoxedError, run_id: &str, what: &str) {
        if self.failed() {
            return;
        }
        let message = log::log_enabled!(log::Level::Error).then(|| describe(&err));
        let stored = {
            let mut guard = self.0.lock().unwrap_or_else(|p| p.into_inner());
            let empty = guard.is_none();
            if empty {
                *guard = Some(err);
            }
            empty
        };
        // Log only when this call actually stored the error, so the message never
        // claims to have "recorded" a failure it silently dropped.
        if let (true, Some(message)) = (stored, message) {
            log::error!("run {run_id}: {what}: {message}");
        }
    }

    /// Whether a failure has been recorded (lets callers short-circuit further
    /// work once evaluation is doomed).
    pub fn failed(&self) -> bool {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).is_some()
    }

    /// Take the recorded failure, if any.
    pub fn take(&self) -> Option<BoxedError> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).take()
    }
}

/// Format `err` for a log line without panicking: `ToString::to_string` panics
/// when a `Display` impl returns an error, and `record` runs on oracle threads
/// where a panic would cross into the optimizer. A failed format falls back to a
/// fixed message.
fn describe(err: &BoxedError) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    match write!(out, "{err}") {
        Ok(()) => out,
        Err(std::fmt::Error) => "<unformattable error>".to_string(),
    }
}

/// Run `method` over `oracles` and return the outcome, or the run's failure.
///
/// The caller releases the GIL around this call; the optimizer loop is pure Rust
/// and the oracle re-acquires the GIL internally where it must. An oracle failure
/// recorded in `errors` during the run is returned in preference to the
/// optimizer's own error (see [`OracleError`]).
///
/// `initial_point` is handed to whichever optimizer `method` names (each
/// optimizer's `initial_point` field says how it starts from it); `None` starts
/// every optimizer as before. [`Method::External`] takes none, since its
/// library chooses its own start: a point is an
/// [`InvalidConfig`](OptimizerError::InvalidConfig) for `initial_point`, after
/// External's configuration is checked and before any `ask()`. It ignores
/// `seed` too (contract C-7: the library owns its seed).
pub fn dispatch_optimizer(
    method: Method,
    oracles: impl Into<Oracles>,
    dimensions: u32,
    errors: &OracleErrorSlot,
    seed: u64,
    initial_point: Option<Vec<f64>>,
) -> Result<OptimizationOutcome, OracleError> {
    dispatch_optimizer_observed(
        method,
        oracles,
        dimensions,
        errors,
        seed,
        initial_point,
        None,
    )
}

/// [`dispatch_optimizer`], showing `observer`, when there is one, the run at
/// the end of every iteration of the optimizer (see [`TrainObserver`]).
///
/// With an observer, the evaluation oracles (`evaluation` and
/// `full_evaluation`) are wrapped in a counter of the candidates they are
/// handed, and the exact gradient in one of the `θ`s it scores itself, which
/// the observer sees as
/// [`TrainProgress::evaluations`](crate::TrainProgress::evaluations), and the
/// run stops at the end of the first iteration after `errors` records a
/// failure, without calling the observer. A stop the observer asks for
/// returns the outcome the run has reached; a failure it records in `errors`
/// is the run's error, like an oracle's. `None` runs exactly as
/// [`dispatch_optimizer`] always has.
pub fn dispatch_optimizer_observed(
    method: Method,
    oracles: impl Into<Oracles>,
    dimensions: u32,
    errors: &OracleErrorSlot,
    seed: u64,
    initial_point: Option<Vec<f64>>,
    observer: Option<&mut dyn TrainObserver>,
) -> Result<OptimizationOutcome, OracleError> {
    let Oracles {
        evaluation: mut oracle,
        mut gradient,
        mut full_evaluation,
    } = oracles.into();
    let mut unobserved = Unobserved;
    let mut watch;
    let observer: &mut dyn IterationObserver = match observer {
        Some(observer) => {
            let evaluations = Arc::new(AtomicU64::new(0));
            oracle = Box::new(Counted::new(oracle, &evaluations));
            full_evaluation = full_evaluation.map(|full| -> Box<dyn EvaluationOracle> {
                Box::new(Counted::new(full, &evaluations))
            });
            gradient = gradient.map(|exact| -> Box<dyn GradientOracle> {
                Box::new(CountedGradient::new(exact, &evaluations))
            });
            watch = Watch {
                observer,
                evaluations,
                errors,
            };
            &mut watch
        }
        None => &mut unobserved,
    };
    let result = match method {
        // Adam tracks its best on the full set itself, so it takes the view and
        // its outcome is not re-scored below.
        Method::Adam(cfg) => dispatch_adam(
            cfg,
            oracle,
            gradient,
            full_evaluation.take(),
            dimensions,
            seed,
            initial_point,
            observer,
        ),
        Method::De(cfg) => AlgorithmDifferentialEvolution.optimize_with_strategy_observed(
            cfg.strategy,
            AlgorithmDifferentialEvolutionArgs {
                oracle,
                population_size: cfg.population_size,
                generations: cfg.generations,
                dimensions,
                tolerance: cfg.tolerance,
                patience: cfg.patience,
                seed: Some(seed),
                initial_point,
                mutation_factor: cfg.mutation_factor,
                crossover_rate: cfg.crossover_rate,
            },
            observer,
        ),
        Method::Pso(cfg) => AlgorithmPSO.optimize_observed(
            AlgorithmPSOArgs {
                oracle,
                population_size: cfg.population_size,
                generations: cfg.generations,
                dimensions,
                bounds: cfg.bounds,
                inertia_weight: cfg.inertia_weight,
                cognitive_weight: cfg.cognitive_weight,
                social_weight: cfg.social_weight,
                tolerance: cfg.tolerance,
                seed: Some(seed),
                initial_point,
            },
            observer,
        ),
        Method::Qng(cfg, variance_oracle) => AlgorithmQNG.optimize_observed(
            AlgorithmQNGArgs {
                oracle,
                max_iters: cfg.max_iters,
                learning_rate: cfg.learning_rate,
                finite_difference_step: cfg.finite_difference_step,
                bounds: cfg.bounds,
                dimensions,
                variance_oracle,
                tikhonov_reg: cfg.tikhonov_reg,
                seed: Some(seed),
                initial_point,
            },
            observer,
        ),
        Method::Cmaes(cfg) => AlgorithmCmaEs.optimize_observed(
            AlgorithmCmaEsArgs {
                oracle,
                population_size: cfg.population_size,
                generations: cfg.generations,
                dimensions,
                sigma0: cfg.sigma0,
                tolerance: cfg.tolerance,
                patience: cfg.patience,
                restarts: cfg.restarts,
                seed: Some(seed),
                initial_point,
            },
            observer,
        ),
        Method::Spsa(cfg) => AlgorithmSPSA.optimize_observed(
            AlgorithmSPSAArgs {
                oracle,
                max_iters: cfg.max_iters,
                resamplings: cfg.resamplings,
                perturbation: cfg.perturbation,
                target_step: cfg.target_step,
                alpha: cfg.alpha,
                gamma: cfg.gamma,
                stability_constant: cfg.stability_constant,
                eval_interval: cfg.eval_interval,
                dimensions,
                seed: Some(seed),
                initial_point,
            },
            observer,
        ),
        Method::Rotosolve(cfg) => AlgorithmRotosolve.optimize_observed(
            AlgorithmRotosolveArgs {
                oracle,
                sweeps: cfg.sweeps,
                reevaluate_every: cfg.reevaluate_every,
                tolerance: cfg.tolerance,
                patience: cfg.patience,
                dimensions,
                seed: Some(seed),
                initial_point,
            },
            observer,
        ),
        // As every optimizer: the configuration, then the point.
        Method::External(cfg, optimizer) => {
            AlgorithmExternal::check_preconditions(cfg.max_evaluations, dimensions)
                .and_then(|()| refuse_external_initial_point(initial_point.as_deref()))
                .and_then(|()| {
                    AlgorithmExternal.optimize_observed(
                        AlgorithmExternalArgs {
                            oracle,
                            optimizer,
                            max_evaluations: cfg.max_evaluations,
                            dimensions,
                        },
                        observer,
                    )
                })
        }
    };
    // With a minibatch a gradient-free optimizer steered by subset estimates,
    // so its `best_fitness` is one; re-score the returned parameters on every
    // sample (C-5). `fitness_history` keeps the estimates the search used.
    let result = match (result, full_evaluation) {
        (Ok(outcome), Some(full)) => rescore(outcome, full.as_ref()),
        (result, _) => result,
    };
    finish(result, errors)
}

/// [`Method::External`] takes no `initial_point`: any point is an
/// [`InvalidConfig`](OptimizerError::InvalidConfig) for it, whatever its
/// length or values.
fn refuse_external_initial_point(initial_point: Option<&[f64]>) -> Result<(), OptimizerError> {
    match initial_point {
        Some(_) => Err(OptimizerError::InvalidConfig {
            parameter: "initial_point",
            reason: "the external optimizer chooses its own start; give the starting point \
                     to the library (or to the adapter's initial_point) instead"
                .to_string(),
        }),
        None => Ok(()),
    }
}

/// Adam: the gradient is the exact one when the oracles carry it, otherwise a
/// finite difference of `evaluation`, which scores θ with its stencil in one
/// batch per iteration. When `evaluation` is a minibatch, the best solution is
/// tracked on every sample (`full_evaluation`), so its history and
/// `best_fitness` are full-dataset values.
fn dispatch_adam(
    cfg: AdamConfig,
    evaluation: Box<dyn EvaluationOracle>,
    gradient: Option<Box<dyn GradientOracle>>,
    full_evaluation: Option<Box<dyn EvaluationOracle>>,
    dimensions: u32,
    seed: u64,
    initial_point: Option<Vec<f64>>,
    observer: &mut dyn IterationObserver,
) -> Result<OptimizationOutcome, OptimizerError> {
    // Checked even when an exact gradient leaves it unused, so a bad step is
    // refused whatever oracles the run has, as `check_preconditions` refuses it.
    finite_difference::check_step(cfg.finite_difference_step)?;
    let (gradient_oracle, tracking): (Box<dyn GradientOracle>, Box<dyn EvaluationOracle>) =
        match (gradient, full_evaluation) {
            (Some(gradient), Some(full)) => (gradient, full),
            (Some(gradient), None) => (gradient, evaluation),
            (None, full) => {
                let evaluation: Arc<dyn EvaluationOracle> = Arc::from(evaluation);
                let difference = FiniteDifferenceGradient::new(
                    Arc::clone(&evaluation),
                    cfg.finite_difference_step,
                )?;
                match full {
                    // The stencil scores θ on the minibatch, and Adam must
                    // report its own oracle's value (C-5), so it scores θ on
                    // the full set and takes only the gradient from here.
                    Some(full) => (Box::new(GradientOnly(difference)), full),
                    // The finite difference and the tracking share one oracle:
                    // θ's value comes from the difference's batch, and Adam
                    // reports it as the run's fitness (C-5).
                    None => (Box::new(difference), Box::new(evaluation)),
                }
            }
        };
    AlgorithmAdam.optimize_observed(
        AlgorithmAdamArgs {
            oracle: tracking,
            gradient_oracle,
            max_iters: cfg.max_iters,
            learning_rate: cfg.learning_rate,
            beta1: cfg.beta1,
            beta2: cfg.beta2,
            epsilon: cfg.epsilon,
            bounds: cfg.bounds,
            dimensions,
            tolerance: cfg.tolerance,
            patience: cfg.patience,
            rule: cfg.rule,
            seed: Some(seed),
            initial_point,
        },
        observer,
    )
}

/// A gradient oracle without its [`GradientOracle::value_and_gradient`], so
/// Adam scores θ with its own oracle before asking for the gradient.
struct GradientOnly<G>(G);

impl<G: GradientOracle> GradientOracle for GradientOnly<G> {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        self.0.gradient(theta, param_index)
    }

    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        self.0.gradient_batch(theta, dims)
    }
}

/// `outcome` with `best_fitness` re-scored by `full` on `best_params`.
fn rescore(
    mut outcome: OptimizationOutcome,
    full: &dyn EvaluationOracle,
) -> Result<OptimizationOutcome, OptimizerError> {
    let scores = full.evaluate_batch(std::slice::from_ref(&outcome.best_params));
    if scores.len() != 1 {
        return Err(OptimizerError::OracleLengthMismatch {
            expected: 1,
            got: scores.len(),
        });
    }
    outcome.best_fitness = scores[0];
    Ok(outcome)
}

/// An oracle failure recorded during the run is the root cause; surface it
/// ahead of the optimizer's own (often downstream) error.
fn finish(
    result: Result<OptimizationOutcome, OptimizerError>,
    errors: &OracleErrorSlot,
) -> Result<OptimizationOutcome, OracleError> {
    if let Some(eval_err) = errors.take() {
        return Err(OracleError::Evaluation(eval_err));
    }
    result.map_err(OracleError::Config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_log::{capture_logs, logged_for};
    use std::fmt;

    // The slot is type-erased, so these exercise it with a trivial pure error —
    // no `EvaluationError`/pyo3 needed (this crate has neither). The behaviours
    // pinned are the ones the oracles rely on: first-error-wins, shared storage
    // across clones, and correctness under the QML oracle's thread contention.
    #[derive(Debug)]
    struct TestError(&'static str);
    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl Error for TestError {}

    fn boxed(msg: &'static str) -> BoxedError {
        Box::new(TestError(msg))
    }

    const RUN_ID: &str = "oracle-error-slot-test";

    #[test]
    fn new_and_default_slots_are_empty() {
        for slot in [OracleErrorSlot::new(), OracleErrorSlot::default()] {
            assert!(!slot.failed(), "a fresh slot must not report a failure");
            assert!(slot.take().is_none(), "a fresh slot must hold no error");
        }
    }

    #[test]
    fn record_marks_the_slot_as_failed() {
        let slot = OracleErrorSlot::new();
        slot.record(boxed("first"), RUN_ID);
        assert!(slot.failed(), "record() must make failed() true");
    }

    #[test]
    fn record_keeps_the_first_error() {
        let slot = OracleErrorSlot::new();
        slot.record(boxed("first"), RUN_ID);
        slot.record(boxed("second"), RUN_ID);
        let kept = slot.take().expect("an error was recorded");
        assert_eq!(
            kept.to_string(),
            "first",
            "the first recorded error must win"
        );
    }

    #[test]
    fn take_returns_the_error_and_clears_the_slot() {
        let slot = OracleErrorSlot::new();
        slot.record(boxed("boom"), RUN_ID);
        assert_eq!(
            slot.take().expect("an error was recorded").to_string(),
            "boom"
        );
        assert!(!slot.failed(), "take() must clear the slot");
        assert!(slot.take().is_none(), "a second take() must yield None");
    }

    #[test]
    fn clone_shares_the_same_slot() {
        // The QML oracle hands a clone to each worker thread; they must all see
        // (and write to) the same underlying slot.
        let slot = OracleErrorSlot::new();
        let handed_out = slot.clone();
        handed_out.record(boxed("x"), RUN_ID);
        assert!(slot.failed(), "a clone must share the original's storage");
    }

    /// Only the error that is actually stored is logged; a later one, dropped by
    /// first-error-wins, leaves no line claiming it was recorded.
    #[test]
    fn record_logs_only_the_error_it_stores() {
        let run_id = "oracle-error-slot-logs-only-the-stored-error";
        capture_logs();
        assert!(logged_for(run_id).is_empty());
        let slot = OracleErrorSlot::new();
        slot.record(boxed("first"), run_id);
        slot.record(boxed("second"), run_id);
        assert_eq!(
            logged_for(run_id),
            vec![format!("run {run_id}: oracle evaluation failed: first")]
        );
        assert_eq!(
            slot.take().expect("an error was recorded").to_string(),
            "first"
        );
    }

    /// `record_as` logs its own account of the failure, and keeps the first
    /// error like `record`.
    #[test]
    fn record_as_logs_its_own_wording() {
        let run_id = "oracle-error-slot-record-as";
        capture_logs();
        let slot = OracleErrorSlot::new();
        slot.record_as(
            boxed("no trials"),
            run_id,
            "the external optimizer's ask() failed",
        );
        slot.record(boxed("later"), run_id);
        assert_eq!(
            logged_for(run_id),
            vec![format!(
                "run {run_id}: the external optimizer's ask() failed: no trials"
            )]
        );
        assert_eq!(slot.take().expect("recorded").to_string(), "no trials");
    }

    /// An error whose `Display` cannot finish until another thread has called
    /// [`OracleErrorSlot::failed`]. Formatting it while holding the slot's mutex
    /// therefore deadlocks until the bounded wait below gives up.
    #[derive(Debug)]
    struct WaitsForFailedCall {
        formatting: std::sync::mpsc::Sender<()>,
        failed_called: std::sync::mpsc::Receiver<()>,
    }

    impl fmt::Display for WaitsForFailedCall {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let _ = self.formatting.send(());
            // Bounded, so a deadlocked run still ends instead of hanging the binary.
            let _ = self
                .failed_called
                .recv_timeout(std::time::Duration::from_secs(10));
            write!(f, "slow to format")
        }
    }

    impl Error for WaitsForFailedCall {}

    /// `record` formats the error (for its log line) without holding the slot's
    /// mutex. Formatting can block: a `DisplaySafePyErr` attaches to the
    /// interpreter to format itself, while an oracle thread that holds the
    /// interpreter calls `failed()`. Formatting under the mutex made each wait
    /// for the other. Here thread A records an error whose `Display` waits for
    /// thread B's `failed()` call; B must get its answer while A is still
    /// formatting.
    #[test]
    fn record_does_not_hold_the_slot_while_formatting_the_error() {
        use std::sync::mpsc;
        use std::time::Duration;

        let run_id = "oracle-error-slot-formats-outside-the-lock";
        // The format only happens when an `ERROR` line will be written.
        capture_logs();
        assert!(logged_for(run_id).is_empty());
        let slot = OracleErrorSlot::new();
        let (formatting_tx, formatting_rx) = mpsc::channel();
        let (failed_called_tx, failed_called_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let err = WaitsForFailedCall {
            formatting: formatting_tx,
            failed_called: failed_called_rx,
        };

        let recorder = {
            let slot = slot.clone();
            std::thread::spawn(move || slot.record(Box::new(err), run_id))
        };
        let checker = {
            let slot = slot.clone();
            std::thread::spawn(move || {
                formatting_rx
                    .recv_timeout(Duration::from_secs(10))
                    .expect("the recorder starts formatting the error");
                let failed = slot.failed();
                let _ = failed_called_tx.send(());
                let _ = done_tx.send(failed);
            })
        };

        let failed = done_rx.recv_timeout(Duration::from_secs(5)).expect(
            "failed() must not wait for record() to finish formatting the error \
             (the slot's mutex was held while formatting)",
        );
        assert!(!failed, "nothing is stored before the format completes");
        recorder.join().expect("the recorder does not panic");
        checker.join().expect("the checker does not panic");
        assert!(slot.failed(), "the error is stored once formatted");
        assert_eq!(
            logged_for(run_id),
            vec![format!(
                "run {run_id}: oracle evaluation failed: slow to format"
            )]
        );
    }

    #[test]
    fn concurrent_records_keep_exactly_one_error() {
        // First-error-wins must hold under contention, not just sequentially.
        let slot = OracleErrorSlot::new();
        let start = Arc::new(std::sync::Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let slot = slot.clone();
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    slot.record(boxed("race"), RUN_ID);
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("no worker may panic");
        }
        assert!(slot.failed(), "at least one writer must have recorded");
        assert!(slot.take().is_some(), "exactly one error survives");
        assert!(
            slot.take().is_none(),
            "only one error is ever stored, however many writers raced"
        );
    }

    /// Concave objective maximised at 1.0 that records every batch's size.
    struct RecordingQuadratic(Arc<Mutex<Vec<usize>>>);
    impl EvaluationOracle for RecordingQuadratic {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            self.0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(candidates.len());
            candidates
                .iter()
                .map(|c| -c.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>())
                .collect()
        }
    }

    fn de_method(strategy: DeStrategy) -> Method {
        Method::De(DeConfig {
            generations: 30,
            population_size: 12,
            tolerance: 0.0,
            patience: 100,
            strategy,
            mutation_factor: None,
            crossover_rate: None,
        })
    }

    /// Dispatch a DE `strategy` and return the outcome plus every batch size.
    fn dispatch_de(strategy: DeStrategy) -> (OptimizationOutcome, Vec<usize>) {
        let sizes = Arc::new(Mutex::new(Vec::new()));
        let oracle = Box::new(RecordingQuadratic(Arc::clone(&sizes)));
        let outcome = match dispatch_optimizer(
            de_method(strategy),
            oracle,
            3,
            &OracleErrorSlot::new(),
            9,
            None,
        ) {
            Ok(outcome) => outcome,
            Err(_) => panic!("a valid DE configuration must run"),
        };
        let sizes = sizes.lock().unwrap_or_else(|p| p.into_inner()).clone();
        (outcome, sizes)
    }

    /// The default strategy is the classic DE: dispatch returns exactly what
    /// `AlgorithmDifferentialEvolution::optimize` returns for the same seed.
    #[test]
    fn default_de_strategy_dispatches_the_classic_optimizer() {
        let (dispatched, _) = dispatch_de(DeStrategy::default());
        let direct = AlgorithmDifferentialEvolution
            .optimize(AlgorithmDifferentialEvolutionArgs {
                oracle: Box::new(RecordingQuadratic(Arc::default())),
                population_size: 12,
                generations: 30,
                dimensions: 3,
                tolerance: 0.0,
                patience: 100,
                seed: Some(9),
                initial_point: None,
                mutation_factor: None,
                crossover_rate: None,
            })
            .expect("a valid DE configuration runs");
        assert_eq!(dispatched, direct);
    }

    /// The configured strategy reaches the optimizer: L-SHADE's population, and
    /// so its batches, shrink; SHADE's do not.
    #[test]
    fn de_strategy_reaches_the_optimizer() {
        let (_, shade) = dispatch_de(DeStrategy::Shade);
        assert!(shade.iter().all(|&n| n == 12), "{shade:?}");
        let (outcome, lshade) = dispatch_de(DeStrategy::LShade);
        assert_eq!(lshade.len(), outcome.iterations_run + 1);
        assert_eq!(lshade[0], 12);
        assert!(*lshade.last().unwrap() < 12, "{lshade:?}");
    }

    /// Scores `-Σ (xᵢ − 1)²` and records the first candidate of the first batch.
    struct FirstCandidate(Arc<Mutex<Option<Vec<f64>>>>);
    impl EvaluationOracle for FirstCandidate {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            let mut first = self.0.lock().unwrap_or_else(|p| p.into_inner());
            if first.is_none() {
                *first = candidates.first().cloned();
            }
            candidates
                .iter()
                .map(|c| -c.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>())
                .collect()
        }
    }

    /// Constant QFIM diagonal for the QNG case.
    struct ConstVariance;
    impl VarianceOracle for ConstVariance {
        fn variance(&self, _theta: &[f64], _param_index: usize) -> f64 {
            1.0
        }
    }

    /// Dispatch hands `initial_point` to every `Method`: it is the first
    /// candidate of the first batch (DE, PSO, and Adam, whose batch is θ₀ and
    /// then its finite-difference stencil), the centre of the first gradient
    /// stencil (QNG, whose first candidate is θ₀ + h·e₀), the first mean
    /// (CMA-ES, whose candidates all sit on it with a tiny σ₀), the centre
    /// of the calibration (SPSA, whose first candidate is θ₀ + c·Δ₁) and the
    /// first point measured (Rotosolve, whose first batch is θ₀ and its two
    /// neighbours along the first parameter).
    #[test]
    fn initial_point_reaches_every_optimizer() {
        let point = vec![0.5, 2.0, 1.25];
        let step = 0.1;
        let mut methods = vec![
            (
                "PSO",
                Method::Pso(PsoConfig {
                    generations: 2,
                    population_size: 6,
                    bounds: (-std::f64::consts::PI, std::f64::consts::PI),
                    inertia_weight: 0.5,
                    cognitive_weight: 1.0,
                    social_weight: 1.0,
                    tolerance: 0.0,
                }),
            ),
            (
                "QNG",
                Method::Qng(
                    QngConfig {
                        max_iters: 2,
                        learning_rate: 0.1,
                        finite_difference_step: step,
                        bounds: (-std::f64::consts::PI, std::f64::consts::PI),
                        tikhonov_reg: 0.05,
                    },
                    Box::new(ConstVariance),
                ),
            ),
            (
                "CMA-ES",
                Method::Cmaes(CmaesConfig {
                    generations: 1,
                    population_size: None,
                    sigma0: 1e-9,
                    tolerance: 0.0,
                    patience: 10,
                    restarts: 0,
                }),
            ),
            (
                "Adam",
                Method::Adam(AdamConfig {
                    max_iters: 2,
                    learning_rate: 0.05,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    bounds: (-std::f64::consts::PI, std::f64::consts::PI),
                    tolerance: 0.0,
                    patience: 10,
                    finite_difference_step: step,
                    rule: AdamRule::Adam,
                }),
            ),
            ("SPSA", spsa_method(2, 1)),
            ("Rotosolve", rotosolve_method(2, 5)),
        ];
        for strategy in [DeStrategy::Rand1Bin, DeStrategy::Shade, DeStrategy::LShade] {
            methods.push((strategy.as_str(), de_method(strategy)));
        }
        for (name, method) in methods {
            let first = Arc::new(Mutex::new(None));
            let oracle = Box::new(FirstCandidate(Arc::clone(&first)));
            let errors = OracleErrorSlot::new();
            if dispatch_optimizer(method, oracle, 3, &errors, 5, Some(point.clone())).is_err() {
                panic!("{name}: a valid configuration must run");
            }
            let first = first
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
                .expect("the oracle was called");
            if name == "SPSA" {
                // The first calibration point, θ₀ + c·Δ₁ with Δ₁ of ±1s.
                for (x, p) in first.iter().zip(&point) {
                    assert!(
                        ((x - p).abs() - SPSA_PERTURBATION).abs() < 1e-12,
                        "{first:?}"
                    );
                }
                continue;
            }
            let expected = match name {
                "QNG" => vec![point[0] + step, point[1], point[2]],
                _ => point.clone(),
            };
            let tolerance = if name == "CMA-ES" { 1e-6 } else { 0.0 };
            assert_eq!(first.len(), expected.len(), "{name}");
            for (x, want) in first.iter().zip(&expected) {
                assert!(
                    (x - want).abs() <= tolerance,
                    "{name}: {first:?} vs {expected:?}"
                );
            }
        }
    }

    /// `Method::check_preconditions` returns exactly what dispatch returns as
    /// its configuration error, in the same order (the configuration before
    /// `initial_point`), and passes whatever dispatch runs.
    #[test]
    fn check_preconditions_agrees_with_dispatch() {
        use std::f64::consts::PI;
        let pso = |bounds| {
            Method::Pso(PsoConfig {
                generations: 2,
                population_size: 6,
                bounds,
                inertia_weight: 0.5,
                cognitive_weight: 1.0,
                social_weight: 1.0,
                tolerance: 0.0,
            })
        };
        let qng = |max_iters, bounds| {
            Method::Qng(
                QngConfig {
                    max_iters,
                    learning_rate: 0.1,
                    finite_difference_step: 0.1,
                    bounds,
                    tikhonov_reg: 0.05,
                },
                Box::new(ConstVariance),
            )
        };
        let cmaes = |sigma0| {
            Method::Cmaes(CmaesConfig {
                generations: 2,
                population_size: None,
                sigma0,
                tolerance: 0.0,
                patience: 10,
                restarts: 0,
            })
        };
        let de = |strategy, population_size| {
            Method::De(DeConfig {
                generations: 2,
                population_size,
                tolerance: 0.0,
                patience: 10,
                strategy,
                mutation_factor: None,
                crossover_rate: None,
            })
        };
        let short = Some(vec![0.5]);
        let outside = Some(vec![0.5, 4.0]);
        // A name, a way to build the method (it is not `Clone`) and a point.
        type Case = (&'static str, Box<dyn Fn() -> Method>, Option<Vec<f64>>);
        let mut cases: Vec<Case> = vec![
            (
                "PSO, empty bounds",
                Box::new(move || pso((1.0, 1.0))),
                short.clone(),
            ),
            (
                "PSO, short point",
                Box::new(move || pso((-PI, PI))),
                short.clone(),
            ),
            (
                "PSO, point outside",
                Box::new(move || pso((-PI, PI))),
                outside.clone(),
            ),
            (
                "PSO, valid",
                Box::new(move || pso((-PI, PI))),
                Some(vec![0.5, 1.0]),
            ),
            (
                "QNG, max_iters 0",
                Box::new(move || qng(0, (-PI, PI))),
                short.clone(),
            ),
            (
                "QNG, empty bounds",
                Box::new(move || qng(1, (1.0, 0.0))),
                short.clone(),
            ),
            (
                "QNG, short point",
                Box::new(move || qng(1, (-PI, PI))),
                short.clone(),
            ),
            (
                "QNG, point outside",
                Box::new(move || qng(1, (-PI, PI))),
                outside.clone(),
            ),
            (
                "CMA-ES, sigma0 0",
                Box::new(move || cmaes(0.0)),
                short.clone(),
            ),
            (
                "CMA-ES, NaN",
                Box::new(move || cmaes(0.3)),
                Some(vec![0.5, f64::NAN]),
            ),
            ("CMA-ES, no point", Box::new(move || cmaes(0.3)), None),
            (
                "CMA-ES, patience 0",
                Box::new(move || match cmaes(0.3) {
                    Method::Cmaes(cfg) => Method::Cmaes(CmaesConfig { patience: 0, ..cfg }),
                    other => other,
                }),
                short.clone(),
            ),
        ];
        let adam = |finite_difference_step, learning_rate| {
            Method::Adam(AdamConfig {
                max_iters: 2,
                learning_rate,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                bounds: (-PI, PI),
                tolerance: 0.0,
                patience: 10,
                finite_difference_step,
                rule: AdamRule::Adam,
            })
        };
        let adam_cases: Vec<Case> = vec![
            (
                "Adam, step 0",
                Box::new(move || adam(0.0, 0.0)),
                short.clone(),
            ),
            (
                "Adam, rate 0",
                Box::new(move || adam(0.1, 0.0)),
                short.clone(),
            ),
            (
                "Adam, patience 0",
                Box::new(move || match adam(0.1, 0.05) {
                    Method::Adam(cfg) => Method::Adam(AdamConfig { patience: 0, ..cfg }),
                    other => other,
                }),
                short.clone(),
            ),
            (
                "Adam, short point",
                Box::new(move || adam(0.1, 0.05)),
                short.clone(),
            ),
            (
                "Adam, point outside",
                Box::new(move || adam(0.1, 0.05)),
                outside.clone(),
            ),
        ];
        cases.extend(adam_cases);
        let spsa = |max_iters, alpha| {
            Method::Spsa(SpsaConfig {
                max_iters,
                resamplings: 1,
                perturbation: 0.1,
                target_step: 0.63,
                alpha,
                gamma: 0.101,
                stability_constant: 0.0,
                eval_interval: 1,
            })
        };
        let spsa_cases: Vec<Case> = vec![
            (
                "SPSA, max_iters 0",
                Box::new(move || spsa(0, 0.602)),
                short.clone(),
            ),
            (
                "SPSA, alpha NaN",
                Box::new(move || spsa(2, f64::NAN)),
                short.clone(),
            ),
            (
                "SPSA, short point",
                Box::new(move || spsa(2, 0.602)),
                short.clone(),
            ),
            (
                "SPSA, valid",
                Box::new(move || spsa(2, 0.602)),
                outside.clone(),
            ),
        ];
        cases.extend(spsa_cases);
        let rotosolve = |sweeps, tolerance| {
            Method::Rotosolve(RotosolveConfig {
                sweeps,
                reevaluate_every: 1,
                tolerance,
                patience: 1,
            })
        };
        let rotosolve_cases: Vec<Case> = vec![
            (
                "Rotosolve, sweeps 0",
                Box::new(move || rotosolve(0, 0.0)),
                short.clone(),
            ),
            (
                "Rotosolve, tolerance -1",
                Box::new(move || rotosolve(1, -1.0)),
                short.clone(),
            ),
            (
                "Rotosolve, short point",
                Box::new(move || rotosolve(1, 0.0)),
                short.clone(),
            ),
            (
                "Rotosolve, valid",
                Box::new(move || rotosolve(1, 0.0)),
                outside.clone(),
            ),
        ];
        cases.extend(rotosolve_cases);
        let external = |max_evaluations| {
            Method::External(
                ExternalConfig { max_evaluations },
                Box::new(Scripted::new(external_batches())),
            )
        };
        let external_cases: Vec<Case> = vec![
            (
                "External, max_evaluations 0",
                Box::new(move || external(0)),
                short.clone(),
            ),
            (
                "External, a point",
                Box::new(move || external(6)),
                outside.clone(),
            ),
            ("External, valid", Box::new(move || external(6)), None),
        ];
        cases.extend(external_cases);
        for strategy in [DeStrategy::Rand1Bin, DeStrategy::Shade, DeStrategy::LShade] {
            cases.push((
                "DE, population 2",
                Box::new(move || de(strategy, 2)),
                short.clone(),
            ));
            cases.push((
                "DE, patience 0",
                Box::new(move || match de(strategy, 6) {
                    Method::De(cfg) => Method::De(DeConfig { patience: 0, ..cfg }),
                    other => other,
                }),
                short.clone(),
            ));
            cases.push((
                "DE, short point",
                Box::new(move || de(strategy, 6)),
                short.clone(),
            ));
            cases.push((
                "DE, valid",
                Box::new(move || de(strategy, 6)),
                outside.clone(),
            ));
        }
        for (name, method, point) in cases {
            let checked = method().check_preconditions(2, point.as_deref());
            let oracle = Box::new(FirstCandidate(Arc::default()));
            let dispatched =
                dispatch_optimizer(method(), oracle, 2, &OracleErrorSlot::new(), 5, point);
            match (checked, dispatched) {
                (Ok(()), Ok(_)) => {}
                (Err(checked), Err(OracleError::Config(dispatched))) => {
                    assert_eq!(checked, dispatched, "{name}");
                }
                (checked, dispatched) => panic!(
                    "{name}: check_preconditions gave {checked:?}, dispatch {}",
                    match dispatched {
                        Ok(_) => "Ok".to_string(),
                        Err(OracleError::Config(e)) => format!("Config({e})"),
                        Err(OracleError::Evaluation(e)) => format!("Evaluation({e})"),
                    }
                ),
            }
        }
    }

    /// A point of the wrong length is the optimizer's configuration error.
    #[test]
    fn a_wrong_initial_point_is_a_config_error() {
        let oracle = Box::new(RecordingQuadratic(Arc::default()));
        let result = dispatch_optimizer(
            de_method(DeStrategy::Rand1Bin),
            oracle,
            3,
            &OracleErrorSlot::new(),
            5,
            Some(vec![0.5]),
        );
        assert!(matches!(
            result,
            Err(OracleError::Config(OptimizerError::InvalidConfig {
                parameter: "initial_point",
                ..
            }))
        ));
    }

    // ── Adam and the run's Oracles ────────────────────────────────────────────

    fn adam_method(max_iters: u32) -> Method {
        Method::Adam(AdamConfig {
            max_iters,
            learning_rate: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            bounds: (-2.0, 2.0),
            tolerance: 0.0,
            patience: 1_000,
            finite_difference_step: 1e-4,
            rule: AdamRule::Adam,
        })
    }

    /// `quadratic`'s exact gradient, counting its calls.
    struct QuadraticGradient(Arc<Mutex<usize>>);
    impl GradientOracle for QuadraticGradient {
        fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
            self.gradient_batch(theta, theta.len())[param_index]
        }
        fn gradient_batch(&self, theta: &[f64], _dims: usize) -> Vec<f64> {
            *self.0.lock().unwrap_or_else(|p| p.into_inner()) += 1;
            theta.iter().map(|x| -2.0 * (x - 1.0)).collect()
        }
    }

    fn recorder() -> (Box<RecordingQuadratic>, Arc<Mutex<Vec<usize>>>) {
        let sizes = Arc::new(Mutex::new(Vec::new()));
        (Box::new(RecordingQuadratic(Arc::clone(&sizes))), sizes)
    }

    fn sizes(log: &Arc<Mutex<Vec<usize>>>) -> Vec<usize> {
        log.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    fn assert_near_the_optimum(outcome: &OptimizationOutcome) {
        for x in &outcome.best_params {
            assert!((x - 1.0).abs() < 1e-2, "{:?}", outcome.best_params);
        }
    }

    #[test]
    fn adam_without_an_exact_gradient_differentiates_the_evaluation_oracle() {
        let (oracle, log) = recorder();
        let outcome = dispatch_optimizer(
            adam_method(300),
            oracle,
            3,
            &OracleErrorSlot::new(),
            1,
            None,
        )
        .unwrap_or_else(|_| panic!("a valid Adam run must succeed"));
        assert_near_the_optimum(&outcome);
        // Each iteration: one batch on the same oracle, θ and its
        // finite-difference stencil (2·dims + 1 candidates); then the final θ
        // alone.
        let sizes = sizes(&log);
        let mut expected = vec![7; outcome.iterations_run];
        expected.push(1);
        assert_eq!(sizes, expected);
    }

    #[test]
    fn adam_uses_the_exact_gradient_when_the_oracles_carry_one() {
        let (oracle, log) = recorder();
        let calls = Arc::new(Mutex::new(0));
        let oracles = Oracles {
            evaluation: oracle,
            gradient: Some(Box::new(QuadraticGradient(Arc::clone(&calls)))),
            full_evaluation: None,
        };
        let outcome = dispatch_optimizer(
            adam_method(300),
            oracles,
            3,
            &OracleErrorSlot::new(),
            1,
            None,
        )
        .unwrap_or_else(|_| panic!("a valid Adam run must succeed"));
        assert_near_the_optimum(&outcome);
        assert_eq!(*calls.lock().unwrap(), outcome.iterations_run);
        // No stencil: the evaluation oracle only scores θ, one candidate a call,
        // once per iteration and once more for the final θ.
        let sizes = sizes(&log);
        assert_eq!(sizes.len(), outcome.iterations_run + 1);
        assert!(sizes.iter().all(|&n| n == 1), "{sizes:?}");
    }

    /// With an exact gradient the step is never used, but a bad one is still
    /// refused, as `check_preconditions` refuses it, before any call.
    #[test]
    fn adam_with_an_exact_gradient_still_rejects_a_bad_step() {
        let (oracle, log) = recorder();
        let calls = Arc::new(Mutex::new(0));
        let oracles = Oracles {
            evaluation: oracle,
            gradient: Some(Box::new(QuadraticGradient(Arc::clone(&calls)))),
            full_evaluation: None,
        };
        let method = match adam_method(10) {
            Method::Adam(cfg) => Method::Adam(AdamConfig {
                finite_difference_step: f64::NAN,
                ..cfg
            }),
            other => other,
        };
        let checked = method
            .check_preconditions(3, None)
            .expect_err("check_preconditions refuses the step");
        match dispatch_optimizer(method, oracles, 3, &OracleErrorSlot::new(), 1, None) {
            Err(OracleError::Config(dispatched)) => {
                assert_eq!(dispatched.to_string(), checked.to_string())
            }
            _ => panic!("expected the step's InvalidConfig"),
        }
        assert!(sizes(&log).is_empty());
        assert_eq!(*calls.lock().unwrap(), 0);
    }

    #[test]
    fn a_minibatched_adam_tracks_its_best_on_the_full_dataset() {
        let (minibatch, minibatch_log) = recorder();
        let (full, full_log) = recorder();
        let oracles = Oracles {
            evaluation: minibatch,
            gradient: None,
            full_evaluation: Some(full),
        };
        let outcome = dispatch_optimizer(
            adam_method(50),
            oracles,
            2,
            &OracleErrorSlot::new(),
            4,
            None,
        )
        .unwrap_or_else(|_| panic!("a valid Adam run must succeed"));
        // The minibatch oracle only feeds the gradient, a bare stencil with no
        // θ in front; θ is scored on the full set, once per iteration, and so is
        // the final θ, and nothing is re-scored after it.
        let minibatch_sizes = sizes(&minibatch_log);
        assert_eq!(minibatch_sizes.len(), outcome.iterations_run);
        assert!(
            minibatch_sizes.iter().all(|&n| n == 4),
            "{minibatch_sizes:?}"
        );
        assert_eq!(sizes(&full_log), vec![1; outcome.iterations_run + 1]);
        assert_eq!(outcome.fitness_history.last(), Some(&outcome.best_fitness));
    }

    #[test]
    fn a_minibatched_adam_with_an_exact_gradient_tracks_on_the_full_dataset() {
        let (minibatch, minibatch_log) = recorder();
        let (full, full_log) = recorder();
        let calls = Arc::new(Mutex::new(0));
        let oracles = Oracles {
            evaluation: minibatch,
            gradient: Some(Box::new(QuadraticGradient(Arc::clone(&calls)))),
            full_evaluation: Some(full),
        };
        let outcome = dispatch_optimizer(
            adam_method(50),
            oracles,
            2,
            &OracleErrorSlot::new(),
            4,
            None,
        )
        .unwrap_or_else(|_| panic!("a valid Adam run must succeed"));
        assert!(sizes(&minibatch_log).is_empty());
        assert_eq!(sizes(&full_log), vec![1; outcome.iterations_run + 1]);
        assert_eq!(*calls.lock().unwrap(), outcome.iterations_run);
    }

    /// Scores every candidate `offset` above `RecordingQuadratic`, so a test can
    /// tell which oracle produced a fitness.
    struct Shifted(f64);
    impl EvaluationOracle for Shifted {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            candidates
                .iter()
                .map(|c| self.0 - c.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>())
                .collect()
        }
    }

    #[test]
    fn a_minibatched_population_run_reports_its_best_on_the_full_dataset() {
        let (minibatch, _) = recorder();
        let oracles = Oracles {
            evaluation: minibatch,
            gradient: None,
            full_evaluation: Some(Box::new(Shifted(100.0))),
        };
        let outcome = dispatch_optimizer(
            de_method(DeStrategy::default()),
            oracles,
            3,
            &OracleErrorSlot::new(),
            9,
            None,
        )
        .unwrap_or_else(|_| panic!("a valid DE run must succeed"));
        let full = Shifted(100.0).evaluate_batch(std::slice::from_ref(&outcome.best_params))[0];
        assert_eq!(outcome.best_fitness, full);
        // The history keeps the estimates the search steered by.
        assert!(outcome.fitness_history.iter().all(|&f| f <= 0.0));
    }

    /// The configured rule reaches the optimizer: dispatch runs what
    /// `AlgorithmAdam` runs with that rule over a finite difference of the
    /// same oracle, and the two rules run differently.
    #[test]
    fn adam_rule_reaches_the_optimizer() {
        let dispatched = |rule| {
            let (oracle, _) = recorder();
            let method = match adam_method(20) {
                Method::Adam(cfg) => Method::Adam(AdamConfig { rule, ..cfg }),
                other => other,
            };
            dispatch_optimizer(method, oracle, 3, &OracleErrorSlot::new(), 1, None)
                .unwrap_or_else(|_| panic!("a valid Adam run must succeed"))
        };
        let direct = |rule| {
            let (oracle, _) = recorder();
            let oracle: Arc<dyn EvaluationOracle> = Arc::new(*oracle);
            let gradient = FiniteDifferenceGradient::new(Arc::clone(&oracle), 1e-4)
                .expect("a finite, positive step is valid");
            AlgorithmAdam
                .optimize(AlgorithmAdamArgs {
                    oracle: Box::new(oracle),
                    gradient_oracle: Box::new(gradient),
                    max_iters: 20,
                    learning_rate: 0.05,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    bounds: (-2.0, 2.0),
                    dimensions: 3,
                    tolerance: 0.0,
                    patience: 1_000,
                    rule,
                    seed: Some(1),
                    initial_point: None,
                })
                .expect("a valid Adam configuration runs")
        };
        for rule in [AdamRule::Adam, AdamRule::Momentum] {
            assert_eq!(dispatched(rule), direct(rule), "{rule}");
        }
        assert_ne!(
            dispatched(AdamRule::Adam).best_params,
            dispatched(AdamRule::Momentum).best_params
        );
    }

    #[test]
    fn adam_rejects_a_bad_finite_difference_step_before_any_call() {
        let (oracle, log) = recorder();
        let method = match adam_method(10) {
            Method::Adam(cfg) => Method::Adam(AdamConfig {
                finite_difference_step: 0.0,
                ..cfg
            }),
            other => other,
        };
        match dispatch_optimizer(method, oracle, 3, &OracleErrorSlot::new(), 1, None) {
            Err(OracleError::Config(OptimizerError::InvalidConfig { parameter, .. })) => {
                assert_eq!(parameter, "finite_difference_step")
            }
            _ => panic!("expected an InvalidConfig for the step"),
        }
        assert!(sizes(&log).is_empty());
    }

    // ── SPSA ──────────────────────────────────────────────────────────────────

    const SPSA_PERTURBATION: f64 = 0.1;

    fn spsa_method(max_iters: u32, resamplings: u32) -> Method {
        Method::Spsa(SpsaConfig {
            max_iters,
            resamplings,
            perturbation: SPSA_PERTURBATION,
            target_step: 0.63,
            alpha: 0.602,
            gamma: 0.101,
            stability_constant: 0.0,
            eval_interval: 4,
        })
    }

    /// Dispatch hands every `SpsaConfig` field and the seed to SPSA: the
    /// outcome is the one `AlgorithmSPSA::optimize` returns for the same
    /// arguments, after a calibration batch of 50, one batch per iteration and
    /// the final θ.
    #[test]
    fn spsa_dispatch_runs_the_configured_optimizer() {
        let (oracle, log) = recorder();
        let dispatched = dispatch_optimizer(
            spsa_method(9, 3),
            oracle,
            3,
            &OracleErrorSlot::new(),
            4,
            None,
        )
        .unwrap_or_else(|_| panic!("a valid SPSA run must succeed"));
        // θ heads the batches of iterations 1, 5 and 9; the final θ is alone.
        assert_eq!(sizes(&log), [50, 7, 6, 6, 6, 7, 6, 6, 6, 7, 1]);
        let (oracle, _) = recorder();
        let direct = AlgorithmSPSA
            .optimize(AlgorithmSPSAArgs {
                oracle,
                max_iters: 9,
                resamplings: 3,
                perturbation: SPSA_PERTURBATION,
                target_step: 0.63,
                alpha: 0.602,
                gamma: 0.101,
                stability_constant: 0.0,
                eval_interval: 4,
                dimensions: 3,
                seed: Some(4),
                initial_point: None,
            })
            .expect("a valid SPSA configuration runs");
        assert_eq!(dispatched, direct);
    }

    #[test]
    fn spsa_rejects_an_invalid_configuration_before_any_batch() {
        let (oracle, log) = recorder();
        let method = match spsa_method(10, 1) {
            Method::Spsa(cfg) => Method::Spsa(SpsaConfig {
                eval_interval: 0,
                ..cfg
            }),
            other => other,
        };
        match dispatch_optimizer(method, oracle, 3, &OracleErrorSlot::new(), 1, None) {
            Err(OracleError::Config(OptimizerError::InvalidConfig { parameter, .. })) => {
                assert_eq!(parameter, "eval_interval")
            }
            _ => panic!("expected an InvalidConfig for eval_interval"),
        }
        assert!(sizes(&log).is_empty());
    }

    // ── Rotosolve ─────────────────────────────────────────────────────────────

    fn rotosolve_method(sweeps: u32, reevaluate_every: u32) -> Method {
        Method::Rotosolve(RotosolveConfig {
            sweeps,
            reevaluate_every,
            tolerance: 0.0,
            patience: 2,
        })
    }

    /// Dispatch hands every `RotosolveConfig` field and the seed to Rotosolve:
    /// the outcome is the one `AlgorithmRotosolve::optimize` returns for the
    /// same arguments, after one batch per step and the final point.
    #[test]
    fn rotosolve_dispatch_runs_the_configured_optimizer() {
        let (oracle, log) = recorder();
        let dispatched = dispatch_optimizer(
            rotosolve_method(3, 4),
            oracle,
            3,
            &OracleErrorSlot::new(),
            4,
            None,
        )
        .unwrap_or_else(|_| panic!("a valid Rotosolve run must succeed"));
        // θ heads the batches of steps 0, 4 and 8; the final point is alone.
        assert_eq!(sizes(&log), [3, 2, 2, 2, 3, 2, 2, 2, 3, 1]);
        let (oracle, _) = recorder();
        let direct = AlgorithmRotosolve
            .optimize(AlgorithmRotosolveArgs {
                oracle,
                sweeps: 3,
                reevaluate_every: 4,
                tolerance: 0.0,
                patience: 2,
                dimensions: 3,
                seed: Some(4),
                initial_point: None,
            })
            .expect("a valid Rotosolve configuration runs");
        assert_eq!(dispatched, direct);
    }

    #[test]
    fn rotosolve_rejects_an_invalid_configuration_before_any_batch() {
        let (oracle, log) = recorder();
        let method = Method::Rotosolve(RotosolveConfig {
            sweeps: 10,
            reevaluate_every: 0,
            tolerance: 0.01,
            patience: 2,
        });
        match dispatch_optimizer(method, oracle, 3, &OracleErrorSlot::new(), 1, None) {
            Err(OracleError::Config(OptimizerError::InvalidConfig { parameter, .. })) => {
                assert_eq!(parameter, "reevaluate_every")
            }
            _ => panic!("expected an InvalidConfig for reevaluate_every"),
        }
        assert!(sizes(&log).is_empty());
    }

    // ── External ──────────────────────────────────────────────────────────────

    /// An ask/tell optimizer that proposes `batches` in order (then nothing),
    /// counting its asks and keeping what it was told. With `fail`, its first
    /// `ask` records the cause in that slot, as the Python bridge does, and
    /// fails.
    struct Scripted {
        batches: std::collections::VecDeque<Vec<Vec<f64>>>,
        asks: Arc<Mutex<usize>>,
        told: Arc<Mutex<Vec<Vec<f64>>>>,
        fail: Option<OracleErrorSlot>,
    }

    impl Scripted {
        fn new(batches: Vec<Vec<Vec<f64>>>) -> Self {
            Scripted {
                batches: batches.into(),
                asks: Arc::default(),
                told: Arc::default(),
                fail: None,
            }
        }
    }

    impl AskTell for Scripted {
        fn ask(&mut self) -> Result<Option<Vec<Vec<f64>>>, polypus_optimizers::AskTellError> {
            *self.asks.lock().unwrap_or_else(|p| p.into_inner()) += 1;
            if let Some(slot) = &self.fail {
                slot.record(boxed("the library's own exception"), RUN_ID);
                return Err("ask() raised".into());
            }
            Ok(self.batches.pop_front())
        }

        fn tell(
            &mut self,
            _candidates: &[Vec<f64>],
            values: &[f64],
        ) -> Result<(), polypus_optimizers::AskTellError> {
            self.told
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(values.to_vec());
            Ok(())
        }
    }

    /// `Method` holds a `Box<dyn AskTell>` for `External`, and stays `Send +
    /// Sync` as it was before: a compile-time check.
    #[test]
    fn method_is_send_and_sync() {
        fn send_and_sync<T: Send + Sync>() {}
        send_and_sync::<Method>();
    }

    /// Three batches of two, three and one points in two dimensions.
    fn external_batches() -> Vec<Vec<Vec<f64>>> {
        vec![
            vec![vec![0.0, 0.0], vec![1.0, 0.5]],
            vec![vec![2.0, 2.0], vec![1.0, 1.0], vec![0.5, 1.5]],
            vec![vec![1.5, 1.0]],
        ]
    }

    /// Dispatch runs the bridge: one oracle batch per `ask()`, the values told
    /// back, the run ended by the empty fourth `ask()` (the budget of seven
    /// leaves room for it), and the outcome the bridge returns for the same
    /// arguments. The seed is the library's (C-7), so dispatch
    /// ignores it.
    #[test]
    fn external_dispatch_runs_the_configured_bridge() {
        let mut outcomes = Vec::new();
        for seed in [1, 77] {
            let (oracle, log) = recorder();
            let optimizer = Scripted::new(external_batches());
            let told = Arc::clone(&optimizer.told);
            let method =
                Method::External(ExternalConfig { max_evaluations: 7 }, Box::new(optimizer));
            let outcome =
                dispatch_optimizer(method, oracle, 2, &OracleErrorSlot::new(), seed, None)
                    .unwrap_or_else(|_| panic!("a valid External run must succeed"));
            assert_eq!(sizes(&log), [2, 3, 1]);
            assert_eq!(told.lock().unwrap_or_else(|p| p.into_inner()).len(), 3);
            assert!(outcome.converged, "the script ended with an empty ask");
            outcomes.push(outcome);
        }
        assert_eq!(outcomes[0], outcomes[1], "the seed must not matter");
        let (oracle, _) = recorder();
        let direct = AlgorithmExternal
            .optimize(AlgorithmExternalArgs {
                oracle,
                optimizer: Box::new(Scripted::new(external_batches())),
                max_evaluations: 7,
                dimensions: 2,
            })
            .expect("a valid External configuration runs");
        assert_eq!(outcomes[0], direct);
        assert_eq!(direct.best_params, vec![1.0, 1.0]);
    }

    /// The budget reaches the bridge: with five evaluations the third batch
    /// (which would make six) is neither evaluated nor told.
    #[test]
    fn external_dispatch_passes_the_budget() {
        let (oracle, log) = recorder();
        let optimizer = Scripted::new(external_batches());
        let told = Arc::clone(&optimizer.told);
        let method = Method::External(ExternalConfig { max_evaluations: 5 }, Box::new(optimizer));
        let outcome = dispatch_optimizer(method, oracle, 2, &OracleErrorSlot::new(), 3, None)
            .unwrap_or_else(|_| panic!("a valid External run must succeed"));
        assert_eq!(sizes(&log), [2, 3]);
        assert_eq!(told.lock().unwrap_or_else(|p| p.into_inner()).len(), 2);
        assert!(!outcome.converged, "the budget stopped the run");
        assert_eq!(outcome.iterations_run, 2);
    }

    /// An `initial_point` is a configuration error before the library is asked
    /// anything or the oracle called: the library chooses its own start. It
    /// comes after External's own configuration errors.
    #[test]
    fn external_rejects_an_initial_point_before_any_ask() {
        let (oracle, log) = recorder();
        let optimizer = Scripted::new(external_batches());
        let asks = Arc::clone(&optimizer.asks);
        let method = Method::External(ExternalConfig { max_evaluations: 6 }, Box::new(optimizer));
        match dispatch_optimizer(
            method,
            oracle,
            2,
            &OracleErrorSlot::new(),
            1,
            Some(vec![0.5, 0.5]),
        ) {
            Err(OracleError::Config(OptimizerError::InvalidConfig { parameter, reason })) => {
                assert_eq!(parameter, "initial_point");
                assert!(reason.contains("own start"), "{reason}");
            }
            _ => panic!("expected an InvalidConfig for initial_point"),
        }
        assert_eq!(*asks.lock().unwrap_or_else(|p| p.into_inner()), 0);
        assert!(sizes(&log).is_empty());

        let (oracle, _) = recorder();
        let method = Method::External(
            ExternalConfig { max_evaluations: 0 },
            Box::new(Scripted::new(external_batches())),
        );
        match dispatch_optimizer(
            method,
            oracle,
            2,
            &OracleErrorSlot::new(),
            1,
            Some(vec![0.5, 0.5]),
        ) {
            Err(OracleError::Config(OptimizerError::InvalidConfig { parameter, .. })) => {
                assert_eq!(parameter, "max_evaluations")
            }
            _ => panic!("expected an InvalidConfig for max_evaluations"),
        }
    }

    /// A failure the bridge recorded in the run's slot (the Python exception
    /// `ask()` raised) is what dispatch returns, not the bridge's own error.
    #[test]
    fn external_returns_the_failure_recorded_in_the_slot() {
        let (oracle, log) = recorder();
        let errors = OracleErrorSlot::new();
        let mut optimizer = Scripted::new(external_batches());
        optimizer.fail = Some(errors.clone());
        let method = Method::External(ExternalConfig { max_evaluations: 6 }, Box::new(optimizer));
        match dispatch_optimizer(method, oracle, 2, &errors, 1, None) {
            Err(OracleError::Evaluation(cause)) => {
                assert_eq!(cause.to_string(), "the library's own exception")
            }
            _ => panic!("expected the recorded failure"),
        }
        assert!(sizes(&log).is_empty());
        assert!(!errors.failed(), "dispatch takes the recorded failure");
    }

    /// Without a recorded failure, the bridge's own error is a configuration
    /// error (it reaches Python as `ValueError`).
    #[test]
    fn external_bridge_errors_are_config_errors() {
        let (oracle, _) = recorder();
        let method = Method::External(
            ExternalConfig { max_evaluations: 0 },
            Box::new(Scripted::new(external_batches())),
        );
        match dispatch_optimizer(method, oracle, 2, &OracleErrorSlot::new(), 1, None) {
            Err(OracleError::Config(OptimizerError::InvalidConfig { parameter, .. })) => {
                assert_eq!(parameter, "max_evaluations")
            }
            _ => panic!("expected an InvalidConfig for max_evaluations"),
        }
    }

    // ── Watching a run: the evaluation count and the failure slot ──────────────

    /// Dispatch `method` over `oracles` with an observer that never stops the
    /// run, and return the outcome and the `evaluations` of every report.
    fn watched(
        method: Method,
        oracles: Oracles,
        errors: &OracleErrorSlot,
    ) -> (Result<OptimizationOutcome, OracleError>, Vec<u64>) {
        let mut seen = Vec::new();
        let mut observer = |progress: &crate::TrainProgress<'_>| {
            seen.push(progress.evaluations);
            std::ops::ControlFlow::Continue(())
        };
        let result =
            dispatch_optimizer_observed(method, oracles, 3, errors, 5, None, Some(&mut observer));
        (result, seen)
    }

    #[test]
    fn de_evaluations_count_the_initial_population_and_each_generation() {
        for strategy in [DeStrategy::Rand1Bin, DeStrategy::Shade] {
            let (oracle, log) = recorder();
            let (result, seen) =
                watched(de_method(strategy), oracle.into(), &OracleErrorSlot::new());
            let outcome = result.unwrap_or_else(|_| panic!("the run succeeds"));
            // 12 members: the initial population, then one batch a generation.
            let expected: Vec<u64> = (1..=outcome.iterations_run as u64)
                .map(|g| 12 * (g + 1))
                .collect();
            assert_eq!(seen, expected, "{strategy}");
            assert_eq!(
                *seen.last().unwrap(),
                sizes(&log).iter().sum::<usize>() as u64,
                "{strategy}"
            );
        }
    }

    #[test]
    fn a_never_stopping_observer_leaves_the_outcome_unchanged() {
        for strategy in [DeStrategy::Rand1Bin, DeStrategy::Shade, DeStrategy::LShade] {
            let (oracle, _) = recorder();
            let plain = dispatch_optimizer(
                de_method(strategy),
                oracle,
                3,
                &OracleErrorSlot::new(),
                5,
                None,
            )
            .unwrap_or_else(|_| panic!("the run succeeds"));
            let (oracle, _) = recorder();
            let (watched, _) = watched(de_method(strategy), oracle.into(), &OracleErrorSlot::new());
            assert_eq!(
                watched.unwrap_or_else(|_| panic!("the run succeeds")),
                plain,
                "{strategy}"
            );
        }
        let (oracle, _) = recorder();
        let plain =
            dispatch_optimizer(adam_method(25), oracle, 3, &OracleErrorSlot::new(), 5, None)
                .unwrap_or_else(|_| panic!("the run succeeds"));
        let (oracle, _) = recorder();
        let (watched, _) = watched(adam_method(25), oracle.into(), &OracleErrorSlot::new());
        assert_eq!(
            watched.unwrap_or_else(|_| panic!("the run succeeds")),
            plain
        );
    }

    #[test]
    fn adam_with_a_finite_difference_counts_theta_and_its_stencil() {
        let (oracle, log) = recorder();
        let (result, seen) = watched(adam_method(10), oracle.into(), &OracleErrorSlot::new());
        let outcome = result.unwrap_or_else(|_| panic!("the run succeeds"));
        assert_eq!(outcome.iterations_run, 10);
        // θ and its central stencil, 2·3 + 1 candidates an iteration, and the
        // final θ, measured on the last iteration before its report.
        let mut expected: Vec<u64> = (1..=10).map(|i| 7 * i).collect();
        *expected.last_mut().unwrap() += 1;
        assert_eq!(seen, expected);
        assert_eq!(
            *seen.last().unwrap(),
            sizes(&log).iter().sum::<usize>() as u64
        );
    }

    #[test]
    fn adam_with_an_exact_gradient_counts_only_theta() {
        let (oracle, _) = recorder();
        let calls = Arc::new(Mutex::new(0));
        let oracles = Oracles {
            evaluation: oracle,
            gradient: Some(Box::new(QuadraticGradient(Arc::clone(&calls)))),
            full_evaluation: None,
        };
        let (result, seen) = watched(adam_method(10), oracles, &OracleErrorSlot::new());
        let outcome = result.unwrap_or_else(|_| panic!("the run succeeds"));
        // The gradient's own circuits are not counted: one θ an iteration, and
        // the final θ on the last.
        let mut expected: Vec<u64> = (1..=outcome.iterations_run as u64).collect();
        *expected.last_mut().unwrap() += 1;
        assert_eq!(seen, expected);
        assert_eq!(*calls.lock().unwrap(), outcome.iterations_run);
    }

    #[test]
    fn a_minibatched_adam_counts_the_minibatch_and_the_full_dataset() {
        let (minibatch, minibatch_log) = recorder();
        let (full, full_log) = recorder();
        let oracles = Oracles {
            evaluation: minibatch,
            gradient: None,
            full_evaluation: Some(full),
        };
        let (result, seen) = watched(adam_method(10), oracles, &OracleErrorSlot::new());
        let outcome = result.unwrap_or_else(|_| panic!("the run succeeds"));
        // θ on every sample, its 2·3 stencil on the minibatch, and the final θ
        // on every sample on the last iteration.
        let mut expected: Vec<u64> = (1..=outcome.iterations_run as u64).map(|i| 7 * i).collect();
        *expected.last_mut().unwrap() += 1;
        assert_eq!(seen, expected);
        let total =
            sizes(&minibatch_log).iter().sum::<usize>() + sizes(&full_log).iter().sum::<usize>();
        assert_eq!(*seen.last().unwrap(), total as u64);
    }

    /// `quadratic`'s exact gradient that also scores θ in its own batch, as a
    /// parameter shift does, counting the θs it scores.
    struct ScoringGradient(Arc<Mutex<usize>>);
    impl GradientOracle for ScoringGradient {
        fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
            -2.0 * (theta[param_index] - 1.0)
        }
        fn value_and_gradient(&self, theta: &[f64], dims: usize) -> Option<(f64, Vec<f64>)> {
            *self.0.lock().unwrap_or_else(|p| p.into_inner()) += 1;
            let value = -theta.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>();
            Some((value, self.gradient_batch(theta, dims)))
        }
        fn scores_theta(&self) -> bool {
            true
        }
    }

    #[test]
    fn adam_counts_the_theta_an_exact_gradient_scores_itself() {
        let (oracle, log) = recorder();
        let scored = Arc::new(Mutex::new(0));
        let oracles = Oracles {
            evaluation: oracle,
            gradient: Some(Box::new(ScoringGradient(Arc::clone(&scored)))),
            full_evaluation: None,
        };
        let (result, seen) = watched(adam_method(10), oracles, &OracleErrorSlot::new());
        let outcome = result.unwrap_or_else(|_| panic!("the run succeeds"));
        // One θ an iteration, scored by the gradient, and the final θ, scored
        // by the evaluation oracle on the last.
        let mut expected: Vec<u64> = (1..=outcome.iterations_run as u64).collect();
        *expected.last_mut().unwrap() += 1;
        assert_eq!(seen, expected);
        assert_eq!(*scored.lock().unwrap(), outcome.iterations_run);
        assert_eq!(sizes(&log), vec![1]);

        // The same run, unwatched, is unchanged.
        let (oracle, _) = recorder();
        let oracles = Oracles {
            evaluation: oracle,
            gradient: Some(Box::new(ScoringGradient(Arc::default()))),
            full_evaluation: None,
        };
        let plain = dispatch_optimizer(
            adam_method(10),
            oracles,
            3,
            &OracleErrorSlot::new(),
            5,
            None,
        )
        .unwrap_or_else(|_| panic!("the run succeeds"));
        assert_eq!(plain, outcome);
    }

    /// Records `Failed` in its slot from its `fail_at`-th call on.
    struct FailingFrom {
        errors: OracleErrorSlot,
        fail_at: usize,
        calls: Arc<Mutex<usize>>,
    }
    impl EvaluationOracle for FailingFrom {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            let mut calls = self.calls.lock().unwrap_or_else(|p| p.into_inner());
            if *calls >= self.fail_at {
                self.errors.record(boxed("backend"), "dispatch-test");
            }
            *calls += 1;
            vec![0.0; candidates.len()]
        }
    }

    #[test]
    fn a_failed_run_stops_without_showing_the_observer_another_iteration() {
        let errors = OracleErrorSlot::new();
        let calls = Arc::new(Mutex::new(0));
        let oracle = Box::new(FailingFrom {
            errors: errors.clone(),
            fail_at: 2,
            calls: Arc::clone(&calls),
        });
        let mut shown = Vec::new();
        let mut observer = |progress: &crate::TrainProgress<'_>| {
            shown.push(progress.iteration);
            std::ops::ControlFlow::Continue(())
        };
        let result = dispatch_optimizer_observed(
            de_method(DeStrategy::Rand1Bin),
            oracle,
            3,
            &errors,
            5,
            None,
            Some(&mut observer),
        );
        match result {
            Err(OracleError::Evaluation(cause)) => assert_eq!(cause.to_string(), "backend"),
            _ => panic!("expected the recorded failure"),
        }
        // The initial population and generation 1 succeed; generation 2 fails,
        // and the run ends there without showing it.
        assert_eq!(shown, vec![1]);
        assert_eq!(*calls.lock().unwrap(), 3);
    }

    #[test]
    fn an_error_the_observer_records_is_the_runs() {
        let (oracle, log) = recorder();
        let errors = OracleErrorSlot::new();
        let mut shown = 0;
        let mut observer = |_: &crate::TrainProgress<'_>| {
            shown += 1;
            if shown == 2 {
                errors.record(boxed("the callback raised"), "dispatch-test");
                return std::ops::ControlFlow::Break(());
            }
            std::ops::ControlFlow::Continue(())
        };
        let result = dispatch_optimizer_observed(
            de_method(DeStrategy::Rand1Bin),
            oracle,
            3,
            &errors,
            5,
            None,
            Some(&mut observer),
        );
        match result {
            Err(OracleError::Evaluation(cause)) => {
                assert_eq!(cause.to_string(), "the callback raised")
            }
            _ => panic!("expected the recorded failure"),
        }
        assert_eq!(shown, 2);
        // The initial population and two generations: nothing after the stop.
        assert_eq!(sizes(&log), vec![12, 12, 12]);
    }
}
