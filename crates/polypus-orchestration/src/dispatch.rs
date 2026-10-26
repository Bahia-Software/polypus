//! Optimizer dispatch — the single place that turns a chosen optimization
//! [`Method`] plus an [`EvaluationOracle`] into an [`OptimizationOutcome`].
//!
//! `train` and `qml_train` used to carry *identical* DE/PSO/QNG dispatch blocks
//! (build the optimizer args, release the GIL, run `optimize`, then surface
//! either the oracle's recorded failure or the optimizer's own config error).
//! That triplication now lives here once:
//! [`dispatch_optimizer`] takes a pyo3-free [`Method`] (the binding parses the
//! `polypus.DE`/`PSO`/`QNG`/`CMAES`/`Adam`/`SPSA` object into it and builds the QNG
//! variance oracle), runs the matching optimizer, and returns an
//! [`OracleError`] the binding converts back into a `PyErr`.
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
use polypus_optimizers::{AlgorithmSPSA, AlgorithmSPSAArgs};
use std::error::Error;
use std::sync::{Arc, Mutex};

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
    pub eval_interval: u32,
}

/// The chosen optimizer and its configuration. `Qng` additionally carries its
/// variance oracle (built from the Python callback at the binding boundary), so
/// dispatch stays Python-free.
pub enum Method {
    De(DeConfig),
    Pso(PsoConfig),
    Qng(QngConfig, Box<dyn VarianceOracle>),
    Cmaes(CmaesConfig),
    Adam(AdamConfig),
    Spsa(SpsaConfig),
}

/// What a training run evaluates through: built by an
/// [`OracleFactory`](crate::OracleFactory) and consumed by [`dispatch_optimizer`].
///
/// - `evaluation` scores candidates; it is what every optimizer searches with.
/// - `gradient` is an exact fitness gradient, when the factory can compute one
///   (a parameter shift). Only Adam reads it; without it, Adam differentiates
///   `evaluation` by finite differences.
pub struct Oracles {
    pub evaluation: Box<dyn EvaluationOracle>,
    pub gradient: Option<Box<dyn GradientOracle>>,
}

impl From<Box<dyn EvaluationOracle>> for Oracles {
    /// An evaluation oracle alone: no exact gradient.
    fn from(evaluation: Box<dyn EvaluationOracle>) -> Self {
        Oracles {
            evaluation,
            gradient: None,
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
                cfg.population_size,
                dimensions,
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
                initial_point,
            ),
            Method::Spsa(cfg) => AlgorithmSPSA::check_preconditions(
                cfg.max_iters,
                cfg.resamplings,
                cfg.perturbation,
                cfg.target_step,
                cfg.alpha,
                cfg.gamma,
                cfg.eval_interval,
                dimensions,
                initial_point,
            ),
            // The step first: dispatch builds the finite difference before it
            // runs Adam whenever the run's oracles carry no exact gradient,
            // which is every `train` and `qml.train` run. With an exact
            // gradient dispatch never reads the step.
            Method::Adam(cfg) => {
                finite_difference::check_step(cfg.finite_difference_step)?;
                AlgorithmAdam::check_preconditions(
                    cfg.max_iters,
                    cfg.learning_rate,
                    cfg.beta1,
                    cfg.beta2,
                    cfg.epsilon,
                    cfg.bounds,
                    dimensions,
                    initial_point,
                )
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
    /// The optimizer rejected its configuration before any evaluation
    /// (population too small, empty/non-finite bounds, `max_iters == 0`, ...).
    /// Surfaces as `ValueError`.
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
            log::error!("run {run_id}: oracle evaluation failed: {message}");
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
/// every optimizer as before.
pub fn dispatch_optimizer(
    method: Method,
    oracles: impl Into<Oracles>,
    dimensions: u32,
    errors: &OracleErrorSlot,
    seed: u64,
    initial_point: Option<Vec<f64>>,
) -> Result<OptimizationOutcome, OracleError> {
    let Oracles {
        evaluation: oracle,
        gradient,
    } = oracles.into();
    let result = match method {
        Method::Adam(cfg) => dispatch_adam(cfg, oracle, gradient, dimensions, seed, initial_point),
        Method::De(cfg) => AlgorithmDifferentialEvolution.optimize_with_strategy(
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
            },
        ),
        Method::Pso(cfg) => AlgorithmPSO.optimize(AlgorithmPSOArgs {
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
        }),
        Method::Qng(cfg, variance_oracle) => AlgorithmQNG.optimize(AlgorithmQNGArgs {
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
        }),
        Method::Cmaes(cfg) => AlgorithmCmaEs.optimize(AlgorithmCmaEsArgs {
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
        }),
        Method::Spsa(cfg) => AlgorithmSPSA.optimize(AlgorithmSPSAArgs {
            oracle,
            max_iters: cfg.max_iters,
            resamplings: cfg.resamplings,
            perturbation: cfg.perturbation,
            target_step: cfg.target_step,
            alpha: cfg.alpha,
            gamma: cfg.gamma,
            eval_interval: cfg.eval_interval,
            dimensions,
            seed: Some(seed),
            initial_point,
        }),
    };
    finish(result, errors)
}

/// Adam: the gradient is the exact one when the oracles carry it, otherwise a
/// finite difference of `evaluation`, which scores θ with its stencil in one
/// batch per iteration.
fn dispatch_adam(
    cfg: AdamConfig,
    evaluation: Box<dyn EvaluationOracle>,
    gradient: Option<Box<dyn GradientOracle>>,
    dimensions: u32,
    seed: u64,
    initial_point: Option<Vec<f64>>,
) -> Result<OptimizationOutcome, OptimizerError> {
    let (gradient_oracle, tracking): (Box<dyn GradientOracle>, Box<dyn EvaluationOracle>) =
        match gradient {
            Some(gradient) => (gradient, evaluation),
            None => {
                // The finite difference and the tracking share one oracle: θ's
                // value comes from the difference's batch, and Adam reports it
                // as the run's fitness (C-5).
                let evaluation: Arc<dyn EvaluationOracle> = Arc::from(evaluation);
                let difference = FiniteDifferenceGradient::new(
                    Arc::clone(&evaluation),
                    cfg.finite_difference_step,
                )?;
                (Box::new(difference), Box::new(evaluation))
            }
        };
    AlgorithmAdam.optimize(AlgorithmAdamArgs {
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
    })
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
    /// (CMA-ES, whose candidates all sit on it with a tiny σ₀) and the centre
    /// of the calibration (SPSA, whose first candidate is θ₀ + c·Δ₁).
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
        for strategy in [DeStrategy::Rand1Bin, DeStrategy::Shade, DeStrategy::LShade] {
            cases.push((
                "DE, population 2",
                Box::new(move || de(strategy, 2)),
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
        // finite-difference stencil (2·dims + 1 candidates).
        let sizes = sizes(&log);
        assert_eq!(sizes.len(), outcome.iterations_run);
        assert!(sizes.iter().all(|&n| n == 7), "{sizes:?}");
    }

    #[test]
    fn adam_uses_the_exact_gradient_when_the_oracles_carry_one() {
        let (oracle, log) = recorder();
        let calls = Arc::new(Mutex::new(0));
        let oracles = Oracles {
            evaluation: oracle,
            gradient: Some(Box::new(QuadraticGradient(Arc::clone(&calls)))),
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
        // once per iteration.
        let sizes = sizes(&log);
        assert_eq!(sizes.len(), outcome.iterations_run);
        assert!(sizes.iter().all(|&n| n == 1), "{sizes:?}");
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
}
