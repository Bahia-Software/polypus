//! Optimizer dispatch — the single place that turns a chosen optimization
//! [`Method`] plus an [`EvaluationOracle`] into an [`OptimizationOutcome`].
//!
//! `train` and `qml_train` used to carry *identical* DE/PSO/QNG dispatch blocks
//! (build the optimizer args, release the GIL, run `optimize`, then surface
//! either the oracle's recorded failure or the optimizer's own config error).
//! That triplication now lives here once:
//! [`dispatch_optimizer`] takes a pyo3-free [`Method`] (the binding parses the
//! `polypus.DE`/`PSO`/`QNG`/`CMAES` object into it and builds the QNG variance oracle),
//! runs the matching optimizer, and returns an [`OracleError`] the binding
//! converts back into a `PyErr`.
//!
//! This crate is deliberately Python-free — the GIL is released by the caller
//! (`py.detach(|| dispatch_optimizer(...))`) and re-acquired only inside
//! the oracle. So a real oracle failure reaches here **type-erased** as a
//! [`BoxedError`] held in the [`OracleErrorSlot`]: the concrete error (today an
//! `EvaluationError`, whose Python exception travels as a `DisplaySafePyErr`) is
//! boxed by the oracle and recovered by the `polypus` edge via `downcast` to
//! `EvaluationError`, keeping this crate free of `pyo3`.

use polypus_optimizers::{AlgorithmCmaEs, AlgorithmCmaEsArgs};
use polypus_optimizers::{
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, AlgorithmPSO,
    AlgorithmPSOArgs, AlgorithmQNG, AlgorithmQNGArgs, DeStrategy, EvaluationOracle,
    OptimizationOutcome, Optimizer, OptimizerError, VarianceOracle,
};
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

/// The chosen optimizer and its configuration. `Qng` additionally carries its
/// variance oracle (built from the Python callback at the binding boundary), so
/// dispatch stays Python-free.
pub enum Method {
    De(DeConfig),
    Pso(PsoConfig),
    Qng(QngConfig, Box<dyn VarianceOracle>),
    Cmaes(CmaesConfig),
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

/// Run `method` over `oracle` and return the outcome, or the run's failure.
///
/// The caller releases the GIL around this call; the optimizer loop is pure Rust
/// and the oracle re-acquires the GIL internally where it must. An oracle failure
/// recorded in `errors` during the run is returned in preference to the
/// optimizer's own error (see [`OracleError`]).
pub fn dispatch_optimizer(
    method: Method,
    oracle: Box<dyn EvaluationOracle>,
    dimensions: u32,
    errors: &OracleErrorSlot,
    seed: u64,
) -> Result<OptimizationOutcome, OracleError> {
    let result = match method {
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
        }),
    };

    // An oracle failure recorded during the run is the root cause; surface it
    // ahead of the optimizer's own (often downstream) error.
    if let Some(eval_err) = errors.take() {
        return Err(OracleError::Evaluation(eval_err));
    }
    result.map_err(OracleError::Config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_log::logged_for;
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
        let outcome =
            match dispatch_optimizer(de_method(strategy), oracle, 3, &OracleErrorSlot::new(), 9) {
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
}
