//! Optimizer dispatch — the single place that turns a chosen optimization
//! [`Method`] plus an [`EvaluationOracle`] into an [`OptimizationOutcome`].
//!
//! `train` and `qml_train` used to carry *identical* DE/PSO/QNG dispatch blocks
//! (build the optimizer args, release the GIL, run `optimize`, then surface
//! either the oracle's recorded failure or the optimizer's own config error).
//! That triplication now lives here once:
//! [`dispatch_optimizer`] takes a pyo3-free [`Method`] (the binding parses the
//! `polypus.DE`/`PSO`/`QNG` object into it and builds the QNG variance oracle),
//! runs the matching optimizer, and returns an [`OracleError`] the binding
//! converts back into a `PyErr`.
//!
//! This crate is deliberately Python-free — the GIL is released by the caller
//! (`py.allow_threads(|| dispatch_optimizer(...))`) and re-acquired only inside
//! the oracle. So a real oracle failure reaches here **type-erased** as a
//! [`BoxedError`] held in the [`OracleErrorSlot`]: the concrete error (today an
//! `EvaluationError` carrying a `PyErr`) is boxed by the oracle and recovered by
//! the `polypus` edge via `downcast`, keeping this crate free of `pyo3`.

use polypus_optimizers::{
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, AlgorithmPSO,
    AlgorithmPSOArgs, AlgorithmQNG, AlgorithmQNGArgs, EvaluationOracle, OptimizationOutcome,
    Optimizer, OptimizerError, VarianceOracle,
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

/// The chosen optimizer and its configuration. `Qng` additionally carries its
/// variance oracle (built from the Python callback at the binding boundary), so
/// dispatch stays Python-free.
pub enum Method {
    De(DeConfig),
    Pso(PsoConfig),
    Qng(QngConfig, Box<dyn VarianceOracle>),
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
/// `EvaluationError` carrying a `PyErr` — is boxed by the oracle so this crate
/// never names `pyo3`; the `polypus` edge downcasts it back to re-raise the
/// original Python exception verbatim (plan §10.1).
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
    pub fn record(&self, err: BoxedError, run_id: &str) {
        let mut guard = self.0.lock().unwrap_or_else(|p| p.into_inner());
        // Log only when this call actually stores the error, so the message never
        // claims to have "recorded" a failure it silently dropped.
        if guard.is_none() {
            log::error!("run {run_id}: oracle evaluation failed: {err}");
            *guard = Some(err);
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
        Method::De(cfg) => {
            AlgorithmDifferentialEvolution.optimize(AlgorithmDifferentialEvolutionArgs {
                oracle,
                population_size: cfg.population_size,
                generations: cfg.generations,
                dimensions,
                tolerance: cfg.tolerance,
                patience: cfg.patience,
                seed: Some(seed),
            })
        }
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
}
