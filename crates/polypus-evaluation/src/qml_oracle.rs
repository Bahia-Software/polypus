use crate::{
    assign_parameters_qiskit, CostObservable, EvaluationError, EvaluationOracle, OracleErrorSlot,
};
use polypus_infrastructure::{
    BoundCircuit, CancelToken, CircuitTask, ExecutionConfig, Planner, QuantumBackend,
};
use polypus_orchestration::{OracleFactory, Resources};
use pyo3::prelude::*;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Multiplier applied to the machine's available parallelism to size the
/// candidate window (see [`candidate_window_size`]).
///
/// Two, not more: the GIL serialises the per-candidate binding and the Aer
/// simulation, so a larger window only raises peak memory (more live
/// [`BoundCircuit`]s) without buying throughput. One window per core, doubled,
/// keeps the pipeline fed while the cap stays proportional to the hardware.
const CONCURRENCY_MULTIPLIER: usize = 2;

/// Parallelism assumed when [`std::thread::available_parallelism`] cannot report
/// it (it returns an error rather than a guess on a platform that hides the value
/// — a restrictive sandbox, an unsupported target).
///
/// One is the deliberate floor: under-guessing only shrinks the window (a smaller,
/// safer batch), whereas over-guessing is exactly the unbounded allocation this
/// cap exists to prevent.
const FALLBACK_PARALLELISM: usize = 1;

/// Maximum number of candidates bound into one in-flight window:
/// `CONCURRENCY_MULTIPLIER × available_parallelism()`.
///
/// Derived, not configured — there is no kwarg or [`ExecutionConfig`] field for
/// it, so `qml.train`'s public signature is untouched. Never zero (both constants
/// are `>= 1`), so the windowing loop is always finite.
fn candidate_window_size() -> usize {
    let parallelism = std::thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(FALLBACK_PARALLELISM);
    // Both constants are >= 1, so this is a floor rather than a correction: it
    // keeps the window usable whatever they become.
    parallelism.saturating_mul(CONCURRENCY_MULTIPLIER).max(1)
}

/// Oracle for QML training with feature-map encoding.
///
/// Holds N pre-bound training circuits (one per training sample, with feature-map
/// parameters already fixed). For each candidate `θ`, it binds `θ` to every
/// training circuit, runs those circuits through the [`Planner`], and returns the
/// **mean** expectation value per candidate as the fitness.
///
/// The oracle owns only the *what*: bind the candidates, submit the tasks, reduce
/// each candidate's expectations to their mean, and validate contract C-5. The
/// [`Planner`] owns the *how* — waves, the per-wave concurrency cap, the
/// between-wave `check_signals` (ENGINEERING §3) and cancellation. The GIL still
/// serialises the Qiskit binding and the Aer simulation calls; genuine parallelism
/// arrives with a native QML backend.
///
/// # Bounded in-flight circuit batch
///
/// The candidates are **not** all bound up front. Binding the whole `candidates ×
/// training_circuits` product eagerly would hold `candidates.len() * n_train`
/// [`BoundCircuit`]s live at once — for a DE/PSO population of 300 candidates over
/// 50 training samples that is 15 000 bound circuits, rebuilt every generation. The
/// [`Planner`]'s wave cap bounds how many circuits *run* concurrently, but it acts
/// on a slice the oracle has already built, so it cannot bound the *construction*;
/// and today's only real QML backend (`LocalBackend`) reports
/// `max_concurrency == usize::MAX`, so nothing downstream caps the batch either.
/// Keeping the peak allocation bounded is therefore this oracle's job.
///
/// So the fallible core binds the candidates in **windows of at most
/// `candidate_window_size` candidates**: it builds one window's `<= window *
/// n_train` circuits, submits that window to the [`Planner`], reduces it to its
/// per-candidate means, and drops it before building the next. At most `window *
/// n_train` [`BoundCircuit`]s are ever live, independent of the population size.
/// The window is derived from [`std::thread::available_parallelism`] (see
/// `candidate_window_size`), not configured, so `qml.train`'s public signature is
/// untouched.
///
/// This changes only peak memory, never results: the circuits are submitted in the
/// same candidate-major order and each candidate's mean is taken over the same
/// values in the same order, so [`EvaluationOracle::evaluate_batch`] returns one
/// fitness per candidate, in input order, byte-for-byte as the eager path did. Two
/// further consequences: the first window whose submission fails short-circuits the
/// rest — no later window is built — and Ctrl+C is now observed between windows,
/// not only between the [`Planner`]'s waves, so a large population is more
/// interruptible, not less.
pub struct QmlOracle {
    /// Pre-bound training circuits (feature-map parameters already fixed).
    pub training_circuits: Vec<Py<PyAny>>,
    pub config: Arc<ExecutionConfig>,
    pub backend: Arc<dyn QuantumBackend>,
    /// Owns how the bound circuits are executed and reduced (waves, concurrency).
    pub planner: Arc<dyn Planner>,
    pub observable: Arc<dyn CostObservable>,
    /// Cooperative cancellation, shared with the run's `Scheduler`/entry point.
    pub cancel: CancelToken,
    /// Shared with the `qml.train` entry point: the first evaluation failure is
    /// recorded here and surfaced as a `PyErr` after `optimize` returns, since
    /// [`EvaluationOracle::evaluate_batch`] cannot return a `Result`.
    pub errors: OracleErrorSlot,
}

impl EvaluationOracle for QmlOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        // Once evaluation has failed, stop doing work: return finite sentinels
        // and let the entry point surface the recorded error.
        if self.errors.failed() {
            return vec![0.0; candidates.len()];
        }
        match self.try_evaluate(candidates) {
            Ok(values) => values,
            Err(e) => {
                // Type-erase the failure into the (pyo3-free) slot; the FFI edge
                // downcasts it back to re-raise the original exception.
                self.errors.record(Box::new(e), &self.config.id);
                vec![0.0; candidates.len()]
            }
        }
    }
}

impl QmlOracle {
    /// Fallible core of [`EvaluationOracle::evaluate_batch`]. Kept separate so the
    /// trait method (which must return `Vec<f64>`) can record any error and yield
    /// finite sentinels while the entry point re-raises it.
    ///
    /// Binds and submits the candidates in windows of [`candidate_window_size`]
    /// candidates so at most `window * n_train` [`BoundCircuit`]s live at once —
    /// see the type-level "Bounded in-flight circuit batch" note.
    fn try_evaluate(&self, candidates: &[Vec<f64>]) -> Result<Vec<f64>, EvaluationError> {
        self.try_evaluate_windowed(candidates, candidate_window_size())
    }

    /// [`try_evaluate`](Self::try_evaluate) with the window size injected, so the
    /// bound can be exercised with a fixed window that does not depend on the test
    /// machine's core count.
    fn try_evaluate_windowed(
        &self,
        candidates: &[Vec<f64>],
        window: usize,
    ) -> Result<Vec<f64>, EvaluationError> {
        let n_train = self.training_circuits.len();
        let mut means: Vec<f64> = Vec::with_capacity(candidates.len());

        // Process the population in windows of at most `window` candidates. Each
        // window's circuits are built, submitted and reduced, then dropped before
        // the next window is built — that drop is what bounds the peak live
        // `BoundCircuit` count to `window * n_train`. `chunks` floors the window at
        // 1, so a degenerate `0` degrades to per-candidate windows instead of
        // panicking; the constructor never yields one, but keep the loop total.
        for window_candidates in candidates.chunks(window.max(1)) {
            // Bind this window's (candidate, training-circuit) pairs,
            // **candidate-major**: candidate `w`'s circuits occupy `[w*n_train,
            // (w+1)*n_train)` within the window, so the global submission order is
            // identical to the former eager flat batch. Binding is Qiskit-specific
            // (feature-map pre-binding) and re-acquires the GIL internally.
            let mut bound: Vec<BoundCircuit> =
                Vec::with_capacity(window_candidates.len() * n_train);
            for theta in window_candidates {
                for qc_xi in &self.training_circuits {
                    bound.push(BoundCircuit::Qiskit(assign_parameters_qiskit(
                        qc_xi, theta,
                    )?));
                }
            }

            // One 2D task per circuit — uniform shots in training; `shots` comes
            // from the task, the single source of truth the Planner reads.
            let tasks: Vec<CircuitTask> = bound
                .iter()
                .map(|circuit| CircuitTask {
                    circuit,
                    shots: self.config.shots,
                })
                .collect();

            // Delegate execution + reduction to the Planner: it owns the waves, the
            // per-wave concurrency cap, the between-wave `check_signals` and the
            // shot merge. A failure here propagates via `?` before the next window
            // is built, so a fault in window N never lets window N+1 be constructed
            // (the eager `dispatch_bounded` short-circuit, preserved).
            let expectations = self.planner.evaluate(
                self.backend.as_ref(),
                &tasks,
                self.observable.as_ref(),
                &self.config,
                &self.cancel,
            )?;

            // Structural C-5: exactly one expectation per submitted circuit. The
            // Planner guarantees this, but a short/long batch would misalign the
            // reshape below, so it is checked before the means are taken.
            if expectations.len() != tasks.len() {
                return Err(EvaluationError::WrongLength {
                    expected: tasks.len(),
                    got: expectations.len(),
                });
            }

            // Reduce each candidate's `n_train` expectations to their mean — the QML
            // fitness. Sliced manually (not `chunks(n_train)`) so an empty training
            // set yields `NaN` (caught by the finiteness check below) instead of a
            // `chunks(0)` panic. The sum is over the same values, in the same order,
            // as the eager flat batch, so the mean is byte-identical.
            for w in 0..window_candidates.len() {
                let slice = &expectations[w * n_train..(w + 1) * n_train];
                means.push(slice.iter().sum::<f64>() / n_train as f64);
            }
            // `tasks` (which borrows `bound`) and `bound` drop here, releasing this
            // window's circuits before the next window is bound.
        }

        // Optimizer-facing C-5: exactly one finite fitness per candidate. A NaN/inf
        // would silently poison the pure-Rust optimizer. Reported as a `Result`,
        // never a panic (this runs under `OracleErrorSlot`).
        if let Some((index, &value)) = means.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(EvaluationError::NonFinite { index, value });
        }
        Ok(means)
    }
}

/// Assembles a [`QmlOracle`] for a training flow from the run's [`Resources`].
///
/// The QML counterpart of [`VqcOracleFactory`](crate::VqcOracleFactory): it carries
/// the pre-bound training circuits and the reducer, and wires them to the run
/// context when [`polypus_orchestration::TrainFlow`] runs. See
/// [`VqcOracleFactory`](crate::VqcOracleFactory) for the order-vs-assembly split.
pub struct QmlOracleFactory {
    /// Pre-bound training circuits (feature-map parameters already fixed).
    pub training_circuits: Vec<Py<PyAny>>,
    /// Reduces each candidate's measurement counts to the fitness scalar.
    pub observable: Arc<dyn CostObservable>,
}

impl OracleFactory for QmlOracleFactory {
    fn build(
        self,
        resources: &Resources,
        cancel: &CancelToken,
        errors: &OracleErrorSlot,
    ) -> Box<dyn EvaluationOracle> {
        Box::new(QmlOracle {
            training_circuits: self.training_circuits,
            config: Arc::clone(&resources.config),
            backend: Arc::clone(&resources.backend),
            planner: Arc::clone(&resources.planner),
            observable: self.observable,
            cancel: cancel.clone(),
            errors: errors.clone(),
        })
    }
}

/// Tests for the in-flight windowing (issue #146).
///
/// `QmlOracle` binds `Py<PyAny>` training circuits through
/// `assign_parameters_qiskit`, which makes a real PyO3 call under the GIL, so —
/// unlike the fully GIL-free `VqcOracle` tests built on `CircuitSource::Native` —
/// these need a live interpreter. They avoid depending on Qiskit by standing a
/// minimal Python `Template` class in for the training circuit: its
/// `assign_parameters(params, inplace=...)` returns an opaque `Bound` object and
/// bumps a module-level counter, so the tests can both feed the oracle's real
/// binding path and read back exactly how many circuits were constructed. The
/// window is injected as a fixed value (`try_evaluate_windowed`) so the assertions
/// do not depend on the test machine's core count.
#[cfg(test)]
mod tests {
    use super::*;
    use polypus_infrastructure::{
        BackendCapabilities, BackendConfig, BackendError, OptLevel, SequentialPlanner,
    };
    use polypus_observable::ObservableError;
    use pyo3::types::PyModule;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// A Python stub standing in for a Qiskit training circuit: its
    /// `assign_parameters` records each invocation (so a test can prove how many
    /// `BoundCircuit`s were constructed) and returns an opaque `Bound` object the
    /// mock backend never inspects. Compiled once and kept, so `assign_calls`
    /// reads the same module-level counter the templates mutate.
    struct Stub {
        module: Py<PyModule>,
    }

    impl Stub {
        fn new() -> Self {
            use std::ffi::CString;
            use std::sync::atomic::{AtomicUsize, Ordering};
            // A unique module name per instance: `from_code` registers the module in
            // `sys.modules`, so a shared name would let parallel tests share one
            // `call_count`.
            static SEQ: AtomicUsize = AtomicUsize::new(0);
            let name = format!("qml_stub_{}", SEQ.fetch_add(1, Ordering::Relaxed));
            let file = CString::new(format!("{name}.py")).expect("no interior NUL");
            let module_name = CString::new(name).expect("no interior NUL");

            pyo3::prepare_freethreaded_python();
            Python::with_gil(|py| {
                let code = cr#"
call_count = 0


class Template:
    def assign_parameters(self, params, inplace=False):
        global call_count
        call_count += 1
        return Bound()


class Bound:
    pass
"#;
                let module = PyModule::from_code(py, code, file.as_c_str(), module_name.as_c_str())
                    .expect("the QML stub module must compile")
                    .unbind();
                Stub { module }
            })
        }

        /// `n` fresh `Template` instances to serve as the oracle's training circuits.
        fn training_circuits(&self, n: usize) -> Vec<Py<PyAny>> {
            Python::with_gil(|py| {
                let cls = self
                    .module
                    .bind(py)
                    .getattr("Template")
                    .expect("the stub exposes Template");
                (0..n)
                    .map(|_| {
                        cls.call0()
                            .expect("Template() constructs")
                            .into_any()
                            .unbind()
                    })
                    .collect()
            })
        }

        /// Total `assign_parameters` calls so far — i.e. `BoundCircuit`s built.
        fn assign_calls(&self) -> usize {
            Python::with_gil(|py| {
                self.module
                    .bind(py)
                    .getattr("call_count")
                    .expect("the stub exposes call_count")
                    .extract()
                    .expect("call_count is an int")
            })
        }
    }

    /// Reads each circuit's "1" count as its expectation (0 when absent), so a
    /// mis-ordered result vector stays detectable without touching Python.
    struct KeyOneObservable;

    impl CostObservable for KeyOneObservable {
        fn expectation_batch(
            &self,
            counts: &[HashMap<String, u64>],
        ) -> Result<Vec<f64>, ObservableError> {
            Ok(counts
                .iter()
                .map(|c| c.get("1").copied().unwrap_or(0) as f64)
                .collect())
        }
    }

    /// A backend that records the size of every `run_circuits` call.
    ///
    /// `max_concurrency` is `usize::MAX` (like the real `LocalBackend`), so the
    /// `SequentialPlanner` submits each window in a single call — the recorded call
    /// size is then exactly the oracle's window, the quantity under test. With
    /// `encode_position` it stamps each circuit's global submission index into its
    /// "1" count (contract C-3 conserving `shots`), so `KeyOneObservable` turns
    /// position into a distinguishable expectation. `fail_on_call` errors out of the
    /// N-th call before returning any counts, modelling a mid-population failure.
    struct MockBackend {
        encode_position: bool,
        fail_on_call: Option<usize>,
        /// Circuit count of each `run_circuits` call, in order.
        call_sizes: Mutex<Vec<usize>>,
    }

    impl MockBackend {
        fn recording() -> Self {
            Self {
                encode_position: false,
                fail_on_call: None,
                call_sizes: Mutex::new(Vec::new()),
            }
        }

        fn positional() -> Self {
            Self {
                encode_position: true,
                fail_on_call: None,
                call_sizes: Mutex::new(Vec::new()),
            }
        }

        fn failing_on(call: usize) -> Self {
            Self {
                encode_position: false,
                fail_on_call: Some(call),
                call_sizes: Mutex::new(Vec::new()),
            }
        }

        fn call_sizes(&self) -> Vec<usize> {
            self.call_sizes
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
        }
    }

    impl QuantumBackend for MockBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            config: &ExecutionConfig,
        ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
            let mut sizes = self.call_sizes.lock().unwrap_or_else(|p| p.into_inner());
            // Global index of this call's first circuit within the whole run — the
            // value `encode_position` stamps, so a mis-ordered result is detectable.
            let offset: usize = sizes.iter().sum();
            let call_index = sizes.len() + 1;
            sizes.push(qcs.len());
            drop(sizes);

            if self.fail_on_call == Some(call_index) {
                return Err(BackendError::Conversion("mock failure".to_string()));
            }

            Ok((0..qcs.len())
                .map(|i| {
                    if self.encode_position {
                        let ones = (offset + i) as u64;
                        HashMap::from([
                            ("1".to_string(), ones),
                            ("0".to_string(), u64::from(config.shots) - ones),
                        ])
                    } else {
                        HashMap::from([("0".to_string(), u64::from(config.shots))])
                    }
                })
                .collect())
        }

        fn capabilities(&self) -> BackendCapabilities {
            // Unbounded, like `LocalBackend`: the whole window reaches the backend in
            // one wave, so a recorded call size is exactly one oracle window.
            BackendCapabilities {
                max_concurrency: usize::MAX,
                supports_shot_distribution: true,
            }
        }
    }

    fn config(shots: u32) -> Arc<ExecutionConfig> {
        Arc::new(ExecutionConfig {
            id: "qml-oracle-test".to_string(),
            shots,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: BackendConfig::LocalNative { fusion: true },
            opt_level: OptLevel::default(),
            seed: Some(7),
        })
    }

    fn oracle(stub: &Stub, n_train: usize, backend: Arc<MockBackend>, shots: u32) -> QmlOracle {
        QmlOracle {
            training_circuits: stub.training_circuits(n_train),
            config: config(shots),
            backend,
            planner: Arc::new(SequentialPlanner),
            observable: Arc::new(KeyOneObservable),
            cancel: CancelToken::default(),
            errors: OracleErrorSlot::new(),
        }
    }

    /// `n` one-dimensional candidates. The mock ignores the angle (it keys off
    /// submission order), so the values only need to be distinct handles.
    fn candidates(n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| vec![0.1 * (i as f64 + 1.0)]).collect()
    }

    /// The window used by the injected-window tests: fixed, so nothing depends on
    /// the number of cores on the machine running the suite.
    const WINDOW: usize = 8;

    #[test]
    fn windowed_binding_keeps_at_most_one_window_of_circuits_in_flight() {
        // The issue's reference case: 300 candidates over 50 training circuits, i.e.
        // 15 000 pairs that must never be bound all at once.
        const POPULATION: usize = 300;
        const N_TRAIN: usize = 50;

        let stub = Stub::new();
        let backend = Arc::new(MockBackend::recording());
        let oracle = oracle(&stub, N_TRAIN, Arc::clone(&backend), 8);

        let means = oracle
            .try_evaluate_windowed(&candidates(POPULATION), WINDOW)
            .expect("a healthy windowed evaluation must succeed");
        assert_eq!(means.len(), POPULATION, "one fitness per candidate");

        let sizes = backend.call_sizes();
        // No call — hence no built window — exceeds `WINDOW * N_TRAIN` circuits.
        let peak = sizes.iter().copied().max().expect("at least one call");
        assert!(
            peak <= WINDOW * N_TRAIN,
            "a window carried {peak} circuits; the bound is {}",
            WINDOW * N_TRAIN
        );
        // …and it is genuinely a full window, not collapsed to per-candidate work.
        assert_eq!(
            peak,
            WINDOW * N_TRAIN,
            "the largest window must be a full {WINDOW}-candidate window"
        );
        // The population really was split into windows (not one giant eager batch),
        // one call per window.
        assert_eq!(
            sizes.len(),
            POPULATION.div_ceil(WINDOW),
            "the population must be submitted one call per window"
        );
        assert!(
            peak < POPULATION * N_TRAIN,
            "the whole cross product was bound at once — the window did not bound it"
        );
        // Nothing dropped or duplicated: every pair reached the backend exactly once.
        assert_eq!(sizes.iter().sum::<usize>(), POPULATION * N_TRAIN);
    }

    #[test]
    fn windowing_does_not_change_the_per_candidate_means() {
        const POPULATION: usize = 7;
        const N_TRAIN: usize = 3;
        // Max submission index is POPULATION*N_TRAIN-1 = 20; keep it below `shots`
        // so the "0" complement never underflows.
        const SHOTS: u32 = 64;

        let cands = candidates(POPULATION);

        let stub = Stub::new();
        let windowed = oracle(&stub, N_TRAIN, Arc::new(MockBackend::positional()), SHOTS)
            .try_evaluate_windowed(&cands, 2)
            .expect("windowed evaluation succeeds");

        // A window wider than the population reproduces the former single flat batch.
        let stub_full = Stub::new();
        let eager = oracle(
            &stub_full,
            N_TRAIN,
            Arc::new(MockBackend::positional()),
            SHOTS,
        )
        .try_evaluate_windowed(&cands, POPULATION * 4)
        .expect("single-batch evaluation succeeds");

        assert_eq!(
            windowed, eager,
            "windowing must be byte-identical to the eager single batch"
        );
        // Candidate i's circuits are the consecutive global indices [i*N, (i+1)*N),
        // whose mean is i*N + (N-1)/2 — pin the absolute values too, not just parity.
        let expected: Vec<f64> = (0..POPULATION)
            .map(|i| (i * N_TRAIN) as f64 + (N_TRAIN as f64 - 1.0) / 2.0)
            .collect();
        assert_eq!(windowed, expected, "each candidate's mean is order-correct");
    }

    #[test]
    fn a_failing_window_short_circuits_before_the_next_is_built() {
        const POPULATION: usize = 30;
        const N_TRAIN: usize = 4;

        let stub = Stub::new();
        // `max_concurrency` is unbounded, so call K is window K; fail window 2.
        let backend = Arc::new(MockBackend::failing_on(2));
        let oracle = oracle(&stub, N_TRAIN, Arc::clone(&backend), 8);

        let err = oracle
            .try_evaluate_windowed(&candidates(POPULATION), WINDOW)
            .expect_err("the backend failure must propagate");
        assert!(
            matches!(err, EvaluationError::Backend(_)),
            "a backend failure surfaces as EvaluationError::Backend, got {err:?}"
        );

        // Only the first two windows were submitted…
        assert_eq!(
            backend.call_sizes().len(),
            2,
            "no window may be submitted after the failing one"
        );
        // …and — the point of the bound — no circuit beyond window 2 was ever built.
        assert_eq!(
            stub.assign_calls(),
            2 * WINDOW * N_TRAIN,
            "window 3+ must not be constructed once window 2 has failed"
        );
    }

    #[test]
    fn candidate_window_size_is_two_per_core() {
        let parallelism = std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(FALLBACK_PARALLELISM);
        assert_eq!(
            candidate_window_size(),
            parallelism * CONCURRENCY_MULTIPLIER
        );
        // Never zero, whatever the platform reports: a zero window would stall the loop.
        assert!(candidate_window_size() >= CONCURRENCY_MULTIPLIER);
    }
}
