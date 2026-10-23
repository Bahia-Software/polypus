use crate::{
    assign_parameters_qiskit, CostObservable, EvaluationError, EvaluationOracle, Label,
    OracleErrorSlot, SupervisedObjective,
};
use polypus_infrastructure::{
    BoundCircuit, CancelToken, CircuitTask, Planner, QiskitCircuit, QuantumBackend, RunParams,
};
use polypus_orchestration::{OracleFactory, Oracles, Resources};
use pyo3::prelude::*;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Multiplier applied to the execution parallelism to size the candidate window
/// (see [`window_size`]).
///
/// Two, not more: one window's worth of work per execution unit, doubled, keeps
/// the pipeline fed while the next window is bound. A larger window only raises
/// peak memory (more live [`BoundCircuit`]s) without buying throughput.
const CONCURRENCY_MULTIPLIER: usize = 2;

/// Parallelism assumed when [`std::thread::available_parallelism`] cannot report
/// it (it returns an error rather than a guess on a platform that hides the value
/// — a restrictive sandbox, an unsupported target).
///
/// One is the deliberate floor: under-guessing only shrinks the window (a smaller,
/// safer batch), whereas over-guessing is exactly the unbounded allocation this
/// cap exists to prevent.
const FALLBACK_PARALLELISM: usize = 1;

/// Live [`BoundCircuit`]s one window may hold when a backend wider than the host
/// lifts it above the local window (see [`window_size`]).
///
/// The same order as the local window of a typical node (#146: `2 × cores ×
/// n_train`, ~800 circuits for 8 cores and 50 samples), and far below the eager
/// `population × n_train` product it replaced (15 000 for 300 × 50).
const MAX_WINDOW_CIRCUITS: usize = 4096;

/// Maximum number of candidates bound into one in-flight window.
///
/// `local` is the host's parallelism, `backend` the backend's batch-agnostic
/// `capabilities().max_concurrency` and `n_train` the circuits bound per
/// candidate. The rule (issue #272):
///
/// ```text
/// window = min(memory_bound, max(local, backend) × CONCURRENCY_MULTIPLIER)
/// memory_bound = max(local × CONCURRENCY_MULTIPLIER, MAX_WINDOW_CIRCUITS / n_train)
/// ```
///
/// - A backend narrower than the host (QMIO: 1) or as wide keeps the local window:
///   binding is host work, and the planner splits the window into waves anyway.
/// - A wider backend (CUNQA with more QPUs than local cores) lifts the window so
///   one window can fill every QPU instead of leaving some idle.
/// - `usize::MAX` means "no backend limit", not "unbounded parallelism": native
///   and local report it whenever the batch fits in memory, and their real
///   parallelism is the host's. It is therefore treated as `local`.
/// - The memory bound caps the lift at [`MAX_WINDOW_CIRCUITS`] live circuits, but
///   never below the #146 local window, so a run without a wide backend is sized
///   exactly as before.
///
/// The batch-agnostic `capabilities()` is used because `capabilities_for` needs
/// the bound tasks, which do not exist until the window is sized.
///
/// Derived, not configured — there is no kwarg or [`ExecutionConfig`] field for
/// it, so `qml.train`'s public signature is untouched. Never zero, so the
/// windowing loop is always finite.
fn window_size(local: usize, backend: usize, n_train: usize) -> usize {
    let local = local.max(1);
    let units = if backend == usize::MAX {
        local
    } else {
        local.max(backend)
    };
    let local_window = local.saturating_mul(CONCURRENCY_MULTIPLIER);
    let memory_bound = local_window.max(MAX_WINDOW_CIRCUITS / n_train.max(1));
    units
        .saturating_mul(CONCURRENCY_MULTIPLIER)
        .min(memory_bound)
        .max(1)
}

/// [`window_size`] for this host: `local` is
/// [`std::thread::available_parallelism`].
fn candidate_window_size(backend: usize, n_train: usize) -> usize {
    let local = std::thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(FALLBACK_PARALLELISM);
    window_size(local, backend, n_train)
}

/// How [`QmlOracle`] turns each training circuit's counts into a per-sample value.
pub enum QmlObjective {
    /// No labels: one cost observable for every sample, applied through
    /// [`Planner::evaluate`].
    Unsupervised(Arc<dyn CostObservable>),
    /// With `y_train`: each sample is scored against its own label.
    Supervised {
        objective: Arc<dyn SupervisedObjective>,
        /// One label per training circuit, in `training_circuits` order.
        labels: Vec<Label>,
    },
}

/// Oracle for QML training with feature-map encoding.
///
/// Holds N pre-bound training circuits (one per training sample, with feature-map
/// parameters already fixed). For each candidate `θ`, it binds `θ` to every
/// training circuit, runs those circuits through the [`Planner`], and returns the
/// **mean** per-sample value per candidate as the fitness ([`QmlObjective`]).
///
/// # Supervised evaluation
///
/// With labels, the circuits run through [`Planner::execute`] and the oracle pairs
/// the counts with the labels itself: the window is candidate-major, so circuit `i`
/// belongs to sample `i % n_train`. The planner's reducer is position-free by
/// contract and cannot do this pairing. The unsupervised path calls
/// [`Planner::evaluate`] as before.
///
/// The oracle owns only the *what*: bind the candidates, submit the tasks, reduce
/// each candidate's per-sample values to their mean, and validate contract C-5. The
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
/// The window is derived from the host's parallelism and the backend's
/// `capabilities().max_concurrency`, capped by a memory bound (see `window_size`),
/// not configured, so `qml.train`'s public signature is untouched.
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
    pub config: Arc<RunParams>,
    pub backend: Arc<dyn QuantumBackend>,
    /// Owns how the bound circuits are executed and reduced (waves, concurrency).
    pub planner: Arc<dyn Planner>,
    /// Turns each training circuit's counts into its per-sample value.
    pub objective: QmlObjective,
    /// Cooperative cancellation, shared with the run's `Scheduler`/entry point.
    pub cancel: CancelToken,
    /// Shared with the `qml.train` entry point: the first evaluation failure is
    /// recorded here and surfaced as a `PyErr` after `optimize` returns, since
    /// [`EvaluationOracle::evaluate_batch`] cannot return a `Result`.
    pub errors: OracleErrorSlot,
}

impl EvaluationOracle for QmlOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        self.evaluate(candidates, self.config.shots)
    }

    /// Every circuit of the batch, each candidate on every training sample,
    /// runs with `shots` shots instead of the run's (`final_shots` of a
    /// training run).
    fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        self.evaluate(candidates, shots)
    }
}

impl QmlOracle {
    /// Both trait methods: one mean per candidate, each circuit measured with
    /// `shots` shots.
    fn evaluate(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        // Once evaluation has failed, stop doing work: return finite sentinels
        // and let the entry point surface the recorded error.
        if self.errors.failed() {
            return vec![0.0; candidates.len()];
        }
        match self.try_evaluate(candidates, shots) {
            Ok(values) => values,
            Err(e) => {
                // Type-erase the failure into the (pyo3-free) slot; the FFI edge
                // downcasts it back to re-raise the original exception.
                self.errors.record(Box::new(e), &self.config.id);
                vec![0.0; candidates.len()]
            }
        }
    }

    /// Fallible core of [`EvaluationOracle::evaluate_batch`]. Kept separate so the
    /// trait method (which must return `Vec<f64>`) can record any error and yield
    /// finite sentinels while the entry point re-raises it.
    ///
    /// Binds and submits the candidates in windows of [`candidate_window_size`]
    /// candidates so at most `window * n_train` [`BoundCircuit`]s live at once —
    /// see the type-level "Bounded in-flight circuit batch" note.
    fn try_evaluate(
        &self,
        candidates: &[Vec<f64>],
        shots: u32,
    ) -> Result<Vec<f64>, EvaluationError> {
        self.try_evaluate_windowed(candidates, self.window(), shots)
    }

    /// The candidate window for this oracle's backend and training set.
    fn window(&self) -> usize {
        candidate_window_size(
            self.backend.capabilities().max_concurrency,
            self.training_circuits.len(),
        )
    }

    /// [`try_evaluate`](Self::try_evaluate) with the window size injected, so the
    /// bound can be exercised with a fixed window that does not depend on the test
    /// machine's core count.
    fn try_evaluate_windowed(
        &self,
        candidates: &[Vec<f64>],
        window: usize,
        shots: u32,
    ) -> Result<Vec<f64>, EvaluationError> {
        let n_train = self.training_circuits.len();
        // Labels are paired with circuits by position. The edge rejects a mismatch
        // up front (contract C-8); this covers direct Rust callers.
        if let QmlObjective::Supervised { labels, .. } = &self.objective {
            if labels.len() != n_train {
                return Err(EvaluationError::LabelCount {
                    labels: labels.len(),
                    samples: n_train,
                });
            }
        }
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
                    bound.push(QiskitCircuit::into_bound(assign_parameters_qiskit(
                        qc_xi, theta,
                    )?));
                }
            }

            // One 2D task per circuit, every one with the batch's `shots`: the
            // run's in training, `final_shots` for the final batch. `shots`
            // comes from the task, the single source of truth the Planner reads.
            let tasks: Vec<CircuitTask> = bound
                .iter()
                .map(|circuit| CircuitTask { circuit, shots })
                .collect();

            // Delegate execution to the Planner: it owns the waves, the per-wave
            // concurrency cap, the between-wave `check_signals` and the shot merge.
            // A failure here propagates via `?` before the next window is built, so
            // a fault in window N never lets window N+1 be constructed (the eager
            // `dispatch_bounded` short-circuit, preserved).
            let values = self.per_sample_values(&tasks, window_candidates.len(), n_train)?;

            // Structural C-5: exactly one value per submitted circuit. The Planner
            // guarantees this, but a short/long batch would misalign the reshape
            // below, so it is checked before the means are taken.
            if values.len() != tasks.len() {
                return Err(EvaluationError::WrongLength {
                    expected: tasks.len(),
                    got: values.len(),
                });
            }

            // Reduce each candidate's `n_train` per-sample values to their mean — the
            // QML fitness. Sliced manually (not `chunks(n_train)`) so an empty
            // training set yields `NaN` (caught by the finiteness check below)
            // instead of a `chunks(0)` panic. The sum is over the same values, in the
            // same order, as the eager flat batch, so the mean is byte-identical.
            for w in 0..window_candidates.len() {
                let slice = &values[w * n_train..(w + 1) * n_train];
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

    /// One value per circuit of a window, in order. The window is candidate-major,
    /// so task `i` belongs to sample `i % n_train`.
    fn per_sample_values(
        &self,
        tasks: &[CircuitTask<'_>],
        n_candidates: usize,
        n_train: usize,
    ) -> Result<Vec<f64>, EvaluationError> {
        match &self.objective {
            // Unchanged: the Planner executes and reduces.
            QmlObjective::Unsupervised(observable) => Ok(self.planner.evaluate(
                self.backend.as_ref(),
                tasks,
                observable.as_ref(),
                &self.config,
                &self.cancel,
            )?),
            QmlObjective::Supervised { objective, labels } => {
                // Execute only; the pairing happens here, where the layout is known.
                let counts = self.planner.execute(
                    self.backend.as_ref(),
                    tasks,
                    &self.config,
                    &self.cancel,
                )?;
                // The caller checked `labels.len() == n_train`, so cycling pairs task
                // `i` with `labels[i % n_train]`.
                let window_labels: Vec<Label> = labels
                    .iter()
                    .copied()
                    .cycle()
                    .take(n_candidates * n_train)
                    .collect();
                let scores = objective.score_batch(&counts, &window_labels)?;
                // Checked per sample so the error names the `x_train` row (typically
                // a `log(0)`).
                if let Some((index, &value)) =
                    scores.iter().enumerate().find(|(_, v)| !v.is_finite())
                {
                    return Err(EvaluationError::NonFiniteScore {
                        sample: index % n_train.max(1),
                        value,
                    });
                }
                Ok(scores)
            }
        }
    }
}

/// Assembles a [`QmlOracle`] for a training flow from the run's [`Resources`].
///
/// The QML counterpart of [`VqcOracleFactory`](crate::VqcOracleFactory): it carries
/// the pre-bound training circuits and the objective, and wires them to the run
/// context when
/// [`polypus_orchestration::TrainFlow`] runs. See
/// [`VqcOracleFactory`](crate::VqcOracleFactory) for the order-vs-assembly split.
pub struct QmlOracleFactory {
    /// Pre-bound training circuits (feature-map parameters already fixed).
    pub training_circuits: Vec<Py<PyAny>>,
    /// Turns each training circuit's counts into its per-sample value.
    pub objective: QmlObjective,
}

impl OracleFactory for QmlOracleFactory {
    fn build(
        self,
        resources: &Resources,
        cancel: &CancelToken,
        errors: &OracleErrorSlot,
    ) -> Oracles {
        Box::new(QmlOracle {
            training_circuits: self.training_circuits,
            config: Arc::clone(&resources.config),
            backend: Arc::clone(&resources.backend),
            planner: Arc::clone(&resources.planner),
            objective: self.objective,
            cancel: cancel.clone(),
            errors: errors.clone(),
        })
        .into()
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
    use polypus_infrastructure::{BackendCapabilities, BackendError, OptLevel, SequentialPlanner};
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

            pyo3::Python::initialize();
            Python::attach(|py| {
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
            Python::attach(|py| {
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
            Python::attach(|py| {
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
        /// Reported as `capabilities().max_concurrency`.
        max_concurrency: usize,
        /// Circuit count of each `run_circuits` call, in order.
        call_sizes: Mutex<Vec<usize>>,
        /// Shots of each `run_circuits` call, in order.
        call_shots: Mutex<Vec<u32>>,
    }

    impl MockBackend {
        fn recording() -> Self {
            Self {
                encode_position: false,
                fail_on_call: None,
                max_concurrency: usize::MAX,
                call_sizes: Mutex::new(Vec::new()),
                call_shots: Mutex::new(Vec::new()),
            }
        }

        fn positional() -> Self {
            Self {
                encode_position: true,
                fail_on_call: None,
                max_concurrency: usize::MAX,
                call_sizes: Mutex::new(Vec::new()),
                call_shots: Mutex::new(Vec::new()),
            }
        }

        fn failing_on(call: usize) -> Self {
            Self {
                encode_position: false,
                fail_on_call: Some(call),
                max_concurrency: usize::MAX,
                call_sizes: Mutex::new(Vec::new()),
                call_shots: Mutex::new(Vec::new()),
            }
        }

        /// A backend that runs at most `max_concurrency` circuits per call.
        fn with_concurrency(max_concurrency: usize) -> Self {
            Self {
                max_concurrency,
                ..Self::recording()
            }
        }

        fn call_sizes(&self) -> Vec<usize> {
            self.call_sizes
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
        }

        fn call_shots(&self) -> Vec<u32> {
            self.call_shots
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
        }
    }

    impl QuantumBackend for MockBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            config: &RunParams,
        ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
            let mut sizes = self.call_sizes.lock().unwrap_or_else(|p| p.into_inner());
            // Global index of this call's first circuit within the whole run — the
            // value `encode_position` stamps, so a mis-ordered result is detectable.
            let offset: usize = sizes.iter().sum();
            let call_index = sizes.len() + 1;
            sizes.push(qcs.len());
            drop(sizes);
            self.call_shots
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(config.shots);

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
            // Unbounded by default, like `LocalBackend`: the whole window reaches the
            // backend in one wave, so a recorded call size is exactly one window.
            BackendCapabilities {
                max_concurrency: self.max_concurrency,
                supports_shot_distribution: true,
            }
        }
    }

    fn config(shots: u32) -> Arc<RunParams> {
        Arc::new(RunParams {
            id: "qml-oracle-test".to_string(),
            shots,
            seed: Some(7),
            opt_level: OptLevel::default(),
        })
    }

    fn oracle(stub: &Stub, n_train: usize, backend: Arc<MockBackend>, shots: u32) -> QmlOracle {
        QmlOracle {
            training_circuits: stub.training_circuits(n_train),
            config: config(shots),
            backend,
            planner: Arc::new(SequentialPlanner),
            objective: QmlObjective::Unsupervised(Arc::new(KeyOneObservable)),
            cancel: CancelToken::default(),
            errors: OracleErrorSlot::new(),
        }
    }

    /// Records `(submission index, label)` pairs, reading the index from the
    /// positional mock's "1" count, and scores each as `index + 1000 * label`.
    struct RecordingObjective {
        seen: Mutex<Vec<(u64, i64)>>,
        /// Return NaN for the pair at this global submission index.
        nan_at: Option<u64>,
    }

    impl RecordingObjective {
        fn new() -> Self {
            Self {
                seen: Mutex::new(Vec::new()),
                nan_at: None,
            }
        }

        fn seen(&self) -> Vec<(u64, i64)> {
            self.seen.lock().unwrap_or_else(|p| p.into_inner()).clone()
        }
    }

    impl SupervisedObjective for RecordingObjective {
        fn score_batch(
            &self,
            counts: &[HashMap<String, u64>],
            labels: &[Label],
        ) -> Result<Vec<f64>, EvaluationError> {
            let mut seen = self.seen.lock().unwrap_or_else(|p| p.into_inner());
            Ok(counts
                .iter()
                .zip(labels)
                .map(|(c, label)| {
                    let index = c.get("1").copied().unwrap_or(0);
                    let Label::Class(label) = *label else {
                        panic!("these tests use class labels only");
                    };
                    seen.push((index, label));
                    if self.nan_at == Some(index) {
                        f64::NAN
                    } else {
                        index as f64 + 1000.0 * label as f64
                    }
                })
                .collect())
        }
    }

    fn supervised_oracle(
        stub: &Stub,
        labels: &[i64],
        objective: Arc<RecordingObjective>,
        shots: u32,
    ) -> QmlOracle {
        QmlOracle {
            objective: QmlObjective::Supervised {
                objective,
                labels: labels.iter().map(|&l| Label::Class(l)).collect(),
            },
            ..oracle(
                stub,
                labels.len(),
                Arc::new(MockBackend::positional()),
                shots,
            )
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
            .try_evaluate_windowed(&candidates(POPULATION), WINDOW, 8)
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
            .try_evaluate_windowed(&cands, 2, SHOTS)
            .expect("windowed evaluation succeeds");

        // A window wider than the population reproduces the former single flat batch.
        let stub_full = Stub::new();
        let eager = oracle(
            &stub_full,
            N_TRAIN,
            Arc::new(MockBackend::positional()),
            SHOTS,
        )
        .try_evaluate_windowed(&cands, POPULATION * 4, SHOTS)
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
            .try_evaluate_windowed(&candidates(POPULATION), WINDOW, 8)
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
    fn supervised_labels_follow_their_samples_across_windows() {
        const POPULATION: usize = 7;
        // Distinct labels, so a mis-paired circuit shows in the record and the mean.
        const LABELS: [i64; 3] = [3, 1, 4];
        const N_TRAIN: usize = LABELS.len();
        // Max submission index is POPULATION*N_TRAIN-1 = 20 < shots (no underflow).
        const SHOTS: u32 = 64;

        let cands = candidates(POPULATION);
        let mut results = Vec::new();
        // Windows that split the population unevenly, exactly, and not at all.
        for window in [1, 2, 3, POPULATION * 4] {
            let stub = Stub::new();
            let objective = Arc::new(RecordingObjective::new());
            let means = supervised_oracle(&stub, &LABELS, Arc::clone(&objective), SHOTS)
                .try_evaluate_windowed(&cands, window, SHOTS)
                .expect("a healthy supervised evaluation succeeds");

            let seen = objective.seen();
            assert_eq!(
                seen.len(),
                POPULATION * N_TRAIN,
                "every circuit scored once"
            );
            for (index, label) in seen {
                assert_eq!(
                    label,
                    LABELS[index as usize % N_TRAIN],
                    "window {window}: circuit {index} was scored against another sample's label"
                );
            }
            results.push(means);
        }

        // Candidate c's circuits are the indices c*N + j, scored against LABELS[j].
        let expected: Vec<f64> = (0..POPULATION)
            .map(|c| {
                let sum: f64 = (0..N_TRAIN)
                    .map(|j| (c * N_TRAIN + j) as f64 + 1000.0 * LABELS[j] as f64)
                    .sum();
                sum / N_TRAIN as f64
            })
            .collect();
        for means in results {
            assert_eq!(
                means, expected,
                "windowing must not change the supervised means"
            );
        }
    }

    #[test]
    fn a_non_finite_supervised_score_names_its_training_sample() {
        const LABELS: [i64; 4] = [0, 1, 0, 1];
        let stub = Stub::new();
        // Global index 11 = candidate 2, sample 3.
        let objective = Arc::new(RecordingObjective {
            nan_at: Some(11),
            ..RecordingObjective::new()
        });
        let err = supervised_oracle(&stub, &LABELS, objective, 64)
            .try_evaluate_windowed(&candidates(5), 2, 64)
            .expect_err("a NaN score must not reach the optimizer");
        match err {
            EvaluationError::NonFiniteScore { sample, value } => {
                assert_eq!(sample, 3, "the x_train row, not the candidate or the index");
                assert!(value.is_nan());
            }
            other => panic!("expected NonFiniteScore, got {other:?}"),
        }
    }

    #[test]
    fn a_label_count_mismatch_is_an_error_before_any_circuit_is_built() {
        let stub = Stub::new();
        let oracle = QmlOracle {
            objective: QmlObjective::Supervised {
                objective: Arc::new(RecordingObjective::new()),
                labels: vec![Label::Class(0), Label::Class(1)],
            },
            // Three training circuits, two labels.
            ..oracle(&stub, 3, Arc::new(MockBackend::recording()), 8)
        };
        match oracle.try_evaluate_windowed(&candidates(4), WINDOW, 8) {
            Err(EvaluationError::LabelCount { labels, samples }) => {
                assert_eq!((labels, samples), (2, 3));
            }
            other => panic!("expected LabelCount, got {other:?}"),
        }
        assert_eq!(stub.assign_calls(), 0, "nothing may be bound or run");
    }

    fn local_parallelism() -> usize {
        std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(FALLBACK_PARALLELISM)
    }

    #[test]
    fn a_backend_no_wider_than_the_host_keeps_the_local_window() {
        // QMIO-like (1), as wide as the host, and "no limit" (local/native).
        for backend in [1, 4, 8, usize::MAX] {
            assert_eq!(
                window_size(8, backend, 50),
                8 * CONCURRENCY_MULTIPLIER,
                "backend concurrency {backend}"
            );
        }
    }

    #[test]
    fn a_backend_wider_than_the_host_lifts_the_window() {
        // 32 QPUs on a 4-core host: 8 candidates would leave 24 QPUs idle.
        assert_eq!(window_size(4, 32, 1), 32 * CONCURRENCY_MULTIPLIER);
        assert_eq!(window_size(4, 32, 50), 32 * CONCURRENCY_MULTIPLIER);
    }

    #[test]
    fn the_lift_never_exceeds_the_memory_bound() {
        // A huge finite concurrency is capped at MAX_WINDOW_CIRCUITS live circuits.
        for n_train in [1, 3, 50, 1000] {
            let window = window_size(4, 1 << 40, n_train);
            let bound = (4 * CONCURRENCY_MULTIPLIER).max(MAX_WINDOW_CIRCUITS / n_train);
            assert_eq!(window, bound, "n_train {n_train}");
        }
        // The memory bound never shrinks the #146 local window.
        assert_eq!(window_size(64, 1 << 40, 1000), 64 * CONCURRENCY_MULTIPLIER);
        assert_eq!(
            window_size(64, usize::MAX, 1000),
            64 * CONCURRENCY_MULTIPLIER
        );
    }

    #[test]
    fn the_window_is_never_zero() {
        assert_eq!(window_size(0, 0, 0), CONCURRENCY_MULTIPLIER);
        assert_eq!(
            window_size(1, usize::MAX, usize::MAX),
            CONCURRENCY_MULTIPLIER
        );
    }

    #[test]
    fn the_oracle_sizes_its_window_from_its_backend() {
        let local = local_parallelism();
        let stub = Stub::new();
        for (backend, expected) in [
            (1, local),
            (local, local),
            (usize::MAX, local),
            (local * 4, local * 4),
        ] {
            let oracle = oracle(
                &stub,
                1,
                Arc::new(MockBackend::with_concurrency(backend)),
                8,
            );
            assert_eq!(
                oracle.window(),
                expected * CONCURRENCY_MULTIPLIER,
                "backend concurrency {backend} on {local} local cores"
            );
        }
    }

    #[test]
    fn a_wider_backend_receives_full_waves() {
        // Before #272 the window was 2 × local, so every call to a backend four times
        // wider than the host carried at most half its capacity.
        let concurrency = local_parallelism() * 4;
        let population = concurrency * 3;
        let stub = Stub::new();
        let backend = Arc::new(MockBackend::with_concurrency(concurrency));
        let oracle = oracle(&stub, 1, Arc::clone(&backend), 8);

        oracle
            .try_evaluate(&candidates(population), 8)
            .expect("a healthy evaluation succeeds");

        let sizes = backend.call_sizes();
        assert!(
            sizes.iter().all(|&n| n == concurrency),
            "every wave must fill the backend's {concurrency} slots, got {sizes:?}"
        );
        assert_eq!(sizes.iter().sum::<usize>(), population);
    }

    /// `evaluate_batch_with_shots` runs every circuit, each candidate on every
    /// training sample and in every window, with the shots it is given, and
    /// `evaluate_batch` with the run's, with either objective.
    #[test]
    fn a_batch_with_shots_puts_them_in_every_task() {
        const LABELS: [i64; 3] = [3, 1, 4];
        const SHOTS: u32 = 64;
        let cands = candidates(5);
        let stub = Stub::new();
        let unsupervised = |backend| oracle(&stub, LABELS.len(), backend, SHOTS);
        let supervised = |backend| QmlOracle {
            objective: QmlObjective::Supervised {
                objective: Arc::new(RecordingObjective::new()),
                labels: LABELS.iter().map(|&l| Label::Class(l)).collect(),
            },
            ..oracle(&stub, LABELS.len(), backend, SHOTS)
        };
        let builders: [&dyn Fn(Arc<MockBackend>) -> QmlOracle; 2] = [&unsupervised, &supervised];
        for build in builders {
            // A wave of at most four circuits, so each window runs in several.
            let backend = Arc::new(MockBackend::with_concurrency(4));
            let oracle = build(Arc::clone(&backend));
            let values = oracle.evaluate_batch(&cands);
            let runs = backend.call_shots().len();
            assert!(runs > 1, "{runs} waves");
            let with_shots = oracle.evaluate_batch_with_shots(&cands, 4096);
            assert!(!oracle.errors.failed());
            assert_eq!(values.len(), 5);
            assert_eq!(with_shots.len(), 5);
            let shots = backend.call_shots();
            assert_eq!(shots.len(), 2 * runs, "{shots:?}");
            assert!(shots[..runs].iter().all(|&s| s == SHOTS), "{shots:?}");
            assert!(shots[runs..].iter().all(|&s| s == 4096), "{shots:?}");
            assert_eq!(backend.call_sizes().iter().sum::<usize>(), 2 * 5 * 3);
            // Windowed too: two candidates a window, every circuit at 4096.
            let windowed = Arc::new(MockBackend::recording());
            build(Arc::clone(&windowed))
                .try_evaluate_windowed(&cands, 2, 4096)
                .expect("a healthy evaluation succeeds");
            assert_eq!(windowed.call_sizes(), vec![6, 6, 3]);
            assert_eq!(windowed.call_shots(), vec![4096; 3]);
        }
    }
}
