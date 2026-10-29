use crate::{
    assign_parameters_qiskit, CostObservable, EvaluationError, EvaluationOracle, Label,
    OracleErrorSlot, SupervisedObjective,
};
use polypus_infrastructure::{
    BoundCircuit, CancelToken, CircuitTask, Counts, InfrastructureError, NativeStatevectorBackend,
    Planner, QiskitCircuit, QuantumBackend, RunParams,
};
use polypus_optimizers::GradientOracle;
use polypus_orchestration::{OracleFactory, Oracles, Resources};
use polypus_qml::{minibatch_indices, QmlError, QmlProblem};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
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
/// # Sources
///
/// What it trains is a [`QmlSource`]. The Qiskit form is described below. The
/// native form holds a `polypus-qml` [`QmlProblem`], which binds a candidate's
/// circuits in Rust (no GIL) and turns their counts into the fitness
/// (`−mean_loss`); with `exact`, the circuits run on a native backend's exact
/// probabilities instead of through the planner, and cancellation is checked
/// between windows. Both forms go through the same window loop, so the bound on
/// live circuits below holds for both; only how a window is bound and scored
/// differs. Only the native form has an exact gradient ([`GradientOracle`]).
///
/// # Minibatches
///
/// With a [`MinibatchConfig`], each call scores a fresh subset of the samples,
/// the same for every candidate of the call (and for the base and every shift
/// of a gradient call); [`evaluate_full`](QmlOracle::evaluate_full) scores every
/// sample regardless, for an honest final fitness.
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
    /// What is trained: Qiskit training circuits with a Python-side objective, or
    /// a native [`QmlProblem`].
    pub source: QmlSource,
    pub config: Arc<RunParams>,
    pub backend: Arc<dyn QuantumBackend>,
    /// Owns how the bound circuits are executed and reduced (waves, concurrency).
    pub planner: Arc<dyn Planner>,
    /// Optional per-call minibatching (`batch_size`); `None` scores every sample.
    pub minibatch: Option<MinibatchConfig>,
    /// Cooperative cancellation, shared with the run's `Scheduler`/entry point.
    pub cancel: CancelToken,
    /// Shared with the `qml.train` entry point: the first evaluation failure is
    /// recorded here and surfaced as a `PyErr` after `optimize` returns, since
    /// [`EvaluationOracle::evaluate_batch`] cannot return a `Result`.
    pub errors: OracleErrorSlot,
}

/// What a [`QmlOracle`] trains.
pub enum QmlSource {
    /// The Qiskit form of `qml.train`: pre-bound training circuits (feature-map
    /// parameters already fixed) scored by a [`QmlObjective`].
    Qiskit {
        /// Pre-bound training circuits (feature-map parameters already fixed).
        training_circuits: Vec<Py<PyAny>>,
        /// Turns each training circuit's counts into its per-sample value.
        objective: QmlObjective,
    },
    /// The native form: a `polypus-qml` problem, which binds its own circuits and
    /// turns their results into a fitness and an exact gradient (C-8).
    Native {
        /// The model, training set and loss.
        problem: Arc<QmlProblem>,
        /// `Some` for exact mode: the circuits run on this backend's exact
        /// probabilities instead of through the planner. The run's backend is a
        /// `dyn QuantumBackend`, which cannot be turned back into a native one, so
        /// the edge builds this one itself.
        exact: Option<Arc<NativeStatevectorBackend>>,
    },
}

impl QmlSource {
    /// Training samples, i.e. circuits per candidate before any minibatch.
    fn num_samples(&self) -> usize {
        match self {
            QmlSource::Qiskit {
                training_circuits, ..
            } => training_circuits.len(),
            QmlSource::Native { problem, .. } => problem.num_circuits(),
        }
    }
}

/// Deterministic minibatching (design doc §17): each [`QmlOracle`] call scores a
/// fresh subset of `batch_size` samples, drawn by [`minibatch_indices`] from
/// `seed` and a per-oracle call counter, so a run stays reproducible (C-7). One
/// draw serves a whole call: every candidate of an `evaluate_batch`, or the base
/// and every shift of a gradient, sees the same samples.
#[derive(Debug)]
pub struct MinibatchConfig {
    batch_size: usize,
    seed: u64,
    calls: AtomicU64,
}

impl MinibatchConfig {
    /// Score `batch_size` samples per call (`>= 1`; the edge also rejects a batch
    /// as large as the training set), drawn from `seed`.
    pub fn new(batch_size: usize, seed: u64) -> Self {
        MinibatchConfig {
            batch_size: batch_size.max(1),
            seed,
            calls: AtomicU64::new(0),
        }
    }

    /// The samples of the next call, out of `num_samples`.
    fn next(&self, num_samples: usize) -> Vec<usize> {
        let call = self.calls.fetch_add(1, AtomicOrdering::Relaxed);
        minibatch_indices(num_samples, self.seed, call, self.batch_size)
    }
}

/// The samples one call scores: all of them, or a minibatch.
enum Samples {
    All,
    Subset(Vec<usize>),
}

impl EvaluationOracle for QmlOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        self.evaluate(candidates, self.config.shots, &self.config)
    }

    /// Every circuit of the batch, each candidate on **every** training sample,
    /// runs with `shots` shots instead of the run's (`final_shots` of a
    /// training run). The backend gives each call its own sampling stream
    /// (contract C-7), so its shot noise is independent of the training's. It
    /// draws no minibatch: the final batch scores the whole training set, and a
    /// seeded run is the same up to it.
    fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        self.evaluate_batch_full(candidates, shots, &self.config)
    }
}

impl QmlOracle {
    /// [`evaluate_batch`](EvaluationOracle::evaluate_batch): one mean per
    /// candidate, each circuit measured with `shots` shots, every call to the
    /// backend with `config`.
    fn evaluate(&self, candidates: &[Vec<f64>], shots: u32, config: &RunParams) -> Vec<f64> {
        // Once evaluation has failed, stop doing work: return finite sentinels
        // and let the entry point surface the recorded error.
        if self.errors.failed() {
            return vec![0.0; candidates.len()];
        }
        match self.try_evaluate(candidates, shots, config) {
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

impl GradientOracle for QmlOracle {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        self.gradient_batch(theta, theta.len())
            .get(param_index)
            .copied()
            .unwrap_or(0.0)
    }

    /// The exact parameter-shift gradient of a native source (one call, one
    /// minibatch). Same sentinel discipline as
    /// [`evaluate_batch`](EvaluationOracle::evaluate_batch): a failure, including
    /// asking a Qiskit source, which has no exact gradient, is recorded and the
    /// result is zeros.
    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        if self.errors.failed() {
            return vec![0.0; dims];
        }
        match self.try_gradient(theta) {
            Ok((_, gradient)) => gradient,
            Err(e) => {
                self.errors.record(Box::new(e), &self.config.id);
                vec![0.0; dims]
            }
        }
    }

    /// `θ`'s fitness and the exact gradient from one call: the gradient's
    /// circuits start with `θ`'s own (C-11 (e)), so Adam need not score `θ`
    /// apart. Only on every sample: under a minibatch those circuits are the
    /// minibatch's, while Adam tracks its best on the full set (C-5), so this is
    /// `None` and Adam scores `θ` with its own oracle. A failure is recorded and
    /// returns finite sentinels, as [`gradient_batch`](Self::gradient_batch)
    /// does.
    fn value_and_gradient(&self, theta: &[f64], dims: usize) -> Option<(f64, Vec<f64>)> {
        if self.minibatch.is_some() {
            return None;
        }
        if self.errors.failed() {
            return Some((0.0, vec![0.0; dims]));
        }
        match self.try_gradient(theta) {
            Ok(measured) => Some(measured),
            Err(e) => {
                self.errors.record(Box::new(e), &self.config.id);
                Some((0.0, vec![0.0; dims]))
            }
        }
    }

    /// `true` without a minibatch, when
    /// [`value_and_gradient`](Self::value_and_gradient) returns `θ`'s fitness,
    /// so the walltime projection counts an Adam iteration as one call.
    fn scores_theta(&self) -> bool {
        self.minibatch.is_none()
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
    fn try_evaluate(
        &self,
        candidates: &[Vec<f64>],
        shots: u32,
        config: &RunParams,
    ) -> Result<Vec<f64>, EvaluationError> {
        self.try_evaluate_windowed(candidates, self.window(), shots, config)
    }

    /// `theta`'s fitness on **every** training sample, whatever the minibatch:
    /// the honest final score of a minibatched run (design doc §17).
    pub fn evaluate_full(&self, theta: &[f64]) -> Result<f64, EvaluationError> {
        let fitness = self.evaluate_samples(
            &[theta.to_vec()],
            self.window(),
            Samples::All,
            self.config.shots,
            &self.config,
        )?;
        Ok(fitness[0])
    }

    /// [`evaluate`](Self::evaluate) on every training sample, with the same
    /// failure discipline: what [`FullDataset`] serves, and the final batch.
    fn evaluate_batch_full(
        &self,
        candidates: &[Vec<f64>],
        shots: u32,
        config: &RunParams,
    ) -> Vec<f64> {
        if self.errors.failed() {
            return vec![0.0; candidates.len()];
        }
        match self.evaluate_samples(candidates, self.window(), Samples::All, shots, config) {
            Ok(values) => values,
            Err(e) => {
                self.errors.record(Box::new(e), &self.config.id);
                vec![0.0; candidates.len()]
            }
        }
    }

    /// The candidate window for this oracle's backend and training set.
    fn window(&self) -> usize {
        candidate_window_size(
            self.backend.capabilities().max_concurrency,
            self.source.num_samples(),
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
        config: &RunParams,
    ) -> Result<Vec<f64>, EvaluationError> {
        let samples = match &self.minibatch {
            Some(minibatch) => Samples::Subset(minibatch.next(self.source.num_samples())),
            None => Samples::All,
        };
        self.evaluate_samples(candidates, window, samples, shots, config)
    }

    /// Score `candidates` on `samples`, through the shared window loop, each
    /// circuit with `shots` shots and every backend call with `config`. Exact
    /// mode draws no shots, so it ignores both.
    fn evaluate_samples(
        &self,
        candidates: &[Vec<f64>],
        window: usize,
        samples: Samples,
        shots: u32,
        config: &RunParams,
    ) -> Result<Vec<f64>, EvaluationError> {
        match &self.source {
            QmlSource::Qiskit {
                training_circuits,
                objective,
            } => self.evaluate_qiskit(
                candidates,
                window,
                training_circuits,
                objective,
                samples,
                shots,
                config,
            ),
            QmlSource::Native { problem, exact } => {
                let problem = subset(problem, samples)?;
                let n = problem.num_circuits();
                self.evaluate_in_windows(
                    candidates,
                    window,
                    n,
                    |theta, bound| {
                        bound.extend(
                            problem
                                .bind_batch(theta)?
                                .into_iter()
                                .map(BoundCircuit::Native),
                        );
                        Ok(())
                    },
                    |bound, n_candidates| match exact {
                        Some(native) => {
                            let probabilities = self.run_exact(native, bound)?;
                            per_candidate(n_candidates, n, &probabilities, |weights| {
                                problem.fitness_from_probabilities(weights)
                            })
                        }
                        None => {
                            let counts = self.run_sampled(bound, shots, config)?;
                            per_candidate(n_candidates, n, &counts, |weights| {
                                problem.fitness_from_counts(weights)
                            })
                        }
                    },
                )
            }
        }
    }

    /// The Qiskit form: the evaluation of #146/#272, unchanged, on the selected
    /// training circuits (and their labels).
    fn evaluate_qiskit(
        &self,
        candidates: &[Vec<f64>],
        window: usize,
        training_circuits: &[Py<PyAny>],
        objective: &QmlObjective,
        samples: Samples,
        shots: u32,
        config: &RunParams,
    ) -> Result<Vec<f64>, EvaluationError> {
        // Labels are paired with circuits by position. The edge rejects a mismatch
        // up front (contract C-8); this covers direct Rust callers.
        if let QmlObjective::Supervised { labels, .. } = objective {
            if labels.len() != training_circuits.len() {
                return Err(EvaluationError::LabelCount {
                    labels: labels.len(),
                    samples: training_circuits.len(),
                });
            }
        }
        let selected: Vec<usize> = match samples {
            Samples::All => (0..training_circuits.len()).collect(),
            Samples::Subset(indices) => indices,
        };
        let labels: Vec<Label> = match objective {
            QmlObjective::Supervised { labels, .. } => {
                selected.iter().map(|&i| labels[i]).collect()
            }
            QmlObjective::Unsupervised(_) => Vec::new(),
        };
        let n_train = selected.len();
        self.evaluate_in_windows(
            candidates,
            window,
            n_train,
            |theta, bound| {
                // Binding is Qiskit-specific (feature-map pre-binding) and
                // re-acquires the GIL internally.
                for &i in &selected {
                    bound.push(QiskitCircuit::into_bound(assign_parameters_qiskit(
                        &training_circuits[i],
                        theta,
                    )?));
                }
                Ok(())
            },
            |bound, n_candidates| {
                // One 2D task per circuit, every one with the batch's `shots`: the
                // run's in training, `final_shots` for the final batch. `shots`
                // comes from the task, the single source of truth the Planner reads.
                let tasks: Vec<CircuitTask> = bound
                    .iter()
                    .map(|circuit| CircuitTask { circuit, shots })
                    .collect();

                // Delegate execution to the Planner: it owns the waves, the per-wave
                // concurrency cap, the between-wave `check_signals` and the shot merge.
                let values = self.per_sample_values(
                    objective,
                    &labels,
                    &tasks,
                    n_candidates,
                    n_train,
                    config,
                )?;

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
                // training set yields `NaN` (caught by the finiteness check)
                // instead of a `chunks(0)` panic. The sum is over the same values, in the
                // same order, as the eager flat batch, so the mean is byte-identical.
                Ok((0..n_candidates)
                    .map(|w| {
                        let slice = &values[w * n_train..(w + 1) * n_train];
                        slice.iter().sum::<f64>() / n_train as f64
                    })
                    .collect())
            },
        )
    }

    /// The window loop both sources share (#146): bind at most `window` candidates
    /// (`n_per_candidate` circuits each, **candidate-major**: candidate `w`'s
    /// circuits occupy `[w*n, (w+1)*n)` within the window), `score` them to one
    /// fitness per candidate, and drop the window before binding the next. That
    /// drop is what bounds the live [`BoundCircuit`]s to `window * n_per_candidate`;
    /// a failing window short-circuits the rest, so a fault in window N never lets
    /// window N+1 be built.
    ///
    /// `bind` appends one candidate's circuits; `score` turns a window's circuits
    /// into its candidates' fitness values, in order.
    fn evaluate_in_windows<B, S>(
        &self,
        candidates: &[Vec<f64>],
        window: usize,
        n_per_candidate: usize,
        mut bind: B,
        mut score: S,
    ) -> Result<Vec<f64>, EvaluationError>
    where
        B: FnMut(&[f64], &mut Vec<BoundCircuit>) -> Result<(), EvaluationError>,
        S: FnMut(&[BoundCircuit], usize) -> Result<Vec<f64>, EvaluationError>,
    {
        let mut fitness: Vec<f64> = Vec::with_capacity(candidates.len());
        // `chunks` floors the window at 1, so a degenerate `0` degrades to
        // per-candidate windows instead of panicking; the constructor never
        // yields one, but keep the loop total.
        for window_candidates in candidates.chunks(window.max(1)) {
            let mut bound: Vec<BoundCircuit> =
                Vec::with_capacity(window_candidates.len() * n_per_candidate);
            for theta in window_candidates {
                bind(theta, &mut bound)?;
            }
            let values = score(&bound, window_candidates.len())?;
            if values.len() != window_candidates.len() {
                return Err(EvaluationError::WrongLength {
                    expected: window_candidates.len(),
                    got: values.len(),
                });
            }
            fitness.extend(values);
            // `bound` drops here, releasing this window's circuits before the next
            // window is bound.
        }

        // Optimizer-facing C-5: exactly one finite fitness per candidate. A NaN/inf
        // would silently poison the pure-Rust optimizer. Reported as a `Result`,
        // never a panic (this runs under `OracleErrorSlot`).
        if let Some((index, &value)) = fitness.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(EvaluationError::NonFinite { index, value });
        }
        Ok(fitness)
    }

    /// One value per circuit of a window, in order. The window is candidate-major,
    /// so task `i` belongs to sample `i % n_train`, labelled `labels[i % n_train]`.
    fn per_sample_values(
        &self,
        objective: &QmlObjective,
        labels: &[Label],
        tasks: &[CircuitTask<'_>],
        n_candidates: usize,
        n_train: usize,
        config: &RunParams,
    ) -> Result<Vec<f64>, EvaluationError> {
        match objective {
            // Unchanged: the Planner executes and reduces.
            QmlObjective::Unsupervised(observable) => Ok(self.planner.evaluate(
                self.backend.as_ref(),
                tasks,
                observable.as_ref(),
                config,
                &self.cancel,
            )?),
            QmlObjective::Supervised { objective, .. } => {
                // Execute only; the pairing happens here, where the layout is known.
                let counts =
                    self.planner
                        .execute(self.backend.as_ref(), tasks, config, &self.cancel)?;
                // `labels` holds one label per selected circuit, so cycling pairs
                // task `i` with `labels[i % n_train]`.
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

    /// Run `bound` through the planner, each circuit with `shots` shots and the
    /// call with `config`, and return one counts map per circuit.
    fn run_sampled(
        &self,
        bound: &[BoundCircuit],
        shots: u32,
        config: &RunParams,
    ) -> Result<Vec<Counts>, EvaluationError> {
        let tasks: Vec<CircuitTask> = bound
            .iter()
            .map(|circuit| CircuitTask { circuit, shots })
            .collect();
        let counts = self
            .planner
            .execute(self.backend.as_ref(), &tasks, config, &self.cancel)?;
        if counts.len() != tasks.len() {
            return Err(EvaluationError::WrongLength {
                expected: tasks.len(),
                got: counts.len(),
            });
        }
        Ok(counts)
    }

    /// Run `bound` on the exact native backend, after the cancellation and
    /// interrupt checks the planner would make between waves.
    fn run_exact(
        &self,
        native: &NativeStatevectorBackend,
        bound: &[BoundCircuit],
    ) -> Result<Vec<HashMap<String, f64>>, EvaluationError> {
        if self.cancel.is_cancelled() {
            return Err(InfrastructureError::Cancelled.into());
        }
        self.cancel.poll_interrupt()?;
        let probabilities = native.run_circuits_exact(bound, &self.config)?;
        if probabilities.len() != bound.len() {
            return Err(EvaluationError::WrongLength {
                expected: bound.len(),
                got: probabilities.len(),
            });
        }
        Ok(probabilities)
    }

    /// Fallible core of [`GradientOracle::gradient_batch`] and
    /// [`GradientOracle::value_and_gradient`]: the native source's
    /// [`gradient_circuits`](QmlProblem::gradient_circuits), run in one planner
    /// call (or exactly), turned by the problem into `θ`'s fitness, from the
    /// base circuits that lead the batch, and `∂fitness/∂θ`.
    fn try_gradient(&self, theta: &[f64]) -> Result<(f64, Vec<f64>), EvaluationError> {
        let QmlSource::Native { problem, exact } = &self.source else {
            return Err(EvaluationError::Qml(QmlError::NoExactGradient {
                reason: "the training circuits are Qiskit circuits, which polypus cannot \
                         differentiate exactly; use a finite-difference gradient"
                    .to_string(),
            }));
        };
        let samples = match &self.minibatch {
            Some(minibatch) => Samples::Subset(minibatch.next(problem.num_circuits())),
            None => Samples::All,
        };
        let problem = subset(problem, samples)?;
        let bound: Vec<BoundCircuit> = problem
            .gradient_circuits(theta)?
            .into_iter()
            .map(BoundCircuit::Native)
            .collect();
        // The base circuits lead the batch, one per sample, exactly as
        // `bind_batch` returns them.
        let n = problem.num_circuits();
        Ok(match exact {
            Some(native) => {
                let probabilities = self.run_exact(native, &bound)?;
                (
                    problem.fitness_from_probabilities(&probabilities[..n])?,
                    problem.gradient_from_probabilities(&probabilities)?,
                )
            }
            None => {
                let counts = self.run_sampled(&bound, self.config.shots, &self.config)?;
                (
                    problem.fitness_from_counts(&counts[..n])?,
                    problem.gradient_from_counts(&counts)?,
                )
            }
        })
    }
}

/// `problem` restricted to `samples`: shared as is for every sample, or a
/// minibatch subset.
fn subset(problem: &Arc<QmlProblem>, samples: Samples) -> Result<Arc<QmlProblem>, EvaluationError> {
    match samples {
        Samples::All => Ok(Arc::clone(problem)),
        Samples::Subset(indices) => problem
            .subset(&indices)
            .map(Arc::new)
            .map_err(|e| EvaluationError::Runtime(format!("minibatch selection failed: {e}"))),
    }
}

/// Split a window's per-circuit results (candidate-major, `n` per candidate) and
/// score each candidate's slice with `fitness`.
fn per_candidate<W>(
    n_candidates: usize,
    n: usize,
    results: &[W],
    fitness: impl Fn(&[W]) -> Result<f64, QmlError>,
) -> Result<Vec<f64>, EvaluationError> {
    if results.len() != n_candidates * n {
        return Err(EvaluationError::WrongLength {
            expected: n_candidates * n,
            got: results.len(),
        });
    }
    (0..n_candidates)
        .map(|w| Ok(fitness(&results[w * n..(w + 1) * n])?))
        .collect()
}

/// Assembles a [`QmlOracle`] for a training flow from the run's [`Resources`].
///
/// The QML counterpart of [`VqcOracleFactory`](crate::VqcOracleFactory): it carries
/// what is trained ([`QmlSource`]) and the optional minibatch, and wires them to
/// the run context when [`polypus_orchestration::TrainFlow`] runs. See
/// [`VqcOracleFactory`](crate::VqcOracleFactory) for the order-vs-assembly split.
pub struct QmlOracleFactory {
    /// What is trained.
    pub source: QmlSource,
    /// Optional per-call minibatching.
    pub minibatch: Option<MinibatchConfig>,
}

impl OracleFactory for QmlOracleFactory {
    /// One [`QmlOracle`] in every role the run needs: the evaluation, the exact
    /// gradient of a native source, and, with a minibatch, the same oracle on
    /// every sample ([`FullDataset`]). Sharing one oracle keeps one minibatch
    /// counter for the whole run.
    fn build(
        self,
        resources: &Resources,
        cancel: &CancelToken,
        errors: &OracleErrorSlot,
    ) -> Oracles {
        let native = matches!(self.source, QmlSource::Native { .. });
        let minibatched = self.minibatch.is_some();
        let oracle = Arc::new(QmlOracle {
            source: self.source,
            config: Arc::clone(&resources.config),
            backend: Arc::clone(&resources.backend),
            planner: Arc::clone(&resources.planner),
            minibatch: self.minibatch,
            cancel: cancel.clone(),
            errors: errors.clone(),
        });
        Oracles {
            evaluation: Box::new(Arc::clone(&oracle)),
            gradient: native.then(|| Box::new(Arc::clone(&oracle)) as Box<dyn GradientOracle>),
            full_evaluation: minibatched
                .then(|| Box::new(FullDataset(Arc::clone(&oracle))) as Box<dyn EvaluationOracle>),
        }
    }
}

/// A [`QmlOracle`] scored on every training sample, whatever its minibatch: the
/// `full_evaluation` of a minibatched run ([`Oracles`]).
pub struct FullDataset(pub Arc<QmlOracle>);

impl EvaluationOracle for FullDataset {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        self.0
            .evaluate_batch_full(candidates, self.0.config.shots, &self.0.config)
    }

    /// As the oracle's own: every sample, `shots` shots; the backend gives the
    /// call its own stream (C-7).
    fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        self.0.evaluate_batch_with_shots(candidates, shots)
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
        /// Shots and seed of each `run_circuits` call, in order.
        call_params: Mutex<Vec<(u32, Option<u64>)>>,
    }

    impl MockBackend {
        fn recording() -> Self {
            Self {
                encode_position: false,
                fail_on_call: None,
                max_concurrency: usize::MAX,
                call_sizes: Mutex::new(Vec::new()),
                call_params: Mutex::new(Vec::new()),
            }
        }

        fn positional() -> Self {
            Self {
                encode_position: true,
                fail_on_call: None,
                max_concurrency: usize::MAX,
                call_sizes: Mutex::new(Vec::new()),
                call_params: Mutex::new(Vec::new()),
            }
        }

        fn failing_on(call: usize) -> Self {
            Self {
                encode_position: false,
                fail_on_call: Some(call),
                max_concurrency: usize::MAX,
                call_sizes: Mutex::new(Vec::new()),
                call_params: Mutex::new(Vec::new()),
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
            self.call_params
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .iter()
                .map(|&(shots, _)| shots)
                .collect()
        }

        fn call_seeds(&self) -> Vec<Option<u64>> {
            self.call_params
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .iter()
                .map(|&(_, seed)| seed)
                .collect()
        }
    }

    impl QuantumBackend for MockBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            config: &RunParams,
        ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
            self.call_params
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push((config.shots, config.seed));
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
            source: QmlSource::Qiskit {
                training_circuits: stub.training_circuits(n_train),
                objective: QmlObjective::Unsupervised(Arc::new(KeyOneObservable)),
            },
            config: config(shots),
            backend,
            planner: Arc::new(SequentialPlanner),
            minibatch: None,
            cancel: CancelToken::default(),
            errors: OracleErrorSlot::new(),
        }
    }

    /// `oracle` with its Qiskit objective replaced.
    fn with_objective(mut oracle: QmlOracle, objective: QmlObjective) -> QmlOracle {
        if let QmlSource::Qiskit {
            objective: current, ..
        } = &mut oracle.source
        {
            *current = objective;
        }
        oracle
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
        with_objective(
            oracle(
                stub,
                labels.len(),
                Arc::new(MockBackend::positional()),
                shots,
            ),
            QmlObjective::Supervised {
                objective,
                labels: labels.iter().map(|&l| Label::Class(l)).collect(),
            },
        )
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
            .try_evaluate_windowed(&candidates(POPULATION), WINDOW, 8, &oracle.config)
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
        let oracle_windowed = oracle(&stub, N_TRAIN, Arc::new(MockBackend::positional()), SHOTS);
        let windowed = oracle_windowed
            .try_evaluate_windowed(&cands, 2, SHOTS, &oracle_windowed.config)
            .expect("windowed evaluation succeeds");

        // A window wider than the population reproduces the former single flat batch.
        let stub_full = Stub::new();
        let oracle_eager = oracle(
            &stub_full,
            N_TRAIN,
            Arc::new(MockBackend::positional()),
            SHOTS,
        );
        let eager = oracle_eager
            .try_evaluate_windowed(&cands, POPULATION * 4, SHOTS, &oracle_eager.config)
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
            .try_evaluate_windowed(&candidates(POPULATION), WINDOW, 8, &oracle.config)
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
            let oracle = supervised_oracle(&stub, &LABELS, Arc::clone(&objective), SHOTS);
            let means = oracle
                .try_evaluate_windowed(&cands, window, SHOTS, &oracle.config)
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
        let oracle = supervised_oracle(&stub, &LABELS, objective, 64);
        let err = oracle
            .try_evaluate_windowed(&candidates(5), 2, 64, &oracle.config)
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
        let oracle = with_objective(
            // Three training circuits, two labels.
            oracle(&stub, 3, Arc::new(MockBackend::recording()), 8),
            QmlObjective::Supervised {
                objective: Arc::new(RecordingObjective::new()),
                labels: vec![Label::Class(0), Label::Class(1)],
            },
        );
        match oracle.try_evaluate_windowed(&candidates(4), WINDOW, 8, &oracle.config) {
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
            .try_evaluate(&candidates(population), 8, &oracle.config)
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
    /// `evaluate_batch` with the run's; both send the run's seed, which the
    /// backend derives each call's from (C-7), with either objective.
    #[test]
    fn a_batch_with_shots_puts_them_in_every_task() {
        const LABELS: [i64; 3] = [3, 1, 4];
        const SHOTS: u32 = 64;
        let cands = candidates(5);
        let stub = Stub::new();
        let unsupervised = |backend| oracle(&stub, LABELS.len(), backend, SHOTS);
        let supervised = |backend| {
            with_objective(
                oracle(&stub, LABELS.len(), backend, SHOTS),
                QmlObjective::Supervised {
                    objective: Arc::new(RecordingObjective::new()),
                    labels: LABELS.iter().map(|&l| Label::Class(l)).collect(),
                },
            )
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
            // The run's seed on every call: the backend derives each call's.
            let seeds = backend.call_seeds();
            assert!(seeds.iter().all(|&s| s == oracle.config.seed), "{seeds:?}");
            // Windowed too: two candidates a window, every circuit at 4096.
            let windowed = Arc::new(MockBackend::recording());
            let oracle = build(Arc::clone(&windowed));
            oracle
                .try_evaluate_windowed(&cands, 2, 4096, &oracle.config)
                .expect("a healthy evaluation succeeds");
            assert_eq!(windowed.call_sizes(), vec![6, 6, 3]);
            assert_eq!(windowed.call_shots(), vec![4096; 3]);
            assert_eq!(windowed.call_seeds(), vec![oracle.config.seed; 3]);
        }
    }

    // ── Native source (polypus-qml), exact mode, gradient, minibatch ──────────

    use polypus_qml::{
        Dataset, Decision, Loss, Observable, Pauli, PauliString, QuantumModel, Readout,
        RotationAxis,
    };

    /// A small native problem: 3 qubits, an angle encoder, a basic convolution
    /// (shared parameters) and a `Z₁` readout, over `n` samples.
    fn native_problem(n: usize) -> Arc<QmlProblem> {
        let z1 =
            Observable::new(vec![(1.0, PauliString::new(vec![(1, Pauli::Z)]).unwrap())]).unwrap();
        let model = QuantumModel::new(3)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .conv(polypus_qml::ConvBlock::Basic)
            .readout(Readout::new(vec![z1], Decision::Raw).unwrap())
            .compile(3)
            .unwrap();
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![0.3 * i as f64, 1.1 - 0.2 * i as f64, 0.7 + 0.1 * i as f64])
            .collect();
        let labels: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.4 })
            .collect();
        let data = Dataset::from_rows(&rows, &labels).unwrap();
        Arc::new(QmlProblem::new(model, data, Loss::SquaredError).unwrap())
    }

    fn native_oracle(
        problem: &Arc<QmlProblem>,
        exact: bool,
        minibatch: Option<MinibatchConfig>,
        seed: u64,
    ) -> QmlOracle {
        let native = Arc::new(NativeStatevectorBackend::new(seed));
        let backend: Arc<dyn QuantumBackend> = native.clone();
        QmlOracle {
            source: QmlSource::Native {
                problem: Arc::clone(problem),
                exact: exact.then_some(native),
            },
            config: config(2000),
            planner: backend.default_planner(),
            backend,
            minibatch,
            cancel: CancelToken::default(),
            errors: OracleErrorSlot::new(),
        }
    }

    fn native_candidates(problem: &QmlProblem, n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|c| {
                (0..problem.num_params())
                    .map(|k| 0.2 + 0.31 * k as f64 - 0.17 * c as f64)
                    .collect()
            })
            .collect()
    }

    fn bound(circuits: Vec<polypus_circuit::ConcreteCircuit>) -> Vec<BoundCircuit> {
        circuits.into_iter().map(BoundCircuit::Native).collect()
    }

    /// Exact probabilities straight from a fresh native backend.
    fn exact_probabilities(
        circuits: Vec<polypus_circuit::ConcreteCircuit>,
    ) -> Vec<HashMap<String, f64>> {
        NativeStatevectorBackend::new(0)
            .run_circuits_exact(&bound(circuits), &config(1))
            .unwrap()
    }

    /// A native source whose backend fails keeps the oracle contract (C-5): one
    /// finite sentinel per candidate, the original error kept in the slot for
    /// the entry point to re-raise, and no further work once failed (the
    /// gradient and the full-dataset view then return sentinels without calling
    /// the backend again).
    #[test]
    fn a_failing_native_run_records_its_error_and_returns_sentinels() {
        let problem = native_problem(4);
        let candidates = native_candidates(&problem, 3);
        let backend = Arc::new(MockBackend::failing_on(1));
        let as_dyn: Arc<dyn QuantumBackend> = backend.clone();
        let oracle = QmlOracle {
            source: QmlSource::Native {
                problem: Arc::clone(&problem),
                exact: None,
            },
            config: config(64),
            planner: as_dyn.default_planner(),
            backend: as_dyn,
            minibatch: None,
            cancel: CancelToken::default(),
            errors: OracleErrorSlot::new(),
        };

        let fitness = oracle.evaluate_batch(&candidates);
        assert_eq!(fitness, vec![0.0; candidates.len()]);
        assert!(oracle.errors.failed());
        let calls = backend.call_sizes().len();

        assert_eq!(
            oracle.gradient_batch(&candidates[0], problem.num_params()),
            vec![0.0; problem.num_params()]
        );
        assert_eq!(
            oracle.evaluate_batch_full(&candidates, 64, &oracle.config),
            vec![0.0; 3]
        );
        assert_eq!(
            oracle.value_and_gradient(&candidates[0], problem.num_params()),
            Some((0.0, vec![0.0; problem.num_params()]))
        );
        assert_eq!(
            backend.call_sizes().len(),
            calls,
            "no backend call after the failure"
        );

        let error = oracle.errors.take().expect("the failure is recorded");
        let error = error
            .downcast::<EvaluationError>()
            .expect("the slot holds the oracle's EvaluationError");
        assert!(
            matches!(*error, EvaluationError::Backend(_)),
            "a backend failure is kept as EvaluationError::Backend, got {error:?}"
        );
    }

    #[test]
    fn native_sampled_fitness_is_the_problems_fitness_of_the_same_counts() {
        let problem = native_problem(4);
        let candidates = native_candidates(&problem, 3);
        let oracle = native_oracle(&problem, false, None, 21);
        let fitness = oracle
            .try_evaluate_windowed(&candidates, WINDOW, 2000, &oracle.config)
            .unwrap();
        // One window, one planner call: the same counts a backend with the same
        // seed gives for the same candidate-major batch.
        let circuits: Vec<_> = candidates
            .iter()
            .flat_map(|theta| problem.bind_batch(theta).unwrap())
            .collect();
        let counts = NativeStatevectorBackend::new(21)
            .run_circuits(&bound(circuits), &config(2000))
            .unwrap();
        let expected: Vec<f64> = counts
            .chunks(problem.num_circuits())
            .map(|c| problem.fitness_from_counts(c).unwrap())
            .collect();
        assert_eq!(fitness, expected);
    }

    #[test]
    fn native_exact_fitness_and_gradient_are_the_problems() {
        let problem = native_problem(4);
        let candidates = native_candidates(&problem, 3);
        let oracle = native_oracle(&problem, true, None, 0);
        let fitness = oracle
            .try_evaluate_windowed(&candidates, 2, 2000, &oracle.config)
            .unwrap();
        for (theta, f) in candidates.iter().zip(&fitness) {
            let probabilities = exact_probabilities(problem.bind_batch(theta).unwrap());
            assert_eq!(
                *f,
                problem.fitness_from_probabilities(&probabilities).unwrap()
            );
        }
        let theta = &candidates[1];
        let gradient = oracle.gradient_batch(theta, problem.num_params());
        let probabilities = exact_probabilities(problem.gradient_circuits(theta).unwrap());
        assert_eq!(
            gradient,
            problem.gradient_from_probabilities(&probabilities).unwrap()
        );
        assert!(!oracle.errors.failed());
    }

    #[test]
    fn native_sampled_gradient_is_reproducible_for_a_seed() {
        let problem = native_problem(3);
        let theta = &native_candidates(&problem, 1)[0];
        let dims = problem.num_params();
        let a = native_oracle(&problem, false, None, 4).gradient_batch(theta, dims);
        let b = native_oracle(&problem, false, None, 4).gradient_batch(theta, dims);
        assert_eq!(a.len(), dims);
        assert!(a.iter().all(|g| g.is_finite()));
        assert_eq!(a, b);
    }

    /// `value_and_gradient` reads `θ`'s fitness from the base circuits that lead
    /// the gradient's own batch: value and gradient are the problem's on the
    /// same counts a backend with the same seed gives for that one batch, so
    /// `θ` is not run apart.
    #[test]
    fn native_value_and_gradient_reads_theta_from_the_gradient_batch() {
        let problem = native_problem(4);
        let theta = &native_candidates(&problem, 1)[0];
        let dims = problem.num_params();
        let (value, gradient) = native_oracle(&problem, false, None, 21)
            .value_and_gradient(theta, dims)
            .expect("without a minibatch the oracle scores θ");
        let counts = NativeStatevectorBackend::new(21)
            .run_circuits(
                &bound(problem.gradient_circuits(theta).unwrap()),
                &config(2000),
            )
            .unwrap();
        let n = problem.num_circuits();
        assert_eq!(value, problem.fitness_from_counts(&counts[..n]).unwrap());
        assert_eq!(gradient, problem.gradient_from_counts(&counts).unwrap());
    }

    /// In exact mode the value is, bit for bit, what the oracle's own
    /// `evaluate_batch` returns for `θ` (the C-5 requirement), and the gradient
    /// is `gradient_batch`'s.
    #[test]
    fn native_exact_value_and_gradient_match_the_separate_calls() {
        let problem = native_problem(4);
        let theta = &native_candidates(&problem, 1)[0];
        let dims = problem.num_params();
        let oracle = native_oracle(&problem, true, None, 0);
        let (value, gradient) = oracle
            .value_and_gradient(theta, dims)
            .expect("without a minibatch the oracle scores θ");
        assert_eq!(value, oracle.evaluate_batch(std::slice::from_ref(theta))[0]);
        assert_eq!(gradient, oracle.gradient_batch(theta, dims));
    }

    /// Under a minibatch the gradient's base circuits are the minibatch's, not
    /// the full set Adam tracks with, so the oracle declines and draws nothing:
    /// the next call still takes the first minibatch.
    #[test]
    fn a_minibatch_keeps_theta_out_of_the_native_gradient() {
        let problem = native_problem(6);
        let theta = &native_candidates(&problem, 1)[0];
        let oracle = native_oracle(&problem, true, Some(MinibatchConfig::new(2, 9)), 0);
        assert_eq!(oracle.value_and_gradient(theta, problem.num_params()), None);
        let fitness = oracle
            .try_evaluate_windowed(std::slice::from_ref(theta), 1, 2000, &oracle.config)
            .unwrap();
        let first = problem.subset(&minibatch_indices(6, 9, 0, 2)).unwrap();
        let probabilities = exact_probabilities(first.bind_batch(theta).unwrap());
        assert_eq!(
            fitness[0],
            first.fitness_from_probabilities(&probabilities).unwrap()
        );
    }

    /// `scores_theta` tells in advance what `value_and_gradient` returns (the
    /// walltime projection counts an Adam iteration by it): `true` on every
    /// sample, `false` under a minibatch.
    #[test]
    fn the_native_gradient_declares_whether_it_scores_theta() {
        let problem = native_problem(6);
        let theta = &native_candidates(&problem, 1)[0];
        let dims = problem.num_params();
        for (minibatch, scores) in [(None, true), (Some(MinibatchConfig::new(2, 9)), false)] {
            let label = format!("minibatch {minibatch:?}");
            let oracle = native_oracle(&problem, true, minibatch, 0);
            assert_eq!(oracle.scores_theta(), scores, "{label}");
            assert_eq!(
                oracle.value_and_gradient(theta, dims).is_some(),
                oracle.scores_theta(),
                "{label}"
            );
        }
    }

    /// A native oracle's final batch (`evaluate_batch_with_shots`) scores every
    /// training sample with the shots it is given, even under a minibatch, and
    /// draws no minibatch, so a seeded run is the same up to it: the counts
    /// are those a backend with the oracle's seed gives that batch at 4096 shots.
    #[test]
    fn a_native_final_batch_scores_every_sample_with_its_shots() {
        let problem = native_problem(6);
        let candidates = native_candidates(&problem, 2);
        let oracle = native_oracle(&problem, false, Some(MinibatchConfig::new(2, 9)), 21);
        let values = oracle.evaluate_batch_with_shots(&candidates, 4096);
        assert!(!oracle.errors.failed());
        let circuits: Vec<_> = candidates
            .iter()
            .flat_map(|theta| problem.bind_batch(theta).unwrap())
            .collect();
        let counts = NativeStatevectorBackend::new(21)
            .run_circuits(&bound(circuits), &config(4096))
            .unwrap();
        let expected: Vec<f64> = counts
            .chunks(problem.num_circuits())
            .map(|c| problem.fitness_from_counts(c).unwrap())
            .collect();
        assert_eq!(values, expected);
        // No minibatch was drawn: the next evaluation takes minibatch 0.
        let exact = native_oracle(&problem, true, Some(MinibatchConfig::new(2, 9)), 0);
        exact.evaluate_batch_with_shots(&candidates, 4096);
        let next = exact
            .try_evaluate_windowed(&candidates[..1], 1, 2000, &exact.config)
            .unwrap();
        let first = problem.subset(&minibatch_indices(6, 9, 0, 2)).unwrap();
        let probabilities = exact_probabilities(first.bind_batch(&candidates[0]).unwrap());
        assert_eq!(
            next[0],
            first.fitness_from_probabilities(&probabilities).unwrap()
        );
    }

    #[test]
    fn a_qiskit_source_has_no_exact_gradient() {
        let stub = Stub::new();
        let oracle = oracle(&stub, 3, Arc::new(MockBackend::recording()), 8);
        assert_eq!(oracle.gradient_batch(&[0.1], 1), vec![0.0]);
        assert_eq!(stub.assign_calls(), 0, "nothing may be bound or run");
        let err = oracle.errors.take().expect("the failure is recorded");
        match err.downcast_ref::<EvaluationError>() {
            Some(EvaluationError::Qml(QmlError::NoExactGradient { reason })) => {
                assert!(reason.contains("Qiskit"), "{reason}")
            }
            other => panic!("expected NoExactGradient, got {other:?}"),
        }
    }

    #[test]
    fn a_minibatch_is_drawn_once_per_call_and_shared_by_the_whole_call() {
        let problem = native_problem(6);
        let candidates = native_candidates(&problem, 3);
        let oracle = native_oracle(&problem, true, Some(MinibatchConfig::new(2, 9)), 0);
        // Calls 0 and 1 score their own minibatch, the same for every candidate;
        // call 2 is a gradient, base and shifts on one minibatch.
        for call in 0..2 {
            let fitness = oracle
                .try_evaluate_windowed(&candidates, 2, 2000, &oracle.config)
                .unwrap();
            let sub = problem.subset(&minibatch_indices(6, 9, call, 2)).unwrap();
            for (theta, f) in candidates.iter().zip(&fitness) {
                let probabilities = exact_probabilities(sub.bind_batch(theta).unwrap());
                assert_eq!(*f, sub.fitness_from_probabilities(&probabilities).unwrap());
            }
        }
        let theta = &candidates[0];
        let gradient = oracle.gradient_batch(theta, problem.num_params());
        let sub = problem.subset(&minibatch_indices(6, 9, 2, 2)).unwrap();
        let probabilities = exact_probabilities(sub.gradient_circuits(theta).unwrap());
        assert_eq!(
            gradient,
            sub.gradient_from_probabilities(&probabilities).unwrap()
        );
    }

    #[test]
    fn evaluate_full_scores_every_sample_and_leaves_the_minibatch_alone() {
        let problem = native_problem(6);
        let theta = native_candidates(&problem, 1).remove(0);
        let oracle = native_oracle(&problem, true, Some(MinibatchConfig::new(2, 9)), 0);
        let full = oracle.evaluate_full(&theta).unwrap();
        let probabilities = exact_probabilities(problem.bind_batch(&theta).unwrap());
        assert_eq!(
            full,
            problem.fitness_from_probabilities(&probabilities).unwrap()
        );
        // The next call still draws minibatch 0.
        let next = oracle
            .try_evaluate_windowed(std::slice::from_ref(&theta), 2, 2000, &oracle.config)
            .unwrap();
        let sub = problem.subset(&minibatch_indices(6, 9, 0, 2)).unwrap();
        let probabilities = exact_probabilities(sub.bind_batch(&theta).unwrap());
        assert_eq!(
            next[0],
            sub.fitness_from_probabilities(&probabilities).unwrap()
        );
    }

    #[test]
    fn a_qiskit_minibatch_binds_and_labels_only_the_drawn_samples() {
        let stub = Stub::new();
        let objective = Arc::new(RecordingObjective::new());
        let labels: Vec<i64> = vec![10, 11, 12, 13, 14];
        let mut oracle = supervised_oracle(&stub, &labels, Arc::clone(&objective), 64);
        oracle.minibatch = Some(MinibatchConfig::new(2, 5));
        oracle
            .try_evaluate_windowed(&candidates(3), WINDOW, 64, &oracle.config)
            .unwrap();
        // 3 candidates × 2 drawn samples, not × 5.
        assert_eq!(stub.assign_calls(), 6);
        let drawn: Vec<i64> = minibatch_indices(5, 5, 0, 2)
            .iter()
            .map(|&i| labels[i])
            .collect();
        let seen: Vec<i64> = objective.seen().iter().map(|&(_, label)| label).collect();
        assert_eq!(seen, [drawn.clone(), drawn.clone(), drawn].concat());
    }

    #[test]
    fn the_factory_hands_out_each_role_the_source_has() {
        use polypus_orchestration::Resources;
        let problem = native_problem(4);
        let theta = native_candidates(&problem, 1).remove(0);
        let resources = |seed| {
            Resources::new(
                Arc::new(NativeStatevectorBackend::new(seed)),
                None,
                config(500),
            )
            .unwrap_or_else(|_| panic!("a native backend pairs with its default planner"))
        };
        let exact = |minibatch| QmlOracleFactory {
            source: QmlSource::Native {
                problem: Arc::clone(&problem),
                exact: Some(Arc::new(NativeStatevectorBackend::new(0))),
            },
            minibatch,
        };
        let errors = OracleErrorSlot::new();
        let cancel = CancelToken::default();

        // Native, full batch: evaluation and exact gradient, which scores θ,
        // no full view.
        let oracles = exact(None).build(&resources(1), &cancel, &errors);
        assert!(oracles.gradient.as_ref().is_some_and(|g| g.scores_theta()));
        assert!(oracles.full_evaluation.is_none());

        // Native, minibatched: the exact gradient does not score θ, and the
        // full view scores every sample.
        let oracles =
            exact(Some(MinibatchConfig::new(2, 3))).build(&resources(1), &cancel, &errors);
        assert!(oracles.gradient.as_ref().is_some_and(|g| !g.scores_theta()));
        let full = oracles
            .full_evaluation
            .expect("a minibatch comes with a full-dataset view");
        let probabilities = exact_probabilities(problem.bind_batch(&theta).unwrap());
        assert_eq!(
            full.evaluate_batch(std::slice::from_ref(&theta)),
            vec![problem.fitness_from_probabilities(&probabilities).unwrap()]
        );
        assert!(!errors.failed());

        // Qiskit: no exact gradient (a finite difference is dispatch's job).
        let stub = Stub::new();
        let qiskit = QmlOracleFactory {
            source: QmlSource::Qiskit {
                training_circuits: stub.training_circuits(2),
                objective: QmlObjective::Unsupervised(Arc::new(KeyOneObservable)),
            },
            minibatch: None,
        }
        .build(&resources(1), &cancel, &errors);
        assert!(qiskit.gradient.is_none());
        assert!(qiskit.full_evaluation.is_none());
    }
}
