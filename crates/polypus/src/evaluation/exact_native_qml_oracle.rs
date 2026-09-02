use crate::evaluation::{EvaluationError, EvaluationOracle, MinibatchConfig, OracleErrorSlot};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, NativeStatevectorBackend};
use polypus_optimizers::GradientOracle;
use polypus_qml::{Loss, QmlProblem};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Oracle for **exact** native (pure-Rust) QML training (design doc §17).
///
/// The exact-mode twin of [`NativeQmlOracle`](crate::evaluation::NativeQmlOracle):
/// it holds the same [`QmlProblem`] and follows the same concurrency,
/// GIL-release and signal-check discipline, but it scores each candidate from
/// the **exact** basis-state probabilities of the statevector rather than from
/// finite-shot counts. There is no sampling anywhere on this path, so there is
/// no shot noise: the fitness and its parameter-shift gradient are the true
/// noiseless values, and two runs are byte-identical regardless of seed.
///
/// This is deliberately a **separate** oracle rather than a mode inside
/// `NativeQmlOracle`, so the already-proven sampled path carries zero regression
/// risk. Two structural differences follow from "exact":
///
/// - `backend` is the concrete [`NativeStatevectorBackend`], **not**
///   `Arc<dyn QuantumBackend>`: the exact read-out
///   ([`run_circuits_exact`](NativeStatevectorBackend::run_circuits_exact)) is an
///   inherent method of that backend, not part of the [`QuantumBackend`] trait,
///   because "exact" has no physical meaning for a noisy Aer backend or real
///   hardware (QMIO/CUNQA).
/// - `config.shots` and `config.seed` are unused (documented on
///   `run_circuits_exact`); only `opt_level` is read.
///
/// [`QuantumBackend`]: crate::infrastructure::QuantumBackend
pub struct ExactNativeQmlOracle {
    /// The trainable problem: bind parameters in, get exact fitness out.
    ///
    /// Held behind an [`Arc`] so the common (no-minibatch) evaluation path shares
    /// it by reference-count bump instead of deep-cloning its per-sample circuit
    /// templates on every `evaluate_batch`/`gradient_batch` call.
    pub problem: Arc<QmlProblem>,
    /// Unused fields on this path: `shots`/`seed` (no sampling). Only
    /// `opt_level` is read when building the transpile options.
    pub config: Arc<ExecutionConfig>,
    /// The concrete native backend — the exact read-out lives on it, not on the
    /// `QuantumBackend` trait.
    pub backend: Arc<NativeStatevectorBackend>,
    /// Shared with the `qml.train` entry point: the first evaluation failure is
    /// recorded here and surfaced as a `PyErr` after `optimize` returns.
    pub errors: OracleErrorSlot,
    /// Optional deterministic minibatching (design doc §17), identical in
    /// meaning to [`NativeQmlOracle`](crate::evaluation::NativeQmlOracle)'s: when
    /// `Some`, each `evaluate_batch`/`gradient_batch` call scores a fresh
    /// minibatch from its own counter; when `None`, every call scores the full
    /// training set. Exact mode removes shot noise, not the dataset, so
    /// minibatching applies here exactly as on the sampled path.
    pub minibatch: Option<MinibatchConfig>,
}

impl EvaluationOracle for ExactNativeQmlOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        // Same sentinel discipline as `NativeQmlOracle`: once evaluation has
        // failed, do no work and let the entry point surface the recorded error.
        if self.errors.failed() {
            return vec![0.0; candidates.len()];
        }
        match self.try_evaluate(candidates) {
            Ok(values) => values,
            Err(e) => {
                self.errors.record(e);
                vec![0.0; candidates.len()]
            }
        }
    }
}

impl GradientOracle for ExactNativeQmlOracle {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        if self.errors.failed() {
            return 0.0;
        }
        match self.try_gradient(theta, &[param_index]) {
            Ok(grad) => grad[0],
            Err(e) => {
                self.errors.record(e);
                0.0
            }
        }
    }

    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        if self.errors.failed() {
            return vec![0.0; dims];
        }
        let indices: Vec<usize> = (0..dims).collect();
        match self.try_gradient(theta, &indices) {
            Ok(grad) => grad,
            Err(e) => {
                self.errors.record(e);
                vec![0.0; dims]
            }
        }
    }
}

impl ExactNativeQmlOracle {
    /// Fallible core of [`EvaluationOracle::evaluate_batch`], mirroring
    /// [`NativeQmlOracle::try_evaluate`](crate::evaluation::NativeQmlOracle) to
    /// the letter: one `spawn_blocking` task per candidate, the GIL released
    /// around the join, and a single `check_signals` afterwards — only the
    /// per-candidate work runs the exact read-out.
    fn try_evaluate(&self, candidates: &[Vec<f64>]) -> Result<Vec<f64>, EvaluationError> {
        let rt = crate::utils::tokio_runtime().map_err(|e| {
            EvaluationError::Python(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to start the Tokio runtime for exact native QML evaluation: {e}"
            )))
        })?;

        // With minibatching (design doc §17) the shared handle is a fresh
        // minibatch drawn once here, so every candidate in this batch scores the
        // same minibatch — identical policy to `NativeQmlOracle::try_evaluate`.
        let problem: Arc<QmlProblem> = match &self.minibatch {
            Some(mb) => {
                let indices = mb.next_indices(&self.problem);
                Arc::new(self.problem.subset(&indices)?)
            }
            None => Arc::clone(&self.problem),
        };

        let handles: Vec<_> = candidates
            .iter()
            .map(|theta| {
                let problem = Arc::clone(&problem);
                let config = Arc::clone(&self.config);
                let backend = Arc::clone(&self.backend);
                let theta = theta.clone();
                rt.spawn_blocking(move || {
                    evaluate_exact_native_qml_single(&problem, &config, backend.as_ref(), &theta)
                })
            })
            .collect();

        // The native exact path never touches Python during the work, but the
        // calling thread still holds the GIL; release it around the join and
        // check signals once afterwards so `qml.train` stays interruptible.
        Python::with_gil(|py| {
            let out = py.allow_threads(|| {
                rt.block_on(async {
                    let mut out = Vec::with_capacity(handles.len());
                    for h in handles {
                        let single = h.await.map_err(|e| {
                            EvaluationError::Python(pyo3::exceptions::PyRuntimeError::new_err(
                                format!("exact native QML evaluation task failed: {e}"),
                            ))
                        })?;
                        out.push(single?);
                    }
                    Ok::<_, EvaluationError>(out)
                })
            })?;
            py.check_signals().map_err(EvaluationError::Python)?;
            Ok(out)
        })
    }

    /// Fallible core of the [`GradientOracle`] methods: the exact
    /// parameter-shift gradient (design doc §17), mirroring
    /// [`NativeQmlOracle::try_gradient`](crate::evaluation::NativeQmlOracle) but
    /// scoring from exact probabilities. It runs the base `θ` once plus
    /// `θ ± π/2·e_k` per requested `k` (`1 + 2·|param_indices|` circuit batches),
    /// then combines them via the problem's exact gradient methods.
    fn try_gradient(
        &self,
        theta: &[f64],
        param_indices: &[usize],
    ) -> Result<Vec<f64>, EvaluationError> {
        let rt = crate::utils::tokio_runtime().map_err(|e| {
            EvaluationError::Python(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to start the Tokio runtime for exact native QML gradient: {e}"
            )))
        })?;

        // One minibatch per gradient call, drawn once so the base θ and every
        // θ±π/2·e_k shift score the same samples — the parameter-shift coherence
        // constraint, identical to `NativeQmlOracle::try_gradient`.
        let problem: Arc<QmlProblem> = match &self.minibatch {
            Some(mb) => {
                let indices = mb.next_indices(&self.problem);
                Arc::new(self.problem.subset(&indices)?)
            }
            None => Arc::clone(&self.problem),
        };

        let shift = std::f64::consts::PI / 2.0;
        let mut thetas: Vec<Vec<f64>> = Vec::with_capacity(1 + 2 * param_indices.len());
        thetas.push(theta.to_vec());
        for &k in param_indices {
            let mut plus = theta.to_vec();
            let mut minus = theta.to_vec();
            plus[k] += shift;
            minus[k] -= shift;
            thetas.push(plus);
            thetas.push(minus);
        }

        let handles: Vec<_> = thetas
            .iter()
            .map(|t| {
                let problem = Arc::clone(&problem);
                let config = Arc::clone(&self.config);
                let backend = Arc::clone(&self.backend);
                let t = t.clone();
                rt.spawn_blocking(move || {
                    run_exact_native_qml_probs(&problem, &config, backend.as_ref(), &t)
                })
            })
            .collect();

        Python::with_gil(|py| {
            let all_probs = py.allow_threads(|| {
                rt.block_on(async {
                    let mut out = Vec::with_capacity(handles.len());
                    for h in handles {
                        let single = h.await.map_err(|e| {
                            EvaluationError::Python(pyo3::exceptions::PyRuntimeError::new_err(
                                format!("exact native QML gradient task failed: {e}"),
                            ))
                        })?;
                        out.push(single?);
                    }
                    Ok::<_, EvaluationError>(out)
                })
            })?;
            py.check_signals().map_err(EvaluationError::Python)?;

            // Assemble the gradient from the exact probabilities (cheap, no
            // circuits). Branch on the loss exactly as `NativeQmlOracle` does for
            // counts: the categorical loss reads every class's expectation and
            // combines them via the multiclass chain rule; a scalar loss reads
            // only ⟨O₀⟩.
            let grad = if problem.loss() == Loss::CategoricalCrossEntropy {
                let base_expectations =
                    problem.expectations_per_class_from_probabilities(&all_probs[0])?;
                param_indices
                    .iter()
                    .enumerate()
                    .map(|(j, _)| {
                        let plus = &all_probs[1 + 2 * j];
                        let minus = &all_probs[1 + 2 * j + 1];
                        problem.param_gradient_categorical_exact(&base_expectations, plus, minus)
                    })
                    .collect::<Result<Vec<f64>, _>>()?
            } else {
                let base_expectations = problem.expectations_from_probabilities(&all_probs[0])?;
                param_indices
                    .iter()
                    .enumerate()
                    .map(|(j, _)| {
                        let plus = &all_probs[1 + 2 * j];
                        let minus = &all_probs[1 + 2 * j + 1];
                        problem.param_gradient_exact(&base_expectations, plus, minus)
                    })
                    .collect::<Result<Vec<f64>, _>>()?
            };
            Ok(grad)
        })
    }

    /// Evaluate `theta` against the **full** dataset, bypassing any configured
    /// minibatch — the exact-mode twin of
    /// [`NativeQmlOracle::evaluate_full`](crate::evaluation::NativeQmlOracle).
    /// Used once, after optimization ends, to report an honest final fitness
    /// (design doc §17) — see `bindings/qml.rs`.
    pub fn evaluate_full(&self, theta: &[f64]) -> Result<f64, EvaluationError> {
        evaluate_exact_native_qml_single(&self.problem, &self.config, self.backend.as_ref(), theta)
    }
}

/// Run one candidate `theta`'s whole training-set circuit batch and return the
/// per-sample **exact** probabilities, in the same (stable, sample-major) order
/// as [`QmlProblem::bind_batch`]. The exact-mode counterpart of
/// `run_native_qml_counts`. Any failure is an [`EvaluationError`], never a panic.
fn run_exact_native_qml_probs(
    problem: &QmlProblem,
    config: &ExecutionConfig,
    backend: &NativeStatevectorBackend,
    theta: &[f64],
) -> Result<Vec<HashMap<String, f64>>, EvaluationError> {
    // Bind the candidate into one native circuit per training sample (C-8 (a)).
    let bound: Vec<BoundCircuit> = problem
        .bind_batch(theta)?
        .into_iter()
        .map(BoundCircuit::Native)
        .collect();

    // The native backend takes the whole batch at once (no QPU/shot batching on
    // this path), so a single exact read-out call covers every sample.
    Ok(backend.run_circuits_exact(&bound, config)?)
}

/// Evaluate one candidate `theta` against the whole training set, returning the
/// exact fitness computed by [`QmlProblem::fitness_from_probabilities`].
fn evaluate_exact_native_qml_single(
    problem: &QmlProblem,
    config: &ExecutionConfig,
    backend: &NativeStatevectorBackend,
    theta: &[f64],
) -> Result<f64, EvaluationError> {
    let all_probs = run_exact_native_qml_probs(problem, config, backend, theta)?;
    Ok(problem.fitness_from_probabilities(&all_probs)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::{BackendConfig, OptLevel};
    use polypus_circuit::CircuitError;
    use polypus_qml::{
        Dataset, Decision, Loss, Observable, Pauli, PauliString, QmlProblem, QuantumModel, Readout,
        RotationAxis,
    };

    // `small_problem`/`categorical_problem` are shared with the sampled oracle's
    // tests, so they live in one place rather than being copied per file.
    use crate::evaluation::test_support::{categorical_problem, small_problem};

    /// A `LocalNative` execution config. `shots`/`seed` are irrelevant to the
    /// exact path, so their values here are arbitrary and never observed.
    fn native_config() -> Arc<ExecutionConfig> {
        Arc::new(ExecutionConfig {
            id: "exact_native_qml_oracle_test".to_string(),
            shots: 256,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: BackendConfig::LocalNative,
            opt_level: OptLevel::default(),
            seed: Some(7),
        })
    }

    fn oracle(problem: QmlProblem, errors: OracleErrorSlot) -> ExactNativeQmlOracle {
        ExactNativeQmlOracle {
            problem: Arc::new(problem),
            config: native_config(),
            // Any seed: the exact path never samples, so it does not matter.
            backend: Arc::new(NativeStatevectorBackend::new(7)),
            errors,
            minibatch: None,
        }
    }

    /// A six-sample problem (well-separated) so a batch of 3 is a genuine subset.
    fn six_sample_problem() -> QmlProblem {
        let readout = Readout::new(
            vec![
                Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())])
                    .unwrap(),
            ],
            Decision::Sign,
        )
        .unwrap();
        let model = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(readout);
        let ds = Dataset::from_rows(
            &[
                vec![0.30, 0.35],
                vec![0.40, 0.30],
                vec![0.35, 0.40],
                vec![2.80, 2.75],
                vec![2.90, 2.80],
                vec![2.75, 2.90],
            ],
            &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let compiled = model.compile(ds.num_features()).unwrap();
        QmlProblem::new(compiled, ds, Loss::Hinge).unwrap()
    }

    /// An exact oracle over [`six_sample_problem`] with minibatching active.
    fn oracle_with_minibatch(batch_size: usize, seed: u64) -> ExactNativeQmlOracle {
        ExactNativeQmlOracle {
            problem: Arc::new(six_sample_problem()),
            config: native_config(),
            backend: Arc::new(NativeStatevectorBackend::new(seed)),
            errors: OracleErrorSlot::new(),
            minibatch: Some(MinibatchConfig::new(batch_size, seed)),
        }
    }

    /// A well-formed batch returns exactly `candidates.len()` finite fitnesses
    /// and records no error.
    #[test]
    fn evaluate_batch_returns_one_finite_fitness_per_candidate() {
        pyo3::prepare_freethreaded_python();
        let errors = OracleErrorSlot::new();
        let oracle = oracle(small_problem(), errors.clone());
        let candidates = vec![vec![0.1_f64; 8], vec![0.2_f64; 8], vec![0.3_f64; 8]];
        let out = oracle.evaluate_batch(&candidates);
        assert_eq!(out.len(), candidates.len());
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(errors.take().is_none(), "a valid batch records no error");
    }

    /// A real failure (a `θ` of the wrong length) yields the finite `0.0`
    /// sentinel in every position and leaves the true error recoverable.
    #[test]
    fn evaluate_batch_sentinels_on_failure_and_records_error() {
        pyo3::prepare_freethreaded_python();
        let errors = OracleErrorSlot::new();
        let oracle = oracle(small_problem(), errors.clone());
        // 3 parameters where the model needs 8: binding fails per candidate.
        let candidates = vec![vec![0.1_f64; 3], vec![0.2_f64; 3]];
        let out = oracle.evaluate_batch(&candidates);
        assert_eq!(out, vec![0.0, 0.0]);
        let err = errors
            .take()
            .expect("the real error is recoverable via the slot");
        assert!(
            matches!(
                err,
                EvaluationError::Qml(polypus_qml::QmlError::Circuit(
                    CircuitError::WrongNumberOfParams { .. }
                ))
            ),
            "unexpected error variant: {err:?}"
        );
    }

    /// The gradient batch is finite and has the requested length (scalar loss).
    #[test]
    fn gradient_batch_is_finite() {
        pyo3::prepare_freethreaded_python();
        let oracle = oracle(small_problem(), OracleErrorSlot::new());
        let dims = 8;
        let theta = vec![0.15_f64; dims];
        let grad = oracle.gradient_batch(&theta, dims);
        assert_eq!(grad.len(), dims);
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    /// The categorical gradient branch is finite and correctly sized.
    #[test]
    fn gradient_batch_is_finite_categorical() {
        pyo3::prepare_freethreaded_python();
        let oracle = oracle(categorical_problem(), OracleErrorSlot::new());
        let dims = 8;
        let theta = vec![0.15_f64; dims];
        let grad = oracle.gradient_batch(&theta, dims);
        assert_eq!(grad.len(), dims);
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    /// Stronger reproducibility than the sampled path can offer: two runs of
    /// `evaluate_batch` and `gradient_batch` are byte-identical **without any
    /// seed being fixed for this purpose**, because the exact path draws no shot
    /// noise at all. Two backends built with *different* seeds must still agree.
    #[test]
    fn evaluate_and_gradient_are_bit_identical_regardless_of_seed() {
        pyo3::prepare_freethreaded_python();
        let a = ExactNativeQmlOracle {
            problem: Arc::new(small_problem()),
            config: native_config(),
            backend: Arc::new(NativeStatevectorBackend::new(1)),
            errors: OracleErrorSlot::new(),
            minibatch: None,
        };
        let b = ExactNativeQmlOracle {
            problem: Arc::new(small_problem()),
            config: native_config(),
            backend: Arc::new(NativeStatevectorBackend::new(999)),
            errors: OracleErrorSlot::new(),
            minibatch: None,
        };
        let candidates: Vec<Vec<f64>> = (0..6).map(|k| vec![0.05 + 0.03 * k as f64; 8]).collect();
        assert_eq!(a.evaluate_batch(&candidates), b.evaluate_batch(&candidates));

        let theta = vec![0.2_f64; 8];
        assert_eq!(a.gradient_batch(&theta, 8), b.gradient_batch(&theta, 8));
    }

    // ── Minibatching (design doc §17) ────────────────────────────────────────

    /// `gradient_batch` draws exactly one minibatch per call over `dims > 1`
    /// (white-box on the counter), mirroring the sampled oracle's guarantee. The
    /// exact path removes shot noise, not the minibatch coherence requirement.
    #[test]
    fn gradient_batch_draws_one_minibatch_for_the_whole_call() {
        pyo3::prepare_freethreaded_python();
        let oracle = oracle_with_minibatch(3, 20);
        let dims = 8;
        let theta = vec![0.15_f64; dims];
        let grad = oracle.gradient_batch(&theta, dims);
        assert_eq!(grad.len(), dims);
        assert!(grad.iter().all(|g| g.is_finite()));
        assert_eq!(oracle.minibatch.as_ref().unwrap().calls_so_far(), 1);
    }

    /// Each `evaluate_batch` advances the counter by one, and `evaluate_full`
    /// scores the whole dataset regardless of how many minibatches were drawn.
    /// In exact mode `evaluate_full` is also seed-independent (no sampling).
    #[test]
    fn evaluate_full_ignores_minibatch_and_is_exact() {
        pyo3::prepare_freethreaded_python();
        let oracle = oracle_with_minibatch(3, 20);
        let theta = vec![0.15_f64; 8];
        let full_before = oracle.evaluate_full(&theta).unwrap();

        let _ = oracle.evaluate_batch(std::slice::from_ref(&theta));
        let _ = oracle.evaluate_batch(std::slice::from_ref(&theta));
        assert_eq!(oracle.minibatch.as_ref().unwrap().calls_so_far(), 2);

        let full_after = oracle.evaluate_full(&theta).unwrap();
        assert_eq!(full_before, full_after);
        assert!(oracle.errors.take().is_none());
    }

    /// C-7 (stronger on the exact path): two oracles with the same batch_size and
    /// the same call sequence are byte-identical even with **different** backend
    /// seeds — the minibatch selection depends only on `MinibatchConfig`'s seed,
    /// and there is no shot noise. Uses the same minibatch seed but different
    /// backend seeds to prove the result is a function of the former alone.
    #[test]
    fn minibatch_is_reproducible_and_seed_independent_for_sampling() {
        pyo3::prepare_freethreaded_python();
        let a = ExactNativeQmlOracle {
            problem: Arc::new(six_sample_problem()),
            config: native_config(),
            backend: Arc::new(NativeStatevectorBackend::new(1)),
            errors: OracleErrorSlot::new(),
            minibatch: Some(MinibatchConfig::new(3, 20)),
        };
        let b = ExactNativeQmlOracle {
            problem: Arc::new(six_sample_problem()),
            config: native_config(),
            backend: Arc::new(NativeStatevectorBackend::new(999)),
            errors: OracleErrorSlot::new(),
            minibatch: Some(MinibatchConfig::new(3, 20)),
        };
        let candidates: Vec<Vec<f64>> = (0..5).map(|k| vec![0.05 + 0.03 * k as f64; 8]).collect();
        assert_eq!(a.evaluate_batch(&candidates), b.evaluate_batch(&candidates));
        let theta = vec![0.2_f64; 8];
        assert_eq!(a.gradient_batch(&theta, 8), b.gradient_batch(&theta, 8));
    }

    /// A six-sample problem whose last two samples are **contradictory**:
    /// identical features, opposite labels. Used by the minibatch/early-stopping
    /// investigation below.
    fn contradictory_pair_problem() -> QmlProblem {
        let readout = Readout::new(
            vec![
                Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())])
                    .unwrap(),
            ],
            Decision::Sign,
        )
        .unwrap();
        let model = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(readout);
        let ds = Dataset::from_rows(
            &[
                vec![0.30, 0.35],
                vec![0.40, 0.30],
                vec![2.80, 2.75],
                vec![2.90, 2.80],
                vec![1.55, 1.60],
                vec![1.55, 1.60],
            ],
            &[-1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
        )
        .unwrap();
        let compiled = model.compile(ds.num_features()).unwrap();
        QmlProblem::new(compiled, ds, Loss::Hinge).unwrap()
    }

    /// The analytic core of the minibatch / gradient-norm early-stopping
    /// interaction (design doc §17): **a minibatch gradient can be exactly zero
    /// at a `θ` where the full-dataset gradient is nowhere near zero.**
    ///
    /// For a hinge loss, two samples with identical features and opposite labels
    /// have `e` and `de/dθ` in common, and their loss derivatives are `−y·de/dθ`,
    /// so they cancel **term by term**: the pair's gradient is `0` in exact
    /// arithmetic, not merely small. Every other sample keeps contributing, so the
    /// full-dataset norm stays O(0.1–0.5) — two orders of magnitude above the
    /// `tolerance = 0.01` that `polypus.Adam`/`polypus.QNG` default to.
    ///
    /// `AlgorithmQNGArgs`/`AlgorithmAdamArgs` compare exactly this per-iteration
    /// norm against `tolerance` to set `converged`. Because the norm is *exactly*
    /// zero, no `tolerance`, however small, keeps a single such iteration below
    /// the bar — which is why the fix was `patience` (requiring `patience`
    /// *consecutive* sub-tolerance iterations, default 3) rather than a tighter
    /// threshold. See the write-up in the design doc §17 and the note beside
    /// C-5/C-7 in `CONTRACTS.md`.
    ///
    /// This test goes through **no optimizer** — it reads the two gradient norms
    /// straight off the oracle — so `patience` does not touch what it asserts, and
    /// the numerical fact both documents rest on is unchanged by the fix: the
    /// cancellation is still exact, and it is still what `patience` has to
    /// tolerate rather than prevent.
    #[test]
    fn minibatch_gradient_can_vanish_where_the_full_gradient_does_not() {
        pyo3::prepare_freethreaded_python();
        let norm = |g: &[f64]| g.iter().map(|x| x * x).sum::<f64>().sqrt();
        let dims = 8;

        let full = oracle(contradictory_pair_problem(), OracleErrorSlot::new());
        // The contradictory pair on its own — what a `batch_size = 2` minibatch
        // draws whenever the shuffle puts samples 4 and 5 first.
        let pair_only = full.problem.subset(&[4, 5]).unwrap();
        let pair = oracle(pair_only, OracleErrorSlot::new());

        // Checked across several θ so the cancellation is shown to be structural,
        // not a coincidence at one point of parameter space.
        for theta_value in [0.15_f64, 0.5, 1.0, 2.0] {
            let theta = vec![theta_value; dims];
            let pair_norm = norm(&pair.gradient_batch(&theta, dims));
            let full_norm = norm(&full.gradient_batch(&theta, dims));
            assert_eq!(
                pair_norm, 0.0,
                "the contradictory pair's gradient must cancel exactly at θ = {theta_value}"
            );
            assert!(
                full_norm > 0.2,
                "the full-dataset gradient must stay far from zero at θ = {theta_value}, \
                 got {full_norm}"
            );
        }
    }
}
