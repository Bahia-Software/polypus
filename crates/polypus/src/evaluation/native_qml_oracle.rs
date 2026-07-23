use crate::evaluation::{EvaluationError, EvaluationOracle, OracleErrorSlot};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use polypus_optimizers::GradientOracle;
use polypus_qml::QmlProblem;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Oracle for native (pure-Rust) QML training (contract C-8).
///
/// It holds a [`QmlProblem`] — a compiled model, a training set and a loss
/// produced entirely by `polypus-qml` — and lets it bind parameters and score
/// counts in pure Rust, where [`QmlOracle`](crate::evaluation::QmlOracle) holds
/// Qiskit `QuantumCircuit`s and computes expectations through Python. It works on
/// **any** simulated backend: the native statevector simulator (GIL-free end to
/// end) or, via the C-1 seam, Aer / CUNQA's simulated QPUs.
///
/// For each candidate `θ` it binds `θ` into one circuit per training sample
/// ([`QmlProblem::bind_batch`]), runs them (batched by `max_batch_size`), and
/// turns the counts into a single fitness ([`QmlProblem::fitness_from_counts`],
/// `= −mean_loss`, since the optimizers maximise).
///
/// ## Concurrent evaluation (mirroring `QmlOracle` to the letter)
///
/// Each candidate is evaluated concurrently via Tokio `spawn_blocking`, exactly
/// like [`QmlOracle`](crate::evaluation::QmlOracle): one blocking task per
/// candidate, the GIL released around the join (`allow_threads` + `block_on`),
/// and a single [`Python::check_signals`] on the calling thread after the join
/// so the run stays interruptible at per-batch granularity.
///
/// This was **not** always safe. An earlier revision fell back to sequential
/// evaluation because [`NativeStatevectorBackend`] used to seed each circuit from
/// a *shared atomic counter* advanced by call-arrival order; concurrent workers
/// raced on that counter, so a native `qml.train` run was not reproducible
/// byte-for-byte (breaking contract C-7). That root cause is now fixed at the
/// backend: it derives every circuit's seed purely from the circuit's own content
/// (an FNV-1a hash of its OpenQASM text) plus its batch index, with **no shared
/// mutable state**. Concurrent candidates therefore can never race for seed
/// assignment, so this oracle is free to reclaim the genuine cross-candidate
/// parallelism that is the whole point of the native path.
///
/// [`NativeStatevectorBackend`]: crate::infrastructure::NativeStatevectorBackend
pub struct NativeQmlOracle {
    /// The trainable problem: bind parameters in, get fitness out (C-8).
    pub problem: QmlProblem,
    pub config: Arc<ExecutionConfig>,
    pub backend: Arc<dyn QuantumBackend>,
    /// Shared with the `qml.train` entry point: the first evaluation failure is
    /// recorded here and surfaced as a `PyErr` after `optimize` returns, since
    /// [`EvaluationOracle::evaluate_batch`] cannot return a `Result`.
    pub errors: OracleErrorSlot,
}

impl EvaluationOracle for NativeQmlOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        // Once evaluation has failed, stop doing work: return finite sentinels
        // and let the entry point surface the recorded error.
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

impl GradientOracle for NativeQmlOracle {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        // Same sentinel discipline as `evaluate_batch`: once evaluation has
        // failed, do no work and let the entry point surface the recorded error.
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

impl NativeQmlOracle {
    /// Fallible core of [`EvaluationOracle::evaluate_batch`]. Kept separate so
    /// the trait method (which must return `Vec<f64>`) can record any error and
    /// yield finite sentinels while the entry point re-raises it.
    fn try_evaluate(&self, candidates: &[Vec<f64>]) -> Result<Vec<f64>, EvaluationError> {
        let rt = crate::utils::tokio_runtime().map_err(|e| {
            EvaluationError::Python(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to start the Tokio runtime for native QML evaluation: {e}"
            )))
        })?;

        // Clone the problem once per batch into an `Arc` so every candidate's
        // `spawn_blocking` task holds an owned, `'static`, cheaply-shared handle
        // to it — the native counterpart to how `QmlOracle` hands its already-
        // `Arc` config/backend (and `clone_ref`'d circuits) to its workers.
        let problem = Arc::new(self.problem.clone());

        let handles: Vec<_> = candidates
            .iter()
            .map(|theta| {
                let problem = Arc::clone(&problem);
                let config = Arc::clone(&self.config);
                let backend = Arc::clone(&self.backend);
                let theta = theta.clone();
                rt.spawn_blocking(move || {
                    evaluate_native_qml_single(&problem, &config, backend.as_ref(), &theta)
                })
            })
            .collect();

        // The calling thread entered from a PyO3 `#[pyfunction]` and still holds
        // the GIL. A native evaluation never touches Python, but the entry point
        // already released the GIL around the whole `optimize()` (ENGINEERING §3);
        // release it again here while blocking on the workers so nothing waits on
        // it. After the join, check signals **once** on this (main) thread — where
        // `PyErr_CheckSignals` is not a no-op — so `qml.train` stays interruptible
        // at per-batch granularity; a pending SIGINT (Ctrl+C) becomes a
        // `KeyboardInterrupt` carried verbatim via `EvaluationError::Python`.
        Python::with_gil(|py| {
            let out = py.allow_threads(|| {
                rt.block_on(async {
                    let mut out = Vec::with_capacity(handles.len());
                    for h in handles {
                        // A `JoinError` means the worker task itself panicked;
                        // turn it into a typed error rather than re-panicking.
                        let single = h.await.map_err(|e| {
                            EvaluationError::Python(pyo3::exceptions::PyRuntimeError::new_err(
                                format!("native QML evaluation task failed: {e}"),
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
    /// parameter-shift gradient (design doc §17) for the requested
    /// `param_indices`, one value each, in order.
    ///
    /// The fitness composes a nonlinear [`Loss`](polypus_qml::Loss) over the raw
    /// per-sample expectations, so the shift rule needs the chain rule: it runs
    /// the base `θ` once to get the base expectations, then `θ ± π/2·e_k` for
    /// each requested `k`, and combines them via [`QmlProblem::param_gradient`].
    ///
    /// **Cost.** This runs `1 + 2·|param_indices|` circuit batches. A full QNG
    /// step therefore submits `2·dims + 1` gradient batches per iteration — one
    /// more than the old finite-difference stencil's `2·dims`, because the chain
    /// rule needs the base expectations separately (finite differences did not).
    /// That extra base batch is a small, deliberately accepted cost: it is *not*
    /// shared with QNG's step-4 fitness-tracking batch, since threading one
    /// evaluation through two independent trait calls would complicate the seam
    /// for a marginal saving (out of scope for this phase).
    fn try_gradient(
        &self,
        theta: &[f64],
        param_indices: &[usize],
    ) -> Result<Vec<f64>, EvaluationError> {
        let rt = crate::utils::tokio_runtime().map_err(|e| {
            EvaluationError::Python(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to start the Tokio runtime for native QML gradient: {e}"
            )))
        })?;

        // Clone the problem once per gradient into an `Arc`, exactly as
        // `try_evaluate` does, so every `spawn_blocking` task owns a cheap,
        // `'static` handle to it.
        let problem = Arc::new(self.problem.clone());

        // The parameter vectors to run: base θ first, then θ ± π/2·e_k for each
        // requested k. The base batch is what makes this cost 2·|indices| + 1.
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

        // One blocking task per parameter vector. Running these concurrently is
        // safe for the same reason `try_evaluate` documents: the phase-4 seeding
        // fix makes `NativeStatevectorBackend` derive every circuit's seed from
        // the circuit's own content (not shared mutable state), so concurrent
        // batches can never race for seed assignment (contract C-7).
        let handles: Vec<_> = thetas
            .iter()
            .map(|t| {
                let problem = Arc::clone(&problem);
                let config = Arc::clone(&self.config);
                let backend = Arc::clone(&self.backend);
                let t = t.clone();
                rt.spawn_blocking(move || {
                    run_native_qml_counts(&problem, &config, backend.as_ref(), &t)
                })
            })
            .collect();

        // Release the GIL around the join and check signals once afterwards,
        // identically to `try_evaluate` — the native path never touches Python
        // during the work, but the calling thread still holds the GIL.
        Python::with_gil(|py| {
            let all_counts = py.allow_threads(|| {
                rt.block_on(async {
                    let mut out = Vec::with_capacity(handles.len());
                    for h in handles {
                        let single = h.await.map_err(|e| {
                            EvaluationError::Python(pyo3::exceptions::PyRuntimeError::new_err(
                                format!("native QML gradient task failed: {e}"),
                            ))
                        })?;
                        out.push(single?);
                    }
                    Ok::<_, EvaluationError>(out)
                })
            })?;
            py.check_signals().map_err(EvaluationError::Python)?;

            // Assemble the gradient from the counts (cheap, no circuits): base
            // expectations once, then one chain-rule component per requested k.
            let base_expectations = problem.expectations_from_counts(&all_counts[0])?;
            let grad = param_indices
                .iter()
                .enumerate()
                .map(|(j, _)| {
                    let plus = &all_counts[1 + 2 * j];
                    let minus = &all_counts[1 + 2 * j + 1];
                    problem.param_gradient(&base_expectations, plus, minus)
                })
                .collect::<Result<Vec<f64>, _>>()?;
            Ok(grad)
        })
    }
}

/// Run one candidate `theta`'s whole training-set circuit batch and return the
/// per-sample counts, in the same (stable, sample-major) order as
/// [`QmlProblem::bind_batch`].
///
/// Binds `theta` into one native circuit per training sample, runs them in
/// batches of `max_batch_size`, and reassembles the counts. Shared by the
/// fitness evaluation ([`evaluate_native_qml_single`]) and the parameter-shift
/// gradient ([`NativeQmlOracle::try_gradient`]), which both need exactly this
/// "θ → per-sample counts" step before diverging (fitness vs. chain-rule
/// gradient). Any failure is returned as an [`EvaluationError`], never panics.
fn run_native_qml_counts(
    problem: &QmlProblem,
    config: &ExecutionConfig,
    backend: &dyn QuantumBackend,
    theta: &[f64],
) -> Result<Vec<HashMap<String, u64>>, EvaluationError> {
    // Bind the candidate into one native circuit per training sample (C-8 (a)).
    let bound: Vec<BoundCircuit> = problem
        .bind_batch(theta)?
        .into_iter()
        .map(BoundCircuit::Native)
        .collect();

    let batch_size = backend.max_batch_size(bound.len()).max(1);
    let mut all_counts: Vec<HashMap<String, u64>> = Vec::with_capacity(bound.len());
    for chunk in bound.chunks(batch_size) {
        // Counts come back one dict per circuit, in submission order (C-3), so
        // extending preserves the sample-major order the callers rely on.
        all_counts.extend(backend.run_circuits(chunk, config)?);
    }
    Ok(all_counts)
}

/// Evaluate one candidate `theta` against the whole training set, returning the
/// fitness computed by [`QmlProblem::fitness_from_counts`].
fn evaluate_native_qml_single(
    problem: &QmlProblem,
    config: &ExecutionConfig,
    backend: &dyn QuantumBackend,
    theta: &[f64],
) -> Result<f64, EvaluationError> {
    let all_counts = run_native_qml_counts(problem, config, backend, theta)?;
    Ok(problem.fitness_from_counts(&all_counts)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::{BackendConfig, NativeStatevectorBackend, OptLevel};
    use polypus_circuit::CircuitError;
    use polypus_qml::{
        Dataset, Decision, Loss, Observable, Pauli, PauliString, QmlProblem, QuantumModel, Readout,
        RotationAxis,
    };

    /// A tiny, fully-Rust `QmlProblem`: a 2-qubit angle-encoder + hardware-efficient
    /// ansatz reading `⟨Z₀⟩` with a `Sign` decision, trained with `Hinge` over two
    /// well-separated samples. Its compiled model reserves 8 trainable parameters.
    fn small_problem() -> QmlProblem {
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
        let ds = Dataset::from_rows(&[vec![0.4, 0.5], vec![2.6, 2.7]], &[-1.0, 1.0]).unwrap();
        let compiled = model.compile(ds.num_features()).unwrap();
        QmlProblem::new(compiled, ds, Loss::Hinge).unwrap()
    }

    /// A `LocalNative` execution config seeded deterministically, with a
    /// caller-chosen shot count (the gradient test wants many more shots than
    /// the fitness tests to tame sampling noise).
    fn native_config_with_shots(shots: u32) -> Arc<ExecutionConfig> {
        Arc::new(ExecutionConfig {
            id: "native_qml_oracle_test".to_string(),
            shots,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: BackendConfig::LocalNative,
            opt_level: OptLevel::default(),
            seed: Some(7),
        })
    }

    /// A `LocalNative` execution config seeded deterministically.
    fn native_config() -> Arc<ExecutionConfig> {
        native_config_with_shots(256)
    }

    fn oracle(errors: OracleErrorSlot) -> NativeQmlOracle {
        NativeQmlOracle {
            problem: small_problem(),
            config: native_config(),
            backend: Arc::new(NativeStatevectorBackend::new(7)),
            errors,
        }
    }

    /// Oracle over `small_problem()` with a caller-chosen shot count.
    fn oracle_with_shots(errors: OracleErrorSlot, shots: u32) -> NativeQmlOracle {
        NativeQmlOracle {
            problem: small_problem(),
            config: native_config_with_shots(shots),
            backend: Arc::new(NativeStatevectorBackend::new(7)),
            errors,
        }
    }

    /// The exact parameter-shift `gradient_batch` matches a central finite
    /// difference of `evaluate_batch` for the existing `small_problem()`.
    ///
    /// Two things make the finite-difference step **not** `1e-5`. First, each
    /// `evaluate_batch` is shot-noisy, and `θ ± h` produce different circuits
    /// (so different content-derived seeds — the noise is independent, not
    /// cancelling); dividing that noise by `2h` blows it up as `h → 0` (at
    /// `h=1e-5` the discrepancy is ~247, pure noise). Second, `h` must stay
    /// large enough that the true `2h·f'` signal dominates the sampling noise.
    /// `h=0.1` with `shots=65536` sits in that window: the reference finite
    /// difference then tracks the (more accurate) parameter-shift value to well
    /// under the loose tolerance below. The run is fully deterministic (fixed
    /// seed + content-derived shot seeds), so this comparison is reproducible.
    #[test]
    fn gradient_batch_matches_finite_difference() {
        pyo3::prepare_freethreaded_python();
        let oracle = oracle_with_shots(OracleErrorSlot::new(), 65536);
        // small_problem() reserves 8 trainable parameters.
        let dims = 8;
        let theta = vec![0.15_f64; dims];

        let grad = oracle.gradient_batch(&theta, dims);
        assert_eq!(grad.len(), dims);
        assert!(grad.iter().all(|g| g.is_finite()));

        let h = 0.1_f64;
        for k in 0..dims {
            let mut tp = theta.clone();
            let mut tm = theta.clone();
            tp[k] += h;
            tm[k] -= h;
            let fp = oracle.evaluate_batch(&[tp])[0];
            let fm = oracle.evaluate_batch(&[tm])[0];
            let fd = (fp - fm) / (2.0 * h);
            assert!(
                (grad[k] - fd).abs() < 5e-2,
                "param {k}: parameter-shift {} vs finite difference {fd}",
                grad[k]
            );
        }
    }

    /// A well-formed batch returns exactly `candidates.len()` finite fitnesses and
    /// records no error (contract C-5 length + C-8 (b) finiteness, via the oracle).
    #[test]
    fn evaluate_batch_returns_one_finite_fitness_per_candidate() {
        // `evaluate_batch` acquires the GIL only for the per-candidate signal
        // check; the interpreter must exist, but no Qiskit/Aer/`polypus_python`
        // is involved — the whole evaluation is native (ENGINEERING §3).
        pyo3::prepare_freethreaded_python();
        let errors = OracleErrorSlot::new();
        let oracle = oracle(errors.clone());
        // The compiled model reserves 8 parameters.
        let candidates = vec![vec![0.1_f64; 8], vec![0.2_f64; 8], vec![0.3_f64; 8]];
        let out = oracle.evaluate_batch(&candidates);
        assert_eq!(out.len(), candidates.len());
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(errors.take().is_none(), "a valid batch records no error");
    }

    /// A real failure (a `θ` of the wrong length) yields the finite `0.0` sentinel
    /// in every position **and** leaves the true error recoverable via the shared
    /// slot — the sentinel + `OracleErrorSlot` contract the entry point relies on.
    #[test]
    fn evaluate_batch_sentinels_on_failure_and_records_error() {
        pyo3::prepare_freethreaded_python();
        let errors = OracleErrorSlot::new();
        let oracle = oracle(errors.clone());
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

    /// Reproducibility (C-7) under **concurrent** evaluation: two runs of the same
    /// candidates on two backends built with the same seed produce byte-identical
    /// fitnesses. Now that candidates are evaluated concurrently via
    /// `spawn_blocking`, this is the real guard against the seed-assignment race —
    /// if the backend still derived seeds from shared mutable state, the two runs
    /// would diverge under nondeterministic worker scheduling. A larger candidate
    /// set widens the window in which such a race could surface.
    #[test]
    fn evaluate_batch_is_reproducible_for_a_fixed_seed() {
        pyo3::prepare_freethreaded_python();
        let a = oracle(OracleErrorSlot::new());
        let b = oracle(OracleErrorSlot::new());
        let candidates: Vec<Vec<f64>> = (0..12).map(|k| vec![0.05 + 0.03 * k as f64; 8]).collect();
        assert_eq!(a.evaluate_batch(&candidates), b.evaluate_batch(&candidates));
    }
}
