use crate::evaluation::{EvaluationError, EvaluationOracle, OracleErrorSlot};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
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
/// ## Sequential evaluation (and why it is *not* `QmlOracle`'s `spawn_blocking`)
///
/// Like `QmlOracle` this runs N training circuits per candidate; unlike it, it
/// evaluates candidates **sequentially**, mirroring [`VqcOracle`]'s structure
/// rather than `QmlOracle`'s concurrent `spawn_blocking`. This is required for
/// reproducibility: [`NativeStatevectorBackend`] seeds each circuit from a
/// per-batch atomic counter (`base_seed + counter`), so counts are deterministic
/// only when circuits are submitted in a deterministic order. `QmlOracle`'s
/// concurrent workers race on that counter, which is harmless for Aer (its seed
/// is not counter-based) but would make a native `qml.train` run non-reproducible
/// — breaking contract C-7 and decision J of the phase-4 plan. `VqcOracle` (the
/// existing native training path) is sequential for exactly this reason. The
/// per-batch signal check that keeps the run interruptible is kept.
///
/// [`VqcOracle`]: crate::evaluation::VqcOracle
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

impl NativeQmlOracle {
    /// Fallible core of [`EvaluationOracle::evaluate_batch`]. Kept separate so
    /// the trait method (which must return `Vec<f64>`) can record any error and
    /// yield finite sentinels while the entry point re-raises it.
    fn try_evaluate(&self, candidates: &[Vec<f64>]) -> Result<Vec<f64>, EvaluationError> {
        let mut results = Vec::with_capacity(candidates.len());
        for theta in candidates {
            // The entry point released the GIL around the whole `optimize()`
            // (see `qml.train` and ENGINEERING §3). A native evaluation never
            // touches Python, so a pending SIGINT (Ctrl+C) would otherwise go
            // unseen until the run ends: reacquire the GIL only to turn it into a
            // `KeyboardInterrupt` at this safe per-candidate boundary, then drop
            // it again for the GIL-free work. The exception is carried verbatim
            // via `EvaluationError::Python` and re-raised as itself by the entry
            // point.
            Python::with_gil(|py| py.check_signals()).map_err(EvaluationError::Python)?;
            results.push(evaluate_native_qml_single(
                &self.problem,
                &self.config,
                self.backend.as_ref(),
                theta,
            )?);
        }
        Ok(results)
    }
}

/// Evaluate one candidate `theta` against the whole training set.
///
/// Binds `theta` into one native circuit per training sample
/// ([`QmlProblem::bind_batch`]), runs them in batches of `max_batch_size`,
/// reassembles the counts in the same (stable, sample-major) order, and returns
/// the fitness computed by [`QmlProblem::fitness_from_counts`]. Any failure is
/// returned as an [`EvaluationError`] instead of panicking.
fn evaluate_native_qml_single(
    problem: &QmlProblem,
    config: &ExecutionConfig,
    backend: &dyn QuantumBackend,
    theta: &[f64],
) -> Result<f64, EvaluationError> {
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
        // extending preserves the sample-major order `fitness_from_counts` needs.
        all_counts.extend(backend.run_circuits(chunk, config)?);
    }

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

    /// A `LocalNative` execution config seeded deterministically.
    fn native_config() -> Arc<ExecutionConfig> {
        Arc::new(ExecutionConfig {
            id: "native_qml_oracle_test".to_string(),
            shots: 256,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: BackendConfig::LocalNative,
            opt_level: OptLevel::default(),
            seed: Some(7),
        })
    }

    fn oracle(errors: OracleErrorSlot) -> NativeQmlOracle {
        NativeQmlOracle {
            problem: small_problem(),
            config: native_config(),
            backend: Arc::new(NativeStatevectorBackend::new(7)),
            errors,
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

    /// Reproducibility (C-7 / decision J): two runs of the same candidates on two
    /// backends built with the same seed produce byte-identical fitnesses. This is
    /// what the sequential evaluation buys — a concurrent oracle would race on the
    /// backend's per-batch seed counter and fail this.
    #[test]
    fn evaluate_batch_is_reproducible_for_a_fixed_seed() {
        pyo3::prepare_freethreaded_python();
        let a = oracle(OracleErrorSlot::new());
        let b = oracle(OracleErrorSlot::new());
        let candidates = vec![vec![0.1_f64; 8], vec![0.2_f64; 8], vec![0.3_f64; 8]];
        assert_eq!(a.evaluate_batch(&candidates), b.evaluate_batch(&candidates));
    }
}
