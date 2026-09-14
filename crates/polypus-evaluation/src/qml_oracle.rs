use crate::{
    assign_parameters_qiskit, CostObservable, EvaluationError, EvaluationOracle, OracleErrorSlot,
};
use polypus_infrastructure::{
    BoundCircuit, CancelToken, CircuitTask, ExecutionConfig, Planner, QuantumBackend,
};
use polypus_orchestration::{OracleFactory, Resources};
use pyo3::prelude::*;
use std::sync::Arc;

/// Oracle for QML training with feature-map encoding.
///
/// Holds N pre-bound training circuits (one per training sample, with feature-map
/// parameters already fixed). For each candidate `θ`, it binds `θ` to every
/// training circuit, runs the whole population's circuits through the
/// [`Planner`], and returns the **mean** expectation value per candidate as the
/// fitness.
///
/// The oracle owns only the *what*: bind the candidates, build one flat 2D batch
/// of `candidates × training_circuits` tasks, reduce each candidate's expectations
/// to their mean, and validate contract C-5. The [`Planner`] owns the *how* —
/// waves, the concurrency/memory cap, the between-wave `check_signals`
/// (ENGINEERING §3) and cancellation — so this oracle no longer runs its own
/// Tokio dispatch loop. The GIL still serialises the Qiskit binding and the Aer
/// simulation calls; genuine parallelism arrives with a native QML backend.
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
    fn try_evaluate(&self, candidates: &[Vec<f64>]) -> Result<Vec<f64>, EvaluationError> {
        let n_train = self.training_circuits.len();

        // Bind every (candidate, training-circuit) pair eagerly into one flat
        // batch, **candidate-major**: candidate `i`'s circuits occupy
        // `[i*n_train, (i+1)*n_train)`. Binding is Qiskit-specific (feature-map
        // pre-binding) and re-acquires the GIL internally, exactly as the former
        // per-candidate path did — the concurrency/memory cap is the Planner's job.
        let mut bound: Vec<BoundCircuit> = Vec::with_capacity(candidates.len() * n_train);
        for theta in candidates {
            for qc_xi in &self.training_circuits {
                bound.push(BoundCircuit::Qiskit(assign_parameters_qiskit(
                    qc_xi, theta,
                )?));
            }
        }

        // One 2D task per circuit — uniform shots in training; `shots` comes from
        // the task, the single source of truth the Planner reads.
        let tasks: Vec<CircuitTask> = bound
            .iter()
            .map(|circuit| CircuitTask {
                circuit,
                shots: self.config.shots,
            })
            .collect();

        // Delegate execution + reduction to the Planner: it owns the waves, the
        // concurrency cap, the between-wave `check_signals` and the shot merge.
        // This dissolves the former per-chunk `run_and_evaluate` loop and the
        // per-candidate Tokio dispatch.
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
        // as the former per-candidate loop, so the mean is byte-identical.
        let means: Vec<f64> = (0..candidates.len())
            .map(|i| {
                let slice = &expectations[i * n_train..(i + 1) * n_train];
                slice.iter().sum::<f64>() / n_train as f64
            })
            .collect();

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
