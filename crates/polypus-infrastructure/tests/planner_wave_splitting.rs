//! Issue #147 regression: `SequentialPlanner::execute` sizes its waves with the
//! *batch-aware* cap (`capabilities_for`), so a high-qubit batch on the native or
//! local backend is split into several memory-capped waves — with the between-wave
//! cancellation/interrupt checkpoint running once per wave — instead of one
//! uninterruptible wave.
//!
//! These live here (not in `polypus-backend`, where the `Planner` now lives)
//! because they exercise the *real* `NativeStatevectorBackend` / `LocalBackend`
//! capability arithmetic, which those backends only implement in this crate. The
//! cap decision is taken by the real backend (`WaveSpy::capabilities_for` delegates
//! to it), but the circuits are never simulated: a `WaveSpy` intercepts
//! `run_circuits`, records each wave's size and returns dummy valid counts — so a
//! 30-qubit circuit (16 GiB statevector ⇒ cap 1 under the default budget) never
//! allocates `2^30` amplitudes.

use std::sync::Mutex;

use polypus_circuit::ParameterizedCircuit;
use polypus_infrastructure::{
    BackendCapabilities, BackendError, BoundCircuit, CancelToken, CircuitTask, Counts,
    InfrastructureError, LocalBackend, NativeStatevectorBackend, OptLevel, Planner, QuantumBackend,
    RunParams, SequentialPlanner,
};

/// Wraps a real backend to borrow its batch-aware cap while stubbing out
/// execution: `run_circuits` records the wave size and returns one valid counts
/// map per circuit, so nothing is ever simulated. Optionally cancels a shared
/// token from inside the first wave.
struct WaveSpy<'b> {
    inner: &'b dyn QuantumBackend,
    calls: Mutex<Vec<usize>>,
    cancel_after_first: Option<CancelToken>,
}

impl QuantumBackend for WaveSpy<'_> {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        params: &RunParams,
    ) -> Result<Vec<Counts>, BackendError> {
        self.calls.lock().unwrap().push(qcs.len());
        if let Some(token) = &self.cancel_after_first {
            token.cancel();
        }
        Ok(qcs
            .iter()
            .map(|_| Counts::from([("0".to_string(), u64::from(params.shots))]))
            .collect())
    }

    // The whole point: waves are sized by the real backend's batch-aware cap.
    fn capabilities_for(
        &self,
        tasks: &[CircuitTask<'_>],
    ) -> Result<BackendCapabilities, InfrastructureError> {
        self.inner.capabilities_for(tasks)
    }
}

fn params() -> RunParams {
    RunParams {
        id: "wave-test".to_string(),
        shots: 8,
        seed: Some(7),
        opt_level: OptLevel::default(),
    }
}

/// Four 30-qubit tasks, referencing one zero-gate circuit (never simulated).
fn wide_batch(circuit: &BoundCircuit) -> Vec<CircuitTask<'_>> {
    (0..4).map(|_| CircuitTask { circuit, shots: 8 }).collect()
}

fn wide_native() -> BoundCircuit {
    BoundCircuit::Native(
        ParameterizedCircuit::new(30)
            .assign_parameters(&[])
            .unwrap(),
    )
}

/// Native: a high-qubit batch is split into several waves respecting the
/// memory-derived cap (cap 1 under the 16 GiB default ⇒ four single-circuit waves).
#[test]
fn execute_splits_high_qubit_native_batch_into_memory_capped_waves() {
    let native = NativeStatevectorBackend::new(0);
    let wide = wide_native();
    let tasks = wide_batch(&wide);
    let spy = WaveSpy {
        inner: &native,
        calls: Mutex::new(Vec::new()),
        cancel_after_first: None,
    };

    let out = SequentialPlanner
        .execute(&spy, &tasks, &params(), &CancelToken::default())
        .unwrap();

    assert_eq!(out.len(), 4);
    assert_eq!(
        *spy.calls.lock().unwrap(),
        vec![1, 1, 1, 1],
        "cap 1 must split the batch into four single-circuit waves"
    );
}

/// Local: same split, cap taken from the real `LocalBackend::capabilities_for`.
#[test]
fn execute_splits_high_qubit_local_batch_into_memory_capped_waves() {
    pyo3::prepare_freethreaded_python();
    let local = LocalBackend::new("AerSimulator".to_string(), "statevector".to_string(), None);
    let wide = wide_native();
    let tasks = wide_batch(&wide);
    let spy = WaveSpy {
        inner: &local,
        calls: Mutex::new(Vec::new()),
        cancel_after_first: None,
    };

    let out = SequentialPlanner
        .execute(&spy, &tasks, &params(), &CancelToken::default())
        .unwrap();

    assert_eq!(out.len(), 4);
    assert_eq!(*spy.calls.lock().unwrap(), vec![1, 1, 1, 1]);
}

/// Native: the wave boundary is honoured *between* waves — cancelling from inside
/// the first wave stops the second from launching.
#[test]
fn execute_checks_between_native_waves_not_only_at_the_end() {
    let native = NativeStatevectorBackend::new(0);
    let wide = wide_native();
    let tasks = wide_batch(&wide);
    let token = CancelToken::default();
    let spy = WaveSpy {
        inner: &native,
        calls: Mutex::new(Vec::new()),
        cancel_after_first: Some(token.clone()),
    };

    let err = SequentialPlanner
        .execute(&spy, &tasks, &params(), &token)
        .unwrap_err();

    assert!(matches!(err, InfrastructureError::Cancelled));
    assert_eq!(
        *spy.calls.lock().unwrap(),
        vec![1],
        "the second wave must not launch after cancellation at the first boundary"
    );
}

/// Local: same between-wave boundary, cap from the real `LocalBackend`.
#[test]
fn execute_checks_between_local_waves_not_only_at_the_end() {
    pyo3::prepare_freethreaded_python();
    let local = LocalBackend::new("AerSimulator".to_string(), "statevector".to_string(), None);
    let wide = wide_native();
    let tasks = wide_batch(&wide);
    let token = CancelToken::default();
    let spy = WaveSpy {
        inner: &local,
        calls: Mutex::new(Vec::new()),
        cancel_after_first: Some(token.clone()),
    };

    let err = SequentialPlanner
        .execute(&spy, &tasks, &params(), &token)
        .unwrap_err();

    assert!(matches!(err, InfrastructureError::Cancelled));
    assert_eq!(*spy.calls.lock().unwrap(), vec![1]);
}
