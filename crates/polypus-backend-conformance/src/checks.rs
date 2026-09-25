//! The individual conformance checks. Each returns a [`Check`]; none panics — a
//! contract violation is a [`Status::Failed`], an inapplicable check a
//! [`Status::Skipped`]. The orchestration that decides which to run lives in
//! [`crate`].

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use polypus_backend::{
    validate_run_results, BackendError, BoundCircuit, CancelToken, CircuitTask, Counts, OptLevel,
    Planner, QuantumBackend, RunParams, SequentialPlanner,
};

use crate::circuits::{alien_foreign, as_qasm2, bell, ones, zeros};
use crate::report::Check;
use crate::Representation;

/// Per-call parameters the battery uses. A fixed seed keeps a seed-consuming
/// backend reproducible; the `id` uses the contract's safe charset so it can reach
/// a SLURM job name unchanged.
pub(crate) fn params(shots: u32) -> RunParams {
    RunParams {
        id: "conformance-run".to_string(),
        shots,
        seed: Some(0xC0FFEE),
        opt_level: OptLevel::default(),
    }
}

/// Realise a circuit in the requested representation.
fn as_repr(circuit: &BoundCircuit, repr: Representation) -> BoundCircuit {
    match repr {
        Representation::Native => circuit.duplicate(),
        Representation::Qasm2 => as_qasm2(circuit),
    }
}

/// The width (number of `0`/`1` characters) shared by every key in a counts map, or
/// an error string if the keys disagree or the map is empty.
fn uniform_key_width(map: &Counts) -> Result<usize, String> {
    let mut widths = map.keys().map(String::len);
    let first = widths
        .next()
        .ok_or_else(|| "counts map is empty".to_string())?;
    if widths.all(|w| w == first) {
        Ok(first)
    } else {
        Err(format!(
            "counts map mixes bitstring widths: {:?}",
            map.keys().collect::<Vec<_>>()
        ))
    }
}

// ----------------------------------------------------------------------------
// Behavioural checks — need only a healthy backend.
// ----------------------------------------------------------------------------

/// A healthy batch's results satisfy the shipped [`validate_run_results`] over each
/// requested representation: one non-empty map per circuit, counts summing to the
/// requested shots (contract C-3), bitstring keys. This is the same validator the
/// planner applies, run against the backend's real output.
pub(crate) fn result_contract(
    backend: &dyn QuantumBackend,
    shots: u32,
    reprs: &[Representation],
) -> Check {
    const NAME: &str = "healthy_run_satisfies_result_contract";
    const DESC: &str = "a healthy batch's counts pass validate_run_results (count, non-empty, shot conservation C-3, bitstring keys)";
    let template = [zeros(1), ones(2), bell()];
    for &repr in reprs {
        let batch: Vec<BoundCircuit> = template.iter().map(|c| as_repr(c, repr)).collect();
        let out = match backend.run_circuits(&batch, &params(shots)) {
            Ok(out) => out,
            Err(e) => {
                return Check::failed(
                    NAME,
                    DESC,
                    format!("run_circuits({repr:?}) errored on a healthy batch: {e}"),
                )
            }
        };
        if let Err(e) = validate_run_results(&out, batch.len(), shots) {
            return Check::failed(
                NAME,
                DESC,
                format!("results for a healthy {repr:?} batch violate the contract: {e}"),
            );
        }
    }
    Check::passed(NAME, DESC)
}

/// The i-th returned map corresponds to the i-th input circuit, checked
/// structurally by bitstring width: a batch of a 1-qubit and a 3-qubit circuit must
/// come back as a width-1 map then a width-3 map, in that order. This detects a
/// backend that reorders or transposes its results without assuming it *simulates*
/// (only that it encodes the measured width in the bitstring, which the counts
/// format requires anyway).
pub(crate) fn order_preserved(backend: &dyn QuantumBackend, shots: u32) -> Check {
    const NAME: &str = "batch_order_is_preserved";
    const DESC: &str =
        "results come back in submission order (checked by per-circuit bitstring width)";
    let batch = [zeros(1), zeros(3)];
    let out = match backend.run_circuits(&batch, &params(shots)) {
        Ok(out) => out,
        Err(e) => return Check::failed(NAME, DESC, format!("run_circuits errored: {e}")),
    };
    if out.len() != 2 {
        return Check::failed(NAME, DESC, format!("expected 2 maps, got {}", out.len()));
    }
    for (i, expected) in [1usize, 3].into_iter().enumerate() {
        match uniform_key_width(&out[i]) {
            Ok(w) if w == expected => {}
            Ok(w) => {
                return Check::failed(
                    NAME,
                    DESC,
                    format!("circuit {i} is {expected} qubits but its counts have width {w} — results are out of order or mis-shaped"),
                )
            }
            Err(e) => return Check::failed(NAME, DESC, format!("circuit {i}: {e}")),
        }
    }
    Check::passed(NAME, DESC)
}

/// An empty batch returns an empty result vector (zero circuits → zero maps), not an
/// error and not a spurious map.
pub(crate) fn empty_batch(backend: &dyn QuantumBackend, shots: u32) -> Check {
    const NAME: &str = "empty_batch_returns_empty";
    const DESC: &str = "run_circuits(&[]) is Ok with zero result maps";
    match backend.run_circuits(&[], &params(shots)) {
        Ok(out) if out.is_empty() => Check::passed(NAME, DESC),
        Ok(out) => Check::failed(
            NAME,
            DESC,
            format!("an empty batch produced {} maps", out.len()),
        ),
        Err(e) => Check::failed(NAME, DESC, format!("an empty batch errored: {e}")),
    }
}

/// A batch larger than the backend's concurrency runs correctly through the shipped
/// [`SequentialPlanner`] — the planner splits it into waves (one `run_circuits` per
/// wave, capped at `max_concurrency`; e.g. six waves for a QMIO cap of 1) and
/// reassembles them in order. Every returned map is validated.
pub(crate) fn multi_wave_batch(backend: &dyn QuantumBackend, shots: u32) -> Check {
    const NAME: &str = "sequential_planner_runs_a_multi_wave_batch";
    const DESC: &str =
        "a batch larger than max_concurrency runs correctly in waves via SequentialPlanner";
    let circuits = [zeros(1), ones(1), zeros(2), ones(2), bell(), zeros(3)];
    let tasks: Vec<CircuitTask> = circuits
        .iter()
        .map(|c| CircuitTask { circuit: c, shots })
        .collect();
    match SequentialPlanner.execute(backend, &tasks, &params(shots), &CancelToken::default()) {
        Ok(out) => match validate_run_results(&out, circuits.len(), shots) {
            Ok(()) => Check::passed(NAME, DESC),
            Err(e) => Check::failed(NAME, DESC, format!("planned results are invalid: {e}")),
        },
        Err(e) => Check::failed(NAME, DESC, format!("the planner errored: {e}")),
    }
}

/// Shot distribution conserves the total exactly (contract C-3): splitting one
/// circuit's shots across replicas via [`QuantumBackend::run_shots_distributed`]
/// returns one map per replica, each summing to its own batch, and the merged total
/// equal to the requested shots. Skipped when the backend declares it does not
/// support shot distribution.
pub(crate) fn shot_distribution(backend: &dyn QuantumBackend, shots: u32) -> Check {
    const NAME: &str = "shot_distribution_conserves_shots";
    const DESC: &str = "run_shots_distributed splits a circuit's shots across replicas and conserves the total (C-3)";
    if !backend.capabilities().supports_shot_distribution {
        return Check::skipped(
            NAME,
            DESC,
            "backend's capabilities() report supports_shot_distribution = false",
        );
    }
    let qc = bell();
    // An uneven split summing to `shots`.
    let base = shots / 3;
    let batches = [base + (shots % 3), base, base];
    let out = match backend.run_shots_distributed(&qc, &batches, &params(shots)) {
        Ok(out) => out,
        Err(e) => return Check::failed(NAME, DESC, format!("run_shots_distributed errored: {e}")),
    };
    if out.len() != batches.len() {
        return Check::failed(
            NAME,
            DESC,
            format!("expected {} replica maps, got {}", batches.len(), out.len()),
        );
    }
    for (i, (map, &expected)) in out.iter().zip(batches.iter()).enumerate() {
        let got: u64 = map.values().sum();
        if got != u64::from(expected) {
            return Check::failed(
                NAME,
                DESC,
                format!("replica {i} was given {expected} shots but its counts sum to {got}"),
            );
        }
    }
    let total: u64 = out.iter().flat_map(|m| m.values()).sum();
    if total != u64::from(shots) {
        return Check::failed(
            NAME,
            DESC,
            format!("merged shots {total} != requested {shots} (C-3 violated)"),
        );
    }
    Check::passed(NAME, DESC)
}

/// A zero-shot replica contributes nothing: `run_shots_distributed` with a `0` entry
/// (which happens whenever `shots < n_qpus`) must submit nothing for it — its map
/// sums to 0 — while the non-zero replicas still conserve the total. Skipped when the
/// backend declares no shot-distribution support.
pub(crate) fn zero_shot_replica(backend: &dyn QuantumBackend, shots: u32) -> Check {
    const NAME: &str = "zero_shot_replica_contributes_nothing";
    const DESC: &str =
        "a zero-shot replica in run_shots_distributed yields a map summing to 0, total still conserved";
    if !backend.capabilities().supports_shot_distribution {
        return Check::skipped(
            NAME,
            DESC,
            "backend's capabilities() report supports_shot_distribution = false",
        );
    }
    let qc = bell();
    let batches = [0u32, shots];
    let out = match backend.run_shots_distributed(&qc, &batches, &params(shots)) {
        Ok(out) => out,
        Err(e) => return Check::failed(NAME, DESC, format!("run_shots_distributed errored: {e}")),
    };
    if out.len() != 2 {
        return Check::failed(NAME, DESC, format!("expected 2 maps, got {}", out.len()));
    }
    let zero_sum: u64 = out[0].values().sum();
    if zero_sum != 0 {
        return Check::failed(
            NAME,
            DESC,
            format!("the zero-shot replica's counts sum to {zero_sum}, not 0"),
        );
    }
    let total: u64 = out.iter().flat_map(|m| m.values()).sum();
    if total != u64::from(shots) {
        return Check::failed(
            NAME,
            DESC,
            format!("total {total} != requested {shots} with a zero-shot replica"),
        );
    }
    Check::passed(NAME, DESC)
}

/// A backend that declares a *finite* `max_concurrency` accepts a full wave of
/// exactly that many circuits in a single `run_circuits` call and returns valid
/// results. Skipped for an unbounded backend (`usize::MAX`), where there is no
/// specific wave size to exercise.
pub(crate) fn declared_concurrency(backend: &dyn QuantumBackend, shots: u32) -> Check {
    const NAME: &str = "declared_finite_concurrency_wave_runs";
    const DESC: &str =
        "a backend declaring a finite max_concurrency runs a full wave of that many circuits";
    let cap = backend.capabilities().max_concurrency;
    if cap == usize::MAX {
        return Check::skipped(
            NAME,
            DESC,
            "backend declares unbounded concurrency (max_concurrency = usize::MAX)",
        );
    }
    // Guard against an absurd cap that would allocate a huge batch.
    let wave = cap.clamp(1, 64);
    let batch: Vec<BoundCircuit> = (0..wave).map(|_| bell()).collect();
    match backend.run_circuits(&batch, &params(shots)) {
        Ok(out) => match validate_run_results(&out, wave, shots) {
            Ok(()) => Check::passed(NAME, DESC),
            Err(e) => Check::failed(NAME, DESC, format!("full-wave results are invalid: {e}")),
        },
        Err(e) => Check::failed(
            NAME,
            DESC,
            format!("a full wave of {wave} circuits errored: {e}"),
        ),
    }
}

// ----------------------------------------------------------------------------
// Error-classification checks.
// ----------------------------------------------------------------------------

/// An unrecognised `Foreign` circuit is rejected with
/// [`BackendError::UnsupportedCircuit`], not mishandled. Uses a provider type no
/// backend owns, so even Aer/CUNQA (which accept a Qiskit `Foreign`) reject it.
pub(crate) fn rejects_foreign(backend: &dyn QuantumBackend, shots: u32) -> Check {
    const NAME: &str = "foreign_circuit_rejected_as_unsupported";
    const DESC: &str = "an unrecognised Foreign circuit yields BackendError::UnsupportedCircuit";
    match backend.run_circuits(&[alien_foreign()], &params(shots)) {
        Err(BackendError::UnsupportedCircuit(_)) => Check::passed(NAME, DESC),
        Err(other) => Check::failed(
            NAME,
            DESC,
            format!("expected UnsupportedCircuit, got {other:?}"),
        ),
        Ok(_) => Check::failed(
            NAME,
            DESC,
            "the backend accepted an alien Foreign circuit instead of rejecting it",
        ),
    }
}

/// A backend rigged to have stopped responding (a dead or hung worker) classifies
/// the failure as [`BackendError::Unresponsive`] — the retryable "stopped
/// responding" variant, distinct from a definitive provider error.
pub(crate) fn unresponsive(faulty: &dyn QuantumBackend, shots: u32) -> Check {
    const NAME: &str = "unresponsive_backend_is_classified";
    const DESC: &str =
        "a dead/hung backend yields BackendError::Unresponsive (retryable), not a generic error";
    match faulty.run_circuits(&[zeros(2)], &params(shots)) {
        Err(BackendError::Unresponsive(_)) => Check::passed(NAME, DESC),
        Err(other) => Check::failed(NAME, DESC, format!("expected Unresponsive, got {other:?}")),
        Ok(_) => Check::failed(
            NAME,
            DESC,
            "the rigged-unresponsive backend returned counts instead of erroring",
        ),
    }
}

/// A run aborted by an out-of-band [`QuantumBackend::cancel`] (the terminal-Ctrl+C
/// path a subprocess backend sees) classifies as [`BackendError::Aborted`] — the
/// cancellation variant the FFI edge raises as `KeyboardInterrupt`. The backend is
/// run on a worker thread and cancelled from here once the call is in flight.
pub(crate) fn aborted(faulty: Arc<dyn QuantumBackend>, shots: u32, grace: Duration) -> Check {
    const NAME: &str = "interrupted_call_is_aborted";
    const DESC: &str =
        "an in-flight call cancelled out-of-band yields BackendError::Aborted (→ KeyboardInterrupt)";
    // Supplying a Fault::Aborted factory asserts that out-of-band cancellation works
    // end-to-end — and in production the *only* caller of `cancel()` is the watcher
    // thread in `Scheduler::run_cancellable`, which is spawned solely for backends that
    // opt in via `wants_cancel_watcher() == true` (the default is false). A backend
    // that implements `cancel()` but forgets to override that method would abort here
    // (we call `cancel()` directly) yet never be cancellable in real use, so the check
    // must reject it before it can pass on the direct call alone.
    if !faulty.wants_cancel_watcher() {
        return Check::failed(
            NAME,
            DESC,
            "the backend supports Fault::Aborted but wants_cancel_watcher() is false, so \
             cancel() is never invoked from Scheduler::run_cancellable in production — override \
             wants_cancel_watcher() to return true",
        );
    }
    let runner = Arc::clone(&faulty);
    let handle = thread::spawn(move || runner.run_circuits(&[zeros(2)], &params(shots)));
    // Let the call get in flight, then abort it from this thread.
    thread::sleep(grace);
    faulty.cancel();
    let result = match handle.join() {
        Ok(r) => r,
        Err(_) => return Check::failed(NAME, DESC, "the run thread panicked"),
    };
    match result {
        Err(BackendError::Aborted(_)) => Check::passed(NAME, DESC),
        Err(other) => Check::failed(NAME, DESC, format!("expected Aborted, got {other:?}")),
        Ok(_) => Check::failed(
            NAME,
            DESC,
            "the cancelled call returned counts instead of Aborted",
        ),
    }
}

/// A backend rigged to hit a *clean, definitive* provider failure classifies it as
/// neither [`Unresponsive`](BackendError::Unresponsive) (retryable) nor
/// [`Aborted`](BackendError::Aborted) (a cancellation): it is a real error the caller
/// must surface. The observed variant is reported so the classification is visible
/// (a provider error is expected to be [`External`](BackendError::External)).
pub(crate) fn clean_error(faulty: &dyn QuantumBackend, shots: u32) -> Check {
    const NAME: &str = "clean_provider_error_is_classified";
    const DESC: &str =
        "a clean provider failure is a definitive error, not Unresponsive and not Aborted";
    match faulty.run_circuits(&[zeros(2)], &params(shots)) {
        Err(BackendError::Unresponsive(_)) => Check::failed(
            NAME,
            DESC,
            "a clean provider error was misclassified as Unresponsive (implies a spurious retry)",
        ),
        Err(BackendError::Aborted(_)) => Check::failed(
            NAME,
            DESC,
            "a clean provider error was misclassified as Aborted (implies a spurious cancellation)",
        ),
        // Any other variant is a definitive failure — the correct classification for
        // a clean provider error (a provider error is expected to be `External`; the
        // native backend's own clean error is `NativeCircuit`). Accepted as a pass.
        Err(_) => Check::passed(NAME, DESC),
        Ok(_) => Check::failed(
            NAME,
            DESC,
            "the rigged-error backend returned counts instead of erroring",
        ),
    }
}
