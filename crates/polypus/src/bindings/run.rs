//! `polypus.run_quantum_circuit`: one circuit, run once, its counts returned.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use pyo3::prelude::*;

use super::extract::extract_bound_circuit;
use super::interrupt::interruptible_token;
use super::options::OptionsArg;
use super::prepare::{self, Entry, TargetArgs};
use super::results::{counts_to_pydict, counts_to_pylist, RunResult};
use super::seeds::{seed_or_entropy, unique_id};
use super::{call_trace, python_version};
use crate::infrastructure::execution_config::random_seed;
use crate::infrastructure::{
    merge_counts, Counts, InfrastructureError, Planner, ShotDistributingPlanner,
};
use crate::orchestration::trace::{TraceFlow, TraceHeader};
use crate::orchestration::RunCircuitFlow;

/// Function to run a quantum circuit called from Python.
///
/// `qc` may be a Qiskit `QuantumCircuit`, a `polypus.Circuit` (fully bound),
/// or an OpenQASM 2.0 string. The meaning of `backend` depends on
/// `infrastructure`: for `"local"` it selects the device — `"aer"` (default)
/// or the pure-Rust `"polypus"` statevector simulator; for `"qmio"` it
/// instead selects the program format submitted to the QPU (`"openqasm"`
/// (default), `"qir"`, or `"qir_bitcode"`); for `"cunqa"`, whose QPUs run Aer,
/// it must be `"aer"` (default) or `"AerSimulator"`. Any other value, or a
/// `noise_model` with `"cunqa"`, raises `ValueError` before any QPU is requested.
///
/// `seed` controls shot sampling (contract C-7) on every simulated backend —
/// native, Aer (`infrastructure="local"`) and CUNQA's simulated QPUs
/// (`infrastructure="cunqa"`): an explicit `seed` reproduces the counts
/// byte-for-byte, and omitting it draws a fresh OS-entropy seed so repeated
/// runs give genuinely independent noise. Passing a `seed` with
/// `infrastructure="qmio"` raises `ValueError` rather than silently ignoring
/// it, since that infrastructure is real hardware and cannot be seeded.
///
/// `nodes` and `cores_per_qpu` size the SLURM allocation and are meaningful
/// only for `infrastructure="cunqa"` (they become CUNQA's `qraise` parameters);
/// the `local` and `qmio` infrastructures ignore them. For `cunqa` both must be
/// `>= 1` — a zero has no sane meaning once it reaches SLURM and is rejected.
/// They default to `nodes=1, cores_per_qpu=2`.
///
/// `walltime` is the SLURM walltime of the CUNQA QPUs (sent to `qraise` as `t`),
/// again meaningful only for `infrastructure="cunqa"`. It must be `"HH:MM:SS"`:
/// hours of one or more digits (`"48:00:00"` is valid), minutes and seconds of
/// two digits below 60, total above zero; SLURM's `D-HH:MM:SS` is rejected. A
/// malformed value raises `ValueError` before any QPU is requested. Omitted, it
/// is `"10:00:00"`.
///
/// Returns a [`RunResult`] carrying the counts — `counts`, always a
/// `list[dict]` with one dict per QPU replica, and `merged_counts`, their total as
/// one `dict` — plus a manifest (`id`, effective `seed`, `backend`,
/// `infrastructure`) for logging and replay.
///
/// `fusion` applies only to `backend="polypus"`, the one backend that fuses
/// gates. Omit it (the default) and each backend does its own thing — the
/// native one fuses, the rest never did. `fusion=False` forces a strictly
/// gate-by-gate run and is accepted everywhere (a non-fusing backend already
/// meets it). `fusion=True` on any backend that cannot fuse is rejected with a
/// `ValueError` rather than silently ignored, so it never looks like it took
/// effect.
///
/// `options` configures a registry-backed backend (`"qmio"`, `"subprocess"`, or a
/// registered third-party name): a `dict` whose values are each a `str` or a
/// `list`/`tuple` of `str` (e.g. `{"command": ["python3", "/my dir/worker.py"]}`).
/// Any other value type is a `TypeError` naming the key; a non-empty `options` on
/// `local`/`cunqa`, which would ignore it, is a `ValueError`.
///
/// With `infrastructure="qmio"` the QPU endpoint (`ZMQ_SERVER`, or the
/// `endpoint` option) must be trusted: the ZeroMQ link is neither encrypted nor
/// authenticated, and the 64 MiB reply cap is checked only after a reply has
/// been received, so a hostile endpoint can make the process allocate as much
/// memory as it announces. Off CESGA's private network, use a tunnel (see "QMIO
/// transport threat model" in `docs/backends.md`).
///
/// `trace` (keyword-only, default `None`) writes the run's call trace
/// (contract C-12) to a new JSON Lines file: a header line, one `batch` line
/// (`execute_replicas`) and one `wave` line per backend call, with their sizes
/// and times (each replica's shots too when `n_qpus > 1`) and nothing else (no
/// circuits or counts), and last an `end` line with how the run ended
/// (`"ok"`, `"error"` or `"cancelled"`). It is a path, a `str` or an
/// `os.PathLike[str]`, to a file that does not exist yet, in a directory that
/// does. If something is already there, a file or a symbolic link, or the
/// directory does not exist, nothing is created or changed and the run goes on
/// without a trace, after one warning in the Polypus log (shown only once a
/// logger is installed with `polypus.init_logger`). A `bytes` path, or a value
/// that is not a path, raises `TypeError` before anything runs. The trace never
/// changes the result or the error. Without `trace`,
/// `POLYPUS_CALL_TRACE=<directory>` writes one trace per run there, named
/// `<id>-<pid>-<n>.jsonl`, as for `train`, after the run's id (`RunResult.id`,
/// cut to its first 100 characters and a hash if longer), the process id and
/// the number of traces the variable named before in the process; the
/// directory must exist. `POLYPUS_CALL_TRACE_MAX_MB` caps a trace, in whole
/// MiB (512 MiB by default).
///
/// `target` (keyword-only, default `None`) is a typed target, `polypus.Aer`,
/// `Native`, `Cunqa`, `Qmio` or `Registered`, used instead of `infrastructure`,
/// `backend` and the provider arguments. Next to it, `infrastructure`,
/// `noise_model`, `fusion`, `nodes`, `cores_per_qpu` or `walltime` given at
/// all, or `backend` or `sim_method` with a non-default value, raises
/// `ConfigurationError`. Without `target`, `infrastructure` is required.
// `nodes`/`cores_per_qpu` are `None` at runtime so an explicit value next to
// `target=` is seen; `text_signature` keeps showing their defaults.
#[pyfunction(signature=(qc, shots, infrastructure=None, n_qpus=1, nodes=None, cores_per_qpu=None, sim_method="automatic", noise_model=None, backend="aer", seed=None, fusion=None, options=None, walltime=None, *, trace=None, target=None), text_signature = "(qc, shots, infrastructure=None, n_qpus=1, nodes=1, cores_per_qpu=2, sim_method=\"automatic\", noise_model=None, backend=\"aer\", seed=None, fusion=None, options=None, walltime=None, *, trace=None, target=None)")]
pub fn run_quantum_circuit<'py>(
    qc: Bound<'py, PyAny>,
    shots: u32,
    infrastructure: Option<String>,
    n_qpus: u32,
    nodes: Option<u32>,
    cores_per_qpu: Option<u32>,
    sim_method: &str,
    noise_model: Option<Bound<'py, PyAny>>,
    backend: &str,
    seed: Option<u64>,
    fusion: Option<bool>,
    options: Option<OptionsArg>,
    walltime: Option<String>,
    trace: Option<PathBuf>,
    target: Option<Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let start = Instant::now();
    let selection = prepare::select(
        Entry::Run,
        TargetArgs {
            target: target.as_ref(),
            infrastructure,
            backend,
            sim_method,
            noise_model,
            fusion,
            nodes,
            cores_per_qpu,
            walltime,
        },
    )?;
    let (infrastructure, backend) = (selection.infrastructure.clone(), selection.backend.clone());
    // Entry-point trace carrying the full circuit `Debug` repr on every call:
    // large and high-volume, so it stays at `debug` rather than the default log.
    log::debug!(
        "run_quantum_circuit called with qc: {qc:?}, shots: {shots}, \
         infrastructure: {infrastructure}, n_qpus: {n_qpus}, backend: {backend}, seed: {seed:?}"
    );
    let id_prefix = prepare::check_arguments(Entry::Run, &selection, shots, n_qpus, None)?;
    let bound_qc = extract_bound_circuit(&qc)?;
    // The native backend and the QMIO path read circuits in Rust (GIL-free) and
    // cannot read a Qiskit QuantumCircuit, whose gates are only accessible via
    // Python.
    prepare::reject_foreign_circuit(Entry::Run, &selection, bound_qc.is_foreign())?;
    // Resolve the shot-sampling seed. Every simulated backend (native, Aer,
    // CUNQA's simulated QPUs) is seeded by Polypus; `qmio` is real hardware, so
    // an explicit seed there would be silently ineffective — reject it rather
    // than give false confidence in reproducibility. When simulated, `None`
    // means "draw a fresh OS-entropy seed", resolved here so the effective
    // value can be reported in the manifest.
    let effective_seed: Option<u64> = if infrastructure == "qmio" {
        if seed.is_some() {
            return Err(crate::exceptions::configuration_error(
                "seed is not supported for the 'qmio' infrastructure (real quantum hardware)",
            ));
        }
        None
    } else {
        Some(seed_or_entropy(seed, random_seed)?)
    };

    // Append a UUID v4 so two concurrent calls with identical `n_qpus`/
    // `infrastructure` never collide: the id names SLURM families/allocations,
    // temp files and log streams (see ExecutionConfig::id), so a byte-identical
    // id across runs is a real hazard, not a cosmetic one. The `run_{n}_{infra}`
    // prefix is kept for human-readable debuggability.
    let id = unique_id(&id_prefix)?;
    // Only the native statevector backend consults the gate-parallel threshold,
    // so it is calibrated lazily (#351), with its one-time default-visible
    // warning, only when this run actually resolves to it — never for
    // Aer/CUNQA/QMIO. Gated on the resolved `BackendConfig`, not the raw
    // `backend` string, so it tracks the real dispatch rather than re-deriving
    // it from the kwargs.
    let backend_config = prepare::backend_config(qc.py(), selection, options)?;
    let config = prepare::execution_config(
        id.clone(),
        shots,
        n_qpus,
        infrastructure.clone(),
        backend_config,
        effective_seed,
    );
    // Lifecycle record at the default level, carrying the C-7 manifest data only
    // (the circuit itself stays in the `debug!` above): enough to see what ran
    // where, and with which seed to replay it.
    log::info!(
        "run {id} starting: infrastructure={infrastructure}, backend={backend}, \
         n_qpus={n_qpus}, shots={shots}, seed={effective_seed:?}"
    );
    // The call trace (contract C-12), if `trace` names a path or
    // `POLYPUS_CALL_TRACE` a directory (where the file is named after `id`,
    // the effective id), as in `train`. It starts before the backend
    // exists, so its clock counts the backend's start-up too (on CUNQA,
    // raising the QPUs), and every way out from here closes it with the
    // status Python sees. Creating it writes the header line to the file, so
    // the GIL is released; a path that cannot be created leaves the run
    // untraced.
    let py = qc.py();
    let header = TraceHeader {
        flow: TraceFlow::RunCircuit,
        id: id.clone(),
        polypus: python_version(env!("CARGO_PKG_VERSION")),
        method: None,
        dims: None,
        backend: backend.to_owned(),
        infrastructure: infrastructure.clone(),
        n_qpus,
        shots,
        seed: effective_seed,
    };
    let tracer = py.detach(|| call_trace::open(trace, header));
    // The flow's handle on the trace, moved into the run below.
    let flow_trace = tracer.clone();

    // Release the GIL for the whole run, mirroring `train`: circuit execution is
    // GIL-free on the native backend (and internally reacquires the GIL where
    // Aer/CUNQA need it), so holding it here would stall every other Python thread
    // and (with the Planner's between-wave check_signals) keep Ctrl+C from taking
    // effect until the run finishes. See docs/ENGINEERING.md §3. `n_qpus > 1`
    // selects the shot-distributing planner (which apportions this one circuit's
    // shots across replicas, conserving the total per C-3, and returns one counts
    // map per replica); otherwise the backend's default atomic-wave planner runs
    // the circuit as-is, as its single replica.
    let counts_result = qc
        .py()
        .detach(move || -> Result<Vec<Counts>, InfrastructureError> {
            let planner: Option<Arc<dyn Planner>> = if n_qpus == 1 {
                None
            } else {
                Some(Arc::new(ShotDistributingPlanner::new(n_qpus)))
            };
            let scheduler = prepare::scheduler(&config, planner)?;
            // A Ctrl+C during the run aborts it at the next wave boundary: the
            // token carries the `check_signals`-backed interrupt guard the
            // pyo3-free planner calls between waves.
            let out = scheduler.run_cancellable(
                RunCircuitFlow {
                    circuits: vec![bound_qc],
                    shots,
                    trace: flow_trace,
                },
                &interruptible_token(),
            );
            scheduler.close();
            out
        });
    // A failed run, creating its backend included, closes the trace with the
    // status of the error Python gets.
    let counts_vec = counts_result
        .map_err(crate::exceptions::infrastructure_error_to_pyerr)
        .map_err(|err| call_trace::close_on_error(py, tracer.as_ref(), err))?;
    // Completion counterpart of the start record above. There is no
    // iterations/convergence notion on this path (those are `TrainResult`
    // fields), so this reports only the run and how long it took.
    log::info!("run {id} completed: duration={:?}", start.elapsed());
    // Convert at the FFI boundary with one shape for every `n_qpus` (C-3):
    // `counts` is a `list[dict]` with one map per replica, and `merged_counts` is
    // their key-by-key sum as a single `dict`. Keys are sorted in both.
    let merged = merge_counts(&counts_vec);
    let out = Python::attach(|py| {
        let counts = counts_to_pylist(py, &counts_vec)?.into_any().unbind();
        let merged_counts = counts_to_pydict(py, &merged)?.into_any().unbind();
        Py::new(
            py,
            RunResult {
                counts,
                merged_counts,
                id,
                seed: effective_seed,
                backend: backend.to_string(),
                infrastructure,
            },
        )
        .map(|result| result.into_any())
    });
    // The outcome returned to Python, the result or a failure to build it,
    // closes the trace.
    call_trace::close(py, tracer.as_ref(), &out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::test_support::uniform3_qasm;
    use pyo3::types::PyString;
    use std::collections::HashMap;

    /// Drive `run_quantum_circuit` on the native backend and return the effective
    /// seed reported in the manifest plus the wrapped counts payload.
    fn native_run(
        py: Python<'_>,
        qasm: &str,
        seed: Option<u64>,
    ) -> (Option<u64>, Vec<HashMap<String, u64>>) {
        let qc = PyString::new(py, qasm).into_any();
        let result = run_quantum_circuit(
            qc,
            500,
            Some("local".to_string()),
            1,
            Some(1),
            Some(2),
            "automatic",
            None,
            "polypus",
            seed,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("native run_quantum_circuit succeeds");
        let bound = result.bind(py);
        let reported = bound
            .getattr("seed")
            .unwrap()
            .extract::<Option<u64>>()
            .unwrap();
        let counts = bound
            .getattr("counts")
            .unwrap()
            .extract::<Vec<HashMap<String, u64>>>()
            .unwrap();
        (reported, counts)
    }

    /// Acceptance criterion (defect #1) through the public entry point: an
    /// explicit seed round-trips into the manifest and reproduces the counts.
    #[test]
    fn native_seed_round_trips_and_reproduces_counts() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let qasm = uniform3_qasm();
            let (s1, c1) = native_run(py, &qasm, Some(42));
            let (s2, c2) = native_run(py, &qasm, Some(42));
            assert_eq!(s1, Some(42));
            assert_eq!(s2, Some(42));
            assert_eq!(c1, c2, "same seed must reproduce counts");
        });
    }

    /// Acceptance criterion (defect #1), negative half: with no seed, each run
    /// draws a fresh entropy seed (reported in the manifest) and the counts
    /// differ across calls.
    #[test]
    fn native_omitted_seed_is_entropy_and_differs() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let qasm = uniform3_qasm();
            let (s1, c1) = native_run(py, &qasm, None);
            let (s2, c2) = native_run(py, &qasm, None);
            assert!(
                s1.is_some() && s2.is_some(),
                "an entropy seed must be reported"
            );
            assert_ne!(s1, s2, "each unseeded run must draw a fresh seed");
            assert_ne!(c1, c2, "unseeded runs must produce independent noise");
        });
    }

    /// The manifest carries the full run metadata for logging/replay.
    #[test]
    fn native_manifest_reports_run_metadata() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let qc = PyString::new(py, &uniform3_qasm()).into_any();
            let result = run_quantum_circuit(
                qc,
                100,
                Some("local".to_string()),
                1,
                Some(1),
                Some(2),
                "automatic",
                None,
                "polypus",
                Some(7),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("native run succeeds");
            let bound = result.bind(py);
            let get = |name: &str| bound.getattr(name).unwrap().extract::<String>().unwrap();
            assert_eq!(get("backend"), "polypus");
            assert_eq!(get("infrastructure"), "local");
            // The id keeps the human-readable `run_{n}_{infra}_` prefix but is
            // suffixed with a per-call UUID (see the uniqueness test below), so
            // only the stable prefix is asserted here.
            assert!(
                get("id").starts_with("run_1_local_"),
                "id must keep the run_{{n}}_{{infra}}_ prefix, got {}",
                get("id")
            );
            assert_eq!(
                bound
                    .getattr("seed")
                    .unwrap()
                    .extract::<Option<u64>>()
                    .unwrap(),
                Some(7)
            );
        });
    }

    /// Regression for #45: two runs with identical arguments must not share an
    /// id. The id names SLURM families/allocations, temp files and log streams,
    /// so a collision (as with the old `run_{n}_{infra}` format) could make two
    /// concurrent runs clobber one another. The per-call UUID v4 guarantees
    /// uniqueness while the `run_1_local_` prefix stays stable for debugging.
    #[test]
    fn auto_generated_id_is_unique_per_call() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let qasm = uniform3_qasm();
            let id_of = || {
                let qc = PyString::new(py, &qasm).into_any();
                let result = run_quantum_circuit(
                    qc,
                    100,
                    Some("local".to_string()),
                    1,
                    Some(1),
                    Some(2),
                    "automatic",
                    None,
                    "polypus",
                    Some(7),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("native run succeeds");
                result
                    .bind(py)
                    .getattr("id")
                    .unwrap()
                    .extract::<String>()
                    .unwrap()
            };
            let id1 = id_of();
            let id2 = id_of();
            assert!(id1.starts_with("run_1_local_") && id2.starts_with("run_1_local_"));
            assert_ne!(
                id1, id2,
                "ids from identical calls must differ (unique per run)"
            );
        });
    }

    /// A seed passed with `infrastructure="qmio"` is rejected, never silently
    /// dropped — that infrastructure is real hardware and Polypus cannot seed
    /// it, unlike the native, Aer, and CUNQA simulated backends.
    #[test]
    fn seed_rejected_for_qmio_hardware() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let qasm = uniform3_qasm();
            let qc = pyo3::types::PyString::new(py, &qasm).into_any();
            let result = run_quantum_circuit(
                qc,
                100,
                Some("qmio".to_string()),
                1,
                Some(1),
                Some(2),
                "automatic",
                None,
                "aer",
                Some(3),
                None,
                None,
                None,
                None,
                None,
            );
            assert!(
                result.is_err(),
                "an explicit seed must be rejected for infrastructure=qmio"
            );
        });
    }
}
