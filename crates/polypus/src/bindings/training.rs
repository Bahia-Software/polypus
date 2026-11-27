//! `polypus.train`: a variational circuit trained with a chosen optimizer.

use std::path::PathBuf;
use std::time::Instant;

use pyo3::prelude::*;

use super::callback::validate_callback;
use super::extract::{extract_circuit_source, extract_cost_observable};
use super::interrupt::interruptible_token;
use super::optimizer::{
    check_rotosolve_circuit, is_rotosolve, method_from_pyclass, training_observer,
    validate_final_shots, validate_method,
};
use super::options::OptionsArg;
use super::prepare::{self, Entry, TargetArgs};
use super::results::{finish_optimization, log_training_start};
use super::seeds::{method_seed, resolve_optimizer_seed, unique_id};
use super::{call_trace, python_log, python_version};
use crate::evaluation::{CircuitSource, OracleErrorSlot, VqcOracleFactory};
use crate::infrastructure::execution_config::random_seed;
use crate::orchestration::trace::{TraceFlow, TraceHeader};
use crate::orchestration::TrainFlow;

/// Unified entry point: train a variational quantum circuit with a chosen optimizer.
///
/// `method` must be an instance of `DE`, `PSO`, `QNG`, `CMAES`, `Adam`, `SPSA`,
/// `Rotosolve`, or `External`. Rotosolve trains only a circuit whose every
/// parameter is the angle of exactly one single-Pauli rotation, as `±θ + c`:
/// the circuit is checked before anything runs, and a circuit that does not
/// qualify raises `ValueError` naming the first gate or parameter that does
/// not. `External` runs an external ask/tell optimizer (Optuna, Nevergrad,
/// pycma or your own object), one batch per `ask()`; an exception its `ask()`
/// or `tell()` raises propagates as itself.
///
/// `seed` makes the optimizer reproducible (contract C-7): precedence is the
/// explicit `seed` kwarg, then the optimizer object's `seed` field, then a fresh
/// OS-entropy value. `External` has no `seed` field and its library owns its
/// seed, so there the seed only fixes the shot sampling. On the native backend
/// the same seed also drives shot sampling, so a native-backend run reproduces
/// exactly. On Aer and CUNQA the resolved seed must be below `2**63`, the
/// range of Aer's `seed_simulator`, or `polypus.ConfigurationError` is raised
/// before anything runs (a drawn seed always is; C-7). Returns a
/// [`TrainResult`] exposing `best_params`, `best_fitness`, `iterations_run`,
/// `converged`, `fitness_history`, `final_params`, `final_fitness`, the
/// effective `seed` and the effective `id`.
///
/// The `id` kwarg is a human-readable *prefix*, not the literal effective id:
/// a UUID v4 is appended for uniqueness (mirroring [`run_quantum_circuit`]),
/// so `TrainResult.id` differs from what was passed in and names the run's
/// SLURM allocation / temp files / log stream (see #75). Because it travels
/// into those downstream names, the prefix is restricted to `[A-Za-z0-9._-]`,
/// non-empty and at most 64 characters (contract C-9); anything else — a space,
/// a path separator, a shell metacharacter — raises `ValueError` naming the
/// offending character, before the UUID suffix is generated.
///
/// `initial_point` (default `None`) starts the search from a given point, such
/// as the `best_params` of an earlier run: a list, tuple or NumPy array of
/// `dimensions` finite numbers, otherwise `ValueError` before anything runs.
/// DE (every `strategy`) evaluates it, wrapped into `[0, 2π)`, as the first
/// member of its initial population; PSO as its first particle, which must lie
/// within the PSO `bounds`; QNG and Adam start from it as θ₀; CMA-ES centres
/// its first run on it (restarts start elsewhere), with its `sigma0`
/// unchanged; SPSA starts from it as θ₀ and centres its calibration batch on
/// it; Rotosolve starts from it as θ₀, the first point it measures. Without it
/// every optimizer starts as before (contract C-5). `External` refuses it with
/// `ValueError`, whatever the point: the external library chooses its start
/// (give it to the library, or to the `initial_point` of a `polypus.ask_tell`
/// adapter).
///
/// `final_shots` (default `None`) re-evaluates the best candidates once the
/// optimizer has stopped, in one more batch with that many shots per circuit:
/// `best_params` and, for DE, PSO, CMA-ES and `External`, the four best other
/// distinct candidates they scored. `TrainResult.final_params` is the one with the
/// highest re-evaluated value (`best_params` on a tie) and `final_fitness`
/// that value; both are `None` without it, and `best_params`, `best_fitness`
/// and `fitness_history` are the same with and without it, seeded runs
/// included (contract C-5). It must be an integer `>= 1`, checked before
/// anything runs: a smaller one raises `ValueError`, and a value that is not
/// an integer (a `bool` or a `float` too) `TypeError`, as `shots` does. The batch is sent
/// whatever stopped the optimizer, but not after an error or an interrupt; it
/// is not part of the CUNQA walltime projection, and if it fails, `train`
/// raises as for any batch.
///
/// `callback` (keyword-only, default `None`) is called once per iteration of
/// the optimizer, `iterations_run` times in all, with a read-only
/// `polypus.TrainProgress`: `iteration` (1, 2, …), `best_fitness` and
/// `best_params` (the entry the iteration leaves in `fitness_history`, and a
/// copy of its parameters) and `evaluations` (the candidates scored so far).
/// An iteration is a generation of DE, PSO or CMA-ES, an iteration of QNG,
/// Adam or SPSA, a step of Rotosolve and an ask/tell batch of `External`. It
/// is not called once the optimizer has returned, nor for the `final_shots`
/// batch. Returning `True` (or a NumPy `bool_` that is true) stops the run
/// after that iteration: `train` returns the `TrainResult` it has reached,
/// with `iterations_run` that iteration, `converged=False` and every other
/// field as after any stop, and the `final_shots` batch is still sent.
/// Returning `None` or `False` lets the run go on; any other value raises
/// `TypeError` and the run returns no result. An exception the callback
/// raises propagates as itself, with no result and no `final_shots` batch
/// (contract C-5). Without a callback, or with one that never stops the run,
/// a seeded run returns exactly the same result. A `callback` that is neither
/// callable nor `None` raises `TypeError` before anything runs.
///
/// `trace` (keyword-only, default `None`) writes the run's call trace
/// (contract C-12) to a new JSON Lines file: a header line, one line per
/// planner batch, backend wave, optimizer iteration and call to QNG's metric,
/// with their sizes and times and nothing else (no circuits, parameters or
/// counts), and last an `end` line with how the run ended (`"ok"`, `"error"`
/// or `"cancelled"`). It is a path, a `str` or an `os.PathLike[str]`, to a
/// file that does not exist yet, in a directory that does. If something is
/// already there, a file or a symbolic link, or the directory does not exist,
/// nothing is created or changed and the run goes on without a trace, after
/// one warning in the Polypus log (shown only once a logger is installed with
/// `polypus.init_logger` or `polypus.log_to_python`). A `bytes` path, or a
/// value that is not a path, raises `TypeError` before anything runs. The trace
/// never changes the result or the error. Without `trace`,
/// `POLYPUS_CALL_TRACE=<directory>` writes one trace per run there, named
/// `<id>-<pid>-<n>.jsonl` after the run's id (`TrainResult.id`, cut to its
/// first 100 characters and a hash if longer), the process id and the number of
/// traces the variable named before in the process; the directory must exist.
/// `POLYPUS_CALL_TRACE_MAX_MB` caps a trace, in whole MiB (512 MiB by default).
///
/// Interruption: pressing Ctrl+C (or otherwise sending `SIGINT`) stops the
/// optimization promptly and raises `KeyboardInterrupt` in Python, instead of
/// waiting for the run to finish, with or without a `callback`. An exception
/// raised by `expectation_function` propagates the same way, as itself.
///
/// `nodes` and `cores_per_qpu` size the SLURM allocation and are consumed only
/// by `infrastructure="cunqa"`; `local`/`qmio` accept but ignore them. For
/// `cunqa` both must be `>= 1` (a zero is meaningless to SLURM and rejected).
/// `walltime` (default `"10:00:00"`) is the SLURM walltime of the CUNQA QPUs,
/// also `cunqa` only: `"HH:MM:SS"` (hours may exceed 24, e.g. `"48:00:00"`;
/// `D-HH:MM:SS` is rejected), otherwise `ValueError` before any QPU is requested.
/// With `infrastructure="cunqa"`, whose QPUs run Aer, `backend` must be `"aer"`
/// (default) or `"AerSimulator"` and `noise_model` is not accepted; either one
/// raises `ValueError` before any QPU is requested.
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
/// Example:
///
/// ```ignore
///     result = polypus.train(
///         qc, polypus.DE(generations=200, population_size=50),
///         shots=1024, n_qpus=1, dimensions=4,
///         expectation_function=my_cost,
///         infrastructure="local", nodes=1, cores_per_qpu=2, id="run1"
///     )
/// ```
///
/// `target` (keyword-only, default `None`) is a typed target, `polypus.Aer`,
/// `Native`, `Cunqa`, `Qmio` or `Registered`, used instead of `infrastructure`,
/// `backend` and the provider arguments. Next to it, `infrastructure`,
/// `noise_model`, `fusion`, `nodes`, `cores_per_qpu` or `walltime` given at
/// all, or `backend` or `sim_method` with a non-default value, raises
/// `ConfigurationError`. Without `target`, `infrastructure` (with `nodes` and
/// `cores_per_qpu`) is required.
#[pyfunction(signature = (qc, method, shots, n_qpus, dimensions, expectation_function, infrastructure=None, nodes=None, cores_per_qpu=None, id=None, sim_method="automatic", noise_model=None, backend="aer", seed=None, fusion=None, options=None, walltime=None, initial_point=None, final_shots=None, *, callback=None, trace=None, target=None))]
pub fn train<'py>(
    qc: Bound<'py, PyAny>,
    method: Bound<'py, PyAny>,
    shots: u32,
    n_qpus: u32,
    dimensions: u32,
    expectation_function: Bound<'py, PyAny>,
    infrastructure: Option<String>,
    nodes: Option<u32>,
    cores_per_qpu: Option<u32>,
    id: Option<String>,
    sim_method: &str,
    noise_model: Option<Bound<'py, PyAny>>,
    backend: &str,
    seed: Option<u64>,
    fusion: Option<bool>,
    options: Option<OptionsArg>,
    walltime: Option<String>,
    initial_point: Option<Vec<f64>>,
    final_shots: Option<Bound<'py, PyAny>>,
    callback: Option<Bound<'py, PyAny>>,
    trace: Option<PathBuf>,
    target: Option<Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Read Python's log levels before the GIL is released below (`python_log`).
    python_log::refresh_levels(qc.py());
    let start = Instant::now();
    let selection = prepare::select(
        Entry::Train,
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
    let id = prepare::check_arguments(Entry::Train, &selection, shots, n_qpus, id)?;
    // Resolve the seed that drives the optimizer's RNG (contract C-7): explicit
    // kwarg > optimizer object's `seed` field > fresh OS entropy. Unlike
    // `run_quantum_circuit`, a seed is always meaningful here — it seeds the
    // optimizer regardless of backend — and it is also threaded into
    // `ExecutionConfig::seed` so the native backend's shot sampling becomes
    // deterministic too, making a native-backend training run fully reproducible.
    let effective_seed = resolve_optimizer_seed(seed, method_seed(&method), random_seed)?;
    // Native circuits know their parameter count — catch a mismatch with the
    // requested optimisation dimensions before any QPU work starts.
    let circuit_source = extract_circuit_source(&qc);
    if let Some(num_params) = circuit_source.num_params() {
        if num_params != dimensions as usize {
            return Err(crate::exceptions::configuration_error(format!(
                "dimensions ({dimensions}) does not match the circuit's free parameters ({num_params})"
            )));
        }
    }
    validate_method(&method, dimensions, initial_point.as_deref())?;
    let final_shots = validate_final_shots(final_shots.as_ref())?;
    let callback = validate_callback(callback.as_ref())?;
    // The native statevector backend runs pure-Rust circuits only, and QMIO
    // serialises GIL-free: neither can read a Qiskit template, which only the
    // interpreter can.
    prepare::reject_foreign_circuit(
        Entry::Train,
        &selection,
        matches!(circuit_source, CircuitSource::Qiskit(_)),
    )?;
    // Rotosolve's step is exact only on single-frequency circuits: checked
    // before any backend, and so any CUNQA allocation, exists.
    if is_rotosolve(&method) {
        check_rotosolve_circuit(&qc, Some(&circuit_source), None)?;
    }
    // As in `run_quantum_circuit`: the native backend's simulator resolves the
    // gate-parallel threshold, so it is calibrated lazily before it is built
    // (#351).
    let backend_config =
        prepare::backend_config(qc.py(), selection, options, Some(effective_seed))?;
    // Suffix the caller-supplied `id` with a UUID v4 so two concurrent training
    // runs sharing the same `id` never collide on the SLURM family/allocation,
    // temp files or log streams named by ExecutionConfig::id (#75, mirroring
    // run_quantum_circuit). The prefix is kept for debuggability and the
    // effective id is reported back in the TrainResult.
    let effective_id = unique_id(&id)?;
    let config = prepare::execution_config(
        effective_id.clone(),
        shots,
        n_qpus,
        infrastructure.clone(),
        backend_config,
        // The run's base seed: the native backend seeds each circuit from it,
        // Aer, CUNQA and the subprocess bridge each call (C-7); QMIO ignores
        // it. Set unconditionally so a seeded training run reproduces exactly.
        Some(effective_seed),
    );
    // Shared error slot: the oracle records the first evaluation failure here
    // (the optimizer traits cannot return a `Result`) and it is surfaced by
    // `finish_optimization` after `optimize` returns.
    let errors = OracleErrorSlot::new();
    // A callable stays a Python-callback observable (optimized fallback); a
    // polypus.Qubo/Ising opts into the native, GIL-free evaluation path.
    // Converted, like the method, before the backend exists: on CUNQA creating
    // it requests a SLURM allocation (issue #418).
    let observable = extract_cost_observable(&expectation_function)?;
    let method_enum = method_from_pyclass(&method, dimensions, &errors, &effective_id)?;
    // The `&str` device selector is what belongs in the manifest record, not
    // the constructed backend object.
    log_training_start(
        &effective_id,
        &infrastructure,
        &backend,
        n_qpus,
        shots,
        effective_seed,
    );
    // The call trace (contract C-12), if `trace` names a path or
    // `POLYPUS_CALL_TRACE` a directory (where the file is named after the
    // effective id). It starts before the backend exists, so its clock counts
    // the backend's start-up too (on CUNQA, raising the QPUs), and every way
    // out from here closes it with the status Python sees. Creating it writes
    // the header line to the file, so the GIL is released; a path that cannot
    // be created leaves the run untraced.
    let py = method.py();
    let header = TraceHeader {
        flow: TraceFlow::Train,
        id: effective_id.clone(),
        polypus: python_version(env!("CARGO_PKG_VERSION")),
        method: Some(method_enum.name()),
        dims: Some(dimensions),
        backend: backend.to_owned(),
        infrastructure: infrastructure.clone(),
        n_qpus,
        shots,
        seed: Some(effective_seed),
    };
    let tracer = py.detach(|| call_trace::open(trace, header));
    // Pair the backend with its default planner (the atomic-wave SequentialPlanner)
    // and validate the pairing up front.
    let scheduler = prepare::scheduler(&config, None)
        .map_err(crate::exceptions::infrastructure_error_to_pyerr)
        .map_err(|err| call_trace::close_on_error(py, tracer.as_ref(), err))?;
    let flow = TrainFlow {
        factory: VqcOracleFactory {
            circuit: circuit_source,
            observable,
        },
        method: method_enum,
        dimensions,
        seed: effective_seed,
        initial_point,
        final_shots,
        observer: training_observer(callback, &errors, &effective_id),
        errors: errors.clone(),
        trace: tracer.clone(),
    };
    // Release the GIL for the whole optimization: parameter binding and native
    // simulation are GIL-free, so holding it would stall every other Python
    // thread and (with the Planner's between-wave check_signals) keep Ctrl+C from
    // taking effect until the run ends. See docs/ENGINEERING.md §3. The run token
    // carries the `check_signals`-backed interrupt guard the pyo3-free planner
    // calls between waves, so Ctrl+C still aborts at the next wave boundary.
    let token = interruptible_token();
    let result = py.detach(|| scheduler.run_cancellable(flow, &token));
    scheduler.close();
    let out = finish_optimization(py, result, effective_seed, effective_id, start);
    call_trace::close(py, tracer.as_ref(), &out);
    out
}
