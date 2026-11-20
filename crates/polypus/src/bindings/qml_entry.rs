//! `polypus.qml.train` and `polypus.qml.predict`: a Qiskit feature map and
//! ansatz, trained on rows of data or run on them.

use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyModule};

use super::callback::validate_callback;
use super::extract::{extract_cost_observable, extract_labels, extract_supervised_objective};
use super::interrupt::interruptible_token;
use super::observable::SampleCost;
use super::optimizer::{
    check_rotosolve_circuit, is_rotosolve, method_from_pyclass, training_observer,
    validate_final_shots, validate_method,
};
use super::options::OptionsArg;
use super::prepare::{self, Entry, TargetArgs};
use super::qml;
use super::results::{counts_to_pylist, finish_optimization, log_training_start, RunResult};
use super::seeds::{method_seed, resolve_optimizer_seed, seed_or_entropy, unique_id};
use crate::evaluation::{OracleErrorSlot, QmlObjective, QmlOracleFactory, QmlSource};
use crate::infrastructure::execution_config::random_seed;
use crate::infrastructure::{BoundCircuit, Counts, InfrastructureError};
use crate::orchestration::{RunCircuitFlow, TrainFlow};

/// Compose `feature_map` with `ansatz` into the QML circuit template, adding a
/// terminal `measure_all` when there are no classical bits (Aer needs them to
/// return counts).
///
/// A Qiskit exception from these calls (an ansatz wider than the feature map,
/// a parameter-name clash) is raised as `polypus.EvaluationError`, chaining the
/// original (issue #218); nothing has run on a backend yet.
fn compose_qml_template<'py>(
    feature_map: &Bound<'py, PyAny>,
    ansatz: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let qiskit = crate::exceptions::qiskit_error_to_evaluation_error;
    let composed = feature_map
        .call_method1("compose", (ansatz,))
        .map_err(qiskit)?;
    let num_clbits: usize = composed.getattr("num_clbits")?.extract()?;
    if num_clbits == 0 {
        composed.call_method0("measure_all").map_err(qiskit)?;
    }
    Ok(composed)
}

/// Bind each row of `rows` by name to the feature-map parameters of `template`,
/// one circuit per row. The binding is partial: the ansatz parameters stay free.
///
/// A row whose length differs from `len(feature_map.parameters)` is a
/// `ValueError` naming `rows_name` and the 0-based row (contract C-8); zipping
/// would otherwise drop extra features or leave some unbound. A Qiskit exception
/// from binding a row (a value Qiskit rejects, such as a complex number) is
/// raised as `polypus.EvaluationError`, chaining the original (issue #218).
fn bind_feature_rows<'py>(
    template: &Bound<'py, PyAny>,
    feature_map: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    rows_name: &str,
) -> PyResult<Vec<Py<PyAny>>> {
    let py = template.py();
    let fm_params = feature_map.getattr("parameters")?;
    let builtins = PyModule::import(py, "builtins")?;
    let fm_params_list = builtins.call_method1("list", (&fm_params,))?;
    let kwargs_assign = [("inplace", false)].into_py_dict(py)?;
    let fm_len = fm_params_list.len()?;
    let mut circuits: Vec<Py<PyAny>> = Vec::new();
    for (row_idx, row_result) in rows.try_iter()?.enumerate() {
        let row = row_result?;
        let row_len = row.len()?;
        if row_len != fm_len {
            return Err(crate::exceptions::configuration_error(format!(
                "{rows_name} row {row_idx} has {row_len} features, but feature_map expects \
                 {fm_len} (len(feature_map.parameters))"
            )));
        }
        let param_dict = PyDict::new(py);
        for (param, val) in fm_params_list.try_iter()?.zip(row.try_iter()?) {
            param_dict.set_item(param?, val?)?;
        }
        circuits.push(
            template
                .call_method("assign_parameters", (&param_dict,), Some(&kwargs_assign))
                .map_err(crate::exceptions::qiskit_error_to_evaluation_error)?
                .unbind(),
        );
    }
    Ok(circuits)
}

/// QML entry point: train a data-encoding VQC where `feature_map` encodes each
/// training sample and `ansatz` holds the trainable weights.
///
/// Internally, this function:
/// 1. Composes `feature_map` and `ansatz` into a single circuit.
/// 2. Pre-binds each row of `x_train` to the feature-map parameters, producing
///    one partially-bound circuit per training sample.
/// 3. Delegates to the chosen optimizer with `TrainMode::Qml`, so that for
///    every candidate parameter vector θ the optimizer binds θ to all training
///    circuits, runs them, and averages the per-sample values into a single
///    fitness value (maximised).
///
/// # Labels (`y_train`)
///
/// Without `y_train`, every sample is scored by the same
/// `expectation_function(bitstring) -> float`. With `y_train` (keyword-only, one
/// label per `x_train` row), each sample is scored against its own label and
/// `expectation_function` is one of:
///
/// - a callable `f(bitstring, label) -> float`, averaged over each sample's shots
///   (the expected accuracy, when `f` returns `1.0` for a correct read-out);
/// - `polypus.CachedCost(f)`, the same memoised by `(label, bitstring)`;
/// - `polypus.SampleCost(g)`, where `g(counts, label) -> float` sees the sample's
///   whole distribution, for non-linear losses such as a log-likelihood.
///
/// Integer labels reach the objective as `int`; if any label is not an integer,
/// all are passed as `float`. `y_train` is validated before anything runs
/// (contract C-8).
///
/// `seed` follows the same precedence as [`train`] and makes the optimizer's
/// search reproducible; it returns a [`TrainResult`]. `qml.train` runs on the
/// Qiskit/Aer path (the native backend and `infrastructure="qmio"` are
/// rejected before anything runs), and Aer's shot sampling
/// is now seeded too (contract C-7), so a `qml.train` run is fully
/// reproducible end-to-end given the same seed.
///
/// As in [`train`], the `id` kwarg is a human-readable *prefix*: a UUID v4 is
/// appended for uniqueness (mirroring [`run_quantum_circuit`]), so the returned
/// `TrainResult.id` differs from what was passed in and names the run's SLURM
/// allocation / temp files / log stream (see #75). It carries the same charset
/// restriction as [`train`] — `[A-Za-z0-9._-]`, non-empty, at most 64
/// characters (contract C-9) — and an invalid prefix raises `ValueError`
/// naming the offending character.
///
/// `initial_point` (keyword-only, default `None`) starts the search from a
/// given point of the ansatz weights, as in [`train`]: `dimensions` finite
/// numbers, otherwise `ValueError` before anything runs. `polypus.External`
/// refuses it, as in [`train`].
///
/// With `method=polypus.Rotosolve(...)`, every ansatz parameter must be the
/// angle of exactly one single-Pauli rotation, as `±θ + c`; the feature map's
/// parameters are bound to each row, so they may appear anywhere. A
/// `polypus.SampleCost`, which is not an average of a cost per bitstring, is
/// rejected too. Both raise `ValueError` before anything runs.
///
/// `final_shots` (keyword-only, default `None`) re-evaluates the best
/// candidates with that many shots once the optimizer has stopped, as in
/// [`train`]: every training circuit of each candidate runs with them, and
/// `TrainResult.final_params` and `final_fitness` hold the result.
///
/// `callback` (keyword-only, default `None`) is called once per iteration of
/// the optimizer with a `polypus.TrainProgress`, and stops the run when it
/// returns `True`, as in [`train`]; its `evaluations` count candidates, each
/// scored with one circuit per training row.
///
/// Interruption: same behaviour as [`train`] — Ctrl+C stops the optimization
/// promptly and raises `KeyboardInterrupt` rather than waiting for the run to
/// finish, and an exception raised by `expectation_function` propagates as
/// itself.
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
/// Example (supervised binary classifier, parity read-out):
///
/// ```ignore
///     def correct(bitstring, label):
///         return float(bitstring.count("1") % 2 == label)
///
///     result = polypus.qml.train(
///         feature_map, ansatz, X_train,
///         polypus.PSO(generations=50, population_size=20, bounds=(0, np.pi)),
///         shots=1024, n_qpus=4, dimensions=12,
///         expectation_function=correct,
///         infrastructure="local", nodes=1, cores_per_qpu=2, id="qml_run",
///         y_train=y_train,
///     )
/// ```
///
/// `target` (keyword-only, default `None`) is a typed target, `polypus.Aer`,
/// `Native`, `Cunqa`, `Qmio` or `Registered`, used instead of `infrastructure`,
/// `backend` and the provider arguments. Next to it, `infrastructure`,
/// `noise_model`, `nodes`, `cores_per_qpu` or `walltime` given at all, or
/// `backend` or `sim_method` with a non-default value, raises
/// `ConfigurationError`. Without `target`, `infrastructure` (with `nodes` and
/// `cores_per_qpu`) is required.
#[pyfunction(name = "train", signature = (feature_map, ansatz, x_train, method, shots, n_qpus, dimensions, expectation_function, infrastructure=None, nodes=None, cores_per_qpu=None, id=None, sim_method="automatic", noise_model=None, backend="aer", seed=None, options=None, walltime=None, *, y_train=None, initial_point=None, final_shots=None, callback=None, target=None))]
pub fn qml_train<'py>(
    feature_map: Bound<'py, PyAny>,
    ansatz: Bound<'py, PyAny>,
    x_train: Bound<'py, PyAny>,
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
    options: Option<OptionsArg>,
    walltime: Option<String>,
    y_train: Option<Bound<'py, PyAny>>,
    initial_point: Option<Vec<f64>>,
    final_shots: Option<Bound<'py, PyAny>>,
    callback: Option<Bound<'py, PyAny>>,
    target: Option<Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let start = Instant::now();
    let selection = prepare::select(
        Entry::QmlTrain,
        TargetArgs {
            target: target.as_ref(),
            infrastructure,
            backend,
            sim_method,
            noise_model,
            fusion: None,
            nodes,
            cores_per_qpu,
            walltime,
        },
    )?;
    let (infrastructure, backend) = (selection.infrastructure.clone(), selection.backend.clone());
    let id = prepare::check_arguments(Entry::QmlTrain, &selection, shots, n_qpus, id)?;
    // Native models have their own entry point (plan D4-R).
    qml::reject_native_pieces("train", &feature_map, &ansatz)?;
    // Same seed precedence as `train` (contract C-7): kwarg > optimizer field >
    // OS entropy. qml.train always runs on a Qiskit/Aer path (native rejected
    // below); this seed governs the optimizer's RNG and, since it's threaded
    // into ExecutionConfig::seed below, Aer's shot sampling too.
    let effective_seed = resolve_optimizer_seed(seed, method_seed(&method), random_seed)?;
    // QML composes Qiskit feature maps and ansätze, so it is inherently a
    // Qiskit path; the native statevector backend cannot consume a Qiskit
    // `QuantumCircuit`, and neither can QMIO. Accept `backend` for API symmetry
    // but reject both.
    prepare::reject_foreign_circuit(Entry::QmlTrain, &selection, true)?;
    // The optimizer searches a `dimensions`-wide vector and binds it to the
    // ansatz's free parameters; a mismatch surfaces later as a cryptic Qiskit
    // binding error inside the oracle. Catch it upfront, mirroring how `train`
    // validates `dimensions` against the circuit's parameter count (contract
    // C-8). `len()` calls `__len__`, working on Qiskit's ParameterView the same
    // as on a list.
    let num_ansatz_params = ansatz.getattr("parameters")?.len()?;
    if num_ansatz_params != dimensions as usize {
        return Err(crate::exceptions::configuration_error(format!(
            "dimensions ({dimensions}) does not match the ansatz's free parameters ({num_ansatz_params})"
        )));
    }
    validate_method(&method, dimensions, initial_point.as_deref())?;
    let final_shots = validate_final_shots(final_shots.as_ref())?;
    let callback = validate_callback(callback.as_ref())?;
    // Type-checked before any circuit is composed; the count is checked once
    // `x_train` has been read (contract C-8).
    let labels = y_train.as_ref().map(extract_labels).transpose()?;
    let py = feature_map.py();

    // Rotosolve needs a fitness that averages a cost per bitstring, which a
    // `SampleCost` (a function of each sample's whole distribution) is not.
    let rotosolve = is_rotosolve(&method);
    if rotosolve
        && expectation_function
            .extract::<PyRef<'_, SampleCost>>()
            .is_ok()
    {
        return Err(crate::exceptions::configuration_error(
            "polypus.Rotosolve cannot train with a polypus.SampleCost: Rotosolve needs a \
             fitness that averages a cost per bitstring over the shots, so that it is a \
             sinusoid of each parameter, and a SampleCost scores each sample's whole \
             distribution. Use a callable (bitstring, label) -> float or a \
             polypus.CachedCost, or another optimizer",
        ));
    }

    let composed = compose_qml_template(&feature_map, &ansatz)?;
    // Only the ansatz is trained: the feature map's parameters are bound to
    // each row below, so they are constants wherever they appear.
    if rotosolve {
        check_rotosolve_circuit(&composed, None, Some(&ansatz.getattr("parameters")?))?;
    }
    // Pre-bind each sample, leaving the ansatz parameters for the optimizer.
    let qcs = bind_feature_rows(&composed, &feature_map, &x_train, "x_train")?;

    if qcs.is_empty() {
        return Err(crate::exceptions::configuration_error(
            "x_train must contain at least one training sample",
        ));
    }

    // One label per row (contract C-8), and an objective that can read labels:
    // both checked before any backend, and so any CUNQA allocation, exists.
    let supervised = match labels {
        Some(labels) => {
            if labels.len() != qcs.len() {
                return Err(crate::exceptions::configuration_error(format!(
                    "y_train has {} labels, but x_train has {} rows; qml.train needs exactly \
                     one label per training sample",
                    labels.len(),
                    qcs.len()
                )));
            }
            Some((extract_supervised_objective(&expectation_function)?, labels))
        }
        None => None,
    };

    // QML composes Qiskit feature maps and ansätze, so it is inherently a
    // Qiskit path (native backend already rejected above): `backend` can only be
    // an Aer variant here, which never fuses. Pass `None` (no fusion opinion)
    // rather than `Some(true)` — the latter would trip the "fusion=True on a
    // non-fusing backend" rejection for a value the caller never chose.
    let backend_config = prepare::backend_config(py, selection, options)?;
    // Suffix the caller-supplied `id` with a UUID v4 (see `train` and #75) so
    // concurrent qml.train runs sharing the same `id` never collide on the
    // SLURM family/allocation, temp files or log streams named by
    // ExecutionConfig::id. The effective id is reported back in the TrainResult.
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
    // Shared error slot (see `train`): the oracle records the first evaluation
    // failure here and `finish_optimization` surfaces it after `optimize`.
    let errors = OracleErrorSlot::new();
    // Converted before the backend exists, as in `train` (issue #418).
    let objective = match supervised {
        Some((objective, labels)) => QmlObjective::Supervised { objective, labels },
        None => QmlObjective::Unsupervised(extract_cost_observable(&expectation_function)?),
    };
    let method_enum = method_from_pyclass(&method, dimensions, &errors, &effective_id)?;
    // As in `train`: the device selector string is the manifest value, not the
    // constructed backend object.
    log_training_start(
        &effective_id,
        &infrastructure,
        &backend,
        n_qpus,
        shots,
        effective_seed,
    );
    let scheduler = prepare::scheduler(&config, None)
        .map_err(crate::exceptions::infrastructure_error_to_pyerr)?;
    let flow = TrainFlow {
        factory: QmlOracleFactory {
            source: QmlSource::Qiskit {
                training_circuits: qcs,
                objective,
            },
            minibatch: None,
        },
        method: method_enum,
        dimensions,
        seed: effective_seed,
        initial_point,
        final_shots,
        observer: training_observer(callback, &errors, &effective_id),
        errors: errors.clone(),
        trace: None,
    };
    // Release the GIL for the optimization (see `train` and docs/ENGINEERING.md
    // §3): the Qiskit binding + Aer calls re-acquire it, and the run token's
    // `check_signals`-backed interrupt guard — which the pyo3-free planner calls
    // between waves — keeps Ctrl+C prompt.
    let token = interruptible_token();
    let result = py.detach(|| scheduler.run_cancellable(flow, &token));
    scheduler.close();
    finish_optimization(py, result, effective_seed, effective_id, start)
}

/// Batched QML inference: run a trained `qml.train` model on every row of `x` in
/// one scheduled run, returning a [`RunResult`] whose `counts` holds one dict per
/// row, in row order, and whose `merged_counts` is `None` (the rows are distinct
/// circuits, so summing them has no meaning).
///
/// The circuits are built as in training: each row bound to the feature map, then
/// `params` (typically `TrainResult.best_params`) bound positionally to the
/// ansatz, as `QmlOracle` binds a candidate. With `n_qpus > 1` the rows are spread
/// over the QPUs; one row's shots are never split. `seed` behaves as in
/// `run_quantum_circuit` (contract C-7). The native backend and `qmio` are
/// rejected, since the model is a Qiskit circuit. Row width, the number and
/// finiteness of `params`, and a non-empty `x` are checked before anything runs
/// (contract C-8).
///
/// `nodes`, `cores_per_qpu` and `walltime` size the SLURM allocation and are
/// consumed only by `infrastructure="cunqa"`, with the same rules as in
/// [`run_quantum_circuit`]; `local` ignores them.
/// With `infrastructure="cunqa"`, whose QPUs run Aer, `backend` must be `"aer"`
/// (default) or `"AerSimulator"` and `noise_model` is not accepted; either one
/// raises `ValueError` before any QPU is requested.
///
/// Example:
///
/// ```ignore
///     run = polypus.qml.predict(
///         feature_map, ansatz, X_test, result.best_params,
///         shots=1024, infrastructure="local", seed=7,
///     )
///     predictions = [readout(counts) for counts in run.counts]
/// ```
///
/// `target` (keyword-only, default `None`) is a typed target, `polypus.Aer`,
/// `Native`, `Cunqa`, `Qmio` or `Registered`, used instead of `infrastructure`,
/// `backend` and the provider arguments. Next to it, `infrastructure`,
/// `noise_model`, `nodes`, `cores_per_qpu` or `walltime` given at all, or
/// `backend` or `sim_method` with a non-default value, raises
/// `ConfigurationError`. Without `target`, `infrastructure` is required.
// `nodes`/`cores_per_qpu`: see `run_quantum_circuit`.
#[pyfunction(name = "predict", signature = (feature_map, ansatz, x, params, shots, infrastructure=None, n_qpus=1, nodes=None, cores_per_qpu=None, sim_method="automatic", noise_model=None, backend="aer", seed=None, walltime=None, *, target=None), text_signature = "(feature_map, ansatz, x, params, shots, infrastructure=None, n_qpus=1, nodes=1, cores_per_qpu=2, sim_method=\"automatic\", noise_model=None, backend=\"aer\", seed=None, walltime=None, *, target=None)")]
pub fn qml_predict<'py>(
    feature_map: Bound<'py, PyAny>,
    ansatz: Bound<'py, PyAny>,
    x: Bound<'py, PyAny>,
    params: Vec<f64>,
    shots: u32,
    infrastructure: Option<String>,
    n_qpus: u32,
    nodes: Option<u32>,
    cores_per_qpu: Option<u32>,
    sim_method: &str,
    noise_model: Option<Bound<'py, PyAny>>,
    backend: &str,
    seed: Option<u64>,
    walltime: Option<String>,
    target: Option<Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let start = Instant::now();
    let selection = prepare::select(
        Entry::QmlPredict,
        TargetArgs {
            target: target.as_ref(),
            infrastructure,
            backend,
            sim_method,
            noise_model,
            fusion: None,
            nodes,
            cores_per_qpu,
            walltime,
        },
    )?;
    let (infrastructure, backend) = (selection.infrastructure.clone(), selection.backend.clone());
    let id_prefix = prepare::check_arguments(Entry::QmlPredict, &selection, shots, n_qpus, None)?;
    qml::reject_native_pieces("predict", &feature_map, &ansatz)?;
    // The model is a Qiskit circuit: neither the native backend nor QMIO runs it.
    prepare::reject_foreign_circuit(Entry::QmlPredict, &selection, true)?;
    // The weights must fill the ansatz exactly (contract C-8).
    let num_ansatz_params = ansatz.getattr("parameters")?.len()?;
    if params.len() != num_ansatz_params {
        return Err(crate::exceptions::configuration_error(format!(
            "params has {} values, but the ansatz has {num_ansatz_params} free parameters",
            params.len()
        )));
    }
    if let Some((index, value)) = params.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(crate::exceptions::configuration_error(format!(
            "params[{index}] is {value}; every weight must be a finite number"
        )));
    }
    let py = feature_map.py();

    // Built as in training: rows by name, then the weights positionally.
    let composed = compose_qml_template(&feature_map, &ansatz)?;
    let rows = bind_feature_rows(&composed, &feature_map, &x, "x")?;
    if rows.is_empty() {
        return Err(crate::exceptions::configuration_error(
            "x must contain at least one sample",
        ));
    }
    let kwargs_assign = [("inplace", false)].into_py_dict(py)?;
    let circuits: Vec<BoundCircuit> = rows
        .iter()
        .map(|row| {
            let circuit = row
                .bind(py)
                .call_method("assign_parameters", (params.clone(),), Some(&kwargs_assign))
                // Qiskit rejecting the weights (issue #218): `polypus.EvaluationError`.
                .map_err(crate::exceptions::qiskit_error_to_evaluation_error)?
                .unbind();
            Ok(crate::infrastructure::QiskitCircuit::into_bound(circuit))
        })
        .collect::<PyResult<_>>()?;

    // Always simulated (qmio is rejected), so a seed always applies (contract C-7).
    let effective_seed = seed_or_entropy(seed, random_seed)?;
    let id = unique_id(&id_prefix)?;
    // predict has no `options=`; a `Registered` target brings its own.
    let backend_config = prepare::backend_config(py, selection, None)?;
    let config = prepare::execution_config(
        id.clone(),
        shots,
        n_qpus,
        infrastructure.clone(),
        backend_config,
        Some(effective_seed),
    );
    let n_rows = circuits.len();
    log::info!(
        "qml.predict {id} starting: rows={n_rows}, infrastructure={infrastructure}, \
         backend={backend}, n_qpus={n_qpus}, shots={shots}, seed={effective_seed}"
    );

    // One run, GIL released. The default planner (never the shot-distributing one)
    // sends the rows in waves of the backend's concurrency; Ctrl+C is honoured
    // between waves.
    let counts_result = py.detach(move || -> Result<Vec<Counts>, InfrastructureError> {
        let scheduler = prepare::scheduler(&config, None)?;
        let out = scheduler.run_cancellable(
            RunCircuitFlow {
                circuits,
                shots,
                trace: None,
            },
            &interruptible_token(),
        );
        scheduler.close();
        out
    });
    let counts_vec = counts_result.map_err(crate::exceptions::infrastructure_error_to_pyerr)?;
    log::info!("qml.predict {id} completed: duration={:?}", start.elapsed());
    let counts = counts_to_pylist(py, &counts_vec)?.into_any().unbind();
    Py::new(
        py,
        RunResult {
            counts,
            // Each entry is a different row's circuit, not a replica of one, so
            // there is no meaningful merged total (C-3).
            merged_counts: py.None(),
            id,
            seed: Some(effective_seed),
            backend: backend.to_string(),
            infrastructure,
        },
    )
    .map(|result| result.into_any())
}
