//! What the four entry points (`run_quantum_circuit`, `train`, `qml.train`,
//! `qml.predict`) share to prepare a run (#419): choosing the target and
//! checking the arguments every one of them takes, the rule that keeps a
//! Qiskit circuit off the native backend and QMIO, and building the backend
//! the run goes to.
//!
//! These are phases, not one call: each entry point calls them in its own
//! order, so its own checks (the optimizer, `final_shots`, the callback, the
//! labels, the dimensions…) still come where they always did, and every
//! check still comes before the backend exists, which on CUNQA requests a
//! SLURM allocation (#418). The seed and the run id stay each entry point's
//! own (contract C-7): only the id's prefix is chosen here.

use std::sync::Arc;

use pyo3::prelude::*;

use super::backend_config::{build_backend_config, validate_id, validate_shots_and_qpus};
use super::calibration;
use super::options::OptionsArg;
use super::seeds::check_seed_for_backend;
use super::targets::{self, Selection};
use crate::exceptions::configuration_error;
use crate::infrastructure::{
    BackendConfig, ExecutionConfig, Infrastructure, InfrastructureError, OptLevel, Planner,
};
use crate::orchestration::{Resources, Scheduler};

/// Why `run_quantum_circuit(..., infrastructure="qmio")` rejects a Qiskit
/// `QuantumCircuit`; `backend_compatibility` reports the same text.
pub(super) const QMIO_REJECTS_QISKIT: &str =
    "the 'qmio' infrastructure runs entirely in Rust (GIL-free) and cannot \
     serialize a Qiskit QuantumCircuit; pass a polypus.Circuit or an OpenQASM \
     2.0 string";

/// Why `qml.train` and `qml.predict` reject `infrastructure="qmio"`.
const QMIO_REJECTS_QML: &str =
    "the 'qmio' infrastructure runs entirely in Rust (GIL-free) and cannot \
     serialize the Qiskit circuits of a QML model";

/// Why `run_quantum_circuit(..., backend="polypus")` rejects a Qiskit
/// `QuantumCircuit`; `backend_compatibility` reports the same text.
pub(super) const NATIVE_REJECTS_QISKIT: &str =
    "the native 'polypus' backend cannot execute a Qiskit QuantumCircuit; \
     pass a polypus.Circuit or an OpenQASM 2.0 string, or use backend=\"aer\"";

/// Why `train(..., backend="polypus")` rejects a Qiskit circuit template.
const TRAIN_NATIVE_REJECTS_QISKIT: &str =
    "the native 'polypus' backend requires a native polypus.Circuit; \
     got a Qiskit circuit. Build it with polypus.Circuit or use backend=\"aer\"";

/// Why `train(..., infrastructure="qmio")` rejects a Qiskit circuit template.
const TRAIN_QMIO_REJECTS_QISKIT: &str =
    "the 'qmio' infrastructure requires a native polypus.Circuit (GIL-free \
     serialization); got a Qiskit circuit. Build it with polypus.Circuit";

/// Why `qml.train(..., backend="polypus")` is rejected.
const QML_TRAIN_NATIVE: &str =
    "the native 'polypus' backend is not supported for qml.train (feature maps \
     and ansätze are Qiskit circuits); use backend=\"aer\"";

/// Why `qml.predict(..., backend="polypus")` is rejected.
const QML_PREDICT_NATIVE: &str =
    "the native 'polypus' backend is not supported for qml.predict (feature maps \
     and ansätze are Qiskit circuits); use backend=\"aer\"";

/// The entry point a run is prepared for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Entry {
    /// `polypus.run_quantum_circuit`.
    Run,
    /// `polypus.train`.
    Train,
    /// `polypus.qml.train`.
    QmlTrain,
    /// `polypus.qml.predict`.
    QmlPredict,
}

impl Entry {
    /// The function name its argument errors show (`train() missing …`):
    /// `qml.train` and `qml.predict` are `train` and `predict` in Python.
    fn name(self) -> &'static str {
        match self {
            Entry::Run => "run_quantum_circuit",
            Entry::Train | Entry::QmlTrain => "train",
            Entry::QmlPredict => "predict",
        }
    }

    /// Whether it trains: `nodes`, `cores_per_qpu` and `id` are then required
    /// on the string path, and `id` is the caller's.
    fn trains(self) -> bool {
        match self {
            Entry::Train | Entry::QmlTrain => true,
            Entry::Run | Entry::QmlPredict => false,
        }
    }

    /// The label its generated run id starts with, or `None` when the
    /// caller's `id` is the run id. The id names the SLURM job family, the
    /// temporary files and the log stream, so a new entry point has to choose
    /// it here rather than inherit one.
    fn id_label(self) -> Option<&'static str> {
        match self {
            Entry::Run => Some("run"),
            Entry::QmlPredict => Some("predict"),
            Entry::Train | Entry::QmlTrain => None,
        }
    }

    /// Why a Qiskit circuit cannot run on the native backend here.
    fn native_rejection(self) -> &'static str {
        match self {
            Entry::Run => NATIVE_REJECTS_QISKIT,
            Entry::Train => TRAIN_NATIVE_REJECTS_QISKIT,
            Entry::QmlTrain => QML_TRAIN_NATIVE,
            Entry::QmlPredict => QML_PREDICT_NATIVE,
        }
    }

    /// Why a Qiskit circuit cannot run on QMIO here.
    fn qmio_rejection(self) -> &'static str {
        match self {
            Entry::Run => QMIO_REJECTS_QISKIT,
            Entry::Train => TRAIN_QMIO_REJECTS_QISKIT,
            Entry::QmlTrain | Entry::QmlPredict => QMIO_REJECTS_QML,
        }
    }
}

/// The arguments that choose where a run executes: `target=`, or the
/// `infrastructure`/`backend` strings and the provider arguments. An entry
/// point without one of them passes `None` (`fusion` in `qml.train` and
/// `qml.predict`).
pub(super) struct TargetArgs<'a, 'py> {
    pub(super) target: Option<&'a Bound<'py, PyAny>>,
    pub(super) infrastructure: Option<String>,
    pub(super) backend: &'a str,
    pub(super) sim_method: &'a str,
    pub(super) noise_model: Option<Bound<'py, PyAny>>,
    pub(super) fusion: Option<bool>,
    pub(super) nodes: Option<u32>,
    pub(super) cores_per_qpu: Option<u32>,
    pub(super) walltime: Option<String>,
}

/// The [`Selection`] `args` make for `entry`, the first check of every entry
/// point: a `target=` next to a string argument is a `ConfigurationError`, and
/// without `target=`, `infrastructure` (and for training also `nodes` and
/// `cores_per_qpu`) is required.
pub(super) fn select(entry: Entry, args: TargetArgs<'_, '_>) -> PyResult<Selection> {
    let selection = Selection::new(
        entry.name(),
        args.target,
        args.infrastructure,
        args.backend,
        args.sim_method,
        args.noise_model,
        args.fusion,
        args.nodes,
        args.cores_per_qpu,
        args.walltime,
    )?;
    if entry.trains() {
        selection.require_allocation(entry.name())?;
    }
    Ok(selection)
}

/// The argument checks every entry point makes right after [`select`], in
/// this order: `id` given (training only), `shots` and `n_qpus` (contract
/// C-3), the CUNQA allocation, and the `id` charset (training only, contract
/// C-9).
///
/// Returns the prefix of the run id, which [`unique_id`](super::seeds::unique_id)
/// suffixes with a UUID: the caller's `id` when training,
/// `run_{n_qpus}_{infrastructure}` for `run_quantum_circuit` and
/// `predict_{n_qpus}_{infrastructure}` for `qml.predict`.
pub(super) fn check_arguments(
    entry: Entry,
    selection: &Selection,
    shots: u32,
    n_qpus: u32,
    id: Option<String>,
) -> PyResult<String> {
    match entry.id_label() {
        Some(label) => {
            validate_shots_and_qpus(shots, n_qpus)?;
            selection.validate_allocation()?;
            Ok(format!("{label}_{n_qpus}_{}", selection.infrastructure))
        }
        None => {
            let id = targets::require_arg(entry.name(), "id", id)?;
            validate_shots_and_qpus(shots, n_qpus)?;
            selection.validate_allocation()?;
            validate_id(&id)?;
            Ok(id)
        }
    }
}

/// The one rule on where a circuit can run: a Qiskit circuit (`foreign`) runs
/// on neither the native backend nor QMIO, which read circuits in Rust,
/// without the interpreter. Each entry point raises its own message, the
/// native one first; `qml.train` and `qml.predict` always pass `foreign`,
/// since a QML model is a Qiskit circuit.
pub(super) fn reject_foreign_circuit(
    entry: Entry,
    selection: &Selection,
    foreign: bool,
) -> PyResult<()> {
    if !foreign {
        return Ok(());
    }
    if selection.is_native() {
        return Err(configuration_error(entry.native_rejection()));
    }
    if selection.infrastructure == "qmio" {
        return Err(configuration_error(entry.qmio_rejection()));
    }
    Ok(())
}

/// The [`BackendConfig`] the run goes to: `selection` resolved with `options`
/// and mapped by [`build_backend_config`], with the run's resolved `seed`
/// checked against it ([`check_seed_for_backend`]: below `2^63` on Aer and
/// CUNQA, #393). When that is the native simulator, its gate-parallel
/// threshold is calibrated here, lazily, before the backend is built (#351);
/// no other backend reads it, so they never pay for it.
pub(super) fn backend_config(
    py: Python<'_>,
    selection: Selection,
    options: Option<OptionsArg>,
    seed: Option<u64>,
) -> PyResult<BackendConfig> {
    let target = selection.resolve(options.map(OptionsArg::into_inner))?;
    let config = build_backend_config(target)?;
    check_seed_for_backend(seed, &config)?;
    if matches!(config, BackendConfig::LocalNative { .. }) {
        calibration::prepare_native_threshold(py)?;
    }
    Ok(config)
}

/// The [`ExecutionConfig`] of a run: its effective `id`, its shots and QPUs,
/// where it goes and its base `seed` (`None` only on QMIO, which is not
/// seeded).
pub(super) fn execution_config(
    id: String,
    shots: u32,
    n_qpus: u32,
    infrastructure: String,
    backend_config: BackendConfig,
    seed: Option<u64>,
) -> ExecutionConfig {
    ExecutionConfig {
        id,
        shots,
        n_qpus,
        infrastructure,
        backend_config,
        opt_level: OptLevel::default(),
        seed,
    }
}

/// Create the backend `config` selects and a single-run scheduler over it,
/// with `planner`, or the backend's default planner when `None`, paired and
/// checked up front. Creating the backend is what requests a CUNQA
/// allocation, so every argument check comes before this.
pub(super) fn scheduler(
    config: &ExecutionConfig,
    planner: Option<Arc<dyn Planner>>,
) -> Result<Scheduler, InfrastructureError> {
    let backend = Infrastructure::create_backend(config).map_err(InfrastructureError::Backend)?;
    let resources = Resources::new(backend, planner, Arc::new(config.run_params()))?;
    Ok(Scheduler::ephemeral(resources))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::targets::{Native, Qmio};

    /// The selection of `infrastructure`/`backend` on the string path, or of
    /// `target` when given.
    fn selection(
        entry: Entry,
        target: Option<&Bound<'_, PyAny>>,
        infrastructure: Option<&str>,
        backend: &str,
    ) -> Selection {
        select(
            entry,
            TargetArgs {
                target,
                infrastructure: infrastructure.map(str::to_owned),
                backend,
                sim_method: "automatic",
                noise_model: None,
                fusion: None,
                // Required for training on the string path, refused next to a
                // target.
                nodes: target.is_none().then_some(1),
                cores_per_qpu: target.is_none().then_some(1),
                walltime: None,
            },
        )
        .expect("a valid selection")
    }

    /// The circuit rule over every entry point, every kind of destination and
    /// both kinds of circuit: a Qiskit circuit is refused on the native
    /// backend (the native message first, whatever the infrastructure) and on
    /// QMIO, each entry point with its own message; a native circuit, and any
    /// circuit elsewhere, passes.
    #[test]
    fn a_qiskit_circuit_is_refused_on_native_and_qmio_with_each_entry_points_message() {
        Python::initialize();
        Python::attach(|py| {
            let native_target = py.get_type::<Native>().call0().unwrap();
            let qmio_target = py.get_type::<Qmio>().call1(("openqasm",)).unwrap();
            let entries = [
                (Entry::Run, NATIVE_REJECTS_QISKIT, QMIO_REJECTS_QISKIT),
                (
                    Entry::Train,
                    "the native 'polypus' backend requires a native polypus.Circuit; got a \
                     Qiskit circuit. Build it with polypus.Circuit or use backend=\"aer\"",
                    "the 'qmio' infrastructure requires a native polypus.Circuit (GIL-free \
                     serialization); got a Qiskit circuit. Build it with polypus.Circuit",
                ),
                (
                    Entry::QmlTrain,
                    "the native 'polypus' backend is not supported for qml.train (feature \
                     maps and ansätze are Qiskit circuits); use backend=\"aer\"",
                    QMIO_REJECTS_QML,
                ),
                (
                    Entry::QmlPredict,
                    "the native 'polypus' backend is not supported for qml.predict (feature \
                     maps and ansätze are Qiskit circuits); use backend=\"aer\"",
                    QMIO_REJECTS_QML,
                ),
            ];
            for (entry, native_message, qmio_message) in entries {
                let destinations = [
                    (
                        selection(entry, None, Some("local"), "polypus"),
                        Some(native_message),
                    ),
                    (
                        selection(entry, Some(&native_target), None, "aer"),
                        Some(native_message),
                    ),
                    (
                        selection(entry, None, Some("qmio"), "polypus"),
                        Some(native_message),
                    ),
                    (
                        selection(entry, None, Some("qmio"), "openqasm"),
                        Some(qmio_message),
                    ),
                    (
                        selection(entry, Some(&qmio_target), None, "aer"),
                        Some(qmio_message),
                    ),
                    (selection(entry, None, Some("local"), "aer"), None),
                    (selection(entry, None, Some("cunqa"), "aer"), None),
                    (
                        selection(entry, None, Some("a-registered-name"), "aer"),
                        None,
                    ),
                ];
                for (selection, refusal) in destinations {
                    let place = format!(
                        "{entry:?} on {}/{}",
                        selection.infrastructure, selection.backend
                    );
                    assert!(
                        reject_foreign_circuit(entry, &selection, false).is_ok(),
                        "{place}: a native circuit must pass"
                    );
                    let outcome = reject_foreign_circuit(entry, &selection, true);
                    match refusal {
                        None => assert!(outcome.is_ok(), "{place}: a Qiskit circuit must pass"),
                        Some(message) => {
                            let err = outcome.expect_err(&place);
                            crate::exceptions::tests::assert_configuration_error(&err, message);
                            assert_eq!(err.value(py).to_string(), message, "{place}");
                        }
                    }
                }
            }
        });
    }

    /// The two per-entry choices about the run id agree: an entry point that
    /// trains takes the caller's `id`, and every other one generates its id
    /// from its own label.
    #[test]
    fn an_entry_point_generates_its_id_exactly_when_it_does_not_train() {
        for entry in [Entry::Run, Entry::Train, Entry::QmlTrain, Entry::QmlPredict] {
            assert_eq!(entry.id_label().is_none(), entry.trains(), "{entry:?}");
        }
    }

    /// The run id's prefix: the caller's `id` when training, checked after
    /// `shots` and the allocation; generated from `n_qpus` and the
    /// infrastructure otherwise.
    #[test]
    fn the_run_id_prefix_is_the_callers_id_or_generated() {
        Python::initialize();
        let run = selection(Entry::Run, None, Some("local"), "aer");
        assert_eq!(
            check_arguments(Entry::Run, &run, 8, 3, None).unwrap(),
            "run_3_local"
        );
        let predict = selection(Entry::QmlPredict, None, Some("cunqa"), "aer");
        assert_eq!(
            check_arguments(Entry::QmlPredict, &predict, 8, 2, None).unwrap(),
            "predict_2_cunqa"
        );
        let train = selection(Entry::Train, None, Some("local"), "aer");
        assert_eq!(
            check_arguments(Entry::Train, &train, 8, 1, Some("run1".to_string())).unwrap(),
            "run1"
        );
        // `shots` before the `id` charset; a missing `id` before both.
        let err = check_arguments(Entry::Train, &train, 0, 1, Some("a b".to_string()))
            .expect_err("shots=0 is rejected");
        crate::exceptions::tests::assert_configuration_error(&err, "shots must be >= 1, got 0");
        let err = check_arguments(Entry::Train, &train, 8, 1, Some("a b".to_string()))
            .expect_err("a space is rejected");
        crate::exceptions::tests::assert_configuration_error(&err, "invalid character ' '");
        let err = check_arguments(Entry::QmlTrain, &train, 0, 1, None)
            .expect_err("a missing id is rejected");
        assert_eq!(
            err.to_string(),
            "TypeError: train() missing 1 required positional argument: 'id'"
        );
    }
}
