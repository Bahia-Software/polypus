//! From a [`Target`] to the [`BackendConfig`] a run is built from, and the
//! argument checks every entry point shares: `shots`/`n_qpus`, the CUNQA
//! allocation and the `id` prefix (contract C-9).

use std::collections::HashMap;

use pyo3::prelude::*;

use super::targets::{target_from_strings, ProgramFormat, Target};
use crate::infrastructure::{validate_walltime, BackendConfig, OptionValue, DEFAULT_WALLTIME};

/// The [`BackendConfig`] a [`Target`] runs on. Every check lives where the
/// target is built ([`targets::target_from_strings`] or the target classes),
/// so this only maps fields; QMIO alone can still fail, in a build without
/// the `qmio` feature.
pub(super) fn build_backend_config(target: Target) -> PyResult<BackendConfig> {
    Ok(match target {
        Target::Aer {
            sim_method,
            noise_model,
        } => BackendConfig::Local {
            sim_method,
            noise_model,
        },
        Target::Native { fusion } => BackendConfig::LocalNative {
            fusion: fusion.unwrap_or(true),
        },
        Target::Cunqa {
            nodes,
            cores_per_qpu,
            walltime,
            sim_method,
        } => BackendConfig::Cunqa {
            sim_method,
            nodes,
            cores_per_qpu,
            walltime: walltime.unwrap_or_else(|| DEFAULT_WALLTIME.to_string()),
        },
        Target::Qmio {
            program_format,
            options,
        } => build_qmio_registered(program_format, options)?,
        Target::Registered { name, options } => BackendConfig::Registered { name, options },
    })
}

/// The [`BackendConfig`] the `infrastructure`/`backend` strings select:
/// [`target_from_strings`], then [`build_backend_config`]. For callers that
/// take only the strings (the native QML model).
pub(super) fn config_from_strings(
    infrastructure: &str,
    backend: &str,
    sim_method: &str,
    noise_model: Option<Py<PyAny>>,
    nodes: u32,
    cores_per_qpu: u32,
    walltime: Option<&str>,
    fusion: Option<bool>,
    options: Option<HashMap<String, OptionValue>>,
) -> PyResult<BackendConfig> {
    let target = target_from_strings(
        infrastructure,
        backend,
        sim_method,
        noise_model,
        nodes,
        cores_per_qpu,
        walltime,
        fusion,
        options,
    )?;
    build_backend_config(target)
}

/// Reject the Aer-only arguments on an infrastructure that cannot honour them:
/// QMIO (real hardware) and every registry-dispatched backend, whose
/// `BackendConfig::Registered` has no field for either, so they would be
/// dropped silently. `sim_method` is compared with the entry points' default,
/// `"automatic"`, so a call that leaves it alone passes.
pub(super) fn reject_aer_only_args(
    infrastructure: &str,
    sim_method: &str,
    noise_model: Option<&Py<PyAny>>,
) -> PyResult<()> {
    if noise_model.is_some() {
        return Err(crate::exceptions::configuration_error(format!(
            "the '{infrastructure}' infrastructure does not accept a noise_model; it applies \
             only to Aer (infrastructure=\"local\" with backend=\"aer\")"
        )));
    }
    if sim_method != "automatic" {
        return Err(crate::exceptions::configuration_error(format!(
            "the '{infrastructure}' infrastructure does not accept sim_method=\"{sim_method}\"; \
             it applies only to Aer (infrastructure=\"local\" or \"cunqa\")"
        )));
    }
    Ok(())
}

/// Build the QMIO backend's registry config for the CESGA QMIO QPU.
///
/// The endpoint is read from the `ZMQ_SERVER` environment variable (the same
/// variable the reference `qmio` Python client uses); there is no default, so with
/// neither `ZMQ_SERVER` nor an `endpoint` option the backend fails to build with
/// `polypus.QmioError`. `program_format` is the program representation
/// submitted to the QPU. Caller-supplied `options` override the
/// computed defaults (e.g. `optimization`, `repetition_period`, `res_format`, a
/// custom `endpoint`). Only available with `--features qmio`; otherwise it returns
/// an actionable error instead of silently degrading.
#[cfg(feature = "qmio")]
pub(super) fn build_qmio_registered(
    program_format: ProgramFormat,
    options: HashMap<String, OptionValue>,
) -> PyResult<BackendConfig> {
    qmio_registered_config(program_format, std::env::var("ZMQ_SERVER").ok(), options)
}

/// [`build_qmio_registered`] with the `ZMQ_SERVER` value passed in, so tests do not
/// depend on the process environment.
#[cfg(feature = "qmio")]
pub(super) fn qmio_registered_config(
    program_format: ProgramFormat,
    zmq_server: Option<String>,
    options: HashMap<String, OptionValue>,
) -> PyResult<BackendConfig> {
    let mut opts: HashMap<String, OptionValue> = HashMap::new();
    if let Some(endpoint) = zmq_server {
        opts.insert("endpoint".to_string(), OptionValue::from(endpoint));
    }
    opts.insert(
        "program_format".to_string(),
        OptionValue::from(program_format.option_value()),
    );
    opts.insert("optimization".to_string(), OptionValue::from("0"));
    opts.insert("res_format".to_string(), OptionValue::from("binary_count"));
    opts.extend(options);
    Ok(BackendConfig::Registered {
        name: "qmio".to_string(),
        options: opts,
    })
}

/// Without the `qmio` feature, selecting the QMIO infrastructure fails with a
/// clear, actionable message instead of pulling a ZeroMQ stack into every build.
#[cfg(not(feature = "qmio"))]
pub(super) fn build_qmio_registered(
    _program_format: ProgramFormat,
    _options: HashMap<String, OptionValue>,
) -> PyResult<BackendConfig> {
    Err(crate::exceptions::configuration_error(QMIO_NOT_COMPILED))
}

/// Why `infrastructure="qmio"` fails in a build without the `qmio` feature;
/// `backend_compatibility` reports the same text.
#[cfg(not(feature = "qmio"))]
pub(crate) const QMIO_NOT_COMPILED: &str =
    "the 'qmio' infrastructure requires compiling polypus with --features qmio";

/// Validate the shot/QPU parameters at the Python-facing boundary, so the
/// orchestration layer can assume `shots >= 1` and `n_qpus >= 1` (contract
/// C-3). Rejecting here also avoids the division-by-zero panic that `n_qpus = 0`
/// would otherwise trigger in `DistributeByShotsRun`. Shared by every entry
/// point rather than duplicated per function.
pub(super) fn validate_shots_and_qpus(shots: u32, n_qpus: u32) -> PyResult<()> {
    if shots < 1 {
        return Err(crate::exceptions::configuration_error(format!(
            "shots must be >= 1, got {shots}"
        )));
    }
    if n_qpus < 1 {
        return Err(crate::exceptions::configuration_error(format!(
            "n_qpus must be >= 1, got {n_qpus}"
        )));
    }
    Ok(())
}

/// Validate the CUNQA allocation parameters at the Python-facing boundary. Only
/// the `"cunqa"` infrastructure consumes `nodes`/`cores_per_qpu`/`walltime` (they
/// flow to SLURM's `qraise` as the C-1 kwargs `n_nodes`/`cores_per_qpu`/`t` via
/// `CunqaBackend`); the `local` and `qmio` arms of [`target_from_strings`]
/// ignore them, so validating unconditionally would reject harmless non-cunqa
/// calls. A zero for either count has no sane meaning once it reaches SLURM, and
/// a `walltime` that is not `HH:MM:SS` would only fail inside `qraise`, so
/// reject both here before any seam call rather than forward them. Shared by
/// every entry point rather than duplicated per function.
pub(super) fn validate_cunqa_allocation(
    infrastructure: &str,
    nodes: u32,
    cores_per_qpu: u32,
    walltime: Option<&str>,
) -> PyResult<()> {
    if infrastructure != "cunqa" {
        return Ok(());
    }
    if nodes < 1 {
        return Err(crate::exceptions::configuration_error(format!(
            "nodes must be >= 1 for the 'cunqa' infrastructure, got {nodes}"
        )));
    }
    if cores_per_qpu < 1 {
        return Err(crate::exceptions::configuration_error(format!(
            "cores_per_qpu must be >= 1 for the 'cunqa' infrastructure, got {cores_per_qpu}"
        )));
    }
    if let Some(walltime) = walltime {
        validate_walltime(walltime)
            .map_err(|e| crate::exceptions::configuration_error(e.to_string()))?;
    }
    Ok(())
}

/// Maximum length accepted for the caller-supplied `id` prefix
/// (`train`/`qml.train`). Bounded so the effective id — prefix plus the 36-char
/// UUID v4 suffix [`unique_id`] appends — stays comfortably inside the limits
/// SLURM job names and filesystem path components impose downstream.
pub(super) const MAX_ID_LEN: usize = 64;

/// Validate the caller-supplied `id` at the Python-facing boundary
/// (defense-in-depth, `docs/ENGINEERING.md` §8; contract C-9).
///
/// `id` becomes the CUNQA SLURM `family_name`/`family_id` (contract C-1,
/// `crate::infrastructure::cunqa`) and is documented as naming temp files and
/// log streams ([`ExecutionConfig::id`]), so an unrestricted string could carry
/// whitespace, path separators (`../`) or shell metacharacters into whatever
/// `qraise`/SLURM does with it downstream — which this crate cannot see. The
/// accepted charset is `[A-Za-z0-9._-]`, non-empty and at most
/// [`MAX_ID_LEN`] characters.
///
/// Checked before [`unique_id`] appends the UUID suffix, so a rejected id never
/// produces a partially-valid effective id. The charset is checked before the
/// length so a non-ASCII id gets the specific "invalid character" message
/// (and, past that check, every character is one byte, making the reported
/// length exact). Shared by every entry point rather than duplicated per
/// function, mirroring [`validate_shots_and_qpus`] and
/// [`validate_cunqa_allocation`].
pub(super) fn validate_id(id: &str) -> PyResult<()> {
    if id.is_empty() {
        return Err(crate::exceptions::configuration_error(
            "id must not be empty",
        ));
    }
    if let Some(c) = id
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')))
    {
        return Err(crate::exceptions::configuration_error(format!(
            "id contains invalid character {c:?}; only ASCII letters, digits, \
			 '.', '_' and '-' are allowed (got {id:?})"
        )));
    }
    if id.len() > MAX_ID_LEN {
        return Err(crate::exceptions::configuration_error(format!(
            "id must be at most {MAX_ID_LEN} characters, got {} ({id:?})",
            id.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Direct unit coverage of the private entry-point helpers.
    //
    // `tests/python/test_backend_selection.py` already exercises these through
    // the full `#[pyfunction]`s end-to-end; these tests pin the helpers
    // themselves, which the audit flagged as having no unit-level coverage. They
    // are deliberately lean — the goal is closing that gap, not re-proving the
    // end-to-end behaviour. Nothing here needs an installed package: the
    // `Py<PyAny>` arguments are `py.None()` against a bare interpreter.
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn validate_shots_and_qpus_accepts_the_minimum() {
        assert!(validate_shots_and_qpus(1, 1).is_ok());
        assert!(validate_shots_and_qpus(1024, 8).is_ok());
    }

    #[test]
    fn validate_shots_and_qpus_rejects_zero() {
        // `n_qpus = 0` used to reach `DistributeByShotsRun` and divide by zero;
        // `shots = 0` silently ran an empty execution (contract C-3).
        assert!(validate_shots_and_qpus(0, 1).is_err());
        assert!(validate_shots_and_qpus(1, 0).is_err());
        assert!(validate_shots_and_qpus(0, 0).is_err());
    }

    #[test]
    fn build_backend_config_selects_the_local_variants() {
        let aer = config_from_strings("local", "aer", "automatic", None, 1, 2, None, None, None)
            .expect("aer is a valid local backend");
        assert!(matches!(
            aer,
            BackendConfig::Local {
                ref sim_method,
                noise_model: None,
            } if sim_method == "automatic"
        ));

        for name in ["polypus", "statevector", "polypus_statevector"] {
            let native = config_from_strings(
                "local",
                name,
                "automatic",
                None,
                1,
                2,
                None,
                Some(true),
                None,
            )
            .unwrap_or_else(|_| panic!("'{name}' selects the native backend"));
            assert!(matches!(
                native,
                BackendConfig::LocalNative { fusion: true }
            ));
        }
    }

    /// The `fusion` kwarg reaches `BackendConfig::LocalNative` unchanged, in
    /// both directions — it is not silently forced to `true`.
    #[test]
    fn build_backend_config_forwards_fusion_for_the_native_backend() {
        let with_fusion = config_from_strings(
            "local",
            "polypus",
            "automatic",
            None,
            1,
            2,
            None,
            Some(true),
            None,
        )
        .expect("polypus is a valid local backend");
        assert!(matches!(
            with_fusion,
            BackendConfig::LocalNative { fusion: true }
        ));

        let without_fusion = config_from_strings(
            "local",
            "polypus",
            "automatic",
            None,
            1,
            2,
            None,
            Some(false),
            None,
        )
        .expect("polypus is a valid local backend");
        assert!(matches!(
            without_fusion,
            BackendConfig::LocalNative { fusion: false }
        ));
    }

    /// Omitting `fusion` (`None`) leaves the native backend on its default
    /// (`true`) — the crate-wide `StatevectorSimulator::default().fusion`.
    #[test]
    fn build_backend_config_defaults_fusion_to_enabled_when_omitted() {
        let native = config_from_strings(
            "local",
            "polypus",
            "automatic",
            None,
            1,
            2,
            None,
            None,
            None,
        )
        .expect("polypus is a valid local backend");
        assert!(matches!(
            native,
            BackendConfig::LocalNative { fusion: true }
        ));
    }

    /// `fusion=True` on a backend that cannot fuse is rejected: a request that
    /// cannot be met must not look like it took effect. `fusion=False` and an
    /// omitted `fusion` — both honourable everywhere, since a non-fusing backend
    /// already runs gate-by-gate — are accepted unchanged.
    #[test]
    fn build_backend_config_rejects_fusion_true_on_non_fusing_backends() {
        pyo3::Python::initialize();
        for (infra, backend) in [("local", "aer"), ("cunqa", "aer")] {
            let err = config_from_strings(
                infra,
                backend,
                "automatic",
                None,
                1,
                2,
                None,
                Some(true),
                None,
            )
            .expect_err("fusion=True on a non-fusing backend must be rejected");
            Python::attach(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(
                    err.to_string().contains("fusion=True applies only to"),
                    "unexpected error: {err}"
                );
            });

            // fusion=False and an omitted fusion must build a config, not error.
            config_from_strings(
                infra,
                backend,
                "automatic",
                None,
                1,
                2,
                None,
                Some(false),
                None,
            )
            .expect("fusion=False is accepted everywhere");
            config_from_strings(infra, backend, "automatic", None, 1, 2, None, None, None)
                .expect("omitted fusion is accepted everywhere");
        }
    }

    #[test]
    fn build_backend_config_rejects_an_unknown_local_backend() {
        let err = config_from_strings(
            "local",
            "does-not-exist",
            "automatic",
            None,
            1,
            2,
            None,
            None,
            None,
        )
        .expect_err("an unknown local backend must be rejected");
        pyo3::Python::initialize();
        Python::attach(|py| {
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(err.to_string().contains("unknown local backend"));
        });
    }

    /// CUNQA's QPUs run Aer, so any other `backend` (the native aliases
    /// included) is rejected instead of being dropped while the manifest echoes it.
    #[test]
    fn build_backend_config_rejects_an_unknown_cunqa_backend() {
        pyo3::Python::initialize();
        for backend in ["polypus", "statevector", "polypus_statevector", "foo"] {
            let err =
                config_from_strings("cunqa", backend, "automatic", None, 1, 2, None, None, None)
                    .expect_err("a non-Aer cunqa backend must be rejected");
            Python::attach(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(
                    err.to_string().contains("unknown cunqa backend"),
                    "unexpected error for {backend}: {err}"
                );
            });
        }
        for backend in ["aer", "AerSimulator"] {
            let config =
                config_from_strings("cunqa", backend, "automatic", None, 1, 2, None, None, None)
                    .expect("cunqa accepts the Aer backend names");
            assert!(matches!(config, BackendConfig::Cunqa { .. }));
        }
    }

    #[test]
    fn build_backend_config_rejects_a_noise_model_on_cunqa() {
        // `BackendConfig::Cunqa` has no noise model to carry, so one must be an
        // error rather than a silently noiseless run. As on the native backend,
        // any Python object will do — the check is `Option::is_some`.
        pyo3::Python::initialize();
        let noise_model = Python::attach(|py| py.None());
        let err = config_from_strings(
            "cunqa",
            "aer",
            "automatic",
            Some(noise_model),
            1,
            2,
            None,
            None,
            None,
        )
        .expect_err("a noise model on cunqa must be rejected");
        Python::attach(|py| {
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.to_string().contains("noise_model"),
                "the message must name the offending kwarg: {err}"
            );
        });
        assert!(matches!(
            config_from_strings("cunqa", "aer", "automatic", None, 1, 2, None, None, None),
            Ok(BackendConfig::Cunqa { .. })
        ));
    }

    #[test]
    fn build_backend_config_rejects_aer_only_args_on_a_registered_backend() {
        // `subprocess` is a built-in registry backend; `Registered` has no field
        // for these, so each must be an error naming the kwarg.
        pyo3::Python::initialize();
        let build = |backend: &str, sim_method: &str, noise: bool| {
            let noise_model = noise.then(|| Python::attach(|py| py.None()));
            config_from_strings(
                "subprocess",
                backend,
                sim_method,
                noise_model,
                1,
                2,
                None,
                None,
                None,
            )
        };
        assert!(build("aer", "automatic", false).is_ok());
        for (backend, sim_method, noise, kwarg) in [
            ("aer", "automatic", true, "noise_model"),
            ("aer", "statevector", false, "sim_method"),
            ("polypus", "automatic", false, "backend"),
        ] {
            let err = build(backend, sim_method, noise)
                .expect_err("an argument the backend cannot honour must be rejected");
            Python::attach(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(err.to_string().contains(kwarg), "{kwarg}: {err}");
            });
        }
    }

    #[cfg(feature = "qmio")]
    #[test]
    fn build_backend_config_rejects_a_noise_model_and_sim_method_on_qmio() {
        // Real hardware: a noise model or an Aer method would be dropped.
        // `backend` stays the program format, so "qir" is accepted.
        pyo3::Python::initialize();
        let build = |sim_method: &str, noise: bool| {
            let noise_model = noise.then(|| Python::attach(|py| py.None()));
            config_from_strings(
                "qmio",
                "qir",
                sim_method,
                noise_model,
                1,
                2,
                None,
                None,
                None,
            )
        };
        assert!(build("automatic", false).is_ok());
        for (sim_method, noise, kwarg) in [
            ("automatic", true, "noise_model"),
            ("density_matrix", false, "sim_method"),
        ] {
            let err = build(sim_method, noise).expect_err("must be rejected on qmio");
            Python::attach(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(err.to_string().contains(kwarg), "{kwarg}: {err}");
            });
        }
    }

    #[test]
    fn build_backend_config_rejects_a_noise_model_on_the_native_backend() {
        // The native simulator is noiseless by construction, so a noise_model
        // must be an error rather than silently ignored. Any Python object will
        // do — the check is `Option::is_some`, not a Qiskit type check.
        pyo3::Python::initialize();
        let noise_model = Python::attach(|py| py.None());
        let err = config_from_strings(
            "local",
            "polypus",
            "automatic",
            Some(noise_model),
            1,
            2,
            None,
            None,
            None,
        )
        .expect_err("a noise model on the native backend must be rejected");
        Python::attach(|py| {
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.to_string().contains("noise_model"),
                "the message must name the offending kwarg: {err}"
            );
        });
    }

    #[test]
    fn build_backend_config_rejects_a_sim_method_on_the_native_backend() {
        // `LocalNative` has no field for `sim_method`, so anything but the
        // default must be an error rather than silently ignored (#334), on
        // every name of the native backend.
        pyo3::Python::initialize();
        for backend in ["polypus", "statevector", "polypus_statevector"] {
            let err = config_from_strings(
                "local",
                backend,
                "statevector",
                None,
                1,
                2,
                None,
                None,
                None,
            )
            .expect_err("a sim_method on the native backend must be rejected");
            Python::attach(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(
                    err.to_string().contains("sim_method"),
                    "the message must name the offending kwarg: {err}"
                );
            });
        }
    }

    #[test]
    fn build_backend_config_keeps_a_noise_model_for_aer() {
        pyo3::Python::initialize();
        let noise_model = Python::attach(|py| py.None());
        let config = config_from_strings(
            "local",
            "aer",
            "density_matrix",
            Some(noise_model),
            1,
            2,
            None,
            None,
            None,
        )
        .expect("aer accepts a noise model");
        assert!(matches!(
            config,
            BackendConfig::Local {
                noise_model: Some(_),
                ref sim_method,
                ..
            } if sim_method == "density_matrix"
        ));
    }

    #[test]
    fn build_backend_config_forwards_the_cunqa_allocation() {
        let config =
            config_from_strings("cunqa", "aer", "statevector", None, 3, 4, None, None, None)
                .expect("cunqa is a valid infrastructure");
        assert!(matches!(
            config,
            BackendConfig::Cunqa {
                nodes: 3,
                cores_per_qpu: 4,
                ref sim_method,
                ref walltime,
                ..
            } if sim_method == "statevector" && walltime == DEFAULT_WALLTIME
        ));
    }

    #[test]
    fn build_backend_config_forwards_an_explicit_cunqa_walltime() {
        let config = config_from_strings(
            "cunqa",
            "aer",
            "statevector",
            None,
            1,
            2,
            Some("48:00:00"),
            None,
            None,
        )
        .expect("cunqa is a valid infrastructure");
        assert!(matches!(
            config,
            BackendConfig::Cunqa { ref walltime, .. } if walltime == "48:00:00"
        ));
    }

    #[test]
    fn validate_cunqa_allocation_checks_the_walltime_only_for_cunqa() {
        let msg = |r: PyResult<()>| r.expect_err("must be rejected").to_string();
        assert!(
            msg(validate_cunqa_allocation("cunqa", 1, 2, Some("1-00:00:00"))).contains("HH:MM:SS")
        );
        assert!(validate_cunqa_allocation("cunqa", 1, 2, Some("48:00:00")).is_ok());
        assert!(validate_cunqa_allocation("cunqa", 1, 2, None).is_ok());
        // local/qmio ignore the kwarg, as they do nodes/cores_per_qpu.
        for infra in ["local", "qmio"] {
            assert!(validate_cunqa_allocation(infra, 1, 2, Some("garbage")).is_ok());
        }
    }

    #[test]
    fn build_backend_config_rejects_an_unknown_infrastructure() {
        let err = config_from_strings(
            "quantum-cloud",
            "aer",
            "automatic",
            None,
            1,
            2,
            None,
            None,
            None,
        )
        .expect_err("an unknown infrastructure must be rejected");
        pyo3::Python::initialize();
        Python::attach(|py| {
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(err.to_string().contains("unknown infrastructure"));
        });
    }

    /// A name registered at runtime resolves to `BackendConfig::Registered`, carrying
    /// the caller's `options` through to the factory — the success path of the
    /// register-by-name dispatch.
    #[test]
    fn build_backend_config_routes_a_registered_name_to_registered() {
        // Register a throwaway backend (build_backend_config only builds the config,
        // it never invokes the factory, so the body is irrelevant here).
        crate::infrastructure::register_backend(
            "edge-registered-test",
            |_ctx: &crate::infrastructure::BackendBuildContext| {
                Err(crate::infrastructure::BackendError::Conversion(
                    "unused".to_string(),
                ))
            },
        );
        let opts = HashMap::from([
            ("endpoint".to_string(), OptionValue::from("tcp://x:1")),
            (
                "command".to_string(),
                OptionValue::from(vec!["python3".to_string(), "/a b/w.py".to_string()]),
            ),
        ]);
        let config = config_from_strings(
            "edge-registered-test",
            "aer",
            "automatic",
            None,
            1,
            2,
            None,
            None,
            Some(opts),
        )
        .expect("a registered name builds a Registered config");
        match config {
            BackendConfig::Registered { name, options } => {
                assert_eq!(name, "edge-registered-test");
                assert_eq!(
                    options.get("endpoint"),
                    Some(&OptionValue::from("tcp://x:1"))
                );
                // A list value reaches `Registered` intact (argument with a space
                // included).
                assert_eq!(
                    options.get("command"),
                    Some(&OptionValue::from(vec![
                        "python3".to_string(),
                        "/a b/w.py".to_string()
                    ]))
                );
            }
            other => panic!("expected BackendConfig::Registered, got {other:?}"),
        }
    }

    /// `options` on a typed built-in (`local`/`cunqa`) that would ignore it is
    /// rejected rather than silently dropped.
    #[test]
    fn build_backend_config_rejects_options_on_a_typed_builtin() {
        pyo3::Python::initialize();
        let str_opts = HashMap::from([("command".to_string(), OptionValue::from("x"))]);
        let list_opts = HashMap::from([(
            "command".to_string(),
            OptionValue::from(vec!["x".to_string()]),
        )]);
        for (infra, opts) in [
            ("local", &str_opts),
            ("cunqa", &str_opts),
            ("local", &list_opts),
            ("cunqa", &list_opts),
        ] {
            let err = config_from_strings(
                infra,
                "aer",
                "automatic",
                None,
                1,
                2,
                None,
                None,
                Some(opts.clone()),
            )
            .expect_err("options on a typed built-in must be rejected");
            Python::attach(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(
                    err.to_string().contains("does not accept 'options'"),
                    "unexpected message: {err}"
                );
            });
        }
        // An empty options dict is fine (nothing to ignore).
        assert!(config_from_strings(
            "local",
            "aer",
            "automatic",
            None,
            1,
            2,
            None,
            None,
            Some(HashMap::new()),
        )
        .is_ok());
    }

    /// The user's options are merged over the qmio defaults: a user value replaces the
    /// default for its key, the other defaults stay, and a list value is carried
    /// through untouched (the factory rejects it later if the key is a string one).
    #[cfg(feature = "qmio")]
    #[test]
    fn build_qmio_registered_merges_user_options_over_the_defaults() {
        let user = HashMap::from([
            ("optimization".to_string(), OptionValue::from("2")),
            (
                "res_format".to_string(),
                OptionValue::from(vec!["raw".to_string()]),
            ),
            (
                "custom".to_string(),
                OptionValue::from(vec!["a b".to_string()]),
            ),
        ]);
        let config = qmio_registered_config(
            ProgramFormat::QirText,
            Some("tcp://qmio.example:5556".to_string()),
            user,
        )
        .expect("qir is a qmio format");
        let BackendConfig::Registered { name, options } = config else {
            panic!("qmio must build a Registered config");
        };
        assert_eq!(name, "qmio");
        // Defaults the user did not override stay.
        assert_eq!(
            options.get("endpoint"),
            Some(&OptionValue::from("tcp://qmio.example:5556"))
        );
        assert_eq!(
            options.get("program_format"),
            Some(&OptionValue::from("qir_text"))
        );
        // User values win, whatever their shape.
        assert_eq!(options.get("optimization"), Some(&OptionValue::from("2")));
        assert_eq!(
            options.get("res_format"),
            Some(&OptionValue::from(vec!["raw".to_string()]))
        );
        assert_eq!(
            options.get("custom"),
            Some(&OptionValue::from(vec!["a b".to_string()]))
        );
        assert_eq!(options.len(), 5);
    }

    /// Without `ZMQ_SERVER` no endpoint is invented: the key is absent (the factory
    /// then rejects the config) unless the caller passes one.
    #[cfg(feature = "qmio")]
    #[test]
    fn qmio_registered_config_has_no_default_endpoint() {
        let BackendConfig::Registered { options, .. } =
            qmio_registered_config(ProgramFormat::OpenQasm, None, HashMap::new())
                .expect("qmio is a qmio format")
        else {
            panic!("qmio must build a Registered config");
        };
        assert_eq!(options.get("endpoint"), None);

        let user = HashMap::from([(
            "endpoint".to_string(),
            OptionValue::from("tcp://qmio.example:5556"),
        )]);
        let BackendConfig::Registered { options, .. } =
            qmio_registered_config(ProgramFormat::OpenQasm, None, user)
                .expect("qmio is a qmio format")
        else {
            panic!("qmio must build a Registered config");
        };
        assert_eq!(
            options.get("endpoint"),
            Some(&OptionValue::from("tcp://qmio.example:5556"))
        );
    }

    /// The argument checks shared by the entry points raise
    /// `polypus.ConfigurationError`.
    #[test]
    fn argument_checks_raise_configuration_error() {
        // Before the cases: without an interpreter the class cannot be looked up.
        Python::initialize();
        let cases: [(PyResult<()>, &str); 6] = [
            (validate_id(""), "id must not be empty"),
            (validate_id("a b"), "' '"),
            (validate_id(&"a".repeat(65)), "64"),
            (validate_cunqa_allocation("cunqa", 0, 1, None), "nodes"),
            (
                validate_cunqa_allocation("cunqa", 1, 0, None),
                "cores_per_qpu",
            ),
            (
                validate_cunqa_allocation("cunqa", 1, 1, Some("1-00:00:00")),
                "walltime",
            ),
        ];
        for (result, message) in cases {
            let err = result.expect_err("the argument is rejected");
            crate::exceptions::tests::assert_configuration_error(&err, message);
        }
    }
}
