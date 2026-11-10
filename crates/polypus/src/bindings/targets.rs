//! Typed execution targets (#434).
//!
//! A [`Target`] says where a run executes: Aer, the native simulator, CUNQA,
//! QMIO or a registered backend, with the settings each one reads. Python
//! builds one either from a target class (`polypus.Aer`, `Native`, `Cunqa`,
//! `Qmio`, `Registered`), each validating its own fields, or from the
//! `infrastructure`/`backend` strings through [`target_from_strings`], whose
//! aliases all live in [`ALIASES`].

use std::collections::{BTreeMap, HashMap};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use super::{reject_aer_only_args, validate_cunqa_allocation, OptionsArg};
use crate::exceptions::configuration_error;
use crate::infrastructure::OptionValue;

/// Entry-point defaults of the string arguments. An argument that differs from
/// its default is "given", and conflicts with `target=`.
pub(crate) const DEFAULT_BACKEND: &str = "aer";
pub(crate) const DEFAULT_SIM_METHOD: &str = "automatic";
pub(crate) const DEFAULT_NODES: u32 = 1;
pub(crate) const DEFAULT_CORES_PER_QPU: u32 = 2;

/// The program representation submitted to the QMIO QPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProgramFormat {
    OpenQasm,
    QirText,
    QirBitcode,
}

impl ProgramFormat {
    /// The name `polypus.Qmio(program_format=...)` reports.
    pub(crate) fn public_name(self) -> &'static str {
        match self {
            Self::OpenQasm => "openqasm",
            Self::QirText => "qir",
            Self::QirBitcode => "qir_bitcode",
        }
    }

    /// The value of the QMIO backend's `program_format` option.
    #[cfg_attr(not(feature = "qmio"), allow(dead_code))]
    pub(crate) fn option_value(self) -> &'static str {
        match self {
            Self::OpenQasm => "openqasm",
            Self::QirText => "qir_text",
            Self::QirBitcode => "qir_bitcode",
        }
    }
}

/// What an alias names: a `backend` device, or a QMIO program format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scope {
    Device,
    ProgramFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Canonical {
    Aer,
    Native,
    Format(ProgramFormat),
}

/// Every string accepted for `backend`, once. On `local` a device selects Aer
/// or the native simulator; `cunqa` accepts the Aer devices; on `qmio` the
/// string is a program format, and the entry-point default `"aer"` means
/// OpenQASM (see [`qmio_program_format`]), not an alias.
const ALIASES: &[(Scope, &str, Canonical)] = &[
    (Scope::Device, "aer", Canonical::Aer),
    (Scope::Device, "AerSimulator", Canonical::Aer),
    (Scope::Device, "polypus", Canonical::Native),
    (Scope::Device, "statevector", Canonical::Native),
    (Scope::Device, "polypus_statevector", Canonical::Native),
    (
        Scope::ProgramFormat,
        "openqasm",
        Canonical::Format(ProgramFormat::OpenQasm),
    ),
    (
        Scope::ProgramFormat,
        "qasm",
        Canonical::Format(ProgramFormat::OpenQasm),
    ),
    (
        Scope::ProgramFormat,
        "qmio",
        Canonical::Format(ProgramFormat::OpenQasm),
    ),
    (
        Scope::ProgramFormat,
        "qir",
        Canonical::Format(ProgramFormat::QirText),
    ),
    (
        Scope::ProgramFormat,
        "qir_text",
        Canonical::Format(ProgramFormat::QirText),
    ),
    (
        Scope::ProgramFormat,
        "qir_bitcode",
        Canonical::Format(ProgramFormat::QirBitcode),
    ),
    (
        Scope::ProgramFormat,
        "qir_compiled",
        Canonical::Format(ProgramFormat::QirBitcode),
    ),
];

fn lookup(scope: Scope, alias: &str) -> Option<Canonical> {
    ALIASES
        .iter()
        .find(|(s, a, _)| *s == scope && *a == alias)
        .map(|(_, _, canonical)| *canonical)
}

/// Whether `backend` names the native statevector simulator.
pub(crate) fn is_native_device(backend: &str) -> bool {
    lookup(Scope::Device, backend) == Some(Canonical::Native)
}

fn program_format(name: &str) -> PyResult<ProgramFormat> {
    match lookup(Scope::ProgramFormat, name) {
        Some(Canonical::Format(format)) => Ok(format),
        _ => Err(configuration_error(format!(
            "unknown qmio program format '{name}'; expected \"openqasm\", \"qir\", or \"qir_bitcode\""
        ))),
    }
}

/// The program format a `backend` string selects on `qmio`; the default
/// `"aer"` stands for OpenQASM.
#[cfg_attr(not(feature = "qmio"), allow(dead_code))]
fn qmio_program_format(backend: &str) -> PyResult<ProgramFormat> {
    if backend == DEFAULT_BACKEND {
        Ok(ProgramFormat::OpenQasm)
    } else {
        program_format(backend)
    }
}

/// Where a run executes and the settings that backend reads.
pub(crate) enum Target {
    Aer {
        sim_method: String,
        noise_model: Option<Py<PyAny>>,
    },
    Native {
        fusion: Option<bool>,
    },
    Cunqa {
        nodes: u32,
        cores_per_qpu: u32,
        walltime: Option<String>,
        sim_method: String,
    },
    /// `options` comes from the entry point's `options=` (endpoint, …).
    Qmio {
        program_format: ProgramFormat,
        options: HashMap<String, OptionValue>,
    },
    Registered {
        name: String,
        options: HashMap<String, OptionValue>,
    },
}

impl Target {
    /// The `infrastructure` string the equivalent string call passes.
    pub(crate) fn infrastructure_label(&self) -> String {
        match self {
            Self::Aer { .. } | Self::Native { .. } => "local".to_owned(),
            Self::Cunqa { .. } => "cunqa".to_owned(),
            Self::Qmio { .. } => "qmio".to_owned(),
            Self::Registered { name, .. } => name.clone(),
        }
    }

    /// The `backend` string the equivalent string call passes (and the
    /// result's manifest echoes).
    pub(crate) fn backend_label(&self) -> String {
        match self {
            Self::Native { .. } => "polypus".to_owned(),
            Self::Qmio { program_format, .. } => program_format.public_name().to_owned(),
            Self::Aer { .. } | Self::Cunqa { .. } | Self::Registered { .. } => {
                DEFAULT_BACKEND.to_owned()
            }
        }
    }

    fn equals(&self, other: &Self, py: Python<'_>) -> PyResult<bool> {
        Ok(match (self, other) {
            (
                Self::Aer {
                    sim_method: a,
                    noise_model: na,
                },
                Self::Aer {
                    sim_method: b,
                    noise_model: nb,
                },
            ) => {
                a == b
                    && match (na, nb) {
                        (None, None) => true,
                        (Some(x), Some(y)) => x.is(y) || x.bind(py).eq(y.bind(py))?,
                        _ => false,
                    }
            }
            (Self::Native { fusion: a }, Self::Native { fusion: b }) => a == b,
            (
                Self::Cunqa {
                    nodes: a1,
                    cores_per_qpu: a2,
                    walltime: a3,
                    sim_method: a4,
                },
                Self::Cunqa {
                    nodes: b1,
                    cores_per_qpu: b2,
                    walltime: b3,
                    sim_method: b4,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4,
            (
                Self::Qmio {
                    program_format: a,
                    options: oa,
                },
                Self::Qmio {
                    program_format: b,
                    options: ob,
                },
            ) => a == b && oa == ob,
            (
                Self::Registered {
                    name: a,
                    options: oa,
                },
                Self::Registered {
                    name: b,
                    options: ob,
                },
            ) => a == b && oa == ob,
            _ => false,
        })
    }

    fn repr(&self, py: Python<'_>) -> PyResult<String> {
        let text = |s: &str| -> PyResult<String> { Ok(PyString::new(py, s).repr()?.to_string()) };
        let optional = |s: &Option<String>| -> PyResult<String> {
            s.as_deref().map_or_else(|| Ok("None".to_owned()), text)
        };
        Ok(match self {
            Self::Aer {
                sim_method,
                noise_model,
            } => format!(
                "Aer(sim_method={}, noise_model={})",
                text(sim_method)?,
                match noise_model {
                    Some(model) => model.bind(py).repr()?.to_string(),
                    None => "None".to_owned(),
                }
            ),
            Self::Native { fusion } => format!(
                "Native(fusion={})",
                fusion.map_or("None", |f| if f { "True" } else { "False" })
            ),
            Self::Cunqa {
                nodes,
                cores_per_qpu,
                walltime,
                sim_method,
            } => format!(
                "Cunqa(nodes={nodes}, cores_per_qpu={cores_per_qpu}, walltime={}, sim_method={})",
                optional(walltime)?,
                text(sim_method)?
            ),
            Self::Qmio { program_format, .. } => format!(
                "Qmio(program_format={})",
                text(program_format.public_name())?
            ),
            Self::Registered { name, options } => format!(
                "Registered({}, options={})",
                text(name)?,
                options_to_py(py, options)?.repr()?
            ),
        })
    }

    /// The Python target object for this target. `Qmio` drops `options`,
    /// which the entry point's `options=` carries.
    fn into_py(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Self::Aer {
                sim_method,
                noise_model,
            } => Py::new(
                py,
                Aer {
                    sim_method,
                    noise_model,
                },
            )?
            .into_any(),
            Self::Native { fusion } => Py::new(py, Native { fusion })?.into_any(),
            Self::Cunqa {
                nodes,
                cores_per_qpu,
                walltime,
                sim_method,
            } => Py::new(
                py,
                Cunqa {
                    nodes,
                    cores_per_qpu,
                    walltime,
                    sim_method,
                },
            )?
            .into_any(),
            Self::Qmio { program_format, .. } => Py::new(py, Qmio { program_format })?.into_any(),
            Self::Registered { name, options } => {
                Py::new(py, Registered { name, options })?.into_any()
            }
        })
    }
}

/// `options` as a `dict`, keys sorted so its repr is stable.
fn options_to_py<'py>(
    py: Python<'py>,
    options: &HashMap<String, OptionValue>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in options.iter().collect::<BTreeMap<_, _>>() {
        match value {
            OptionValue::Str(text) => dict.set_item(key, text)?,
            OptionValue::List(items) => dict.set_item(key, PyList::new(py, items)?)?,
            _ => dict.set_item(key, py.None())?,
        }
    }
    Ok(dict)
}

fn options_not_accepted(infrastructure: &str, count: usize) -> PyErr {
    configuration_error(format!(
        "the '{infrastructure}' infrastructure does not accept 'options' ({count} given); \
         the 'options' dict applies only to registry-backed backends (\"qmio\", \
         \"subprocess\", or a registered third-party backend)"
    ))
}

fn unknown_infrastructure(name: &str) -> PyErr {
    configuration_error(format!(
        "unknown infrastructure '{name}'; expected \"local\", \"cunqa\", \"qmio\", \
         or a registered backend name (see docs/backends.md)"
    ))
}

/// Map the `infrastructure` + `backend` strings and provider arguments to a
/// [`Target`], with the checks (and their order) the entry points always had.
///
/// `backend` selects the *device* within an infrastructure: on `local`, Aer or
/// the native simulator. CUNQA's QPUs always run Aer, with the simulator fixed
/// when they are raised, so `cunqa` accepts only an Aer device and rejects a
/// `noise_model`: both would otherwise be dropped silently. For the same
/// reason the native backend, `qmio` and registry-dispatched backends reject a
/// `noise_model` and a non-default `sim_method`, and registry backends a
/// non-default `backend`. On `qmio`, `backend` is the program format.
///
/// `fusion` is an `Option` so three cases stay distinct: omitted (`None`, each
/// backend does its own thing), `Some(false)` ("do not fuse", which every
/// backend honours) and `Some(true)` ("fuse", which only the native backend
/// can, so anywhere else it is rejected rather than silently ignored).
///
/// `options` is consumed only by registry-dispatched backends; a non-empty one
/// on `local`/`cunqa` is rejected. The CUNQA allocation is checked separately,
/// by `validate_cunqa_allocation`.
pub(crate) fn target_from_strings(
    infrastructure: &str,
    backend: &str,
    sim_method: &str,
    noise_model: Option<Py<PyAny>>,
    nodes: u32,
    cores_per_qpu: u32,
    walltime: Option<&str>,
    fusion: Option<bool>,
    options: Option<HashMap<String, OptionValue>>,
) -> PyResult<Target> {
    let is_native = infrastructure == "local" && is_native_device(backend);
    if fusion == Some(true) && !is_native {
        return Err(configuration_error(format!(
            "fusion=True applies only to the native backend=\"polypus\"; backend \"{backend}\" \
             on infrastructure \"{infrastructure}\" cannot fuse gates. Omit fusion, or pass \
             fusion=False for a gate-by-gate run."
        )));
    }
    if matches!(infrastructure, "local" | "cunqa") {
        if let Some(opts) = &options {
            if !opts.is_empty() {
                return Err(options_not_accepted(infrastructure, opts.len()));
            }
        }
    }
    if infrastructure == "cunqa" {
        if lookup(Scope::Device, backend) != Some(Canonical::Aer) {
            return Err(configuration_error(format!(
                "unknown cunqa backend '{backend}'; CUNQA's QPUs run Aer, expected \"aer\""
            )));
        }
        if noise_model.is_some() {
            return Err(configuration_error(
                "the 'cunqa' infrastructure does not accept a noise_model: its simulated QPUs \
                 are configured when they are raised, not per run; omit it, or use \
                 infrastructure=\"local\" with backend=\"aer\"",
            ));
        }
    }
    match infrastructure {
        "local" => match lookup(Scope::Device, backend) {
            Some(Canonical::Aer) => Ok(Target::Aer {
                sim_method: sim_method.to_owned(),
                noise_model,
            }),
            Some(Canonical::Native) => {
                if noise_model.is_some() {
                    return Err(configuration_error(
                        "the native 'polypus' backend is a noiseless statevector simulator \
                         and does not accept a noise_model; use backend=\"aer\"",
                    ));
                }
                // The native simulator has no simulation method to choose, so
                // a non-default `sim_method` would be dropped silently (#334).
                if sim_method != DEFAULT_SIM_METHOD {
                    return Err(configuration_error(format!(
                        "the native 'polypus' backend is a statevector simulator and does \
                         not accept sim_method=\"{sim_method}\"; omit it, or use \
                         backend=\"aer\""
                    )));
                }
                Ok(Target::Native { fusion })
            }
            _ => Err(configuration_error(format!(
                "unknown local backend '{backend}'; expected \"aer\" or \"polypus\""
            ))),
        },
        "cunqa" => Ok(Target::Cunqa {
            nodes,
            cores_per_qpu,
            walltime: walltime.map(str::to_owned),
            sim_method: sim_method.to_owned(),
        }),
        "qmio" => {
            #[cfg(not(feature = "qmio"))]
            {
                Err(configuration_error(super::QMIO_NOT_COMPILED))
            }
            #[cfg(feature = "qmio")]
            {
                let program_format = qmio_program_format(backend)?;
                reject_aer_only_args(infrastructure, sim_method, noise_model.as_ref())?;
                Ok(Target::Qmio {
                    program_format,
                    options: options.unwrap_or_default(),
                })
            }
        }
        // Any other name is dispatched through the runtime backend registry:
        // the built-in `"subprocess"` bridge, or a backend a Rust embedder
        // registered before this call.
        other => {
            crate::infrastructure::register_builtin_backends();
            if !crate::infrastructure::is_registered(other) {
                return Err(unknown_infrastructure(other));
            }
            reject_aer_only_args(other, sim_method, noise_model.as_ref())?;
            // `Registered` carries only `options`, so `backend` would be
            // dropped while `RunResult.backend` echoed it.
            if backend != DEFAULT_BACKEND {
                return Err(configuration_error(format!(
                    "the registered backend '{other}' does not accept backend=\"{backend}\"; \
                     omit it and pass backend-specific settings through 'options'"
                )));
            }
            Ok(Target::Registered {
                name: other.to_owned(),
                options: options.unwrap_or_default(),
            })
        }
    }
}

/// `target` with the entry point's `options=`: QMIO takes them; a registered
/// target takes them unless it already has its own; the typed built-ins reject
/// a non-empty `options` as the string path does.
fn with_options(target: Target, options: Option<HashMap<String, OptionValue>>) -> PyResult<Target> {
    let Some(options) = options.filter(|o| !o.is_empty()) else {
        return Ok(target);
    };
    match target {
        Target::Qmio { program_format, .. } => Ok(Target::Qmio {
            program_format,
            options,
        }),
        Target::Registered { name, options: own } => {
            if own.is_empty() {
                Ok(Target::Registered { name, options })
            } else {
                Err(configuration_error(format!(
                    "options were given both in polypus.Registered('{name}', options=...) and \
                     as options=; pass them once"
                )))
            }
        }
        builtin => Err(options_not_accepted(
            &builtin.infrastructure_label(),
            options.len(),
        )),
    }
}

/// A `TypeError` like the one Python raises for a missing argument.
pub(crate) fn require_arg<T>(func: &str, name: &str, value: Option<T>) -> PyResult<T> {
    value.ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{func}() missing 1 required positional argument: '{name}'"
        ))
    })
}

/// The target an entry point runs on, chosen by `target=` or by the strings.
pub(crate) struct Selection {
    /// The infrastructure and backend strings of the manifest: the caller's on
    /// the string path, those of the equivalent string call with `target=`.
    pub(crate) infrastructure: String,
    pub(crate) backend: String,
    typed: Option<Target>,
}

impl Selection {
    /// `given` lists each other exclusive argument, in signature order, and
    /// whether the caller set it to something other than its default.
    pub(crate) fn new(
        func: &str,
        target: Option<&Bound<'_, PyAny>>,
        infrastructure: Option<String>,
        backend: &str,
        given: &[(&str, bool)],
    ) -> PyResult<Self> {
        let Some(target) = target else {
            return Ok(Self {
                infrastructure: require_arg(func, "infrastructure", infrastructure)?,
                backend: backend.to_owned(),
                typed: None,
            });
        };
        let target = extract_target(target)?;
        let conflicts: Vec<&str> = std::iter::once(("infrastructure", infrastructure.is_some()))
            .chain(given.iter().copied())
            .filter_map(|(name, set)| set.then_some(name))
            .collect();
        if !conflicts.is_empty() {
            return Err(configuration_error(format!(
                "target= cannot be combined with {}; the target already selects the \
                 infrastructure, the backend and their settings",
                conflicts.join(", ")
            )));
        }
        Ok(Self {
            infrastructure: target.infrastructure_label(),
            backend: target.backend_label(),
            typed: Some(target),
        })
    }

    pub(crate) fn is_typed(&self) -> bool {
        self.typed.is_some()
    }

    /// Whether the run goes to the native simulator. On the string path this
    /// reads `backend` alone, whatever the infrastructure, as the native-only
    /// circuit checks always did.
    pub(crate) fn is_native(&self) -> bool {
        match &self.typed {
            Some(target) => matches!(target, Target::Native { .. }),
            None => is_native_device(&self.backend),
        }
    }

    /// The final target: the typed one with `options` added, or `strings`
    /// applied to them.
    pub(crate) fn resolve(
        self,
        options: Option<HashMap<String, OptionValue>>,
        strings: impl FnOnce(Option<HashMap<String, OptionValue>>) -> PyResult<Target>,
    ) -> PyResult<Target> {
        match self.typed {
            Some(target) => with_options(target, options),
            None => strings(options),
        }
    }
}

/// Read a `target=` argument.
pub(crate) fn extract_target(obj: &Bound<'_, PyAny>) -> PyResult<Target> {
    let py = obj.py();
    if let Ok(t) = obj.cast::<Aer>() {
        let t = t.get();
        return Ok(Target::Aer {
            sim_method: t.sim_method.clone(),
            noise_model: t.noise_model.as_ref().map(|nm| nm.clone_ref(py)),
        });
    }
    if let Ok(t) = obj.cast::<Native>() {
        return Ok(Target::Native {
            fusion: t.get().fusion,
        });
    }
    if let Ok(t) = obj.cast::<Cunqa>() {
        let t = t.get();
        return Ok(Target::Cunqa {
            nodes: t.nodes,
            cores_per_qpu: t.cores_per_qpu,
            walltime: t.walltime.clone(),
            sim_method: t.sim_method.clone(),
        });
    }
    if let Ok(t) = obj.cast::<Qmio>() {
        return Ok(Target::Qmio {
            program_format: t.get().program_format,
            options: HashMap::new(),
        });
    }
    if let Ok(t) = obj.cast::<Registered>() {
        let t = t.get();
        return Ok(Target::Registered {
            name: t.name.clone(),
            options: t.options.clone(),
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "target must be polypus.Aer, polypus.Native, polypus.Cunqa, polypus.Qmio or \
         polypus.Registered, got {}",
        obj.get_type().name()?
    )))
}

fn target_eq(this: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    match extract_target(other) {
        Ok(other) => extract_target(this)?.equals(&other, this.py()),
        Err(_) => Ok(false),
    }
}

fn target_repr(this: &Bound<'_, PyAny>) -> PyResult<String> {
    extract_target(this)?.repr(this.py())
}

/// Qiskit Aer on this machine (`infrastructure="local"`, `backend="aer"`).
#[pyclass(module = "polypus", frozen)]
pub struct Aer {
    sim_method: String,
    noise_model: Option<Py<PyAny>>,
}

#[pymethods]
impl Aer {
    #[new]
    #[pyo3(signature = (sim_method="automatic", noise_model=None))]
    fn new(sim_method: &str, noise_model: Option<Py<PyAny>>) -> Self {
        Self {
            sim_method: sim_method.to_owned(),
            noise_model,
        }
    }

    #[getter]
    fn sim_method(&self) -> &str {
        &self.sim_method
    }

    #[getter]
    fn noise_model(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.noise_model.as_ref().map(|nm| nm.clone_ref(py))
    }

    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        target_eq(slf.as_any(), other)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        target_repr(slf.as_any())
    }
}

/// The native statevector simulator (`infrastructure="local"`,
/// `backend="polypus"`).
#[pyclass(module = "polypus", frozen)]
pub struct Native {
    #[pyo3(get)]
    fusion: Option<bool>,
}

#[pymethods]
impl Native {
    #[new]
    #[pyo3(signature = (fusion=None))]
    fn new(fusion: Option<bool>) -> Self {
        Self { fusion }
    }

    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        target_eq(slf.as_any(), other)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        target_repr(slf.as_any())
    }
}

/// CUNQA's simulated QPUs, raised through SLURM (`infrastructure="cunqa"`).
#[pyclass(module = "polypus", frozen)]
pub struct Cunqa {
    #[pyo3(get)]
    nodes: u32,
    #[pyo3(get)]
    cores_per_qpu: u32,
    #[pyo3(get)]
    walltime: Option<String>,
    #[pyo3(get)]
    sim_method: String,
}

#[pymethods]
impl Cunqa {
    #[new]
    #[pyo3(signature = (nodes=1, cores_per_qpu=2, walltime=None, sim_method="automatic"))]
    fn new(
        nodes: u32,
        cores_per_qpu: u32,
        walltime: Option<String>,
        sim_method: &str,
    ) -> PyResult<Self> {
        validate_cunqa_allocation("cunqa", nodes, cores_per_qpu, walltime.as_deref())?;
        Ok(Self {
            nodes,
            cores_per_qpu,
            walltime,
            sim_method: sim_method.to_owned(),
        })
    }

    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        target_eq(slf.as_any(), other)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        target_repr(slf.as_any())
    }
}

/// The CESGA QMIO QPU (`infrastructure="qmio"`); `program_format` is what
/// `backend` selected there.
#[pyclass(module = "polypus", frozen)]
pub struct Qmio {
    program_format: ProgramFormat,
}

#[pymethods]
impl Qmio {
    #[new]
    #[pyo3(signature = (program_format="openqasm"))]
    fn new(program_format: &str) -> PyResult<Self> {
        Ok(Self {
            program_format: self::program_format(program_format)?,
        })
    }

    #[getter]
    fn program_format(&self) -> &'static str {
        self.program_format.public_name()
    }

    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        target_eq(slf.as_any(), other)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        target_repr(slf.as_any())
    }
}

/// A backend from the runtime registry (the built-in `"subprocess"` bridge or
/// one a Rust embedder registered), configured by `options`.
#[pyclass(module = "polypus", frozen)]
pub struct Registered {
    #[pyo3(get)]
    name: String,
    options: HashMap<String, OptionValue>,
}

#[pymethods]
impl Registered {
    #[new]
    #[pyo3(signature = (name, options=None))]
    fn new(name: String, options: Option<OptionsArg>) -> PyResult<Self> {
        let typed = match name.as_str() {
            "" => return Err(configuration_error("Registered name must not be empty")),
            "local" => Some("polypus.Aer or polypus.Native"),
            "cunqa" => Some("polypus.Cunqa"),
            "qmio" => Some("polypus.Qmio"),
            _ => None,
        };
        if let Some(class) = typed {
            return Err(configuration_error(format!(
                "'{name}' is a built-in infrastructure, not a registered backend; use {class}"
            )));
        }
        crate::infrastructure::register_builtin_backends();
        if !crate::infrastructure::is_registered(&name) {
            return Err(unknown_infrastructure(&name));
        }
        Ok(Self {
            name,
            options: options.map(OptionsArg::into_inner).unwrap_or_default(),
        })
    }

    #[getter]
    fn options<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        options_to_py(py, &self.options)
    }

    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        target_eq(slf.as_any(), other)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        target_repr(slf.as_any())
    }
}

/// Test helper: the target the strings select, or the error the entry points
/// raise for them.
#[pyfunction(name = "_target_of", signature = (infrastructure, backend="aer", sim_method="automatic", noise_model=None, nodes=1, cores_per_qpu=2, walltime=None, fusion=None, options=None))]
fn target_of(
    py: Python<'_>,
    infrastructure: &str,
    backend: &str,
    sim_method: &str,
    noise_model: Option<Py<PyAny>>,
    nodes: u32,
    cores_per_qpu: u32,
    walltime: Option<String>,
    fusion: Option<bool>,
    options: Option<OptionsArg>,
) -> PyResult<Py<PyAny>> {
    validate_cunqa_allocation(infrastructure, nodes, cores_per_qpu, walltime.as_deref())?;
    target_from_strings(
        infrastructure,
        backend,
        sim_method,
        noise_model,
        nodes,
        cores_per_qpu,
        walltime.as_deref(),
        fusion,
        options.map(OptionsArg::into_inner),
    )?
    .into_py(py)
}

/// Add the target classes, and `_target_of` outside `__all__`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Aer>()?;
    m.add_class::<Native>()?;
    m.add_class::<Cunqa>()?;
    m.add_class::<Qmio>()?;
    m.add_class::<Registered>()?;
    m.setattr("_target_of", wrap_pyfunction!(target_of, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_have_no_duplicate_keys() {
        for (i, (scope, alias, _)) in ALIASES.iter().enumerate() {
            assert!(
                !ALIASES[i + 1..]
                    .iter()
                    .any(|(s, a, _)| s == scope && a == alias),
                "{alias:?} appears twice in {scope:?}"
            );
        }
    }

    #[test]
    fn every_alias_resolves_to_its_canonical_target() {
        for (scope, alias, canonical) in ALIASES {
            assert_eq!(lookup(*scope, alias), Some(*canonical));
        }
        assert!(is_native_device("statevector"));
        assert!(!is_native_device("AerSimulator"));
        assert_eq!(lookup(Scope::Device, "openqasm"), None);
        assert_eq!(lookup(Scope::ProgramFormat, "aer"), None);
    }

    #[test]
    fn the_qmio_default_backend_means_openqasm() {
        assert_eq!(
            qmio_program_format(DEFAULT_BACKEND).unwrap(),
            ProgramFormat::OpenQasm
        );
        assert_eq!(
            qmio_program_format("qir_compiled").unwrap(),
            ProgramFormat::QirBitcode
        );
        assert_eq!(ProgramFormat::QirText.option_value(), "qir_text");
        assert_eq!(ProgramFormat::QirText.public_name(), "qir");
    }

    #[test]
    fn labels_match_the_equivalent_string_call() {
        let cases = [
            (
                Target::Aer {
                    sim_method: "automatic".into(),
                    noise_model: None,
                },
                ("local", "aer"),
            ),
            (Target::Native { fusion: None }, ("local", "polypus")),
            (
                Target::Qmio {
                    program_format: ProgramFormat::QirBitcode,
                    options: HashMap::new(),
                },
                ("qmio", "qir_bitcode"),
            ),
            (
                Target::Registered {
                    name: "subprocess".into(),
                    options: HashMap::new(),
                },
                ("subprocess", "aer"),
            ),
        ];
        for (target, (infrastructure, backend)) in cases {
            assert_eq!(target.infrastructure_label(), infrastructure);
            assert_eq!(target.backend_label(), backend);
        }
    }
}
