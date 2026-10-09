//! A **runtime backend registry** — the pyo3-free mechanism that lets a backend
//! implemented outside the Polypus workspace be selected *by name* without editing
//! Polypus's own dispatch.
//!
//! # Why a registry
//!
//! Polypus's built-in backends are dispatched by a closed `match` in
//! `polypus-infrastructure` (`Infrastructure::create_backend`). Adding a backend
//! that way means editing that `match` — i.e. editing *our* repository. This
//! registry is the escape hatch: an embedder registers a [`BackendFactory`] under a
//! name, and any name-driven construction path (including Polypus's own factory,
//! which routes unknown names here) can then build it.
//!
//! # The two stories (see `docs/backends.md`)
//!
//! - **Rust embedder** (compiles their own binary embedding Polypus): calls
//!   [`register_backend`] explicitly at startup. This avoids touching our
//!   repository, but — being ordinary Rust — it does *not* avoid recompiling their
//!   binary. An explicit call (rather than link-time collection à la `inventory`/
//!   `linkme`) was chosen deliberately: no linker-`--gc-sections` surprises, no
//!   platform-specific static-collection magic, and it composes cleanly with
//!   `#[cfg(feature = …)]`-gated backends. The trade-off is the one ordering
//!   requirement: register before use (an omission surfaces as a clear
//!   [`BackendError::UnknownInfrastructure`], never a silent failure).
//! - **Python user** (`pip install polypus`): does not touch this registry
//!   directly. They select the built-in `"subprocess"` backend, which is registered
//!   here by `polypus-infrastructure` and bridges to a provider's Python SDK running
//!   in its own process (see the `polypus-subprocess-backend` crate).
//!
//! # Construction inputs
//!
//! A factory receives a [`BackendBuildContext`]: the pyo3-free, provider-agnostic
//! construction inputs (`id`, `shots`, `n_qpus`, `seed`, `opt_level`) plus an
//! `options` bag for whatever provider-specific configuration the backend needs
//! (an endpoint, a worker command, …). Each value is an [`OptionValue`]: a string or
//! a list of strings (an argv). Strings keep the registry — and every third-party
//! backend that depends on it — free of any provider type coupling; the list form
//! carries values a single string cannot, such as an argument containing spaces.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::error::BackendError;
use crate::transpiler::OptLevel;
use crate::QuantumBackend;

/// One value of a [`BackendBuildContext`]'s `options` bag: a string, or a list of
/// strings (e.g. a worker argv whose arguments contain spaces).
///
/// A factory reads values through [`BackendBuildContext::option_str`] and
/// [`BackendBuildContext::option_list`], which reject a value of the other shape
/// instead of silently ignoring it. The enum is `#[non_exhaustive]` so a further
/// value shape can be added without another breaking change; code outside this crate
/// that matches on it needs a wildcard arm.
///
/// ```
/// use polypus_backend::OptionValue;
///
/// assert_eq!(OptionValue::from("tcp://qpu:5556"), OptionValue::Str("tcp://qpu:5556".into()));
/// let argv = OptionValue::from(vec!["python3".to_string(), "/my dir/w.py".to_string()]);
/// assert!(matches!(argv, OptionValue::List(ref v) if v.len() == 2));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OptionValue {
    /// A single string value.
    Str(String),
    /// A list of string values, kept in order (an empty list is representable; a
    /// backend that needs a non-empty one rejects it).
    List(Vec<String>),
}

impl From<&str> for OptionValue {
    fn from(value: &str) -> Self {
        OptionValue::Str(value.to_string())
    }
}

impl From<String> for OptionValue {
    fn from(value: String) -> Self {
        OptionValue::Str(value)
    }
}

impl From<Vec<String>> for OptionValue {
    fn from(value: Vec<String>) -> Self {
        OptionValue::List(value)
    }
}

/// Provider-agnostic, **pyo3-free** construction inputs handed to a
/// [`BackendFactory`] when a backend is built by name.
///
/// This is the construction-time counterpart of the per-call
/// [`RunParams`](crate::RunParams): it carries the generic run parameters plus an
/// `options` bag for provider-specific configuration that would otherwise require a
/// typed config enum in Polypus's own crates. A registered backend reads only what
/// it needs and ignores the rest.
#[derive(Debug, Clone, Default)]
pub struct BackendBuildContext {
    /// Run identifier (logging, temp files, SLURM job names). Same validated string
    /// as [`RunParams::id`](crate::RunParams::id) when it comes from a Python entry
    /// point.
    pub id: String,
    /// Shots per circuit for this run.
    pub shots: u32,
    /// Number of execution units (QPUs) requested. `1` unless the caller asked for
    /// a distributed run; a backend that has no notion of replicas ignores it.
    pub n_qpus: u32,
    /// Explicit RNG seed, or `None` for the backend's own unseeded default.
    pub seed: Option<u64>,
    /// Transpiler optimization effort.
    pub opt_level: OptLevel,
    /// Provider-specific configuration: string keys, each mapped to a string or a
    /// list of strings ([`OptionValue`]). Documented per backend (e.g. the subprocess
    /// bridge reads `command`/`recv_timeout_ms`, QMIO reads
    /// `endpoint`/`program_format`/…). Unknown keys are ignored by design so a caller
    /// can sweep the same options across backends.
    pub options: HashMap<String, OptionValue>,
}

impl BackendBuildContext {
    /// Read a string option by key.
    ///
    /// `Ok(None)` when the key is absent (the only case a factory should fall back to
    /// its default), `Ok(Some(_))` when it holds a string, and
    /// [`BackendError::Conversion`] naming the key when it holds a list — a value of
    /// the wrong shape is a configuration mistake, never a silent default.
    ///
    /// ```
    /// use polypus_backend::{BackendBuildContext, BackendError, OptionValue};
    ///
    /// let mut ctx = BackendBuildContext::default();
    /// assert_eq!(ctx.option_str("endpoint").unwrap(), None);
    /// ctx.options.insert("endpoint".into(), OptionValue::from("tcp://qpu:5556"));
    /// assert_eq!(ctx.option_str("endpoint").unwrap(), Some("tcp://qpu:5556"));
    /// ctx.options.insert("endpoint".into(), OptionValue::from(vec!["a".to_string()]));
    /// assert!(matches!(ctx.option_str("endpoint"), Err(BackendError::Conversion(_))));
    /// ```
    pub fn option_str(&self, key: &str) -> Result<Option<&str>, BackendError> {
        match self.options.get(key) {
            None => Ok(None),
            Some(OptionValue::Str(s)) => Ok(Some(s)),
            Some(OptionValue::List(_)) => Err(BackendError::Conversion(format!(
                "option '{key}' must be a string, got a list of strings"
            ))),
        }
    }

    /// Read a list-of-strings option by key.
    ///
    /// The mirror of [`option_str`](Self::option_str): `Ok(None)` when the key is
    /// absent, `Ok(Some(_))` when it holds a list (possibly empty), and
    /// [`BackendError::Conversion`] naming the key when it holds a string.
    pub fn option_list(&self, key: &str) -> Result<Option<&[String]>, BackendError> {
        match self.options.get(key) {
            None => Ok(None),
            Some(OptionValue::List(v)) => Ok(Some(v)),
            Some(OptionValue::Str(_)) => Err(BackendError::Conversion(format!(
                "option '{key}' must be a list of strings, got a string"
            ))),
        }
    }
}

/// Builds a [`QuantumBackend`] from a [`BackendBuildContext`].
///
/// Implemented for any `Fn(&BackendBuildContext) -> Result<Arc<dyn QuantumBackend>,
/// BackendError>` that is `Send + Sync`, so the common case is a plain closure:
///
/// ```
/// use std::sync::Arc;
/// use polypus_backend::{register_backend, BackendBuildContext, BackendError, QuantumBackend};
/// # struct MyBackend;
/// # impl MyBackend { fn new(_: &BackendBuildContext) -> Result<Self, BackendError> { Ok(MyBackend) } }
/// # impl QuantumBackend for MyBackend {
/// #     fn run_circuits(&self, _: &[polypus_backend::BoundCircuit], _: &polypus_backend::RunParams)
/// #         -> Result<Vec<std::collections::HashMap<String, u64>>, BackendError> { Ok(vec![]) }
/// # }
/// register_backend("my_qpu", |ctx: &BackendBuildContext| {
///     Ok(Arc::new(MyBackend::new(ctx)?) as Arc<dyn QuantumBackend>)
/// });
/// ```
pub trait BackendFactory: Send + Sync {
    /// Build the backend, or report why it could not be built.
    fn create(&self, ctx: &BackendBuildContext) -> Result<Arc<dyn QuantumBackend>, BackendError>;
}

impl<F> BackendFactory for F
where
    F: Fn(&BackendBuildContext) -> Result<Arc<dyn QuantumBackend>, BackendError> + Send + Sync,
{
    fn create(&self, ctx: &BackendBuildContext) -> Result<Arc<dyn QuantumBackend>, BackendError> {
        self(ctx)
    }
}

/// The process-wide registry: backend name → factory. Created on first use.
fn registry() -> &'static RwLock<HashMap<String, Arc<dyn BackendFactory>>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, Arc<dyn BackendFactory>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register `factory` under `name`, so a later `create_registered_backend(name, …)`
/// (or Polypus's own factory routing an unknown infrastructure name here) builds it.
///
/// **Last registration wins:** re-registering a name replaces the previous factory
/// (logged at debug). This lets an embedder deliberately override a built-in — e.g.
/// swap Polypus's `"subprocess"` bridge for their own — and keeps the call
/// idempotent when a crate registers itself more than once.
///
/// Call this from a Rust embedder's startup, *before* the name is used. Being an
/// explicit call, it does not eliminate recompiling the embedder's binary — it
/// eliminates editing Polypus.
pub fn register_backend(name: impl Into<String>, factory: impl BackendFactory + 'static) {
    let name = name.into();
    let mut map = registry().write().unwrap_or_else(|p| p.into_inner());
    if map.insert(name.clone(), Arc::new(factory)).is_some() {
        log::debug!("backend registry: replaced existing factory for '{name}'");
    } else {
        log::debug!("backend registry: registered backend '{name}'");
    }
}

/// Whether a backend is registered under `name`.
pub fn is_registered(name: &str) -> bool {
    registry()
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .contains_key(name)
}

/// The names of all currently registered backends (diagnostic; unordered).
pub fn registered_names() -> Vec<String> {
    registry()
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .keys()
        .cloned()
        .collect()
}

/// Build the backend registered under `name` from `ctx`.
///
/// Returns [`BackendError::UnknownInfrastructure`] if no factory is registered under
/// that name — the same error the built-in factory raises for an unknown
/// infrastructure, so a caller need not special-case the registry path.
pub fn create_registered_backend(
    name: &str,
    ctx: &BackendBuildContext,
) -> Result<Arc<dyn QuantumBackend>, BackendError> {
    // Clone the `Arc<dyn BackendFactory>` out under the read lock, then build
    // *without* holding the lock: a factory may itself register backends (or take a
    // while), and we must not hold the registry lock across that.
    let factory = {
        let map = registry().read().unwrap_or_else(|p| p.into_inner());
        map.get(name).cloned()
    };
    match factory {
        Some(factory) => factory.create(ctx),
        None => Err(BackendError::UnknownInfrastructure {
            name: name.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BoundCircuit, Counts, RunParams};
    use std::collections::HashMap;

    /// A pure-Rust backend a third party could ship: it depends on nothing but
    /// `polypus-backend`, and reads its "width" from an option to prove the context
    /// reaches it.
    struct EchoBackend {
        width: usize,
    }

    impl QuantumBackend for EchoBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            params: &RunParams,
        ) -> Result<Vec<Counts>, BackendError> {
            Ok(qcs
                .iter()
                .map(|_| HashMap::from([("0".repeat(self.width), u64::from(params.shots))]))
                .collect())
        }
    }

    fn ctx_with(options: &[(&str, &str)]) -> BackendBuildContext {
        BackendBuildContext {
            id: "reg-test".to_string(),
            shots: 16,
            n_qpus: 1,
            seed: None,
            opt_level: OptLevel::default(),
            options: options
                .iter()
                .map(|(k, v)| (k.to_string(), OptionValue::from(*v)))
                .collect(),
        }
    }

    #[test]
    fn registers_and_builds_a_backend_by_name() {
        register_backend("echo-test-build", |ctx: &BackendBuildContext| {
            let width = ctx
                .option_str("width")?
                .and_then(|w| w.parse().ok())
                .unwrap_or(1);
            Ok(Arc::new(EchoBackend { width }) as Arc<dyn QuantumBackend>)
        });
        assert!(is_registered("echo-test-build"));

        let backend =
            create_registered_backend("echo-test-build", &ctx_with(&[("width", "3")])).unwrap();
        let out = backend
            .run_circuits(
                &[BoundCircuit::Qasm2(String::new())],
                &RunParams {
                    id: "reg-test".to_string(),
                    shots: 16,
                    seed: None,
                    opt_level: OptLevel::default(),
                },
            )
            .unwrap();
        assert_eq!(out[0].get("000"), Some(&16));
    }

    #[test]
    fn unknown_name_is_unknown_infrastructure() {
        // `Arc<dyn QuantumBackend>` is not `Debug`, so match on the `Result` rather
        // than `unwrap_err()`.
        match create_registered_backend("definitely-not-registered", &ctx_with(&[])) {
            Err(BackendError::UnknownInfrastructure { name }) => {
                assert_eq!(name, "definitely-not-registered");
            }
            Err(other) => panic!("expected UnknownInfrastructure, got {other:?}"),
            Ok(_) => panic!("expected an error for an unregistered name"),
        }
    }

    #[test]
    fn last_registration_wins() {
        register_backend("echo-test-override", |_: &BackendBuildContext| {
            Ok(Arc::new(EchoBackend { width: 1 }) as Arc<dyn QuantumBackend>)
        });
        register_backend("echo-test-override", |_: &BackendBuildContext| {
            Ok(Arc::new(EchoBackend { width: 2 }) as Arc<dyn QuantumBackend>)
        });
        let backend = create_registered_backend("echo-test-override", &ctx_with(&[])).unwrap();
        let out = backend
            .run_circuits(
                &[BoundCircuit::Qasm2(String::new())],
                &RunParams {
                    id: "reg-test".to_string(),
                    shots: 4,
                    seed: None,
                    opt_level: OptLevel::default(),
                },
            )
            .unwrap();
        // The second registration (width 2) is the one that built.
        assert_eq!(out[0].get("00"), Some(&4));
    }

    /// A context holding one string option and one list option.
    fn mixed_ctx() -> BackendBuildContext {
        let mut ctx = ctx_with(&[("endpoint", "tcp://qpu:1")]);
        ctx.options.insert(
            "command".to_string(),
            OptionValue::from(vec!["python3".to_string(), "/a b/w.py".to_string()]),
        );
        ctx.options
            .insert("empty".to_string(), OptionValue::from(Vec::new()));
        ctx
    }

    #[test]
    fn option_str_reads_absent_string_and_rejects_a_list() {
        let ctx = mixed_ctx();
        assert_eq!(ctx.option_str("missing").unwrap(), None);
        assert_eq!(ctx.option_str("endpoint").unwrap(), Some("tcp://qpu:1"));
        match ctx.option_str("command") {
            Err(BackendError::Conversion(m)) => {
                assert!(m.contains("'command'"), "message should name the key: {m}");
                assert!(m.contains("must be a string"), "unexpected message: {m}");
            }
            other => panic!("a list read as a string must be a Conversion error, got {other:?}"),
        }
    }

    #[test]
    fn option_list_reads_absent_list_and_rejects_a_string() {
        let ctx = mixed_ctx();
        assert_eq!(ctx.option_list("missing").unwrap(), None);
        assert_eq!(
            ctx.option_list("command").unwrap(),
            Some(&["python3".to_string(), "/a b/w.py".to_string()][..])
        );
        // An empty list is a list: the accessor returns it, the backend decides.
        assert_eq!(ctx.option_list("empty").unwrap(), Some(&[][..]));
        match ctx.option_list("endpoint") {
            Err(BackendError::Conversion(m)) => {
                assert!(m.contains("'endpoint'"), "message should name the key: {m}");
                assert!(
                    m.contains("must be a list of strings"),
                    "unexpected message: {m}"
                );
            }
            other => panic!("a string read as a list must be a Conversion error, got {other:?}"),
        }
    }

    #[test]
    fn option_value_conversions() {
        assert_eq!(OptionValue::from("a"), OptionValue::Str("a".to_string()));
        assert_eq!(
            OptionValue::from("a".to_string()),
            OptionValue::Str("a".to_string())
        );
        assert_eq!(
            OptionValue::from(vec!["a".to_string()]),
            OptionValue::List(vec!["a".to_string()])
        );
        assert_ne!(
            OptionValue::from("a"),
            OptionValue::from(vec!["a".to_string()])
        );
    }
}
