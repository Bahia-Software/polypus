//! Acceptance test for the Fase-4 Rust story: a backend that lives **outside the
//! Polypus workspace** can be registered at runtime and then built *by name* through
//! the very same factory Polypus's own backends go through — **without editing any
//! `match` in the workspace**.
//!
//! The backend here depends only on the pyo3-free contract (`QuantumBackend`,
//! `BoundCircuit`, …), exactly as a third party's would; `register_backend` +
//! `Infrastructure::create_backend` do the rest.

use std::collections::HashMap;
use std::sync::Arc;

use polypus_infrastructure::{
    register_backend, BackendBuildContext, BackendConfig, BackendError, BoundCircuit, Counts,
    ExecutionConfig, Infrastructure, OptLevel, QuantumBackend, RunParams,
};

/// A pure-Rust "third-party" backend: returns all shots on the all-zeros bitstring,
/// with a width taken from a registration option (to prove config reaches it).
struct ExternalQpu {
    width: usize,
}

impl QuantumBackend for ExternalQpu {
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

fn execution_config(name: &str, options: HashMap<String, String>) -> ExecutionConfig {
    ExecutionConfig {
        id: "third-party-run".to_string(),
        shots: 256,
        n_qpus: 1,
        infrastructure: name.to_string(),
        backend_config: BackendConfig::Registered {
            name: name.to_string(),
            options,
        },
        opt_level: OptLevel::default(),
        seed: None,
    }
}

#[test]
fn a_backend_outside_the_workspace_is_registered_and_built_by_name() {
    // 1. The third party registers their backend once, at startup. No Polypus source
    //    is edited to do this.
    register_backend("acme-qpu", |ctx: &BackendBuildContext| {
        let width = ctx
            .option("width")
            .and_then(|w| w.parse().ok())
            .unwrap_or(1);
        Ok(Arc::new(ExternalQpu { width }) as Arc<dyn QuantumBackend>)
    });

    // 2. Polypus's own factory — the same one that builds Local/CUNQA/QMIO — dispatches
    //    it by name through the registry.
    let options = HashMap::from([("width".to_string(), "4".to_string())]);
    let backend = Infrastructure::create_backend(&execution_config("acme-qpu", options))
        .expect("the registered third-party backend is built by name");

    // 3. And it runs like any built-in backend.
    let out = backend
        .run_circuits(
            &[BoundCircuit::Qasm2(String::new())],
            &RunParams {
                id: "third-party-run".to_string(),
                shots: 256,
                seed: None,
                opt_level: OptLevel::default(),
            },
        )
        .expect("the third-party backend runs");
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].get("0000"),
        Some(&256),
        "width option reached the factory"
    );
}

#[test]
fn an_unregistered_name_is_a_clear_error() {
    // A name nobody registered is a clean ValueError-shaped error at the edge, never
    // a panic or a silent fallthrough.
    let result =
        Infrastructure::create_backend(&execution_config("nobody-registered-this", HashMap::new()));
    match result {
        Err(BackendError::UnknownInfrastructure { name }) => {
            assert_eq!(name, "nobody-registered-this");
        }
        Err(other) => panic!("expected UnknownInfrastructure, got {other:?}"),
        Ok(_) => panic!("an unregistered name must not build a backend"),
    }
}

/// The built-in `"subprocess"` bridge is registered by `create_backend` (via
/// `register_builtin_backends`) without anyone calling it explicitly — proving the
/// Python story's backend is discoverable by name out of the box.
#[test]
fn the_subprocess_bridge_is_a_registered_builtin() {
    // Building it will fail (no `command` option here), but the failure is the
    // factory's *own* Conversion error — which proves the name resolved to the
    // subprocess factory rather than falling through as unknown.
    let result = Infrastructure::create_backend(&execution_config("subprocess", HashMap::new()));
    match result {
        Err(BackendError::Conversion(msg)) => {
            assert!(msg.contains("command"), "unexpected message: {msg}");
        }
        Err(BackendError::UnknownInfrastructure { .. }) => {
            panic!("the 'subprocess' bridge must be a registered builtin, not unknown")
        }
        Err(other) => panic!("expected the subprocess factory's Conversion error, got {other:?}"),
        Ok(_) => panic!("building the subprocess bridge with no 'command' must fail"),
    }
}
