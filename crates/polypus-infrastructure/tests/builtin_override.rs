//! An embedder who registers their own factory under a built-in name (e.g.
//! `"subprocess"`) **before** the first `create_backend` must keep it: Polypus's
//! `register_builtin_backends` fills in only names that are still free, so it never
//! clobbers an override.
//!
//! This lives in its own test binary (its own process) because it mutates the
//! process-global registry for the `"subprocess"` name, which would otherwise race
//! the assertions in `third_party_registration.rs`.

use std::collections::HashMap;
use std::sync::Arc;

use polypus_infrastructure::{
    register_backend, BackendBuildContext, BackendError, BoundCircuit, Counts, ExecutionConfig,
    Infrastructure, OptLevel, QuantumBackend, RunParams,
};

/// A stand-in "subprocess" the embedder prefers: it needs no worker `command`
/// (the real bridge would error without one), returning a fixed counts map.
struct MySubprocess;
impl QuantumBackend for MySubprocess {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        params: &RunParams,
    ) -> Result<Vec<Counts>, BackendError> {
        Ok(qcs
            .iter()
            .map(|_| HashMap::from([("0".to_string(), u64::from(params.shots))]))
            .collect())
    }
}

#[test]
fn an_embedder_override_of_a_builtin_name_is_not_clobbered() {
    // Register BEFORE any create_backend triggers register_builtin_backends.
    register_backend("subprocess", |_ctx: &BackendBuildContext| {
        Ok(Arc::new(MySubprocess) as Arc<dyn QuantumBackend>)
    });

    let config = ExecutionConfig {
        id: "override-run".to_string(),
        shots: 32,
        n_qpus: 1,
        infrastructure: "subprocess".to_string(),
        backend_config: polypus_infrastructure::BackendConfig::Registered {
            name: "subprocess".to_string(),
            // No `command`: the real bridge factory would fail here. Our override
            // ignores options and succeeds, which is how we tell them apart.
            options: HashMap::new(),
        },
        opt_level: OptLevel::default(),
        seed: None,
    };

    // create_backend runs register_builtin_backends internally; the override must
    // survive it.
    let backend = Infrastructure::create_backend(&config)
        .expect("the embedder's own 'subprocess' backend must still build (not the bridge)");
    let out = backend
        .run_circuits(&[BoundCircuit::Qasm2(String::new())], &config.run_params())
        .expect("the override backend runs");
    assert_eq!(
        out[0].get("0"),
        Some(&32),
        "the override backend produced the counts"
    );
}
