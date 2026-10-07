//! The acceptance test for the native template: it passes the Polypus conformance
//! battery. An external author running this against their own filled-in backend is
//! exactly the "done when" of the templates+conformance phase.

use std::sync::Arc;

use example_rust_backend::{register, TemplateBackend, BACKEND_NAME};
use polypus_backend::{
    is_registered, BackendError, BoundCircuit, OptLevel, QuantumBackend, RunParams,
};
use polypus_backend_conformance::{Conformance, Fault};

/// Build a backend at `endpoint`, boxed for the battery's factory closures.
fn at(endpoint: &'static str) -> impl Fn() -> Result<Arc<dyn QuantumBackend>, String> {
    move || {
        TemplateBackend::connect(endpoint)
            .map(|b| Arc::new(b) as Arc<dyn QuantumBackend>)
            .map_err(|e| e.to_string())
    }
}

#[test]
fn the_native_template_passes_the_conformance_battery() {
    let report = Conformance::new(BACKEND_NAME, at("sim://loopback"))
        .shots(1024)
        // The template can be driven into every documented fault, so nothing skips:
        .fault(Fault::Unresponsive, at("sim://unreachable"))
        .fault(Fault::CleanError, at("sim://reject"))
        .fault(Fault::Aborted, at("sim://slow"))
        .run();

    println!("{report}");
    report.assert_conformant();
    // Every error-classification check is exercised (the template can reach each
    // fault). Only `declared_finite_concurrency_wave_runs` skips, because this
    // template declares unbounded concurrency — a legitimate not-applicable.
    for name in [
        "unresponsive_backend_is_classified",
        "clean_provider_error_is_classified",
        "interrupted_call_is_aborted",
        "foreign_circuit_rejected_as_unsupported",
    ] {
        let check = report.checks.iter().find(|c| c.name == name).unwrap();
        assert!(
            matches!(check.status, polypus_backend_conformance::Status::Passed),
            "{name} should pass, got {:?}",
            check.status
        );
    }
}

#[test]
fn it_registers_under_its_name() {
    register();
    assert!(is_registered(BACKEND_NAME));
}

#[test]
fn a_circuit_openqasm2_cannot_express_is_unsupported() {
    let circuit = polypus_circuit::ParameterizedCircuit::from_qasm3(
        "OPENQASM 3.0;\ninclude \"stdgates.inc\";\ngate g(t) a { rx(arcsin(t)) a; }\nqubit[1] q;\ng(0.5) q[0];\n",
    )
    .unwrap()
    .assign_parameters(&[])
    .unwrap();
    let params = RunParams {
        id: "qasm2-unsupported".to_string(),
        shots: 16,
        seed: None,
        opt_level: OptLevel::default(),
    };
    let backend = TemplateBackend::connect("sim://loopback").unwrap();
    match backend.run_circuits(&[BoundCircuit::Native(circuit)], &params) {
        Err(BackendError::UnsupportedCircuit(message)) => {
            assert!(message.contains("arcsin"), "{message}")
        }
        other => panic!("expected UnsupportedCircuit, got {other:?}"),
    }
}
