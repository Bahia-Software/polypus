//! The native backend's structural pre-check logs nothing (issue #218).
//!
//! `NativeStatevectorBackend::check_circuit` backs `polypus.backend_compatibility`,
//! which is documented as side-effect free: a rejected circuit is the expected
//! answer to the query, not an error, so it must not write an `ERROR` record.
//! Running the same circuit still logs its failure. `log::set_logger` is global
//! per process, so this lives in its own test binary with a single test.

use std::sync::Mutex;

use log::{Level, Log, Metadata, Record};
use polypus_infrastructure::{
    BackendConfig, BoundCircuit, ExecutionConfig, NativeStatevectorBackend, OptLevel,
    QuantumBackend,
};

/// Records every `(level, message)` it receives.
struct Capture(Mutex<Vec<(Level, String)>>);

impl Log for Capture {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &Record<'_>) {
        self.0
            .lock()
            .expect("the capture lock is never poisoned")
            .push((record.level(), record.args().to_string()));
    }

    fn flush(&self) {}
}

static CAPTURE: Capture = Capture(Mutex::new(Vec::new()));

/// Drain the records captured so far.
fn take() -> Vec<(Level, String)> {
    std::mem::take(
        &mut *CAPTURE
            .0
            .lock()
            .expect("the capture lock is never poisoned"),
    )
}

#[test]
fn check_circuit_logs_nothing_while_execution_logs_the_rejection() {
    log::set_logger(&CAPTURE).expect("the only logger this binary installs");
    log::set_max_level(log::LevelFilter::Trace);

    let rejected = BoundCircuit::Qasm2(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ncreg c[1];\n\
         reset q[0];\nmeasure q[0] -> c[0];\n"
            .to_string(),
    );
    let accepted = BoundCircuit::Qasm2(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ncreg c[1];\n\
         x q[0];\nmeasure q[0] -> c[0];\n"
            .to_string(),
    );

    take();
    assert!(NativeStatevectorBackend::check_circuit(&rejected).is_err());
    assert!(NativeStatevectorBackend::check_circuit(&accepted).is_ok());
    assert_eq!(take(), Vec::new(), "the pre-check must not log");

    let config = ExecutionConfig {
        id: "precheck-logging".to_string(),
        shots: 8,
        n_qpus: 1,
        infrastructure: "local".to_string(),
        backend_config: BackendConfig::LocalNative { fusion: true },
        opt_level: OptLevel::default(),
        seed: Some(1),
    };
    let run = NativeStatevectorBackend::new(1)
        .run_circuits(std::slice::from_ref(&rejected), &config.run_params());
    assert!(run.is_err());
    let errors: Vec<String> = take()
        .into_iter()
        .filter(|(level, _)| *level == Level::Error)
        .map(|(_, message)| message)
        .collect();
    assert_eq!(
        errors.len(),
        1,
        "execution logs its rejection once: {errors:?}"
    );
    assert!(
        errors[0].contains("native backend could not parse OpenQASM 2.0")
            && errors[0].contains("'reset' is not supported"),
        "{errors:?}"
    );
}
