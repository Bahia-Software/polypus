//! Integration tests for the native circuit path through the evaluation layer.
//!
//! The decisive property verified here: binding a native circuit template
//! (`CircuitSource::Native`) NEVER touches Python. These tests run in a plain
//! `cargo test` binary where no Python interpreter has been initialised —
//! if `bind` (or QASM generation) acquired the GIL, `Python::with_gil` would
//! panic ("the Python interpreter is not initialized"). Passing tests are
//! therefore proof of GIL-freedom, not just a convention.

use polypus::circuit::{Param, ParameterizedCircuit};
use polypus::evaluation::CircuitSource;
use polypus::infrastructure::BoundCircuit;

/// QAOA MaxCut ansatz used across the test suite (4 qubits, ring graph, p=1).
fn qaoa_template() -> ParameterizedCircuit {
    let edges = [(0usize, 1usize), (1, 2), (2, 3), (3, 0)];
    let mut qc = ParameterizedCircuit::new(4);
    for q in 0..4 {
        qc = qc.h(q);
    }
    for &(a, b) in &edges {
        qc = qc.rzz(a, b, Param(0));
    }
    for q in 0..4 {
        qc = qc.rx(q, Param(1));
    }
    qc.measure_all()
}

// ─────────────────────────────────────────────────────────────────────────────
// GIL-freedom: native binding works with no Python interpreter at all
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn native_bind_requires_no_python_interpreter() {
    let source = CircuitSource::Native(qaoa_template());
    let bound = source.bind(&[0.4, 0.8]);
    match bound {
        // A native template binds to the native variant; serialising it to
        // OpenQASM 2.0 is pure Rust, so this still needs no interpreter.
        BoundCircuit::Native(circuit) => {
            let qasm = circuit.to_qasm2();
            assert!(qasm.starts_with("OPENQASM 2.0;"));
            assert!(qasm.contains("rzz(0.400000000000) q[0],q[1];"));
            assert!(qasm.contains("rx(0.800000000000) q[3];"));
            assert!(qasm.ends_with("measure q -> c;\n"));
        }
        BoundCircuit::Qasm2(_) | BoundCircuit::Qiskit(_) => {
            panic!("native template must bind to the native variant")
        }
    }
}

#[test]
fn native_bind_is_threadsafe_without_gil() {
    // Simulates the QML-oracle scenario: many worker threads binding
    // candidates concurrently. With Qiskit this serialises on the GIL; the
    // native path must scale freely across threads.
    let source = std::sync::Arc::new(CircuitSource::Native(qaoa_template()));
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let src = std::sync::Arc::clone(&source);
            std::thread::spawn(move || {
                for j in 0..100 {
                    let theta = [0.01 * i as f64, 0.02 * j as f64];
                    let BoundCircuit::Native(circuit) = src.bind(&theta) else {
                        panic!("expected native circuit");
                    };
                    assert!(circuit.to_qasm2().starts_with("OPENQASM 2.0;"));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("binding thread panicked");
    }
}

#[test]
fn native_num_params_known_without_python() {
    let source = CircuitSource::Native(qaoa_template());
    assert_eq!(source.num_params(), Some(2));
}

#[test]
#[should_panic(expected = "wrong number of parameter values")]
fn native_bind_panics_on_wrong_param_count() {
    let source = CircuitSource::Native(qaoa_template());
    let _ = source.bind(&[0.1]); // template declares 2 params
}

// ─────────────────────────────────────────────────────────────────────────────
// BoundCircuit utilities that must not need Python for the QASM variant
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bound_qasm_duplicate_requires_no_python() {
    let original = BoundCircuit::Qasm2("OPENQASM 2.0;\n".to_string());
    let copy = original.duplicate();
    match (original, copy) {
        (BoundCircuit::Qasm2(a), BoundCircuit::Qasm2(b)) => assert_eq!(a, b),
        _ => panic!("duplicate changed the variant"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread-safety contracts (compile-time): if any of these types loses
// Send/Sync, distributed evaluation breaks — fail loudly here.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn evaluation_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ParameterizedCircuit>();
    assert_send_sync::<CircuitSource>();
    assert_send_sync::<BoundCircuit>();
    assert_send_sync::<polypus::evaluation::VqcOracle>();
    assert_send_sync::<polypus::evaluation::QmlOracle>();
}
