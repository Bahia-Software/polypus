//! Hardening of the untrusted OpenQASM 2.0 import surface against
//! denial-of-service inputs:
//!
//! - expression recursion depth cap (no stack overflow), and long flat sums or
//!   products evaluated without recursion;
//! - register-size cap (no multi-gigabyte allocation).
//!
//! (The caps on `gate` declarations — nesting, expansion size, validation
//! budget — are covered in `gate_declarations.rs`.)
//!
//! (Non-finite angle rejection is covered by the crate's existing tests, which
//! exercise `CircuitError::NonFiniteParam`.)

use polypus_circuit::{CircuitError, ParameterizedCircuit};

const HEADER: &str = "OPENQASM 2.0;\nqreg q[1];\n";

fn parse_err(src: &str) -> (usize, String) {
    match ParameterizedCircuit::from_qasm2(src) {
        Err(CircuitError::Parse { line, message }) => (line, message),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

// ───────────────────────── recursion depth ──────────────────────────

#[test]
fn deeply_nested_parentheses_are_rejected_not_overflowed() {
    let src = format!(
        "{HEADER}rx({}1{}) q[0];\n",
        "(".repeat(500),
        ")".repeat(500)
    );
    let (_, message) = parse_err(&src);
    assert!(
        message.contains("nested too deeply"),
        "unexpected message: {message:?}"
    );
}

#[test]
fn deeply_nested_unary_minus_is_rejected_not_overflowed() {
    let src = format!("{HEADER}rx({}1) q[0];\n", "-".repeat(500));
    let (_, message) = parse_err(&src);
    assert!(
        message.contains("nested too deeply"),
        "unexpected message: {message:?}"
    );
}

#[test]
fn deeply_nested_exponent_is_rejected_not_overflowed() {
    // Right-associative `^` recurses through `factor`; a long chain must be
    // bounded too.
    let src = format!("{HEADER}rx(2{}) q[0];\n", "^2".repeat(500));
    let (_, message) = parse_err(&src);
    assert!(
        message.contains("nested too deeply"),
        "unexpected message: {message:?}"
    );
}

#[test]
fn moderately_nested_expression_still_parses() {
    // Well within the limit: must not be rejected.
    let src = format!(
        "{HEADER}rx({}0.5{}) q[0];\n",
        "(".repeat(20),
        ")".repeat(20)
    );
    let qc = ParameterizedCircuit::from_qasm2(&src).unwrap();
    assert_eq!(qc.gates.len(), 1);
}

// ─────────────────────────── register cap ───────────────────────────

#[test]
fn oversized_quantum_register_is_rejected() {
    let src = "OPENQASM 2.0;\nqreg q[4000000000];\nbarrier q;\n";
    let (line, message) = parse_err(src);
    assert_eq!(line, 2);
    assert!(
        message.contains("MAX_REGISTER_BITS"),
        "unexpected message: {message:?}"
    );
}

#[test]
fn oversized_classical_register_is_rejected() {
    let src = "OPENQASM 2.0;\ncreg c[4000000000];\n";
    let (_, message) = parse_err(src);
    assert!(
        message.contains("MAX_REGISTER_BITS"),
        "unexpected message: {message:?}"
    );
}

#[test]
fn accumulated_register_size_over_cap_is_rejected() {
    // Each is under the cap, but their sum exceeds it.
    let src = "OPENQASM 2.0;\nqreg a[600000];\nqreg b[600000];\n";
    let (line, message) = parse_err(src);
    assert_eq!(line, 3);
    assert!(message.contains("MAX_REGISTER_BITS"));
}

#[test]
fn register_size_near_usize_max_does_not_overflow() {
    // `1_000_000` (== cap) then a `usize::MAX` register would overflow the
    // running total; the checked add must reject it, not wrap.
    let src = format!("OPENQASM 2.0;\nqreg a[1000000];\nqreg b[{}];\n", usize::MAX);
    let (_, message) = parse_err(&src);
    assert!(message.contains("MAX_REGISTER_BITS"));
}

#[test]
fn register_at_a_reasonable_size_is_accepted() {
    let src = "OPENQASM 2.0;\nqreg q[1000];\ncreg c[1000];\nh q[0];\n";
    let qc = ParameterizedCircuit::from_qasm2(src).unwrap();
    assert_eq!(qc.num_qubits, 1000);
}

// ─────────────────── long (not deep) expressions ────────────────────

/// A sum or product of hundreds of thousands of terms is not nesting: it must
/// evaluate — without overflowing the stack — to the value the terms spell,
/// whether at top level or through a declared gate's body.
#[test]
fn very_long_sums_and_products_evaluate_without_recursion() {
    let terms = 300_000;
    let sum = vec!["1"; terms].join("+");
    let src = format!("{HEADER}rx({sum}) q[0];\n");
    let qc = ParameterizedCircuit::from_qasm2(&src).unwrap();
    assert_eq!(
        qc.gates,
        [polypus_circuit::GateInstruction::Rx {
            qubit: 0,
            theta: polypus_circuit::GateParam::Fixed(terms as f64),
        }]
    );

    let product = vec!["t"; terms].join("*");
    let src =
        format!("OPENQASM 2.0;\ngate g(t) a {{ rz({product}) a; }}\nqreg q[1];\ng(1) q[0];\n");
    let qc = ParameterizedCircuit::from_qasm2(&src).unwrap();
    let polypus_circuit::GateInstruction::Custom(call) = &qc.gates[0] else {
        panic!("expected a call");
    };
    assert_eq!(
        call.expand(&[]).unwrap(),
        [polypus_circuit::GateInstruction::Rz {
            qubit: 0,
            theta: polypus_circuit::GateParam::Fixed(1.0),
        }]
    );
}
