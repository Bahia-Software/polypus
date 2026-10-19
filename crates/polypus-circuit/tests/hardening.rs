//! Hardening of the untrusted OpenQASM 2.0 import surface against
//! denial-of-service inputs:
//!
//! - expression recursion depth cap (no stack overflow), and long flat sums or
//!   products evaluated without recursion;
//! - register-size cap (no multi-gigabyte allocation);
//! - the budgets shared with the OpenQASM 3 importer (source and token size,
//!   declarations, instructions after broadcasting, barrier operands,
//!   expression nodes), with the audit's inputs (#343) among them.
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

// ─────────────────────── huge integer literals ──────────────────────

/// An integer literal too large for `usize` is still a number: it is accepted
/// in an expression (as the nearest `f64`), and where a register size or
/// index is required it is reported as such — not as an unreadable token that
/// hides the statement it belongs to.
#[test]
fn integer_literals_beyond_usize_are_numbers_not_lexer_errors() {
    let huge = "1427247692705959881058285969449495136382746624";
    let qc =
        ParameterizedCircuit::from_qasm2(&format!("{HEADER}rz({huge}/{huge}) q[0];\n")).unwrap();
    assert_eq!(
        qc.gates,
        [polypus_circuit::GateInstruction::Rz {
            qubit: 0,
            theta: polypus_circuit::GateParam::Fixed(1.0),
        }]
    );

    // QASMBench's `cc_n151` compares a classical register with such a value:
    // the error names the unsupported `if`, not the literal.
    let src = format!("{HEADER}creg c[1];\nif(c=={huge}) x q[0];\n");
    let (line, message) = parse_err(&src);
    assert_eq!(line, 4);
    assert!(
        message.contains("'if' statements are not supported"),
        "{message}"
    );

    let (line, message) = parse_err(&format!("OPENQASM 2.0;\nqreg q[{huge}];\n"));
    assert_eq!(line, 2);
    assert!(message.contains("expected register size"), "{message}");
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

// ──────────────────────── shared import budgets ─────────────────────

const MAX_SOURCE_BYTES: usize = 32 << 20;
const MAX_TOKEN_BYTES: usize = 4096;
const MAX_DECLARATIONS: usize = 10_000;
const MAX_EXPR_NODES: usize = 1_000_000;

#[test]
fn the_source_size_is_bounded() {
    let at = |len: usize| format!("{HEADER}//{}\n", "x".repeat(len - HEADER.len() - 3));
    let src = at(MAX_SOURCE_BYTES);
    assert_eq!(src.len(), MAX_SOURCE_BYTES);
    assert!(ParameterizedCircuit::from_qasm2(&src).is_ok());
    let (line, message) = parse_err(&at(MAX_SOURCE_BYTES + 1));
    assert_eq!(line, 1);
    assert!(message.contains("exceeds MAX_SOURCE_BYTES"), "{message}");
}

#[test]
fn tokens_are_bounded() {
    let name = format!("r{}", "a".repeat(MAX_TOKEN_BYTES - 1));
    let qc = ParameterizedCircuit::from_qasm2(&format!(
        "OPENQASM 2.0;\nqreg {name}[1];\nh {name}[0];\n"
    ))
    .unwrap();
    assert_eq!(qc.num_qubits, 1);
    let number = "0".repeat(MAX_TOKEN_BYTES);
    assert!(ParameterizedCircuit::from_qasm2(&format!("{HEADER}rx({number}) q[0];\n")).is_ok());
    let string = "a".repeat(MAX_TOKEN_BYTES - 2);
    assert!(ParameterizedCircuit::from_qasm2(&format!("{HEADER}include \"{string}\";\n")).is_ok());

    for over in [
        format!("qreg {name}b[1];"),
        format!("rx({number}0) q[0];"),
        format!("include \"{string}b\";"),
    ] {
        let (line, message) = parse_err(&format!("{HEADER}{over}\n"));
        assert_eq!(line, 3);
        assert!(
            message.contains("token longer than MAX_TOKEN_BYTES (4096 bytes)"),
            "{message}"
        );
    }
}

#[test]
fn declarations_are_bounded() {
    let mut src = String::from(HEADER);
    for i in 0..MAX_DECLARATIONS {
        src.push_str(&format!("gate g{i} a {{ }}\n"));
    }
    assert!(ParameterizedCircuit::from_qasm2(&src).is_ok());
    src.push_str("gate one_more a { }\n");
    let (line, message) = parse_err(&src);
    assert_eq!(line, MAX_DECLARATIONS + 3);
    assert!(
        message.contains("more than MAX_DECLARATIONS (10000)"),
        "{message}"
    );
}

/// The audit's first input: 153 bytes that broadcast to 20 000 000
/// instructions. The fifth `h q;` is rejected before it is expanded.
#[test]
fn instructions_after_broadcasting_are_bounded() {
    let src = format!("OPENQASM 2.0;\nqreg q[1000000];\n{}", "h q;\n".repeat(20));
    let (line, message) = parse_err(&src);
    assert_eq!(line, 7);
    assert!(
        message.contains("more than MAX_INSTRUCTIONS (4000000)"),
        "{message}"
    );
    // Register-broadcast measurements count alike.
    let src = "OPENQASM 2.0;\nqreg q[1000000];\ncreg c[1000000];\n\
               h q;\nh q;\nh q;\nh q[0];\nmeasure q -> c;\n";
    let (line, message) = parse_err(src);
    assert_eq!(line, 8);
    assert!(message.contains("MAX_INSTRUCTIONS"), "{message}");
}

/// The audit's third input: each `barrier q;` lists every qubit until the
/// circuit is assembled, so the operands of all barriers are bounded.
#[test]
fn barrier_operands_are_bounded() {
    let src = |barriers: &str| format!("OPENQASM 2.0;\nqreg q[1000000];\n{barriers}");
    let at = src(&"barrier q;\n".repeat(4));
    assert_eq!(
        ParameterizedCircuit::from_qasm2(&at).unwrap().gates,
        vec![polypus_circuit::GateInstruction::Barrier(vec![]); 4]
    );
    let (line, message) = parse_err(&format!("{at}barrier q[0];\n"));
    assert_eq!(line, 7);
    assert!(
        message.contains("more than MAX_BARRIER_OPERANDS (4000000) qubits"),
        "{message}"
    );
    let (line, _) = parse_err(&src(&"barrier q;\n".repeat(100)));
    assert_eq!(line, 7);
}

#[test]
fn a_parameter_list_is_bounded() {
    // n terms of a sum are 2n − 1 nodes; with the first parameter, n = N/2
    // fill MAX_EXPR_NODES exactly, and a unary minus is one more.
    let sum = format!("0{}", "+0".repeat(MAX_EXPR_NODES / 2 - 1));
    assert!(ParameterizedCircuit::from_qasm2(&format!("{HEADER}u2(0,{sum}) q[0];\n")).is_ok());
    let (line, message) = parse_err(&format!("{HEADER}u2(-0,{sum}) q[0];\n"));
    assert_eq!(line, 3);
    assert!(
        message.contains("parameter list has more than 1000000 expression nodes"),
        "{message}"
    );
}

#[test]
fn the_expression_nodes_kept_in_gate_bodies_are_bounded() {
    // 999 999 nodes per body: four fit in 4 000 000, a fifth does not.
    let sum = format!("t{}", "+t".repeat(MAX_EXPR_NODES / 2 - 1));
    let gates = |n: usize| {
        let mut src = String::from(HEADER);
        for i in 0..n {
            src.push_str(&format!("gate g{i}(t) a {{ rz({sum}) a; }}\n"));
        }
        src
    };
    assert!(ParameterizedCircuit::from_qasm2(&gates(4)).is_ok());
    let (line, message) = parse_err(&gates(5));
    assert_eq!(line, 7);
    assert!(message.contains("MAX_PROGRAM_NODES (4000000"), "{message}");
}
