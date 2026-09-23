//! OpenQASM 2.0 `gate` declarations: a declaration becomes a template, each
//! call one [`GateInstruction::Custom`] instruction (never its expanded body),
//! and the exporter re-emits the declaration verbatim plus the calls.

use polypus_circuit::{CircuitError, GateInstruction, GateParam, ParameterizedCircuit};
use std::f64::consts::PI;

const HEADER: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n";

fn parse(src: &str) -> ParameterizedCircuit {
    ParameterizedCircuit::from_qasm2(src).unwrap_or_else(|e| panic!("{e}\n{src}"))
}

fn assert_parse_err(src: &str, needle: &str, expected_line: usize) {
    match ParameterizedCircuit::from_qasm2(src) {
        Err(CircuitError::Parse { line, message }) => {
            assert!(
                message.contains(needle),
                "expected message containing {needle:?}, got {message:?}"
            );
            assert_eq!(line, expected_line, "wrong line for {message:?}");
        }
        other => panic!("expected Parse error, got {other:?}"),
    }
}

fn call(gate: &GateInstruction) -> &polypus_circuit::CustomGate {
    match gate {
        GateInstruction::Custom(call) => call,
        other => panic!("expected a call of a declared gate, got {other:?}"),
    }
}

// ───────────────────────────── Round-trip ─────────────────────────────────

/// A program in canonical form — declarations (as Qiskit's exporter writes
/// them) before the registers, calls with 12-decimal angles — is re-emitted
/// byte-identically: the same declarations, the same calls, nothing expanded.
#[test]
fn declarations_and_calls_reemit_byte_identically() {
    let src = format!(
        "{HEADER}\
gate ryy(param0) q0,q1 {{ sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }}\n\
gate xx_plus_yy(param0,param1) q0,q1 {{ rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }}\n\
gate ecr q0,q1 {{ s q0; sx q1; cx q0,q1; x q0; }}\n\
qreg q[3];\n\
creg c[3];\n\
h q[0];\n\
ryy(0.300000000000) q[2],q[0];\n\
xx_plus_yy(0.300000000000,-0.100000000000) q[1],q[2];\n\
ecr q[0],q[1];\n\
measure q -> c;\n"
    );
    let imported = parse(&src);
    assert_eq!(imported.gates.len(), 5); // h, three calls, measure_all
    assert_eq!(imported.to_qasm2_with_params(&[]).unwrap(), src);
}

/// A call is one instruction holding its definition, its angles in order and
/// its qubits in source order (not sorted).
#[test]
fn a_call_is_one_instruction_with_source_order_operands() {
    let src = format!(
        "{HEADER}gate g(a,b) x,y,z {{ rz(a) x; ry(b) z; ccx x,y,z; }}\nqreg q[4];\ng(0.5,-0.25) q[3],q[0],q[2];\n"
    );
    let imported = parse(&src);
    assert_eq!(imported.gates.len(), 1);
    let call = call(&imported.gates[0]);
    assert_eq!(call.name(), "g");
    assert_eq!(
        call.params(),
        [GateParam::Fixed(0.5), GateParam::Fixed(-0.25)]
    );
    assert_eq!(call.qubits(), [3, 0, 2]);
    assert_eq!(call.definition().num_params(), 2);
    assert_eq!(call.definition().num_qubits(), 3);
    assert_eq!(
        call.definition().declaration(),
        "gate g(a,b) x,y,z { rz(a) x; ry(b) z; ccx x,y,z; }"
    );
}

/// Nested declarations: calling only the outer gate re-emits both
/// declarations, in source order (inner first), before the registers.
#[test]
fn nested_declarations_are_reemitted_with_their_dependencies() {
    let src = format!(
        "{HEADER}gate inner(t) a {{ rz(t/2) a; }}\n\
gate outer(t) a,b {{ inner(t) b; cx a,b; inner(-t) b; }}\n\
qreg q[2];\nouter(1.000000000000) q[0],q[1];\n"
    );
    let imported = parse(&src);
    assert_eq!(imported.gates.len(), 1);
    assert_eq!(imported.to_qasm2_with_params(&[]).unwrap(), src);
}

/// A declaration no instruction reaches is not re-emitted: it does not change
/// the program a backend builds.
#[test]
fn unused_declarations_are_not_reemitted() {
    let src = format!("{HEADER}gate unused a {{ x a; }}\nqreg q[1];\nh q[0];\n");
    let out = parse(&src).to_qasm2_with_params(&[]).unwrap();
    assert_eq!(out, format!("{HEADER}qreg q[1];\nh q[0];\n"));
}

/// Declarations can appear anywhere before their first use; the canonical
/// export puts them right after the include, still in source order.
#[test]
fn declarations_after_registers_are_moved_before_them() {
    let src = format!("{HEADER}qreg q[2];\nh q[0];\ngate g a,b {{ cx a,b; }}\ng q[0],q[1];\n");
    let out = parse(&src).to_qasm2_with_params(&[]).unwrap();
    assert_eq!(
        out,
        format!("{HEADER}gate g a,b {{ cx a,b; }}\nqreg q[2];\nh q[0];\ng q[0],q[1];\n")
    );
}

/// The declaration text is kept verbatim (comments, line breaks); only CRLF
/// line endings are normalised.
#[test]
fn declaration_text_is_verbatim_with_crlf_normalised() {
    let src = "OPENQASM 2.0;\r\ninclude \"qelib1.inc\";\r\ngate g a,b\r\n{\r\n  cx a,b; // entangle\r\n  h a;\r\n}\r\nqreg q[2];\r\ng q[1],q[0];\r\n";
    let imported = parse(src);
    let out = imported.to_qasm2_with_params(&[]).unwrap();
    assert!(
        out.contains("gate g a,b\n{\n  cx a,b; // entangle\n  h a;\n}\n"),
        "{out}"
    );
    // … and the re-emitted text imports to the same circuit.
    assert_eq!(parse(&out).gates, imported.gates);
}

// ───────────────────────────── Expansion ──────────────────────────────────

/// Expansion (the lowering the simulator and QIR use) binds the formal
/// parameters and qubits, evaluates the body expressions, and goes through
/// nested declarations.
#[test]
fn expansion_binds_parameters_qubits_and_nested_gates() {
    let src = format!(
        "{HEADER}gate inner(t) a {{ rz(t/2) a; }}\n\
gate outer(t,u) a,b {{ inner(t) b; cx a,b; ry(u*t) a; barrier a,b; id b; }}\n\
qreg q[3];\nouter(pi,2) q[2],q[0];\n"
    );
    let imported = parse(&src);
    let expanded = call(&imported.gates[0]).expand(&[]).unwrap();
    assert_eq!(
        expanded,
        [
            GateInstruction::Rz {
                qubit: 0,
                theta: GateParam::Fixed(PI / 2.0),
            },
            GateInstruction::Cx(2, 0),
            GateInstruction::Ry {
                qubit: 2,
                theta: GateParam::Fixed(2.0 * PI),
            },
            GateInstruction::Barrier(vec![2, 0]),
            GateInstruction::Id(0),
        ]
    );
}

/// Built-in gates inside a body keep their own spelling and semantics
/// (`U`/`CX` builtins included), and an empty body expands to nothing.
#[test]
fn bodies_use_the_builtin_vocabulary() {
    let src = format!(
        "{HEADER}gate g a,b {{ U(0.1,0.2,0.3) a; CX a,b; cu1(0.5) b,a; sx b; }}\n\
gate nop a {{ }}\nqreg q[2];\ng q[0],q[1];\nnop q[1];\n"
    );
    let imported = parse(&src);
    let expanded = call(&imported.gates[0]).expand(&[]).unwrap();
    assert_eq!(
        expanded,
        [
            // The builtin `U` is Qiskit's `u`.
            GateInstruction::UGate {
                qubit: 0,
                theta: GateParam::Fixed(0.1),
                phi: GateParam::Fixed(0.2),
                lam: GateParam::Fixed(0.3),
            },
            GateInstruction::Cx(0, 1),
            GateInstruction::Cu1 {
                q0: 1,
                q1: 0,
                theta: GateParam::Fixed(0.5),
            },
            GateInstruction::Sx(1),
        ]
    );
    assert!(call(&imported.gates[1]).expand(&[]).unwrap().is_empty());
}

/// Calls broadcast over registers like any gate (OpenQASM 2.0 §3.1).
#[test]
fn calls_broadcast_over_registers() {
    let src =
        format!("{HEADER}gate g a,b {{ cx a,b; }}\nqreg a[2];\nqreg b[2];\ng a,b;\ng a[1],b;\n");
    let imported = parse(&src);
    let operands: Vec<Vec<usize>> = imported
        .gates
        .iter()
        .map(|g| call(g).qubits().to_vec())
        .collect();
    assert_eq!(operands, [vec![0, 2], vec![1, 3], vec![1, 2], vec![1, 3]]);
}

// ────────────────────────────── Validation ────────────────────────────────

#[test]
fn recursion_is_impossible() {
    let src = format!("{HEADER}gate g a {{ g a; }}\nqreg q[1];\n");
    assert_parse_err(&src, "calls itself", 3);
    // A body may only call gates declared before it.
    let src = format!("{HEADER}gate f a {{ later a; }}\ngate later a {{ x a; }}\n");
    assert_parse_err(&src, "unsupported gate 'later'", 3);
}

#[test]
fn declarations_cannot_be_repeated_or_shadow_qelib1() {
    let src = format!("{HEADER}gate g a {{ x a; }}\ngate g a {{ y a; }}\n");
    assert_parse_err(&src, "gate 'g' is already declared", 4);
    let src = format!("{HEADER}gate h a {{ U(pi/2,0,pi) a; }}\n");
    assert_parse_err(&src, "already defined by qelib1.inc", 3);
    let src = format!("{HEADER}gate CX a,b {{ }}\n");
    assert_parse_err(&src, "already defined by qelib1.inc", 3);
}

#[test]
fn malformed_declarations_are_rejected_with_the_gate_named() {
    let cases: [(&str, &str); 10] = [
        (
            "gate g(t,t) a { rz(t) a; }",
            "parameter 't' of gate 'g' is declared twice",
        ),
        (
            "gate g a,a { x a; }",
            "argument 'a' of gate 'g' is declared twice",
        ),
        (
            "gate g(a) a { x a; }",
            "argument 'a' of gate 'g' is declared twice",
        ),
        (
            "gate g a { x b; }",
            "'b' is not a qubit argument of gate 'g'",
        ),
        ("gate g a { x a[0]; }", "cannot be indexed"),
        (
            "gate g a { rz(u) a; }",
            "unknown identifier 'u' in expression",
        ),
        (
            "gate g a { measure a -> c[0]; }",
            "'measure' is not allowed inside the body of gate 'g'",
        ),
        (
            "gate g a,b { cx a,a; }",
            "two-qubit gate requires distinct qubits",
        ),
        (
            "gate g a { cx a; }",
            "gate 'cx' expects 2 argument(s), found 1",
        ),
        (
            "gate g a { rz a; }",
            "gate 'rz' expects 1 parameter(s), found 0",
        ),
    ];
    for (declaration, needle) in cases {
        let src = format!("{HEADER}{declaration}\n");
        assert_parse_err(&src, needle, 3);
    }
    let src = format!("{HEADER}gate g a {{ x a;\n");
    assert_parse_err(&src, "unterminated body of gate 'g'", 3);
}

#[test]
fn calls_are_checked_against_the_declaration() {
    let decl = format!("{HEADER}gate g(t) a,b {{ rz(t) a; cx a,b; }}\nqreg q[3];\n");
    assert_parse_err(
        &format!("{decl}g(0.1) q[0];\n"),
        "gate 'g' expects 2 argument(s), found 1",
        5,
    );
    assert_parse_err(
        &format!("{decl}g q[0],q[1];\n"),
        "gate 'g' expects 1 parameter(s), found 0",
        5,
    );
    assert_parse_err(
        &format!("{decl}g(0.1) q[1],q[1];\n"),
        "two-qubit gate requires distinct qubits",
        5,
    );
    assert_parse_err(&format!("{decl}g(0.1) q[0],q[7];\n"), "out of range", 5);
}

/// Each call's angles are evaluated through the whole body at parse time, so a
/// division by zero or a non-finite angle inside the body is rejected at the
/// call (contract C-2: no non-finite angle is ever accepted).
#[test]
fn body_angles_are_validated_at_each_call() {
    let decl = format!(
        "{HEADER}gate inner(t) a {{ rz(ln(t)) a; }}\ngate g(t) a {{ rz(1/t) a; inner(t) a; }}\nqreg q[1];\n"
    );
    assert!(ParameterizedCircuit::from_qasm2(&format!("{decl}g(2) q[0];\n")).is_ok());
    assert_parse_err(
        &format!("{decl}g(0) q[0];\n"),
        "in gate 'g': division by zero",
        6,
    );
    assert_parse_err(
        &format!("{decl}g(-1) q[0];\n"),
        "in gate 'g': parameter expression evaluated to a non-finite value",
        6,
    );
}

/// C-4 holds for calls: a declared gate is a unitary on all its qubits.
#[test]
fn a_call_on_a_measured_qubit_is_rejected() {
    let src = format!(
        "{HEADER}gate g a,b {{ cx a,b; }}\nqreg q[3];\ncreg c[1];\nmeasure q[2] -> c[0];\ng q[0],q[2];\n"
    );
    assert_parse_err(&src, "qubit 2 after it was measured", 7);
}

// ─────────────────────── Resource limits (hostile input) ──────────────────

/// Nesting is capped: every level calls the previous one, 70 levels deep.
#[test]
fn nesting_depth_is_capped() {
    let mut src = format!("{HEADER}gate g0 a {{ x a; }}\n");
    for level in 1..70 {
        src.push_str(&format!("gate g{level} a {{ g{} a; }}\n", level - 1));
    }
    assert_parse_err(&src, "nests gate calls more than 64 levels deep", 67);
}

/// `gate g0 a { <leaf> }` followed by `gate gk a { g(k-1) a; g(k-1) a; }` for
/// k = 1..=levels: each level calls the previous one twice.
fn doubling_declarations(leaf: &str, levels: usize) -> String {
    let mut src = format!("{HEADER}gate g0 a {{ {leaf} }}\n");
    for level in 1..=levels {
        src.push_str(&format!(
            "gate g{level} a {{ g{p} a; g{p} a; }}\n",
            p = level - 1
        ));
    }
    src
}

/// Exponential blow-up is capped. A full expansion of level k visits
/// 3·2^k − 2 body statements (2^k `x` gates plus 2^(k+1) − 2 nested calls), so
/// level 19 (1572862) is the first beyond the 1000000 cap.
#[test]
fn exponential_expansion_is_capped() {
    assert!(ParameterizedCircuit::from_qasm2(&doubling_declarations("x a;", 18)).is_ok());
    assert_parse_err(
        &doubling_declarations("x a;", 19),
        "gate 'g19' expands to more than 1000000",
        22,
    );
}

/// The cap counts nested calls, not only the built-in gates they reach: with
/// an empty `g0` every level expands to *nothing*, yet expanding (or
/// validating a call of) level k walks 2^(k+1) − 2 nested calls. Were calls
/// not counted, all 64 levels the nesting cap allows would be accepted, and a
/// single call of the last one would take 2^64 steps to validate.
#[test]
fn nested_calls_of_empty_gates_count_toward_the_cap() {
    assert!(ParameterizedCircuit::from_qasm2(&doubling_declarations("", 18)).is_ok());
    assert_parse_err(
        &doubling_declarations("", 19),
        "gate 'g19' expands to more than 1000000",
        22,
    );
}

/// Calling a large declaration many times hits the program-wide validation
/// budget (20000000) instead of taking unbounded time — also when the calls
/// expand to nothing.
#[test]
fn repeated_calls_of_large_declarations_are_bounded() {
    // A call of g18 visits 786430 body statements with `x` leaves, 524286 with
    // an empty g0: 64 calls exceed the budget either way.
    for leaf in ["x a;", ""] {
        let mut src = doubling_declarations(leaf, 18);
        src.push_str("qreg q[1];\n");
        for _ in 0..64 {
            src.push_str("g18 q[0];\n");
        }
        match ParameterizedCircuit::from_qasm2(&src) {
            Err(CircuitError::Parse { message, .. }) => assert!(
                message.contains("validating the calls of declared gates"),
                "{message}"
            ),
            other => panic!("expected the validation budget to trip, got {other:?}"),
        }
    }
}

// ─────────────────────────────── QIR ──────────────────────────────────────

/// QIR lowers a call exactly as it lowers the expanded body.
#[test]
fn qir_lowers_a_call_as_its_expansion() {
    let src = format!(
        "{HEADER}gate g(t) a,b {{ h a; cx a,b; rz(t) b; }}\nqreg q[2];\ng(0.5) q[1],q[0];\n"
    );
    let imported = parse(&src);
    let mut expanded = ParameterizedCircuit::new(2);
    expanded.gates = call(&imported.gates[0]).expand(&[]).unwrap();
    assert_eq!(
        imported.to_qir_with_params(&[]).unwrap(),
        expanded.to_qir_with_params(&[]).unwrap()
    );
}

// ──────────────────────── Combining imported programs ─────────────────────

/// Two different declarations under one name cannot be exported together
/// (one OpenQASM 2.0 program cannot declare both); identical ones can.
#[test]
fn conflicting_declarations_are_rejected_at_export() {
    let a = parse(&format!(
        "{HEADER}gate g a {{ x a; }}\nqreg q[1];\ng q[0];\n"
    ));
    let b = parse(&format!(
        "{HEADER}gate g a {{ y a; }}\nqreg q[1];\ng q[0];\n"
    ));
    let mut combined = ParameterizedCircuit::new(1);
    combined.try_push(a.gates[0].clone()).unwrap();
    combined.try_push(b.gates[0].clone()).unwrap();
    assert_eq!(
        combined.to_qasm2_with_params(&[]),
        Err(CircuitError::ConflictingGateDefinitions {
            name: "g".to_string()
        })
    );

    let same = parse(&format!(
        "{HEADER}gate g a {{ x a; }}\nqreg q[1];\ng q[0];\n"
    ));
    let mut combined = ParameterizedCircuit::new(1);
    combined.try_push(a.gates[0].clone()).unwrap();
    combined.try_push(same.gates[0].clone()).unwrap();
    let out = combined.to_qasm2_with_params(&[]).unwrap();
    assert_eq!(out.matches("gate g a").count(), 1, "{out}");
}

/// A call is validated like any instruction when pushed onto a builder.
#[test]
fn builder_validates_pushed_calls() {
    let imported = parse(&format!(
        "{HEADER}gate g a,b {{ cx a,b; }}\nqreg q[3];\ng q[2],q[0];\n"
    ));
    let mut small = ParameterizedCircuit::new(2);
    assert_eq!(
        small.try_push(imported.gates[0].clone()),
        Err(CircuitError::QubitOutOfRange {
            qubit: 2,
            num_qubits: 2
        })
    );
    let mut measured = ParameterizedCircuit::new(3);
    measured
        .try_push(GateInstruction::Measure { qubit: 0, cbit: 0 })
        .unwrap();
    assert_eq!(
        measured.try_push(imported.gates[0].clone()),
        Err(CircuitError::QubitAlreadyMeasured { qubit: 0 })
    );
}
