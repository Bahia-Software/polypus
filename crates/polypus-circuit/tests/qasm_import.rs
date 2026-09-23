//! Integration tests for OpenQASM 2.0 import ([`ParameterizedCircuit::from_qasm2`])
//! and the export → import → export round-trip guarantee.

use polypus_circuit::{CircuitError, GateInstruction, GateParam, Param, ParameterizedCircuit};

/// The full-vocabulary circuit used by the round-trip tests: every gate the
/// builder offers, plus subset barriers and individual measurements.
fn full_vocabulary_circuit() -> ParameterizedCircuit {
    ParameterizedCircuit::new(3)
        .h(0)
        .x(1)
        .y(2)
        .z(0)
        .s(1)
        .t(2)
        .sdg(0)
        .tdg(1)
        .id(2)
        .rx(0, 0.25)
        .ry(1, Param(0))
        .rz(2, -1.5)
        .u(0, 0.1, Param(1), 0.3)
        .cx(0, 1)
        .cz(1, 2)
        .swap(0, 2)
        .rzz(0, 2, Param(0))
        .rxx(1, 2, 2.0)
        .barrier()
        .barrier_on(&[0, 2])
        .measure(0, 0)
        .measure(2, 1)
}

// ───────────────────────────── Round-trip ─────────────────────────────────

/// Core guarantee: for circuits produced by this crate,
/// export → import → export is byte-identical.
#[test]
fn roundtrip_full_vocabulary_is_byte_stable() {
    let qasm1 = full_vocabulary_circuit()
        .to_qasm2_with_params(&[0.75, -0.4])
        .unwrap();
    let imported = ParameterizedCircuit::from_qasm2(&qasm1).unwrap();
    let qasm2 = imported.to_qasm2_with_params(&[]).unwrap();
    assert_eq!(qasm1, qasm2);
}

#[test]
fn roundtrip_measure_all_collapses_back() {
    let qasm1 = ParameterizedCircuit::new(4)
        .h(0)
        .cx(0, 1)
        .cx(1, 2)
        .cx(2, 3)
        .measure_all()
        .to_qasm2_with_params(&[])
        .unwrap();
    let imported = ParameterizedCircuit::from_qasm2(&qasm1).unwrap();
    // `measure q -> c;` must come back as a single MeasureAll, not 4 Measures.
    assert_eq!(imported.gates.last(), Some(&GateInstruction::MeasureAll));
    assert_eq!(imported.to_qasm2_with_params(&[]).unwrap(), qasm1);
}

#[test]
fn roundtrip_preserves_structure_and_counts() {
    let original = full_vocabulary_circuit();
    let qasm = original.to_qasm2_with_params(&[0.75, -0.4]).unwrap();
    let imported = ParameterizedCircuit::from_qasm2(&qasm).unwrap();

    assert_eq!(imported.num_qubits, original.num_qubits);
    assert_eq!(imported.num_clbits(), original.num_clbits());
    // QASM 2.0 is parameter-free: the import is fully concrete.
    assert_eq!(imported.num_params, 0);
    assert_eq!(imported.gates.len(), original.gates.len());
}

#[test]
fn imported_angles_match_bound_values() {
    let qasm = ParameterizedCircuit::new(1)
        .ry(0, Param(0))
        .to_qasm2_with_params(&[std::f64::consts::PI])
        .unwrap();
    let imported = ParameterizedCircuit::from_qasm2(&qasm).unwrap();
    match &imported.gates[0] {
        GateInstruction::Ry {
            qubit: 0,
            theta: GateParam::Fixed(v),
        } => {
            // 12 decimal places of precision survive the round-trip.
            assert!((v - std::f64::consts::PI).abs() < 1e-11);
        }
        other => panic!("expected Ry, got {other:?}"),
    }
}

// ───────────────────────── Qiskit interoperability ────────────────────────

/// Parse the exact shape Qiskit's `qasm2.dumps` produces: `creg meas`,
/// `u`/`p`/`swap`/`id` gates, explicit barrier list, per-qubit measures.
#[test]
fn parses_qiskit_dumps_output() {
    let src = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg meas[3];
h q[0];
rzz(0.4) q[0],q[1];
rx(0.8) q[2];
u(0.1,0.2,0.3) q[0];
p(0.5) q[1];
swap q[0],q[2];
id q[1];
barrier q[0],q[1],q[2];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
"#;
    let qc = ParameterizedCircuit::from_qasm2(src).unwrap();
    assert_eq!(qc.num_qubits, 3);
    assert_eq!(qc.num_clbits(), 3);

    // swap → native Swap; id kept one-to-one (it counts towards gate count and
    // depth); explicit full barrier → whole-register form; 3 contiguous
    // measures → MeasureAll.
    let expected = [
        GateInstruction::H(0),
        GateInstruction::Rzz {
            q0: 0,
            q1: 1,
            theta: GateParam::Fixed(0.4),
        },
        GateInstruction::Rx {
            qubit: 2,
            theta: GateParam::Fixed(0.8),
        },
        GateInstruction::U {
            qubit: 0,
            theta: GateParam::Fixed(0.1),
            phi: GateParam::Fixed(0.2),
            lam: GateParam::Fixed(0.3),
        },
        GateInstruction::U {
            qubit: 1,
            theta: GateParam::Fixed(0.0),
            phi: GateParam::Fixed(0.0),
            lam: GateParam::Fixed(0.5),
        },
        GateInstruction::Swap(0, 2),
        GateInstruction::Id(1),
        GateInstruction::Barrier(Vec::new()),
        GateInstruction::MeasureAll,
    ];
    assert_eq!(qc.gates, expected);
}

#[test]
fn supports_register_broadcast() {
    let src = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q;
rx(pi/2) q;
measure q -> c;
"#;
    let qc = ParameterizedCircuit::from_qasm2(src).unwrap();
    let h_count = qc
        .gates
        .iter()
        .filter(|g| matches!(g, GateInstruction::H(_)))
        .count();
    assert_eq!(h_count, 3);
    for q in 0..3 {
        assert!(qc.gates.contains(&GateInstruction::Rx {
            qubit: q,
            theta: GateParam::Fixed(std::f64::consts::FRAC_PI_2),
        }));
    }
    assert_eq!(qc.gates.last(), Some(&GateInstruction::MeasureAll));
}

#[test]
fn flattens_multiple_registers_in_declaration_order() {
    let src = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg a[2];
qreg b[2];
creg c[4];
x a[1];
x b[0];
cx a[0],b[1];
measure b[1] -> c[3];
"#;
    let qc = ParameterizedCircuit::from_qasm2(src).unwrap();
    assert_eq!(qc.num_qubits, 4);
    // a = [0,1], b = [2,3]
    assert_eq!(
        qc.gates,
        [
            GateInstruction::X(1),
            GateInstruction::X(2),
            GateInstruction::Cx(0, 3),
            GateInstruction::Measure { qubit: 3, cbit: 3 },
        ]
    );
}

// ─────────────────────── Expressions and lexing ───────────────────────────

#[test]
fn evaluates_constant_angle_expressions() {
    let src = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rx(pi/2) q[0];
ry(-pi/4) q[0];
rz(2*pi) q[0];
u3(1.5e-3, (1+2)*pi, cos(0)) q[0];
rx(2^3) q[0];
"#;
    let qc = ParameterizedCircuit::from_qasm2(src).unwrap();
    use std::f64::consts::PI;
    let angle = |g: &GateInstruction| -> f64 {
        match g {
            GateInstruction::Rx {
                theta: GateParam::Fixed(v),
                ..
            }
            | GateInstruction::Ry {
                theta: GateParam::Fixed(v),
                ..
            }
            | GateInstruction::Rz {
                theta: GateParam::Fixed(v),
                ..
            } => *v,
            GateInstruction::U {
                theta: GateParam::Fixed(v),
                ..
            } => *v,
            other => panic!("unexpected gate {other:?}"),
        }
    };
    assert!((angle(&qc.gates[0]) - PI / 2.0).abs() < 1e-15);
    assert!((angle(&qc.gates[1]) + PI / 4.0).abs() < 1e-15);
    assert!((angle(&qc.gates[2]) - 2.0 * PI).abs() < 1e-15);
    assert!((angle(&qc.gates[3]) - 1.5e-3).abs() < 1e-18);
    match &qc.gates[3] {
        GateInstruction::U {
            phi: GateParam::Fixed(phi),
            lam: GateParam::Fixed(lam),
            ..
        } => {
            assert!((phi - 3.0 * PI).abs() < 1e-14);
            assert!((lam - 1.0).abs() < 1e-15);
        }
        other => panic!("expected U, got {other:?}"),
    }
    assert!((angle(&qc.gates[4]) - 8.0).abs() < 1e-15);
}

#[test]
fn skips_comments_and_blank_lines() {
    let src = "// leading comment\nOPENQASM 2.0;\n\ninclude \"qelib1.inc\"; // trailing\n// gate-less\nqreg q[1];\nh q[0];\n";
    let qc = ParameterizedCircuit::from_qasm2(src).unwrap();
    assert_eq!(qc.gates, [GateInstruction::H(0)]);
}

// ───────────────────────────── Error cases ────────────────────────────────

/// Helper: assert parsing fails with a message containing `needle` at `line`.
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

#[test]
fn rejects_missing_header() {
    assert_parse_err("qreg q[1];\nh q[0];\n", "OPENQASM 2.0", 1);
}

#[test]
fn rejects_wrong_version() {
    assert_parse_err("OPENQASM 3.0;\n", "version 2.0", 1);
}

#[test]
fn rejects_unsupported_gate_with_line_number() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nccz q[0],q[1];\n";
    assert_parse_err(src, "unsupported gate 'ccz'", 4);
}

#[test]
fn rejects_opaque_declarations_by_name() {
    let src = "OPENQASM 2.0;\nqreg q[1];\nopaque magic(t) a;\n";
    assert_parse_err(src, "opaque gate 'magic' has no definition", 3);
}

#[test]
fn rejects_undeclared_register() {
    let src = "OPENQASM 2.0;\nqreg q[1];\nh r[0];\n";
    assert_parse_err(src, "undeclared quantum register 'r'", 3);
}

#[test]
fn rejects_out_of_range_index() {
    let src = "OPENQASM 2.0;\nqreg q[2];\nh q[5];\n";
    assert_parse_err(src, "out of range", 3);
}

#[test]
fn rejects_identical_qubits_in_two_qubit_gate() {
    let src = "OPENQASM 2.0;\nqreg q[2];\ncx q[1],q[1];\n";
    assert_parse_err(src, "distinct qubits", 3);
}

#[test]
fn rejects_wrong_parameter_count() {
    let src = "OPENQASM 2.0;\nqreg q[1];\nrx(0.1,0.2) q[0];\n";
    assert_parse_err(src, "expects 1 parameter(s), found 2", 3);
}

#[test]
fn rejects_broadcast_size_mismatch() {
    let src = "OPENQASM 2.0;\nqreg a[2];\nqreg b[3];\ncx a,b;\n";
    assert_parse_err(src, "register size mismatch", 4);
}

#[test]
fn rejects_measure_size_mismatch() {
    let src = "OPENQASM 2.0;\nqreg q[3];\ncreg c[2];\nmeasure q -> c;\n";
    assert_parse_err(src, "measure size mismatch", 4);
}

#[test]
fn rejects_duplicate_register_name() {
    let src = "OPENQASM 2.0;\nqreg q[2];\ncreg q[2];\n";
    assert_parse_err(src, "already declared", 3);
}

#[test]
fn rejects_reset_and_if() {
    assert_parse_err("OPENQASM 2.0;\nqreg q[1];\nreset q[0];\n", "'reset'", 3);
    let src = "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nif (c==1) x q[0];\n";
    assert_parse_err(src, "'if'", 4);
}

/// The known limitations are reported by name, with the reason, rather than as
/// a generic failure.
#[test]
fn known_limitations_have_actionable_errors() {
    // Classical control: names the construct and the terminal-measurement model.
    let src = "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nif (c==1) x q[0];\n";
    assert_parse_err(src, "classical control", 4);
    assert_parse_err(src, "contract C-4", 4);
    assert_parse_err(
        "OPENQASM 2.0;\nqreg q[1];\nreset q[0];\n",
        "known limitation",
        3,
    );
    // An arbitrary-control multi-controlled gate without a declaration...
    assert_parse_err(
        "OPENQASM 2.0;\nqreg q[4];\nmcx q[0],q[1],q[2],q[3];\n",
        "multi-controlled gates with an arbitrary number of controls",
        3,
    );
    // ... is fine when the program declares it (Qiskit's exporter does).
    let declared = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\ngate mcx q0,q1,q2,q3 { h q3; ccx q0,q1,q2; h q3; }\nqreg q[4];\nmcx q[0],q[1],q[2],q[3];\n";
    assert!(ParameterizedCircuit::from_qasm2(declared).is_ok());
    // Any other unknown name says it is neither built in nor declared.
    assert_parse_err(
        "OPENQASM 2.0;\nqreg q[1];\nfoo q[0];\n",
        "neither a qelib1.inc gate nor declared with a `gate` block",
        3,
    );
}

#[test]
fn rejects_division_by_zero_in_parameter() {
    let src = "OPENQASM 2.0;\nqreg q[1];\nrx(1/0) q[0];\n";
    assert_parse_err(src, "division by zero", 3);
}

#[test]
fn rejects_non_finite_parameter_expression() {
    // ln(0) = -inf: caught by the general finiteness guard.
    let src = "OPENQASM 2.0;\nqreg q[1];\nrx(ln(0)) q[0];\n";
    assert_parse_err(src, "non-finite", 3);

    // 0^-1 = inf via powf: same guard, different route.
    let src = "OPENQASM 2.0;\nqreg q[1];\nrx(0^-1) q[0];\n";
    assert_parse_err(src, "non-finite", 3);
}

#[test]
fn rejects_truncated_input() {
    let src = "OPENQASM 2.0;\nqreg q[1];\nh q[0]";
    assert!(matches!(
        ParameterizedCircuit::from_qasm2(src),
        Err(CircuitError::Parse { .. })
    ));
}

// ───────────────────── `id` is an instruction, not a no-op ────────────────

/// Regression: `id` used to be dropped at parse time, which silently changed
/// the gate count and depth of the imported circuit relative to the source
/// (and to what Qiskit reports for the same file).
#[test]
fn id_is_preserved_and_roundtrips_byte_identically() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\nid q[1];\nid q[0];\ncx q[0],q[1];\n";
    let qc = ParameterizedCircuit::from_qasm2(src).unwrap();
    assert_eq!(
        qc.gates,
        [
            GateInstruction::H(0),
            GateInstruction::Id(1),
            GateInstruction::Id(0),
            GateInstruction::Cx(0, 1),
        ]
    );
    assert_eq!(qc.to_qasm2_with_params(&[]).unwrap(), src);
}

#[test]
fn id_broadcasts_over_a_register() {
    let src = "OPENQASM 2.0;\nqreg q[3];\nid q;\n";
    let qc = ParameterizedCircuit::from_qasm2(src).unwrap();
    assert_eq!(
        qc.gates,
        [
            GateInstruction::Id(0),
            GateInstruction::Id(1),
            GateInstruction::Id(2)
        ]
    );
}

#[test]
fn id_rejects_wrong_arity_parameters_and_range() {
    assert_parse_err(
        "OPENQASM 2.0;\nqreg q[2];\nid q[0],q[1];\n",
        "gate 'id' expects 1 argument(s), found 2",
        3,
    );
    assert_parse_err(
        "OPENQASM 2.0;\nqreg q[1];\nid(0.5) q[0];\n",
        "gate 'id' expects 0 parameter(s), found 1",
        3,
    );
    assert_parse_err("OPENQASM 2.0;\nqreg q[2];\nid q[2];\n", "out of range", 3);
}

// ─────────────── Tier-1 qelib1.inc gates: one-to-one import ───────────────

/// Each gate imports into its own instruction with the operands in source
/// order (control(s) first) and the angles in source order. The operands are
/// deliberately not ascending, so a swapped control/target shows up.
#[test]
fn qelib1_gates_import_with_their_operand_and_parameter_order() {
    use GateInstruction as G;
    use GateParam::Fixed;
    let cases: Vec<(&str, GateInstruction)> = vec![
        ("sx q[2];", G::Sx(2)),
        ("sxdg q[1];", G::Sxdg(1)),
        ("cy q[2],q[0];", G::Cy(2, 0)),
        ("ch q[1],q[0];", G::Ch(1, 0)),
        ("csx q[2],q[1];", G::Csx(2, 1)),
        ("ccx q[2],q[0],q[1];", G::Ccx(2, 0, 1)),
        ("cswap q[1],q[2],q[0];", G::Cswap(1, 2, 0)),
        (
            "crx(0.5) q[2],q[0];",
            G::Crx {
                control: 2,
                target: 0,
                theta: Fixed(0.5),
            },
        ),
        (
            "cry(-0.5) q[1],q[0];",
            G::Cry {
                control: 1,
                target: 0,
                theta: Fixed(-0.5),
            },
        ),
        (
            "crz(pi/4) q[0],q[2];",
            G::Crz {
                control: 0,
                target: 2,
                theta: Fixed(std::f64::consts::FRAC_PI_4),
            },
        ),
        (
            "cu1(0.25) q[2],q[1];",
            G::Cu1 {
                q0: 2,
                q1: 1,
                theta: Fixed(0.25),
            },
        ),
        (
            "cu3(0.1,0.2,0.3) q[1],q[2];",
            G::Cu3 {
                control: 1,
                target: 2,
                theta: Fixed(0.1),
                phi: Fixed(0.2),
                lam: Fixed(0.3),
            },
        ),
        (
            "cu(0.1,0.2,0.3,0.4) q[2],q[0];",
            G::Cu {
                control: 2,
                target: 0,
                theta: Fixed(0.1),
                phi: Fixed(0.2),
                lam: Fixed(0.3),
                gamma: Fixed(0.4),
            },
        ),
    ];
    for (statement, expected) in cases {
        let src = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\n{statement}\n");
        let qc =
            ParameterizedCircuit::from_qasm2(&src).unwrap_or_else(|e| panic!("{statement}: {e}"));
        assert_eq!(qc.gates, [expected], "{statement}");
    }
}

/// (name, parameters, qubits) of every gate added for the benchmark suites
/// (Tier 1, then the multi-qubit `qelib1.inc` gates), for the signature checks.
const TIER1_SIGNATURES: [(&str, usize, usize); 19] = [
    ("sx", 0, 1),
    ("sxdg", 0, 1),
    ("cy", 0, 2),
    ("ch", 0, 2),
    ("csx", 0, 2),
    ("ccx", 0, 3),
    ("cswap", 0, 3),
    ("crx", 1, 2),
    ("cry", 1, 2),
    ("crz", 1, 2),
    ("cu1", 1, 2),
    ("cu3", 3, 2),
    ("cu", 4, 2),
    ("u0", 1, 1),
    ("rccx", 0, 3),
    ("rc3x", 0, 4),
    ("c3x", 0, 4),
    ("c3sqrtx", 0, 4),
    ("c4x", 0, 5),
];

/// The multi-qubit qelib1.inc gates import one-to-one, operands in source order.
#[test]
fn multi_qubit_qelib1_gates_import_with_their_operand_order() {
    use GateInstruction as G;
    let cases: Vec<(&str, GateInstruction)> = vec![
        (
            "u0(2) q[4];",
            G::U0 {
                qubit: 4,
                gamma: GateParam::Fixed(2.0),
            },
        ),
        ("rccx q[4],q[0],q[2];", G::Rccx(4, 0, 2)),
        ("rc3x q[3],q[1],q[4],q[0];", G::Rc3x(3, 1, 4, 0)),
        ("c3x q[2],q[4],q[0],q[3];", G::C3x(2, 4, 0, 3)),
        ("c3sqrtx q[4],q[3],q[1],q[2];", G::C3sqrtx(4, 3, 1, 2)),
        ("c4x q[1],q[4],q[0],q[3],q[2];", G::C4x(1, 4, 0, 3, 2)),
    ];
    for (statement, expected) in cases {
        let src = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[5];\n{statement}\n");
        let qc =
            ParameterizedCircuit::from_qasm2(&src).unwrap_or_else(|e| panic!("{statement}: {e}"));
        assert_eq!(qc.gates, [expected], "{statement}");
    }
    // Repeated operands are rejected at any arity.
    assert_parse_err(
        "OPENQASM 2.0;\nqreg q[5];\nc4x q[0],q[1],q[2],q[1],q[4];\n",
        "5-qubit gate requires distinct qubits, got (0, 1, 2, 1, 4)",
        3,
    );
}

/// Render a gate application with `params` angles on the qubits `operands`.
fn application(name: &str, params: usize, operands: impl IntoIterator<Item = usize>) -> String {
    let angles: Vec<String> = (0..params).map(|i| format!("0.{}", i + 1)).collect();
    let angle_list = if params == 0 {
        String::new()
    } else {
        format!("({})", angles.join(","))
    };
    let operands: Vec<String> = operands.into_iter().map(|q| format!("q[{q}]")).collect();
    format!("{name}{angle_list} {};", operands.join(","))
}

#[test]
fn qelib1_gates_reject_wrong_argument_and_parameter_counts() {
    for (name, params, qubits) in TIER1_SIGNATURES {
        // One argument too many (and, for multi-qubit gates, one too few).
        let mut wrong_arities = vec![qubits + 1];
        if qubits > 1 {
            wrong_arities.push(qubits - 1);
        }
        for wrong in wrong_arities {
            let src = format!(
                "OPENQASM 2.0;\nqreg q[8];\n{}\n",
                application(name, params, 0..wrong)
            );
            assert_parse_err(
                &src,
                &format!("gate '{name}' expects {qubits} argument(s), found {wrong}"),
                3,
            );
        }
        // One parameter too many (and one too few for parameterised gates).
        let mut wrong_counts = vec![params + 1];
        if params > 0 {
            wrong_counts.push(params - 1);
        }
        for wrong in wrong_counts {
            let src = format!(
                "OPENQASM 2.0;\nqreg q[8];\n{}\n",
                application(name, wrong, 0..qubits)
            );
            assert_parse_err(
                &src,
                &format!("gate '{name}' expects {params} parameter(s), found {wrong}"),
                3,
            );
        }
    }
}

#[test]
fn qelib1_gates_reject_out_of_range_qubits() {
    for (name, params, qubits) in TIER1_SIGNATURES {
        // A `qubits`-wide register; the last operand is one past its end.
        let statement = application(name, params, (0..qubits - 1).chain([qubits]));
        let src = format!("OPENQASM 2.0;\nqreg q[{qubits}];\n{statement}\n");
        assert_parse_err(&src, "out of range", 3);
    }
}

#[test]
fn three_qubit_gates_reject_repeated_operands() {
    assert_parse_err(
        "OPENQASM 2.0;\nqreg q[3];\nccx q[0],q[1],q[0];\n",
        "three-qubit gate requires distinct qubits, got (0, 1, 0)",
        3,
    );
    assert_parse_err(
        "OPENQASM 2.0;\nqreg q[3];\ncswap q[2],q[1],q[1];\n",
        "three-qubit gate requires distinct qubits, got (2, 1, 1)",
        3,
    );
}

/// OpenQASM 2.0 broadcasting for a three-qubit gate over mixed register and
/// single-bit arguments: `ccx a,b,c[0];` is `ccx a[k],b[k],c[0];` for each k —
/// the same expansion Qiskit's loader produces.
#[test]
fn three_qubit_gates_broadcast_over_registers() {
    let src = "OPENQASM 2.0;\nqreg a[2];\nqreg b[2];\nqreg c[1];\nccx a,b,c[0];\ncswap c[0],a,b;\n";
    let qc = ParameterizedCircuit::from_qasm2(src).unwrap();
    // a = q[0..2], b = q[2..4], c = q[4]
    assert_eq!(
        qc.gates,
        [
            GateInstruction::Ccx(0, 2, 4),
            GateInstruction::Ccx(1, 3, 4),
            GateInstruction::Cswap(4, 0, 2),
            GateInstruction::Cswap(4, 1, 3),
        ]
    );
    // Registers of different sizes cannot broadcast together.
    assert_parse_err(
        "OPENQASM 2.0;\nqreg a[2];\nqreg b[3];\nqreg c[1];\nccx a,b,c[0];\n",
        "register size mismatch",
        5,
    );
}

// ─────────────────────── Imported circuits are usable ─────────────────────

/// An imported circuit must behave like any other ParameterizedCircuit:
/// concrete, exportable, and extensible via the builder.
#[test]
fn imported_circuit_is_a_first_class_citizen() {
    let qasm = ParameterizedCircuit::new(2)
        .h(0)
        .cx(0, 1)
        .to_qasm2_with_params(&[])
        .unwrap();
    let qc = ParameterizedCircuit::from_qasm2(&qasm)
        .unwrap()
        .rz(1, 0.5) // keep building on top of the import
        .measure_all();
    let out = qc.to_qasm2_with_params(&[]).unwrap();
    assert!(out.contains("rz(0.500000000000) q[1];"));
    assert!(out.ends_with("measure q -> c;\n"));
}
