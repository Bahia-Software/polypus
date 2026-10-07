//! The OpenQASM 3 profile of `from_qasm3` and `to_qasm3`: what the importer
//! accepts and how it reads it, what it rejects (naming the construct and its
//! line), its budgets, and the canonical text the exporter writes — layout,
//! numbers, parentheses, renaming — and the exporter's own budgets, which keep
//! everything it writes importable.
//!
//! The matrices the profile reads are checked in `polypus-sim`'s
//! `qasm3_semantics.rs`; the round trip of the whole vocabulary in
//! `contracts.rs` (C-2).

use polypus_circuit::expr::Function;
use polypus_circuit::{
    CircuitError, ConcreteCircuit, Fixed, GateInstruction as G, GateParam, Param, ParamExpr,
    ParameterizedCircuit,
};

const HEADER: &str = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n";

fn import(src: &str) -> ParameterizedCircuit {
    ParameterizedCircuit::from_qasm3(src).unwrap_or_else(|e| panic!("{e}\n{src}"))
}

/// The line and message of the error importing `src`.
fn rejected(src: &str) -> (usize, String) {
    match ParameterizedCircuit::from_qasm3(src) {
        Err(CircuitError::Parse { line, message }) => (line, message),
        other => panic!("expected a parse error, got {other:?}\n{src}"),
    }
}

/// `statements` after the header and the registers `qubit[3] q;` and
/// `bit[3] c;` (lines 3 and 4): the first statement is on line 5.
fn program(statements: &str) -> String {
    format!("{HEADER}qubit[3] q;\nbit[3] c;\n{statements}\n")
}

/// The single angle `statement` (in [`program`]) imports to.
fn angle(statement: &str) -> GateParam {
    match import(&program(statement)).gates.as_slice() {
        [G::Rx { theta, .. }] => *theta,
        other => panic!("{statement}: {other:?}"),
    }
}

/// The exporter's output, checked to be a fixed point of import and export.
fn canonical(circuit: &ParameterizedCircuit) -> String {
    let out = circuit.to_qasm3().unwrap();
    let again = import(&out).to_qasm3().unwrap();
    assert_eq!(again, out, "not a fixed point");
    out
}

fn export_limit(circuit: &ParameterizedCircuit) -> &'static str {
    match circuit.to_qasm3() {
        Err(CircuitError::ExportLimit { limit, .. }) => limit,
        other => panic!("expected an export limit, got {:?}", other.map(|s| s.len())),
    }
}

// ───────────────────────────── Accepted ──────────────────────────────────

#[test]
fn the_version_statement_is_optional() {
    let body = "include \"stdgates.inc\";\nqubit[1] q;\nh q[0];\n";
    let expected = import(&format!("OPENQASM 3.0;\n{body}"));
    assert_eq!(expected.gates, [G::H(0)]);
    assert_eq!(import(&format!("OPENQASM 3;\n{body}")), expected);
    assert_eq!(import(body), expected);
}

#[test]
fn registers_are_flattened_in_declaration_order() {
    let qc = import(&format!(
        "{HEADER}qubit[2] a;\nqubit b;\nqubit[3] d;\nbit[2] m;\nbit n;\n\
         x b;\nh d[1];\ncx a[1], d[2];\nm[1] = measure a[0];\nn = measure d[2];\n"
    ));
    assert_eq!(qc.num_qubits, 6);
    assert_eq!(
        qc.gates,
        [
            G::X(2),
            G::H(4),
            G::Cx(1, 5),
            G::Measure { qubit: 0, cbit: 1 },
            G::Measure { qubit: 5, cbit: 2 },
        ]
    );
}

#[test]
fn register_arguments_broadcast() {
    let qc = import(&format!(
        "{HEADER}qubit[2] a;\nqubit[2] b;\nh a;\ncx a, b;\ncx a[0], b;\n"
    ));
    assert_eq!(
        qc.gates,
        [
            G::H(0),
            G::H(1),
            G::Cx(0, 2),
            G::Cx(1, 3),
            G::Cx(0, 2),
            G::Cx(0, 3),
        ]
    );
}

#[test]
fn inputs_are_the_parameters_in_declaration_order_under_their_names() {
    let qc = import(&format!(
        "{HEADER}input float[64] beta;\ninput float alpha;\ninput float[64] unused;\n\
         qubit[2] q;\nrx(alpha) q[0];\nry(beta) q[1];\n"
    ));
    assert_eq!(qc.num_params, 3);
    assert_eq!(qc.param_names(), ["beta", "alpha", "unused"]);
    assert_eq!(
        qc.gates,
        [
            G::Rx {
                qubit: 0,
                theta: Param(1)
            },
            G::Ry {
                qubit: 1,
                theta: Param(0)
            },
        ]
    );
    // An unused input is a parameter too: binding takes a value for it.
    assert_eq!(
        qc.assign_parameters(&[0.1, 0.2]).unwrap_err(),
        CircuitError::WrongNumberOfParams {
            expected: 3,
            got: 2
        }
    );
    assert_eq!(
        qc.assign_parameters(&[0.1, 0.2, 0.3]).unwrap().gates,
        [
            G::Rx {
                qubit: 0,
                theta: Fixed(0.2)
            },
            G::Ry {
                qubit: 1,
                theta: Fixed(0.1)
            },
        ]
    );
}

#[test]
fn input_names_keep_unicode_and_survive_the_round_trip() {
    let qc = import(&format!(
        "{HEADER}input float[64] θ;\ninput float[64] _θ_0_;\nqubit[1] q;\nrz(θ + _θ_0_) q[0];\n"
    ));
    assert_eq!(qc.param_names(), ["θ", "_θ_0_"]);
    let out = canonical(&qc);
    assert!(
        out.contains("input float[64] θ;\ninput float[64] _θ_0_;\n"),
        "{out}"
    );
    assert!(out.contains("rz(θ + _θ_0_) q[0];\n"), "{out}");
    assert_eq!(import(&out).param_names(), qc.param_names());
    assert_eq!(qc.clone().param_names(), qc.param_names());
}

#[test]
fn an_input_cannot_take_a_stdgates_name_even_without_the_include() {
    let (line, message) = rejected("OPENQASM 3.0;\nqubit q;\ninput float[64] t;\nU(t, 0, 0) q;\n");
    assert_eq!(line, 3);
    assert!(
        message.contains("'t' is a stdgates.inc gate name"),
        "{message}"
    );
    let (line, message) = rejected(&format!("{HEADER}input float[64] t;\n"));
    assert_eq!(line, 3);
    assert!(
        message.contains("'t' is already defined by stdgates.inc"),
        "{message}"
    );
}

/// Every `stdgates.inc` gate, and `U`, is the instruction of its own name —
/// except that `U` becomes `u`, `CX` becomes `cx`, `phase` becomes `p` and
/// `cphase` becomes `cp`.
#[test]
fn every_stdgates_gate_is_the_instruction_of_its_name() {
    let (t, f, l, g) = (Fixed(0.5), Fixed(-0.25), Fixed(1.5), Fixed(0.75));
    let cases = [
        ("p(0.5) q[1];", G::P { qubit: 1, lam: t }),
        ("x q[0];", G::X(0)),
        ("y q[1];", G::Y(1)),
        ("z q[2];", G::Z(2)),
        ("h q[0];", G::H(0)),
        ("s q[1];", G::S(1)),
        ("sdg q[2];", G::Sdg(2)),
        ("t q[0];", G::T(0)),
        ("tdg q[1];", G::Tdg(1)),
        ("sx q[2];", G::Sx(2)),
        ("rx(0.5) q[0];", G::Rx { qubit: 0, theta: t }),
        ("ry(0.5) q[1];", G::Ry { qubit: 1, theta: t }),
        ("rz(0.5) q[2];", G::Rz { qubit: 2, theta: t }),
        ("cx q[2], q[0];", G::Cx(2, 0)),
        ("cy q[2], q[0];", G::Cy(2, 0)),
        ("cz q[2], q[0];", G::Cz(2, 0)),
        (
            "cp(0.5) q[2], q[0];",
            G::Cp {
                q0: 2,
                q1: 0,
                theta: t,
            },
        ),
        (
            "crx(0.5) q[2], q[0];",
            G::Crx {
                control: 2,
                target: 0,
                theta: t,
            },
        ),
        (
            "cry(0.5) q[2], q[0];",
            G::Cry {
                control: 2,
                target: 0,
                theta: t,
            },
        ),
        (
            "crz(0.5) q[2], q[0];",
            G::Crz {
                control: 2,
                target: 0,
                theta: t,
            },
        ),
        ("ch q[2], q[0];", G::Ch(2, 0)),
        ("swap q[2], q[0];", G::Swap(2, 0)),
        ("ccx q[2], q[0], q[1];", G::Ccx(2, 0, 1)),
        ("cswap q[2], q[0], q[1];", G::Cswap(2, 0, 1)),
        (
            "cu(0.5, -0.25, 1.5, 0.75) q[2], q[0];",
            G::Cu {
                control: 2,
                target: 0,
                theta: t,
                phi: f,
                lam: l,
                gamma: g,
            },
        ),
        ("CX q[2], q[0];", G::Cx(2, 0)),
        ("phase(0.5) q[1];", G::P { qubit: 1, lam: t }),
        (
            "cphase(0.5) q[2], q[0];",
            G::Cp {
                q0: 2,
                q1: 0,
                theta: t,
            },
        ),
        ("id q[1];", G::Id(1)),
        ("u1(0.5) q[1];", G::U1 { qubit: 1, lam: t }),
        (
            "u2(-0.25, 1.5) q[1];",
            G::U2 {
                qubit: 1,
                phi: f,
                lam: l,
            },
        ),
        (
            "u3(0.5, -0.25, 1.5) q[1];",
            G::U {
                qubit: 1,
                theta: t,
                phi: f,
                lam: l,
            },
        ),
        (
            "U(0.5, -0.25, 1.5) q[1];",
            G::UGate {
                qubit: 1,
                theta: t,
                phi: f,
                lam: l,
            },
        ),
    ];
    assert_eq!(cases.len(), 33, "the 32 stdgates.inc gates and U");
    for (statement, expected) in cases {
        assert_eq!(import(&program(statement)).gates, [expected], "{statement}");
    }
}

#[test]
fn a_declared_gate_stays_a_declared_gate_whatever_its_name() {
    let qc = import(&program(
        "gate rzz(t) a, b { cx a, b; rz(t) b; cx a, b; }\nrzz(0.5) q[1], q[0];",
    ));
    match qc.gates.as_slice() {
        [G::Custom(call)] => {
            assert_eq!(call.name(), "rzz");
            assert_eq!(call.params(), [Fixed(0.5)]);
            assert_eq!(call.qubits(), [1, 0]);
        }
        other => panic!("{other:?}"),
    }
}

/// A declaration keeps its source text (`GateDefinition::declaration`) with
/// every run of carriage returns before a line feed dropped, as the
/// OpenQASM 2.0 importer keeps it.
#[test]
fn a_declarations_text_keeps_its_source_with_line_endings_normalised() {
    let qc = import(&format!(
        "{HEADER}qubit[1] q;\r\ngate g a {{\r\r\n  x a;\r\n}}\r\ng q[0];\r\n"
    ));
    let G::Custom(call) = &qc.gates[0] else {
        panic!("{:?}", qc.gates)
    };
    assert_eq!(call.definition().declaration(), "gate g a {\n  x a;\n}");
}

#[test]
fn measurements_are_assigned_to_bits() {
    assert_eq!(
        import(&program("h q[0];\nc[1] = measure q[0];")).gates,
        [G::H(0), G::Measure { qubit: 0, cbit: 1 }]
    );
    for statement in ["c = measure q;", "measure q -> c;"] {
        assert_eq!(
            import(&program(&format!("h q[0];\n{statement}"))).gates,
            [G::H(0), G::MeasureAll],
            "{statement}"
        );
    }
    assert_eq!(
        import(&program("measure q[2] -> c[0];")).gates,
        [G::Measure { qubit: 2, cbit: 0 }]
    );
}

#[test]
fn barriers() {
    assert_eq!(
        import(&program("barrier;\nbarrier q;\nbarrier q[2], q[0];")).gates,
        [
            G::Barrier(vec![]),
            G::Barrier(vec![]),
            G::Barrier(vec![2, 0])
        ]
    );
}

#[test]
fn comments_are_skipped_and_lines_still_counted() {
    let src = format!("{HEADER}/* a\nblock\ncomment */ qubit[1] q; // trailing\nh q[0]; /**/ x q[0];\nfoo q[0];\n");
    let (line, message) = rejected(&src);
    assert_eq!(line, 7, "{message}");
    assert!(message.contains("unsupported gate 'foo'"), "{message}");
    let qc = import(&src.replace("foo q[0];\n", ""));
    assert_eq!(qc.gates, [G::H(0), G::X(0)]);
}

// ─────────────────────────── Angle expressions ───────────────────────────

#[test]
fn constants_and_functions_are_evaluated_in_binary64() {
    use std::f64::consts::{E, PI, TAU};
    let cases = [
        ("pi", PI),
        ("π", PI),
        ("tau", TAU),
        ("τ", TAU),
        ("euler", E),
        ("ℇ", E),
        ("sin(0.5)", 0.5f64.sin()),
        ("cos(0.5)", 0.5f64.cos()),
        ("tan(0.5)", 0.5f64.tan()),
        ("arcsin(0.5)", 0.5f64.asin()),
        ("arccos(0.5)", 0.5f64.acos()),
        ("arctan(0.5)", 0.5f64.atan()),
        ("exp(0.5)", 0.5f64.exp()),
        ("log(0.5)", 0.5f64.ln()),
        ("sqrt(0.5)", 0.5f64.sqrt()),
    ];
    for (text, value) in cases {
        assert_eq!(angle(&format!("rx({text}) q[0];")), Fixed(value), "{text}");
    }
}

#[test]
fn number_literals() {
    let cases = [
        ("0x1F", 31.0),
        ("0XfF", 255.0),
        ("0o17", 15.0),
        ("0b101", 5.0),
        ("0B11", 3.0),
        ("1_000", 1000.0),
        ("1.5e-3", 1.5e-3),
        ("1.5E+3", 1500.0),
        (".5", 0.5),
        ("1.", 1.0),
        ("1_0.2_5", 10.25),
        ("2e3", 2000.0),
    ];
    for (text, value) in cases {
        assert_eq!(angle(&format!("rx({text}) q[0];")), Fixed(value), "{text}");
    }
}

#[test]
fn power_is_right_associative_and_binds_tighter_than_unary_minus() {
    let cases = [
        ("2.0 ** 3 ** 2", 512.0),
        ("(2.0 ** 3) ** 2", 64.0),
        ("-2.0 ** 2", -4.0),
        ("(-2.0) ** 2", 4.0),
        ("2.0 ** -1", 0.5),
        ("2.0 * 3 ** 2", 18.0),
        ("-3.0 * 2", -6.0),
    ];
    for (text, value) in cases {
        assert_eq!(angle(&format!("rx({text}) q[0];")), Fixed(value), "{text}");
    }
}

/// Expressions are evaluated as written: left to right, never reassociated
/// (in binary64, `(0.1 + 0.2) + 0.3` and `0.1 + (0.2 + 0.3)` differ).
#[test]
fn evaluation_follows_the_source() {
    assert_eq!(
        angle("rx(0.1 + 0.2 + 0.3) q[0];"),
        Fixed(0.6000000000000001)
    );
    assert_eq!(angle("rx(0.1 + (0.2 + 0.3)) q[0];"), Fixed(0.6));
    let qc = import(&format!(
        "{HEADER}input float[64] a;\nqubit[1] q;\nrx(a + 0.2 + 0.3) q[0];\nry(a + (0.2 + 0.3)) q[0];\n"
    ));
    assert_eq!(
        qc.assign_parameters(&[0.1]).unwrap().gates,
        [
            G::Rx {
                qubit: 0,
                theta: Fixed(0.6000000000000001)
            },
            G::Ry {
                qubit: 0,
                theta: Fixed(0.6)
            },
        ]
    );
}

#[test]
fn a_division_of_two_integer_expressions_is_rejected_with_a_hint() {
    for text in ["1/2", "(1 + 1)/2", "-1/2", "2 ** 2/3", "0x10/0b10"] {
        let (line, message) = rejected(&program(&format!("rx({text}) q[0];")));
        assert_eq!(line, 5, "{text}");
        assert!(
            message.contains("integer division in OpenQASM 3")
                && message.contains("1.0/2 instead of 1/2"),
            "{text}: {message}"
        );
    }
    for (text, value) in [
        ("1.0/2", 0.5),
        ("1/2.0", 0.5),
        ("1/2e0", 0.5),
        ("pi/2", std::f64::consts::FRAC_PI_2),
        ("2/pi", 2.0 / std::f64::consts::PI),
        ("3 * 1.0/2", 1.5),
    ] {
        assert_eq!(angle(&format!("rx({text}) q[0];")), Fixed(value), "{text}");
    }
}

#[test]
fn a_constant_division_by_zero_is_reported_on_the_line_of_the_division() {
    let (line, message) = rejected(&program("rx(1.0\n/ 0.0) q[0];"));
    assert_eq!(line, 6);
    assert!(message.contains("division by zero"), "{message}");
}

/// A constant angle must be finite; only its final value is checked, so a
/// non-finite intermediate result that yields a finite angle is accepted.
#[test]
fn a_constant_angle_must_be_finite_but_its_intermediates_need_not_be() {
    let (line, message) = rejected(&program("rx(exp(1000.0)) q[0];"));
    assert_eq!(line, 5);
    assert!(message.contains("non-finite"), "{message}");
    assert_eq!(angle("rx(1.0/exp(1000.0)) q[0];"), Fixed(0.0));
    assert_eq!(
        angle("rx(arctan(exp(1000.0))) q[0];"),
        Fixed(std::f64::consts::FRAC_PI_2)
    );
    let (_, message) = rejected(&program("rx(1e400) q[0];"));
    assert!(
        message.contains("'1e400' is out of range for binary64"),
        "{message}"
    );
}

#[test]
fn expressions_of_inputs_are_evaluated_when_bound() {
    let qc = import(&format!(
        "{HEADER}input float[64] a;\nqubit[1] q;\nrx(1.0/a) q[0];\nry(exp(a)) q[0];\n"
    ));
    assert!(matches!(
        qc.gates[0],
        G::Rx {
            theta: GateParam::Expr(_),
            ..
        }
    ));
    assert_eq!(
        qc.assign_parameters(&[0.0]).unwrap_err(),
        CircuitError::DivisionByZero
    );
    assert_eq!(
        qc.assign_parameters(&[1000.0]).unwrap_err(),
        CircuitError::NonFiniteParam
    );
    assert_eq!(
        qc.assign_parameters(&[2.0]).unwrap().gates,
        [
            G::Rx {
                qubit: 0,
                theta: Fixed(0.5)
            },
            G::Ry {
                qubit: 0,
                theta: Fixed(2.0f64.exp())
            },
        ]
    );
}

#[test]
fn a_gate_body_is_over_the_gates_parameters_and_checked_when_called() {
    let src = program("gate g(t) a { rx(1.0/t) a; }\ng(2.0) q[0];\ng(0.0) q[1];");
    let (line, message) = rejected(&src);
    assert_eq!(line, 7);
    assert!(
        message.contains("in gate 'g'") && message.contains("division by zero"),
        "{message}"
    );
    let qc = import(&src.replace("g(0.0) q[1];\n", ""));
    let G::Custom(call) = &qc.gates[0] else {
        panic!("{:?}", qc.gates)
    };
    assert_eq!(
        call.expand(&[]).unwrap(),
        [G::Rx {
            qubit: 0,
            theta: Fixed(0.5)
        }]
    );
}

/// The largest and deepest expressions the profile reads go through every
/// stage — parsing, binding, printing, cloning, dropping — without deep
/// recursion: a 999 999-node sum of an input, and a gate body nested to the
/// depth limit.
#[test]
fn adversarial_expressions_survive_every_stage() {
    let sum = format!("rx(a{}) q[0];", " + a".repeat(MAX_EXPR_NODES / 2 - 1));
    let nested = format!(
        "gate g(t) b {{ rz({}t{}) b; }}\ng(a) q[0];",
        "sin(".repeat(64),
        ")".repeat(64)
    );
    let src = format!("{HEADER}input float[64] a;\nqubit[1] q;\n{sum}\n{nested}\n");
    let qc = import(&src);
    assert!(matches!(
        qc.gates[0],
        G::Rx {
            theta: GateParam::Expr(_),
            ..
        }
    ));
    let mut sum = 0.5;
    for _ in 1..MAX_EXPR_NODES / 2 {
        sum += 0.5;
    }
    // The bound circuit, as its instructions expand: the export renames the
    // gate's formal `t` (a stdgates.inc gate), so the definitions differ.
    let expanded = |qc: &ParameterizedCircuit| -> Vec<G> {
        let bound = qc.assign_parameters(&[0.5]).unwrap();
        let G::Custom(call) = &bound.gates[1] else {
            panic!("{:?}", bound.gates[1])
        };
        let mut gates = vec![bound.gates[0].clone()];
        gates.extend(call.expand(&[]).unwrap());
        gates
    };
    let expected = expanded(&qc);
    assert_eq!(
        expected[0],
        G::Rx {
            qubit: 0,
            theta: Fixed(sum)
        }
    );
    let copy = qc.clone();
    let out = canonical(&copy);
    assert!(out.len() > MAX_EXPR_NODES, "{}", out.len());
    drop(copy);
    assert_eq!(expanded(&import(&out)), expected);
    drop(qc);
}

// ───────────────────────────── Rejected ──────────────────────────────────

/// Each construct outside the profile, and the error that names it on its
/// line (line 5 is the first statement after [`program`]'s preamble).
#[test]
fn constructs_outside_the_profile_are_rejected_on_their_line() {
    let cases: &[(&str, usize, &str)] = &[
        // Control flow and non-unitary operations.
        ("if (c[0]) x q[0];", 5, "'if' statements are not supported"),
        ("for int i in [0:2] { x q[0]; }", 5, "'for' is not supported"),
        ("while (true) { x q[0]; }", 5, "'while' is not supported"),
        ("switch (1) { }", 5, "'switch' is not supported"),
        ("reset q[0];", 5, "'reset' is not supported"),
        // Global phase and modifiers, at the top level and in bodies.
        ("gphase(0.5);", 5, "'gphase' is not supported"),
        ("ctrl @ x q[0], q[1];", 5, "gate modifiers ('ctrl @')"),
        ("negctrl @ x q[0], q[1];", 5, "gate modifiers ('negctrl @')"),
        ("inv @ s q[0];", 5, "gate modifiers ('inv @')"),
        ("pow(2) @ s q[0];", 5, "gate modifiers ('pow @')"),
        (
            "gate g a {\n  gphase(0.1);\n}",
            6,
            "'gphase' is not supported (inside the body of gate 'g')",
        ),
        (
            "gate g a, b {\n  ctrl @ x a, b;\n}",
            6,
            "gate modifiers ('ctrl @') are not supported (inside the body of gate 'g')",
        ),
        // Subroutines, pulses, timing.
        ("def f(qubit a) { x a; }", 5, "'def' is not supported"),
        ("extern f(float[64]) -> float[64];", 5, "'extern' is not supported"),
        ("defcal x $0 { }", 5, "'defcal' is not supported"),
        ("cal { }", 5, "'cal' is not supported"),
        ("defcalgrammar \"openpulse\";", 5, "'defcalgrammar' is not supported"),
        ("box { x q[0]; }", 5, "'box' is not supported: the profile has no timing"),
        ("delay[100ns] q[0];", 5, "'delay' is not supported"),
        ("stretch s;", 5, "'stretch' is not supported"),
        ("rx(durationof({x q[0];})) q[0];", 5, "'durationof' is not supported"),
        ("rx(100ns) q[0];", 5, "duration literals ('…ns') are not supported"),
        // Other classical declarations and computation.
        ("let a = q[0];", 5, "'let' is not supported"),
        ("const float[64] a = 1.0;", 5, "'const' is not supported"),
        ("output bit[3] r;", 5, "'output' is not supported"),
        ("int[32] i;", 5, "classical type 'int' is not supported"),
        ("float[64] f;", 5, "classical type 'float' is not supported"),
        ("angle[20] a;", 5, "classical type 'angle' is not supported"),
        ("bool b;", 5, "classical type 'bool' is not supported"),
        ("array[float[64], 2] a;", 5, "classical type 'array' is not supported"),
        ("input int[32] n;", 5, "inputs of type 'int' are not supported"),
        ("input float[32] a;", 5, "only float[64] inputs are supported"),
        ("c += 1;", 5, "assignments ('+=') other than measurement are not supported"),
        ("c[0] = 1;", 5, "assignments other than measurement are not supported"),
        ("bit d = measure q[0];", 5, "a declaration initialised by a measurement"),
        ("bit d = 1;", 5, "initialising a bit is not supported"),
        ("measure q[0];", 5, "a measurement must be assigned to bits"),
        ("measure q -> c[0];", 5, "measure size mismatch: 3 qubit(s) -> 1 classical bit(s)"),
        ("true;", 5, "'true' is not a statement"),
        // OpenQASM 2.0 and other syntax.
        ("qreg r[2];", 5, "'qreg' is OpenQASM 2.0 syntax"),
        ("creg r[2];", 5, "'creg' is OpenQASM 2.0 syntax"),
        ("opaque g a;", 5, "'opaque' declarations are not supported"),
        ("pragma foo;", 5, "pragmas are not supported"),
        ("@bind x q[0];", 5, "annotations ('@…') are not supported"),
        ("#pragma foo", 5, "pragmas and directives ('#…') are not supported"),
        ("{ x q[0]; }", 5, "blocks ('{ … }') are not supported"),
        ("x $0;", 5, "physical qubits ('$0') are not supported"),
        ("include \"qelib1.inc\";", 5, "only \"stdgates.inc\" can be included"),
        ("include \"stdgates.inc\";", 5, "\"stdgates.inc\" is included twice"),
        ("OPENQASM 3.0;", 5, "must be the program's first statement"),
        // Operators, literals and names outside angle expressions.
        ("rx(+0.5) q[0];", 5, "unary '+' is not OpenQASM 3"),
        ("rx(2 ^ 3) q[0];", 5, "the operator '^' is bitwise xor in OpenQASM 3"),
        ("rx(5.0 % 2) q[0];", 5, "the operator '%' (mod) is not supported"),
        ("rx(1 << 2) q[0];", 5, "the operator '<<' is not supported"),
        ("rx(~1) q[0];", 5, "the operator '~' is not supported"),
        ("rx(float(1)) q[0];", 5, "casts ('float(…)') are not supported"),
        ("rx(true) q[0];", 5, "boolean literals are not supported"),
        ("rx(\"01\") q[0];", 5, "bitstring and string literals are not supported"),
        ("rx(2im) q[0];", 5, "imaginary literals are not supported"),
        ("rx(mod(1.0)) q[0];", 5, "the function 'mod' is not supported"),
        ("rx(sin(1.0, 2.0)) q[0];", 5, "'sin' takes one argument"),
        ("rx(a) q[0];", 5, "unknown identifier 'a' in expression"),
        ("rx(q) q[0];", 5, "'q' is a qubit register, not a value"),
        ("rx(1a) q[0];", 5, "letters directly after the digits"),
        ("rx(0x) q[0];", 5, "invalid number literal"),
        ("x q[0]; ?", 5, "unexpected character '?'"),
        // Operands.
        ("x q[0:1];", 5, "slices of 'q' are not supported"),
        ("x q[{0, 1}];", 5, "index sets of 'q' are not supported"),
        ("x q[-1];", 5, "negative indices of 'q' are not supported"),
        ("x q[3];", 5, "index 3 out of range for register 'q' of size 3"),
        ("x q ++ q;", 5, "register concatenation ('++') is not supported"),
        ("rx(0.5) c[0];", 5, "'c' is a bit register, not a quantum register"),
        ("x r[0];", 5, "undeclared quantum register 'r'"),
        ("cx q[0], q[0];", 5, "requires distinct qubits, got (0, 0)"),
        ("x q[0], q[1];", 5, "gate 'x' expects 1 argument(s), found 2"),
        ("rx q[0];", 5, "gate 'rx' expects 1 parameter(s), found 0"),
        ("c[0] = measure q[0];\nx q[0];", 6, "gate acts on qubit 0 after it was measured"),
        ("foo q[0];", 5, "unsupported gate 'foo'"),
        // Declarations.
        ("qubit[2] q;", 5, "'q' is already declared as a qubit register"),
        ("qubit[0] r;", 5, "register 'r' has size 0"),
        ("qubit[2] pi;", 5, "'pi' is reserved in OpenQASM 3 and cannot name a qubit register"),
        ("qubit r;\nx r[0];", 6, "'r' is a single qubit, not a register"),
        ("gate x a { }", 5, "'x' is already defined by stdgates.inc and cannot name a gate"),
        ("gate g { }", 5, "gate 'g' must act on at least one qubit"),
        ("gate g a {\n  g a;\n}", 6, "gate 'g' calls itself"),
        ("gate g(t, t) a { }", 5, "parameter 't' of gate 'g' is declared twice"),
        ("gate g(t) t { }", 5, "qubit argument 't' of gate 'g' is declared twice"),
        ("gate g(pi) a { }", 5, "'pi' is reserved in OpenQASM 3 and cannot name a parameter of gate 'g'"),
        ("gate g a {\n  rx(0.5) a[0];\n}", 6, "argument 'a' of gate 'g' cannot be indexed"),
        ("gate g a {\n  barrier a;\n}", 6, "'barrier' is not allowed inside the body of gate 'g'"),
        ("gate g a {\n  reset a;\n}", 6, "'reset' is not allowed inside the body of gate 'g'"),
        ("gate g a {\n  x b;\n}", 6, "'b' is not a qubit argument of gate 'g'"),
        (
            "gate g(t) a {\n  t a;\n}",
            6,
            "'t' is a parameter of gate 'g', not a gate: inside the body, the parameter shadows the gate",
        ),
        ("gate g h {\n  h h;\n}", 6, "'h' is a qubit argument of gate 'g', not a gate"),
        ("gate g a {\n  rx(t) a;\n}", 6, "unknown identifier 't' in expression (in the body of gate 'g')"),
        (
            "input float[64] a;\ngate g b {\n  rx(a) b;\n}",
            7,
            "'a' is a global input, which a gate body cannot use",
        ),
        ("input float[64] a;\ninput float[64] a;", 6, "'a' is already declared as an input"),
        // Lexical errors.
        ("x q[0]; /* never closed", 5, "unterminated block comment"),
        ("include \"stdgates.inc;", 5, "unterminated string literal"),
        ("x q[0]", 5, "unexpected end of input, expected ',' or ';'"),
    ];
    for &(statements, line, fragment) in cases {
        let (got_line, message) = rejected(&program(statements));
        assert!(
            message.contains(fragment),
            "{statements:?}: {message:?} lacks {fragment:?}"
        );
        assert_eq!(got_line, line, "{statements:?}: {message}");
    }
}

#[test]
fn the_version_must_be_3() {
    let (line, message) = rejected("OPENQASM 2.0;\nqreg q[1];\n");
    assert_eq!(line, 1);
    assert!(
        message.contains("import OpenQASM 2.0 with from_qasm2"),
        "{message}"
    );
    let (_, message) = rejected("OPENQASM 3.1;\n");
    assert!(message.contains("found '3.1'"), "{message}");
}

#[test]
fn stdgates_gates_need_the_include() {
    let (line, message) = rejected("OPENQASM 3.0;\nqubit q;\nh q;\n");
    assert_eq!(line, 3);
    assert!(
        message.contains("it is a stdgates.inc gate, but \"stdgates.inc\" is not included"),
        "{message}"
    );
    // `U` is a builtin, included or not.
    assert_eq!(
        import("OPENQASM 3.0;\nqubit q;\nU(0.5, 0, 0) q;\n").gates,
        [G::UGate {
            qubit: 0,
            theta: Fixed(0.5),
            phi: Fixed(0.0),
            lam: Fixed(0.0)
        }]
    );
}

// ───────────────────────────── Import budgets ────────────────────────────

const MAX_SOURCE_BYTES: usize = 32 << 20;
const MAX_TOKEN_BYTES: usize = 4096;
const MAX_INPUTS: usize = 100_000;
const MAX_DECLARATIONS: usize = 10_000;
const MAX_EXPR_NODES: usize = 1_000_000;
const MAX_REGISTER_BITS: usize = 1_000_000;

#[test]
fn the_source_size_is_bounded() {
    let at = |len: usize| format!("{HEADER}//{}\n", "x".repeat(len - HEADER.len() - 3));
    let src = at(MAX_SOURCE_BYTES);
    assert_eq!(src.len(), MAX_SOURCE_BYTES);
    assert!(ParameterizedCircuit::from_qasm3(&src).is_ok());
    let (line, message) = rejected(&at(MAX_SOURCE_BYTES + 1));
    assert_eq!(line, 1);
    assert!(message.contains("exceeds MAX_SOURCE_BYTES"), "{message}");
}

#[test]
fn tokens_are_bounded() {
    let name = "a".repeat(MAX_TOKEN_BYTES);
    assert_eq!(
        import(&format!("{HEADER}qubit[1] {name};\nh {name}[0];\n")).num_qubits,
        1
    );
    let (line, message) = rejected(&format!("{HEADER}qubit[1] {name}b;\n"));
    assert_eq!(line, 3);
    assert!(
        message.contains("token longer than MAX_TOKEN_BYTES"),
        "{message}"
    );
}

#[test]
fn inputs_are_bounded() {
    let mut src = String::from(HEADER);
    for i in 0..MAX_INPUTS {
        src.push_str(&format!("input float[64] a{i};\n"));
    }
    assert_eq!(import(&src).num_params, MAX_INPUTS);
    src.push_str("input float[64] one_more;\n");
    let (line, message) = rejected(&src);
    assert_eq!(line, MAX_INPUTS + 3);
    assert!(
        message.contains("more than MAX_INPUTS (100000)"),
        "{message}"
    );
}

#[test]
fn declarations_are_bounded() {
    let mut src = String::from(HEADER);
    for i in 0..MAX_DECLARATIONS {
        src.push_str(&format!("gate g{i} a {{ }}\n"));
    }
    assert!(ParameterizedCircuit::from_qasm3(&src).is_ok());
    src.push_str("gate one_more a { }\n");
    let (line, message) = rejected(&src);
    assert_eq!(line, MAX_DECLARATIONS + 3);
    assert!(
        message.contains("more than MAX_DECLARATIONS (10000)"),
        "{message}"
    );
}

#[test]
fn expression_depth_is_bounded() {
    let nested = |open: &str, close: &str, depth: usize| {
        format!("rx({}0.5{}) q[0];", open.repeat(depth), close.repeat(depth))
    };
    for (open, close) in [("(", ")"), ("-", ""), ("sin(", ")"), ("1.0 ** ", "")] {
        assert!(
            ParameterizedCircuit::from_qasm3(&program(&nested(open, close, 64))).is_ok(),
            "{open}"
        );
        let (line, message) = rejected(&program(&nested(open, close, 65)));
        assert_eq!(line, 5);
        assert!(
            message.contains("expression nested too deeply (max 64)"),
            "{open}: {message}"
        );
    }
    // A long flat chain does not nest.
    let chain = format!("rx(0.5{}) q[0];", " + 0.5".repeat(10_000));
    assert!(ParameterizedCircuit::from_qasm3(&program(&chain)).is_ok());
}

#[test]
fn expression_size_is_bounded() {
    // n terms of a sum are 2n − 1 nodes.
    let sum = |terms: usize| format!("rx(0.0{}) q[0];", " + 0.0".repeat(terms - 1));
    assert!(ParameterizedCircuit::from_qasm3(&program(&sum(MAX_EXPR_NODES / 2))).is_ok());
    let (line, message) = rejected(&program(&sum(MAX_EXPR_NODES / 2 + 1)));
    assert_eq!(line, 5);
    assert!(message.contains("more than 1000000 nodes"), "{message}");
}

#[test]
fn the_programs_expression_nodes_are_bounded() {
    // 999 999 nodes per statement: four fit in 4 000 000, a fifth does not.
    let statement = format!("rx(0.0{}) q[0];\n", "+0.0".repeat(MAX_EXPR_NODES / 2 - 1));
    let (line, message) = rejected(&program(&statement.repeat(5)));
    assert_eq!(line, 9);
    assert!(message.contains("MAX_PROGRAM_NODES (4000000"), "{message}");
    assert!(ParameterizedCircuit::from_qasm3(&program(&statement.repeat(4))).is_ok());
}

#[test]
fn register_sizes_are_bounded() {
    let (line, message) = rejected(&format!("{HEADER}qubit[{}] q;\n", MAX_REGISTER_BITS + 1));
    assert_eq!(line, 3);
    assert!(
        message.contains("total qubit count would exceed MAX_REGISTER_BITS (1000000)"),
        "{message}"
    );
    let (line, message) = rejected(&format!("{HEADER}qubit[600000] a;\nqubit[400001] b;\n"));
    assert_eq!(line, 4);
    assert!(
        message.contains("register 'b' has size 400001"),
        "{message}"
    );
    let (_, message) = rejected(&format!("{HEADER}bit[{}] c;\n", MAX_REGISTER_BITS + 1));
    assert!(message.contains("total classical bit count"), "{message}");
    assert_eq!(
        import(&format!("{HEADER}qubit[{MAX_REGISTER_BITS}] q;\n")).num_qubits,
        MAX_REGISTER_BITS
    );
}

#[test]
fn instructions_after_broadcasting_are_bounded() {
    let src = format!("{HEADER}qubit[{MAX_REGISTER_BITS}] q;\nx q;\nx q;\nx q;\nx q;\nx q[0];\n");
    let (line, message) = rejected(&src);
    assert_eq!(line, 8);
    assert!(
        message.contains("more than MAX_INSTRUCTIONS (4000000)"),
        "{message}"
    );
}

/// Declarations whose expansion doubles at each level: `g<k>` visits
/// 3·2^k − 2 statements.
fn doubling(levels: usize) -> String {
    let mut src = String::from("gate g0(t) a { rx(t) a; }\n");
    for k in 1..=levels {
        src.push_str(&format!(
            "gate g{k}(t) a {{ g{}(t) a; g{}(t) a; }}\n",
            k - 1,
            k - 1
        ));
    }
    src
}

#[test]
fn declared_gates_are_bounded_in_nesting_and_expansion() {
    let mut chain = String::from("gate g0 a { x a; }\n");
    for k in 1..64 {
        chain.push_str(&format!("gate g{k} a {{ g{} a; }}\n", k - 1));
    }
    assert!(ParameterizedCircuit::from_qasm3(&program(&chain)).is_ok());
    chain.push_str("gate g64 a { g63 a; }");
    let (line, message) = rejected(&program(&chain));
    assert_eq!(line, 5 + 64);
    assert!(
        message.contains("gate 'g64' nests gate calls more than 64 levels deep"),
        "{message}"
    );

    // 3·2^18 − 2 = 786 430 statements fit in 1 000 000; 3·2^19 − 2 do not.
    assert!(ParameterizedCircuit::from_qasm3(&program(&doubling(18))).is_ok());
    let (line, message) = rejected(&program(&doubling(19)));
    assert_eq!(line, 5 + 19);
    assert!(
        message.contains("gate 'g19' expands to more than 1000000 instructions"),
        "{message}"
    );
}

#[test]
fn the_expansion_of_all_calls_is_bounded() {
    // 25 calls of g18 expand 19 660 750 statements; a 26th passes 20 000 000.
    let calls = "g18(a) q[0];\n".repeat(25);
    let src = format!(
        "{HEADER}input float[64] a;\nqubit[1] q;\n{}{calls}",
        doubling(18)
    );
    assert!(ParameterizedCircuit::from_qasm3(&src).is_ok());
    let (line, message) = rejected(&format!("{src}g18(a) q[0];\n"));
    assert_eq!(line, 4 + 19 + 25 + 1);
    assert!(
        message.contains("would expand more than 20000000 instructions"),
        "{message}"
    );
}

// ─────────────────────────── Canonical output ────────────────────────────

/// The canonical layout: header, inputs, definitions (callees first, in order
/// of first use), registers, statements. `u` is written as `U`; `rzz` (which
/// `stdgates.inc` lacks) through a definition the output carries.
#[test]
fn the_canonical_layout() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
               gate inner(t) a { rz(t/2) a; }\n\
               gate outer(t) a,b { inner(t) b; cx a,b; }\n\
               qreg q[3];\nouter(0.25) q[2],q[0];\n";
    let G::Custom(call) = ParameterizedCircuit::from_qasm2(src).unwrap().gates[0].clone() else {
        unreachable!()
    };
    let mut qc = ParameterizedCircuit::new(3).rzz(0, 1, Param(1)).h(2);
    qc.try_push(G::Custom(
        call.with_arguments(vec![Param(0)], vec![1, 2]).unwrap(),
    ))
    .unwrap();
    let qc = qc
        .push(G::UGate {
            qubit: 0,
            theta: Fixed(0.5),
            phi: Param(0),
            lam: Fixed(-1.5),
        })
        .barrier()
        .measure(0, 0)
        .measure(2, 1);
    assert_eq!(
        canonical(&qc),
        "OPENQASM 3.0;
include \"stdgates.inc\";
input float[64] theta_0;
input float[64] theta_1;
gate rzz(p0) a, b {
  cx a, b;
  rz(p0) b;
  cx a, b;
}
gate inner(t_) a {
  rz(t_/2.0) a;
}
gate outer(t_) a, b {
  inner(t_) b;
  cx a, b;
}
qubit[3] q;
bit[2] c;
rzz(theta_1) q[0], q[1];
h q[2];
outer(theta_0) q[1], q[2];
U(0.5, theta_0, -1.5) q[0];
barrier q;
c[0] = measure q[0];
c[1] = measure q[2];
"
    );
}

#[test]
fn a_full_measurement_is_one_statement() {
    let out = canonical(&ParameterizedCircuit::new(2).h(0).measure_all());
    assert!(
        out.ends_with("qubit[2] q;\nbit[2] c;\nh q[0];\nc = measure q;\n"),
        "{out}"
    );
    // With a wider bit register, the measurement is written bit by bit.
    let out = canonical(&ParameterizedCircuit::new(2).measure(1, 2).measure_all());
    assert!(
        out.ends_with(
            "bit[3] c;\nc[2] = measure q[1];\nc[0] = measure q[0];\nc[1] = measure q[1];\n"
        ),
        "{out}"
    );
}

#[test]
fn numbers_are_the_shortest_decimal_that_reads_back_and_keep_negative_zero() {
    let cases = [
        (0.1, "0.1"),
        (0.30000000000000004, "0.30000000000000004"),
        (1.0, "1.0"),
        (-2.5, "-2.5"),
        (1e-5, "0.00001"),
        (1.5e-7, "1.5e-7"),
        (123456789012345.6, "123456789012345.6"),
        (1e16, "1.0e16"),
        (f64::MAX, "1.7976931348623157e308"),
        (5e-324, "5.0e-324"),
        (-0.0, "-0.0"),
    ];
    for (value, text) in cases {
        let out = canonical(&ParameterizedCircuit::new(1).rz(0, value));
        assert!(
            out.ends_with(&format!("rz({text}) q[0];\n")),
            "{value}: {out}"
        );
        match import(&out).gates.as_slice() {
            [G::Rz {
                theta: Fixed(read), ..
            }] => assert_eq!(read.to_bits(), value.to_bits(), "{text}"),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn expressions_are_written_with_the_fewest_parentheses() {
    let x = |i| ParamExpr::param(i);
    let cases = [
        (x(0) - (x(1) - x(2)), "theta_0 - (theta_1 - theta_2)"),
        ((x(0) - x(1)) - x(2), "theta_0 - theta_1 - theta_2"),
        ((x(0) + x(1)) * x(2), "(theta_0 + theta_1)*theta_2"),
        (x(0) / (x(1) * x(2)), "theta_0/(theta_1*theta_2)"),
        (-(x(0).pow(2.0)), "-theta_0**2.0"),
        ((-x(0)).pow(2.0), "(-theta_0)**2.0"),
        (x(0).pow(x(1).pow(x(2))), "theta_0**theta_1**theta_2"),
        (x(0).pow(x(1)).pow(x(2)), "(theta_0**theta_1)**theta_2"),
        (x(0).pow(-x(1)), "theta_0**-theta_1"),
        (-x(0) * x(1), "-theta_0*theta_1"),
        (x(0) * -x(1), "theta_0*-theta_1"),
        (x(0) + -1.5, "theta_0 + -1.5"),
        (-(x(0) + x(1)), "-(theta_0 + theta_1)"),
        (
            x(0).apply(Function::Sin).pow(2.0) + x(1).apply(Function::Ln),
            "sin(theta_0)**2.0 + log(theta_1)",
        ),
    ];
    for (expr, text) in cases {
        let mut qc = ParameterizedCircuit::new(1);
        let theta = qc.add_expr(expr).unwrap();
        let qc = qc.rz(0, theta).rx(0, Param(2));
        let out = canonical(&qc);
        assert!(
            out.contains(&format!("rz({text}) q[0];\n")),
            "{text}: {out}"
        );
        // The text reads back as the same expression: equal bindings.
        let values = [0.3, 0.7, 1.1];
        assert_eq!(
            import(&out).assign_parameters(&values).unwrap(),
            qc.assign_parameters(&values).unwrap(),
            "{text}"
        );
    }
}

#[test]
fn default_names_are_written_as_inputs() {
    let qc = ParameterizedCircuit::new(1).rx(0, Param(0)).rz(0, Param(2));
    let out = canonical(&qc);
    assert!(
        out.contains(
            "input float[64] theta_0;\ninput float[64] theta_1;\ninput float[64] theta_2;\n"
        ),
        "{out}"
    );
    assert_eq!(import(&out).param_names(), qc.param_names());
}

#[test]
fn binding_first_writes_numbers_only() {
    let qc = import(&format!(
        "{HEADER}input float[64] a;\nqubit[1] q;\nrx(2.0*a) q[0];\n"
    ));
    let out = qc.to_qasm3_with_params(&[0.25]).unwrap();
    assert_eq!(out, format!("{HEADER}qubit[1] q;\nrx(0.5) q[0];\n"));
    assert_eq!(
        qc.to_qasm3_with_params(&[]).unwrap_err(),
        CircuitError::WrongNumberOfParams {
            expected: 1,
            got: 0
        }
    );
}

/// A composed circuit keeps its own parameter names: the gates of another
/// circuit refer to parameters by index.
#[test]
fn parameter_names_belong_to_the_circuit() {
    let named = import(&format!(
        "{HEADER}input float[64] a;\ninput float[64] b;\nqubit[1] q;\nrx(a) q[0];\n"
    ));
    let mut qc = named.clone();
    qc.try_push(G::Ry {
        qubit: 0,
        theta: Param(1),
    })
    .unwrap();
    qc.try_push(G::Rz {
        qubit: 0,
        theta: Param(3),
    })
    .unwrap();
    assert_eq!(qc.param_names(), ["a", "b", "theta_2", "theta_3"]);
    let out = canonical(&qc);
    assert!(out.contains("ry(b) q[0];\nrz(theta_3) q[0];\n"), "{out}");
    assert_eq!(import(&out).param_names(), qc.param_names());
}

// ─────────────────────────────── Renaming ────────────────────────────────

#[test]
fn names_that_would_clash_are_renamed() {
    // A declared gate named like a register, and one like a stdgates.inc gate
    // (legal without the include).
    let qc = import(
        "OPENQASM 3.0;\ngate q a { U(0.5, 0, 0) a; }\ngate h a { q a; }\nqubit[1] r;\nh r[0];\nq r[0];\n",
    );
    let out = canonical(&qc);
    assert!(
        out.contains("gate q a {\n  U(0.5, 0.0, 0.0) a;\n}\n"),
        "{out}"
    );
    assert!(out.contains("gate h_ a {\n  q a;\n}\n"), "{out}");
    assert!(
        out.contains("qubit[1] q_1;\nh_ q_1[0];\nq q_1[0];\n"),
        "{out}"
    );

    // An input named like a register.
    let out = canonical(&import(&format!(
        "{HEADER}input float[64] q;\nqubit[1] r;\nrx(q) r[0];\n"
    )));
    assert!(out.contains("qubit[1] q_1;\nrx(q) q_1[0];\n"), "{out}");
}

/// A native `rzz` and a declared `rzz` are different gates: both are written,
/// the one used first keeps the name.
#[test]
fn a_native_and_a_declared_rzz_are_kept_apart() {
    let declared = import(&program(
        "gate rzz(t) a, b { rx(t) a; ry(t) b; }\nrzz(0.5) q[0], q[1];",
    ));
    let mut qc = declared.clone();
    qc.try_push(G::Rzz {
        q0: 1,
        q1: 2,
        theta: Fixed(0.25),
    })
    .unwrap();
    let out = canonical(&qc);
    assert!(
        out.contains("gate rzz(t_) a, b {\n  rx(t_) a;\n  ry(t_) b;\n}\n"),
        "{out}"
    );
    assert!(
        out.contains("gate rzz_1(p0) a, b {\n  cx a, b;\n  rz(p0) b;\n  cx a, b;\n}\n"),
        "{out}"
    );
    assert!(
        out.contains("rzz(0.5) q[0], q[1];\nrzz_1(0.25) q[1], q[2];\n"),
        "{out}"
    );

    // In OpenQASM 2.0 the native one is qelib1.inc's `rzz`; the declared one
    // is renamed.
    let qasm2 = qc.to_qasm2_with_params(&[]).unwrap();
    assert!(
        qasm2.contains("gate rzz_(t_) a,b { rx(t_) a; ry(t_) b; }\n"),
        "{qasm2}"
    );
    assert!(
        qasm2.ends_with("rzz_(0.500000000000) q[0],q[1];\nrzz(0.250000000000) q[1],q[2];\n"),
        "{qasm2}"
    );
}

#[test]
fn names_openqasm3_cannot_take_are_renamed() {
    // OpenQASM 2.0 allows what OpenQASM 3 reserves or lacks.
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
               gate input(output) box,delay { rx(output) box; cx box,delay; }\n\
               qreg q[2];\ninput(0.5) q[0],q[1];\n";
    let out = canonical(&ParameterizedCircuit::from_qasm2(src).unwrap());
    assert!(
        out.contains(
            "gate input_(output_) box_, delay_ {\n  rx(output_) box_;\n  cx box_, delay_;\n}\n"
        ),
        "{out}"
    );
    assert!(out.ends_with("input_(0.5) q[0], q[1];\n"), "{out}");

    // An identifier longer than the importer reads is cut.
    let long = format!("g{}", "a".repeat(5000));
    let src = format!("OPENQASM 2.0;\ngate {long} x {{ U(0,0,0) x; }}\nqreg q[1];\n{long} q[0];\n");
    let out = canonical(&ParameterizedCircuit::from_qasm2(&src).unwrap());
    let written = out
        .lines()
        .find_map(|l| {
            l.strip_prefix("gate ")
                .and_then(|l| l.strip_suffix(" x_ {"))
        })
        .unwrap();
    assert!(
        written.len() <= MAX_TOKEN_BYTES && long.starts_with(written),
        "{}",
        written.len()
    );
}

// ──────────────────────────── Cross-dialect ──────────────────────────────

#[test]
fn an_openqasm2_declaration_is_printed_in_openqasm3() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
               gate g(t) a,b { rz(ln(t)^2) a; cu1(-t) a,b; u(t,0,pi) b; }\n\
               qreg q[2];\ng(0.5) q[0],q[1];\n";
    let out = canonical(&ParameterizedCircuit::from_qasm2(src).unwrap());
    assert!(
        out.contains(
            "gate g(t_) a, b {\n  rz(log(t_)**2.0) a;\n  cu1(-t_) a, b;\n  U(t_, 0.0, pi) b;\n}\n"
        ),
        "{out}"
    );
    assert!(
        out.contains("gate cu1(p0) a, b {\n  cp(p0) a, b;\n}\n"),
        "{out}"
    );
}

#[test]
fn an_openqasm2_body_with_a_barrier_cannot_be_written_in_openqasm3() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\ngate g a,b { h a; barrier a,b; cx a,b; }\nqreg q[2];\ng q[0],q[1];\n";
    let qc = ParameterizedCircuit::from_qasm2(src).unwrap();
    match qc.to_qasm3().unwrap_err() {
        CircuitError::GateNotExpressible { name, reason } => {
            assert_eq!(name, "g");
            assert!(reason.contains("barrier"), "{reason}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_openqasm3_declaration_is_printed_in_openqasm2() {
    let qc = import(&format!(
        "{HEADER}input float[64] θ;\n\
         gate rzz(Theta) _a, b {{ cx _a, b; rz(Theta ** 2 / 2.0 + log(2.0)) b; cx _a, b; }}\n\
         gate q(x) a {{ U(x, -x, tau) a; phase(euler) a; }}\n\
         gate wrap(t) a, b {{ rzz(t) a, b; q(-t) b; CX a, b; cphase(t) a, b; }}\n\
         qubit[2] r;\nbit[2] m;\nwrap(θ * 0.5) r[0], r[1];\nm = measure r;\n"
    ));
    let qasm2 = qc.to_qasm2_with_params(&[0.25]).unwrap();
    assert_eq!(
        qasm2,
        "OPENQASM 2.0;
include \"qelib1.inc\";
gate rzz_(pTheta) q_a,b { cx q_a,b; rz(pTheta^2.0/2.0 + ln(2.0)) b; cx q_a,b; }
gate q_1(x_) a { u(x_,-x_,6.283185307179586) a; p(2.718281828459045) a; }
gate wrap(t_) a,b { rzz_(t_) a,b; q_1(-t_) b; cx a,b; cp(t_) a,b; }
qreg q[2];
creg c[2];
wrap(0.125000000000) q[0],q[1];
measure q -> c;
"
    );
    // From there on it is an OpenQASM 2.0 program, re-emitted verbatim.
    let again = ParameterizedCircuit::from_qasm2(&qasm2).unwrap();
    assert_eq!(again.to_qasm2_with_params(&[]).unwrap(), qasm2);
}

#[test]
fn an_openqasm3_body_with_arcsin_cannot_be_written_in_openqasm2() {
    let qc = import(&program("gate g(t) a { rx(arcsin(t)) a; }\ng(0.5) q[0];"));
    let expected = CircuitError::GateNotExpressible {
        name: "g".to_string(),
        reason: "cannot be written in OpenQASM 2.0: it uses the function 'arcsin'".into(),
    };
    assert_eq!(qc.to_qasm2_with_params(&[]).unwrap_err(), expected);
    let bound: ConcreteCircuit = qc.assign_parameters(&[]).unwrap();
    assert_eq!(bound.try_to_qasm2().unwrap_err(), expected);
    assert_eq!(
        expected.to_string(),
        "declared gate 'g' cannot be written in OpenQASM 2.0: it uses the function 'arcsin'"
    );
    // OpenQASM 3 has the function.
    assert!(canonical(&qc).contains("rx(arcsin(t_)) a;"));
}

#[test]
#[should_panic(expected = "a declared gate OpenQASM 2.0 cannot express")]
fn to_qasm2_panics_where_try_to_qasm2_fails() {
    let qc = import(&program("gate g(t) a { rx(arccos(t)) a; }\ng(0.5) q[0];"));
    let _ = qc.assign_parameters(&[]).unwrap().to_qasm2();
}

// ──────────────────────────── Export budgets ─────────────────────────────

/// What the exporter writes, the importer reads: an export that would pass
/// one of the importer's budgets fails instead.
#[test]
fn the_export_stays_within_the_importers_budgets() {
    // Inputs.
    let at = ParameterizedCircuit::new(1).rx(0, Param(MAX_INPUTS - 1));
    assert_eq!(import(&at.to_qasm3().unwrap()).num_params, MAX_INPUTS);
    assert_eq!(
        export_limit(&ParameterizedCircuit::new(1).rx(0, Param(MAX_INPUTS))),
        "MAX_INPUTS"
    );

    // Register size, of qubits and of bits.
    let at = ParameterizedCircuit::new(MAX_REGISTER_BITS).h(0);
    assert_eq!(
        import(&at.to_qasm3().unwrap()).num_qubits,
        MAX_REGISTER_BITS
    );
    assert_eq!(
        export_limit(&ParameterizedCircuit::new(MAX_REGISTER_BITS + 1)),
        "MAX_REGISTER_BITS"
    );
    assert_eq!(
        export_limit(&ParameterizedCircuit::new(1).measure(0, MAX_REGISTER_BITS)),
        "MAX_REGISTER_BITS"
    );

    // Instructions as the importer pushes them: `c = measure q;` is one
    // measurement per qubit.
    let mut qc = ParameterizedCircuit::new(MAX_REGISTER_BITS);
    for _ in 0..4 {
        qc = qc.measure_all();
    }
    assert!(qc.to_qasm3().is_ok());
    assert_eq!(export_limit(&qc.measure_all()), "MAX_INSTRUCTIONS");

    // Expression nodes, one expression at a time: OpenQASM 2.0 does not
    // bound the size of an expression in a gate body.
    let src = format!(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\ngate g(t) a {{ rz(t{}) a; }}\nqreg q[1];\ng(0.5) q[0];\n",
        "+t".repeat(MAX_EXPR_NODES / 2)
    );
    let declared = ParameterizedCircuit::from_qasm2(&src).unwrap();
    assert_eq!(export_limit(&declared), "MAX_EXPR_NODES");
}

#[test]
fn the_export_counts_the_programs_expression_nodes() {
    // Five expressions of 900 001 nodes: 4 500 005 nodes in all.
    let mut qc = ParameterizedCircuit::new(1);
    for _ in 0..5 {
        let mut sum = ParamExpr::param(0);
        for _ in 0..450_000 {
            sum = sum + ParamExpr::param(0);
        }
        let theta = qc.add_expr(sum).unwrap();
        qc = qc.rz(0, theta);
    }
    assert_eq!(export_limit(&qc), "MAX_PROGRAM_NODES");
}

#[test]
fn the_export_counts_declarations_and_their_calls() {
    // Every definition counts, helpers included: 9 989 declared gates and the
    // twelve helpers are one more than the importer reads.
    let mut src = String::from("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n");
    for i in 0..MAX_DECLARATIONS - 11 {
        src.push_str(&format!("gate g{i} a {{ x a; }}\n"));
    }
    src.push_str("qreg q[5];\n");
    for i in 0..MAX_DECLARATIONS - 11 {
        src.push_str(&format!("g{i} q[0];\n"));
    }
    let declared = ParameterizedCircuit::from_qasm2(&src).unwrap();
    let helpers = |qc: ParameterizedCircuit| {
        qc.rzz(0, 1, 0.5)
            .rxx(0, 1, 0.5)
            .sxdg(0)
            .csx(0, 1)
            .cu1(0, 1, 0.5)
            .cu3(0, 1, 0.1, 0.2, 0.3)
            .u0(0, 0.5)
            .rccx(0, 1, 2)
            .rc3x(0, 1, 2, 3)
            .c3x(0, 1, 2, 3)
            .c3sqrtx(0, 1, 2, 3)
            .c4x(0, 1, 2, 3, 4)
    };
    assert_eq!(export_limit(&helpers(declared)), "MAX_DECLARATIONS");

    // The calls of declared gates, with helpers counted by their size: the
    // definition of c4x expands to 98 statements, so 250 000 calls pass
    // 20 000 000.
    let mut qc = ParameterizedCircuit::new(5);
    for _ in 0..250_000 {
        qc = qc.c4x(0, 1, 2, 3, 4);
    }
    assert_eq!(export_limit(&qc), "MAX_VALIDATED_EXPANSION");
}

/// A declaration that OpenQASM 2.0 nests within the importers' limits can
/// pass them in OpenQASM 3, where `c4x` becomes a call of a definition two
/// levels deep: `g<k>` nests k + 1 levels in OpenQASM 2.0 and k + 3 in 3.
#[test]
fn a_declaration_too_deep_once_written_in_openqasm3_is_not_expressible() {
    let chain = |top: usize| {
        let mut src = String::from(
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\ngate g0 a,b,c,d,e { c4x a,b,c,d,e; }\n",
        );
        for k in 1..=top {
            src.push_str(&format!(
                "gate g{k} a,b,c,d,e {{ g{} a,b,c,d,e; }}\n",
                k - 1
            ));
        }
        src.push_str(&format!("qreg q[5];\ng{top} q[0],q[1],q[2],q[3],q[4];\n"));
        ParameterizedCircuit::from_qasm2(&src).unwrap()
    };
    assert!(chain(61).to_qasm3().is_ok());
    match chain(62).to_qasm3().unwrap_err() {
        CircuitError::GateNotExpressible { name, reason } => {
            assert_eq!(name, "g62");
            assert!(reason.contains("more than 64 levels deep"), "{reason}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_declaration_too_large_once_written_in_openqasm3_is_not_expressible() {
    let body = "c4x a,b,c,d,e; ".repeat(20_000);
    let src = format!(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\ngate big a,b,c,d,e {{ {body}}}\nqreg q[5];\nbig q[0],q[1],q[2],q[3],q[4];\n"
    );
    match ParameterizedCircuit::from_qasm2(&src)
        .unwrap()
        .to_qasm3()
        .unwrap_err()
    {
        CircuitError::GateNotExpressible { name, reason } => {
            assert_eq!(name, "big");
            assert!(
                reason.contains("expands to more than 1000000 instructions"),
                "{reason}"
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_export_size_is_bounded() {
    let mut qc = ParameterizedCircuit::new(1);
    for _ in 0..1_200_000 {
        qc = qc.rz(0, 0.30000000000000004);
    }
    assert_eq!(export_limit(&qc), "MAX_SOURCE_BYTES");
}
