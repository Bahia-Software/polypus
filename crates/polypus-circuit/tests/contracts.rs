//! Enforcement tests for the inter-layer contracts owned by `polypus-circuit`
//! (see `docs/CONTRACTS.md`):
//!
//! - **C-2 · Gate vocabulary symmetry.** Every gate in the vocabulary survives
//!   the export → import → export round-trip byte-for-byte, and the imported
//!   instruction sequence matches the original. The QIR-vs-simulator unitary
//!   equivalence half of C-2 lives in `crates/polypus-sim/tests/contracts.rs`
//!   (it needs the simulator).
//! - **C-4 · Terminal measurement placement.** A gate acting on an
//!   already-measured qubit is rejected by the builder, the QASM importer and
//!   the QIR exporter (the simulator half lives in the `polypus-sim` tests).

use polypus_circuit::{CircuitError, GateInstruction, GateParam, Param, ParameterizedCircuit};

// ─────────────────────────── C-2 · round-trip ─────────────────────────────

/// One circuit exercising the whole vocabulary, including `cp` (audit item C2).
/// Free parameters cover the `Param` path; everything else is fixed.
fn full_vocabulary() -> ParameterizedCircuit {
    ParameterizedCircuit::new(3)
        .h(0)
        .x(1)
        .y(2)
        .z(0)
        .s(1)
        .t(2)
        .sdg(0)
        .tdg(1)
        .rx(0, 0.25)
        .ry(1, Param(0))
        .rz(2, -1.5)
        .u(0, 0.1, Param(1), 0.3)
        .cx(0, 1)
        .cz(1, 2)
        .rzz(0, 2, Param(0))
        .rxx(1, 2, 2.0)
        .cp(0, 1, 0.75)
        .barrier()
        .barrier_on(&[0, 2])
        .measure(0, 0)
        .measure(2, 1)
}

#[test]
fn c2_full_vocabulary_roundtrip_is_byte_stable() {
    let c = full_vocabulary();
    let qasm1 = c.to_qasm2_with_params(&[0.4, -0.9]).unwrap();
    assert!(
        qasm1.contains("cp(0.750000000000) q[0],q[1];"),
        "exporter must emit cp:\n{qasm1}"
    );

    let imported = ParameterizedCircuit::from_qasm2(&qasm1).unwrap();
    let qasm2 = imported.to_qasm2_with_params(&[]).unwrap();
    assert_eq!(
        qasm1, qasm2,
        "export → import → export must be a fixed point"
    );

    // The imported instruction sequence reproduces the (bound) original.
    let bound = c.assign_parameters(&[0.4, -0.9]).unwrap();
    assert_eq!(imported.gates, bound.gates);
}

/// The whole vocabulary, gate by gate: each single-gate circuit is a fixed
/// point under export → import → export. This is the "for each gate in the set"
/// half of the C-2 round-trip guarantee.
#[test]
fn c2_every_gate_roundtrips_individually() {
    let cases: Vec<(&str, ParameterizedCircuit)> = vec![
        ("h", ParameterizedCircuit::new(1).h(0)),
        ("x", ParameterizedCircuit::new(1).x(0)),
        ("y", ParameterizedCircuit::new(1).y(0)),
        ("z", ParameterizedCircuit::new(1).z(0)),
        ("s", ParameterizedCircuit::new(1).s(0)),
        ("t", ParameterizedCircuit::new(1).t(0)),
        ("sdg", ParameterizedCircuit::new(1).sdg(0)),
        ("tdg", ParameterizedCircuit::new(1).tdg(0)),
        ("rx", ParameterizedCircuit::new(1).rx(0, 0.3)),
        ("ry", ParameterizedCircuit::new(1).ry(0, 0.3)),
        ("rz", ParameterizedCircuit::new(1).rz(0, 0.3)),
        ("cx", ParameterizedCircuit::new(2).cx(0, 1)),
        ("cz", ParameterizedCircuit::new(2).cz(0, 1)),
        ("rzz", ParameterizedCircuit::new(2).rzz(0, 1, 0.3)),
        ("rxx", ParameterizedCircuit::new(2).rxx(0, 1, 0.3)),
        ("cp", ParameterizedCircuit::new(2).cp(0, 1, 0.3)),
        ("u3", ParameterizedCircuit::new(1).u(0, 0.1, 0.2, 0.3)),
        ("barrier", ParameterizedCircuit::new(2).h(0).barrier()),
        // A partial measurement (qubit 1 left unmeasured) so the importer does
        // not canonicalise a full q[k]->c[k] run into `measure_all`.
        ("measure", ParameterizedCircuit::new(2).h(0).measure(0, 0)),
        (
            "measure_all",
            ParameterizedCircuit::new(2).h(0).measure_all(),
        ),
    ];

    for (name, c) in cases {
        let qasm1 = c.to_qasm2_with_params(&[]).unwrap();
        let imported = ParameterizedCircuit::from_qasm2(&qasm1).unwrap();
        let qasm2 = imported.to_qasm2_with_params(&[]).unwrap();
        assert_eq!(
            qasm1, qasm2,
            "gate '{name}' is not a round-trip fixed point"
        );
        assert_eq!(
            imported.gates, c.gates,
            "gate '{name}' instruction sequence changed on round-trip"
        );
    }
}

/// `cp` specifically must import back to a `Cp` instruction (audit item C2:
/// it used to be missing from the importer, so it round-tripped to nothing).
#[test]
fn c2_cp_imports_to_cp_instruction() {
    let qasm = ParameterizedCircuit::new(2)
        .cp(0, 1, std::f64::consts::FRAC_PI_2)
        .to_qasm2_with_params(&[])
        .unwrap();
    let imported = ParameterizedCircuit::from_qasm2(&qasm).unwrap();
    match imported.gates.as_slice() {
        [GateInstruction::Cp {
            q0: 0,
            q1: 1,
            theta: GateParam::Fixed(v),
        }] => assert!((v - std::f64::consts::FRAC_PI_2).abs() < 1e-11),
        other => panic!("expected a single Cp, got {other:?}"),
    }
}

// ───────────────────────── C-4 · terminal measurement ─────────────────────

#[test]
fn c4_builder_rejects_gate_after_measure() {
    let mut qc = ParameterizedCircuit::new(1);
    qc.try_push(GateInstruction::Measure { qubit: 0, cbit: 0 })
        .unwrap();
    let err = qc.try_push(GateInstruction::X(0)).unwrap_err();
    assert_eq!(err, CircuitError::QubitAlreadyMeasured { qubit: 0 });
}

#[test]
fn c4_builder_rejects_two_qubit_gate_touching_measured_qubit() {
    let mut qc = ParameterizedCircuit::new(2);
    qc.try_push(GateInstruction::Measure { qubit: 1, cbit: 0 })
        .unwrap();
    // cx control=0 (unmeasured), target=1 (measured) -> reject on qubit 1.
    let err = qc.try_push(GateInstruction::Cx(0, 1)).unwrap_err();
    assert_eq!(err, CircuitError::QubitAlreadyMeasured { qubit: 1 });
}

#[test]
#[should_panic(expected = "after it was measured")]
fn c4_fluent_builder_panics_on_gate_after_measure() {
    let _ = ParameterizedCircuit::new(1).measure(0, 0).h(0);
}

#[test]
fn c4_barrier_and_remeasure_are_allowed_after_measure() {
    // Barrier on a measured qubit and re-measuring it are both legal.
    let qc = ParameterizedCircuit::new(2)
        .h(0)
        .measure(0, 0)
        .barrier()
        .measure(0, 0)
        .measure_all();
    assert_eq!(qc.num_qubits, 2);
}

#[test]
fn c4_importer_rejects_gate_after_measure_with_line() {
    let src = "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nx q[0];\n";
    match ParameterizedCircuit::from_qasm2(src) {
        Err(CircuitError::Parse { line, message }) => {
            assert_eq!(line, 5, "wrong line for {message:?}");
            assert!(
                message.contains("after it was measured"),
                "unexpected message: {message:?}"
            );
        }
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn c4_qir_exporter_rejects_gate_after_measure() {
    // Hand-assembled (bypasses the builder's own check) to reach the exporter.
    let qc = ParameterizedCircuit {
        num_qubits: 1,
        num_params: 0,
        gates: vec![
            GateInstruction::Measure { qubit: 0, cbit: 0 },
            GateInstruction::X(0),
        ],
    };
    let err = qc.to_qir_with_params(&[]).unwrap_err();
    assert_eq!(err, CircuitError::QubitAlreadyMeasured { qubit: 0 });
}

/// A terminal circuit must still export to QIR unchanged (the C-4 check is not
/// over-eager: measurements last are fine).
#[test]
fn c4_qir_accepts_terminal_measurement() {
    let ir = ParameterizedCircuit::new(2)
        .h(0)
        .cx(0, 1)
        .measure_all()
        .to_qir_with_params(&[])
        .unwrap();
    assert!(ir.contains("__quantum__qis__mz__body"));
}
