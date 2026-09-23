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

/// One circuit exercising the whole vocabulary, including `cp` (audit item C2)
/// and `swap` (native gate added alongside the QFT template).
/// Free parameters cover the `Param` path; everything else is fixed.
fn full_vocabulary() -> ParameterizedCircuit {
    ParameterizedCircuit::new(5)
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
        .cp(0, 1, 0.75)
        .sx(1)
        .sxdg(2)
        .cy(2, 0)
        .ch(1, 2)
        .csx(0, 2)
        .ccx(2, 0, 1)
        .cswap(1, 2, 0)
        .crx(1, 0, Param(1))
        .cry(2, 1, -0.6)
        .crz(0, 2, 1.1)
        .cu1(2, 1, Param(0))
        .cu3(1, 0, 0.2, -0.4, Param(1))
        .cu(0, 2, 0.3, Param(0), -0.1, 0.9)
        .u0(3, Param(1))
        .rccx(4, 0, 2)
        .rc3x(3, 1, 4, 0)
        .c3x(2, 4, 0, 3)
        .c3sqrtx(4, 3, 1, 2)
        .c4x(1, 4, 0, 3, 2)
        .push(declared_gate_call())
        .barrier()
        .barrier_on(&[0, 2])
        .measure(0, 0)
        .measure(2, 1)
}

/// A call of a gate declared with a `gate` block (only the importer creates
/// declarations): `g(0.25) q[2],q[0];`, whose body calls a second declared gate.
fn declared_gate_call() -> GateInstruction {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
               gate inner(t) a { rz(t/2) a; }\n\
               gate g(t) a,b { inner(t) b; cx a,b; }\n\
               qreg q[3];\ng(0.25) q[2],q[0];\n";
    let imported = ParameterizedCircuit::from_qasm2(src).unwrap();
    assert_eq!(imported.gates.len(), 1);
    imported.gates[0].clone()
}

/// Number of instruction kinds in [`GateInstruction`].
const INSTRUCTION_KINDS: usize = 42;

/// Every instruction kind, numbered `0..INSTRUCTION_KINDS`. The match is
/// exhaustive, so adding a variant fails to build here until it is numbered;
/// `c2_full_vocabulary_covers_every_instruction` then fails until
/// `full_vocabulary` exercises it.
fn instruction_kind(gate: &GateInstruction) -> usize {
    use GateInstruction as G;
    match gate {
        G::H(_) => 0,
        G::X(_) => 1,
        G::Y(_) => 2,
        G::Z(_) => 3,
        G::S(_) => 4,
        G::T(_) => 5,
        G::Sdg(_) => 6,
        G::Tdg(_) => 7,
        G::Id(_) => 8,
        G::Rx { .. } => 9,
        G::Ry { .. } => 10,
        G::Rz { .. } => 11,
        G::Cx(..) => 12,
        G::Cz(..) => 13,
        G::Swap(..) => 14,
        G::Rzz { .. } => 15,
        G::Rxx { .. } => 16,
        G::Cp { .. } => 17,
        G::U { .. } => 18,
        G::Sx(_) => 19,
        G::Sxdg(_) => 20,
        G::Cy(..) => 21,
        G::Ch(..) => 22,
        G::Csx(..) => 23,
        G::Ccx(..) => 24,
        G::Cswap(..) => 25,
        G::Crx { .. } => 26,
        G::Cry { .. } => 27,
        G::Crz { .. } => 28,
        G::Cu1 { .. } => 29,
        G::Cu3 { .. } => 30,
        G::Cu { .. } => 31,
        G::Barrier(_) => 32,
        G::Measure { .. } => 33,
        G::MeasureAll => 34,
        G::Custom(_) => 35,
        G::U0 { .. } => 36,
        G::Rccx(..) => 37,
        G::Rc3x(..) => 38,
        G::C3x(..) => 39,
        G::C3sqrtx(..) => 40,
        G::C4x(..) => 41,
    }
}

#[test]
fn c2_full_vocabulary_covers_every_instruction() {
    let mut seen = [false; INSTRUCTION_KINDS];
    // `MeasureAll` has its own round-trip case below: `full_vocabulary` keeps a
    // partial measurement so the importer does not collapse it.
    for gate in full_vocabulary()
        .gates
        .iter()
        .chain([&GateInstruction::MeasureAll])
    {
        seen[instruction_kind(gate)] = true;
    }
    let missing: Vec<usize> = (0..INSTRUCTION_KINDS).filter(|&k| !seen[k]).collect();
    assert!(
        missing.is_empty(),
        "instruction kinds missing from full_vocabulary: {missing:?}"
    );
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
        ("id", ParameterizedCircuit::new(1).id(0)),
        ("rx", ParameterizedCircuit::new(1).rx(0, 0.3)),
        ("ry", ParameterizedCircuit::new(1).ry(0, 0.3)),
        ("rz", ParameterizedCircuit::new(1).rz(0, 0.3)),
        ("cx", ParameterizedCircuit::new(2).cx(0, 1)),
        ("cz", ParameterizedCircuit::new(2).cz(0, 1)),
        ("swap", ParameterizedCircuit::new(2).swap(0, 1)),
        ("rzz", ParameterizedCircuit::new(2).rzz(0, 1, 0.3)),
        ("rxx", ParameterizedCircuit::new(2).rxx(0, 1, 0.3)),
        ("cp", ParameterizedCircuit::new(2).cp(0, 1, 0.3)),
        ("u3", ParameterizedCircuit::new(1).u(0, 0.1, 0.2, 0.3)),
        ("sx", ParameterizedCircuit::new(1).sx(0)),
        ("sxdg", ParameterizedCircuit::new(1).sxdg(0)),
        // Operands deliberately out of ascending order: a swapped control and
        // target (or a re-sorted operand list) would change the instruction.
        ("cy", ParameterizedCircuit::new(2).cy(1, 0)),
        ("ch", ParameterizedCircuit::new(2).ch(1, 0)),
        ("csx", ParameterizedCircuit::new(2).csx(1, 0)),
        ("ccx", ParameterizedCircuit::new(3).ccx(2, 0, 1)),
        ("cswap", ParameterizedCircuit::new(3).cswap(1, 2, 0)),
        ("crx", ParameterizedCircuit::new(2).crx(1, 0, 0.3)),
        ("cry", ParameterizedCircuit::new(2).cry(1, 0, 0.3)),
        ("crz", ParameterizedCircuit::new(2).crz(1, 0, 0.3)),
        ("cu1", ParameterizedCircuit::new(2).cu1(1, 0, 0.3)),
        ("cu3", ParameterizedCircuit::new(2).cu3(1, 0, 0.1, 0.2, 0.3)),
        (
            "cu",
            ParameterizedCircuit::new(2).cu(1, 0, 0.1, 0.2, 0.3, 0.4),
        ),
        ("u0", ParameterizedCircuit::new(1).u0(0, 0.5)),
        ("rccx", ParameterizedCircuit::new(3).rccx(2, 0, 1)),
        ("rc3x", ParameterizedCircuit::new(4).rc3x(3, 1, 0, 2)),
        ("c3x", ParameterizedCircuit::new(4).c3x(2, 3, 0, 1)),
        ("c3sqrtx", ParameterizedCircuit::new(4).c3sqrtx(1, 3, 2, 0)),
        ("c4x", ParameterizedCircuit::new(5).c4x(4, 0, 3, 1, 2)),
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

/// The text-first direction, gate by gate: an OpenQASM statement in canonical
/// form parses into one instruction and is re-emitted byte-identically. This is
/// what keeps a benchmark file's circuit intact on its way to Aer: the same
/// names, the same operand order, no decomposition.
#[test]
fn c2_every_gate_statement_reemits_byte_identically() {
    let statements = [
        "h q[2];",
        "x q[0];",
        "y q[1];",
        "z q[2];",
        "s q[0];",
        "t q[1];",
        "sdg q[2];",
        "tdg q[0];",
        "id q[1];",
        "sx q[2];",
        "sxdg q[0];",
        "rx(0.250000000000) q[1];",
        "ry(-1.500000000000) q[2];",
        "rz(3.141592653590) q[0];",
        "u3(0.100000000000,0.200000000000,0.300000000000) q[1];",
        "cx q[2],q[0];",
        "cz q[1],q[2];",
        "cy q[2],q[1];",
        "ch q[0],q[2];",
        "csx q[2],q[0];",
        "swap q[1],q[0];",
        "ccx q[2],q[0],q[1];",
        "cswap q[1],q[2],q[0];",
        "rzz(0.400000000000) q[2],q[1];",
        "rxx(-0.400000000000) q[1],q[0];",
        "cp(0.750000000000) q[2],q[0];",
        "cu1(0.750000000000) q[2],q[0];",
        "crx(0.500000000000) q[1],q[0];",
        "cry(-0.500000000000) q[2],q[1];",
        "crz(1.250000000000) q[0],q[2];",
        "cu3(0.100000000000,-0.200000000000,0.300000000000) q[2],q[0];",
        "cu(0.100000000000,0.200000000000,-0.300000000000,0.400000000000) q[1],q[2];",
        "u0(0.500000000000) q[4];",
        "rccx q[2],q[4],q[0];",
        "rc3x q[3],q[0],q[4],q[1];",
        "c3x q[1],q[4],q[0],q[2];",
        "c3sqrtx q[4],q[2],q[3],q[0];",
        "c4x q[3],q[0],q[4],q[2],q[1];",
    ];
    for statement in statements {
        let src = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[5];\n{statement}\n");
        let imported = ParameterizedCircuit::from_qasm2(&src)
            .unwrap_or_else(|e| panic!("{statement}: failed to parse: {e}"));
        assert_eq!(imported.gates.len(), 1, "{statement}: not one instruction");
        assert_eq!(
            imported.to_qasm2_with_params(&[]).unwrap(),
            src,
            "{statement}: not re-emitted byte-identically"
        );
    }
}

/// The fuzz target's round-trip property (`fuzz/fuzz_targets/from_qasm2.rs`),
/// on its seed corpus — so it runs in every test build, not only under
/// `cargo fuzz`: whatever imports, exports to a fixed point.
#[test]
fn c2_export_is_a_fixed_point_on_the_fuzz_corpus() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/corpus/from_qasm2");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let src = std::fs::read_to_string(&path).unwrap();
        let circuit = ParameterizedCircuit::from_qasm2(&src)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let exported = circuit.to_qasm2_with_params(&[]).unwrap();
        let again = ParameterizedCircuit::from_qasm2(&exported)
            .and_then(|c| c.to_qasm2_with_params(&[]))
            .unwrap_or_else(|e| panic!("{}: re-import failed: {e}", path.display()));
        assert_eq!(again, exported, "{}", path.display());
        checked += 1;
    }
    assert!(checked >= 7, "corpus not found in {}", dir.display());
}

/// `cu1` and `cp` are the same operator but distinct instructions: each keeps
/// its own spelling through import and export (the byte-identical round-trip
/// guarantee), rather than one being normalised into the other.
#[test]
fn c2_cu1_and_cp_keep_their_own_spelling() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncu1(0.500000000000) q[0],q[1];\ncp(0.500000000000) q[0],q[1];\n";
    let imported = ParameterizedCircuit::from_qasm2(src).unwrap();
    assert_eq!(
        imported.gates,
        [
            GateInstruction::Cu1 {
                q0: 0,
                q1: 1,
                theta: GateParam::Fixed(0.5)
            },
            GateInstruction::Cp {
                q0: 0,
                q1: 1,
                theta: GateParam::Fixed(0.5)
            },
        ]
    );
    assert_eq!(imported.to_qasm2_with_params(&[]).unwrap(), src);
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

/// `id` is a unitary (the identity) like any gate: it may not act on a measured
/// qubit, at push time or at parse time.
#[test]
fn c4_id_after_measure_is_rejected_by_builder_and_importer() {
    let mut qc = ParameterizedCircuit::new(1);
    qc.try_push(GateInstruction::Measure { qubit: 0, cbit: 0 })
        .unwrap();
    assert_eq!(
        qc.try_push(GateInstruction::Id(0)),
        Err(CircuitError::QubitAlreadyMeasured { qubit: 0 })
    );

    let src = "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nid q[0];\n";
    match ParameterizedCircuit::from_qasm2(src) {
        Err(CircuitError::Parse { line: 5, message }) => {
            assert!(message.contains("after it was measured"), "{message}")
        }
        other => panic!("expected a C-4 parse error at line 5, got {other:?}"),
    }
}

/// C-4 holds at every arity: a three-qubit gate is rejected if *any* of its
/// operands was measured, and the first measured operand (in operand order) is
/// the one reported — by the builder, the importer and the QIR exporter alike.
#[test]
fn c4_three_qubit_gates_reject_any_measured_operand() {
    for (measured, gate, offending) in [
        (2, GateInstruction::Ccx(0, 1, 2), 2), // the target
        (0, GateInstruction::Ccx(0, 1, 2), 0), // a control
        (1, GateInstruction::Cswap(0, 1, 2), 1),
        (2, GateInstruction::Cswap(2, 0, 1), 2),
    ] {
        let mut qc = ParameterizedCircuit::new(3);
        qc.try_push(GateInstruction::Measure {
            qubit: measured,
            cbit: 0,
        })
        .unwrap();
        assert_eq!(
            qc.try_push(gate.clone()),
            Err(CircuitError::QubitAlreadyMeasured { qubit: offending }),
            "builder, {gate:?}"
        );

        let mut hand = ParameterizedCircuit::new(3);
        hand.gates = vec![
            GateInstruction::Measure {
                qubit: measured,
                cbit: 0,
            },
            gate.clone(),
        ];
        assert_eq!(
            hand.to_qir_with_params(&[]),
            Err(CircuitError::QubitAlreadyMeasured { qubit: offending }),
            "QIR exporter, {gate:?}"
        );
    }

    let src =
        "OPENQASM 2.0;\nqreg q[3];\ncreg c[1];\nmeasure q[1] -> c[0];\ncswap q[0],q[2],q[1];\n";
    match ParameterizedCircuit::from_qasm2(src) {
        Err(CircuitError::Parse { line: 5, message }) => {
            assert!(
                message.contains("qubit 1 after it was measured"),
                "{message}"
            )
        }
        other => panic!("expected a C-4 parse error at line 5, got {other:?}"),
    }
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
    let mut qc = ParameterizedCircuit::new(1);
    qc.gates = vec![
        GateInstruction::Measure { qubit: 0, cbit: 0 },
        GateInstruction::X(0),
    ];
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
