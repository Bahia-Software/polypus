//! Integration tests for polypus-circuit circuit templates.

use polypus_circuit::templates::MAX_QFT_QUBITS;
use polypus_circuit::templates::{qft, qft_with_options, shor_order_finding};
use polypus_circuit::{
    CircuitError, GateInstruction, GateInstruction as G, GateParam, ParameterizedCircuit,
};
use std::f64::consts::PI;

/// Controlled-phase angle used by the QFT between target `j` and control `k`.
fn cp(k: usize, j: usize, sign: f64) -> GateInstruction {
    GateInstruction::Cp {
        q0: k,
        q1: j,
        theta: GateParam::Fixed(sign * PI / 2f64.powi((j - k) as i32)),
    }
}

#[test]
fn qft_three_qubits_matches_qiskit_convention() {
    let qc = qft(3).unwrap();

    // No free parameters, no measurements: directly exportable / composable.
    assert_eq!(qc.num_qubits(), 3);
    assert_eq!(qc.num_params(), 0);
    assert_eq!(qc.num_clbits(), 0);

    assert_eq!(
        qc.gates(),
        vec![
            GateInstruction::H(2),
            cp(1, 2, 1.0), // π/2
            cp(0, 2, 1.0), // π/4
            GateInstruction::H(1),
            cp(0, 1, 1.0), // π/2
            GateInstruction::H(0),
            GateInstruction::Swap(0, 2),
        ]
    );
}

#[test]
fn qft_inverse_is_the_adjoint_of_the_forward() {
    let inv = qft_with_options(3, true, true).unwrap();

    // Reverse of the forward circuit, phases negated, swaps first.
    assert_eq!(
        inv.gates(),
        vec![
            GateInstruction::Swap(0, 2),
            GateInstruction::H(0),
            cp(0, 1, -1.0), // −π/2
            GateInstruction::H(1),
            cp(0, 2, -1.0), // −π/4
            cp(1, 2, -1.0), // −π/2
            GateInstruction::H(2),
        ]
    );
}

#[test]
fn qft_without_swaps_omits_the_swap_network() {
    let qc = qft_with_options(3, false, false).unwrap();
    assert!(!qc
        .gates()
        .iter()
        .any(|g| matches!(g, GateInstruction::Swap(..))));
    // Same rotation block as the default, just without the trailing swap.
    assert_eq!(qc.gates().len(), 6);
}

#[test]
fn qft_single_qubit_is_a_lone_hadamard() {
    let qc = qft(1).unwrap();
    assert_eq!(qc.gates(), vec![GateInstruction::H(0)]);
}

#[test]
fn qft_zero_qubits_is_empty() {
    let qc = qft(0).unwrap();
    assert_eq!(qc.num_qubits(), 0);
    assert!(qc.gates().is_empty());
}

#[test]
fn qft_exports_to_qasm() {
    let qasm = qft(3)
        .unwrap()
        .measure_all()
        .to_qasm2_with_params(&[])
        .unwrap();
    assert!(qasm.contains("h q[2];"));
    assert!(qasm.contains("cp(1.5707963267948966) q[1],q[2];"));
    assert!(qasm.contains("swap q[0],q[2];"));
    assert!(qasm.contains("measure q -> c;"));
}

/// Its gate count grows quadratically: a QFT larger than the importers read
/// back is an error, not an allocation that exhausts memory (#356).
#[test]
fn qft_beyond_max_qft_qubits_is_an_error() {
    assert_eq!(MAX_QFT_QUBITS, 2827);
    for inverse in [false, true] {
        let err = qft_with_options(MAX_QFT_QUBITS + 1, inverse, true).unwrap_err();
        assert_eq!(
            err,
            CircuitError::TemplateTooLarge {
                template: "qft",
                num_qubits: MAX_QFT_QUBITS + 1,
                max_qubits: MAX_QFT_QUBITS,
            }
        );
    }
    assert!(qft(100_000).is_err());
    assert!(qft(usize::MAX).is_err());
}

#[test]
fn shor_order_finding_sizes_match_the_closed_form() {
    for (a, n, t, qubits, gates) in [
        (7, 15, None, 18, 8_561),
        (7, 15, Some(4), 14, 4_273),
        (2, 21, Some(6), 18, 10_165),
    ] {
        let qc = shor_order_finding(a, n, t).unwrap();
        assert_eq!((qc.num_qubits(), qc.gates().len()), (qubits, gates));
        assert_eq!((qc.num_params(), qc.num_clbits()), (0, 0));
    }
}

#[test]
fn shor_order_finding_uses_only_the_template_vocabulary() {
    let qc = shor_order_finding(7, 15, None).unwrap();
    assert!(qc.gates().iter().all(|g| matches!(
        g,
        G::H(_) | G::X(_) | G::P { .. } | G::Cp { .. } | G::Cx(..) | G::Swap(..) | G::Cswap(..)
    )));
}

#[test]
fn shor_order_finding_validates_its_arguments() {
    for (a, n, t, argument) in [
        (2, 2, None, "modulus"),
        (2, (1 << 24) + 1, None, "modulus"),
        (0, 15, None, "a"),
        (15, 15, None, "a"),
        (6, 15, None, "a"),
        (7, 15, Some(0), "counting_qubits"),
        (7, 15, Some(9), "counting_qubits"),
    ] {
        match shor_order_finding(a, n, t) {
            Err(CircuitError::InvalidTemplateArgument {
                template: "shor_order_finding",
                argument: got,
                ..
            }) => assert_eq!(got, argument),
            other => panic!("({a}, {n}, {t:?}): {other:?}"),
        }
    }
}

#[test]
#[ignore = "builds 3.7 million gates; run with --release -- --ignored"]
fn shor_order_finding_builds_the_largest_modulus() {
    let qc = shor_order_finding(2, 16_777_213, None).unwrap();
    assert_eq!((qc.num_qubits(), qc.gates().len()), (98, 3_716_641));
    assert!(matches!(
        qc.to_qasm3(),
        Err(CircuitError::ExportLimit {
            limit: "MAX_SOURCE_BYTES",
            ..
        })
    ));
}

#[test]
fn shor_order_finding_round_trips_through_openqasm_3() {
    let text = shor_order_finding(7, 15, Some(4))
        .unwrap()
        .to_qasm3()
        .unwrap();
    assert_eq!(
        ParameterizedCircuit::from_qasm3(&text)
            .unwrap()
            .to_qasm3()
            .unwrap(),
        text
    );
}
