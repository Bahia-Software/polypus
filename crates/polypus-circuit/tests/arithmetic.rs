//! Integration tests for polypus-circuit: the arithmetic templates
//! (`templates::arithmetic`), checked gate by gate. Their action on basis
//! states is checked by simulation in `polypus-sim/tests/arithmetic.rs`.

use polypus_circuit::templates::arithmetic::phase_adder;
use polypus_circuit::templates::MAX_QFT_QUBITS;
use polypus_circuit::{CircuitError, GateInstruction as G, GateParam::Fixed};
use std::f64::consts::PI;

#[test]
fn phase_adder_angles_follow_the_draper_formula() {
    assert_eq!(
        phase_adder(3, 5, 0).unwrap().gates(),
        [
            G::P {
                qubit: 0,
                lam: Fixed(PI)
            },
            G::P {
                qubit: 1,
                lam: Fixed(PI / 2.0)
            },
            G::P {
                qubit: 2,
                lam: Fixed(5.0 * PI / 4.0)
            },
        ]
    );
}

#[test]
fn phase_adder_gate_counts() {
    for m in 1..=6 {
        assert_eq!(phase_adder(m, 11, 0).unwrap().gates().len(), m);
        assert_eq!(phase_adder(m, 11, 1).unwrap().gates().len(), m);
        assert_eq!(phase_adder(m, 11, 2).unwrap().gates().len(), 3 * m + 2);
    }
}

#[test]
fn phase_adder_handles_registers_wider_than_64_qubits() {
    let qc = phase_adder(200, u64::MAX, 0).unwrap();
    assert_eq!(qc.gates().len(), 200);
    assert!(qc
        .gates()
        .iter()
        .all(|g| matches!(g, G::P { lam: Fixed(x), .. } if x.is_finite())));
}

#[test]
fn phase_adder_validates_its_arguments() {
    assert!(matches!(
        phase_adder(3, 1, 3),
        Err(CircuitError::InvalidTemplateArgument {
            argument: "num_controls",
            ..
        })
    ));
    assert!(matches!(
        phase_adder(0, 1, 0),
        Err(CircuitError::InvalidTemplateArgument {
            argument: "num_qubits",
            ..
        })
    ));
    assert_eq!(
        phase_adder(MAX_QFT_QUBITS + 1, 1, 0).unwrap_err(),
        CircuitError::TemplateTooLarge {
            template: "phase_adder",
            num_qubits: MAX_QFT_QUBITS + 1,
            max_qubits: MAX_QFT_QUBITS
        }
    );
}
