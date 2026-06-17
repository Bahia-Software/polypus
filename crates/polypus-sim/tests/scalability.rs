//! Scalability guards: a 20-qubit circuit stays well-formed, and oversized
//! requests fail cleanly instead of panicking or attempting a doomed
//! allocation. When built with `--features parallel`, the 20-qubit case
//! exercises the parallel kernels (its qubit count is above the threshold).

use polypus_circuit::{ConcreteCircuit, ParameterizedCircuit};
use polypus_sim::{SimError, Simulator, StatevectorSimulator, C64, MAX_QUBITS};

fn close(a: C64, b: C64) -> bool {
    (a - b).norm() < 1e-9
}

#[test]
fn twenty_qubit_circuit_stays_normalized() {
    let n = 20;
    let mut qc = ParameterizedCircuit::new(n);
    for q in 0..n {
        qc = qc.h(q);
    }
    for q in 0..n - 1 {
        qc = qc.cx(q, q + 1);
    }
    for q in 0..n {
        qc = qc.rz(q, 0.3);
    }
    let concrete = qc.assign_parameters(&[]).unwrap();

    let sv = StatevectorSimulator::new().run(&concrete).unwrap();
    assert_eq!(sv.dim(), 1usize << n);
    assert!((sv.norm() - 1.0).abs() < 1e-9);
}

#[test]
fn large_ghz_has_two_equal_peaks() {
    // A GHZ across 18 qubits: probability 1/2 on |0…0> and |1…1>, exercising
    // the parallel path (when enabled) with an analytically known answer.
    let n = 18;
    let mut qc = ParameterizedCircuit::new(n).h(0);
    for q in 0..n - 1 {
        qc = qc.cx(q, q + 1);
    }
    let concrete = qc.assign_parameters(&[]).unwrap();
    let sv = StatevectorSimulator::new().run(&concrete).unwrap();

    let all_ones = (1usize << n) - 1;
    assert!(close(
        sv.amplitudes()[0],
        C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0)
    ));
    assert!(close(
        sv.amplitudes()[all_ones],
        C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0)
    ));
    assert!((sv.norm() - 1.0).abs() < 1e-9);
}

#[test]
fn statevector_new_rejects_oversized_request_without_panic() {
    let err = polypus_sim::Statevector::new(64).unwrap_err();
    assert_eq!(
        err,
        SimError::TooManyQubits {
            requested: 64,
            max: MAX_QUBITS
        }
    );
}

#[test]
fn simulator_rejects_circuit_above_limit() {
    // Construct an oversized circuit header with no gates; run() must reject it
    // before allocating anything.
    let circuit = ConcreteCircuit {
        num_qubits: 40,
        gates: Vec::new(),
    };
    let err = StatevectorSimulator::new().run(&circuit).unwrap_err();
    assert_eq!(
        err,
        SimError::TooManyQubits {
            requested: 40,
            max: MAX_QUBITS
        }
    );
}
