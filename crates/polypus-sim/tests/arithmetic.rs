//! The arithmetic templates of `polypus_circuit::templates::arithmetic`, run on
//! every basis state they are specified for: each block must map a basis state
//! to the basis state its arithmetic predicts, with the controls untouched.

use polypus_circuit::templates::arithmetic::{modular_phase_adder, phase_adder};
use polypus_circuit::templates::qft_with_options;
use polypus_circuit::{GateInstruction as G, ParameterizedCircuit};
use polypus_sim::{Simulator, StatevectorSimulator};

/// The basis state `qc` maps `|input⟩` to: prepare `input` with X gates, run
/// `qc` after them, and require a single amplitude of probability `1`.
fn output_state(qc: &ParameterizedCircuit, input: usize) -> usize {
    let mut full = ParameterizedCircuit::new(qc.num_qubits());
    for qubit in (0..qc.num_qubits()).filter(|&q| input >> q & 1 == 1) {
        full.try_push(G::X(qubit)).unwrap();
    }
    for gate in qc.gates() {
        full.try_push(gate.clone()).unwrap();
    }
    let state = StatevectorSimulator::new()
        .run(&full.assign_parameters(&[]).unwrap())
        .unwrap();
    let (index, amplitude) = state
        .amplitudes()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm_sqr().total_cmp(&b.1.norm_sqr()))
        .unwrap();
    assert!(
        amplitude.norm_sqr() > 1.0 - 1e-9,
        "|{input}⟩ does not map to a basis state (largest probability {})",
        amplitude.norm_sqr()
    );
    index
}

/// `block` between the QFT without swaps and its inverse, both placed on
/// `register`: the block then acts on the values of `register`, not on their
/// Fourier transforms.
fn in_fourier_basis(block: &ParameterizedCircuit, register: &[usize]) -> ParameterizedCircuit {
    let m = register.len();
    let mut qc = ParameterizedCircuit::new(block.num_qubits());
    let push_on_register = |qft: ParameterizedCircuit, qc: &mut ParameterizedCircuit| {
        for gate in qft.gates() {
            let placed = match *gate {
                G::H(q) => G::H(register[q]),
                G::Cp { q0, q1, theta } => G::Cp {
                    q0: register[q0],
                    q1: register[q1],
                    theta,
                },
                ref other => panic!("unexpected QFT gate {other:?}"),
            };
            qc.try_push(placed).unwrap();
        }
    };
    push_on_register(qft_with_options(m, false, false).unwrap(), &mut qc);
    for gate in block.gates() {
        qc.try_push(gate.clone()).unwrap();
    }
    push_on_register(qft_with_options(m, true, false).unwrap(), &mut qc);
    qc
}

#[test]
fn phase_adder_adds_modulo_two_to_the_m() {
    let m = 4;
    for num_controls in 0..=2 {
        let register: Vec<usize> = (num_controls..num_controls + m).collect();
        let all_set = (1 << num_controls) - 1;
        for a in 0..16 {
            let qc = in_fourier_basis(&phase_adder(m, a, num_controls).unwrap(), &register);
            for controls in 0..=all_set {
                for b in 0..16usize {
                    let input = controls | (b << num_controls);
                    let expected = if controls == all_set {
                        controls | (((b + a as usize) % 16) << num_controls)
                    } else {
                        input
                    };
                    assert_eq!(
                        output_state(&qc, input),
                        expected,
                        "a = {a}, b = {b}, controls = {controls:0num_controls$b}"
                    );
                }
            }
        }
    }
}

#[test]
fn modular_phase_adder_adds_modulo_n() {
    for modulus in [15usize, 21] {
        // [c1, c2, b_0..b_n, anc]: an (n + 1)-qubit register after the controls.
        let n = (usize::BITS - modulus.leading_zeros()) as usize;
        let register: Vec<usize> = (2..n + 3).collect();
        for a in 0..modulus {
            let block = modular_phase_adder(a as u64, modulus as u64).unwrap();
            let qc = in_fourier_basis(&block, &register);
            for controls in 0..4 {
                for b in 0..modulus {
                    let sum = if controls == 3 { (a + b) % modulus } else { b };
                    assert_eq!(
                        output_state(&qc, controls | (b << 2)),
                        controls | (sum << 2),
                        "N = {modulus}, a = {a}, b = {b}, controls = {controls:02b}"
                    );
                }
            }
        }
    }
}
