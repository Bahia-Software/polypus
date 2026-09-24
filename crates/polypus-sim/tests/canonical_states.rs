//! End-to-end checks on canonical entangled states and a QAOA layer, plus the
//! measurement read-outs (expectation values and shot sampling).

use polypus_circuit::ParameterizedCircuit;
use polypus_sim::{Simulator, SplitMix64, StatevectorSimulator, C64};
use std::f64::consts::FRAC_1_SQRT_2;

fn close(a: C64, b: C64) -> bool {
    (a - b).norm() < 1e-12
}

fn run(circuit: ParameterizedCircuit, params: &[f64]) -> polypus_sim::Statevector {
    let concrete = circuit.assign_parameters(params).unwrap();
    StatevectorSimulator::new().run(&concrete).unwrap()
}

#[test]
fn bell_state() {
    let sv = run(ParameterizedCircuit::new(2).h(0).cx(0, 1), &[]);
    let a = sv.amplitudes();
    assert!(close(a[0], C64::new(FRAC_1_SQRT_2, 0.0)));
    assert!(close(a[1], C64::new(0.0, 0.0)));
    assert!(close(a[2], C64::new(0.0, 0.0)));
    assert!(close(a[3], C64::new(FRAC_1_SQRT_2, 0.0)));
    assert!((sv.norm() - 1.0).abs() < 1e-12);
}

#[test]
fn ghz_state() {
    let sv = run(ParameterizedCircuit::new(3).h(0).cx(0, 1).cx(1, 2), &[]);
    let a = sv.amplitudes();
    assert!(close(a[0], C64::new(FRAC_1_SQRT_2, 0.0)));
    assert!(close(a[7], C64::new(FRAC_1_SQRT_2, 0.0)));
    for (i, amp) in a.iter().enumerate() {
        if i != 0 && i != 7 {
            assert!(close(*amp, C64::new(0.0, 0.0)));
        }
    }
    assert!((sv.norm() - 1.0).abs() < 1e-12);
}

#[test]
fn bell_expectation_values() {
    let sv = run(ParameterizedCircuit::new(2).h(0).cx(0, 1), &[]);
    // Perfectly correlated: <Z0 Z1> = +1, each marginal <Z> = 0.
    assert!((sv.expectation_z(&[0, 1]) - 1.0).abs() < 1e-12);
    assert!(sv.expectation_z(&[0]).abs() < 1e-12);
    assert!(sv.expectation_z(&[1]).abs() < 1e-12);
}

#[test]
fn qaoa_layer_stays_normalized() {
    // One QAOA MaxCut layer on a 4-cycle: H on all, ZZ cost, X mixer.
    let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
    let mut qc = ParameterizedCircuit::new(4);
    for q in 0..4 {
        qc = qc.h(q);
    }
    for &(a, b) in &edges {
        qc = qc.rzz(a, b, polypus_circuit::Param(1));
    }
    for q in 0..4 {
        qc = qc.rx(q, polypus_circuit::Param(0));
    }
    let sv = run(qc, &[0.8, 0.4]);

    assert!((sv.norm() - 1.0).abs() < 1e-12);
    let total: f64 = sv.probabilities().iter().sum();
    assert!((total - 1.0).abs() < 1e-12);
}

#[test]
fn sampling_bell_only_hits_correlated_outcomes() {
    let concrete = ParameterizedCircuit::new(2)
        .h(0)
        .cx(0, 1)
        .measure_all()
        .assign_parameters(&[])
        .unwrap();
    let sim = StatevectorSimulator::new();
    let counts = sim.run_and_sample(&concrete, 4000, 12345).unwrap();

    // Only |00> (0) and |11> (3) are possible.
    let total: u64 = counts.values().sum();
    assert_eq!(total, 4000);
    for &state in counts.keys() {
        assert!(state == 0 || state == 3, "unexpected outcome {state}");
    }
    assert!(counts.contains_key(&0) && counts.contains_key(&3));
    // Roughly balanced for a fair coin (generous band).
    let n0 = counts[&0] as f64;
    assert!((n0 / 4000.0 - 0.5).abs() < 0.05);
}

// ── States built with the qelib1.inc gates and declared gates ───────────────

/// The only non-zero amplitude, as (basis index, amplitude).
fn single_basis_state(sv: &polypus_sim::Statevector) -> (usize, C64) {
    let nonzero: Vec<(usize, C64)> = sv
        .amplitudes()
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, a)| a.norm() > 1e-12)
        .collect();
    assert_eq!(nonzero.len(), 1, "not a basis state: {nonzero:?}");
    nonzero[0]
}

#[test]
fn toffoli_and_fredkin_truth_tables() {
    for input in 0..8usize {
        let mut prep = ParameterizedCircuit::new(3);
        for q in 0..3 {
            if (input >> q) & 1 == 1 {
                prep = prep.x(q);
            }
        }
        // ccx q0,q1 -> q2 flips bit 2 iff bits 0 and 1 are set.
        let (out, amp) = single_basis_state(&run(prep.clone().ccx(0, 1, 2), &[]));
        let expected = if input & 0b011 == 0b011 {
            input ^ 0b100
        } else {
            input
        };
        assert_eq!(out, expected, "ccx on |{input:03b}>");
        assert!(close(amp, C64::new(1.0, 0.0)));
        // cswap q2; q0,q1 swaps bits 0 and 1 iff bit 2 is set.
        let (out, amp) = single_basis_state(&run(prep.cswap(2, 0, 1), &[]));
        let swapped = (input & 0b100) | ((input & 1) << 1) | ((input >> 1) & 1);
        let expected = if input & 0b100 != 0 { swapped } else { input };
        assert_eq!(out, expected, "cswap on |{input:03b}>");
        assert!(close(amp, C64::new(1.0, 0.0)));
    }
}

/// The 3-qubit W state (|001⟩ + |010⟩ + |100⟩)/√3, built with `ry`, `ch`,
/// `ccx`, `cx` and `x` (the QASMBench `wstate` construction).
#[test]
fn w_state_with_controlled_hadamard_and_toffoli() {
    let theta = 2.0 * (1.0f64 / 3f64.sqrt()).acos();
    let sv = run(
        ParameterizedCircuit::new(3)
            .ry(0, theta)
            .ch(0, 1)
            .ccx(0, 1, 2)
            .x(0)
            .x(1)
            .cx(0, 1),
        &[],
    );
    let a = sv.amplitudes();
    let third = 1.0 / 3.0;
    for (i, amp) in a.iter().enumerate() {
        let expected = if [0b001, 0b010, 0b100].contains(&i) {
            third
        } else {
            0.0
        };
        assert!(
            (amp.norm_sqr() - expected).abs() < 1e-12,
            "P(|{i:03b}>) = {}",
            amp.norm_sqr()
        );
    }
    assert!((sv.norm() - 1.0).abs() < 1e-12);
}

/// A Bell pair through a gate declared with a `gate` block: the call is one
/// instruction, simulated as its body.
#[test]
fn bell_state_through_a_declared_gate() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\ngate bell a,b { h a; cx a,b; }\nqreg q[2];\nbell q[0],q[1];\n";
    let circuit = ParameterizedCircuit::from_qasm2(src).unwrap();
    assert_eq!(circuit.gates.len(), 1);
    let a = run(circuit, &[]).amplitudes().to_vec();
    assert!(close(a[0], C64::new(FRAC_1_SQRT_2, 0.0)));
    assert!(close(a[3], C64::new(FRAC_1_SQRT_2, 0.0)));
    assert!(close(a[1], C64::new(0.0, 0.0)) && close(a[2], C64::new(0.0, 0.0)));
}

/// A 4-controlled X flips the target only from |1111⟩ on the controls.
#[test]
fn c4x_flips_only_when_every_control_is_set() {
    let all = ParameterizedCircuit::new(5)
        .x(0)
        .x(1)
        .x(2)
        .x(3)
        .c4x(0, 1, 2, 3, 4);
    assert_eq!(single_basis_state(&run(all, &[])).0, 0b11111);
    let three = ParameterizedCircuit::new(5)
        .x(0)
        .x(1)
        .x(3)
        .c4x(0, 1, 2, 3, 4);
    assert_eq!(single_basis_state(&run(three, &[])).0, 0b01011);
}

#[test]
fn sampling_is_reproducible() {
    let concrete = ParameterizedCircuit::new(3)
        .h(0)
        .h(1)
        .h(2)
        .assign_parameters(&[])
        .unwrap();
    let sv = StatevectorSimulator::new().run(&concrete).unwrap();
    let a = sv.sample(1000, &mut SplitMix64::new(99));
    let b = sv.sample(1000, &mut SplitMix64::new(99));
    assert_eq!(a, b);
}
