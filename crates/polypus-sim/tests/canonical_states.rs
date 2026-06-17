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
