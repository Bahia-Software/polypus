//! Integration tests for polypus-circuit: build-time scaling.
//!
//! Regression guard for issue #109: `ParameterizedCircuit::try_push` used to
//! enforce contract C-4 by rescanning every instruction already in the circuit,
//! so appending gate `k` cost O(k) and building a circuit of `G` gates cost
//! O(G²). The check is now answered from an incremental per-circuit cache.
//!
//! The assertion is on *scaling*, not on wall-clock time: doubling the gate
//! count must roughly double the build time (linear), where a reinstated rescan
//! would quadruple it. CI runners are shared and noisy, so absolute budgets
//! would be flaky while the ratio is not, and the threshold below leaves a wide
//! margin between the two regimes.

use polypus_circuit::ParameterizedCircuit;
use std::time::{Duration, Instant};

/// Qubit count of the timed circuits: enough width for a realistic mix of
/// single- and two-qubit gates (the issue's repro used 11).
const NUM_QUBITS: usize = 11;

/// Repetitions per measurement; the minimum is kept, being the sample least
/// polluted by scheduling noise.
const REPEATS: usize = 5;

/// Build a circuit of `num_gates` instructions, alternating single-qubit
/// rotations with two-qubit entanglers across the whole register. No
/// measurements: the point is to time the C-4 check on unitaries.
fn build(num_gates: usize) -> ParameterizedCircuit {
    let mut qc = ParameterizedCircuit::new(NUM_QUBITS);
    for i in 0..num_gates {
        let q = i % NUM_QUBITS;
        qc = if i % 2 == 0 {
            qc.rx(q, 0.1)
        } else {
            qc.cx(q, (q + 1) % NUM_QUBITS)
        };
    }
    qc
}

/// Fastest of [`REPEATS`] builds of `num_gates` gates.
fn best_build_time(num_gates: usize) -> Duration {
    (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            let qc = build(num_gates);
            let elapsed = start.elapsed();
            // Consume the result so the build cannot be optimised away.
            assert_eq!(qc.gates.len(), num_gates);
            elapsed
        })
        .min()
        .expect("REPEATS > 0")
}

#[test]
fn building_a_circuit_scales_linearly_with_its_gate_count() {
    // Grow the small size until one build is long enough that timer noise cannot
    // dominate the ratio. Under the old quadratic behaviour the floor is reached
    // immediately, which only makes the ratio below more pronounced.
    let mut n = 2_000;
    while best_build_time(n) < Duration::from_millis(2) && n < 64_000 {
        n *= 2;
    }

    let small = best_build_time(n);
    let large = best_build_time(2 * n);
    let ratio = large.as_secs_f64() / small.as_secs_f64();
    // Visible with `--nocapture` when investigating a regression.
    println!(
        "{n} gates: {small:?}; {} gates: {large:?}; {ratio:.2}x",
        2 * n
    );

    // Linear ⇒ ~2x. Quadratic ⇒ ~4x. Assert well inside the gap.
    assert!(
        ratio < 3.0,
        "build time grew {ratio:.2}x when the gate count doubled \
         ({n} gates in {small:?}, {} gates in {large:?}); \
         expected ~2x for a linear build, ~4x means an O(G²) rescan is back",
        2 * n
    );
}
