//! Integration tests for polypus-circuit: build-time scaling.
//!
//! Regression guard for issue #109: `ParameterizedCircuit::try_push` used to
//! enforce contract C-4 by rescanning every instruction already in the circuit,
//! so appending gate `k` cost O(k) and building a circuit of `G` gates cost
//! O(G²). The check is now answered from an incremental per-circuit cache.
//! `try_compose` keeps the same promise for a loop of composes: a call costs
//! what it appends, not what the circuit already holds.
//!
//! The assertions are on *scaling*, not on wall-clock time: doubling the size
//! of a build must roughly double its time (linear), where a reinstated rescan
//! would quadruple it. CI runners are shared and noisy, so absolute budgets
//! would be flaky while the ratio is not, and the threshold below leaves a wide
//! margin between the two regimes.

use polypus_circuit::ParameterizedCircuit;
use std::fmt::Write;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

/// Qubit count of the timed circuits: enough width for a realistic mix of
/// single- and two-qubit gates (the issue's repro used 11).
const NUM_QUBITS: usize = 11;

/// Repetitions per measurement; the minimum is kept, being the sample least
/// polluted by scheduling noise.
const REPEATS: usize = 5;

/// Upper bound on the gate count the calibration loop below will grow to.
/// Purely a safety net against an unbounded search (e.g. a broken clock
/// always reading ~0); on any realistic hardware the 2ms floor is reached
/// far below this (~30-60k gates on current dev/CI machines), so raising it
/// costs nothing in practice but removes the (very unlikely) risk of the
/// search being cut off before the floor is reached on unusually fast
/// hardware, which would leave the ratio comparison on sub-millisecond,
/// noise-sensitive timings.
const MAX_CALIBRATION_GATES: usize = 1_000_000;

/// The calibration bound for a circuit with named parameters: twice it stays
/// within the 100,000 inputs the OpenQASM 3 importer reads.
const MAX_CALIBRATION_NAMES: usize = 32_000;

/// Run side by side, the timed tests would slow each other down in the middle
/// of a measurement.
static TIMING: Mutex<()> = Mutex::new(());

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

/// A one-qubit circuit whose `n` parameters are OpenQASM 3 inputs named `p0`,
/// `p1`...
fn with_named_parameters(n: usize) -> ParameterizedCircuit {
    let mut source = String::from("OPENQASM 3.0;\nqubit[1] q;\n");
    for i in 0..n {
        writeln!(source, "input float[64] p{i};").expect("writing to a String");
    }
    ParameterizedCircuit::from_qasm3(&source).expect("valid OpenQASM 3")
}

/// How much longer `run` takes when its size doubles: `run(n)` builds at size
/// `n` and returns how long the part being timed took, and the fastest of
/// [`REPEATS`] runs is kept. The size starts at `start` and doubles, up to
/// `max`, until one run is long enough that timer noise cannot dominate the
/// ratio. Returns the size, the two times and their ratio.
fn growth_when_doubled(
    start: usize,
    max: usize,
    run: impl Fn(usize) -> Duration,
) -> (usize, Duration, Duration, f64) {
    let _serial = TIMING.lock().unwrap_or_else(PoisonError::into_inner);
    let best = |n| (0..REPEATS).map(|_| run(n)).min().expect("REPEATS > 0");
    // Under a quadratic cost the floor is reached immediately, which only
    // makes the ratio below more pronounced.
    let mut n = start;
    while best(n) < Duration::from_millis(2) && n < max {
        n *= 2;
    }
    // The two sizes alternate, so a slowdown of the machine that lasts several
    // runs slows both rather than one.
    let (mut small, mut large) = (Duration::MAX, Duration::MAX);
    for _ in 0..REPEATS {
        small = small.min(run(n));
        large = large.min(run(2 * n));
    }
    let ratio = large.as_secs_f64() / small.as_secs_f64();
    // Visible with `--nocapture` when investigating a regression.
    println!("{n}: {small:?}; {}: {large:?}; {ratio:.2}x", 2 * n);
    (n, small, large, ratio)
}

#[test]
fn building_a_circuit_scales_linearly_with_its_gate_count() {
    let (n, small, large, ratio) = growth_when_doubled(2_000, MAX_CALIBRATION_GATES, |n| {
        let start = Instant::now();
        let qc = build(n);
        let elapsed = start.elapsed();
        // Consume the result so the build cannot be optimised away.
        assert_eq!(qc.gates().len(), n);
        elapsed
    });

    // Linear ⇒ ~2x. Quadratic ⇒ ~4x. Assert well inside the gap.
    assert!(
        ratio < 3.0,
        "build time grew {ratio:.2}x when the gate count doubled \
         ({n} gates in {small:?}, {} gates in {large:?}); \
         expected ~2x for a linear build, ~4x means an O(G²) rescan is back",
        2 * n
    );
}

#[test]
fn composing_measurements_in_a_loop_scales_linearly() {
    // Each compose measures one more qubit: a compose that copied the record
    // of measured qubits would cost as many steps as the composes before it.
    let measure = ParameterizedCircuit::new(1).measure(0, 0);
    let (n, small, large, ratio) = growth_when_doubled(2_000, MAX_CALIBRATION_GATES, |n| {
        let mut qc = ParameterizedCircuit::new(n);
        let start = Instant::now();
        for qubit in 0..n {
            qc.try_compose(&measure, &[qubit]).expect("a valid compose");
        }
        let elapsed = start.elapsed();
        assert_eq!(qc.gates().len(), n);
        elapsed
    });

    assert!(
        ratio < 3.0,
        "{n} measuring composes took {small:?} and {} took {large:?}, {ratio:.2}x; \
         expected ~2x, ~4x means a compose copies the record of measured qubits",
        2 * n
    );
}

#[test]
fn composing_onto_named_parameters_scales_linearly() {
    // `n` composes of a block that names nothing, onto a circuit with `n` named
    // parameters: a compose that went over the circuit's names would cost
    // O(n) steps.
    let block = ParameterizedCircuit::new(1).x(0);
    let (n, small, large, ratio) = growth_when_doubled(2_000, MAX_CALIBRATION_NAMES, |n| {
        let mut qc = with_named_parameters(n);
        let start = Instant::now();
        for _ in 0..n {
            qc.try_compose(&block, &[0]).expect("a valid compose");
        }
        let elapsed = start.elapsed();
        assert_eq!((qc.gates().len(), qc.num_params()), (n, n));
        elapsed
    });

    assert!(
        ratio < 3.0,
        "{n} composes onto {n} named parameters took {small:?}, and {} took {large:?}, \
         {ratio:.2}x; expected ~2x, ~4x means a compose goes over the circuit's names",
        2 * n
    );
}
