//! Mid-run cancellation (issue #110): the hook passed to
//! [`Simulator::run_cancellable`] stops the gate loop, is polled until it says
//! stop, is throttled far below the gate count, and leaves a run it does not
//! cancel byte-identical to [`Simulator::run`].
//!
//! **Timing-independent by construction.** Every long circuit here ends in a
//! rotation whose angle is `NaN`, which `Statevector::apply` rejects with
//! [`SimError::NonFiniteAmplitude`]. The returned *error variant alone* therefore
//! says whether the loop reached the end of the circuit — `Cancelled` proves it
//! stopped early, `NonFiniteAmplitude` proves it did not — so nothing here
//! asserts on wall-clock time, which would be flaky on a loaded CI runner and
//! would have to be re-tuned for the ~40x gap between the debug and release
//! profiles. What the circuits *are* sized for is duration: long enough that the
//! throttle's checkpoints come due while the loop is still running, with a wide
//! enough margin to hold on a much faster machine (the sizes below are ~10x and
//! ~30x the checkpoint interval in release, ~40x that in debug).

use polypus_circuit::{ConcreteCircuit, GateInstruction as G, GateParam::Fixed};
use polypus_sim::{SimError, Simulator, StatevectorSimulator, MAX_QUBITS};

/// Qubits for the long circuits. High enough that a single gate is expensive
/// (16k amplitudes), so a modest gate count buys a run of a few hundred
/// milliseconds.
const N: usize = 14;

/// Gates for a circuit that must outlive several checkpoints (~0.8s in release,
/// far longer in debug — it is never run to completion).
const LONG: usize = 60_000;

/// Gates for a circuit that must outlive *at least one* checkpoint and then be
/// run to completion, so it is kept as short as that allows.
const SHORT: usize = 3_000;

/// A simulator pinned to the sequential kernels (`parallel_threshold` above any
/// qubit count used here), so these tests take the same path — and last the same
/// time — whether or not the `parallel` feature is enabled, and regardless of
/// the runner's core count.
fn sequential_sim() -> StatevectorSimulator {
    StatevectorSimulator {
        max_qubits: MAX_QUBITS,
        parallel_threshold: MAX_QUBITS + 1,
    }
}

/// `gates` cheap rotations, then one rotation by `NaN` — see the module docs for
/// why the last gate is poisoned.
fn nan_terminated_circuit(gates: usize) -> ConcreteCircuit {
    let mut instructions = Vec::with_capacity(gates + 1);
    for i in 0..gates {
        instructions.push(G::Rx {
            qubit: i % N,
            theta: Fixed(0.1),
        });
    }
    instructions.push(G::Rx {
        qubit: 0,
        theta: Fixed(f64::NAN),
    });
    ConcreteCircuit {
        num_qubits: N,
        gates: instructions,
    }
}

#[test]
fn a_hook_that_asks_to_stop_ends_the_run_early() {
    let circuit = nan_terminated_circuit(LONG);
    let mut calls = 0usize;
    let err = {
        let mut cancel = || {
            calls += 1;
            true
        };
        sequential_sim()
            .run_cancellable(&circuit, Some(&mut cancel))
            .unwrap_err()
    };
    // `Cancelled`, not `NonFiniteAmplitude`: the loop never reached the last
    // gate, i.e. the run really was abandoned mid-circuit.
    assert_eq!(err, SimError::Cancelled);
    // And it stopped at the *first* "yes" rather than polling on.
    assert_eq!(calls, 1);
}

#[test]
fn the_hook_is_polled_until_it_asks_to_stop() {
    let circuit = nan_terminated_circuit(LONG);
    let mut calls = 0usize;
    let err = {
        let mut cancel = || {
            calls += 1;
            calls >= 3
        };
        sequential_sim()
            .run_cancellable(&circuit, Some(&mut cancel))
            .unwrap_err()
    };
    assert_eq!(err, SimError::Cancelled);
    assert_eq!(calls, 3);
}

#[test]
fn a_hook_that_never_cancels_lets_the_run_finish() {
    let circuit = nan_terminated_circuit(SHORT);
    let mut calls = 0usize;
    let err = {
        let mut never = || {
            calls += 1;
            false
        };
        sequential_sim()
            .run_cancellable(&circuit, Some(&mut never))
            .unwrap_err()
    };
    // Reached the poisoned final gate: a hook answering "no" must not cut the
    // circuit short.
    assert_eq!(err, SimError::NonFiniteAmplitude);
    // The checkpoints did fire (otherwise the test above proves nothing) …
    assert!(calls >= 1, "the hook was never polled");
    // … but nowhere near once per gate: that is the throttle doing its job, and
    // the whole reason an expensive hook (the `polypus` crate's reacquires the
    // GIL) cannot slow a simulation down.
    assert!(
        calls < SHORT / 10,
        "hook polled {calls} times for {SHORT} gates — the throttle is not working"
    );
}

#[test]
fn no_hook_is_exactly_run() {
    let circuit = ConcreteCircuit {
        num_qubits: 3,
        gates: vec![
            G::H(0),
            G::Cx(0, 1),
            G::Rx {
                qubit: 2,
                theta: Fixed(0.7),
            },
        ],
    };
    let sim = sequential_sim();
    let with_none = sim.run_cancellable(&circuit, None).unwrap();
    let plain = sim.run(&circuit).unwrap();
    // Bit-identical, not merely close: `run` is defined as `run_cancellable`
    // with no hook, so the two share one gate loop.
    assert_eq!(with_none.amplitudes(), plain.amplitudes());
}
