//! Ctrl+C during a run: the between-wave guard the entry points attach to
//! their run token (docs/ENGINEERING.md §3).

use std::sync::Arc;

/// Between-wave interrupt guard backed by CPython's pending-signal check, so a
/// Ctrl+C (`SIGINT`) aborts a run at the next wave boundary.
///
/// The [`Planner`] is now pyo3-free (in `polypus-backend`): instead of calling
/// `py.check_signals()` itself, it invokes the [`Interrupt`](crate::infrastructure::Interrupt)
/// guard optionally attached to the run's `CancelToken`. This is the guard the edge
/// attaches. A pending signal is boxed into `BackendError::External` and re-raised
/// verbatim at the FFI edge (contract C-1), preserving its `KeyboardInterrupt` class.
///
/// `poll` runs while the thread is detached (the run is inside `py.detach`), so
/// attaching here is a fresh attach, not a reuse of the caller's: `Python::attach`
/// could panic, hang, or attach to a finalizing interpreter, if the interpreter
/// were shutting down (on any Python version). It attaches with
/// [`attach_or`](crate::infrastructure::attach_or) instead, and an unreachable
/// interpreter aborts the run with
/// [`InterpreterUnavailable`](crate::infrastructure::InterpreterUnavailable), the
/// error every other site reached while detached returns. The edge raises it as
/// `KeyboardInterrupt`, the same class a Ctrl+C gets, since the run's result
/// could never be handed back to Python (docs/ENGINEERING.md §3 and §9).
struct SignalInterrupt;

impl crate::infrastructure::Interrupt for SignalInterrupt {
    fn poll(&self) -> Result<(), crate::infrastructure::BackendError> {
        crate::infrastructure::attach_or(
            || Err(crate::infrastructure::InterpreterUnavailable.into()),
            |py| {
                py.check_signals()
                    .map_err(crate::infrastructure::seam_error)
            },
        )
    }
}

/// A run token whose wave-boundary check honours a pending Ctrl+C, restoring the
/// interruptibility the planner's former inline `check_signals` provided.
pub(super) fn interruptible_token() -> crate::infrastructure::CancelToken {
    crate::infrastructure::CancelToken::with_interrupt(Arc::new(SignalInterrupt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::circuit;
    use pyo3::prelude::*;

    /// Set in the child process started by [`run_in_fresh_process`].
    const FRESH_PROCESS_ENV: &str = "POLYPUS_FRESH_PROCESS_CHILD";

    /// Run the test `name` alone in a fresh copy of this test binary, one that
    /// never initializes the interpreter (other tests here call
    /// `Python::initialize()`, which cannot be undone). Mirrors the helper in
    /// `polypus_infrastructure::attach`, which is test-only and so not visible
    /// from this crate.
    fn run_in_fresh_process(name: &str) {
        let exe = std::env::current_exe().expect("the test binary path is available");
        let output = std::process::Command::new(exe)
            .args(["--exact", name, "--test-threads=1", "--nocapture"])
            .env(FRESH_PROCESS_ENV, "1")
            .output()
            .expect("the test binary can be re-executed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "child run failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            stdout.contains("1 passed"),
            "the child must actually run the test, not filter it out:\n{stdout}"
        );
    }

    /// Without an interpreter, both signal-polling hooks that run while
    /// detached cancel instead of panicking: the planner's guard aborts the run
    /// with `InterpreterUnavailable` boxed in `BackendError::External`, the
    /// error every other site reached while detached returns (the edge's
    /// `KeyboardInterrupt`), and the `statevector` hook asks the gate loop to
    /// stop, with no pending exception. "Not initialized" is also how a
    /// shutting-down interpreter looks to PyO3 (CPython clears its
    /// `initialized` flag right after marking itself finalizing), but a real
    /// shutdown cannot be reproduced from a Rust test.
    #[test]
    fn signal_polling_cancels_when_the_interpreter_is_unavailable() {
        if std::env::var_os(FRESH_PROCESS_ENV).is_none() {
            run_in_fresh_process(
                "bindings::interrupt::tests::signal_polling_cancels_when_the_interpreter_is_unavailable",
            );
            return;
        }
        use crate::infrastructure::{BackendError, InterpreterUnavailable, Interrupt};
        match SignalInterrupt.poll() {
            Err(BackendError::External(boxed)) => assert!(
                boxed.is::<InterpreterUnavailable>(),
                "an unreachable interpreter must abort the run with InterpreterUnavailable: {boxed}"
            ),
            other => panic!("an unreachable interpreter must abort the run, got {other:?}"),
        }

        let mut pending = None;
        assert!(
            circuit::poll_for_cancellation(&mut pending),
            "an unreachable interpreter must cancel the simulation"
        );
        assert!(
            pending.is_none(),
            "no exception can be recorded without an interpreter"
        );
    }

    /// `statevector`'s detached simulation really goes through
    /// `poll_for_cancellation`: with no interpreter the first checkpoint
    /// cancels the run, with no pending exception, instead of panicking.
    ///
    /// Timing-independent in the same way as `polypus-sim`'s
    /// `tests/cancellation.rs`: the circuit ends in an `Rx(NaN)`, so
    /// `Cancelled` proves the loop stopped early and `NonFiniteAmplitude`
    /// would prove it never hit a checkpoint. The size (14 qubits, 60 000
    /// barrier-separated rotations) outlives the 25ms checkpoint interval by a
    /// wide margin in both profiles; the run is never completed.
    #[test]
    fn statevector_simulation_cancels_when_the_interpreter_is_unavailable() {
        if std::env::var_os(FRESH_PROCESS_ENV).is_none() {
            run_in_fresh_process(
                "bindings::interrupt::tests::statevector_simulation_cancels_when_the_interpreter_is_unavailable",
            );
            return;
        }
        use polypus_circuit::{ConcreteCircuit, GateInstruction as G, GateParam::Fixed};
        const N: usize = 14;
        let mut gates = Vec::new();
        for i in 0..60_000 {
            gates.push(G::Rx {
                qubit: i % N,
                theta: Fixed(0.1),
            });
            gates.push(G::Barrier(Vec::new()));
        }
        gates.push(G::Rx {
            qubit: 0,
            theta: Fixed(f64::NAN),
        });
        let circuit = ConcreteCircuit::new_unchecked(N, gates);

        let mut pending = None;
        let outcome = circuit::simulate_cancellable(&circuit, true, &mut pending);
        assert!(
            matches!(outcome, Err(polypus_sim::SimError::Cancelled)),
            "an unreachable interpreter must cancel the simulation mid-run, got {:?}",
            outcome.map(|_| "a completed run")
        );
        assert!(
            pending.is_none(),
            "no exception can be recorded without an interpreter"
        );
    }

    /// A cancelled `statevector` re-raises the pending signal's own exception,
    /// and, when the interpreter became unreachable, the same
    /// `KeyboardInterrupt` every other site reached while detached ends in:
    /// same class, same text.
    #[test]
    fn statevector_cancellation_raises_keyboard_interrupt() {
        Python::initialize();
        Python::attach(|py| {
            let unreachable = circuit::cancellation_error(None);
            assert!(unreachable.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py));
            assert_eq!(
                unreachable.value(py).to_string(),
                format!(
                    "the run was aborted: {}",
                    crate::infrastructure::InterpreterUnavailable
                )
            );

            let signal = pyo3::exceptions::PyKeyboardInterrupt::new_err("Ctrl+C");
            let signal_value = signal.value(py).clone();
            let raised = circuit::cancellation_error(Some(signal));
            assert!(
                raised.value(py).is(&signal_value),
                "the pending exception must be re-raised verbatim"
            );
        });
    }
}
