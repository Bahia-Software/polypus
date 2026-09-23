//! Point 2a: cancellation. With a call in flight, we signal the worker from
//! the Rust side (this is what a "Ctrl-C / CancelToken" would drive) and check
//! how the subprocess learns it must abort.
//!
//! Finding demonstrated here: the ONLY thing that reaches a worker busy inside
//! a compute/QPU call is an OS signal. Closing the pipe does not, because the
//! worker is not reading it while it works. So the bridge's cancel primitive
//! must be "signal the child", and the protocol needs an explicit `aborted`
//! reply so the worker can survive cancellation and be reused.

use std::time::{Duration, Instant};

use subprocess_bridge_spike::{python_bin, worker_script, Circuit, Request, Worker};

#[test]
fn sigint_aborts_in_flight_call_and_worker_survives() {
    let mut w = Worker::spawn(&python_bin(), &worker_script(), false).expect("spawn");

    // Start a long "QPU" call (2s) but do not read the reply yet.
    let circuits = vec![Circuit {
        qasm: "long".into(),
        n_qubits: 2,
    }];
    w.send(&Request::run("cancel-me", circuits, 10, 2000))
        .expect("send");

    // Let it get well into the call, then cancel via SIGINT.
    std::thread::sleep(Duration::from_millis(200));
    let t0 = Instant::now();
    w.signal(libc::SIGINT).expect("kill(SIGINT)");

    // The worker catches the signal, aborts the in-flight op, and replies.
    let resp = w.recv().expect("recv after signal");
    let elapsed = t0.elapsed();

    assert_eq!(resp.op, "aborted", "worker acknowledges cancellation");
    assert_eq!(resp.reason.as_deref(), Some("signal"));
    assert!(
        elapsed < Duration::from_millis(1500),
        "cancellation was prompt, not run-to-completion: {elapsed:?}"
    );

    // Crucial: the worker is still alive and usable — cancellation is graceful,
    // not a crash. No need to respawn.
    let ok = w
        .call(&Request::ping("still-alive"))
        .expect("ping after abort");
    assert_eq!(ok.op, "pong");
}

/// Contrast: SIGKILL is NOT catchable, so cancelling with it destroys the
/// worker. That collapses cancellation into the crash path (see crash.rs).
/// Kept here to make the "graceful SIGINT vs hard SIGKILL" boundary explicit.
#[test]
fn sigkill_cannot_be_caught_so_it_becomes_a_crash() {
    let mut w = Worker::spawn(&python_bin(), &worker_script(), false).expect("spawn");
    w.send(&Request::run(
        "kill-me",
        vec![Circuit {
            qasm: "x".into(),
            n_qubits: 1,
        }],
        10,
        2000,
    ))
    .expect("send");
    std::thread::sleep(Duration::from_millis(150));
    w.signal(libc::SIGKILL).expect("kill(SIGKILL)");

    let err = w.recv().expect_err("worker died, no reply");
    match err {
        subprocess_bridge_spike::BridgeError::WorkerDied { wait_status } => {
            assert!(
                wait_status.contains("signal 9"),
                "reaped as SIGKILL: {wait_status}"
            );
        }
        other => panic!("expected WorkerDied, got {other:?}"),
    }
}
