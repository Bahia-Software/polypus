//! A live worker that answers in well-formed frames, but not with the reply the bridge
//! asked for: a handshake in another protocol version, a handshake that is not
//! `ready`, a payload that is no protocol message, an in-band error, a `ready` in
//! reply to a run. Each fails the call with an error that names it, and none hangs it:
//! the worker stays blocked on stdin after its reply, so a bridge that waited for it
//! to exit would wait forever. Also the worker's `cwd`.

mod common;

use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

use polypus_backend::{BackendError, Counts, OptLevel, QuantumBackend, RunParams};
use polypus_subprocess_backend::protocol::PROTOCOL_VERSION;
use polypus_subprocess_backend::{BridgeError, SubprocessBackend, SubprocessConfig};

/// How long a call may take before the test calls it hung. Well under the scripted
/// worker's read timeout, so a pass proves the timeout did not end the call.
const LIMIT: Duration = Duration::from_secs(30);

/// Answers each request with the reply given for its `op` on the command line (`hello`
/// first, then `run`), framed as the protocol asks, whatever the reply says. Then it
/// waits for the next request, as a live worker does.
const SCRIPTED_WORKER: &str = r#"
import json, struct, sys
replies = dict(zip(["hello", "run"], sys.argv[1:]))
while True:
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        break
    request = json.loads(sys.stdin.buffer.read(struct.unpack("<I", header)[0]))
    if request["op"] == "shutdown":
        break
    reply = replies[request["op"]].encode()
    sys.stdout.buffer.write(struct.pack("<I", len(reply)) + reply)
    sys.stdout.buffer.flush()
"#;

fn python() -> Option<String> {
    let python = common::resolve_python();
    if python.is_none() {
        eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
    }
    python
}

fn ready() -> String {
    format!(r#"{{"op": "ready", "protocol": {PROTOCOL_VERSION}}}"#)
}

/// A config launching [`SCRIPTED_WORKER`] with these replies.
fn scripted(python: &str, replies: &[&str]) -> SubprocessConfig {
    let mut command = vec![
        python.to_string(),
        "-c".to_string(),
        SCRIPTED_WORKER.to_string(),
    ];
    command.extend(replies.iter().map(|reply| reply.to_string()));
    SubprocessConfig {
        command,
        recv_timeout: Duration::from_secs(120),
        ..SubprocessConfig::default()
    }
}

/// Run `f` on its own thread and fail the test if it does not return within [`LIMIT`]:
/// a regression here is a hang, which would otherwise stall the whole job.
fn within<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(LIMIT) {
        Ok(value) => value,
        Err(RecvTimeoutError::Timeout) => panic!("the bridge hung for {LIMIT:?}"),
        Err(RecvTimeoutError::Disconnected) => panic!("the call panicked"),
    }
}

fn spawn_ok(config: SubprocessConfig) -> Arc<SubprocessBackend> {
    Arc::new(within(move || SubprocessBackend::spawn(config)).expect("the worker handshakes"))
}

fn spawn_err(config: SubprocessConfig) -> BridgeError {
    match within(move || SubprocessBackend::spawn(config)) {
        Err(e) => e,
        Ok(_) => panic!("the spawn must fail"),
    }
}

fn run(backend: &Arc<SubprocessBackend>) -> Result<Vec<Counts>, BackendError> {
    let backend = Arc::clone(backend);
    let params = RunParams {
        id: "misbehaving".to_string(),
        shots: 16,
        seed: None,
        opt_level: OptLevel::default(),
    };
    within(move || backend.run_circuits(&[common::native(1)], &params))
}

/// The bridge error inside a [`BackendError::External`], which is how the bridge
/// reports everything but a dead, hung or aborted worker.
fn bridge_error(err: &BackendError) -> &BridgeError {
    match err {
        BackendError::External(e) => e
            .downcast_ref::<BridgeError>()
            .unwrap_or_else(|| panic!("not a BridgeError: {e:?}")),
        other => panic!("expected BackendError::External, got {other:?}"),
    }
}

#[test]
fn a_worker_speaking_another_protocol_version_is_refused_at_the_handshake() {
    let Some(python) = python() else { return };
    match spawn_err(scripted(&python, &[r#"{"op": "ready", "protocol": 0}"#])) {
        BridgeError::Handshake(m) => assert_eq!(
            m,
            format!("worker speaks protocol v0, this bridge speaks v{PROTOCOL_VERSION}")
        ),
        other => panic!("expected BridgeError::Handshake, got {other:?}"),
    }
}

#[test]
fn a_handshake_answered_with_anything_but_ready_is_refused() {
    let Some(python) = python() else { return };
    let reply = r#"{"op": "error", "message": "no licence for this device"}"#;
    match spawn_err(scripted(&python, &[reply])) {
        BridgeError::Handshake(m) => {
            assert!(m.starts_with("expected a 'ready' frame, got "), "{m}");
            assert!(m.contains("no licence for this device"), "{m}");
        }
        other => panic!("expected BridgeError::Handshake, got {other:?}"),
    }
}

#[test]
fn an_undecodable_reply_to_the_handshake_fails_fast_instead_of_hanging() {
    let Some(python) = python() else { return };
    // An op the protocol does not have, and a payload that is not JSON at all.
    for reply in [r#"{"op": "welcome"}"#, "not json"] {
        match spawn_err(scripted(&python, &[reply])) {
            BridgeError::Protocol(m) => assert!(m.starts_with("undecodable frame: "), "{m}"),
            other => panic!("expected BridgeError::Protocol for {reply:?}, got {other:?}"),
        }
    }
}

#[test]
fn an_undecodable_reply_to_a_run_fails_fast_and_the_worker_is_killed() {
    let Some(python) = python() else { return };
    let backend = spawn_ok(scripted(
        &python,
        &[&ready(), r#"{"op": "result", "counts": "all of them"}"#],
    ));
    let err = run(&backend).expect_err("a reply that is no protocol message has no counts");
    match bridge_error(&err) {
        BridgeError::Protocol(m) => assert!(m.starts_with("undecodable frame: "), "{m}"),
        other => panic!("expected BridgeError::Protocol, got {other:?}"),
    }
    // The worker was killed, so a later call fails at once instead of waiting on it.
    match run(&backend) {
        Err(BackendError::Unresponsive(m)) => {
            assert_eq!(m, "worker process ended: worker already marked dead")
        }
        other => panic!("expected BackendError::Unresponsive, got {other:?}"),
    }
}

#[test]
fn an_in_band_error_carries_the_workers_message_and_keeps_the_worker() {
    let Some(python) = python() else { return };
    let backend = spawn_ok(scripted(
        &python,
        &[
            &ready(),
            r#"{"op": "error", "id": "misbehaving", "message": "device offline"}"#,
        ],
    ));
    // A clean failure, not a dead worker: the second call is answered the same way.
    for _ in 0..2 {
        let err = run(&backend).expect_err("an error reply has no counts");
        assert!(
            matches!(bridge_error(&err), BridgeError::Worker(m) if m == "device offline"),
            "{err:?}"
        );
        assert_eq!(err.to_string(), "subprocess worker error: device offline");
    }
    backend.close();
}

#[test]
fn a_ready_in_reply_to_a_run_is_a_protocol_error() {
    let Some(python) = python() else { return };
    let backend = spawn_ok(scripted(&python, &[&ready(), &ready()]));
    let err = run(&backend).expect_err("a run answered with 'ready' has no counts");
    assert!(
        matches!(bridge_error(&err), BridgeError::Protocol(m) if m == "unexpected 'ready' during a run"),
        "{err:?}"
    );
    backend.close();
}

#[test]
fn a_relative_worker_path_resolves_against_the_configured_cwd() {
    let Some(python) = python() else { return };
    let backend = spawn_ok(SubprocessConfig {
        command: vec![python, "worker_template.py".to_string()],
        cwd: Some(concat!(env!("CARGO_MANIFEST_DIR"), "/python").to_string()),
        ..SubprocessConfig::default()
    });
    let counts = run(&backend).expect("the reference worker runs from its own directory");
    assert_eq!(counts.len(), 1);
    assert_eq!(counts[0].values().sum::<u64>(), 16);
    backend.close();
}

#[test]
fn a_missing_cwd_fails_the_spawn_with_an_io_error() {
    let Some(python) = python() else { return };
    match spawn_err(SubprocessConfig {
        command: vec![python, "-c".to_string(), "pass".to_string()],
        cwd: Some("/nonexistent/polypus-worker-cwd".to_string()),
        ..SubprocessConfig::default()
    }) {
        BridgeError::Io(m) => assert!(m.starts_with("spawning worker: "), "{m}"),
        other => panic!("expected BridgeError::Io, got {other:?}"),
    }
}
