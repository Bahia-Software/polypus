//! `worker.py` exits with code 0 when stopped cleanly, both on a `shutdown` request
//! and on stdin EOF. The bridge never observes this itself (it kills the child right
//! after sending `shutdown`), so these tests launch the worker directly and speak the
//! framing by hand. Skipped (not failed) on a host without a Python interpreter.

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// Locate a Python interpreter, or `None` to skip.
fn resolve_python() -> Option<String> {
    let mut candidates = Vec::new();
    if let Ok(p) = std::env::var("POLYPUS_BRIDGE_PYTHON") {
        candidates.push(p);
    }
    candidates.push("python3".to_string());
    candidates.push("python".to_string());
    candidates.into_iter().find(|c| {
        Command::new(c)
            .arg("-c")
            .arg("pass")
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

fn worker_path() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/worker.py").to_string()
}

/// How long a worker may take to exit after being told to stop.
const EXIT_DEADLINE: Duration = Duration::from_secs(10);

/// Launch `worker.py` with piped stdin/stdout and a silenced stderr.
fn spawn_worker(python: &str) -> Child {
    Command::new(python)
        .arg(worker_path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn worker.py")
}

/// Write one protocol frame: `4-byte little-endian length || UTF-8 JSON payload`.
fn write_frame(stdin: &mut ChildStdin, json: &str) {
    let len = u32::try_from(json.len()).expect("a test frame fits in u32");
    stdin
        .write_all(&len.to_le_bytes())
        .expect("write the frame length");
    stdin
        .write_all(json.as_bytes())
        .expect("write the frame payload");
    stdin.flush().expect("flush the frame");
}

/// Wait for the worker to exit, or kill it and fail if it outlives the deadline.
fn wait_with_deadline(child: &mut Child) -> ExitStatus {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().expect("poll the worker") {
            return status;
        }
        if started.elapsed() > EXIT_DEADLINE {
            let _ = child.kill();
            let _ = child.wait();
            panic!("the worker did not exit within {EXIT_DEADLINE:?} of being told to stop");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn a_worker_exits_0_after_serving_a_hello_and_a_shutdown() {
    let python = match resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };
    let mut child = spawn_worker(&python);
    let mut stdin = child.stdin.take().expect("stdin is piped");
    write_frame(&mut stdin, r#"{"op": "hello", "protocol": 1}"#);
    write_frame(&mut stdin, r#"{"op": "shutdown"}"#);

    let status = wait_with_deadline(&mut child);
    assert_eq!(
        status.code(),
        Some(0),
        "a shutdown is a clean exit: {status}"
    );

    // The `ready` reply sits in the pipe, so reading after exit cannot block.
    let mut stdout = Vec::new();
    child
        .stdout
        .take()
        .expect("stdout is piped")
        .read_to_end(&mut stdout)
        .expect("read the worker's stdout");
    assert!(stdout.len() >= 4, "expected a ready frame, got {stdout:?}");
    let (header, payload) = stdout.split_at(4);
    let len = u32::from_le_bytes(header.try_into().expect("a 4-byte header"));
    assert_eq!(len as usize, payload.len(), "exactly one frame on stdout");
    let payload = std::str::from_utf8(payload).expect("a UTF-8 payload");
    assert!(
        payload.contains(r#""op": "ready""#) && payload.contains(r#""protocol": 1"#),
        "the worker served the handshake before stopping: {payload}"
    );
}

#[test]
fn a_worker_exits_0_when_its_stdin_closes() {
    let python = match resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };
    let mut child = spawn_worker(&python);
    // Closing the pipe without a frame is the bridge going away.
    drop(child.stdin.take().expect("stdin is piped"));

    let status = wait_with_deadline(&mut child);
    assert_eq!(
        status.code(),
        Some(0),
        "stdin EOF is a clean exit: {status}"
    );
}
