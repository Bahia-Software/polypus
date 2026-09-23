//! Fase 2 spike: a minimal subprocess bridge to a Python worker.
//!
//! NOT production code. This validates the design decision recorded in the
//! `polypus-backend-abstraction-plan` memory: third-party Python backends run
//! in their own process/interpreter and talk to Rust over a fixed protocol,
//! rather than embedding PyO3. Here the protocol is length-prefixed JSON over
//! the child's stdin/stdout:
//!
//! ```text
//! frame = u32 little-endian length  ||  UTF-8 JSON payload
//! ```
//!
//! The public surface is deliberately tiny: spawn a [`Worker`], `call` it, and
//! get either a typed response or a typed [`BridgeError`]. The error taxonomy
//! here is the raw material for Fase 3's `BackendError::External` /
//! "backend stopped responding" variant.

use std::io::{self, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde::{Deserialize, Serialize};

/// One request frame sent to the worker.
#[derive(Debug, Serialize)]
pub struct Request {
    pub op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub circuits: Option<Vec<Circuit>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shots: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sleep_ms: Option<u64>,
}

impl Request {
    pub fn ping(id: &str) -> Self {
        Request {
            op: "ping".into(),
            id: Some(id.into()),
            circuits: None,
            shots: None,
            sleep_ms: None,
        }
    }

    pub fn run(id: &str, circuits: Vec<Circuit>, shots: u32, sleep_ms: u64) -> Self {
        Request {
            op: "run".into(),
            id: Some(id.into()),
            circuits: Some(circuits),
            shots: Some(shots),
            sleep_ms: Some(sleep_ms),
        }
    }

    pub fn shutdown() -> Self {
        Request {
            op: "shutdown".into(),
            id: None,
            circuits: None,
            shots: None,
            sleep_ms: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Circuit {
    pub qasm: String,
    pub n_qubits: u32,
}

/// One response frame received from the worker. Untyped `counts` kept as
/// serde_json values — a spike does not need the real counts type.
#[derive(Debug, Deserialize)]
pub struct Response {
    pub op: String,
    pub id: Option<String>,
    #[serde(default)]
    pub counts: Option<Vec<serde_json::Map<String, serde_json::Value>>>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
}

/// The typed failure modes the bridge can surface. In Fase 3 these collapse
/// into `BackendError`: `Protocol`/`Io` are internal bugs, while `WorkerDied`
/// is exactly the externally-caused "backend stopped responding" variant that
/// the plan calls for — note it carries whether the child was signalled, which
/// tells a caller whether a fresh worker is worth trying.
#[derive(Debug)]
pub enum BridgeError {
    /// Low-level I/O error writing to / reading from the pipe.
    Io(io::Error),
    /// A frame arrived but was not valid JSON / not the expected shape.
    Protocol(String),
    /// The worker closed its stdout (EOF) before replying: it crashed, was
    /// killed, or exited. `wait_status` is the OS exit description if known.
    /// This is *always* detected, never a hang or a panic.
    WorkerDied { wait_status: String },
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::Io(e) => write!(f, "bridge I/O error: {e}"),
            BridgeError::Protocol(m) => write!(f, "bridge protocol error: {m}"),
            BridgeError::WorkerDied { wait_status } => {
                write!(f, "worker stopped responding (backend died): {wait_status}")
            }
        }
    }
}

impl std::error::Error for BridgeError {}

impl From<io::Error> for BridgeError {
    fn from(e: io::Error) -> Self {
        BridgeError::Io(e)
    }
}

/// A live worker subprocess plus its pipes.
pub struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

impl Worker {
    /// Spawn `python worker.py`. `arm_pdeathsig` requests that the kernel send
    /// the child a SIGKILL if *this* (parent) process dies — the orphan guard
    /// exercised by the SLURM / parent-crash tests.
    pub fn spawn(python: &str, worker_script: &str, arm_pdeathsig: bool) -> io::Result<Worker> {
        Worker::spawn_with_env(python, worker_script, arm_pdeathsig, &[])
    }

    /// Like [`Worker::spawn`], but sets `envs` on the child only. Used by the
    /// crash test to arm the worker's self-destruct without touching the
    /// parent's (process-global, race-prone) environment.
    pub fn spawn_with_env(
        python: &str,
        worker_script: &str,
        arm_pdeathsig: bool,
        envs: &[(&str, &str)],
    ) -> io::Result<Worker> {
        let mut cmd = Command::new(python);
        cmd.arg(worker_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()); // worker logs flow to our stderr
        for (k, v) in envs {
            cmd.env(k, v);
        }

        if arm_pdeathsig {
            // SAFETY: pre_exec runs in the forked child before exec. We only
            // call the async-signal-safe prctl(2); no allocation, no locks.
            unsafe {
                use std::os::unix::process::CommandExt;
                cmd.pre_exec(|| {
                    // PR_SET_PDEATHSIG = 1. Deliver SIGKILL (9) on parent death.
                    let rc = libc::prctl(
                        libc::PR_SET_PDEATHSIG,
                        libc::SIGKILL as libc::c_ulong,
                        0,
                        0,
                        0,
                    );
                    if rc != 0 {
                        return Err(io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }

        let mut child = cmd.spawn()?;
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        Ok(Worker {
            child,
            stdin,
            stdout,
        })
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Send one request frame. Kept separate from the read so tests can send a
    /// request, then interfere with the worker (signal/kill) before reading.
    pub fn send(&mut self, req: &Request) -> Result<(), BridgeError> {
        let payload = serde_json::to_vec(req).map_err(|e| BridgeError::Protocol(e.to_string()))?;
        let len = payload.len() as u32;
        self.stdin.write_all(&len.to_le_bytes())?;
        self.stdin.write_all(&payload)?;
        self.stdin.flush()?;
        Ok(())
    }

    /// Read one response frame. On EOF (worker gone) this reaps the child and
    /// returns [`BridgeError::WorkerDied`] — it never hangs and never panics.
    pub fn recv(&mut self) -> Result<Response, BridgeError> {
        let mut header = [0u8; 4];
        if let Err(e) = self.stdout.read_exact(&mut header) {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                return Err(BridgeError::WorkerDied {
                    wait_status: self.reap(),
                });
            }
            return Err(BridgeError::Io(e));
        }
        let len = u32::from_le_bytes(header) as usize;
        let mut body = vec![0u8; len];
        if let Err(e) = self.stdout.read_exact(&mut body) {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                return Err(BridgeError::WorkerDied {
                    wait_status: self.reap(),
                });
            }
            return Err(BridgeError::Io(e));
        }
        serde_json::from_slice(&body).map_err(|e| BridgeError::Protocol(e.to_string()))
    }

    /// Convenience: send then receive (the normal blocking round-trip).
    pub fn call(&mut self, req: &Request) -> Result<Response, BridgeError> {
        // If the send itself fails with a broken pipe, the worker already died.
        if let Err(BridgeError::Io(e)) = self.send(req) {
            if e.kind() == io::ErrorKind::BrokenPipe {
                return Err(BridgeError::WorkerDied {
                    wait_status: self.reap(),
                });
            }
            return Err(BridgeError::Io(e));
        }
        self.recv()
    }

    /// Send POSIX `signal` to the worker (for the cancellation test). Returns
    /// the raw `kill(2)` result.
    pub fn signal(&self, signal: i32) -> io::Result<()> {
        let rc = unsafe { libc::kill(self.child.id() as libc::pid_t, signal) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Wait for the child and describe how it exited.
    fn reap(&mut self) -> String {
        match self.child.wait() {
            Ok(status) => {
                use std::os::unix::process::ExitStatusExt;
                if let Some(code) = status.code() {
                    format!("exited with code {code}")
                } else if let Some(sig) = status.signal() {
                    format!("killed by signal {sig}")
                } else {
                    format!("terminated ({status})")
                }
            }
            Err(e) => format!("wait() failed: {e}"),
        }
    }

    /// Force-kill the worker (SIGKILL) — used by tests to simulate a crash.
    pub fn kill(&mut self) -> io::Result<()> {
        self.child.kill()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Best-effort: don't leave a worker behind when the handle is dropped.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Resolve the path to `python/worker.py` relative to this crate.
pub fn worker_script() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/python/worker.py").to_string()
}

/// Which Python to launch. Overridable so the SLURM job can point at the same
/// interpreter. Defaults to `python3` on PATH.
pub fn python_bin() -> String {
    std::env::var("SPIKE_PYTHON").unwrap_or_else(|_| "python3".to_string())
}
