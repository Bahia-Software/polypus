//! # polypus-subprocess-backend
//!
//! A [`QuantumBackend`] that runs a provider's **Python SDK in its own
//! subprocess**, talking to it over the versioned length-prefixed JSON protocol in
//! [`protocol`]. This is the productionised form of the Fase-2 subprocess-bridge
//! spike, and the mechanism by which a `pip install polypus` user reaches a
//! third-party Python backend **without embedding the interpreter** — this crate is
//! pyo3-free (verified in CI with `cargo tree`); the interpreter lives entirely in
//! the child process.
//!
//! It registers itself under the name `"subprocess"` (see [`register`]), so the
//! Polypus factory can build it by name from configuration alone. A Python user
//! points it at a worker command; the worker speaks [`protocol`] and drives the
//! real SDK. A reference worker lives in `python/worker_template.py`.
//!
//! ## The four Fase-2 risks, closed
//!
//! - **Liveness (worker hung).** Reads are bounded by a configurable timeout
//!   ([`SubprocessConfig::recv_timeout`]). A worker that is alive but wedged (a
//!   deadlocked SDK, a mute QPU) trips the timeout and surfaces as
//!   [`BackendError::Unresponsive`] — the same variant a *dead* worker (EOF) maps
//!   to — instead of blocking forever. See `docs/backends.md` for how to size it.
//! - **Orphan guard.** On Linux the child is armed with `PR_SET_PDEATHSIG` so it
//!   dies with us; this is Linux-only (documented, with the portable fallback: an
//!   explicit `close()`/`Drop` that kills the child).
//! - **SLURM resource sharing.** The worker shares the job's `--mem`/cores with the
//!   Rust process — the usage guide requires explicit `--cpus-per-task`/`--mem`.
//! - **IPC overhead.** Measured for large payloads by `src/bin/payload_overhead.rs`.
//!
//! ## Cancellation
//!
//! [`SubprocessBackend::cancel`] sends `SIGINT` to the worker so a call blocked in
//! the bridge's `recv` unblocks and the worker replies `aborted`. It is invoked
//! out-of-band by the planner's watcher thread (see
//! [`QuantumBackend::wants_cancel_watcher`], which this backend enables) when the
//! run's `CancelToken` flips mid-wave.

pub mod protocol;

use std::error::Error;
use std::fmt;
use std::io::{self, Read, Write};
// `ExitStatusExt::signal` works on every Unix; `CommandExt::pre_exec` is only
// called to arm the Linux-only `PR_SET_PDEATHSIG`, so its import is Linux-gated
// to avoid an unused-import warning on other Unix (e.g. macOS).
#[cfg(target_os = "linux")]
use std::os::unix::process::CommandExt;
use std::os::unix::process::ExitStatusExt;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use polypus_backend::{
    register_backend, BackendBuildContext, BackendError, BoundCircuit, Counts, QuantumBackend,
    RunParams,
};

use protocol::{Circuit, Request, Response, PROTOCOL_VERSION};

/// The name this backend registers itself under.
pub const BACKEND_NAME: &str = "subprocess";

/// Default read timeout: how long the bridge waits for a worker reply before
/// declaring it [`Unresponsive`](BackendError::Unresponsive). Matches QMIO's
/// default (`QMIO_RECV_TIMEOUT_MS`): generous enough for a slow real QPU, and the
/// sole knob that also bounds how fast a genuine hang is detected — size it to your
/// workload (see `docs/backends.md`).
pub const DEFAULT_RECV_TIMEOUT_MS: u64 = 300_000;

/// Failure modes of the bridge itself, before mapping to [`BackendError`].
///
/// The mapping ([`From<BridgeError>`](BackendError)) is what ties the Fase-2
/// findings to the contract: both a *dead* worker ([`WorkerDied`](Self::WorkerDied),
/// EOF) and a *hung* worker ([`Timeout`](Self::Timeout)) become
/// [`BackendError::Unresponsive`], the variant reserved for "the backend stopped
/// responding".
#[derive(Debug)]
pub enum BridgeError {
    /// Low-level I/O error writing to / reading from the worker's pipes.
    Io(String),
    /// A frame arrived but was not valid protocol JSON, or an unexpected message
    /// shape (e.g. a second `ready`). An internal/worker bug, not a hang.
    Protocol(String),
    /// The handshake failed: the worker did not answer `ready`, or answered a
    /// protocol version this bridge does not speak.
    Handshake(String),
    /// The worker closed its stdout (EOF) before replying: it crashed, was killed,
    /// or exited. Always *detected*, never a hang or a panic.
    WorkerDied {
        /// How the child exited, if known.
        detail: String,
    },
    /// No reply arrived within the read timeout: the worker is alive but wedged (a
    /// deadlocked SDK, a mute QPU). The Fase-2 open risk, now handled.
    Timeout {
        /// The elapsed deadline, in milliseconds.
        millis: u128,
    },
    /// The worker aborted an in-flight call in response to a cancellation signal and
    /// stayed alive. Not a failure — it is how a cooperative cancel completes.
    Aborted,
    /// The worker reported a clean, in-band failure (bad input, provider error).
    Worker(String),
    /// A circuit representation the bridge cannot serialise to send to the worker
    /// (a provider-native `Foreign` object cannot cross a process boundary).
    UnsupportedCircuit(String),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgeError::Io(m) => write!(f, "subprocess bridge I/O error: {m}"),
            BridgeError::Protocol(m) => write!(f, "subprocess bridge protocol error: {m}"),
            BridgeError::Handshake(m) => write!(f, "subprocess worker handshake failed: {m}"),
            BridgeError::WorkerDied { detail } => {
                write!(f, "subprocess worker died mid-call: {detail}")
            }
            BridgeError::Timeout { millis } => {
                write!(f, "no response within {millis} ms (worker hung)")
            }
            BridgeError::Aborted => write!(f, "the run was aborted by a cancellation signal"),
            BridgeError::Worker(m) => write!(f, "subprocess worker error: {m}"),
            BridgeError::UnsupportedCircuit(m) => write!(f, "{m}"),
        }
    }
}

impl Error for BridgeError {}

impl From<BridgeError> for BackendError {
    fn from(err: BridgeError) -> Self {
        match err {
            // Both a dead worker and a hung worker are "stopped responding".
            BridgeError::WorkerDied { detail } => {
                BackendError::Unresponsive(format!("worker process ended: {detail}"))
            }
            BridgeError::Timeout { millis } => BackendError::Unresponsive(format!(
                "no response within {millis} ms; the worker is alive but unresponsive \
                 (a hung SDK or a mute QPU)"
            )),
            BridgeError::UnsupportedCircuit(m) => BackendError::UnsupportedCircuit(m),
            // The worker aborted an in-flight call in response to a signal (a terminal
            // Ctrl+C reaches it directly through the process group). That is a
            // cancellation, whatever its source, so it maps to the contract's
            // `Aborted` variant — which the FFI edge raises as `KeyboardInterrupt` —
            // rather than to a generic `External` error.
            BridgeError::Aborted => BackendError::Aborted(
                "subprocess worker received a cancellation signal".to_string(),
            ),
            // Worker / Io / Protocol / Handshake carry their own Display and cross the
            // contract type-erased; the FFI edge surfaces them.
            other => BackendError::External(Box::new(other)),
        }
    }
}

/// One item the reader thread hands to the bridge: either a decoded frame or the
/// terminal reason the reader stopped (EOF, I/O error, or a protocol decode error).
enum FromWorker {
    Frame(Response),
    Ended(String),
}

/// A live worker subprocess: its stdin, a background reader thread draining stdout
/// into a channel, and the bookkeeping to detect death, time out a hang, and reap.
struct Worker {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<FromWorker>,
    reader: Option<JoinHandle<()>>,
    /// Set once the worker is known unusable (died, or a timeout desynced the
    /// stream) so we fail fast instead of sending into a broken pipe.
    dead: bool,
    /// Set once the child has been `wait()`ed (reaped), so teardown does not try to
    /// kill/wait an already-reaped child and mistake the benign failure for a fault.
    reaped: bool,
}

/// Read one length-prefixed frame from `stream`. `Ok(None)` is a clean EOF at a
/// frame boundary (the worker closed stdout); `Ok(Some(bytes))` is a full payload.
fn read_frame(stream: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut header = [0u8; 4];
    match stream.read_exact(&mut header) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(header) as usize;
    let mut body = vec![0u8; len];
    match stream.read_exact(&mut body) {
        Ok(()) => Ok(Some(body)),
        // A partial frame after the header is a truncated stream — treat as EOF.
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e),
    }
}

impl Worker {
    /// Spawn `argv` (argv[0] is the program), wiring stdin/stdout as pipes and
    /// leaving stderr inherited so worker logs reach our stderr. `arm_pdeathsig`
    /// requests a kernel SIGKILL to the child if this process dies (Linux only).
    fn spawn(
        argv: &[String],
        cwd: Option<&str>,
        env: &[(String, String)],
        arm_pdeathsig: bool,
    ) -> Result<Worker, BridgeError> {
        let (program, args) = argv
            .split_first()
            .ok_or_else(|| BridgeError::Io("empty worker command".to_string()))?;
        let mut cmd = Command::new(program);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        for (k, v) in env {
            cmd.env(k, v);
        }
        if arm_pdeathsig {
            // `PR_SET_PDEATHSIG` is a Linux-only kernel feature (`nix::sys::prctl`
            // does not exist on other Unix such as macOS), so the orphan guard is
            // armed only on Linux. Elsewhere the request is honoured as a no-op and
            // logged — the worker will not be auto-killed if this process dies. This
            // is the portable-fallback caveat documented in docs/backends.md.
            #[cfg(target_os = "linux")]
            {
                // POLICY EXCEPTION (docs/ENGINEERING.md §5, recorded there as the one
                // signed-off `unsafe` outside polypus-sim/kernels.rs): this is the
                // crate's **only** `unsafe`, and it is unavoidable — `Command::pre_exec`
                // is an `unsafe fn` in std, with no safe equivalent, and
                // `PR_SET_PDEATHSIG` resets across `fork` so it can only be armed here,
                // in the child between fork and exec. It is not a performance
                // optimisation. The body itself is safe: it calls `nix`'s audited
                // `set_pdeathsig` wrapper, not hand-written FFI.
                //
                // SAFETY: `pre_exec` runs in the forked child before `exec`. We call
                // only the async-signal-safe `prctl(2)` via `nix`; no alloc, no locks.
                unsafe {
                    cmd.pre_exec(|| {
                        nix::sys::prctl::set_pdeathsig(nix::sys::signal::Signal::SIGKILL)
                            .map_err(|errno| io::Error::from_raw_os_error(errno as i32))
                    });
                }
            }
            #[cfg(not(target_os = "linux"))]
            log::warn!(
                "arm_pdeathsig was requested but PR_SET_PDEATHSIG is Linux-only; \
                 the orphan guard is not armed on this platform, so the worker will \
                 not be auto-killed if this process dies unexpectedly"
            );
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| BridgeError::Io(format!("spawning worker: {e}")))?;
        let stdin = child.stdin.take().expect("stdin piped");
        let mut stdout = child.stdout.take().expect("stdout piped");
        let (tx, rx) = mpsc::channel::<FromWorker>();
        let reader = std::thread::spawn(move || loop {
            match read_frame(&mut stdout) {
                Ok(Some(bytes)) => match serde_json::from_slice::<Response>(&bytes) {
                    Ok(resp) => {
                        if tx.send(FromWorker::Frame(resp)).is_err() {
                            break; // bridge dropped the receiver
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(FromWorker::Ended(format!("undecodable frame: {e}")));
                        break;
                    }
                },
                Ok(None) => {
                    let _ = tx.send(FromWorker::Ended("worker closed stdout (EOF)".to_string()));
                    break;
                }
                Err(e) => {
                    let _ = tx.send(FromWorker::Ended(format!("stdout read error: {e}")));
                    break;
                }
            }
        });
        Ok(Worker {
            child,
            stdin,
            rx,
            reader: Some(reader),
            dead: false,
            reaped: false,
        })
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Send one request frame. A broken pipe means the worker already died.
    fn send(&mut self, req: &Request) -> Result<(), BridgeError> {
        if self.dead {
            return Err(BridgeError::WorkerDied {
                detail: "worker already marked dead".to_string(),
            });
        }
        let payload = serde_json::to_vec(req).map_err(|e| BridgeError::Protocol(e.to_string()))?;
        let len = payload.len() as u32;
        let write = self
            .stdin
            .write_all(&len.to_le_bytes())
            .and_then(|()| self.stdin.write_all(&payload))
            .and_then(|()| self.stdin.flush());
        if let Err(e) = write {
            if e.kind() == io::ErrorKind::BrokenPipe {
                self.dead = true;
                return Err(BridgeError::WorkerDied {
                    detail: self.reap(),
                });
            }
            return Err(BridgeError::Io(e.to_string()));
        }
        Ok(())
    }

    /// Wait up to `timeout` for the next frame. A timeout leaves the worker alive
    /// but the stream desynced, so the caller must treat the worker as unusable.
    fn recv_timeout(&mut self, timeout: Duration) -> Result<Response, BridgeError> {
        match self.rx.recv_timeout(timeout) {
            Ok(FromWorker::Frame(resp)) => Ok(resp),
            Ok(FromWorker::Ended(detail)) => {
                self.dead = true;
                Err(BridgeError::WorkerDied {
                    detail: format!("{detail} ({})", self.reap()),
                })
            }
            Err(RecvTimeoutError::Timeout) => Err(BridgeError::Timeout {
                millis: timeout.as_millis(),
            }),
            Err(RecvTimeoutError::Disconnected) => {
                self.dead = true;
                Err(BridgeError::WorkerDied {
                    detail: format!("reader thread ended ({})", self.reap()),
                })
            }
        }
    }

    /// Wait for the child and describe how it exited. Marks the child reaped so
    /// teardown does not wait on it again.
    fn reap(&mut self) -> String {
        let result = match self.child.wait() {
            Ok(status) => {
                if let Some(code) = status.code() {
                    format!("exited with code {code}")
                } else if let Some(sig) = status.signal() {
                    format!("killed by signal {sig}")
                } else {
                    format!("terminated ({status})")
                }
            }
            Err(e) => format!("wait() failed: {e}"),
        };
        self.reaped = true;
        result
    }

    /// Mark the worker unusable and force-kill it (after a timeout).
    fn kill_and_mark_dead(&mut self) {
        self.dead = true;
        let _ = self.child.kill();
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Best-effort teardown, but not *silent*: a cleanup failure is logged. Only
        // touch the child if it was not already reaped (a crash/timeout path already
        // waited it), so we do not mistake an already-reaped child's benign
        // kill/wait failure for a real fault.
        if !self.reaped {
            // Killing closes the child's stdout, unblocking the reader thread (EOF).
            if let Err(e) = self.child.kill() {
                log::error!("subprocess worker teardown: kill() failed: {e}");
            }
            if let Err(e) = self.child.wait() {
                log::error!("subprocess worker teardown: wait() failed: {e}");
            }
        }
        if let Some(reader) = self.reader.take() {
            if reader.join().is_err() {
                log::error!("subprocess worker teardown: reader thread panicked");
            }
        }
    }
}

/// Configuration for spawning a [`SubprocessBackend`].
#[derive(Debug, Clone)]
pub struct SubprocessConfig {
    /// The worker command as an argv (`argv[0]` is the program). Must be non-empty.
    pub command: Vec<String>,
    /// Working directory for the worker, or `None` to inherit ours.
    pub cwd: Option<String>,
    /// Extra environment variables set **on the child only** (the worker otherwise
    /// inherits our environment). A provider passes its own config here — e.g. an
    /// endpoint or credentials the SDK reads — without touching the parent's
    /// process-global environment.
    pub env: Vec<(String, String)>,
    /// Read timeout: how long to wait for a reply before declaring the worker
    /// unresponsive.
    pub recv_timeout: Duration,
    /// Arm the `PR_SET_PDEATHSIG` orphan guard (Linux only; ignored elsewhere at
    /// runtime — the guard is set via `prctl`, a no-op path on non-Linux is not
    /// compiled here since Polypus targets Linux/CESGA).
    pub arm_pdeathsig: bool,
}

impl Default for SubprocessConfig {
    fn default() -> Self {
        SubprocessConfig {
            command: Vec::new(),
            cwd: None,
            env: Vec::new(),
            recv_timeout: Duration::from_millis(DEFAULT_RECV_TIMEOUT_MS),
            arm_pdeathsig: true,
        }
    }
}

/// A [`QuantumBackend`] bridging to a Python worker subprocess.
pub struct SubprocessBackend {
    /// The worker behind a mutex: `run_circuits` serialises access to the pipe. A
    /// single worker services one call at a time.
    worker: Mutex<Worker>,
    /// The worker's PID, copied out so [`cancel`](Self::cancel) can signal it
    /// **without taking the pipe lock** (which an in-flight call holds).
    pid: u32,
    /// Whether a cancellation signal was already sent (so a repeated `cancel` is a
    /// no-op rather than a second SIGINT).
    cancel_sent: AtomicBool,
    recv_timeout: Duration,
}

impl SubprocessBackend {
    /// Spawn the worker and complete the protocol handshake.
    pub fn spawn(config: SubprocessConfig) -> Result<SubprocessBackend, BridgeError> {
        let mut worker = Worker::spawn(
            &config.command,
            config.cwd.as_deref(),
            &config.env,
            config.arm_pdeathsig,
        )?;
        let pid = worker.pid();
        // Handshake, bounded by the read timeout so a worker that never answers at
        // startup does not hang construction.
        worker.send(&Request::Hello {
            protocol: PROTOCOL_VERSION,
        })?;
        match worker.recv_timeout(config.recv_timeout)? {
            Response::Ready { protocol } if protocol == PROTOCOL_VERSION => {}
            Response::Ready { protocol } => {
                return Err(BridgeError::Handshake(format!(
                    "worker speaks protocol v{protocol}, this bridge speaks v{PROTOCOL_VERSION}"
                )));
            }
            other => {
                return Err(BridgeError::Handshake(format!(
                    "expected a 'ready' frame, got {other:?}"
                )));
            }
        }
        Ok(SubprocessBackend {
            worker: Mutex::new(worker),
            pid,
            cancel_sent: AtomicBool::new(false),
            recv_timeout: config.recv_timeout,
        })
    }

    /// Build a [`SubprocessBackend`] from a registry [`BackendBuildContext`].
    ///
    /// Options read: `command` (required, argv split on whitespace),
    /// `recv_timeout_ms` (default [`DEFAULT_RECV_TIMEOUT_MS`]; a present-but-malformed
    /// value is an error, not a fallback), `arm_pdeathsig` (default true; false
    /// spellings, case-insensitive: `false`/`0`/`no`/`off`), `cwd` (optional).
    pub fn from_context(
        ctx: &BackendBuildContext,
    ) -> Result<Arc<dyn QuantumBackend>, BackendError> {
        let config = config_from_context(ctx)?;
        let backend = SubprocessBackend::spawn(config)?;
        Ok(Arc::new(backend))
    }
}

/// Parse a [`SubprocessConfig`] from a registry [`BackendBuildContext`]'s options.
///
/// Split out from [`SubprocessBackend::from_context`] so the option parsing (which
/// validates rather than silently defaulting a malformed value) is unit-testable
/// without spawning a process.
fn config_from_context(ctx: &BackendBuildContext) -> Result<SubprocessConfig, BackendError> {
    let command_str = ctx.option("command").ok_or_else(|| {
        BackendError::Conversion(
            "the 'subprocess' backend requires a 'command' option (the worker argv)".to_string(),
        )
    })?;
    let command: Vec<String> = command_str.split_whitespace().map(str::to_string).collect();
    if command.is_empty() {
        return Err(BackendError::Conversion(
            "the 'subprocess' backend's 'command' option is empty".to_string(),
        ));
    }
    // A malformed value is a configuration mistake — surface it, don't silently
    // fall back to the default (which only an *absent* key uses).
    let recv_timeout_ms = match ctx.option("recv_timeout_ms") {
        None => DEFAULT_RECV_TIMEOUT_MS,
        Some(v) => v.parse::<u64>().map_err(|_| {
            BackendError::Conversion(format!(
                "the 'subprocess' backend's 'recv_timeout_ms' option must be a non-negative \
                 integer (milliseconds), got {v:?}"
            ))
        })?,
    };
    // Boolean option: absent → default true. Present → true unless it is a
    // recognised false spelling. Case- and whitespace-insensitive; the false
    // values are "false", "0", "no", "off".
    let arm_pdeathsig = match ctx.option("arm_pdeathsig") {
        None => true,
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "false" | "0" | "no" | "off"
        ),
    };
    Ok(SubprocessConfig {
        command,
        cwd: ctx.option("cwd").map(str::to_string),
        env: Vec::new(),
        recv_timeout: Duration::from_millis(recv_timeout_ms),
        arm_pdeathsig,
    })
}

/// Register this backend under [`BACKEND_NAME`] in the process-wide registry, so the
/// Polypus factory (or any name-driven path) can build it. Idempotent — safe to call
/// more than once (last registration wins).
pub fn register() {
    register_backend(BACKEND_NAME, SubprocessBackend::from_context);
}

/// Serialise a [`BoundCircuit`] to the wire form. A `Foreign` provider object cannot
/// cross a process boundary, so it is rejected up front.
fn to_wire_circuit(bc: &BoundCircuit) -> Result<Circuit, BackendError> {
    match bc {
        BoundCircuit::Native(cc) => Ok(Circuit {
            qasm: cc.to_qasm2(),
            n_qubits: cc.num_qubits as u32,
        }),
        BoundCircuit::Qasm2(qasm) => Ok(Circuit {
            qasm: qasm.clone(),
            n_qubits: bc.native_qubit_width().unwrap_or(0) as u32,
        }),
        BoundCircuit::Foreign(_) => Err(BackendError::UnsupportedCircuit(
            "the subprocess bridge cannot serialise a provider-native (Foreign) circuit across a \
             process boundary; submit Native or Qasm2"
                .to_string(),
        )),
    }
}

impl QuantumBackend for SubprocessBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        params: &RunParams,
    ) -> Result<Vec<Counts>, BackendError> {
        let circuits: Vec<Circuit> = qcs.iter().map(to_wire_circuit).collect::<Result<_, _>>()?;
        // A fresh call clears any prior cancel latch, so a new cancel is delivered.
        self.cancel_sent.store(false, Ordering::SeqCst);
        let mut worker = self.worker.lock().unwrap_or_else(|p| p.into_inner());
        worker.send(&Request::Run {
            id: params.id.clone(),
            circuits,
            shots: params.shots,
            seed: params.seed,
        })?;
        match worker.recv_timeout(self.recv_timeout) {
            Ok(Response::Result { counts, .. }) => Ok(counts),
            Ok(Response::Aborted { .. }) => Err(BridgeError::Aborted.into()),
            Ok(Response::Error { message, .. }) => Err(BridgeError::Worker(message).into()),
            Ok(Response::Ready { .. }) => {
                Err(BridgeError::Protocol("unexpected 'ready' during a run".to_string()).into())
            }
            Err(e @ BridgeError::Timeout { .. }) => {
                // A hung worker leaves the stream desynced: kill it so a later call
                // fails fast rather than reading a stale late reply.
                worker.kill_and_mark_dead();
                Err(e.into())
            }
            Err(e) => Err(e.into()),
        }
    }

    fn cancel(&self) {
        // Out-of-band: signal the worker WITHOUT taking the pipe lock (an in-flight
        // call holds it). SIGINT unblocks a worker parked in the bridge's recv; the
        // worker's handler aborts and replies `aborted`. Sent at most once per call.
        if self
            .cancel_sent
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            // Deliver SIGINT to the worker by pid, without the pipe lock (an
            // in-flight call holds it), via `nix`'s safe `kill` wrapper — no `unsafe`.
            // std can only `Child::kill()` (SIGKILL, and via `&mut`), which does not
            // fit here.
            if let Err(errno) = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(self.pid as i32),
                nix::sys::signal::Signal::SIGINT,
            ) {
                log::debug!(
                    "subprocess backend: kill(SIGINT) on worker {} failed: {errno}",
                    self.pid,
                );
            }
        }
    }

    fn wants_cancel_watcher(&self) -> bool {
        true
    }

    fn close(&self) {
        let mut worker = self.worker.lock().unwrap_or_else(|p| p.into_inner());
        if !worker.dead {
            let _ = worker.send(&Request::Shutdown);
        }
        // The Drop on the inner Worker kills + reaps + joins regardless.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_error_maps_dead_and_hung_to_unresponsive() {
        let dead: BackendError = BridgeError::WorkerDied {
            detail: "killed by signal 9".to_string(),
        }
        .into();
        assert!(matches!(dead, BackendError::Unresponsive(_)));

        let hung: BackendError = BridgeError::Timeout { millis: 300 }.into();
        assert!(matches!(hung, BackendError::Unresponsive(_)));
    }

    #[test]
    fn bridge_error_maps_worker_error_to_external() {
        let worker: BackendError = BridgeError::Worker("bad input".to_string()).into();
        assert!(matches!(worker, BackendError::External(_)));
    }

    #[test]
    fn bridge_error_maps_aborted_to_the_contract_aborted_variant() {
        // A worker abort is a cancellation, not a generic provider error: it must map
        // to `BackendError::Aborted` (→ KeyboardInterrupt at the edge), not `External`.
        let aborted: BackendError = BridgeError::Aborted.into();
        assert!(
            matches!(aborted, BackendError::Aborted(_)),
            "BridgeError::Aborted must map to BackendError::Aborted, got {aborted:?}"
        );
    }

    #[test]
    fn foreign_circuit_is_rejected_without_a_worker() {
        #[derive(Debug)]
        struct F;
        impl polypus_backend::ForeignCircuit for F {
            fn clone_boxed(&self) -> Box<dyn polypus_backend::ForeignCircuit> {
                Box::new(F)
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        let err = to_wire_circuit(&BoundCircuit::Foreign(Box::new(F))).unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedCircuit(_)));
    }

    fn ctx(options: &[(&str, &str)]) -> BackendBuildContext {
        BackendBuildContext {
            id: "cfg-test".to_string(),
            shots: 1,
            n_qpus: 1,
            seed: None,
            opt_level: Default::default(),
            options: options
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    #[test]
    fn missing_command_is_rejected() {
        match config_from_context(&ctx(&[])) {
            Err(BackendError::Conversion(m)) => assert!(m.contains("command")),
            other => panic!("expected a Conversion error naming 'command', got {other:?}"),
        }
    }

    #[test]
    fn malformed_recv_timeout_ms_is_an_error_not_a_default() {
        // A present-but-unparseable value must error, not silently use the default.
        match config_from_context(&ctx(&[
            ("command", "python3 w.py"),
            ("recv_timeout_ms", "soon"),
        ])) {
            Err(BackendError::Conversion(m)) => assert!(m.contains("recv_timeout_ms")),
            other => panic!("expected a Conversion error, got {other:?}"),
        }
        // An absent value uses the default.
        let cfg = config_from_context(&ctx(&[("command", "python3 w.py")])).unwrap();
        assert_eq!(cfg.recv_timeout.as_millis() as u64, DEFAULT_RECV_TIMEOUT_MS);
        // A well-formed value is honoured.
        let cfg = config_from_context(&ctx(&[
            ("command", "python3 w.py"),
            ("recv_timeout_ms", "1500"),
        ]))
        .unwrap();
        assert_eq!(cfg.recv_timeout.as_millis(), 1500);
    }

    #[test]
    fn arm_pdeathsig_parsing_is_case_and_whitespace_insensitive() {
        let armed = |v: &str| {
            config_from_context(&ctx(&[("command", "python3 w.py"), ("arm_pdeathsig", v)]))
                .unwrap()
                .arm_pdeathsig
        };
        // Recognised false spellings (any case, trimmed) → disarmed.
        for v in ["false", "FALSE", "False", " off ", "No", "0"] {
            assert!(!armed(v), "{v:?} should disable the orphan guard");
        }
        // Anything else, and absence, → armed (default true).
        for v in ["true", "TRUE", "yes", "1", "anything"] {
            assert!(armed(v), "{v:?} should leave the orphan guard armed");
        }
        assert!(
            config_from_context(&ctx(&[("command", "python3 w.py")]))
                .unwrap()
                .arm_pdeathsig,
            "an absent arm_pdeathsig defaults to armed"
        );
    }
}
