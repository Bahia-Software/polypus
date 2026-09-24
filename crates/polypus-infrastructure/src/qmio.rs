//! CESGA **QMIO** real-QPU backend.
//!
//! [`QmioBackend`] talks to the QMIO quantum computer directly over its ZeroMQ
//! endpoint, **without going through the Python interpreter**. For the
//! [`BoundCircuit::Native`] and [`BoundCircuit::Qasm2`] variants the whole
//! round-trip — circuit serialisation, request framing, response parsing — runs
//! in Rust, so it never acquires the GIL. (A [`BoundCircuit::Foreign`] Qiskit object cannot be
//! read without the interpreter and is rejected with an actionable message.)
//!
//! # Wire protocol (verified against the `qmio` 0.1.3 client)
//!
//! - **Transport:** a ZeroMQ **REQ** socket connected to `ZMQ_SERVER`
//!   (e.g. `tcp://10.133.29.226:5556`). REQ is strict lock-step: exactly one
//!   request in flight, one reply, repeat. We use the pure-Rust `zeromq` crate
//!   (ZMTP 3.x, interoperable with the server's `pyzmq`/`libzmq`).
//! - **Request payload** = a Python **pickle** of the 2-tuple `(circuit, config)`
//!   — exactly what `socket.send_pyobj((circuit, config))` produces. It is *not*
//!   plain text or JSON: it is the binary pickle format.
//!   - `circuit` = the program: an OpenQASM / QIR text **string**, or QIR
//!     bitcode **bytes**.
//!   - `config` = a **JSON string** (`json.dumps`) that mimics
//!     `qat.purr.compiler.config.CompilerConfig` via `$type`/`$data`/`$value`
//!     tags. `qat` does **not** need to be installed; we build it by hand with
//!     `serde_json` (see [`build_config_json`]).
//! - **Reply** = a pickle of the results (the live QPU pickles a JSON string;
//!   a server that pickles a `dict` directly instead is also accepted as a
//!   defensive fallback). Decode path: `recv` bytes → `serde_pickle` →
//!   `String`/`dict` → `serde_json::Value` → per-bitstring counts.
//!
//! ## Security
//!
//! `serde-pickle` only *builds data* — it never executes `__reduce__`/`GLOBAL`
//! opcodes — so a malicious or buggy peer cannot achieve remote code execution
//! the way Python's `pickle.loads` can. We additionally cap the accepted reply
//! size ([`QmioBackend::max_reply_bytes`]). The ZMQ traffic itself is in clear
//! text over a trusted private HPC network (CURVE/auth is out of scope).
//!
//! # Points to verify against the live QPU (parametrised TODOs — do not invent)
//!
//! 1. **Reply JSON schema** for `binary_count`, verified against the live QPU:
//!    `{"results": {"<register>": {"<bitstring>": count}}, "execution_metrics":
//!    {…}}`. [`counts_from_json`] reads this register-grouped shape and keeps
//!    defensive fallbacks for flatter layouts.
//! 2. **Bit/qubit order** of the returned bitstrings vs Polypus' Qiskit
//!    little-endian convention. [`normalize_bitstring`] is the single hook to
//!    flip it.
//! 3. **Acceptance of QIR `.ll` text and `.bc` bitcode** by the server's
//!    compiler, and whether an extra header/flag is required.
//! 4. **Pickle protocol** the server's Python can load. We write protocol 3
//!    (`SerOptions::new()`), which any Python 3 loads and which natively encodes
//!    `bytes` (needed for the bitcode path); switch to `.proto_v2()` only for a
//!    Python-2 server.
//! 5. **`n_qpus > 1` mapping**: there is a single endpoint, so QMIO is treated as
//!    one QPU (`capabilities().max_concurrency` is `1`).
//! 6. **OpenQASM header**: Polypus exports and submits an `OPENQASM 2.0` program
//!    (header and body), which matches the 2.0-style body the QMIO examples use.
//!    Verify acceptance against the live QPU (point 6).

use crate::error::BackendError;
use crate::{record_cleanup_failure, BoundCircuit, QuantumBackend, RunParams};
use polypus_backend::BackendBuildContext;
use polypus_circuit::{CircuitError, ConcreteCircuit, ParameterizedCircuit};
use serde_json::json;
use serde_pickle::{DeOptions, SerOptions, Value as PickleValue};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use zeromq::{ReqSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

pub use crate::execution_config::QmioProgramFormat;

/// Registry factory for the QMIO backend — the bridge between the pyo3-free runtime
/// registry ([`polypus_backend::register_backend`]) and [`QmioBackend::new`].
///
/// Registered under the name `"qmio"` by `register_builtin_backends` (only with
/// `--features qmio`). It reads its configuration from the
/// [`BackendBuildContext`]'s `options`, mirroring the former typed
/// `BackendConfig::Qmio`:
///
/// | option | meaning | default |
/// |--------|---------|---------|
/// | `endpoint` | ZMQ REQ endpoint | the built-in default endpoint |
/// | `program_format` | `openqasm` / `qir_text` / `qir_bitcode` | `openqasm` |
/// | `optimization` | Tket optimisation level (u8) | `0` |
/// | `repetition_period` | seconds (f64), or absent for the server default | absent |
/// | `res_format` | results format | `binary_count` |
///
/// The provider-specific [`QmioError`] is boxed into
/// [`BackendError::External`], keeping the registry contract pyo3-free; the FFI edge
/// downcasts it back to the typed `polypus.QmioError`.
pub fn qmio_factory(ctx: &BackendBuildContext) -> Result<Arc<dyn QuantumBackend>, BackendError> {
    let endpoint = ctx.option("endpoint").unwrap_or("").to_string();
    let program_format = match ctx.option("program_format").unwrap_or("openqasm") {
        "openqasm" => QmioProgramFormat::OpenQasm,
        "qir_text" => QmioProgramFormat::QirText,
        "qir_bitcode" => QmioProgramFormat::QirBitcode,
        other => {
            return Err(BackendError::Conversion(format!(
                "unknown qmio program_format '{other}'; expected \"openqasm\", \"qir_text\", or \
                 \"qir_bitcode\""
            )))
        }
    };
    // Present-but-malformed numeric options are a configuration mistake — surface
    // them, like the unknown-program_format branch above, rather than silently
    // falling back to the default (which only an *absent* key uses).
    let optimization = match ctx.option("optimization") {
        None => 0,
        Some(v) => v.parse::<u8>().map_err(|_| {
            BackendError::Conversion(format!(
                "qmio 'optimization' must be an integer 0-255, got {v:?}"
            ))
        })?,
    };
    let repetition_period = match ctx.option("repetition_period") {
        None => None,
        Some(v) => Some(v.parse::<f64>().map_err(|_| {
            BackendError::Conversion(format!(
                "qmio 'repetition_period' must be a number of seconds, got {v:?}"
            ))
        })?),
    };
    let res_format = ctx
        .option("res_format")
        .unwrap_or("binary_count")
        .to_string();
    let backend = QmioBackend::new(
        endpoint,
        program_format,
        optimization,
        repetition_period,
        res_format,
    )
    .map_err(|e| BackendError::External(Box::new(e)))?;
    Ok(Arc::new(backend))
}

/// Default endpoint, used only when `ZMQ_SERVER` is unset. Documented fallback,
/// never silently hard-coded over an explicit configuration.
const DEFAULT_ENDPOINT: &str = "tcp://10.255.3.70:5556";

/// A serialised program ready to be pickled as the first tuple element.
///
/// Text programs pickle to a Python `str`; bitcode pickles to Python `bytes`.
#[derive(Debug)]
enum ProgramPayload {
    Text(String),
    Bytes(Vec<u8>),
}

/// Errors raised on the QMIO network/serialisation path.
///
/// These bubble up through [`QuantumBackend::run_circuits`] as a
/// [`BackendError::External`] (a boxed `QmioError`), which the FFI boundary maps to the typed
/// `polypus.QmioError` Python exception — never a panic. The enum keeps its own
/// rich variants (verified against the wire protocol) instead of being
/// flattened into [`BackendError`]; see [`crate::error`] for the
/// crate-wide granularity decision.
#[derive(Debug)]
pub enum QmioError {
    /// A Qiskit `QuantumCircuit` reached the GIL-free QMIO path.
    UnsupportedCircuit,
    /// An OpenQASM 2.0 program could not be parsed back into a circuit.
    Circuit(String),
    /// QIR bitcode assembly failed (typically `llvm-as` missing from `PATH`).
    QirBitcode(String),
    /// Invalid results-format / optimisation selection.
    Config(String),
    /// Pickle (de)serialisation failed.
    Pickle(String),
    /// The reply JSON could not be parsed.
    Json(String),
    /// The reply did not match an expected results schema.
    Schema(String),
    /// Could not establish the ZMQ connection.
    Connect { endpoint: String, source: String },
    /// Sending the request failed.
    Send(String),
    /// Receiving the reply failed.
    Recv(String),
    /// No reply within the configured timeout, after all retries.
    Timeout {
        endpoint: String,
        attempts: usize,
        millis: u128,
    },
    /// The request was successfully sent but no reply was received (a receive
    /// timeout or error). Unlike a connect/send failure — where the request never
    /// left our socket and can be retried safely — the QPU may already have
    /// *executed* this request, so it is deliberately **not** resent: a blind
    /// retry would risk a double execution. Surfaced as a distinct terminal
    /// error so the caller can decide whether re-submitting is safe.
    ResultUnknown {
        endpoint: String,
        /// What went wrong on receive (a timeout, or the underlying error).
        detail: String,
    },
    /// The reply exceeded [`QmioBackend::max_reply_bytes`].
    ResponseTooLarge { bytes: usize, limit: usize },
    /// The dedicated Tokio runtime could not be built at construction time.
    Runtime(String),
    /// An internal invariant of the request loop did not hold (e.g. the socket
    /// was unexpectedly absent after a successful connect). Returned rather than
    /// panicked so it can never abort the process; the request is retried.
    Internal(String),
}

impl fmt::Display for QmioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QmioError::UnsupportedCircuit => write!(
                f,
                "the QMIO backend cannot serialize a Qiskit QuantumCircuit without the \
                 interpreter; pass a polypus.Circuit or an OpenQASM 2.0 string"
            ),
            QmioError::Circuit(m) => write!(f, "could not build a circuit for QMIO: {m}"),
            QmioError::QirBitcode(m) => write!(f, "QIR bitcode export failed: {m}"),
            QmioError::Config(m) => write!(f, "invalid QMIO config: {m}"),
            QmioError::Pickle(m) => write!(f, "pickle error: {m}"),
            QmioError::Json(m) => write!(f, "reply JSON error: {m}"),
            QmioError::Schema(m) => write!(f, "unexpected QMIO reply schema: {m}"),
            QmioError::Connect { endpoint, source } => {
                write!(f, "could not connect to QMIO endpoint {endpoint}: {source}")
            }
            QmioError::Send(m) => write!(f, "failed to send request to QMIO: {m}"),
            QmioError::Recv(m) => write!(f, "failed to receive reply from QMIO: {m}"),
            QmioError::Timeout { endpoint, attempts, millis } => write!(
                f,
                "no reply from QMIO endpoint {endpoint} within {millis} ms after {attempts} attempt(s)"
            ),
            QmioError::ResultUnknown { endpoint, detail } => write!(
                f,
                "the request to QMIO endpoint {endpoint} was delivered but no reply was received \
                 ({detail}); the QPU may have executed it, so it was not resent (to avoid a double \
                 execution)"
            ),
            QmioError::ResponseTooLarge { bytes, limit } => write!(
                f,
                "QMIO reply of {bytes} bytes exceeds the {limit}-byte safety limit"
            ),
            QmioError::Runtime(m) => {
                write!(f, "could not build the Tokio runtime for the QMIO backend: {m}")
            }
            QmioError::Internal(m) => write!(f, "internal QMIO backend error: {m}"),
        }
    }
}

impl std::error::Error for QmioError {}

/// Backend that executes circuits on the CESGA QMIO QPU over ZeroMQ.
///
/// The REQ socket is created lazily and held behind a [`tokio::sync::Mutex`] so
/// the backend is `Send + Sync` (required by [`QuantumBackend`]) even though a
/// single REQ socket is inherently serial — which is exactly the semantics we
/// want: one request at a time. An **async** mutex (not [`std::sync::Mutex`]) is
/// deliberate: [`run_one`](Self::run_one) holds the guard across the request's
/// `.await` points (connect/send/recv and the retry backoff), so a second
/// concurrent caller must wait cooperatively — suspending its task rather than
/// blocking an OS thread — as required by `ENGINEERING.md` §9 ("Don't hold a
/// lock across an `.await`", which a `std::sync::Mutex` here would violate).
/// On a network fault the socket is dropped and recreated (the "Lazy Pirate"
/// pattern: a REQ socket is unusable after a failed `recv`).
pub struct QmioBackend {
    endpoint: String,
    program_format: QmioProgramFormat,
    optimization: u8,
    repetition_period: Option<f64>,
    res_format: String,
    /// Dedicated Tokio runtime driving the async `zeromq` sockets. Owned by the
    /// backend so the socket's background tasks outlive individual calls.
    runtime: Runtime,
    /// Lazily-created REQ socket; `None` until the first request or after a
    /// fault forces a reconnect.
    socket: Mutex<Option<ReqSocket>>,
    /// Per-request receive timeout.
    recv_timeout: Duration,
    /// Base delay applied (linearly) between reconnect attempts.
    retry_backoff: Duration,
    /// Reconnect+retry attempts before giving up on a request.
    max_retries: usize,
    /// Hard deadline for the whole request across all reconnect/retry attempts —
    /// a global wall-clock cap on top of the per-operation `recv_timeout`, so no
    /// request (including a pathologically slow reconnect) can run unboundedly.
    global_timeout: Duration,
    /// Hard cap on an accepted reply, guarding against a hostile/buggy peer.
    max_reply_bytes: usize,
    /// Idempotency guard for [`close`](Self::close)/[`Drop`].
    closed: AtomicBool,
}

impl QmioBackend {
    /// Create a backend targeting `endpoint`.
    ///
    /// Timeouts and retry behaviour can be overridden via the `QMIO_RECV_TIMEOUT_MS`,
    /// `QMIO_MAX_RETRIES` and `QMIO_RETRY_BACKOFF_MS` environment variables
    /// (sensible defaults otherwise).
    pub fn new(
        endpoint: String,
        program_format: QmioProgramFormat,
        optimization: u8,
        repetition_period: Option<f64>,
        res_format: String,
    ) -> Result<Self, QmioError> {
        let endpoint = if endpoint.is_empty() {
            DEFAULT_ENDPOINT.to_string()
        } else {
            endpoint
        };
        // A single worker thread is enough (REQ is serial) but a multi-thread
        // runtime guarantees the socket's background IO tasks make progress
        // independently of the `block_on` call driving a request.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .map_err(|e| QmioError::Runtime(e.to_string()))?;
        let recv_timeout = Duration::from_millis(env_u64("QMIO_RECV_TIMEOUT_MS", 300_000));
        let max_retries = env_u64("QMIO_MAX_RETRIES", 3) as usize;
        let retry_backoff = Duration::from_millis(env_u64("QMIO_RETRY_BACKOFF_MS", 100));
        // Global deadline default: a generous worst-case bound so it never cuts a
        // legitimate reconnect/retry sequence short, only a pathological hang.
        // Each attempt bounds connect + send + recv at `recv_timeout` each (3x),
        // over `max_retries + 1` attempts, plus the linear-backoff sum. Overridable
        // via QMIO_GLOBAL_TIMEOUT_MS to impose a tighter cap.
        let attempts = max_retries as u64 + 1;
        let backoff_sum =
            retry_backoff.as_millis() as u64 * (max_retries as u64 * (max_retries as u64 + 1) / 2);
        let default_global = attempts
            .saturating_mul(3)
            .saturating_mul(recv_timeout.as_millis() as u64)
            .saturating_add(backoff_sum);
        let global_timeout =
            Duration::from_millis(env_u64("QMIO_GLOBAL_TIMEOUT_MS", default_global));
        Ok(QmioBackend {
            endpoint,
            program_format,
            optimization,
            repetition_period,
            res_format,
            runtime,
            socket: Mutex::new(None),
            recv_timeout,
            retry_backoff,
            max_retries,
            global_timeout,
            // 64 MiB: far above any realistic counts payload, far below "OOM".
            max_reply_bytes: 64 * 1024 * 1024,
            closed: AtomicBool::new(false),
        })
    }

    /// Override the timeouts and retry budget on a constructed backend. Test-only:
    /// lets the simulated-server tests drive fast, deterministic timeout/retry
    /// behaviour without depending on process-wide environment variables.
    #[cfg(test)]
    fn with_test_timeouts(mut self, recv_ms: u64, max_retries: usize, global_ms: u64) -> Self {
        self.recv_timeout = Duration::from_millis(recv_ms);
        self.max_retries = max_retries;
        self.global_timeout = Duration::from_millis(global_ms);
        self
    }

    /// Serialise one bound circuit into the program payload for the configured
    /// [`QmioProgramFormat`].
    fn serialize_program(&self, circuit: &BoundCircuit) -> Result<ProgramPayload, QmioError> {
        match self.program_format {
            QmioProgramFormat::OpenQasm => Ok(ProgramPayload::Text(self.qasm_text(circuit)?)),
            QmioProgramFormat::QirText => {
                Ok(ProgramPayload::Text(self.concrete(circuit)?.to_qir()))
            }
            QmioProgramFormat::QirBitcode => Ok(ProgramPayload::Bytes(self.qir_bitcode(circuit)?)),
        }
    }

    /// OpenQASM 2.0 text exported from the circuit, submitted as-is to the QMIO
    /// compiler (header and body both `OPENQASM 2.0`).
    fn qasm_text(&self, circuit: &BoundCircuit) -> Result<String, QmioError> {
        match circuit {
            BoundCircuit::Native(cc) => Ok(cc.to_qasm2()),
            BoundCircuit::Qasm2(s) => Ok(s.clone()),
            BoundCircuit::Foreign(_) => Err(QmioError::UnsupportedCircuit),
        }
    }

    /// QIR LLVM bitcode (`.bc`), mapping the `llvm-as`-missing case to an
    /// actionable error.
    fn qir_bitcode(&self, circuit: &BoundCircuit) -> Result<Vec<u8>, QmioError> {
        self.concrete(circuit)?
            .to_qir_bitcode()
            .map_err(|e| match e {
                CircuitError::QirAssemblyToolNotFound { tool } => QmioError::QirBitcode(format!(
                    "'{tool}' was not found on PATH; install LLVM (llvm-as) or select \
                 program_format \"openqasm\"/\"qir\""
                )),
                other => QmioError::QirBitcode(other.to_string()),
            })
    }

    /// Obtain a [`ConcreteCircuit`] without touching Python (parses an OpenQASM
    /// 2.0 string when necessary).
    fn concrete(&self, circuit: &BoundCircuit) -> Result<ConcreteCircuit, QmioError> {
        match circuit {
            BoundCircuit::Native(cc) => Ok(cc.clone()),
            BoundCircuit::Qasm2(s) => ParameterizedCircuit::from_qasm2(s)
                .and_then(|pc| pc.assign_parameters(&[]))
                .map_err(|e| QmioError::Circuit(e.to_string())),
            BoundCircuit::Foreign(_) => Err(QmioError::UnsupportedCircuit),
        }
    }

    /// Send one pickled request and parse the reply into counts, reconnecting
    /// and retrying on transient faults (Lazy Pirate).
    fn run_one(
        &self,
        program: &ProgramPayload,
        config_json: &str,
    ) -> Result<HashMap<String, u64>, QmioError> {
        // Full request payload (program text/bitcode + config): a large, per-call
        // diagnostic dump, so it stays behind `debug` rather than flooding the
        // default `info` log.
        log::debug!("QMIO request: {program:?}, config: {config_json}");
        let request = pickle_request(program, config_json)?;

        self.runtime.block_on(async {
            // The async mutex is acquired *inside* the runtime and held across the
            // request's `.await` points (ENGINEERING.md §9): a `tokio::sync::Mutex`
            // suspends a competing caller cooperatively instead of blocking an OS
            // thread, while still enforcing "one request at a time" over the single
            // REQ socket. Unlike `std::sync::Mutex` it has no poisoning, so there is
            // no poisoned-guard recovery to perform here.
            let mut guard = self.socket.lock().await;
            let deadline = Instant::now() + self.global_timeout;
            let mut last_err: Option<QmioError> = None;
            for attempt in 0..=self.max_retries {
                // Global deadline: never start another attempt once the whole-call
                // budget is spent. Each individual op below is separately bounded,
                // so this caps total wall time even across reconnect/retry.
                if Instant::now() >= deadline {
                    return Err(last_err.unwrap_or_else(|| QmioError::Timeout {
                        endpoint: self.endpoint.clone(),
                        attempts: attempt,
                        millis: self.global_timeout.as_millis(),
                    }));
                }
                // Linear backoff before a retry; the first attempt is immediate.
                if attempt > 0 {
                    // A retry means the previous attempt hit a transient fault
                    // and the REQ socket was discarded (Lazy Pirate). Surface it
                    // at `warn` — recoverable, but an operator should see the QPU
                    // link is flapping. Emitted before the `.await`, so the
                    // logger's lock is never held across a suspension point.
                    if let Some(err) = &last_err {
                        log::warn!(
                            "QMIO request to {} failed (attempt {}/{}), reconnecting and retrying: {err}",
                            self.endpoint,
                            attempt,
                            self.max_retries + 1
                        );
                    }
                    tokio::time::sleep(self.retry_backoff * attempt as u32).await;
                }
                // (Re)connect if we have no live socket, bounded by the timeout so
                // an unreachable or hung endpoint cannot block the whole call.
                if guard.is_none() {
                    match tokio::time::timeout(self.recv_timeout, connect(&self.endpoint)).await {
                        Ok(Ok(s)) => *guard = Some(s),
                        Ok(Err(e)) => {
                            last_err = Some(e);
                            continue;
                        }
                        Err(_elapsed) => {
                            last_err = Some(QmioError::Connect {
                                endpoint: self.endpoint.clone(),
                                source: format!(
                                    "connect timed out after {} ms",
                                    self.recv_timeout.as_millis()
                                ),
                            });
                            continue;
                        }
                    }
                }
                let socket = match guard.as_mut() {
                    Some(socket) => socket,
                    None => {
                        // Cannot happen (we just connected), but never panic:
                        // record it and let the loop reconnect and retry.
                        last_err = Some(QmioError::Internal(
                            "socket unexpectedly absent after connect".to_string(),
                        ));
                        continue;
                    }
                };

                // Send, bounded by the timeout. A REQ socket that fails or times
                // out while sending is unusable and must be discarded.
                match tokio::time::timeout(
                    self.recv_timeout,
                    socket.send(ZmqMessage::from(request.clone())),
                )
                .await
                {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        drop_socket(&mut guard).await;
                        last_err = Some(QmioError::Send(e.to_string()));
                        continue;
                    }
                    Err(_elapsed) => {
                        drop_socket(&mut guard).await;
                        last_err = Some(QmioError::Timeout {
                            endpoint: self.endpoint.clone(),
                            attempts: attempt + 1,
                            millis: self.recv_timeout.as_millis(),
                        });
                        continue;
                    }
                }

                // The request has now been delivered. Receive, bounded by the
                // timeout — but from here a failure means the QPU may already have
                // executed the request, so the result is UNKNOWN: we discard the
                // stuck socket but must NOT resend (a blind retry risks a double
                // execution). This is the send-failed vs result-unknown boundary.
                let socket = match guard.as_mut() {
                    Some(socket) => socket,
                    None => {
                        last_err = Some(QmioError::Internal(
                            "socket unexpectedly absent after send".to_string(),
                        ));
                        continue;
                    }
                };
                match tokio::time::timeout(self.recv_timeout, socket.recv()).await {
                    Ok(Ok(reply)) => {
                        // Raw wire reply (unparsed ZMQ frames): the noisiest,
                        // lowest-level dump, kept at `trace`.
                        log::trace!("QMIO reply: {reply:?}");
                        let bytes = first_frame(reply);
                        if bytes.len() > self.max_reply_bytes {
                            return Err(QmioError::ResponseTooLarge {
                                bytes: bytes.len(),
                                limit: self.max_reply_bytes,
                            });
                        }
                        return parse_counts(&bytes);
                    }
                    Ok(Err(e)) => {
                        drop_socket(&mut guard).await;
                        return Err(QmioError::ResultUnknown {
                            endpoint: self.endpoint.clone(),
                            detail: format!("receive failed: {e}"),
                        });
                    }
                    Err(_elapsed) => {
                        drop_socket(&mut guard).await;
                        return Err(QmioError::ResultUnknown {
                            endpoint: self.endpoint.clone(),
                            detail: format!("no reply within {} ms", self.recv_timeout.as_millis()),
                        });
                    }
                }
            }
            Err(last_err.unwrap_or_else(|| QmioError::Timeout {
                endpoint: self.endpoint.clone(),
                attempts: self.max_retries + 1,
                millis: self.global_timeout.as_millis(),
            }))
        })
    }
}

impl QuantumBackend for QmioBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        config: &RunParams,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
        self.run_all(qcs, config).map_err(|e| {
            log::error!("QMIO backend error talking to {}: {e}", self.endpoint);
            // The provider-specific QMIO error crosses the pyo3-free backend
            // contract type-erased in `External`; the FFI edge downcasts it back to
            // raise the typed `polypus.QmioError`.
            BackendError::External(Box::new(e))
        })
    }

    fn capabilities(&self) -> super::BackendCapabilities {
        // One circuit per call over a single REQ endpoint.
        super::BackendCapabilities {
            max_concurrency: 1,
            supports_shot_distribution: true,
        }
    }

    fn close(&self) {
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        // Graceful close must run inside the runtime context; the async mutex is
        // therefore locked inside `block_on` (its `.lock()` is a future). There is
        // no poisoning to recover from with `tokio::sync::Mutex`, and this stays
        // panic-free — `Drop` requires it. `close` returns any errors it hit
        // instead of a `Result`; log (never discard) them and record the failed
        // cleanup.
        let errors = self.runtime.block_on(async {
            let mut guard = self.socket.lock().await;
            match guard.take() {
                Some(socket) => socket.close().await,
                None => Vec::new(),
            }
        });
        if !errors.is_empty() {
            log::error!(
                "QMIO socket close reported {} error(s): {errors:?}",
                errors.len()
            );
            record_cleanup_failure();
        }
    }
}

impl QmioBackend {
    /// Execute every circuit in the batch, returning the first failure as a
    /// [`QmioError`]. Split out from `run_circuits` so the whole batch is a
    /// single `?`-threaded `Result` with no per-circuit panic.
    fn run_all(
        &self,
        qcs: &[BoundCircuit],
        config: &RunParams,
    ) -> Result<Vec<HashMap<String, u64>>, QmioError> {
        // The config JSON is identical for every circuit in the batch.
        let config_value = build_config_json(
            config.shots,
            self.repetition_period,
            &self.res_format,
            self.optimization,
        )?;
        let config_json =
            serde_json::to_string(&config_value).map_err(|e| QmioError::Json(e.to_string()))?;

        // One REQ/REP exchange per circuit: the endpoint is a single QPU.
        qcs.iter()
            .map(|qc| {
                let program = self.serialize_program(qc)?;
                self.run_one(&program, &config_json)
            })
            .collect()
    }
}

/// RAII: make sure the socket is closed even if the caller forgets to.
impl Drop for QmioBackend {
    fn drop(&mut self) {
        self.close();
    }
}

/// Connect a fresh REQ socket to `endpoint`.
async fn connect(endpoint: &str) -> Result<ReqSocket, QmioError> {
    let mut socket = ReqSocket::new();
    socket
        .connect(endpoint)
        .await
        .map_err(|e| QmioError::Connect {
            endpoint: endpoint.to_string(),
            source: e.to_string(),
        })?;
    Ok(socket)
}

/// Discard a faulted REQ socket (gracefully, inside the runtime). The next
/// request will transparently reconnect.
async fn drop_socket(guard: &mut Option<ReqSocket>) {
    if let Some(socket) = guard.take() {
        // Don't discard the close errors silently (ENGINEERING.md §9): this is a
        // transient reconnect on a faulted socket, so they are informational.
        let errors = socket.close().await;
        if !errors.is_empty() {
            log::debug!("QMIO faulted-socket close reported errors: {errors:?}");
        }
    }
}

/// Extract the first frame of a reply as raw bytes.
fn first_frame(reply: ZmqMessage) -> Vec<u8> {
    reply
        .into_vec()
        .into_iter()
        .next()
        .map(|frame| frame.to_vec())
        .unwrap_or_default()
}

/// Pickle the `(circuit, config)` 2-tuple.
///
/// Built explicitly from [`PickleValue`] so a text program pickles to a Python
/// `str` and bitcode pickles to Python `bytes` (a `Vec<u8>` would otherwise
/// serialise as a list/tuple of integers).
fn pickle_request(program: &ProgramPayload, config_json: &str) -> Result<Vec<u8>, QmioError> {
    let program_value = match program {
        ProgramPayload::Text(s) => PickleValue::String(s.clone()),
        ProgramPayload::Bytes(b) => PickleValue::Bytes(b.clone()),
    };
    let tuple = PickleValue::Tuple(vec![
        program_value,
        PickleValue::String(config_json.to_string()),
    ]);
    // Protocol 3 (SerOptions default): loadable by any Python 3 and natively
    // encodes `bytes` (needed for the bitcode path).
    serde_pickle::value_to_vec(&tuple, SerOptions::new())
        .map_err(|e| QmioError::Pickle(e.to_string()))
}

/// Decode a pickled reply into per-bitstring counts.
fn parse_counts(reply: &[u8]) -> Result<HashMap<String, u64>, QmioError> {
    let value = serde_pickle::value_from_slice(reply, DeOptions::new())
        .map_err(|e| QmioError::Pickle(e.to_string()))?;
    match value {
        // Expected shape: a pickled JSON string.
        PickleValue::String(json_str) => {
            let json: serde_json::Value =
                serde_json::from_str(&json_str).map_err(|e| QmioError::Json(e.to_string()))?;
            counts_from_json(&json)
        }
        // Defensive: a server might pickle a dict directly instead of a JSON
        // string. Convert it to JSON and reuse the same extraction.
        other => {
            let json = pickle_to_json(&other)?;
            counts_from_json(&json)
        }
    }
}

/// Best-effort conversion of a non-string pickle reply into a JSON value, so the
/// same [`counts_from_json`] extraction can handle a dict-shaped reply.
fn pickle_to_json(value: &PickleValue) -> Result<serde_json::Value, QmioError> {
    match value {
        PickleValue::Dict(entries) => {
            let mut map = serde_json::Map::new();
            for (k, v) in entries {
                let key = match k {
                    serde_pickle::HashableValue::String(s) => s.clone(),
                    serde_pickle::HashableValue::I64(i) => i.to_string(),
                    other => format!("{other:?}"),
                };
                map.insert(key, pickle_to_json(v)?);
            }
            Ok(serde_json::Value::Object(map))
        }
        PickleValue::I64(i) => Ok(json!(i)),
        PickleValue::F64(x) => Ok(json!(x)),
        PickleValue::String(s) => Ok(json!(s)),
        other => Err(QmioError::Schema(format!(
            "reply was neither a JSON string nor a counts dict (got {other:?})"
        ))),
    }
}

/// Locate the bitstring→count object inside the reply JSON.
///
/// The QMIO/`qat` reply groups the counts one level below `results`, keyed by
/// the classical register name — verified against the live QPU, e.g.
/// `{"results": {"c": {"00": 1000}}}`. We also accept a flat `{bitstring:
/// count}` object or one nested directly under a common container key, and fall
/// back to the first nested counts-like object.
fn counts_from_json(value: &serde_json::Value) -> Result<HashMap<String, u64>, QmioError> {
    if let Some(map) = as_counts_object(value) {
        return Ok(map);
    }
    if let Some(obj) = value.as_object() {
        // Container key whose value is *directly* a `{bitstring: count}` object.
        for key in [
            "counts",
            "result",
            "results",
            "data",
            "c",
            "register",
            "measurements",
        ] {
            if let Some(map) = obj.get(key).and_then(as_counts_object) {
                return Ok(map);
            }
        }
        // QMIO/`qat` schema: the counts sit one level deeper, grouped by the
        // classical register name (e.g. `{"results": {"c": {"00": 1000}}}`).
        // Merge every register — a single register is the common case.
        for key in ["results", "result", "data"] {
            if let Some(map) = obj.get(key).and_then(as_register_counts) {
                return Ok(map);
            }
        }
        // Fall back to the first nested counts-like object.
        for inner in obj.values() {
            if let Some(map) = as_counts_object(inner) {
                return Ok(map);
            }
        }
    }
    Err(QmioError::Schema(
        "could not locate a bitstring->count object in the QPU JSON reply".to_string(),
    ))
}

/// Interpret a JSON value as a `{bitstring: count}` map, or `None` if it is not
/// a non-empty object whose values are all non-negative integers.
fn as_counts_object(value: &serde_json::Value) -> Option<HashMap<String, u64>> {
    let obj = value.as_object()?;
    if obj.is_empty() {
        return None;
    }
    let mut counts = HashMap::with_capacity(obj.len());
    for (key, val) in obj {
        let count = val.as_u64()?;
        counts.insert(normalize_bitstring(key), count);
    }
    Some(counts)
}

/// Interpret a JSON value as a map of `register -> {bitstring: count}`, merging
/// every register into a single counts map.
///
/// This is the QMIO/`qat` reply shape, where the measured shots are grouped
/// under the classical register name (e.g. `{"c": {"00": 1000}}`). A single
/// register is the common case and is returned unchanged; multiple registers are
/// summed per bitstring. Returns `None` unless *every* entry is itself a counts
/// object, so it never matches metadata such as `execution_metrics`.
fn as_register_counts(value: &serde_json::Value) -> Option<HashMap<String, u64>> {
    let obj = value.as_object()?;
    if obj.is_empty() {
        return None;
    }
    let mut merged: HashMap<String, u64> = HashMap::new();
    for register in obj.values() {
        let counts = as_counts_object(register)?;
        for (bitstring, count) in counts {
            *merged.entry(bitstring).or_insert(0) += count;
        }
    }
    Some(merged)
}

/// Normalise a returned bitstring key to Polypus' convention.
///
/// Polypus uses Qiskit little-endian with the most-significant classical bit on
/// the left (see `infrastructure::native`). The QMIO bit order is **unverified**
/// (point 2); for now we only strip whitespace. If the live QPU returns the
/// opposite order, reverse the string here — this is the single hook.
fn normalize_bitstring(key: &str) -> String {
    key.split_whitespace().collect()
}

/// Map a Tket optimisation level to the `CompilerConfig` `$value` enum integer.
fn tket_opt_value(optimization: u8) -> Result<i64, QmioError> {
    match optimization {
        0 => Ok(0),  // TketOptimizations::Empty — no Tket compilation/routing
        1 => Ok(1),  // DefaultMappingPass only
        2 => Ok(18), // DefaultMappingPass + circuit simplifications
        3 => Ok(30), // full optimisation including SWAP routing
        other => Err(QmioError::Config(format!(
            "unsupported optimization level {other}; expected 0, 1, 2 or 3"
        ))),
    }
}

/// Map a results-format name to its `(InlineResultsProcessing, ResultsFormatting)`
/// `$value` pair. Only `binary_count` is wired end-to-end; the rest of the table
/// is reproduced for forward-compatibility (the `config` stays extensible).
fn results_format_values(res_format: &str) -> Result<(i64, i64), QmioError> {
    match res_format {
        "binary_count" => Ok((1, 3)),
        "raw" => Ok((1, 2)),
        "binary" => Ok((2, 2)),
        "squash_binary_result_arrays" => Ok((2, 6)),
        other => Err(QmioError::Config(format!(
            "unsupported results format '{other}'; only 'binary_count' is implemented"
        ))),
    }
}

/// Build the `config` JSON object, reproducing `qmio`'s `_config_build`.
///
/// The structure (key order aside, which the server's `json.loads` ignores)
/// matches `qat.purr.compiler.config.CompilerConfig` serialised via the
/// `$type`/`$data`/`$value` tagging scheme.
pub fn build_config_json(
    shots: u32,
    repetition_period: Option<f64>,
    res_format: &str,
    optimization: u8,
) -> Result<serde_json::Value, QmioError> {
    let (format_value, transforms_value) = results_format_values(res_format)?;
    let optimization_value = tket_opt_value(optimization)?;
    let repetition = match repetition_period {
        Some(value) => json!(value),
        None => serde_json::Value::Null,
    };

    Ok(json!({
        "$type": "<class 'qat.purr.compiler.config.CompilerConfig'>",
        "$data": {
            "repeats": shots,
            "repetition_period": repetition,
            "results_format": {
                "$type": "<class 'qat.purr.compiler.config.QuantumResultsFormat'>",
                "$data": {
                    "format": {
                        "$type": "<enum 'qat.purr.compiler.config.InlineResultsProcessing'>",
                        "$value": format_value
                    },
                    "transforms": {
                        "$type": "<enum 'qat.purr.compiler.config.ResultsFormatting'>",
                        "$value": transforms_value
                    }
                }
            },
            "metrics": {
                "$type": "<enum 'qat.purr.compiler.config.MetricsType'>",
                "$value": 6
            },
            "active_calibrations": [],
            "optimizations": {
                "$type": "<enum 'qat.purr.compiler.config.TketOptimizations'>",
                "$value": optimization_value
            }
        }
    }))
}

/// Read a `u64` from an environment variable, falling back to `default` when the
/// variable is unset or unparsable.
fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bell() -> ConcreteCircuit {
        ParameterizedCircuit::new(2)
            .h(0)
            .cx(0, 1)
            .measure_all()
            .assign_parameters(&[])
            .unwrap()
    }

    /// A `BackendBuildContext` with the given qmio options (endpoint left empty so
    /// `QmioBackend::new` uses its default; it connects lazily, so no live server is
    /// needed to *build* the backend).
    fn qmio_ctx(options: &[(&str, &str)]) -> BackendBuildContext {
        BackendBuildContext {
            id: "qmio-factory-test".to_string(),
            shots: 1024,
            n_qpus: 1,
            seed: None,
            opt_level: crate::OptLevel::default(),
            options: options
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    /// The registry factory reads the qmio options and validates them: the three
    /// program formats build, an unknown format and malformed numerics error, and a
    /// present optimization / repetition_period is consulted (a bad value is
    /// rejected, proving it is not ignored).
    #[test]
    fn qmio_factory_reads_and_validates_options() {
        // The three valid program formats build (no server needed to construct).
        for pf in ["openqasm", "qir_text", "qir_bitcode"] {
            assert!(
                qmio_factory(&qmio_ctx(&[("program_format", pf)])).is_ok(),
                "program_format {pf:?} should build"
            );
        }
        // Absent program_format defaults to openqasm and builds.
        assert!(qmio_factory(&qmio_ctx(&[])).is_ok());

        // Unknown program_format → Conversion error.
        match qmio_factory(&qmio_ctx(&[("program_format", "bogus")])) {
            Err(BackendError::Conversion(m)) => assert!(m.contains("program_format")),
            Err(other) => panic!("expected Conversion, got {other:?}"),
            Ok(_) => panic!("an unknown program_format must be rejected"),
        }
        // Malformed optimization → error (proves it is read).
        match qmio_factory(&qmio_ctx(&[("optimization", "high")])) {
            Err(BackendError::Conversion(m)) => assert!(m.contains("optimization")),
            Err(other) => panic!("expected Conversion, got {other:?}"),
            Ok(_) => panic!("a malformed optimization must be rejected, not defaulted"),
        }
        // Malformed repetition_period → error (proves it is read).
        match qmio_factory(&qmio_ctx(&[("repetition_period", "fast")])) {
            Err(BackendError::Conversion(m)) => assert!(m.contains("repetition_period")),
            Err(other) => panic!("expected Conversion, got {other:?}"),
            Ok(_) => panic!("a malformed repetition_period must be rejected, not defaulted"),
        }
        // Valid optimization + repetition_period build.
        assert!(qmio_factory(&qmio_ctx(&[
            ("optimization", "3"),
            ("repetition_period", "0.001"),
        ]))
        .is_ok());
    }

    fn backend(format: QmioProgramFormat) -> QmioBackend {
        QmioBackend::new(
            "tcp://10.255.3.70:5556".to_string(),
            format,
            0,
            None,
            "binary_count".to_string(),
        )
        .unwrap()
    }

    #[test]
    fn config_json_matches_binary_count_schema() {
        let config = build_config_json(1024, None, "binary_count", 0).unwrap();
        let expected = json!({
            "$type": "<class 'qat.purr.compiler.config.CompilerConfig'>",
            "$data": {
                "repeats": 1024,
                "repetition_period": serde_json::Value::Null,
                "results_format": {
                    "$type": "<class 'qat.purr.compiler.config.QuantumResultsFormat'>",
                    "$data": {
                        "format": {
                            "$type": "<enum 'qat.purr.compiler.config.InlineResultsProcessing'>",
                            "$value": 1
                        },
                        "transforms": {
                            "$type": "<enum 'qat.purr.compiler.config.ResultsFormatting'>",
                            "$value": 3
                        }
                    }
                },
                "metrics": {
                    "$type": "<enum 'qat.purr.compiler.config.MetricsType'>",
                    "$value": 6
                },
                "active_calibrations": [],
                "optimizations": {
                    "$type": "<enum 'qat.purr.compiler.config.TketOptimizations'>",
                    "$value": 0
                }
            }
        });
        assert_eq!(config, expected);
    }

    #[test]
    fn config_json_optimization_levels_map_to_tket_values() {
        for (level, expected) in [(0u8, 0i64), (1, 1), (2, 18), (3, 30)] {
            let config = build_config_json(100, None, "binary_count", level).unwrap();
            let value = &config["$data"]["optimizations"]["$value"];
            assert_eq!(value.as_i64(), Some(expected), "opt level {level}");
        }
        assert!(build_config_json(100, None, "binary_count", 4).is_err());
    }

    #[test]
    fn config_json_repetition_period_is_serialised() {
        let config = build_config_json(10, Some(0.25), "binary_count", 0).unwrap();
        assert_eq!(config["$data"]["repetition_period"].as_f64(), Some(0.25));
    }

    #[test]
    fn pickle_request_text_roundtrips_to_str_tuple() {
        let payload =
            pickle_request(&ProgramPayload::Text("OPENQASM 3.0;".into()), "{\"a\":1}").unwrap();
        let value = serde_pickle::value_from_slice(&payload, DeOptions::new()).unwrap();
        match value {
            PickleValue::Tuple(items) => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], PickleValue::String("OPENQASM 3.0;".into()));
                assert_eq!(items[1], PickleValue::String("{\"a\":1}".into()));
            }
            other => panic!("expected a 2-tuple, got {other:?}"),
        }
    }

    #[test]
    fn pickle_request_bytes_roundtrips_to_bytes_tuple() {
        let bitcode = vec![0x42u8, 0x43, 0xC0, 0xDE, 0x01];
        let payload = pickle_request(&ProgramPayload::Bytes(bitcode.clone()), "{}").unwrap();
        let value = serde_pickle::value_from_slice(&payload, DeOptions::new()).unwrap();
        match value {
            PickleValue::Tuple(items) => {
                assert_eq!(items[0], PickleValue::Bytes(bitcode));
                assert_eq!(items[1], PickleValue::String("{}".into()));
            }
            other => panic!("expected a 2-tuple, got {other:?}"),
        }
    }

    #[test]
    fn program_format_openqasm_produces_text_with_20_header() {
        let circuit = bell();
        let exported = circuit.to_qasm2();
        assert!(
            exported.starts_with("OPENQASM 2.0"),
            "exporter changed: {exported}"
        );

        let payload = backend(QmioProgramFormat::OpenQasm)
            .serialize_program(&BoundCircuit::Native(circuit))
            .unwrap();
        match payload {
            ProgramPayload::Text(qasm) => {
                assert!(qasm.starts_with("OPENQASM 2.0"), "header changed: {qasm}");
                // The program is submitted exactly as exported (header and body).
                assert_eq!(qasm, exported);
            }
            ProgramPayload::Bytes(_) => panic!("OpenQASM must be text"),
        }
    }

    #[test]
    fn program_format_qir_text_produces_llvm_module() {
        let payload = backend(QmioProgramFormat::QirText)
            .serialize_program(&BoundCircuit::Native(bell()))
            .unwrap();
        match payload {
            ProgramPayload::Text(qir) => assert!(qir.contains("define"), "not LLVM IR: {qir}"),
            ProgramPayload::Bytes(_) => panic!("QIR text must be text"),
        }
    }

    #[test]
    fn program_format_qir_bitcode_produces_bc_magic() {
        // `llvm-as` is required for this path; skip gracefully if unavailable.
        match backend(QmioProgramFormat::QirBitcode)
            .serialize_program(&BoundCircuit::Native(bell()))
        {
            Ok(ProgramPayload::Bytes(bc)) => {
                assert_eq!(&bc[0..2], b"BC", "missing LLVM bitcode magic");
            }
            Ok(ProgramPayload::Text(_)) => panic!("bitcode must be bytes"),
            Err(QmioError::QirBitcode(_)) => { /* llvm-as not installed: acceptable */ }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn rejects_qiskit_circuits() {
        // Constructing a Qiskit-variant bound circuit needs a Python object, so
        // initialise the interpreter for this test only (the GIL-free paths in
        // native.rs and native_circuit_path.rs are unaffected).
        pyo3::prepare_freethreaded_python();
        let circuit = pyo3::Python::with_gil(|py| crate::QiskitCircuit::into_bound(py.None()));
        let err = backend(QmioProgramFormat::OpenQasm)
            .serialize_program(&circuit)
            .unwrap_err();
        assert!(matches!(err, QmioError::UnsupportedCircuit));
    }

    #[test]
    fn parse_counts_accepts_flat_object() {
        let json = "{\"00\": 10, \"11\": 6}";
        let pickled =
            serde_pickle::value_to_vec(&PickleValue::String(json.into()), SerOptions::new())
                .unwrap();
        let counts = parse_counts(&pickled).unwrap();
        assert_eq!(counts.get("00"), Some(&10));
        assert_eq!(counts.get("11"), Some(&6));
        assert_eq!(counts.values().sum::<u64>(), 16);
    }

    #[test]
    fn parse_counts_accepts_nested_container() {
        let json = "{\"results\": {\"01\": 3, \"10\": 5}, \"meta\": \"x\"}";
        let pickled =
            serde_pickle::value_to_vec(&PickleValue::String(json.into()), SerOptions::new())
                .unwrap();
        let counts = parse_counts(&pickled).unwrap();
        assert_eq!(counts.get("01"), Some(&3));
        assert_eq!(counts.get("10"), Some(&5));
    }

    #[test]
    fn counts_from_json_reads_qmio_register_schema() {
        // The exact reply shape returned by the live QMIO QPU: counts grouped
        // under the classical register name inside `results`, alongside an
        // `execution_metrics` sibling that must be ignored.
        let reply = json!({
            "results": {"c": {"00": 600, "11": 400}},
            "execution_metrics": {
                "optimized_circuit": "OPENQASM 3.0;",
                "optimized_instruction_count": 98,
            },
        });
        let counts = counts_from_json(&reply).unwrap();
        assert_eq!(counts.get("00"), Some(&600));
        assert_eq!(counts.get("11"), Some(&400));
        assert_eq!(counts.values().sum::<u64>(), 1000);
    }

    #[test]
    fn counts_from_json_merges_multiple_registers() {
        let reply = json!({"results": {"c0": {"0": 3}, "c1": {"1": 7}}});
        let counts = counts_from_json(&reply).unwrap();
        assert_eq!(counts.get("0"), Some(&3));
        assert_eq!(counts.get("1"), Some(&7));
    }

    #[test]
    fn parse_counts_accepts_pickled_register_dict() {
        // End-to-end of the decode path for a pickled `dict` (not a JSON
        // string), mirroring what the QPU actually sends.
        let reply_value = json!({
            "results": {"c": {"00": 512, "11": 488}},
            "execution_metrics": {"optimized_instruction_count": 98},
        });
        let pickled = serde_pickle::to_vec(&reply_value, SerOptions::new()).unwrap();
        let counts = parse_counts(&pickled).unwrap();
        assert_eq!(counts.get("00"), Some(&512));
        assert_eq!(counts.get("11"), Some(&488));
        assert_eq!(counts.values().sum::<u64>(), 1000);
    }

    #[test]
    fn normalize_bitstring_strips_whitespace() {
        assert_eq!(normalize_bitstring("01 10"), "0110");
    }

    /// End-to-end test against a simulated QMIO server: a real ZMQ REP socket on
    /// a background thread deserialises the pickled `(circuit, config)` tuple,
    /// validates it, and replies with pickled JSON counts — exercising the whole
    /// [`QmioBackend`] path without the actual QPU.
    #[test]
    fn simulated_rep_server_end_to_end() {
        use crate::{BackendConfig, ExecutionConfig};
        use std::sync::mpsc;
        use zeromq::RepSocket;

        // Reserve a free port, then bind the simulated server to it.
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let endpoint = format!("tcp://127.0.0.1:{port}");

        let (ready_tx, ready_rx) = mpsc::channel::<()>();
        let server_endpoint = endpoint.clone();
        let server = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let mut rep = RepSocket::new();
                rep.bind(&server_endpoint).await.unwrap();
                ready_tx.send(()).unwrap();

                // Handle exactly one request/response exchange.
                let msg = rep.recv().await.unwrap();
                let bytes = msg.into_vec().into_iter().next().unwrap().to_vec();
                let request = serde_pickle::value_from_slice(&bytes, DeOptions::new()).unwrap();
                let (program, config) = match request {
                    PickleValue::Tuple(items) if items.len() == 2 => {
                        (items[0].clone(), items[1].clone())
                    }
                    other => panic!("expected a 2-tuple request, got {other:?}"),
                };
                match program {
                    PickleValue::String(qasm) => {
                        assert!(qasm.starts_with("OPENQASM 2.0"), "header: {qasm}")
                    }
                    other => panic!("expected QASM text, got {other:?}"),
                }
                let config_json = match config {
                    PickleValue::String(s) => s,
                    other => panic!("expected a config JSON string, got {other:?}"),
                };
                let parsed: serde_json::Value = serde_json::from_str(&config_json).unwrap();
                assert_eq!(parsed["$data"]["repeats"].as_u64(), Some(1024));

                // Reply with the verified live-QPU shape: a pickled `dict` whose
                // counts are grouped under the classical register name.
                let reply_value = json!({
                    "results": {"c": {"00": 500, "11": 524}},
                    "execution_metrics": {"optimized_instruction_count": 98},
                });
                let reply = serde_pickle::to_vec(&reply_value, SerOptions::new()).unwrap();
                rep.send(ZmqMessage::from(reply)).await.unwrap();
            });
        });

        // Wait until the server is bound before connecting.
        ready_rx.recv().unwrap();

        let backend = QmioBackend::new(
            endpoint.clone(),
            QmioProgramFormat::OpenQasm,
            0,
            None,
            "binary_count".to_string(),
        )
        .unwrap();
        let config = ExecutionConfig {
            id: "qmio-sim".to_string(),
            shots: 1024,
            n_qpus: 1,
            infrastructure: "qmio".to_string(),
            // QMIO is now registry-dispatched; this test builds the backend directly
            // above and only needs a config to derive `run_params()` (which ignores
            // `backend_config`), so the registry variant with empty options suffices.
            backend_config: BackendConfig::Registered {
                name: "qmio".to_string(),
                options: std::collections::HashMap::new(),
            },
            opt_level: crate::OptLevel::default(),
            // QMIO does not consume the sampling seed (real QPU / server-side).
            seed: None,
        };

        let counts = backend
            .run_circuits(&[BoundCircuit::Native(bell())], &config.run_params())
            .unwrap();
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[0].get("00"), Some(&500));
        assert_eq!(counts[0].get("11"), Some(&524));
        assert_eq!(counts[0].values().sum::<u64>(), 1024);

        server.join().unwrap();
    }

    /// A receive failure *after the request was delivered* must surface as
    /// [`QmioError::ResultUnknown`] — never a blind resend/retry, which would risk
    /// a double execution on the QPU. The simulated server receives the request
    /// but never replies, so the client's receive times out. `ResultUnknown` is
    /// only reachable on the no-resend path (a retry would instead exhaust
    /// `max_retries` and return `Timeout`), so the error type alone proves the
    /// request was not resent.
    #[test]
    fn recv_timeout_after_delivery_is_result_unknown_not_resent() {
        use crate::{BackendConfig, BackendError, ExecutionConfig};
        use std::sync::mpsc;
        use zeromq::RepSocket;

        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let endpoint = format!("tcp://127.0.0.1:{port}");

        let (ready_tx, ready_rx) = mpsc::channel::<()>();
        let (delivered_tx, delivered_rx) = mpsc::channel::<()>();
        let server_endpoint = endpoint.clone();
        let server = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let mut rep = RepSocket::new();
                rep.bind(&server_endpoint).await.unwrap();
                ready_tx.send(()).unwrap();
                // Receive the request but deliberately never reply, forcing the
                // client's receive to time out. Signal receipt so the test knows
                // the request reached the post-send (delivered) path.
                let _ = rep.recv().await.unwrap();
                delivered_tx.send(()).unwrap();
                // Hold the socket open long enough that a (buggy) resend would have
                // had time to arrive before teardown.
                tokio::time::sleep(Duration::from_millis(600)).await;
            });
        });
        ready_rx.recv().unwrap();

        let backend = QmioBackend::new(
            endpoint.clone(),
            QmioProgramFormat::OpenQasm,
            0,
            None,
            "binary_count".to_string(),
        )
        .unwrap()
        // Short receive timeout, retries still allowed: without the fix the client
        // would resend and eventually return Timeout instead of ResultUnknown.
        .with_test_timeouts(150, 3, 10_000);
        let config = ExecutionConfig {
            id: "qmio-unknown".to_string(),
            shots: 1024,
            n_qpus: 1,
            infrastructure: "qmio".to_string(),
            // QMIO is now registry-dispatched; this test builds the backend directly
            // above and only needs a config to derive `run_params()` (which ignores
            // `backend_config`), so the registry variant with empty options suffices.
            backend_config: BackendConfig::Registered {
                name: "qmio".to_string(),
                options: std::collections::HashMap::new(),
            },
            opt_level: crate::OptLevel::default(),
            seed: None,
        };

        let err = backend
            .run_circuits(&[BoundCircuit::Native(bell())], &config.run_params())
            .unwrap_err();
        // The QMIO error crosses the pyo3-free contract type-erased in
        // `External`; recover the concrete `QmioError` to assert the variant.
        let qmio_err = match &err {
            BackendError::External(boxed) => boxed
                .downcast_ref::<QmioError>()
                .expect("the boxed error must be the original QmioError"),
            other => panic!("expected BackendError::External(QmioError), got: {other:?}"),
        };
        assert!(
            matches!(qmio_err, QmioError::ResultUnknown { .. }),
            "a delivered-but-unreplied request must be ResultUnknown (not resent); got: {qmio_err:?}"
        );
        // The request was delivered to the server exactly once.
        delivered_rx.recv().unwrap();
        server.join().unwrap();
    }

    /// Two callers hitting the **same** [`QmioBackend`] concurrently must be
    /// serialised over the single REQ socket and both complete correctly — the
    /// property the async-mutex hardening protects. It exercises the guard being
    /// held across the request's `.await` points from two OS threads at once: the
    /// second caller waits (cooperatively) for the first to release rather than
    /// racing it, so exactly one REQ socket ever talks to the endpoint.
    ///
    /// Determinism without sleeps: the simulated REP socket is itself lock-step
    /// (one recv, one reply, repeat), so it hands out its two *distinct* replies
    /// strictly in the order the requests arrive. Whichever thread wins the mutex
    /// first gets the first reply; the other gets the second. The test therefore
    /// asserts on the order-independent *set* of the two received counts — never on
    /// which thread got which — so it is robust to the (legitimately arbitrary)
    /// thread scheduling. Server readiness is signalled over an `mpsc` channel, as
    /// in the other simulated-server tests.
    #[test]
    fn concurrent_callers_are_serialized_over_a_single_socket() {
        use crate::{BackendConfig, ExecutionConfig};
        use std::sync::mpsc;
        use std::sync::Arc;
        use zeromq::RepSocket;

        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let endpoint = format!("tcp://127.0.0.1:{port}");

        let (ready_tx, ready_rx) = mpsc::channel::<()>();
        let server_endpoint = endpoint.clone();
        // The server replies to request #k with a `"00"` count of `1000 + k`, so
        // the two replies are distinguishable. Because it is lock-step, the first
        // request to arrive gets 1000 and the second gets 1001 — regardless of
        // which client thread that is.
        let server = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let mut rep = RepSocket::new();
                rep.bind(&server_endpoint).await.unwrap();
                ready_tx.send(()).unwrap();

                for k in 0..2u64 {
                    let msg = rep.recv().await.unwrap();
                    let bytes = msg.into_vec().into_iter().next().unwrap().to_vec();
                    let request = serde_pickle::value_from_slice(&bytes, DeOptions::new()).unwrap();
                    match request {
                        PickleValue::Tuple(items) if items.len() == 2 => {}
                        other => panic!("expected a 2-tuple request, got {other:?}"),
                    }
                    let reply_value = json!({"results": {"c": {"00": 1000 + k}}});
                    let reply = serde_pickle::to_vec(&reply_value, SerOptions::new()).unwrap();
                    rep.send(ZmqMessage::from(reply)).await.unwrap();
                }
            });
        });
        ready_rx.recv().unwrap();

        let backend = Arc::new(
            QmioBackend::new(
                endpoint.clone(),
                QmioProgramFormat::OpenQasm,
                0,
                None,
                "binary_count".to_string(),
            )
            .unwrap(),
        );

        // `endpoint` is no longer part of the config (QMIO reads it from the
        // registry options now); the closure keeps the arg only so its two call
        // sites stay unchanged.
        let make_config = |_endpoint: String| ExecutionConfig {
            id: "qmio-concurrent".to_string(),
            shots: 1024,
            n_qpus: 1,
            infrastructure: "qmio".to_string(),
            // QMIO is now registry-dispatched; this test builds the backend directly
            // above and only needs a config to derive `run_params()` (which ignores
            // `backend_config`), so the registry variant with empty options suffices.
            backend_config: BackendConfig::Registered {
                name: "qmio".to_string(),
                options: std::collections::HashMap::new(),
            },
            opt_level: crate::OptLevel::default(),
            seed: None,
        };

        let clients: Vec<_> = (0..2)
            .map(|_| {
                let backend = Arc::clone(&backend);
                let config = make_config(endpoint.clone());
                std::thread::spawn(move || {
                    let counts = backend
                        .run_circuits(&[BoundCircuit::Native(bell())], &config.run_params())
                        .unwrap();
                    assert_eq!(counts.len(), 1);
                    *counts[0]
                        .get("00")
                        .expect("reply must carry a \"00\" count")
                })
            })
            .collect();

        let mut received: Vec<u64> = clients.into_iter().map(|h| h.join().unwrap()).collect();
        received.sort_unstable();
        // Both distinct replies were delivered, one to each caller: the two
        // concurrent requests were serialised over the single socket without one
        // being dropped, duplicated, or crossed with the other.
        assert_eq!(received, vec![1000, 1001]);

        server.join().unwrap();
    }

    /// Smoke test against the real QMIO QPU. Ignored by default: it needs the
    /// `ZMQ_SERVER` environment variable set and live access to the CESGA
    /// network. Run with `cargo test -p polypus --features qmio -- --ignored`.
    #[test]
    #[ignore = "requires ZMQ_SERVER and live access to the CESGA QMIO QPU"]
    fn real_qpu_smoke() {
        use crate::{BackendConfig, ExecutionConfig};

        let endpoint = std::env::var("ZMQ_SERVER")
            .expect("set ZMQ_SERVER to the QMIO endpoint, e.g. tcp://10.255.3.70:5556");
        let backend = QmioBackend::new(
            endpoint.clone(),
            QmioProgramFormat::OpenQasm,
            1,
            None,
            "binary_count".to_string(),
        )
        .unwrap();
        let config = ExecutionConfig {
            id: "qmio-real".to_string(),
            shots: 1000,
            n_qpus: 1,
            infrastructure: "qmio".to_string(),
            // QMIO is now registry-dispatched; only `run_params()` is read here.
            backend_config: BackendConfig::Registered {
                name: "qmio".to_string(),
                options: std::collections::HashMap::new(),
            },
            opt_level: crate::OptLevel::default(),
            // QMIO does not consume the sampling seed (real QPU / server-side).
            seed: None,
        };
        let counts = backend
            .run_circuits(&[BoundCircuit::Native(bell())], &config.run_params())
            .unwrap();
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[0].values().sum::<u64>(), 1000);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // `counts_from_json` — the rejection half.
    //
    // Every case above is a *successful* extraction. These pin the final
    // `QmioError::Schema` fallback, so a reply the extractor cannot understand
    // is a typed error (mapped to `polypus.QmioError` at the FFI) rather than
    // silently-empty counts. Pure `serde_json`: no Python, no network.
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn counts_from_json_rejects_an_empty_object() {
        // `as_counts_object` rejects an empty object, no container key is
        // present, and there is nothing nested to fall back to.
        assert!(matches!(
            counts_from_json(&json!({})),
            Err(QmioError::Schema(_))
        ));
    }

    #[test]
    fn counts_from_json_rejects_metadata_only_replies() {
        // Nothing here is `{bitstring: count}`-shaped at any depth: the leaves
        // are strings, floats and booleans rather than non-negative integers.
        //
        // Note the deliberate absence of an integer leaf. A metadata object such
        // as `{"optimized_instruction_count": 98}` *is* accepted by the naive
        // `{key: u64}` test and would be picked up by the last-resort nested
        // scan — that permissiveness is the documented "fall back to the first
        // nested counts-like object" behaviour, not a case this test asserts on.
        let reply = json!({
            "execution_metrics": {
                "optimized_circuit": "OPENQASM 3.0;",
                "duration_seconds": 1.5,
                "cached": false,
            },
        });
        assert!(matches!(
            counts_from_json(&reply),
            Err(QmioError::Schema(_))
        ));
    }

    #[test]
    fn counts_from_json_rejects_a_register_container_with_a_bad_register() {
        // `as_register_counts` requires *every* entry to be a counts object, so
        // one malformed register invalidates the whole container — and with
        // nothing else counts-shaped in the reply, this must reach `Schema`
        // rather than silently returning the registers it could read.
        let reply = json!({
            "results": {
                "c0": {"00": 10, "11": 6},
                "c1": "unexpected",
            },
        });
        assert!(matches!(
            counts_from_json(&reply),
            Err(QmioError::Schema(_))
        ));
    }

    #[test]
    fn counts_from_json_rejects_negative_and_non_integer_counts() {
        // `as_counts_object` requires `u64` values, so a negative or fractional
        // count is not a counts object.
        assert!(matches!(
            counts_from_json(&json!({"counts": {"00": -1, "11": 6}})),
            Err(QmioError::Schema(_))
        ));
        assert!(matches!(
            counts_from_json(&json!({"counts": {"00": 1.5}})),
            Err(QmioError::Schema(_))
        ));
    }

    #[test]
    fn counts_from_json_rejects_a_non_object_reply() {
        // `value.as_object()` is `None`, so the whole search is skipped.
        for reply in [json!([1, 2, 3]), json!(42), json!("counts"), json!(null)] {
            assert!(
                matches!(counts_from_json(&reply), Err(QmioError::Schema(_))),
                "a non-object reply must be a schema error: {reply}"
            );
        }
    }

    // The mapping of a QMIO failure to the typed `polypus.QmioError` Python class
    // is tested at the `polypus` FFI edge (`exceptions::backend_error_to_pyerr`),
    // which owns that `#[pyclass]` and downcasts the `QmioError` back out of the
    // pyo3-free `BackendError::External` box; this crate tests only that a bad reply
    // becomes a `QmioError::Schema` (above).
}
