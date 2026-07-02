//! CESGA **QMIO** real-QPU backend.
//!
//! [`QmioBackend`] talks to the QMIO quantum computer directly over its ZeroMQ
//! endpoint, **without going through the Python interpreter**. For the
//! [`BoundCircuit::Native`] and [`BoundCircuit::Qasm2`] variants the whole
//! round-trip — circuit serialisation, request framing, response parsing — runs
//! in Rust, so it never acquires the GIL. (A [`BoundCircuit::Qiskit`] cannot be
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
//! - **Reply** = a pickle of a Python `dict` with the results (the live QPU
//!   pickles the dict directly; some builds may instead pickle a JSON string).
//!   Decode path: `recv` bytes → `serde_pickle` → `dict`/`String` →
//!   `serde_json::Value` → per-bitstring counts.
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
//!    one QPU ([`max_batch_size`](QmioBackend::max_batch_size) returns `1`).
//! 6. **OpenQASM header**: Polypus exports and submits an `OPENQASM 2.0` program
//!    (header and body), which matches the 2.0-style body the QMIO examples use.
//!    Verify acceptance against the live QPU (point 6).

use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use polypus_circuit::{CircuitError, ConcreteCircuit, ParameterizedCircuit};
use serde_json::json;
use serde_pickle::{DeOptions, SerOptions, Value as PickleValue};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tokio::runtime::Runtime;
use zeromq::{ReqSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

pub use crate::infrastructure::execution_config::QmioProgramFormat;

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
/// [`QuantumBackend::run_circuits`] cannot return a `Result` (its signature is
/// shared by every backend), so a fatal error is logged and turned into a panic
/// — which PyO3 surfaces to Python as an exception. This mirrors how
/// [`CunqaBackend`](crate::infrastructure::CunqaBackend) and
/// [`NativeStatevectorBackend`](crate::infrastructure::NativeStatevectorBackend)
/// report unrecoverable failures, while keeping the network code itself free of
/// scattered `unwrap()`s.
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
    /// The reply exceeded [`QmioBackend::max_reply_bytes`].
    ResponseTooLarge { bytes: usize, limit: usize },
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
            QmioError::ResponseTooLarge { bytes, limit } => write!(
                f,
                "QMIO reply of {bytes} bytes exceeds the {limit}-byte safety limit"
            ),
        }
    }
}

impl std::error::Error for QmioError {}

/// Backend that executes circuits on the CESGA QMIO QPU over ZeroMQ.
///
/// The REQ socket is created lazily and held behind a [`Mutex`] so the backend
/// is `Send + Sync` (required by [`QuantumBackend`]) even though a single REQ
/// socket is inherently serial — which is exactly the semantics we want: one
/// request at a time. On a network fault the socket is dropped and recreated
/// (the "Lazy Pirate" pattern: a REQ socket is unusable after a failed `recv`).
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
    /// Hard cap on an accepted reply, guarding against a hostile/buggy peer.
    max_reply_bytes: usize,
    /// Idempotency guard for [`close`](Self::close)/[`Drop`].
    closed: AtomicBool,
}

impl QmioBackend {
    /// Create a backend targeting `endpoint`.
    ///
    /// Timeouts and retry limits can be overridden via the `QMIO_RECV_TIMEOUT_MS`
    /// and `QMIO_MAX_RETRIES` environment variables (sensible defaults otherwise).
    pub fn new(
        endpoint: String,
        program_format: QmioProgramFormat,
        optimization: u8,
        repetition_period: Option<f64>,
        res_format: String,
    ) -> Self {
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
            .expect("failed to build the Tokio runtime for the QMIO backend");
        let recv_timeout = Duration::from_millis(env_u64("QMIO_RECV_TIMEOUT_MS", 300_000));
        let max_retries = env_u64("QMIO_MAX_RETRIES", 3) as usize;
        QmioBackend {
            endpoint,
            program_format,
            optimization,
            repetition_period,
            res_format,
            runtime,
            socket: Mutex::new(None),
            recv_timeout,
            retry_backoff: Duration::from_millis(env_u64("QMIO_RETRY_BACKOFF_MS", 100)),
            max_retries,
            // 64 MiB: far above any realistic counts payload, far below "OOM".
            max_reply_bytes: 64 * 1024 * 1024,
            closed: AtomicBool::new(false),
        }
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
            BoundCircuit::Qiskit(_) => Err(QmioError::UnsupportedCircuit),
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
            BoundCircuit::Qiskit(_) => Err(QmioError::UnsupportedCircuit),
        }
    }

    /// Send one pickled request and parse the reply into counts, reconnecting
    /// and retrying on transient faults (Lazy Pirate).
    fn run_one(
        &self,
        program: &ProgramPayload,
        config_json: &str,
    ) -> Result<HashMap<String, u64>, QmioError> {
        println!("QMIO request: {program:?}, config: {config_json}");
        let request = pickle_request(program, config_json)?;
        let mut guard = self.socket.lock().expect("QMIO socket mutex poisoned");

        self.runtime.block_on(async {
            let mut last_err: Option<QmioError> = None;
            for attempt in 0..=self.max_retries {
                // Linear backoff before a retry; the first attempt is immediate.
                if attempt > 0 {
                    tokio::time::sleep(self.retry_backoff * attempt as u32).await;
                }
                // (Re)connect if we have no live socket.
                if guard.is_none() {
                    match connect(&self.endpoint).await {
                        Ok(s) => *guard = Some(s),
                        Err(e) => {
                            last_err = Some(e);
                            continue;
                        }
                    }
                }
                let socket = guard.as_mut().expect("socket present after connect");

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

                // Receive, bounded by the configured timeout. On timeout or
                // error the REQ socket is stuck, so we discard and retry.
                let socket = guard.as_mut().expect("socket present after send");
                match tokio::time::timeout(self.recv_timeout, socket.recv()).await {
                    Ok(Ok(reply)) => {
                        println!("QMIO reply: {reply:?}");
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
                        last_err = Some(QmioError::Recv(e.to_string()));
                    }
                    Err(_elapsed) => {
                        drop_socket(&mut guard).await;
                        last_err = Some(QmioError::Timeout {
                            endpoint: self.endpoint.clone(),
                            attempts: attempt + 1,
                            millis: self.recv_timeout.as_millis(),
                        });
                    }
                }
            }
            Err(last_err.unwrap_or_else(|| QmioError::Timeout {
                endpoint: self.endpoint.clone(),
                attempts: self.max_retries + 1,
                millis: self.recv_timeout.as_millis(),
            }))
        })
    }
}

impl QuantumBackend for QmioBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        config: &ExecutionConfig,
    ) -> Vec<HashMap<String, u64>> {
        // The config JSON is identical for every circuit in the batch.
        let config_json = build_config_json(
            config.shots,
            self.repetition_period,
            &self.res_format,
            self.optimization,
        )
        .and_then(|v| serde_json::to_string(&v).map_err(|e| QmioError::Json(e.to_string())))
        .unwrap_or_else(|e| {
            log::error!("QMIO config build failed: {e}");
            panic!("QMIO backend error: {e}");
        });

        // One REQ/REP exchange per circuit: the endpoint is a single QPU.
        qcs.iter()
            .map(|qc| {
                let program = self.serialize_program(qc).unwrap_or_else(|e| {
                    log::error!("QMIO circuit serialisation failed: {e}");
                    panic!("QMIO backend error: {e}");
                });
                self.run_one(&program, &config_json).unwrap_or_else(|e| {
                    log::error!("QMIO execution failed: {e}");
                    panic!("QMIO backend error talking to {}: {e}", self.endpoint);
                })
            })
            .collect()
    }

    fn max_batch_size(&self, _total: usize) -> usize {
        // A single QPU behind one REQ endpoint: at most one circuit per call.
        1
    }

    fn close(&self) {
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Ok(mut guard) = self.socket.lock() {
            if let Some(socket) = guard.take() {
                // Graceful close must run inside the runtime context.
                let _ = self.runtime.block_on(socket.close());
            }
        }
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
        let _ = socket.close().await;
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

    fn backend(format: QmioProgramFormat) -> QmioBackend {
        QmioBackend::new(
            "tcp://10.255.3.70:5556".to_string(),
            format,
            0,
            None,
            "binary_count".to_string(),
        )
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
        let circuit = pyo3::Python::with_gil(|py| BoundCircuit::Qiskit(py.None()));
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
        use crate::infrastructure::BackendConfig;
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
        );
        let config = ExecutionConfig {
            id: "qmio-sim".to_string(),
            shots: 1024,
            n_qpus: 1,
            infrastructure: "qmio".to_string(),
            backend_config: BackendConfig::Qmio {
                endpoint,
                program_format: QmioProgramFormat::OpenQasm,
                optimization: 0,
                repetition_period: None,
                res_format: "binary_count".to_string(),
            },
            opt_level: crate::infrastructure::OptLevel::default(),
        };

        let counts = backend.run_circuits(&[BoundCircuit::Native(bell())], &config);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[0].get("00"), Some(&500));
        assert_eq!(counts[0].get("11"), Some(&524));
        assert_eq!(counts[0].values().sum::<u64>(), 1024);

        server.join().unwrap();
    }

    /// Smoke test against the real QMIO QPU. Ignored by default: it needs the
    /// `ZMQ_SERVER` environment variable set and live access to the CESGA
    /// network. Run with `cargo test -p polypus --features qmio -- --ignored`.
    #[test]
    #[ignore = "requires ZMQ_SERVER and live access to the CESGA QMIO QPU"]
    fn real_qpu_smoke() {
        use crate::infrastructure::BackendConfig;

        let endpoint = std::env::var("ZMQ_SERVER")
            .expect("set ZMQ_SERVER to the QMIO endpoint, e.g. tcp://10.255.3.70:5556");
        let backend = QmioBackend::new(
            endpoint.clone(),
            QmioProgramFormat::OpenQasm,
            1,
            None,
            "binary_count".to_string(),
        );
        let config = ExecutionConfig {
            id: "qmio-real".to_string(),
            shots: 1000,
            n_qpus: 1,
            infrastructure: "qmio".to_string(),
            backend_config: BackendConfig::Qmio {
                endpoint,
                program_format: QmioProgramFormat::OpenQasm,
                optimization: 1,
                repetition_period: None,
                res_format: "binary_count".to_string(),
            },
            opt_level: crate::infrastructure::OptLevel::default(),
        };
        let counts = backend.run_circuits(&[BoundCircuit::Native(bell())], &config);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[0].values().sum::<u64>(), 1000);
    }
}
