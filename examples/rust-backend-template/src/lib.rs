//! # A native (option-D) Polypus backend template
//!
//! Copy this crate and make it talk to your device. It is the **native / wire-protocol
//! path** — the one QMIO uses: a pure-Rust backend that serialises each circuit,
//! sends it to a device over your own protocol, and turns the reply into counts. It
//! depends on `polypus-backend` **alone** — no PyO3, no Qiskit, no Polypus runtime —
//! and passes the conformance battery (see `tests/conformance.rs`).
//!
//! If your provider only ships a **Python SDK** with no documented wire protocol, use
//! the *subprocess bridge* template instead (`examples/python-backend-template`),
//! which runs that SDK in its own process; this native path is preferable whenever
//! the provider exposes a protocol you can speak from Rust directly.
//!
//! ## What you replace
//!
//! Exactly one thing: [`TemplateBackend::transport`] — the function that carries a
//! serialised program to your device and returns its counts. Everything else (the
//! trait wiring, the error classification, the result self-check, the registration)
//! is the reusable shape.
//!
//! ## Error classification (the part worth getting right)
//!
//! `run_circuits` must never panic; it returns a [`BackendError`]. The three
//! distinctions the battery checks:
//! - the device **stopped responding** (crashed, dropped the connection, or blew a
//!   liveness deadline) → [`BackendError::Unresponsive`] — the *retryable* variant;
//! - your call was **aborted by a signal** (a terminal Ctrl+C, or the planner's
//!   out-of-band [`cancel`](QuantumBackend::cancel)) → [`BackendError::Aborted`],
//!   which the FFI edge raises as `KeyboardInterrupt`;
//! - the device reported a **clean, definitive failure** (bad program, rejected job)
//!   → [`BackendError::External`], carrying your own error type-erased.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use polypus_backend::{
    register_backend, validate_run_results, BackendBuildContext, BackendError, BoundCircuit,
    Counts, QuantumBackend, RunParams,
};

/// The name this backend registers itself under. A Python user selects it with
/// `infrastructure="example-rust-qpu"` after your crate has called [`register`].
pub const BACKEND_NAME: &str = "example-rust-qpu";

/// A native backend that speaks a wire protocol to a device.
pub struct TemplateBackend {
    /// Where the device lives. A real backend opens/holds its connection here.
    endpoint: String,
    /// Set by [`cancel`](QuantumBackend::cancel) so a blocked [`transport`](Self::transport)
    /// call can notice and abort — the out-of-band cancellation hook.
    cancelled: AtomicBool,
}

impl TemplateBackend {
    /// Connect to the device at `endpoint`. A real backend would open its socket /
    /// session here and return an error if that fails.
    pub fn connect(endpoint: &str) -> Result<Self, BackendError> {
        Ok(TemplateBackend {
            endpoint: endpoint.to_string(),
            cancelled: AtomicBool::new(false),
        })
    }

    /// Build from the registry's [`BackendBuildContext`]: read the `endpoint` option
    /// (defaulting to a loopback), plus whatever else your device needs.
    pub fn from_context(
        ctx: &BackendBuildContext,
    ) -> Result<Arc<dyn QuantumBackend>, BackendError> {
        let endpoint = ctx.option("endpoint").unwrap_or("sim://loopback");
        Ok(Arc::new(TemplateBackend::connect(endpoint)?))
    }

    /// Run one circuit on the device and return its counts.
    fn sample_one(
        &self,
        qc: &BoundCircuit,
        shots: u32,
        seed: Option<u64>,
    ) -> Result<Counts, BackendError> {
        // Serialise to your wire format. A provider-native `Foreign` object cannot
        // cross a wire, so reject it up front — do NOT panic.
        let program = match qc {
            BoundCircuit::Native(cc) => cc.to_qasm2(),
            BoundCircuit::Qasm2(qasm) => qasm.clone(),
            BoundCircuit::Foreign(_) => {
                return Err(BackendError::UnsupportedCircuit(
                    "example-rust-qpu speaks OpenQASM 2.0 over the wire and cannot send a \
                     provider-native (Foreign) circuit"
                        .to_string(),
                ))
            }
        };
        let width = qc.native_qubit_width().unwrap_or(1).max(1);
        self.transport(&program, width, shots, seed)
    }

    /// **Replace this with your device's wire protocol.** Send `program` (+ `shots`,
    /// `seed`) to the device and return one bitstring→count map.
    ///
    /// The `sim://…` endpoints below are *only* scaffolding so the conformance battery
    /// can drive this template into each failure mode — delete them in your real
    /// backend and put your `send`/`recv` here instead. They double as the worked
    /// examples of the three error classifications.
    fn transport(
        &self,
        program: &str,
        width: usize,
        shots: u32,
        seed: Option<u64>,
    ) -> Result<Counts, BackendError> {
        let _ = (program, seed); // a real device would use these
        match self.endpoint.as_str() {
            // The device never answered / dropped the link → retryable "stopped responding".
            "sim://unreachable" => Err(BackendError::Unresponsive(
                "no reply from device at sim://unreachable".to_string(),
            )),
            // The device answered with a clean, definitive failure → provider error.
            "sim://reject" => Err(BackendError::External(
                "device rejected the program: unsupported gate".into(),
            )),
            // A long call the out-of-band `cancel()` can abort → Aborted.
            "sim://slow" => {
                let deadline = Instant::now() + Duration::from_secs(5);
                while Instant::now() < deadline {
                    if self.cancelled.load(Ordering::SeqCst) {
                        return Err(BackendError::Aborted(
                            "device call cancelled by signal".to_string(),
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Ok(loopback_counts(width, shots))
            }
            // The healthy path: a real device returns real counts here.
            _ => Ok(loopback_counts(width, shots)),
        }
    }
}

/// A stand-in "device" result: all shots on the all-zeros bitstring of the right
/// width. Deterministic, valid, and hardware-free — your real device returns its own
/// measured counts.
fn loopback_counts(width: usize, shots: u32) -> Counts {
    HashMap::from([("0".repeat(width), u64::from(shots))])
}

impl QuantumBackend for TemplateBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        params: &RunParams,
    ) -> Result<Vec<Counts>, BackendError> {
        // A fresh call clears any prior cancellation latch.
        self.cancelled.store(false, Ordering::SeqCst);
        let counts: Vec<Counts> = qcs
            .iter()
            .map(|qc| self.sample_one(qc, params.shots, params.seed))
            .collect::<Result<_, _>>()?;
        // Recommended: self-check your output against the contract before returning
        // it, so a device/serialisation bug surfaces here as InvalidResults rather
        // than as a confusing failure downstream.
        validate_run_results(&counts, qcs.len(), params.shots)?;
        Ok(counts)
    }

    /// Out-of-band abort: flip the flag a blocked [`transport`](Self::transport) call
    /// watches. Called from another thread by the planner's watcher (enabled via
    /// [`wants_cancel_watcher`](Self::wants_cancel_watcher)) when the run is cancelled.
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    fn wants_cancel_watcher(&self) -> bool {
        // This backend can block on the device, so it opts into the watcher thread
        // that calls `cancel()` mid-run.
        true
    }
}

/// Register this backend under [`BACKEND_NAME`]. A real crate calls this from its
/// startup (or exposes it for the embedder to call).
pub fn register() {
    register_backend(BACKEND_NAME, TemplateBackend::from_context);
}
