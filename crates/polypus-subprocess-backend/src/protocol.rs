//! The **subprocess bridge wire protocol, version 1** — a stable, documented
//! contract between the Rust bridge and a provider's Python worker.
//!
//! # Framing
//!
//! Every message is one *frame* on the worker's stdin (requests) or stdout
//! (responses):
//!
//! ```text
//! frame = u32 little-endian length  ||  UTF-8 JSON payload of that length
//! ```
//!
//! stdout carries protocol frames **only**; anything the worker logs goes to
//! stderr (mixing the two corrupts the stream). JSON was chosen over a binary
//! codec deliberately: the Fase-2 measurement showed the format is never the
//! bottleneck next to QPU latency, and human-readable frames keep the protocol
//! debuggable. See `docs/backends.md` for the full specification and the worker
//! author's contract.
//!
//! # Handshake
//!
//! On spawn the bridge sends [`Request::Hello`] with its [`PROTOCOL_VERSION`]; the
//! worker must answer [`Response::Ready`] with the version it speaks. A mismatch is
//! a hard error — the bridge refuses to run against a worker it cannot understand.
//!
//! # Messages
//!
//! Requests (bridge → worker): [`Request::Hello`], [`Request::Run`],
//! [`Request::Shutdown`]. Responses (worker → bridge): [`Response::Ready`],
//! [`Response::Result`], [`Response::Aborted`], [`Response::Error`].

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// The wire-protocol version this crate speaks. Bumped only on a breaking change to
/// the frame shapes below; a worker declares the version it supports in its
/// [`Response::Ready`], and the bridge refuses a mismatch.
pub const PROTOCOL_VERSION: u32 = 1;

/// One circuit in a [`Request::Run`], as OpenQASM 2.0 text plus its qubit width.
///
/// The bridge sends circuits as QASM2 (a provider-native `Foreign` circuit cannot
/// cross a subprocess boundary and is rejected before it gets here). `n_qubits` is
/// included so a worker can size its device without re-parsing the program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circuit {
    /// OpenQASM 2.0 source of the fully-bound circuit.
    pub qasm: String,
    /// Number of qubits in the circuit.
    pub n_qubits: u32,
}

/// A request frame sent from the bridge to the worker.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// Handshake: "I speak protocol `protocol`; do you?". Sent once, first.
    Hello {
        /// The protocol version the bridge speaks ([`PROTOCOL_VERSION`]).
        protocol: u32,
    },
    /// Execute `circuits` with `shots` shots each, optionally seeded. The worker
    /// replies [`Response::Result`] (counts, one map per circuit, in order),
    /// [`Response::Aborted`] (a cancellation signal cut the call short), or
    /// [`Response::Error`].
    Run {
        /// Run identifier, echoed back in the response.
        id: String,
        /// Circuits to execute, in order.
        circuits: Vec<Circuit>,
        /// Shots per circuit.
        shots: u32,
        /// Optional RNG seed for shot sampling.
        #[serde(skip_serializing_if = "Option::is_none")]
        seed: Option<u64>,
    },
    /// Ask the worker to exit cleanly. The worker closes stdout and returns.
    Shutdown,
}

/// A response frame received from the worker.
///
/// Untagged fields default to absent so a slightly older/newer worker that omits an
/// optional field still decodes (forward compatibility within a protocol major).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Response {
    /// Handshake reply: the protocol version the worker speaks.
    Ready {
        /// The worker's protocol version; must equal the bridge's.
        protocol: u32,
    },
    /// A completed run: one bitstring→count map per submitted circuit, in order.
    Result {
        /// Echoes the request `id` (advisory).
        #[serde(default)]
        id: Option<String>,
        /// Measurement counts, one map per circuit.
        counts: Vec<HashMap<String, u64>>,
    },
    /// The in-flight call was aborted by a cancellation signal; the worker stayed
    /// alive and is reusable. Carries no counts.
    Aborted {
        /// Echoes the request `id` (advisory).
        #[serde(default)]
        id: Option<String>,
        /// Why it aborted (e.g. `"signal"`); advisory.
        #[serde(default)]
        reason: Option<String>,
    },
    /// The worker failed to service the request (bad input, provider error, …).
    /// A clean, reportable failure — distinct from the worker *dying*, which the
    /// bridge detects as EOF, and from it *hanging*, which the bridge detects as a
    /// read timeout.
    Error {
        /// Echoes the request `id` if known (advisory).
        #[serde(default)]
        id: Option<String>,
        /// Human-readable failure message.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_request_serialises_with_op_tag_and_skips_none_seed() {
        let req = Request::Run {
            id: "r1".to_string(),
            circuits: vec![Circuit {
                qasm: "OPENQASM 2.0;".to_string(),
                n_qubits: 2,
            }],
            shots: 100,
            seed: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"op\":\"run\""));
        assert!(json.contains("\"shots\":100"));
        // seed is None → omitted.
        assert!(!json.contains("seed"));
    }

    #[test]
    fn hello_carries_the_protocol_version() {
        let json = serde_json::to_string(&Request::Hello {
            protocol: PROTOCOL_VERSION,
        })
        .unwrap();
        assert_eq!(json, r#"{"op":"hello","protocol":1}"#);
    }

    #[test]
    fn result_response_decodes_counts() {
        let raw = r#"{"op":"result","id":"r1","counts":[{"00":50,"11":50}]}"#;
        match serde_json::from_str::<Response>(raw).unwrap() {
            Response::Result { id, counts } => {
                assert_eq!(id.as_deref(), Some("r1"));
                assert_eq!(counts.len(), 1);
                assert_eq!(counts[0].get("11"), Some(&50));
            }
            other => panic!("expected Result, got {other:?}"),
        }
    }

    #[test]
    fn ready_and_aborted_and_error_decode() {
        assert!(matches!(
            serde_json::from_str::<Response>(r#"{"op":"ready","protocol":1}"#).unwrap(),
            Response::Ready { protocol: 1 }
        ));
        assert!(matches!(
            serde_json::from_str::<Response>(r#"{"op":"aborted","id":"r1","reason":"signal"}"#)
                .unwrap(),
            Response::Aborted { .. }
        ));
        match serde_json::from_str::<Response>(r#"{"op":"error","message":"boom"}"#).unwrap() {
            Response::Error { message, .. } => assert_eq!(message, "boom"),
            other => panic!("expected Error, got {other:?}"),
        }
    }
}
