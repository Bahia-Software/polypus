//! # polypus-circuit
//!
//! Pure-Rust quantum circuit representation with OpenQASM 2.0 and QIR Base
//! Profile export.
//!
//! This crate is the GIL-free circuit layer of Polypus: circuits can be
//! defined, parameterized and serialized entirely in Rust, with no PyO3 or
//! Qiskit dependency. The resulting OpenQASM 2.0 can be sent to any simulator
//! or backend that accepts it (e.g. Qiskit Aer, CUNQA), and the QIR Base
//! Profile output (LLVM IR) targets QIR-consuming providers (Azure Quantum,
//! Quantinuum, and other QIR Alliance adopters).
//!
//! ## Example: QAOA MaxCut ansatz
//!
//! ```
//! use polypus_circuit::{ParameterizedCircuit, Param};
//!
//! let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
//!
//! // One QAOA layer: mixer parameter Param(0), cost parameter Param(1).
//! let mut qc = ParameterizedCircuit::new(4);
//! for q in 0..4 {
//!     qc = qc.h(q);
//! }
//! for &(a, b) in &edges {
//!     qc = qc.rzz(a, b, Param(1));
//! }
//! for q in 0..4 {
//!     qc = qc.rx(q, Param(0));
//! }
//! let qc = qc.measure_all();
//!
//! // Bind (beta, gamma) and export.
//! let qasm = qc.to_qasm2_with_params(&[0.8, 0.4]).unwrap();
//! assert!(qasm.contains("rzz(0.400000000000) q[0],q[1];"));
//! assert!(qasm.ends_with("measure q -> c;\n"));
//! ```

mod circuit;
mod error;
mod gate;
mod qasm;
mod qasm_import;
mod qir;

pub use circuit::{ConcreteCircuit, ParameterizedCircuit};
pub use error::CircuitError;
pub use gate::{terminal_measurement_violation, GateInstruction, GateParam};
// Variant re-exports so call sites can write `Param(0)` / `Fixed(0.5)` directly.
pub use gate::GateParam::{Fixed, Param};
