//! # polypus-sim
//!
//! A pure-Rust statevector simulator for [`polypus_circuit`] circuits.
//!
//! Unlike the Qiskit path, this backend consumes a
//! [`ConcreteCircuit`](polypus_circuit::ConcreteCircuit) **directly** — there
//! is no OpenQASM round-trip and no Python interpreter, so simulation runs
//! entirely in Rust and free of the GIL. With the `parallel` feature it spreads
//! each gate across a rayon thread pool.
//!
//! ## Example: a Bell state
//!
//! ```
//! use polypus_circuit::ParameterizedCircuit;
//! use polypus_sim::{Simulator, StatevectorSimulator};
//!
//! let circuit = ParameterizedCircuit::new(2)
//!     .h(0)
//!     .cx(0, 1)
//!     .assign_parameters(&[])
//!     .unwrap();
//!
//! let sv = StatevectorSimulator::new().run(&circuit).unwrap();
//! let amps = sv.amplitudes();
//!
//! // (|00> + |11>) / sqrt(2)
//! assert!((amps[0].norm_sqr() - 0.5).abs() < 1e-12);
//! assert!((amps[3].norm_sqr() - 0.5).abs() < 1e-12);
//! assert!(amps[1].norm_sqr() < 1e-12);
//! assert!(amps[2].norm_sqr() < 1e-12);
//! ```
//!
//! ## Scalability
//!
//! A statevector stores `2^n` complex amplitudes (`16 · 2^n` bytes), so memory
//! — not Rust — is the ceiling: ~16 MiB at 20 qubits, ~16 GiB at 30. Requests
//! above [`MAX_QUBITS`] return [`SimError::TooManyQubits`] instead of
//! attempting a doomed allocation. Gate kernels are applied in place (no
//! per-gate allocation), with a diagonal fast path and optional parallelism.

#![deny(clippy::all)]

/// 64-bit complex amplitude type used throughout the simulator.
pub type C64 = num_complex::Complex<f64>;

mod error;
mod gates;
mod kernels;
mod measure;
mod rng;
mod simulator;
mod statevector;

pub use error::SimError;
pub use rng::SplitMix64;
pub use simulator::{sample_projected, Simulator, StatevectorSimulator};
pub use statevector::Statevector;

/// Maximum number of qubits the statevector backend will allocate.
///
/// A statevector needs `2^n` complex amplitudes (`16 · 2^n` bytes), so 30
/// qubits already requires 16 GiB. Requests above this return
/// [`SimError::TooManyQubits`] rather than attempting the allocation. The value
/// is also well below `usize::BITS`, so `1 << n` can never overflow.
pub const MAX_QUBITS: usize = 30;

/// Circuits with at least this many qubits use the parallel kernels (only when
/// the `parallel` feature is enabled). Smaller circuits run sequentially to
/// avoid thread-pool overhead that would dwarf the tiny workload.
pub(crate) const DEFAULT_PARALLEL_THRESHOLD: usize = 12;
