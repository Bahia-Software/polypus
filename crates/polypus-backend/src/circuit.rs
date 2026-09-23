//! [`BoundCircuit`] — a fully bound, parameter-free circuit ready for execution —
//! and [`ForeignCircuit`], the pyo3-free escape hatch that lets a backend carry a
//! provider-native circuit object (e.g. a Qiskit `QuantumCircuit`) through this
//! crate without `polypus-backend` ever depending on that provider's SDK.

use std::any::Any;
use std::fmt;

use polypus_circuit::ConcreteCircuit;

use crate::transpiler::{TranspileOptions, Transpiler};

/// A provider-native circuit object, type-erased so it can travel inside a
/// [`BoundCircuit::Foreign`] without `polypus-backend` naming the provider's type.
///
/// This is the seam that keeps this crate free of PyO3 and Qiskit while still
/// letting the Polypus Python backends (and any third-party backend) submit a
/// circuit object their SDK understands. The provider implements this trait for a
/// newtype wrapping its object (Polypus wraps a Qiskit `QuantumCircuit` in
/// `polypus_infrastructure::QiskitCircuit`), and its own backend downcasts back to
/// that concrete type via [`as_any`](Self::as_any) at execution time. A backend
/// handed a `Foreign` it does not recognise returns
/// [`BackendError::UnsupportedCircuit`](crate::BackendError::UnsupportedCircuit).
///
/// A pure-Rust, provider-agnostic backend never needs this trait: it consumes the
/// [`Native`](BoundCircuit::Native) / [`Qasm2`](BoundCircuit::Qasm2) variants,
/// which carry no foreign types at all.
pub trait ForeignCircuit: Send + Sync + fmt::Debug {
    /// Clone into a fresh box. The Qiskit implementation bumps the Python object's
    /// reference count under the GIL; a pure-Rust implementation is a plain clone.
    /// This is what makes a [`BoundCircuit`] carrying a `Foreign` variant
    /// [`duplicate`](BoundCircuit::duplicate)-able without this crate knowing how.
    fn clone_boxed(&self) -> Box<dyn ForeignCircuit>;

    /// Downcast handle so the owning backend can recover the concrete provider
    /// type (`self.as_any().downcast_ref::<MyCircuit>()`).
    fn as_any(&self) -> &dyn Any;
}

impl Clone for Box<dyn ForeignCircuit> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

/// A fully bound (parameter-free) circuit ready for execution, in one of the
/// representations Polypus supports.
///
/// Backends receive a slice of these and decide how to consume each variant. The
/// two native variants are provider-agnostic and pyo3-free; the
/// [`Foreign`](Self::Foreign) variant is the opt-in escape hatch for a
/// provider-native object (see [`ForeignCircuit`]).
///
/// Being an enum (rather than a bare `Box<dyn Any>`) makes the contract explicit:
/// adding a new native representation is a compile-time-checked change in every
/// backend, while an unrecognised `Foreign` payload is a runtime
/// [`UnsupportedCircuit`](crate::BackendError::UnsupportedCircuit), not a panic.
#[derive(Debug)]
pub enum BoundCircuit {
    /// A fully bound native circuit from `polypus-circuit`. Carries the circuit
    /// structure directly so a native statevector backend can simulate it without
    /// any OpenQASM round-trip or GIL; Python-based backends serialise it to
    /// OpenQASM 2.0 on demand.
    Native(ConcreteCircuit),
    /// An OpenQASM 2.0 program (produced by the native circuit layer, or supplied
    /// as a raw string). Consumed directly by native/wire-protocol backends and
    /// forwarded as a `str` to Python-based backends.
    Qasm2(String),
    /// A provider-native circuit object, carried opaquely (see [`ForeignCircuit`]).
    /// In Polypus this is always a Qiskit `QuantumCircuit`; the Python backends
    /// (Aer/CUNQA) submit it as-is, and every GIL-free backend rejects it.
    Foreign(Box<dyn ForeignCircuit>),
}

impl BoundCircuit {
    /// Cheap copy. Only the `Foreign` variant needs any provider machinery
    /// (delegated to [`ForeignCircuit::clone_boxed`], e.g. a GIL-held ref-count
    /// bump for Qiskit); the `Native` and `Qasm2` variants are plain Rust clones.
    pub fn duplicate(&self) -> BoundCircuit {
        match self {
            BoundCircuit::Native(circuit) => BoundCircuit::Native(circuit.clone()),
            BoundCircuit::Qasm2(qasm) => BoundCircuit::Qasm2(qasm.clone()),
            BoundCircuit::Foreign(foreign) => BoundCircuit::Foreign(foreign.clone_boxed()),
        }
    }

    /// Whether this is a [`Foreign`](Self::Foreign) provider object. Backends that
    /// cannot consume a provider-native circuit (the native statevector simulator,
    /// the QMIO wire path) use this to reject it up front with a clear error.
    pub fn is_foreign(&self) -> bool {
        matches!(self, BoundCircuit::Foreign(_))
    }

    /// Rewrite the *native domain* of this circuit with `transpiler`, returning a
    /// transpiled `BoundCircuit`. This is the composition point Python-backed
    /// backends share: each applies its injected [`Transpiler`] to every circuit
    /// just before submission.
    ///
    /// The transpiler is intentionally confined to the GIL-free native domain:
    ///
    /// - [`Native`](Self::Native) is transpiled directly on its
    ///   [`ConcreteCircuit`].
    /// - [`Qasm2`](Self::Qasm2) is parsed back to a
    ///   [`ConcreteCircuit`], transpiled, and
    ///   re-emitted as OpenQASM 2.0. This is *best-effort*: if the QASM cannot be
    ///   parsed (an unsupported construct, a non-native program), the original
    ///   text is returned untouched rather than panicking.
    /// - [`Foreign`](Self::Foreign) is returned as-is: a provider transpiles its
    ///   own object internally, and rewriting one here would require reading its
    ///   gates through the provider SDK, crossing the deliberate native/foreign
    ///   boundary.
    pub fn transpiled(&self, transpiler: &dyn Transpiler, opts: &TranspileOptions) -> BoundCircuit {
        // A guaranteed no-op transpiler changes nothing, so skip the parse →
        // transpile → re-emit round trip (Qasm2) and the trait-dispatch clone
        // (Native) entirely, keeping only the cheap representation-preserving
        // copy. `Foreign` is already a passthrough via `duplicate`.
        if transpiler.is_identity() {
            return match self {
                BoundCircuit::Native(cc) => BoundCircuit::Native(cc.clone()),
                BoundCircuit::Qasm2(qasm) => BoundCircuit::Qasm2(qasm.clone()),
                BoundCircuit::Foreign(_) => self.duplicate(),
            };
        }
        match self {
            BoundCircuit::Native(cc) => BoundCircuit::Native(transpiler.transpile(cc, opts)),
            BoundCircuit::Qasm2(qasm) => {
                match polypus_circuit::ParameterizedCircuit::from_qasm2(qasm)
                    .and_then(|pc| pc.assign_parameters(&[]))
                {
                    Ok(cc) => BoundCircuit::Qasm2(transpiler.transpile(&cc, opts).to_qasm2()),
                    // Unparseable QASM is left intact (best-effort, no new panic).
                    Err(_) => BoundCircuit::Qasm2(qasm.clone()),
                }
            }
            BoundCircuit::Foreign(_) => self.duplicate(),
        }
    }

    /// Qubit width of this circuit as read from its **native domain**, GIL-free:
    /// [`Native`](Self::Native) exposes it directly and [`Qasm2`](Self::Qasm2) is
    /// parsed for it (returning `None` if the program cannot be parsed). A
    /// [`Foreign`](Self::Foreign) circuit returns `None`, because reading its
    /// width would require the provider SDK and "what to do about a foreign
    /// circuit" is caller-specific — the native backend rejects it, and the local
    /// backend layers its own provider-side read on top of this primitive.
    ///
    /// This is the single width-extraction rule shared by the memory-budget scans
    /// in the native and local backends, sizing the widest statevector
    /// identically. It plays the same cross-backend "shared arithmetic" role as
    /// [`wave_concurrency`](crate::wave_concurrency).
    pub fn native_qubit_width(&self) -> Option<usize> {
        match self {
            BoundCircuit::Native(cc) => Some(cc.num_qubits),
            BoundCircuit::Qasm2(qasm) => polypus_circuit::ParameterizedCircuit::from_qasm2(qasm)
                .ok()
                .map(|pc| pc.num_qubits),
            BoundCircuit::Foreign(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transpiler::IdentityTranspiler;
    use polypus_circuit::ParameterizedCircuit;

    /// A pure-Rust `ForeignCircuit` stand-in, proving the escape hatch needs no
    /// PyO3: `clone_boxed` is a plain clone and `as_any` downcasts back.
    #[derive(Debug, Clone, PartialEq)]
    struct DummyForeign(u32);
    impl ForeignCircuit for DummyForeign {
        fn clone_boxed(&self) -> Box<dyn ForeignCircuit> {
            Box::new(self.clone())
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn reads_native_and_qasm2_widths_and_is_none_for_foreign() {
        let native =
            BoundCircuit::Native(ParameterizedCircuit::new(5).assign_parameters(&[]).unwrap());
        assert_eq!(native.native_qubit_width(), Some(5));

        let qasm = BoundCircuit::Qasm2(
            ParameterizedCircuit::new(3)
                .h(0)
                .measure_all()
                .assign_parameters(&[])
                .unwrap()
                .to_qasm2(),
        );
        assert_eq!(qasm.native_qubit_width(), Some(3));

        let foreign = BoundCircuit::Foreign(Box::new(DummyForeign(30)));
        assert_eq!(foreign.native_qubit_width(), None);
        assert!(foreign.is_foreign());
        assert!(!native.is_foreign());
    }

    #[test]
    fn unparseable_qasm2_yields_none() {
        let bad = BoundCircuit::Qasm2("this is definitely not valid openqasm".to_string());
        assert_eq!(bad.native_qubit_width(), None);
    }

    /// `duplicate` round-trips a `Foreign` payload through `clone_boxed` and the
    /// concrete type is recoverable via `as_any` — the downcast a real backend
    /// performs at execution.
    #[test]
    fn foreign_duplicates_and_downcasts() {
        let original = BoundCircuit::Foreign(Box::new(DummyForeign(7)));
        let copy = original.duplicate();
        match copy {
            BoundCircuit::Foreign(f) => {
                let recovered = f.as_any().downcast_ref::<DummyForeign>().unwrap();
                assert_eq!(recovered, &DummyForeign(7));
            }
            _ => panic!("duplicate must preserve the Foreign variant"),
        }
    }

    /// Under the identity transpiler, a `Qasm2` program is returned byte-identical
    /// (no parse + re-emit), and a `Foreign` circuit is passed through unchanged.
    #[test]
    fn identity_transpile_preserves_qasm_bytes_and_passes_foreign_through() {
        let original = "OPENQASM 2.0;\n// a comment dropped on parse\nqreg q[1];\nh q[0];\n";
        let out = BoundCircuit::Qasm2(original.to_string())
            .transpiled(&IdentityTranspiler, &TranspileOptions::default());
        match out {
            BoundCircuit::Qasm2(text) => assert_eq!(text, original),
            _ => panic!("expected the original Qasm2 variant"),
        }

        let foreign = BoundCircuit::Foreign(Box::new(DummyForeign(2)))
            .transpiled(&IdentityTranspiler, &TranspileOptions::default());
        assert!(foreign.is_foreign());
    }
}
