//! Pure-Rust, GIL-free circuit transpilation contract.
//!
//! A [`Transpiler`] rewrites a native [`ConcreteCircuit`] so that it is valid
//! for a backend's *target* (native gate-set decomposition, qubit-connectivity
//! mapping, optimization passes, …). It is the transpilation counterpart of
//! [`crate::infrastructure::QuantumBackend`]: an execution backend *composes*
//! the `Transpiler` appropriate for its target and invokes it inside its own
//! `run_circuits`, so algorithm code never has to know about hardware-specific
//! rewriting.
//!
//! Two axes of variation are kept deliberately separate:
//!
//! - The **strategy** (which rewriting algorithm) is a *dependency*: it is
//!   injected by composition as a `Box<dyn Transpiler>` on the backend.
//! - The **tuning** (how hard it optimizes, and future knobs) is an *argument*:
//!   it travels per call in [`TranspileOptions`].
//!
//! Mnemonic: *the level is an argument; the transpilation strategy is a
//! dependency. Arguments go in the signature; dependencies are injected by
//! composition.*
//!
//! This module is intentionally free of any PyO3/Python dependency so that
//! transpilation can run at HPC scale without ever taking the GIL.

use polypus_circuit::ConcreteCircuit;

/// Optimization effort for a transpilation pass, mirroring Qiskit's 0..=3 scale.
///
/// It is a *parameter* of transpilation, not a transpiler variant: the same
/// [`Transpiler`] honors different levels via [`TranspileOptions`], instead of
/// having one type per level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimization (Qiskit level 0).
    None,
    /// Light, cheap optimization (Qiskit level 1). Default.
    #[default]
    Light,
    /// Medium optimization (Qiskit level 2).
    Medium,
    /// Heavy optimization (Qiskit level 3).
    Heavy,
}

/// Provider-agnostic tuning knobs for a transpilation pass.
///
/// Each [`Transpiler`] interprets the fields it understands and ignores the
/// rest. New knobs (seed, layout strategy, approximation degree, …) are added
/// here so that the [`Transpiler`] trait signature stays stable.
#[derive(Debug, Clone, Copy, Default)]
pub struct TranspileOptions {
    /// Optimization effort.
    pub level: OptLevel,
    // future: seed: Option<u64>, approximation_degree: Option<f64>, …
}

/// Hardware-aware, pure-Rust rewriting of a native circuit: native gate-set
/// decomposition, qubit-connectivity mapping, optimization passes, etc.
///
/// Operates entirely on [`ConcreteCircuit`] so it never touches the Python GIL,
/// keeping it usable at HPC scale. Each [`crate::infrastructure::QuantumBackend`]
/// *composes* the `Transpiler` appropriate for its target (the *strategy*), and
/// passes per-run [`TranspileOptions`] (the *tuning*); algorithm code stays
/// completely unaware of any hardware-specific rewriting.
///
/// # Extending
///
/// - **New strategy** (a different rewriting algorithm): create a type that
///   implements `Transpiler` and inject it by composition (e.g.
///   [`NativeStatevectorBackend::with_transpiler`](crate::infrastructure::NativeStatevectorBackend::with_transpiler)).
///   No change to [`QuantumBackend`](crate::infrastructure::QuantumBackend),
///   the algorithms, or [`BoundCircuit`](crate::infrastructure::BoundCircuit).
/// - **New tuning knob** (seed, layout, approximation degree, …): add a field
///   to [`TranspileOptions`]. The `transpile` signature stays stable.
///
/// # Contract
///
/// Implementations MUST be semantics-preserving: the transpiled circuit must be
/// logically equivalent (up to the target's gate set / connectivity) to the
/// input. They MUST be pure (no observable side effects) and `Send + Sync`.
pub trait Transpiler: Send + Sync {
    /// Rewrite `circuit` into an equivalent one valid for the backend's target,
    /// honoring `opts` (e.g. optimization [`OptLevel`]).
    fn transpile(&self, circuit: &ConcreteCircuit, opts: &TranspileOptions) -> ConcreteCircuit;

    /// Whether [`transpile`](Self::transpile) is guaranteed to be a no-op
    /// (returns a circuit equal to its input) regardless of `opts`. Callers on
    /// hot paths use this to skip calling `transpile` — and any clone/parse
    /// leading up to it — entirely. Default `false` (conservative: only opt in
    /// when it's actually true for every [`TranspileOptions`]).
    fn is_identity(&self) -> bool {
        false
    }
}

/// No-op transpiler: returns the circuit unchanged, ignoring `opts`.
///
/// Used by backends that already transpile internally (Aer, CUNQA), so that
/// introducing [`Transpiler`] is purely additive and changes no behavior.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityTranspiler;

impl Transpiler for IdentityTranspiler {
    #[inline]
    fn transpile(&self, circuit: &ConcreteCircuit, _opts: &TranspileOptions) -> ConcreteCircuit {
        circuit.clone()
    }

    #[inline]
    fn is_identity(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polypus_circuit::{GateInstruction, ParameterizedCircuit};

    fn sample_circuit() -> ConcreteCircuit {
        ParameterizedCircuit::new(2)
            .h(0)
            .cx(0, 1)
            .rz(0, 0.5)
            .measure_all()
            .assign_parameters(&[])
            .unwrap()
    }

    /// `IdentityTranspiler` returns a circuit equal to the input for *every*
    /// optimization level — proving the level is an honored argument rather than
    /// a transpiler variant, and that the identity is a true no-op.
    #[test]
    fn identity_returns_equal_circuit_for_all_levels() {
        let circuit = sample_circuit();
        let transpiler = IdentityTranspiler;
        for level in [
            OptLevel::None,
            OptLevel::Light,
            OptLevel::Medium,
            OptLevel::Heavy,
        ] {
            let opts = TranspileOptions { level };
            assert_eq!(transpiler.transpile(&circuit, &opts), circuit);
        }
    }

    /// The defaults match the documented contract: `OptLevel::Light` and a
    /// `TranspileOptions` whose level is `Light`.
    #[test]
    fn defaults_are_light() {
        assert_eq!(OptLevel::default(), OptLevel::Light);
        assert_eq!(TranspileOptions::default().level, OptLevel::Light);
    }

    /// A custom `Transpiler` receives exactly the `OptLevel` it was passed,
    /// demonstrating end-to-end that tuning travels as an argument.
    #[test]
    fn custom_transpiler_observes_the_passed_level() {
        use std::sync::atomic::{AtomicU8, Ordering};
        use std::sync::Arc;

        struct Recorder {
            seen: Arc<AtomicU8>,
        }
        impl Transpiler for Recorder {
            fn transpile(
                &self,
                circuit: &ConcreteCircuit,
                opts: &TranspileOptions,
            ) -> ConcreteCircuit {
                self.seen.store(opts.level as u8, Ordering::SeqCst);
                circuit.clone()
            }
        }

        let seen = Arc::new(AtomicU8::new(0xFF));
        let recorder = Recorder {
            seen: Arc::clone(&seen),
        };
        let circuit = sample_circuit();
        recorder.transpile(
            &circuit,
            &TranspileOptions {
                level: OptLevel::Heavy,
            },
        );
        assert_eq!(seen.load(Ordering::SeqCst), OptLevel::Heavy as u8);
    }

    /// A strategy that rewrites the circuit (here: appends a `barrier`, a
    /// simulation no-op) is honored by the trait, proving the injection point
    /// accepts non-identity strategies.
    #[test]
    fn custom_strategy_can_rewrite_the_circuit() {
        struct AppendBarrier;
        impl Transpiler for AppendBarrier {
            fn transpile(
                &self,
                circuit: &ConcreteCircuit,
                _opts: &TranspileOptions,
            ) -> ConcreteCircuit {
                let mut out = circuit.clone();
                out.gates.push(GateInstruction::Barrier(Vec::new()));
                out
            }
        }

        let circuit = sample_circuit();
        let out = AppendBarrier.transpile(&circuit, &TranspileOptions::default());
        assert_eq!(out.gates.len(), circuit.gates.len() + 1);
        assert!(matches!(
            out.gates.last(),
            Some(GateInstruction::Barrier(_))
        ));
    }

    /// `IdentityTranspiler` advertises itself as a guaranteed no-op, while a
    /// custom `Transpiler` that does not override the default (even one that
    /// happens to return the circuit unchanged) reports `false` — the hint is
    /// opt-in, so callers only skip `transpile` when it is provably safe.
    #[test]
    fn is_identity_true_only_for_identity_transpiler() {
        // A custom strategy that never overrides `is_identity`, keeping the
        // conservative default. It even returns the circuit unchanged, proving
        // the hint is about the opt-in, not the observed behavior.
        struct DefaultingPassthrough;
        impl Transpiler for DefaultingPassthrough {
            fn transpile(
                &self,
                circuit: &ConcreteCircuit,
                _opts: &TranspileOptions,
            ) -> ConcreteCircuit {
                circuit.clone()
            }
        }

        assert!(IdentityTranspiler.is_identity());
        assert!(!DefaultingPassthrough.is_identity());
    }
}
