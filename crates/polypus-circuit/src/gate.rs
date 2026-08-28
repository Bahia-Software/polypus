//! Core gate data types: [`GateParam`] and [`GateInstruction`].

use crate::error::CircuitError;
use std::collections::BTreeSet;

/// An angle argument of a rotation gate.
///
/// Either a concrete value ([`Fixed`](GateParam::Fixed)) or a reference to the
/// free parameter at a given index ([`Param`](GateParam::Param)), to be bound
/// later via [`ParameterizedCircuit::assign_parameters`](crate::ParameterizedCircuit::assign_parameters).
///
/// `GateParam` implements `From<f64>`, so builder methods accept plain floats:
///
/// ```
/// use polypus_circuit::{ParameterizedCircuit, Param};
///
/// let qc = ParameterizedCircuit::new(1)
///     .rx(0, 0.5)        // fixed angle
///     .rz(0, Param(0));  // free parameter #0
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateParam {
    /// A concrete angle value (radians).
    Fixed(f64),
    /// A reference to the free parameter at this index.
    Param(usize),
}

impl From<f64> for GateParam {
    fn from(value: f64) -> Self {
        GateParam::Fixed(value)
    }
}

impl GateParam {
    /// Resolve to a concrete value, looking up `Param` indices in `params`.
    ///
    /// Rejects a non-finite result — whether from a `Fixed` angle or from a
    /// caller-supplied value bound to a `Param` — with
    /// [`CircuitError::NonFiniteParam`], since `NaN`/infinity is not a valid
    /// rotation angle (mirrors the simulator, contract C-2).
    pub(crate) fn resolve(&self, params: &[f64]) -> Result<f64, CircuitError> {
        let value = match *self {
            GateParam::Fixed(v) => v,
            GateParam::Param(i) => *params.get(i).ok_or(CircuitError::ParamIndexOutOfBounds {
                index: i,
                num_params: params.len(),
            })?,
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(CircuitError::NonFiniteParam)
        }
    }
}

/// A single instruction in a quantum circuit.
///
/// Gate names follow the standard `qelib1.inc` vocabulary of OpenQASM 2.0
/// (`h`, `cx`, `rz`, `rzz`, `u3`, …), so every variant maps one-to-one onto a
/// QASM statement.
#[derive(Debug, Clone, PartialEq)]
pub enum GateInstruction {
    /// Hadamard gate.
    H(usize),
    /// Pauli-X gate.
    X(usize),
    /// Pauli-Y gate.
    Y(usize),
    /// Pauli-Z gate.
    Z(usize),
    /// Phase gate S (√Z).
    S(usize),
    /// T gate (√S).
    T(usize),
    /// Conjugate transpose of S.
    Sdg(usize),
    /// Conjugate transpose of T.
    Tdg(usize),
    /// Rotation around the X axis.
    Rx { qubit: usize, theta: GateParam },
    /// Rotation around the Y axis.
    Ry { qubit: usize, theta: GateParam },
    /// Rotation around the Z axis.
    Rz { qubit: usize, theta: GateParam },
    /// Controlled-NOT: control, target.
    Cx(usize, usize),
    /// Controlled-Z: control, target.
    Cz(usize, usize),
    /// Two-qubit ZZ-interaction rotation, exp(-i θ/2 Z⊗Z).
    Rzz {
        q0: usize,
        q1: usize,
        theta: GateParam,
    },
    /// Two-qubit XX-interaction rotation, exp(-i θ/2 X⊗X).
    Rxx {
        q0: usize,
        q1: usize,
        theta: GateParam,
    },
    /// Controlled phase gate: control, target, angle.
    Cp {
        q0: usize,
        q1: usize,
        theta: GateParam,
    },
    /// Generic single-qubit gate `u3(theta, phi, lambda)`.
    U {
        qubit: usize,
        theta: GateParam,
        phi: GateParam,
        lam: GateParam,
    },
    /// Barrier. An empty vector means "all qubits" (`barrier q;`).
    Barrier(Vec<usize>),
    /// Measure one qubit into one classical bit.
    Measure { qubit: usize, cbit: usize },
    /// Measure every qubit `i` into classical bit `i` (`measure q -> c;`).
    MeasureAll,
}

/// The qubits an instruction evolves *unitarily*.
///
/// Used by the terminal-measurement check (contract C-4): only unitary
/// operations are forbidden on an already-measured qubit. `Barrier`,
/// `Measure` and `MeasureAll` do not evolve the state and therefore report
/// [`ActsOn::None`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActsOn {
    /// A non-unitary instruction (barrier / measure / measure_all).
    None,
    /// A single-qubit unitary on this qubit.
    One(usize),
    /// A two-qubit unitary on these qubits.
    Two(usize, usize),
}

impl GateInstruction {
    /// Largest classical-bit index used by this instruction, if any.
    /// `MeasureAll` is handled separately by the circuit (it needs `num_qubits`).
    pub(crate) fn max_cbit(&self) -> Option<usize> {
        match self {
            GateInstruction::Measure { cbit, .. } => Some(*cbit),
            _ => None,
        }
    }

    /// Which qubits this instruction acts on *as a unitary* (see [`ActsOn`]).
    pub(crate) fn acts_on(&self) -> ActsOn {
        match self {
            GateInstruction::H(q)
            | GateInstruction::X(q)
            | GateInstruction::Y(q)
            | GateInstruction::Z(q)
            | GateInstruction::S(q)
            | GateInstruction::T(q)
            | GateInstruction::Sdg(q)
            | GateInstruction::Tdg(q)
            | GateInstruction::Rx { qubit: q, .. }
            | GateInstruction::Ry { qubit: q, .. }
            | GateInstruction::Rz { qubit: q, .. }
            | GateInstruction::U { qubit: q, .. } => ActsOn::One(*q),
            GateInstruction::Cx(a, b)
            | GateInstruction::Cz(a, b)
            | GateInstruction::Rzz { q0: a, q1: b, .. }
            | GateInstruction::Rxx { q0: a, q1: b, .. }
            | GateInstruction::Cp { q0: a, q1: b, .. } => ActsOn::Two(*a, *b),
            GateInstruction::Barrier(_)
            | GateInstruction::Measure { .. }
            | GateInstruction::MeasureAll => ActsOn::None,
        }
    }
}

/// Incremental record of which qubits a circuit has already measured, used for
/// the push-time C-4 check in
/// [`ParameterizedCircuit::try_push`](crate::ParameterizedCircuit::try_push).
///
/// Rescanning the whole instruction list on every push made building a circuit
/// of `G` gates cost O(G²); this cache makes each push O(log G) in the number of
/// *distinct measured qubits* (O(1) amortized in the gate count). It mirrors the
/// `measured: BTreeSet<usize>` the QASM importer's parser already carries, plus
/// an `all` flag, because unlike the importer the builder is handed
/// [`GateInstruction::MeasureAll`] directly by user code.
///
/// The cache is derived state: it is *not* part of a circuit's identity (see the
/// `PartialEq` impl for [`ParameterizedCircuit`](crate::ParameterizedCircuit)),
/// and a default value means "not derived yet" rather than "nothing measured",
/// so a circuit assembled field-by-field — bypassing `try_push` entirely, as the
/// QASM importer and several tests do — still gets the exact same answers as the
/// old full rescan on its first push.
#[derive(Debug, Clone, Default)]
pub(crate) struct MeasuredQubits {
    /// `false` in a freshly built value: [`Self::sync`] has not yet reconstructed
    /// the cache from the circuit's instruction list.
    derived: bool,
    /// Qubits covered by an explicit [`GateInstruction::Measure`].
    qubits: BTreeSet<usize>,
    /// Set by [`GateInstruction::MeasureAll`], which measures *every* qubit. Kept
    /// as a flag rather than expanded into `qubits` so that, exactly as before,
    /// an out-of-range qubit index after a `MeasureAll` is reported as
    /// already-measured rather than out-of-range.
    all: bool,
}

impl MeasuredQubits {
    /// Reconstruct the cache from `gates` unless it is already up to date. O(G)
    /// once per circuit that was assembled without going through `try_push`,
    /// O(1) on every subsequent push.
    ///
    /// This deliberately does not validate `gates`: hand-assembled sequences may
    /// violate C-4, and the builder's job is only to answer "was this qubit
    /// measured in the prefix", which is what the old rescan did too.
    pub(crate) fn sync(&mut self, gates: &[GateInstruction]) {
        if self.derived {
            return;
        }
        for gate in gates {
            self.record(gate);
        }
        self.derived = true;
    }

    /// Whether `qubit` has already been measured. Only meaningful after
    /// [`Self::sync`].
    pub(crate) fn contains(&self, qubit: usize) -> bool {
        self.all || self.qubits.contains(&qubit)
    }

    /// Fold one instruction into the cache. Called for every gate the builder
    /// accepts, on the success path only; non-measurement instructions
    /// (including `Barrier`, which C-4 always allows) are inert.
    pub(crate) fn record(&mut self, gate: &GateInstruction) {
        match gate {
            GateInstruction::Measure { qubit, .. } => {
                self.qubits.insert(*qubit);
            }
            GateInstruction::MeasureAll => self.all = true,
            _ => {}
        }
    }
}

/// Scan a full instruction sequence for a violation of the terminal-measurement
/// model (contract C-4): a **unitary** gate acting on a qubit that an earlier
/// instruction already measured. Returns the offending qubit, or `None` when
/// the sequence is terminal.
///
/// Semantics (see `docs/adr/0001-terminal-measurements.md`):
/// - a unitary on a measured qubit is a violation;
/// - `Barrier` is always allowed (a scheduling hint, it touches no state);
/// - re-measuring an already-measured qubit is allowed (idempotent).
///
/// This is the shared reference used by the builder, the QASM importer, the QIR
/// exporter and the native simulator so all four reject identically.
pub fn terminal_measurement_violation(gates: &[GateInstruction]) -> Option<usize> {
    let mut measure_all = false;
    let mut measured: Vec<usize> = Vec::new();
    for gate in gates {
        let offending = match gate.acts_on() {
            ActsOn::One(q) if measure_all || measured.contains(&q) => Some(q),
            ActsOn::Two(a, _) if measure_all || measured.contains(&a) => Some(a),
            ActsOn::Two(_, b) if measure_all || measured.contains(&b) => Some(b),
            _ => None,
        };
        if offending.is_some() {
            return offending;
        }
        match gate {
            GateInstruction::Measure { qubit, .. } => {
                if !measured.contains(qubit) {
                    measured.push(*qubit);
                }
            }
            GateInstruction::MeasureAll => measure_all = true,
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_eval_integration() {
        let param_fixed = GateParam::Fixed(1.5);
        let param_variable = GateParam::Param(0);
        let external_values = vec![3.0];

        let fixed_instruction = GateInstruction::Rx {
            qubit: 0,
            theta: param_fixed,
        };
        let variable_instruction = GateInstruction::Rx {
            qubit: 0,
            theta: param_variable,
        };

        let fixed_result = match fixed_instruction {
            GateInstruction::Rx { theta, .. } => theta.resolve(&external_values),
            _ => panic!("Expected Rx"),
        };

        let variable_result = match variable_instruction {
            GateInstruction::Rx { theta, .. } => theta.resolve(&external_values),
            _ => panic!("Expected Rx"),
        };

        assert_eq!(fixed_result.unwrap(), 1.5);
        assert_eq!(variable_result.unwrap(), 3.0);
    }

    #[test]
    fn test_param_multiple_index() {
        let params = vec![10.0, 20.0, 30.0];

        assert_eq!(GateParam::Param(0).resolve(&params).unwrap(), 10.0);
        assert_eq!(GateParam::Param(1).resolve(&params).unwrap(), 20.0);
        assert_eq!(GateParam::Param(2).resolve(&params).unwrap(), 30.0);
    }

    #[test]
    fn test_param_eval_out_of_bounds() {
        let param_variable = GateParam::Param(1);
        let external_values = vec![3.0];

        let variable_instruction = GateInstruction::Rx {
            qubit: 0,
            theta: param_variable,
        };

        let variable_result = match variable_instruction {
            GateInstruction::Rx { theta, .. } => theta.resolve(&external_values),
            _ => panic!("Expected Rx"),
        };

        match variable_result {
            Err(CircuitError::ParamIndexOutOfBounds { index, num_params }) => {
                assert_eq!(index, 1);
                assert_eq!(num_params, 1);
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_param_resolve_empty_params() {
        let param = GateParam::Param(0);
        let params = vec![];

        let result = param.resolve(&params);

        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_keeps_original_values() {
        let param = GateParam::Param(0);
        let params = vec![42.3];

        let _ = param.resolve(&params);

        assert_eq!(params, vec![42.3]);
    }

    #[test]
    fn test_resolve_special_values() {
        // A `Fixed` non-finite angle is rejected directly.
        assert_eq!(
            GateParam::Fixed(f64::NAN).resolve(&[]),
            Err(CircuitError::NonFiniteParam)
        );
        assert_eq!(
            GateParam::Fixed(f64::INFINITY).resolve(&[]),
            Err(CircuitError::NonFiniteParam)
        );

        // A caller-supplied non-finite value bound to a `Param` is also rejected.
        let params = vec![f64::INFINITY, f64::NAN];
        assert_eq!(
            GateParam::Param(0).resolve(&params),
            Err(CircuitError::NonFiniteParam)
        );
        assert_eq!(
            GateParam::Param(1).resolve(&params),
            Err(CircuitError::NonFiniteParam)
        );
    }

    #[test]
    fn test_max_cbit_measure() {
        let instruction = GateInstruction::Measure { qubit: 0, cbit: 5 };

        let result = instruction.max_cbit();

        assert_eq!(result, Some(5));
    }

    #[test]
    fn test_max_cbit_non_measure() {
        let instruction = GateInstruction::H(0);

        let result = instruction.max_cbit();

        assert_eq!(result, None);
    }

    #[test]
    fn test_max_cbit_measure_all() {
        let instruction = GateInstruction::MeasureAll;

        let result = instruction.max_cbit();

        assert_eq!(result, None);
    }

    // ── MeasuredQubits (push-time C-4 cache) ─────────────────────────────

    #[test]
    fn measured_qubits_records_only_measurements() {
        let mut measured = MeasuredQubits::default();
        measured.sync(&[]);

        measured.record(&GateInstruction::H(0));
        measured.record(&GateInstruction::Cx(0, 1));
        measured.record(&GateInstruction::Barrier(vec![0, 1]));
        assert!(!measured.contains(0));
        assert!(!measured.contains(1));

        measured.record(&GateInstruction::Measure { qubit: 1, cbit: 0 });
        assert!(!measured.contains(0));
        assert!(measured.contains(1));

        // Re-measuring is idempotent, and a barrier on a measured qubit is inert.
        measured.record(&GateInstruction::Measure { qubit: 1, cbit: 1 });
        measured.record(&GateInstruction::Barrier(vec![1]));
        assert!(measured.contains(1));
        assert!(!measured.contains(0));
    }

    #[test]
    fn measured_qubits_measure_all_covers_every_index() {
        let mut measured = MeasuredQubits::default();
        measured.sync(&[]);

        measured.record(&GateInstruction::MeasureAll);

        assert!(measured.contains(0));
        assert!(measured.contains(7));
        // Deliberately beyond any plausible register: `MeasureAll` answers for
        // out-of-range indices too, so `try_push` reports them as
        // already-measured rather than out-of-range, exactly as the old rescan did.
        assert!(measured.contains(usize::MAX));
    }

    #[test]
    fn measured_qubits_sync_derives_from_an_existing_gate_list_once() {
        let gates = vec![
            GateInstruction::H(0),
            GateInstruction::Measure { qubit: 0, cbit: 0 },
            GateInstruction::Measure { qubit: 2, cbit: 1 },
        ];

        let mut measured = MeasuredQubits::default();
        measured.sync(&gates);

        assert!(measured.contains(0));
        assert!(!measured.contains(1));
        assert!(measured.contains(2));

        // A second sync is a no-op: once derived, the cache is maintained by
        // `record` alone and must not re-fold the (now stale) prefix.
        measured.sync(&[GateInstruction::Measure { qubit: 1, cbit: 2 }]);
        assert!(!measured.contains(1));
    }

    #[test]
    fn measured_qubits_sync_does_not_validate() {
        // A hand-assembled sequence that violates C-4 still yields the plain
        // "was this qubit measured in the prefix" answer.
        let gates = vec![
            GateInstruction::Measure { qubit: 0, cbit: 0 },
            GateInstruction::X(0),
        ];

        let mut measured = MeasuredQubits::default();
        measured.sync(&gates);

        assert!(measured.contains(0));
    }
}
