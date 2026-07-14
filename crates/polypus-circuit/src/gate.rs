//! Core gate data types: [`GateParam`] and [`GateInstruction`].

use crate::error::CircuitError;

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

/// Whether `gates` already measured `qubit` (via a matching `Measure` or any
/// `MeasureAll`). Used for the incremental C-4 check in
/// [`ParameterizedCircuit::try_push`](crate::ParameterizedCircuit::try_push),
/// where the existing prefix is known to already respect the contract.
pub(crate) fn is_qubit_measured(gates: &[GateInstruction], qubit: usize) -> bool {
    gates.iter().any(|g| match g {
        GateInstruction::MeasureAll => true,
        GateInstruction::Measure { qubit: q, .. } => *q == qubit,
        _ => false,
    })
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
}
