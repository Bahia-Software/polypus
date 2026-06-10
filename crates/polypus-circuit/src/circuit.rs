//! Circuit types: [`ParameterizedCircuit`] (free parameters) and
//! [`ConcreteCircuit`] (all angles bound).

use crate::error::CircuitError;
use crate::gate::{GateInstruction, GateParam};
use crate::qasm;

/// A quantum circuit that may contain free parameters ([`GateParam::Param`]).
///
/// Built with a fluent API; bind values with [`assign_parameters`](Self::assign_parameters)
/// or export directly with [`to_qasm2_with_params`](Self::to_qasm2_with_params).
///
/// ```
/// use polypus_circuit::{ParameterizedCircuit, Param};
///
/// let qc = ParameterizedCircuit::new(3)
///     .h(0).h(1).h(2)
///     .rzz(0, 1, Param(0))
///     .rx(0, Param(1))
///     .measure_all();
///
/// let qasm = qc.to_qasm2_with_params(&[0.4, 0.8]).unwrap();
/// assert!(qasm.starts_with("OPENQASM 2.0;"));
/// ```
///
/// # Panics
///
/// Builder methods panic immediately on structurally invalid input (qubit
/// index out of range, two-qubit gate on identical qubits). These are
/// programming errors, mirroring how Qiskit raises `CircuitError` at
/// construction time. Parameter *values* are validated fallibly at binding
/// time instead, returning [`CircuitError`].
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterizedCircuit {
    /// Number of qubits in the (single) quantum register.
    pub num_qubits: usize,
    /// Number of free parameters. Kept in sync automatically by the builder:
    /// adding a gate with `Param(i)` grows it to at least `i + 1`.
    pub num_params: usize,
    /// The instruction sequence, in execution order.
    pub gates: Vec<GateInstruction>,
}

impl ParameterizedCircuit {
    /// Create an empty circuit over `num_qubits` qubits with no parameters.
    pub fn new(num_qubits: usize) -> Self {
        ParameterizedCircuit {
            num_qubits,
            num_params: 0,
            gates: Vec::new(),
        }
    }

    // ── Internal validation helpers ──────────────────────────────────────

    fn check_qubit(&self, qubit: usize) {
        assert!(
            qubit < self.num_qubits,
            "qubit index {qubit} out of range for circuit with {} qubits",
            self.num_qubits
        );
    }

    fn check_pair(&self, q0: usize, q1: usize) {
        self.check_qubit(q0);
        self.check_qubit(q1);
        assert!(
            q0 != q1,
            "two-qubit gate requires distinct qubits, got ({q0}, {q1})"
        );
    }

    fn track_param(&mut self, param: &GateParam) {
        if let GateParam::Param(i) = param {
            self.num_params = self.num_params.max(i + 1);
        }
    }

    /// Append a raw [`GateInstruction`], validating qubit indices and keeping
    /// `num_params` in sync. All fluent builder methods funnel through here;
    /// it is also useful for programmatic construction (e.g. loops over graph
    /// edges).
    ///
    /// # Panics
    ///
    /// Panics on out-of-range qubit indices or a two-qubit gate whose qubits
    /// coincide (see type-level docs).
    pub fn push(mut self, gate: GateInstruction) -> Self {
        match &gate {
            GateInstruction::H(q)
            | GateInstruction::X(q)
            | GateInstruction::Y(q)
            | GateInstruction::Z(q)
            | GateInstruction::S(q)
            | GateInstruction::T(q)
            | GateInstruction::Sdg(q)
            | GateInstruction::Tdg(q) => self.check_qubit(*q),
            GateInstruction::Rx { qubit, theta }
            | GateInstruction::Ry { qubit, theta }
            | GateInstruction::Rz { qubit, theta } => {
                self.check_qubit(*qubit);
                self.track_param(theta);
            }
            GateInstruction::Cx(q0, q1) | GateInstruction::Cz(q0, q1) => {
                self.check_pair(*q0, *q1)
            }
            GateInstruction::Rzz { q0, q1, theta }
            | GateInstruction::Rxx { q0, q1, theta } => {
                self.check_pair(*q0, *q1);
                self.track_param(theta);
            }
            GateInstruction::U {
                qubit,
                theta,
                phi,
                lam,
            } => {
                self.check_qubit(*qubit);
                self.track_param(theta);
                self.track_param(phi);
                self.track_param(lam);
            }
            GateInstruction::Barrier(qubits) => {
                for q in qubits {
                    self.check_qubit(*q);
                }
            }
            GateInstruction::Measure { qubit, .. } => self.check_qubit(*qubit),
            GateInstruction::MeasureAll => {}
        }
        self.gates.push(gate);
        self
    }

    // ── Single-qubit gates ───────────────────────────────────────────────

    /// Hadamard gate on `qubit`.
    pub fn h(self, qubit: usize) -> Self {
        self.push(GateInstruction::H(qubit))
    }

    /// Pauli-X gate on `qubit`.
    pub fn x(self, qubit: usize) -> Self {
        self.push(GateInstruction::X(qubit))
    }

    /// Pauli-Y gate on `qubit`.
    pub fn y(self, qubit: usize) -> Self {
        self.push(GateInstruction::Y(qubit))
    }

    /// Pauli-Z gate on `qubit`.
    pub fn z(self, qubit: usize) -> Self {
        self.push(GateInstruction::Z(qubit))
    }

    /// S gate on `qubit`.
    pub fn s(self, qubit: usize) -> Self {
        self.push(GateInstruction::S(qubit))
    }

    /// T gate on `qubit`.
    pub fn t(self, qubit: usize) -> Self {
        self.push(GateInstruction::T(qubit))
    }

    /// S† gate on `qubit`.
    pub fn sdg(self, qubit: usize) -> Self {
        self.push(GateInstruction::Sdg(qubit))
    }

    /// T† gate on `qubit`.
    pub fn tdg(self, qubit: usize) -> Self {
        self.push(GateInstruction::Tdg(qubit))
    }

    /// X-rotation on `qubit`; `theta` is a fixed `f64` or a [`Param`](GateParam::Param).
    pub fn rx(self, qubit: usize, theta: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::Rx {
            qubit,
            theta: theta.into(),
        })
    }

    /// Y-rotation on `qubit`; `theta` is a fixed `f64` or a [`Param`](GateParam::Param).
    pub fn ry(self, qubit: usize, theta: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::Ry {
            qubit,
            theta: theta.into(),
        })
    }

    /// Z-rotation on `qubit`; `theta` is a fixed `f64` or a [`Param`](GateParam::Param).
    pub fn rz(self, qubit: usize, theta: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::Rz {
            qubit,
            theta: theta.into(),
        })
    }

    /// Generic single-qubit gate `u3(theta, phi, lambda)` on `qubit`.
    pub fn u(
        self,
        qubit: usize,
        theta: impl Into<GateParam>,
        phi: impl Into<GateParam>,
        lam: impl Into<GateParam>,
    ) -> Self {
        self.push(GateInstruction::U {
            qubit,
            theta: theta.into(),
            phi: phi.into(),
            lam: lam.into(),
        })
    }

    // ── Two-qubit gates ──────────────────────────────────────────────────

    /// Controlled-NOT with `control` and `target`.
    pub fn cx(self, control: usize, target: usize) -> Self {
        self.push(GateInstruction::Cx(control, target))
    }

    /// Controlled-Z with `control` and `target`.
    pub fn cz(self, control: usize, target: usize) -> Self {
        self.push(GateInstruction::Cz(control, target))
    }

    /// ZZ-interaction rotation on `(q0, q1)`.
    pub fn rzz(self, q0: usize, q1: usize, theta: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::Rzz {
            q0,
            q1,
            theta: theta.into(),
        })
    }

    /// XX-interaction rotation on `(q0, q1)`.
    pub fn rxx(self, q0: usize, q1: usize, theta: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::Rxx {
            q0,
            q1,
            theta: theta.into(),
        })
    }

    // ── Non-unitary instructions ─────────────────────────────────────────

    /// Barrier across the whole quantum register (`barrier q;`).
    pub fn barrier(self) -> Self {
        self.push(GateInstruction::Barrier(Vec::new()))
    }

    /// Barrier on a subset of qubits (`barrier q[i],q[j],…;`).
    pub fn barrier_on(self, qubits: &[usize]) -> Self {
        self.push(GateInstruction::Barrier(qubits.to_vec()))
    }

    /// Measure `qubit` into classical bit `cbit`. Classical bits are allocated
    /// implicitly: the classical register is sized to the largest index used.
    pub fn measure(self, qubit: usize, cbit: usize) -> Self {
        self.push(GateInstruction::Measure { qubit, cbit })
    }

    /// Measure every qubit `i` into classical bit `i` (`measure q -> c;`).
    pub fn measure_all(self) -> Self {
        self.push(GateInstruction::MeasureAll)
    }

    // ── Introspection ────────────────────────────────────────────────────

    /// Size of the implicit classical register: `num_qubits` if the circuit
    /// contains a `MeasureAll`, otherwise `max(cbit) + 1` over all `Measure`
    /// instructions (0 when nothing is measured).
    pub fn num_clbits(&self) -> usize {
        num_clbits(self.num_qubits, &self.gates)
    }

    // ── Binding and export ───────────────────────────────────────────────

    /// Bind concrete values to the circuit's free parameters, producing a
    /// [`ConcreteCircuit`] in which every [`GateParam`] is `Fixed`.
    ///
    /// # Errors
    ///
    /// - [`CircuitError::WrongNumberOfParams`] if `params.len() != self.num_params`.
    /// - [`CircuitError::ParamIndexOutOfBounds`] if a gate references an index
    ///   `>= params.len()` (only possible for manually assembled circuits).
    pub fn assign_parameters(&self, params: &[f64]) -> Result<ConcreteCircuit, CircuitError> {
        if params.len() != self.num_params {
            return Err(CircuitError::WrongNumberOfParams {
                expected: self.num_params,
                got: params.len(),
            });
        }

        let resolve = |p: &GateParam| -> Result<GateParam, CircuitError> {
            Ok(GateParam::Fixed(p.resolve(params)?))
        };

        let mut gates = Vec::with_capacity(self.gates.len());
        for gate in &self.gates {
            let bound = match gate {
                GateInstruction::Rx { qubit, theta } => GateInstruction::Rx {
                    qubit: *qubit,
                    theta: resolve(theta)?,
                },
                GateInstruction::Ry { qubit, theta } => GateInstruction::Ry {
                    qubit: *qubit,
                    theta: resolve(theta)?,
                },
                GateInstruction::Rz { qubit, theta } => GateInstruction::Rz {
                    qubit: *qubit,
                    theta: resolve(theta)?,
                },
                GateInstruction::Rzz { q0, q1, theta } => GateInstruction::Rzz {
                    q0: *q0,
                    q1: *q1,
                    theta: resolve(theta)?,
                },
                GateInstruction::Rxx { q0, q1, theta } => GateInstruction::Rxx {
                    q0: *q0,
                    q1: *q1,
                    theta: resolve(theta)?,
                },
                GateInstruction::U {
                    qubit,
                    theta,
                    phi,
                    lam,
                } => GateInstruction::U {
                    qubit: *qubit,
                    theta: resolve(theta)?,
                    phi: resolve(phi)?,
                    lam: resolve(lam)?,
                },
                other => other.clone(),
            };
            gates.push(bound);
        }

        Ok(ConcreteCircuit {
            num_qubits: self.num_qubits,
            gates,
        })
    }

    /// Bind `params` and serialize to OpenQASM 2.0 in one step.
    /// Equivalent to `self.assign_parameters(params)?.to_qasm2()`.
    pub fn to_qasm2_with_params(&self, params: &[f64]) -> Result<String, CircuitError> {
        if params.len() != self.num_params {
            return Err(CircuitError::WrongNumberOfParams {
                expected: self.num_params,
                got: params.len(),
            });
        }
        qasm::write_qasm2(self.num_qubits, self.num_clbits(), &self.gates, params)
    }
}

/// A quantum circuit whose angles are all concrete values
/// ([`GateParam::Fixed`]). Produced by
/// [`ParameterizedCircuit::assign_parameters`]; ready for OpenQASM 2.0 export.
#[derive(Debug, Clone, PartialEq)]
pub struct ConcreteCircuit {
    /// Number of qubits in the (single) quantum register.
    pub num_qubits: usize,
    /// The instruction sequence; every [`GateParam`] is `Fixed`.
    pub gates: Vec<GateInstruction>,
}

impl ConcreteCircuit {
    /// Size of the implicit classical register (see
    /// [`ParameterizedCircuit::num_clbits`]).
    pub fn num_clbits(&self) -> usize {
        num_clbits(self.num_qubits, &self.gates)
    }

    /// Serialize to OpenQASM 2.0.
    ///
    /// # Panics
    ///
    /// Panics if a gate contains an unbound [`GateParam::Param`]. This cannot
    /// happen for circuits produced by
    /// [`ParameterizedCircuit::assign_parameters`]; it is only possible when
    /// the `gates` field was assembled manually.
    pub fn to_qasm2(&self) -> String {
        qasm::write_qasm2(self.num_qubits, self.num_clbits(), &self.gates, &[]).expect(
            "ConcreteCircuit contains an unbound Param; use ParameterizedCircuit::assign_parameters",
        )
    }
}

/// Shared classical-register sizing logic.
fn num_clbits(num_qubits: usize, gates: &[GateInstruction]) -> usize {
    let mut n = 0;
    for gate in gates {
        if matches!(gate, GateInstruction::MeasureAll) {
            n = n.max(num_qubits);
        } else if let Some(cbit) = gate.max_cbit() {
            n = n.max(cbit + 1);
        }
    }
    n
}
