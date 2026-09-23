//! Core gate data types: [`GateParam`] and [`GateInstruction`].

use crate::custom_gate::CustomGate;
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
    /// Identity (`id`). It has no effect on the state, but it is an instruction
    /// of the circuit: kept (never dropped) so gate counts and depth match the
    /// source program and what Qiskit computes for it. As a unitary, it is
    /// subject to the terminal-measurement rule (contract C-4) like any gate.
    Id(usize),
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
    /// SWAP: exchange the states of two qubits.
    Swap(usize, usize),
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
    /// √X (`sx`).
    Sx(usize),
    /// Conjugate transpose of √X (`sxdg`).
    Sxdg(usize),
    /// Controlled-Y (`cy`): control, target.
    Cy(usize, usize),
    /// Controlled-Hadamard (`ch`): control, target.
    Ch(usize, usize),
    /// Controlled-√X (`csx`): control, target.
    Csx(usize, usize),
    /// Toffoli (`ccx`): control, control, target.
    Ccx(usize, usize, usize),
    /// Fredkin (`cswap`): control, target, target.
    Cswap(usize, usize, usize),
    /// Controlled X-rotation (`crx`).
    Crx {
        control: usize,
        target: usize,
        theta: GateParam,
    },
    /// Controlled Y-rotation (`cry`).
    Cry {
        control: usize,
        target: usize,
        theta: GateParam,
    },
    /// Controlled Z-rotation (`crz`).
    Crz {
        control: usize,
        target: usize,
        theta: GateParam,
    },
    /// Controlled phase in its `cu1` spelling: the same operator as
    /// [`Cp`](GateInstruction::Cp), kept as its own variant so that a `cu1`
    /// statement is re-emitted as `cu1`, never rewritten to `cp`.
    Cu1 {
        q0: usize,
        q1: usize,
        theta: GateParam,
    },
    /// Controlled `u3(theta, phi, lambda)` (`cu3`).
    Cu3 {
        control: usize,
        target: usize,
        theta: GateParam,
        phi: GateParam,
        lam: GateParam,
    },
    /// Controlled `u(theta, phi, lambda)` with an extra phase `gamma` on the
    /// controlled branch (`cu`, Qiskit's `CUGate`): applies
    /// `e^{i·gamma} · U(theta, phi, lambda)` to the target when the control is 1.
    Cu {
        control: usize,
        target: usize,
        theta: GateParam,
        phi: GateParam,
        lam: GateParam,
        gamma: GateParam,
    },
    /// `u0(gamma)`: the identity, "idle for `gamma` units" in `qelib1.inc`.
    /// Kept (like [`Id`](GateInstruction::Id)) so gate counts and depth match.
    U0 { qubit: usize, gamma: GateParam },
    /// Phase gate `p(lambda)` = diag(1, e^{iλ}). The same operator as
    /// `u1(λ)` and `u3(0,0,λ)`, kept under its own spelling.
    P { qubit: usize, lam: GateParam },
    /// `u1(lambda)`: the phase gate in its `u1` spelling (see [`P`](GateInstruction::P)).
    U1 { qubit: usize, lam: GateParam },
    /// `u2(phi, lambda)` = `u3(π/2, φ, λ)`, kept under its own spelling.
    U2 {
        qubit: usize,
        phi: GateParam,
        lam: GateParam,
    },
    /// `u(theta, phi, lambda)` (Qiskit's `UGate`): the same operator as
    /// [`U`](GateInstruction::U), which is spelled `u3`; kept as its own
    /// variant so a `u` statement is re-emitted as `u`.
    UGate {
        qubit: usize,
        theta: GateParam,
        phi: GateParam,
        lam: GateParam,
    },
    /// Simplified Toffoli (`rccx`): a Toffoli up to relative phases, with
    /// controls `a`, `b` and target `c`.
    Rccx(usize, usize, usize),
    /// Simplified 3-controlled Toffoli (`rc3x`), up to relative phases: three
    /// controls, then the target.
    Rc3x(usize, usize, usize, usize),
    /// 3-controlled X (`c3x`): three controls, then the target.
    C3x(usize, usize, usize, usize),
    /// 3-controlled √X (`c3sqrtx`): three controls, then the target.
    C3sqrtx(usize, usize, usize, usize),
    /// 4-controlled X (`c4x`): four controls, then the target.
    C4x(usize, usize, usize, usize, usize),
    /// A call of a gate declared in the source program (an OpenQASM 2.0
    /// `gate` block), on qubits of any number. One instruction, like any other
    /// gate: it is re-emitted as the declaration plus the call, never as its
    /// expanded body (see [`CustomGate`]).
    Custom(CustomGate),
    /// Barrier. An empty vector means "all qubits" (`barrier q;`).
    Barrier(Vec<usize>),
    /// Measure one qubit into one classical bit.
    Measure { qubit: usize, cbit: usize },
    /// Measure every qubit `i` into classical bit `i` (`measure q -> c;`).
    MeasureAll,
}

/// The most qubits a built-in instruction acts on. Sized for the four- and
/// five-qubit multi-controlled gates of `qelib1.inc` (`c3x`, `c4x`, …) so that
/// adding them never forces [`Operands`] to change shape.
pub(crate) const MAX_GATE_ARITY: usize = 5;

/// The most angle parameters a built-in instruction takes (`cu`).
const MAX_GATE_PARAMS: usize = 4;

/// The first qubit that appears more than once in `qubits` (reported at its
/// second occurrence), or `None` when all are distinct. A unitary may never
/// name the same qubit twice, whatever its arity.
pub(crate) fn first_repeated_qubit(qubits: &[usize]) -> Option<usize> {
    qubits
        .iter()
        .enumerate()
        .find(|&(i, q)| qubits[..i].contains(q))
        .map(|(_, &q)| q)
}

/// The qubit operands of a unitary instruction, in operand order (control(s)
/// before target(s) for controlled gates).
///
/// A built-in gate's operands are stored inline rather than in a `Vec`:
/// [`GateInstruction::acts_on`] runs on every builder push and on every gate
/// the simulator applies, so building an operand list must not allocate. A
/// call of a declared gate, which may act on any number of qubits, lends its
/// own operand list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operands<'a> {
    Inline {
        qubits: [usize; MAX_GATE_ARITY],
        len: usize,
    },
    Borrowed(&'a [usize]),
}

impl Operands<'_> {
    fn new(qubits: &[usize]) -> Self {
        let mut buf = [0; MAX_GATE_ARITY];
        buf[..qubits.len()].copy_from_slice(qubits);
        Operands::Inline {
            qubits: buf,
            len: qubits.len(),
        }
    }

    /// The operands, in operand order.
    pub(crate) fn as_slice(&self) -> &[usize] {
        match self {
            Operands::Inline { qubits, len } => &qubits[..*len],
            Operands::Borrowed(qubits) => qubits,
        }
    }
}

/// The qubits an instruction evolves *unitarily*.
///
/// Used by the terminal-measurement check (contract C-4): only unitary
/// operations are forbidden on an already-measured qubit. `Barrier`,
/// `Measure` and `MeasureAll` do not evolve the state and therefore report
/// [`ActsOn::None`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActsOn<'a> {
    /// A non-unitary instruction (barrier / measure / measure_all).
    None,
    /// A unitary on these qubits, of any arity.
    Unitary(Operands<'a>),
}

impl ActsOn<'_> {
    /// The unitary operands, in operand order; empty for [`ActsOn::None`].
    pub(crate) fn qubits(&self) -> &[usize] {
        match self {
            ActsOn::None => &[],
            ActsOn::Unitary(operands) => operands.as_slice(),
        }
    }
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

    /// The angle parameters of this instruction, in QASM argument order
    /// (`theta, phi, lambda` for `u3`). Empty for parameter-free instructions.
    ///
    /// Exhaustive over the vocabulary (no wildcard arm): a new parameterised
    /// gate cannot compile until it says what its parameters are, so builder
    /// validation and `num_params` tracking can never silently skip it.
    pub(crate) fn params(&self) -> impl Iterator<Item = &GateParam> + '_ {
        use GateInstruction as G;
        let params: [Option<&GateParam>; MAX_GATE_PARAMS] = match self {
            G::Rx { theta, .. }
            | G::Ry { theta, .. }
            | G::Rz { theta, .. }
            | G::Rzz { theta, .. }
            | G::Rxx { theta, .. }
            | G::Cp { theta, .. }
            | G::Crx { theta, .. }
            | G::Cry { theta, .. }
            | G::Crz { theta, .. }
            | G::Cu1 { theta, .. } => [Some(theta), None, None, None],
            G::U0 { gamma, .. } => [Some(gamma), None, None, None],
            G::P { lam, .. } | G::U1 { lam, .. } => [Some(lam), None, None, None],
            G::U2 { phi, lam, .. } => [Some(phi), Some(lam), None, None],
            G::U {
                theta, phi, lam, ..
            }
            | G::UGate {
                theta, phi, lam, ..
            }
            | G::Cu3 {
                theta, phi, lam, ..
            } => [Some(theta), Some(phi), Some(lam), None],
            G::Cu {
                theta,
                phi,
                lam,
                gamma,
                ..
            } => [Some(theta), Some(phi), Some(lam), Some(gamma)],
            G::H(_)
            | G::X(_)
            | G::Y(_)
            | G::Z(_)
            | G::S(_)
            | G::T(_)
            | G::Sdg(_)
            | G::Tdg(_)
            | G::Id(_)
            | G::Sx(_)
            | G::Sxdg(_)
            | G::Cx(..)
            | G::Cz(..)
            | G::Swap(..)
            | G::Cy(..)
            | G::Ch(..)
            | G::Csx(..)
            | G::Ccx(..)
            | G::Cswap(..)
            | G::Rccx(..)
            | G::Rc3x(..)
            | G::C3x(..)
            | G::C3sqrtx(..)
            | G::C4x(..)
            | G::Custom(_)
            | G::Barrier(_)
            | G::Measure { .. }
            | G::MeasureAll => [None; MAX_GATE_PARAMS],
        };
        // A call of a declared gate takes any number of arguments.
        let call_params: &[GateParam] = match self {
            G::Custom(call) => call.params(),
            _ => &[],
        };
        params.into_iter().flatten().chain(call_params)
    }

    /// A copy of this instruction with every angle parameter replaced by
    /// `f(parameter)`, visited in QASM argument order; the first error aborts.
    /// Used to bind free parameters
    /// ([`ParameterizedCircuit::assign_parameters`](crate::ParameterizedCircuit::assign_parameters)).
    ///
    /// Exhaustive over the vocabulary, for the same reason as [`Self::params`].
    pub(crate) fn try_map_params<E>(
        &self,
        mut f: impl FnMut(&GateParam) -> Result<GateParam, E>,
    ) -> Result<GateInstruction, E> {
        use GateInstruction as G;
        Ok(match self {
            G::Rx { qubit, theta } => G::Rx {
                qubit: *qubit,
                theta: f(theta)?,
            },
            G::Ry { qubit, theta } => G::Ry {
                qubit: *qubit,
                theta: f(theta)?,
            },
            G::Rz { qubit, theta } => G::Rz {
                qubit: *qubit,
                theta: f(theta)?,
            },
            G::Rzz { q0, q1, theta } => G::Rzz {
                q0: *q0,
                q1: *q1,
                theta: f(theta)?,
            },
            G::Rxx { q0, q1, theta } => G::Rxx {
                q0: *q0,
                q1: *q1,
                theta: f(theta)?,
            },
            G::Cp { q0, q1, theta } => G::Cp {
                q0: *q0,
                q1: *q1,
                theta: f(theta)?,
            },
            G::U {
                qubit,
                theta,
                phi,
                lam,
            } => G::U {
                qubit: *qubit,
                theta: f(theta)?,
                phi: f(phi)?,
                lam: f(lam)?,
            },
            G::Crx {
                control,
                target,
                theta,
            } => G::Crx {
                control: *control,
                target: *target,
                theta: f(theta)?,
            },
            G::Cry {
                control,
                target,
                theta,
            } => G::Cry {
                control: *control,
                target: *target,
                theta: f(theta)?,
            },
            G::Crz {
                control,
                target,
                theta,
            } => G::Crz {
                control: *control,
                target: *target,
                theta: f(theta)?,
            },
            G::Cu1 { q0, q1, theta } => G::Cu1 {
                q0: *q0,
                q1: *q1,
                theta: f(theta)?,
            },
            G::Cu3 {
                control,
                target,
                theta,
                phi,
                lam,
            } => G::Cu3 {
                control: *control,
                target: *target,
                theta: f(theta)?,
                phi: f(phi)?,
                lam: f(lam)?,
            },
            G::Cu {
                control,
                target,
                theta,
                phi,
                lam,
                gamma,
            } => G::Cu {
                control: *control,
                target: *target,
                theta: f(theta)?,
                phi: f(phi)?,
                lam: f(lam)?,
                gamma: f(gamma)?,
            },
            G::U0 { qubit, gamma } => G::U0 {
                qubit: *qubit,
                gamma: f(gamma)?,
            },
            G::P { qubit, lam } => G::P {
                qubit: *qubit,
                lam: f(lam)?,
            },
            G::U1 { qubit, lam } => G::U1 {
                qubit: *qubit,
                lam: f(lam)?,
            },
            G::U2 { qubit, phi, lam } => G::U2 {
                qubit: *qubit,
                phi: f(phi)?,
                lam: f(lam)?,
            },
            G::UGate {
                qubit,
                theta,
                phi,
                lam,
            } => G::UGate {
                qubit: *qubit,
                theta: f(theta)?,
                phi: f(phi)?,
                lam: f(lam)?,
            },
            G::Custom(call) => G::Custom(
                call.with_params(call.params().iter().map(&mut f).collect::<Result<_, E>>()?),
            ),
            G::H(_)
            | G::X(_)
            | G::Y(_)
            | G::Z(_)
            | G::S(_)
            | G::T(_)
            | G::Sdg(_)
            | G::Tdg(_)
            | G::Id(_)
            | G::Sx(_)
            | G::Sxdg(_)
            | G::Cx(..)
            | G::Cz(..)
            | G::Swap(..)
            | G::Cy(..)
            | G::Ch(..)
            | G::Csx(..)
            | G::Ccx(..)
            | G::Cswap(..)
            | G::Rccx(..)
            | G::Rc3x(..)
            | G::C3x(..)
            | G::C3sqrtx(..)
            | G::C4x(..)
            | G::Barrier(_)
            | G::Measure { .. }
            | G::MeasureAll => self.clone(),
        })
    }

    /// The exact decomposition of a composite `qelib1.inc` gate — `ccx`,
    /// `cswap`, `rccx`, `rc3x`, `c3x`, `c3sqrtx`, `c4x` — into simpler
    /// instructions: the gate's own definition in `qelib1.inc` (with `u1`/`u2`
    /// written as the equal `t`/`tdg`/`h` and `p` as the equal `u3(0,0,λ)`),
    /// exact including the global phase. `None` for every other instruction.
    ///
    /// This is for backends with no native implementation of these gates (the
    /// native simulator, the QIR exporter) to *lower* them. It is never applied
    /// to the circuit itself: import and export keep each gate as one
    /// instruction under its own name. The result may contain composite gates
    /// again (`c4x` uses `c3x` and `c3sqrtx`), each lowered in turn.
    pub fn lowering(&self) -> Option<Vec<GateInstruction>> {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_8};
        use GateInstruction as G;
        let phase = |qubit: usize, lam: f64| G::U {
            qubit,
            theta: GateParam::Fixed(0.0),
            phi: GateParam::Fixed(0.0),
            lam: GateParam::Fixed(lam),
        };
        let cu1 = |q0: usize, q1: usize, theta: f64| G::Cu1 {
            q0,
            q1,
            theta: GateParam::Fixed(theta),
        };
        Some(match *self {
            G::Ccx(a, b, c) => vec![
                G::H(c),
                G::Cx(b, c),
                G::Tdg(c),
                G::Cx(a, c),
                G::T(c),
                G::Cx(b, c),
                G::Tdg(c),
                G::Cx(a, c),
                G::T(b),
                G::T(c),
                G::H(c),
                G::Cx(a, b),
                G::T(a),
                G::Tdg(b),
                G::Cx(a, b),
            ],
            G::Cswap(a, b, c) => vec![G::Cx(c, b), G::Ccx(a, b, c), G::Cx(c, b)],
            G::Rccx(a, b, c) => vec![
                G::H(c),
                G::T(c),
                G::Cx(b, c),
                G::Tdg(c),
                G::Cx(a, c),
                G::T(c),
                G::Cx(b, c),
                G::Tdg(c),
                G::H(c),
            ],
            G::Rc3x(a, b, c, d) => vec![
                G::H(d),
                G::T(d),
                G::Cx(c, d),
                G::Tdg(d),
                G::H(d),
                G::Cx(a, d),
                G::T(d),
                G::Cx(b, d),
                G::Tdg(d),
                G::Cx(a, d),
                G::T(d),
                G::Cx(b, d),
                G::Tdg(d),
                G::H(d),
                G::T(d),
                G::Cx(c, d),
                G::Tdg(d),
                G::H(d),
            ],
            G::C3x(a, b, c, d) => {
                let p = FRAC_PI_8;
                vec![
                    G::H(d),
                    phase(a, p),
                    phase(b, p),
                    phase(c, p),
                    phase(d, p),
                    G::Cx(a, b),
                    phase(b, -p),
                    G::Cx(a, b),
                    G::Cx(b, c),
                    phase(c, -p),
                    G::Cx(a, c),
                    phase(c, p),
                    G::Cx(b, c),
                    phase(c, -p),
                    G::Cx(a, c),
                    G::Cx(c, d),
                    phase(d, -p),
                    G::Cx(b, d),
                    phase(d, p),
                    G::Cx(c, d),
                    phase(d, -p),
                    G::Cx(a, d),
                    phase(d, p),
                    G::Cx(c, d),
                    phase(d, -p),
                    G::Cx(b, d),
                    phase(d, p),
                    G::Cx(c, d),
                    phase(d, -p),
                    G::Cx(a, d),
                    G::H(d),
                ]
            }
            G::C3sqrtx(a, b, c, d) => {
                let p = FRAC_PI_8;
                let mut ops = Vec::with_capacity(27);
                // h d; cu1(±π/8) ctrl,d; h d — interleaved with the cx ladder.
                let rotation = |ops: &mut Vec<G>, ctrl: usize, theta: f64| {
                    ops.extend([G::H(d), cu1(ctrl, d, theta), G::H(d)]);
                };
                rotation(&mut ops, a, p);
                ops.push(G::Cx(a, b));
                rotation(&mut ops, b, -p);
                ops.push(G::Cx(a, b));
                rotation(&mut ops, b, p);
                ops.push(G::Cx(b, c));
                rotation(&mut ops, c, -p);
                ops.push(G::Cx(a, c));
                rotation(&mut ops, c, p);
                ops.push(G::Cx(b, c));
                rotation(&mut ops, c, -p);
                ops.push(G::Cx(a, c));
                rotation(&mut ops, c, p);
                ops
            }
            G::C4x(a, b, c, d, e) => vec![
                G::H(e),
                cu1(d, e, FRAC_PI_2),
                G::H(e),
                G::C3x(a, b, c, d),
                G::H(e),
                cu1(d, e, -FRAC_PI_2),
                G::H(e),
                G::C3x(a, b, c, d),
                G::C3sqrtx(a, b, c, e),
            ],
            _ => return None,
        })
    }

    /// Which qubits this instruction acts on *as a unitary* (see [`ActsOn`]).
    pub(crate) fn acts_on(&self) -> ActsOn<'_> {
        match self {
            // A declared gate is a unitary on all its qubits, whatever its body.
            GateInstruction::Custom(call) => ActsOn::Unitary(Operands::Borrowed(call.qubits())),
            GateInstruction::H(q)
            | GateInstruction::X(q)
            | GateInstruction::Y(q)
            | GateInstruction::Z(q)
            | GateInstruction::S(q)
            | GateInstruction::T(q)
            | GateInstruction::Sdg(q)
            | GateInstruction::Tdg(q)
            | GateInstruction::Id(q)
            | GateInstruction::Rx { qubit: q, .. }
            | GateInstruction::Ry { qubit: q, .. }
            | GateInstruction::Rz { qubit: q, .. }
            | GateInstruction::U { qubit: q, .. }
            | GateInstruction::Sx(q)
            | GateInstruction::Sxdg(q)
            | GateInstruction::U0 { qubit: q, .. }
            | GateInstruction::P { qubit: q, .. }
            | GateInstruction::U1 { qubit: q, .. }
            | GateInstruction::U2 { qubit: q, .. }
            | GateInstruction::UGate { qubit: q, .. } => ActsOn::Unitary(Operands::new(&[*q])),
            GateInstruction::Rccx(a, b, c) => ActsOn::Unitary(Operands::new(&[*a, *b, *c])),
            GateInstruction::Rc3x(a, b, c, d)
            | GateInstruction::C3x(a, b, c, d)
            | GateInstruction::C3sqrtx(a, b, c, d) => {
                ActsOn::Unitary(Operands::new(&[*a, *b, *c, *d]))
            }
            GateInstruction::C4x(a, b, c, d, e) => {
                ActsOn::Unitary(Operands::new(&[*a, *b, *c, *d, *e]))
            }
            GateInstruction::Cx(a, b)
            | GateInstruction::Cz(a, b)
            | GateInstruction::Swap(a, b)
            | GateInstruction::Rzz { q0: a, q1: b, .. }
            | GateInstruction::Rxx { q0: a, q1: b, .. }
            | GateInstruction::Cp { q0: a, q1: b, .. }
            | GateInstruction::Cy(a, b)
            | GateInstruction::Ch(a, b)
            | GateInstruction::Csx(a, b)
            | GateInstruction::Crx {
                control: a,
                target: b,
                ..
            }
            | GateInstruction::Cry {
                control: a,
                target: b,
                ..
            }
            | GateInstruction::Crz {
                control: a,
                target: b,
                ..
            }
            | GateInstruction::Cu1 { q0: a, q1: b, .. }
            | GateInstruction::Cu3 {
                control: a,
                target: b,
                ..
            }
            | GateInstruction::Cu {
                control: a,
                target: b,
                ..
            } => ActsOn::Unitary(Operands::new(&[*a, *b])),
            GateInstruction::Ccx(a, b, c) | GateInstruction::Cswap(a, b, c) => {
                ActsOn::Unitary(Operands::new(&[*a, *b, *c]))
            }
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
        // The first measured operand in operand order, for any arity.
        let offending = gate
            .acts_on()
            .qubits()
            .iter()
            .copied()
            .find(|q| measure_all || measured.contains(q));
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

/// The first qubit index in `gates` that is out of range for a register of
/// `num_qubits` qubits (i.e. `>= num_qubits`), if any.
///
/// Every qubit *reference* is checked — unitary operands, [`Measure`] targets
/// and [`Barrier`] operands. ([`MeasureAll`] spans `0..num_qubits` by definition
/// and is always in range; classical-bit indices are a separate concern.) The
/// fluent builder already rejects an out-of-range qubit at push time
/// ([`CircuitError::QubitOutOfRange`](crate::CircuitError::QubitOutOfRange)), but
/// a [`ConcreteCircuit`](crate::ConcreteCircuit) can be assembled or mutated
/// through its public `gates` field, bypassing that. This is the shared
/// reference — the sibling of [`terminal_measurement_violation`] — that the
/// builder-free consumers (notably the native simulator) use to reject such a
/// circuit *before it reaches a kernel*, where an out-of-range index maps to an
/// out-of-bounds amplitude (`1 << qubit`) and is undefined behaviour in release.
///
/// [`Measure`]: GateInstruction::Measure
/// [`Barrier`]: GateInstruction::Barrier
/// [`MeasureAll`]: GateInstruction::MeasureAll
pub fn qubit_index_violation(gates: &[GateInstruction], num_qubits: usize) -> Option<usize> {
    for gate in gates {
        let offending = match gate.acts_on() {
            // The first out-of-range operand in operand order, for any arity.
            ActsOn::Unitary(operands) => operands
                .as_slice()
                .iter()
                .copied()
                .find(|&q| q >= num_qubits),
            ActsOn::None => match gate {
                GateInstruction::Measure { qubit, .. } => (*qubit >= num_qubits).then_some(*qubit),
                GateInstruction::Barrier(qubits) => {
                    qubits.iter().copied().find(|&q| q >= num_qubits)
                }
                _ => None,
            },
        };
        if offending.is_some() {
            return offending;
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

    #[test]
    fn qubit_index_violation_flags_every_kind_of_reference() {
        // Unitary operands (one- and two-qubit), a measurement target and a
        // barrier operand are all checked; the offending index is returned.
        assert_eq!(qubit_index_violation(&[GateInstruction::H(5)], 2), Some(5));
        assert_eq!(
            qubit_index_violation(&[GateInstruction::Cx(0, 9)], 2),
            Some(9)
        );
        assert_eq!(
            qubit_index_violation(&[GateInstruction::Measure { qubit: 4, cbit: 0 }], 2),
            Some(4)
        );
        assert_eq!(
            qubit_index_violation(&[GateInstruction::Barrier(vec![0, 7])], 2),
            Some(7)
        );
    }

    // ── Per-gate parameter access ─────────────────────────────────────────

    #[test]
    fn params_are_listed_in_qasm_argument_order() {
        let u = GateInstruction::U {
            qubit: 0,
            theta: GateParam::Fixed(0.1),
            phi: GateParam::Param(2),
            lam: GateParam::Fixed(0.3),
        };
        let listed: Vec<GateParam> = u.params().copied().collect();
        assert_eq!(
            listed,
            [
                GateParam::Fixed(0.1),
                GateParam::Param(2),
                GateParam::Fixed(0.3)
            ]
        );
        let cp = GateInstruction::Cp {
            q0: 0,
            q1: 1,
            theta: GateParam::Param(0),
        };
        assert_eq!(
            cp.params().copied().collect::<Vec<_>>(),
            [GateParam::Param(0)]
        );
        for gate in [
            GateInstruction::H(0),
            GateInstruction::Cx(0, 1),
            GateInstruction::Barrier(vec![]),
            GateInstruction::Measure { qubit: 0, cbit: 0 },
            GateInstruction::MeasureAll,
        ] {
            assert_eq!(gate.params().count(), 0, "{gate:?}");
        }
    }

    #[test]
    fn try_map_params_rewrites_every_parameter_in_order() {
        let u = GateInstruction::U {
            qubit: 2,
            theta: GateParam::Param(0),
            phi: GateParam::Param(1),
            lam: GateParam::Fixed(0.5),
        };
        let mut seen = Vec::new();
        let mapped = u
            .try_map_params(|p| -> Result<GateParam, CircuitError> {
                seen.push(*p);
                Ok(GateParam::Fixed(p.resolve(&[10.0, 20.0])?))
            })
            .unwrap();
        assert_eq!(
            seen,
            [
                GateParam::Param(0),
                GateParam::Param(1),
                GateParam::Fixed(0.5)
            ]
        );
        assert_eq!(
            mapped,
            GateInstruction::U {
                qubit: 2,
                theta: GateParam::Fixed(10.0),
                phi: GateParam::Fixed(20.0),
                lam: GateParam::Fixed(0.5),
            }
        );
        // Parameter-free instructions come back unchanged.
        let barrier = GateInstruction::Barrier(vec![0, 1]);
        assert_eq!(
            barrier.try_map_params(|_| Err::<GateParam, ()>(())),
            Ok(barrier.clone())
        );
    }

    #[test]
    fn try_map_params_stops_at_the_first_error() {
        let rzz = GateInstruction::Rzz {
            q0: 0,
            q1: 1,
            theta: GateParam::Param(3),
        };
        assert_eq!(
            rzz.try_map_params(|p| p.resolve(&[]).map(GateParam::Fixed)),
            Err(CircuitError::ParamIndexOutOfBounds {
                index: 3,
                num_params: 0
            })
        );
    }

    #[test]
    fn first_repeated_qubit_reports_the_second_occurrence() {
        assert_eq!(first_repeated_qubit(&[]), None);
        assert_eq!(first_repeated_qubit(&[0, 1, 2]), None);
        assert_eq!(first_repeated_qubit(&[1, 1]), Some(1));
        assert_eq!(first_repeated_qubit(&[2, 0, 1, 0, 2]), Some(0));
    }

    // ── ActsOn / Operands (arity-generic operand lists) ──────────────────

    #[test]
    fn operands_keep_operand_order_for_every_arity() {
        for qubits in [
            &[][..],
            &[4][..],
            &[3, 1][..],
            &[2, 0, 1][..],
            &[4, 3, 2, 1, 0][..],
        ] {
            assert_eq!(Operands::new(qubits).as_slice(), qubits);
        }
    }

    #[test]
    fn acts_on_reports_unitary_operands_in_operand_order() {
        assert_eq!(GateInstruction::H(3).acts_on().qubits(), &[3]);
        // Control before target: the order the C-4 and index checks report in.
        assert_eq!(GateInstruction::Cx(2, 0).acts_on().qubits(), &[2, 0]);
        let rzz = GateInstruction::Rzz {
            q0: 1,
            q1: 4,
            theta: GateParam::Fixed(0.1),
        };
        assert_eq!(rzz.acts_on().qubits(), &[1, 4]);
        // Non-unitary instructions have no unitary operands.
        for gate in [
            GateInstruction::Barrier(vec![0, 1]),
            GateInstruction::Measure { qubit: 0, cbit: 0 },
            GateInstruction::MeasureAll,
        ] {
            assert_eq!(gate.acts_on(), ActsOn::None);
            assert!(gate.acts_on().qubits().is_empty());
        }
    }

    #[test]
    fn terminal_measurement_violation_reports_first_measured_operand() {
        // Only the target is measured: the target is reported.
        let gates = [
            GateInstruction::Measure { qubit: 1, cbit: 0 },
            GateInstruction::Cx(0, 1),
        ];
        assert_eq!(terminal_measurement_violation(&gates), Some(1));
        // Both measured: the first operand in operand order wins.
        let gates = [
            GateInstruction::Measure { qubit: 0, cbit: 0 },
            GateInstruction::Measure { qubit: 1, cbit: 1 },
            GateInstruction::Cx(1, 0),
        ];
        assert_eq!(terminal_measurement_violation(&gates), Some(1));
        // After MeasureAll every operand is measured.
        let gates = [GateInstruction::MeasureAll, GateInstruction::Cz(2, 3)];
        assert_eq!(terminal_measurement_violation(&gates), Some(2));
    }

    #[test]
    fn qubit_index_violation_reports_first_out_of_range_operand() {
        assert_eq!(
            qubit_index_violation(&[GateInstruction::Cx(7, 9)], 2),
            Some(7)
        );
        assert_eq!(
            qubit_index_violation(&[GateInstruction::Cx(1, 9)], 2),
            Some(9)
        );
    }

    #[test]
    fn qubit_index_violation_accepts_in_range_circuits() {
        // In-range references (including `MeasureAll`, which spans 0..num_qubits
        // by definition) are clean; the boundary index `num_qubits` is not.
        let gates = vec![
            GateInstruction::H(0),
            GateInstruction::Cx(0, 1),
            GateInstruction::MeasureAll,
        ];
        assert_eq!(qubit_index_violation(&gates, 2), None);
        assert_eq!(qubit_index_violation(&[GateInstruction::X(2)], 2), Some(2));
        assert_eq!(qubit_index_violation(&[], 0), None);
    }
}
