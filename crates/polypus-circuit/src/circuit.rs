//! Circuit types: [`ParameterizedCircuit`] (free parameters) and
//! [`ConcreteCircuit`] (all angles bound).

use crate::custom_gate::GateDefinition;
use crate::error::CircuitError;
use crate::expr::{evaluate_constant, Classified, ExprArena, ParamExpr};
use crate::gate::{
    first_repeated_qubit, qubit_index_violation, GateInstruction, GateParam, MeasuredQubits,
};
use crate::qasm;
use crate::qasm_import;
use crate::qir;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::convert::Infallible;

/// A quantum circuit that may contain free parameters ([`GateParam::Param`])
/// and angle expressions of them ([`GateParam::Expr`], see
/// [`add_expr`](Self::add_expr)).
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
#[derive(Debug, Clone)]
pub struct ParameterizedCircuit {
    /// Number of qubits in the (single) quantum register.
    pub num_qubits: usize,
    /// Number of free parameters. Kept in sync automatically by the builder:
    /// adding a gate with `Param(i)` grows it to at least `i + 1`.
    pub num_params: usize,
    /// The instruction sequence, in execution order.
    pub gates: Vec<GateInstruction>,
    /// Push-time cache backing the C-4 check in [`try_push`](Self::try_push).
    /// Derived from `gates`, never part of the circuit's identity; assigning
    /// `gates` directly leaves it in its "not derived yet" state, from which the
    /// next push rebuilds it.
    pub(crate) measured: MeasuredQubits,
    /// The expressions [`GateParam::Expr`] angles refer to.
    pub(crate) exprs: ExprArena,
    /// The names given to free parameters `0..param_names.len()` (by an
    /// importer). Every other parameter has a default name; see
    /// [`param_names`](Self::param_names).
    pub(crate) param_names: Vec<String>,
    /// The total size of the classical registers the OpenQASM program this
    /// circuit was imported from declares (the sum of its `creg`s, or of its
    /// `bit` registers in OpenQASM 3); `None` when it declares none or the
    /// circuit was built, not imported. Import
    /// metadata, never part of the circuit's identity; read through the hidden
    /// `declared_clbits` getter.
    pub(crate) declared_clbits: Option<usize>,
}

/// Structural equality over the circuit itself. The `measured` cache is derived
/// from `gates`, so two circuits that differ only in whether that cache has been
/// materialised yet are the same circuit. Expressions compare by content, not
/// by where they are stored: two circuits whose gates use equal expressions are
/// equal even if one of them also stores an expression no gate uses. The
/// parameters' names ([`ParameterizedCircuit::param_names`]) are part of the
/// circuit. The classical width an OpenQASM import declares is not: like the
/// `measured` cache it is not part of the circuit's identity (it is import
/// metadata), so two imports that differ only in the size of a `creg` are
/// equal, even though the native backend keys their counts at different widths
/// (contract C-3).
impl PartialEq for ParameterizedCircuit {
    fn eq(&self, other: &Self) -> bool {
        self.num_qubits == other.num_qubits
            && self.num_params == other.num_params
            && self.gates.len() == other.gates.len()
            && self
                .gates
                .iter()
                .zip(&other.gates)
                .all(|(a, b)| same_instruction(a, &self.exprs, b, &other.exprs))
            && self.names().eq(other.names())
    }
}

/// Whether `a`, with its expressions in `a_exprs`, and `b`, with its
/// expressions in `b_exprs`, are the same instruction: equal once their angles
/// are set aside, and with equal angles, where two expressions are equal if
/// their content is.
fn same_instruction(
    a: &GateInstruction,
    a_exprs: &ExprArena,
    b: &GateInstruction,
    b_exprs: &ExprArena,
) -> bool {
    let without_angles = |g: &GateInstruction| {
        g.try_map_params(|_| Ok::<_, Infallible>(GateParam::Fixed(0.0)))
            .unwrap_or_else(|never| match never {})
    };
    a.params().count() == b.params().count()
        && a.params().zip(b.params()).all(|(p, q)| match (p, q) {
            (GateParam::Expr(x), GateParam::Expr(y)) => {
                match (a_exprs.nodes(*x), b_exprs.nodes(*y)) {
                    (Some(n), Some(m)) => n == m,
                    (None, None) => x == y,
                    _ => false,
                }
            }
            _ => p == q,
        })
        && without_angles(a) == without_angles(b)
}

impl ParameterizedCircuit {
    /// Create an empty circuit over `num_qubits` qubits with no parameters.
    pub fn new(num_qubits: usize) -> Self {
        ParameterizedCircuit {
            num_qubits,
            num_params: 0,
            gates: Vec::new(),
            measured: MeasuredQubits::default(),
            exprs: ExprArena::default(),
            param_names: Vec::new(),
            declared_clbits: None,
        }
    }

    /// Import an OpenQASM 2.0 program (the inverse of
    /// [`to_qasm2_with_params`](Self::to_qasm2_with_params)).
    ///
    /// Supports the `qelib1.inc` vocabulary produced by Qiskit's `qasm2.dumps`
    /// (`u`, `p`, `u1`, `u2`, `sx`, `ccx`, `cu3`, `id`, …), `gate`
    /// declarations (each call becomes one [`GateInstruction::Custom`], never
    /// its expanded body), multiple `qreg`/`creg` declarations (flattened in
    /// declaration order), register broadcasting, and angle expressions such
    /// as `pi/2`.
    ///
    /// Since OpenQASM 2.0 has no free parameters, the result is always fully
    /// concrete (`num_params == 0`). Round-trip guarantee: for any circuit
    /// `qc` produced by this crate,
    /// `from_qasm2(&qc.to_qasm2_with_params(&p)?)?` exports byte-identical
    /// QASM again.
    ///
    /// Known model differences (semantics-preserving):
    /// - The language builtins `U` and `CX` are re-emitted as `u` and `cx`
    ///   (what Qiskit names them too). Every other instruction — `p`, `u1`,
    ///   `u2`, `u`, `u3`, `id` included — is kept one-to-one, as spelled.
    /// - Gate declarations are re-emitted right after the include, in source
    ///   order; a declaration no instruction uses is not re-emitted.
    /// - The classical register is implicit (sized by the measurements), so
    ///   trailing *unmeasured* classical bits are not re-exported. The total
    ///   the program declares is kept only as import metadata, for the native
    ///   backend to key counts at the declared width (contract C-3); it does
    ///   not count for `==` and is dropped by binding and by every export.
    ///
    /// # Errors
    ///
    /// [`CircuitError::Parse`] (with a 1-based line number) on malformed
    /// input, undeclared registers, out-of-range indices, unsupported
    /// statements (`opaque`, `if`, `reset`) or gates that are neither built in
    /// nor declared — each naming the construct.
    pub fn from_qasm2(source: &str) -> Result<Self, CircuitError> {
        qasm_import::parse_qasm2(source)
    }

    /// Import a program in the OpenQASM 3 profile (the inverse of
    /// [`to_qasm3`](Self::to_qasm3)).
    ///
    /// **OpenQASM 3 profile with Qiskit phase conventions.** The profile is
    /// the straight-line part of OpenQASM 3 that carries a parameterised,
    /// terminal-measurement circuit: `include "stdgates.inc";` (provided
    /// internally; no file is ever read), `qubit` and `bit` registers,
    /// `input float[64]` parameters, calls of `U`, of the `stdgates.inc`
    /// gates and of gates declared with `gate` blocks, `barrier`, and
    /// measurements assigned to bits. Angles are expressions of the inputs
    /// and the constants `pi`, `tau` and `euler` with `+ - * / **`, unary
    /// minus and `sin cos tan arcsin arccos arctan exp log sqrt`, evaluated
    /// in binary64 exactly as written. Everything else — control flow,
    /// `reset`, other classical types and computation, subroutines, gate
    /// modifiers, `gphase`, timing, pulses, arrays, physical qubits,
    /// annotations — is rejected with the construct and its line. So is a
    /// division of two integer expressions (`1/2`), which OpenQASM 3 types as
    /// integer division: write `1.0/2`.
    ///
    /// `U`, `u2` and `u3` are read with **Qiskit's** matrices (Polypus's `u`,
    /// `u2` and `u3`), which differ from the OpenQASM 3 specification's by the
    /// global phases `e^{-iθ/2}` (`U`) and `e^{i(φ+λ)/2}` (`u2`, `u3`).
    /// Amplitudes, as [`polypus_sim`](https://docs.rs/polypus-sim)'s
    /// statevector reports them, may therefore differ from those of a reader
    /// that follows the specification by these global factors; probabilities,
    /// counts and expectation values do not. This is sound only because the
    /// profile rejects gate modifiers and `gphase`, under which a global phase
    /// would become observable. Every other `stdgates.inc` gate follows the
    /// specification exactly. That Polypus reads Qiskit's `qasm3.dumps`
    /// output as Qiskit does is tested for the Qiskit versions and circuits
    /// the test suite lists, not guaranteed in general.
    ///
    /// The inputs are the circuit's free parameters, in declaration order
    /// (unused ones included), under their names
    /// ([`param_names`](Self::param_names)). The builtin `U` becomes `u`, `CX`
    /// becomes `cx`, `phase` becomes `p` and `cphase` becomes `cp`; every
    /// other gate keeps its name, and a declared gate stays a declared gate
    /// ([`GateInstruction::Custom`]) whatever its name. Registers are
    /// flattened in declaration order.
    ///
    /// # Errors
    ///
    /// [`CircuitError::Parse`], with a 1-based line number, for malformed
    /// input, a construct outside the profile, a gate that is neither built in
    /// nor declared, a gate after a measurement of one of its qubits (contract
    /// C-4), or an input beyond the importer's budgets (source size, token
    /// length, inputs, declarations, expression size and depth, register size,
    /// instructions, declared-gate nesting and expansion).
    pub fn from_qasm3(source: &str) -> Result<Self, CircuitError> {
        crate::qasm3_import::parse_qasm3(source)
    }

    // ── Expressions ──────────────────────────────────────────────────────

    /// Store the angle expression `expr` in this circuit and return the angle
    /// to pass to a gate.
    ///
    /// The angle is canonical: a bare free parameter comes back as
    /// [`GateParam::Param`]; an expression without free parameters is
    /// evaluated now and comes back as [`GateParam::Fixed`]; anything else is
    /// stored and comes back as a [`GateParam::Expr`], which only this circuit
    /// (and its clones) can resolve. Like a `Param`, the free parameters an
    /// expression refers to count towards [`num_params`](Self::num_params)
    /// once a gate that uses it is pushed. Binding evaluates the expression
    /// exactly as it is written, left to right and without reassociation.
    ///
    /// # Errors
    ///
    /// - [`CircuitError::NonFiniteParam`] if a number in `expr` is `NaN` or
    ///   infinite, or a constant expression evaluates to one.
    /// - [`CircuitError::DivisionByZero`] if a constant expression divides by
    ///   zero.
    /// - [`CircuitError::InvalidExpression`] if `expr` has more than 1 000 000
    ///   nodes, or nests more than 64 levels deep as OpenQASM writes it (one
    ///   level per parenthesised subexpression, unary minus, function argument
    ///   and exponent; a chain such as `a + b + c` does not nest). Beyond that
    ///   depth an exported expression could not be imported again.
    pub fn add_expr(&mut self, expr: ParamExpr) -> Result<GateParam, CircuitError> {
        Ok(match ExprArena::classify(&expr)? {
            Classified::Param(index) => GateParam::Param(index),
            Classified::Constant => GateParam::Fixed(evaluate_constant(&expr)?),
            Classified::Expr { max_input } => GateParam::Expr(self.exprs.insert(&expr, max_input)?),
        })
    }

    // ── Parameter names ──────────────────────────────────────────────────

    /// The name of each free parameter, in index order: exactly
    /// [`num_params`](Self::num_params) names.
    ///
    /// A parameter that was given a name keeps it (an importer names the
    /// parameters it reads). Every other parameter, including an index no gate
    /// uses, is named `theta_<index>` — or, if that name is already taken by a
    /// named parameter or by a declared gate the circuit calls (directly or
    /// through other declarations), the first free `theta_<index>_<k>` for
    /// `k = 1, 2, …`. The names are therefore a deterministic function of the
    /// circuit. They are part of its identity (`==`) and kept by `clone`.
    /// Names belong to the circuit, not to its gates: a gate pushed from
    /// another circuit refers to parameters by index and takes this circuit's
    /// names. Binding removes the parameters, and their names with them (a
    /// [`ConcreteCircuit`] has neither).
    ///
    /// ```
    /// use polypus_circuit::{Param, ParameterizedCircuit};
    ///
    /// let qc = ParameterizedCircuit::new(1).rx(0, Param(2));
    /// assert_eq!(qc.param_names(), ["theta_0", "theta_1", "theta_2"]);
    /// ```
    pub fn param_names(&self) -> Vec<String> {
        self.names().map(Cow::into_owned).collect()
    }

    /// [`Self::param_names`], one name at a time.
    fn names(&self) -> impl Iterator<Item = Cow<'_, str>> + '_ {
        let mut taken: BTreeSet<&str> = self.param_names.iter().map(String::as_str).collect();
        taken.extend(self.declared_gate_names());
        (0..self.num_params).map(move |index| match self.param_names.get(index) {
            Some(name) => Cow::Borrowed(name.as_str()),
            None => Cow::Owned(default_param_name(index, &taken)),
        })
    }

    /// The names of the declared gates the circuit calls, directly or through
    /// other declarations. Each definition is visited once.
    fn declared_gate_names(&self) -> BTreeSet<&str> {
        let mut names = BTreeSet::new();
        let mut visited: BTreeSet<*const GateDefinition> = BTreeSet::new();
        let mut pending: Vec<&GateDefinition> = self
            .gates
            .iter()
            .filter_map(|gate| match gate {
                GateInstruction::Custom(call) => Some(call.definition()),
                _ => None,
            })
            .collect();
        while let Some(definition) = pending.pop() {
            if visited.insert(definition) {
                names.insert(definition.name());
                pending.extend(definition.callees().map(|callee| &**callee));
            }
        }
        names
    }

    // ── Internal validation helpers ──────────────────────────────────────

    fn track_param(&mut self, param: &GateParam) {
        let highest = match param {
            GateParam::Fixed(_) => None,
            GateParam::Param(i) => Some(*i),
            GateParam::Expr(id) => self.exprs.max_input(*id),
        };
        if let Some(i) = highest {
            self.num_params = self.num_params.max(i + 1);
        }
    }

    /// Reject, at construction time, a `Fixed` angle that is not finite (`NaN`
    /// or infinity) and an `Expr` this circuit does not hold. `Param` angles
    /// are unchecked here — their values are only known at binding time, where
    /// [`GateParam::resolve`] enforces the same rule.
    fn check_param(&self, param: &GateParam) -> Result<(), CircuitError> {
        match param {
            GateParam::Fixed(v) if !v.is_finite() => Err(CircuitError::NonFiniteParam),
            GateParam::Expr(id) if self.exprs.nodes(*id).is_none() => {
                Err(CircuitError::UnknownExpression)
            }
            _ => Ok(()),
        }
    }

    /// Fallible version of [`push`](Self::push): append a raw
    /// [`GateInstruction`], validating qubit indices and keeping `num_params`
    /// in sync. Use this from host languages (e.g. Python bindings) where
    /// invalid input must surface as a recoverable error, not a panic.
    pub fn try_push(&mut self, gate: GateInstruction) -> Result<(), CircuitError> {
        // Terminal-measurement model (contract C-4): a unitary gate may not act
        // on a qubit that an earlier instruction already measured. The existing
        // prefix is already valid, so only the new gate can offend. Answered from
        // the incremental `measured` cache — rescanning `gates` here made building
        // a circuit quadratic in its gate count. Checked before any mutation
        // below; the cache itself is only advanced on the success path.
        self.measured.sync(&self.gates);
        // The first measured operand in operand order, for any arity.
        if let Some(&qubit) = gate
            .acts_on()
            .qubits()
            .iter()
            .find(|&&q| self.measured.contains(q))
        {
            return Err(CircuitError::QubitAlreadyMeasured { qubit });
        }
        // Every qubit reference in range: unitary operands, a measurement target
        // and barrier operands (`MeasureAll` spans the register by definition).
        if let Some(qubit) = qubit_index_violation(std::slice::from_ref(&gate), self.num_qubits) {
            return Err(CircuitError::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        // A unitary never names the same qubit twice (a barrier may).
        if let Some(qubit) = first_repeated_qubit(gate.acts_on().qubits()) {
            return Err(CircuitError::IdenticalQubits { qubit });
        }
        // Every angle is validated before any is tracked, so a rejected gate
        // leaves `num_params` untouched.
        for param in gate.params() {
            self.check_param(param)?;
        }
        for param in gate.params() {
            self.track_param(param);
        }
        self.measured.record(&gate);
        self.gates.push(gate);
        Ok(())
    }

    /// Append a raw [`GateInstruction`], validating qubit indices and keeping
    /// `num_params` in sync. All fluent builder methods funnel through here;
    /// it is also useful for programmatic construction (e.g. loops over graph
    /// edges).
    ///
    /// # Panics
    ///
    /// Panics on out-of-range qubit indices, a two-qubit gate whose qubits
    /// coincide (see type-level docs), or a non-finite fixed angle (`NaN` or
    /// infinity). For a fallible variant, use [`try_push`](Self::try_push).
    pub fn push(mut self, gate: GateInstruction) -> Self {
        if let Err(e) = self.try_push(gate) {
            panic!("{e}");
        }
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

    /// Identity gate on `qubit` (`id`): no effect on the state, but it is kept
    /// as an instruction, so it counts towards gate count and depth.
    pub fn id(self, qubit: usize) -> Self {
        self.push(GateInstruction::Id(qubit))
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

    /// Controlled phase gate: control, target, angle.
    pub fn cp(self, q0: usize, q1: usize, theta: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::Cp {
            q0,
            q1,
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

    /// SWAP: exchange the states of qubits `q0` and `q1`.
    pub fn swap(self, q0: usize, q1: usize) -> Self {
        self.push(GateInstruction::Swap(q0, q1))
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

    // ── The rest of qelib1.inc ───────────────────────────────────────────

    /// √X gate on `qubit`.
    pub fn sx(self, qubit: usize) -> Self {
        self.push(GateInstruction::Sx(qubit))
    }

    /// √X† gate on `qubit`.
    pub fn sxdg(self, qubit: usize) -> Self {
        self.push(GateInstruction::Sxdg(qubit))
    }

    /// Controlled-Y with `control` and `target`.
    pub fn cy(self, control: usize, target: usize) -> Self {
        self.push(GateInstruction::Cy(control, target))
    }

    /// Controlled-Hadamard with `control` and `target`.
    pub fn ch(self, control: usize, target: usize) -> Self {
        self.push(GateInstruction::Ch(control, target))
    }

    /// Controlled-√X with `control` and `target`.
    pub fn csx(self, control: usize, target: usize) -> Self {
        self.push(GateInstruction::Csx(control, target))
    }

    /// Toffoli: flips `target` when both `control0` and `control1` are 1.
    pub fn ccx(self, control0: usize, control1: usize, target: usize) -> Self {
        self.push(GateInstruction::Ccx(control0, control1, target))
    }

    /// Fredkin: swaps `target0` and `target1` when `control` is 1.
    pub fn cswap(self, control: usize, target0: usize, target1: usize) -> Self {
        self.push(GateInstruction::Cswap(control, target0, target1))
    }

    /// Controlled X-rotation.
    pub fn crx(self, control: usize, target: usize, theta: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::Crx {
            control,
            target,
            theta: theta.into(),
        })
    }

    /// Controlled Y-rotation.
    pub fn cry(self, control: usize, target: usize, theta: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::Cry {
            control,
            target,
            theta: theta.into(),
        })
    }

    /// Controlled Z-rotation.
    pub fn crz(self, control: usize, target: usize, theta: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::Crz {
            control,
            target,
            theta: theta.into(),
        })
    }

    /// Controlled phase in its `cu1` spelling (the same operator as
    /// [`cp`](Self::cp), exported as `cu1`).
    pub fn cu1(self, q0: usize, q1: usize, theta: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::Cu1 {
            q0,
            q1,
            theta: theta.into(),
        })
    }

    /// Controlled `u3(theta, phi, lam)`.
    pub fn cu3(
        self,
        control: usize,
        target: usize,
        theta: impl Into<GateParam>,
        phi: impl Into<GateParam>,
        lam: impl Into<GateParam>,
    ) -> Self {
        self.push(GateInstruction::Cu3 {
            control,
            target,
            theta: theta.into(),
            phi: phi.into(),
            lam: lam.into(),
        })
    }

    /// Controlled `u(theta, phi, lam)` with phase `gamma` on the controlled
    /// branch (Qiskit's `cu`).
    pub fn cu(
        self,
        control: usize,
        target: usize,
        theta: impl Into<GateParam>,
        phi: impl Into<GateParam>,
        lam: impl Into<GateParam>,
        gamma: impl Into<GateParam>,
    ) -> Self {
        self.push(GateInstruction::Cu {
            control,
            target,
            theta: theta.into(),
            phi: phi.into(),
            lam: lam.into(),
            gamma: gamma.into(),
        })
    }

    /// Phase gate `p(lam)` on `qubit`.
    pub fn p(self, qubit: usize, lam: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::P {
            qubit,
            lam: lam.into(),
        })
    }

    /// `u1(lam)` on `qubit`: the phase gate in its `u1` spelling.
    pub fn u1(self, qubit: usize, lam: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::U1 {
            qubit,
            lam: lam.into(),
        })
    }

    /// `u2(phi, lam)` on `qubit` (= `u3(π/2, phi, lam)`).
    pub fn u2(self, qubit: usize, phi: impl Into<GateParam>, lam: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::U2 {
            qubit,
            phi: phi.into(),
            lam: lam.into(),
        })
    }

    /// `u0(gamma)` on `qubit`: the identity ("idle for `gamma` units"), kept
    /// as an instruction like [`id`](Self::id).
    pub fn u0(self, qubit: usize, gamma: impl Into<GateParam>) -> Self {
        self.push(GateInstruction::U0 {
            qubit,
            gamma: gamma.into(),
        })
    }

    /// Simplified Toffoli (`rccx`): a Toffoli up to relative phases.
    pub fn rccx(self, control0: usize, control1: usize, target: usize) -> Self {
        self.push(GateInstruction::Rccx(control0, control1, target))
    }

    /// Simplified 3-controlled Toffoli (`rc3x`), up to relative phases.
    pub fn rc3x(self, c0: usize, c1: usize, c2: usize, target: usize) -> Self {
        self.push(GateInstruction::Rc3x(c0, c1, c2, target))
    }

    /// 3-controlled X (`c3x`).
    pub fn c3x(self, c0: usize, c1: usize, c2: usize, target: usize) -> Self {
        self.push(GateInstruction::C3x(c0, c1, c2, target))
    }

    /// 3-controlled √X (`c3sqrtx`).
    pub fn c3sqrtx(self, c0: usize, c1: usize, c2: usize, target: usize) -> Self {
        self.push(GateInstruction::C3sqrtx(c0, c1, c2, target))
    }

    /// 4-controlled X (`c4x`).
    pub fn c4x(self, c0: usize, c1: usize, c2: usize, c3: usize, target: usize) -> Self {
        self.push(GateInstruction::C4x(c0, c1, c2, c3, target))
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

    /// The number of classical bits declared by the OpenQASM program this
    /// circuit was imported from (the sum of its `creg` sizes, or of its `bit`
    /// register sizes in OpenQASM 3), or `None` when it declares none or the
    /// circuit was built rather than imported.
    ///
    /// Not part of the public API: the native backend reads it so that a
    /// measured program's counts are keyed at the declared width, as on Aer
    /// (contract C-3). It is kept by `clone` and by the builder methods,
    /// ignored by `==`, and dropped by [`assign_parameters`](Self::assign_parameters)
    /// and by every export.
    #[doc(hidden)]
    pub fn declared_clbits(&self) -> Option<usize> {
        self.declared_clbits
    }

    // ── Binding and export ───────────────────────────────────────────────

    /// Bind concrete values to the circuit's free parameters, producing a
    /// [`ConcreteCircuit`] in which every [`GateParam`] is `Fixed`. Every
    /// expression is evaluated with these values.
    ///
    /// # Errors
    ///
    /// - [`CircuitError::WrongNumberOfParams`] if `params.len() != self.num_params`.
    /// - [`CircuitError::ParamIndexOutOfBounds`] if a gate references an index
    ///   `>= params.len()` (only possible for manually assembled circuits).
    /// - [`CircuitError::NonFiniteParam`] if a value bound to a parameter a
    ///   gate uses, or the value of an angle, is `NaN` or infinite. Only the
    ///   final value of an expression is an angle: an infinite intermediate
    ///   whose result is finite (`1/exp(x)` for a large `x`) is accepted.
    /// - [`CircuitError::DivisionByZero`] if an expression divides by zero.
    ///   The same two errors report an angle of a declared gate's body that
    ///   the values bound to the call make unusable.
    /// - [`CircuitError::UnknownExpression`] if a gate refers to an expression
    ///   this circuit does not hold (only possible when `gates` was assembled
    ///   by hand).
    pub fn assign_parameters(&self, params: &[f64]) -> Result<ConcreteCircuit, CircuitError> {
        if params.len() != self.num_params {
            return Err(CircuitError::WrongNumberOfParams {
                expected: self.num_params,
                got: params.len(),
            });
        }

        // Scratch space for expression evaluation; never allocates for
        // circuits without expressions.
        let mut stack = Vec::new();
        let mut resolve = |p: &GateParam| -> Result<GateParam, CircuitError> {
            Ok(GateParam::Fixed(p.resolve(
                params,
                &self.exprs,
                &mut stack,
            )?))
        };

        let mut gates = Vec::with_capacity(self.gates.len());
        for gate in &self.gates {
            let bound = match gate {
                // A call with free angles is checked through its body now that
                // they have values; one with fixed angles already was.
                GateInstruction::Custom(call) if call.has_free_angles() => {
                    GateInstruction::Custom(call.bind(&mut resolve)?)
                }
                // Exhaustive over the vocabulary (no wildcard arm), so a new
                // parameterised gate can never slip through unbound.
                _ => gate.try_map_params(&mut resolve)?,
            };
            gates.push(bound);
        }

        Ok(ConcreteCircuit {
            num_qubits: self.num_qubits,
            gates,
        })
    }

    /// Bind `params` and serialize to OpenQASM 2.0 in one step.
    /// Equivalent to `self.assign_parameters(params)?.try_to_qasm2()`.
    ///
    /// # Errors
    ///
    /// Those of [`assign_parameters`](Self::assign_parameters) and of
    /// [`ConcreteCircuit::try_to_qasm2`].
    pub fn to_qasm2_with_params(&self, params: &[f64]) -> Result<String, CircuitError> {
        if params.len() != self.num_params {
            return Err(CircuitError::WrongNumberOfParams {
                expected: self.num_params,
                got: params.len(),
            });
        }
        qasm::write_qasm2(
            self.num_qubits,
            self.num_clbits(),
            &self.gates,
            params,
            &self.exprs,
        )
    }

    /// Serialize to a program in the OpenQASM 3 profile with Qiskit phase
    /// conventions (see [`from_qasm3`](Self::from_qasm3)), free parameters
    /// included: each is an `input float[64]` under its name
    /// ([`param_names`](Self::param_names)), and angles are written as the
    /// expressions they are.
    ///
    /// The output is canonical: `to_qasm3(from_qasm3(to_qasm3(c)))` is
    /// byte-identical to `to_qasm3(c)`. Every instruction keeps its name,
    /// except that `u` is written as the builtin `U`; the Polypus instructions
    /// `stdgates.inc` lacks (`rzz`, `rxx`, `sxdg`, `csx`, `cu1`, `cu3`, `u0`,
    /// `rccx`, `rc3x`, `c3x`, `c3sqrtx`, `c4x`) are written as calls of gates
    /// the output defines, whose bodies reproduce their matrices exactly under
    /// the profile's (Qiskit's) phase conventions. Declared gates are printed
    /// from their definitions. A gate or register name that OpenQASM 3 does
    /// not accept, or that would clash, is renamed deterministically; a
    /// parameter's name never changes. Numbers are written as the shortest
    /// decimal that reads back as the same `f64`.
    ///
    /// `U`, `u2` and `u3` mean Qiskit's matrices, which differ from the
    /// OpenQASM 3 specification's by global phases: a reader that follows the
    /// specification computes the same probabilities, counts and expectation
    /// values, but amplitudes that may differ by those global factors (see
    /// [`from_qasm3`](Self::from_qasm3)).
    ///
    /// # Errors
    ///
    /// [`CircuitError::GateNotExpressible`] for a declared gate OpenQASM 3
    /// cannot express (an OpenQASM 2.0 body with a barrier, or a number too
    /// large for binary64); [`CircuitError::ExportLimit`] if the program would
    /// exceed the importer's source-size limit, so it could not be read back;
    /// the errors of [`assign_parameters`](Self::assign_parameters) for a
    /// hand-assembled circuit whose angles cannot be written.
    pub fn to_qasm3(&self) -> Result<String, CircuitError> {
        crate::qasm3::write_qasm3(self, None)
    }

    /// Bind `params` and serialize the bound circuit to the OpenQASM 3
    /// profile, with no inputs. Equivalent to exporting
    /// `self.assign_parameters(params)?` with [`to_qasm3`](Self::to_qasm3).
    pub fn to_qasm3_with_params(&self, params: &[f64]) -> Result<String, CircuitError> {
        crate::qasm3::write_qasm3(self, Some(params))
    }

    /// Bind `params` and serialize to a QIR Base Profile LLVM IR module in one
    /// step. Equivalent to `self.assign_parameters(params)?.to_qir()`.
    ///
    /// See [`ConcreteCircuit::to_qir`] for the gate-to-intrinsic mapping and
    /// the decompositions applied to `rzz`/`rxx`/`u3`.
    pub fn to_qir_with_params(&self, params: &[f64]) -> Result<String, CircuitError> {
        if params.len() != self.num_params {
            return Err(CircuitError::WrongNumberOfParams {
                expected: self.num_params,
                got: params.len(),
            });
        }
        qir::write_qir(
            self.num_qubits,
            self.num_clbits(),
            &self.gates,
            params,
            &self.exprs,
        )
    }

    /// Bind `params` and serialize to QIR LLVM bitcode (`.bc`) in one step.
    /// Equivalent to `self.assign_parameters(params)?.to_qir_bitcode()`.
    ///
    /// Requires `llvm-as` on `PATH`.
    pub fn to_qir_bitcode_with_params(&self, params: &[f64]) -> Result<Vec<u8>, CircuitError> {
        if params.len() != self.num_params {
            return Err(CircuitError::WrongNumberOfParams {
                expected: self.num_params,
                got: params.len(),
            });
        }
        qir::write_qir_bitcode(
            self.num_qubits,
            self.num_clbits(),
            &self.gates,
            params,
            &self.exprs,
        )
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
    /// Panics where [`try_to_qasm2`](Self::try_to_qasm2) returns an error:
    /// if the circuit calls a gate declared in OpenQASM 3 whose body OpenQASM
    /// 2.0 cannot express ([`CircuitError::GateNotExpressible`]), or two
    /// different gates declared in OpenQASM 2.0 under one name
    /// ([`CircuitError::ConflictingGateDefinitions`], only possible when calls
    /// from different imported programs are combined in one circuit), or if a
    /// gate parameter cannot be resolved: an unbound [`GateParam::Param`] or
    /// [`GateParam::Expr`], or a [`GateParam::Fixed`] holding a non-finite
    /// value (`NaN` or infinity). The last cannot happen for circuits produced
    /// by [`ParameterizedCircuit::assign_parameters`] (which rejects
    /// non-finite values at binding time); it is only possible when the
    /// `gates` field was assembled manually.
    pub fn to_qasm2(&self) -> String {
        self.try_to_qasm2().expect(
            "ConcreteCircuit contains an unbound Param or Expr, a non-finite fixed angle, two different declared gates under one name, or a declared gate OpenQASM 2.0 cannot express; use ParameterizedCircuit::assign_parameters and to_qasm2_with_params, or try_to_qasm2",
        )
    }

    /// Serialize to OpenQASM 2.0, or report why the circuit cannot be
    /// written in it.
    ///
    /// # Errors
    ///
    /// [`CircuitError::GateNotExpressible`] for a declared gate imported
    /// from OpenQASM 3 whose body uses `arcsin`, `arccos` or `arctan`, which
    /// OpenQASM 2.0 lacks; [`CircuitError::ConflictingGateDefinitions`] if the
    /// circuit calls two different gates declared in OpenQASM 2.0 under one
    /// name; and, for a hand-assembled `gates` field only, the errors of an
    /// unbound or non-finite angle.
    pub fn try_to_qasm2(&self) -> Result<String, CircuitError> {
        qasm::write_qasm2(
            self.num_qubits,
            self.num_clbits(),
            &self.gates,
            &[],
            &ExprArena::EMPTY,
        )
    }

    /// Serialize to a QIR Base Profile LLVM IR module.
    ///
    /// Most gates map to a standard QIS intrinsic; `rzz`/`rxx`/`u3` are
    /// decomposed to the standard set and `barrier` is dropped. The result is
    /// a complete, self-contained `.ll` module with a single `@main` entry
    /// point, suitable for any QIR Base Profile consumer.
    ///
    /// # Panics
    ///
    /// Panics if a gate parameter cannot be resolved (an unbound
    /// [`GateParam::Param`] or [`GateParam::Expr`], or a [`GateParam::Fixed`]
    /// holding a non-finite value), or if the sequence violates the terminal-measurement model (a
    /// gate acting on an already-measured qubit; contract C-4). None can happen
    /// for circuits produced by [`ParameterizedCircuit::assign_parameters`];
    /// all are only possible when the `gates` field was assembled manually. For
    /// a fallible export, use
    /// [`ParameterizedCircuit::to_qir_with_params`](crate::ParameterizedCircuit::to_qir_with_params).
    pub fn to_qir(&self) -> String {
        qir::write_qir(
            self.num_qubits,
            self.num_clbits(),
            &self.gates,
            &[],
            &ExprArena::EMPTY,
        )
        .expect(
            "ConcreteCircuit is invalid (unbound Param or Expr, non-finite fixed angle, or a gate after a measurement); build it via ParameterizedCircuit",
        )
    }

    /// Serialize to QIR LLVM bitcode (`.bc`).
    ///
    /// Requires `llvm-as` on `PATH`.
    ///
    /// # Errors
    ///
    /// Returns an error when `llvm-as` is unavailable or fails to assemble the
    /// generated textual QIR module.
    pub fn to_qir_bitcode(&self) -> Result<Vec<u8>, CircuitError> {
        qir::write_qir_bitcode(
            self.num_qubits,
            self.num_clbits(),
            &self.gates,
            &[],
            &ExprArena::EMPTY,
        )
    }
}

/// The default name of parameter `index`: `theta_<index>`, or the first
/// `theta_<index>_<k>` (k = 1, 2, …) not in `taken`. Default names never
/// collide with each other: `<index>` has no underscore, so every candidate
/// names one index only.
fn default_param_name(index: usize, taken: &BTreeSet<&str>) -> String {
    let base = format!("theta_{index}");
    if !taken.contains(base.as_str()) {
        return base;
    }
    (1usize..)
        .map(|k| format!("{base}_{k}"))
        .find(|name| !taken.contains(name.as_str()))
        .unwrap_or(base)
}

/// Shared classical-register sizing logic.
pub(crate) fn num_clbits(num_qubits: usize, gates: &[GateInstruction]) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Param;

    fn named(names: &[&str]) -> ParameterizedCircuit {
        let mut qc = ParameterizedCircuit::new(1).rx(0, Param(names.len().saturating_sub(1)));
        qc.param_names = names.iter().map(|n| n.to_string()).collect();
        qc
    }

    #[test]
    fn given_names_are_kept_and_the_rest_default() {
        let mut qc = named(&["gamma", "beta"]);
        assert_eq!(qc.param_names(), ["gamma", "beta"]);
        qc = qc.rz(0, Param(3));
        assert_eq!(qc.param_names(), ["gamma", "beta", "theta_2", "theta_3"]);
        // Names beyond `num_params` are not reported.
        qc.num_params = 1;
        assert_eq!(qc.param_names(), ["gamma"]);
    }

    #[test]
    fn default_names_avoid_given_ones() {
        // Parameter 0 was named like parameter 1's default.
        let qc = named(&["theta_1"]).ry(0, Param(1));
        assert_eq!(qc.param_names(), ["theta_1", "theta_1_1"]);
        let qc = named(&["theta_1", "theta_1_1"]).ry(0, Param(2));
        assert_eq!(qc.param_names(), ["theta_1", "theta_1_1", "theta_2"]);
        let qc = named(&["theta_2", "theta_2_1"]).ry(0, Param(2));
        assert_eq!(qc.param_names(), ["theta_2", "theta_2_1", "theta_2_2"]);
    }

    #[test]
    fn names_are_part_of_the_circuit() {
        assert_eq!(named(&["a", "b"]), named(&["a", "b"]));
        assert_ne!(named(&["a", "b"]), named(&["a", "c"]));
        // A given name equal to the default is the same name.
        assert_eq!(
            named(&["theta_0"]),
            ParameterizedCircuit::new(1).rx(0, Param(0))
        );
        let qc = named(&["gamma"]);
        assert_eq!(qc.clone().param_names(), ["gamma"]);
    }
}
