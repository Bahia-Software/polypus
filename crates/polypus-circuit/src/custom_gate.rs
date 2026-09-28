//! Gates declared in the source program: OpenQASM 2.0 `gate` blocks.
//!
//! A declaration `gate name(params) qargs { body }` becomes a
//! [`GateDefinition`]: a template over its formal parameters and qubit
//! arguments, plus the verbatim text of the declaration. A call of the gate is
//! **one** instruction, [`GateInstruction::Custom`](crate::GateInstruction::Custom),
//! holding its definition — the circuit never contains the expanded body, so
//! the OpenQASM exporter re-emits the declaration and the call exactly as the
//! source had them, and a backend that parses the export (Qiskit, then Aer)
//! builds the same program as from the original file.
//!
//! Expanding a call into built-in instructions ([`CustomGate::expand`]) is a
//! *lowering* step, used only where a backend needs built-in gates: the native
//! simulator and the QIR exporter.
//!
//! Declarations come only from the importer, which validates them once:
//! recursion is impossible (a body may only call gates declared *before* it),
//! nesting is capped at [`MAX_GATE_NESTING`] levels and a single declaration's
//! full expansion may visit at most [`MAX_GATE_EXPANSION`] body statements, so
//! expansion is always bounded (`qasm_import` is an untrusted input surface).

use crate::error::CircuitError;
use crate::expr::{EvalError, ExprArena, FormalExpr};
use crate::gate::{GateInstruction, GateParam};
use crate::qasm_import::BuiltinGate;
use std::sync::Arc;

/// Deepest nesting of gate calls inside gate bodies. Declared gates can only
/// call earlier ones, so nesting can never be infinite; the cap bounds the
/// recursion of expansion itself. Real hierarchies nest a handful of levels.
pub(crate) const MAX_GATE_NESTING: usize = 64;

/// Most body statements one full expansion of a declaration may visit
/// (built-in instructions, barriers and nested calls alike). Without it, a few
/// lines of nested declarations (each calling the previous one twice) would
/// describe an exponentially large circuit — or, with empty bodies, an
/// exponentially long walk that produces nothing.
pub(crate) const MAX_GATE_EXPANSION: usize = 1_000_000;

// ───────────────────────────── Definitions ──────────────────────────────

/// One statement of a gate body, over the gate's formal arguments (qubits by
/// position in the declaration's qubit list).
#[derive(Debug, PartialEq)]
pub(crate) enum BodyOp {
    /// A built-in (`qelib1.inc`) gate.
    Builtin {
        gate: &'static BuiltinGate,
        params: Vec<FormalExpr>,
        qubits: Vec<usize>,
    },
    /// A gate declared earlier in the same program.
    Call {
        definition: Arc<GateDefinition>,
        params: Vec<FormalExpr>,
        qubits: Vec<usize>,
    },
    /// `barrier` over some of the formal qubits.
    Barrier(Vec<usize>),
}

/// A gate declared with an OpenQASM 2.0 `gate` block: a template over its
/// formal parameters and qubit arguments, and the declaration's source text.
///
/// Only the QASM importer creates definitions, after validating them (see the
/// module docs); calls hold them through an [`Arc`], so a definition is shared
/// by all its calls and by every clone of the circuit.
#[derive(Debug, PartialEq)]
pub struct GateDefinition {
    name: String,
    param_names: Vec<String>,
    qubit_names: Vec<String>,
    body: Vec<BodyOp>,
    /// The declaration, from `gate` to the closing `}`, as written (line
    /// endings normalised to `\n`). The exporter emits it unchanged.
    declaration: String,
    /// Position among the declarations of its source program, so the exporter
    /// can emit declarations in their original order.
    ordinal: usize,
    /// Number of body statements one full expansion of a call visits: every
    /// built-in instruction and barrier, *and* every nested call. Counting the
    /// calls themselves matters: gates with empty bodies expand to nothing, yet
    /// `g1 { g0; g0; }`, `g2 { g1; g1; }`, … still make expansion (and the
    /// importer's validation of each call) walk an exponential number of nodes.
    expansion_size: usize,
    /// Nesting depth: 1 for a body of built-in gates only.
    depth: usize,
}

/// Why a declaration was rejected (the importer adds the line and gate name).
#[derive(Debug, PartialEq)]
pub(crate) enum DefinitionError {
    TooDeep,
    TooLarge,
}

impl GateDefinition {
    /// Build a validated definition. The parser has already checked the body's
    /// operands and signatures; this computes (and bounds) the expansion.
    pub(crate) fn new(
        name: String,
        param_names: Vec<String>,
        qubit_names: Vec<String>,
        body: Vec<BodyOp>,
        declaration: String,
        ordinal: usize,
    ) -> Result<Self, DefinitionError> {
        let mut expansion_size = 0usize;
        let mut depth = 1;
        for op in &body {
            match op {
                BodyOp::Builtin { .. } | BodyOp::Barrier(_) => {
                    expansion_size = expansion_size.saturating_add(1);
                }
                BodyOp::Call { definition, .. } => {
                    // The call node itself, then everything its body visits.
                    expansion_size = expansion_size
                        .saturating_add(1)
                        .saturating_add(definition.expansion_size);
                    depth = depth.max(definition.depth + 1);
                }
            }
        }
        if depth > MAX_GATE_NESTING {
            return Err(DefinitionError::TooDeep);
        }
        if expansion_size > MAX_GATE_EXPANSION {
            return Err(DefinitionError::TooLarge);
        }
        Ok(GateDefinition {
            name,
            param_names,
            qubit_names,
            body,
            declaration,
            ordinal,
            expansion_size,
            depth,
        })
    }

    /// The gate's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of angle parameters a call takes.
    pub fn num_params(&self) -> usize {
        self.param_names.len()
    }

    /// Number of qubits a call acts on.
    pub fn num_qubits(&self) -> usize {
        self.qubit_names.len()
    }

    /// The declaration's OpenQASM 2.0 source text, re-emitted as is by the
    /// exporter.
    pub fn declaration(&self) -> &str {
        &self.declaration
    }

    pub(crate) fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub(crate) fn expansion_size(&self) -> usize {
        self.expansion_size
    }

    /// The definitions this body calls directly.
    pub(crate) fn callees(&self) -> impl Iterator<Item = &Arc<GateDefinition>> {
        self.body.iter().filter_map(|op| match op {
            BodyOp::Call { definition, .. } => Some(definition),
            _ => None,
        })
    }

    /// Check that one call with angles `args` is valid: every angle of the
    /// (fully) expanded body evaluates to a finite value. The same walk as
    /// [`Self::instantiate`] without building any instruction, so the importer
    /// can validate every call cheaply.
    pub(crate) fn validate(&self, args: &[f64]) -> Result<(), EvalError> {
        self.validate_with(args, &mut Vec::new())
    }

    /// [`Self::validate`], with `stack` as the expressions' scratch space.
    fn validate_with(&self, args: &[f64], stack: &mut Vec<f64>) -> Result<(), EvalError> {
        for op in &self.body {
            match op {
                BodyOp::Builtin { params, .. } => {
                    for e in params {
                        e.eval_angle(args, stack)?;
                    }
                }
                BodyOp::Call {
                    definition, params, ..
                } => {
                    let values = params
                        .iter()
                        .map(|e| e.eval_angle(args, stack))
                        .collect::<Result<Vec<_>, _>>()?;
                    definition.validate_with(&values, stack)?;
                }
                BodyOp::Barrier(_) => {}
            }
        }
        Ok(())
    }

    /// Instantiate the body for one call: `args` bound to the formal
    /// parameters, `qubits` to the formal qubits. Every built-in instruction of
    /// the (fully) expanded body is handed to `sink`, in order. `stack` is the
    /// expressions' scratch space.
    pub(crate) fn instantiate(
        &self,
        args: &[f64],
        qubits: &[usize],
        sink: &mut impl FnMut(GateInstruction),
        stack: &mut Vec<f64>,
    ) -> Result<(), EvalError> {
        let actual =
            |formal: &[usize]| -> Vec<usize> { formal.iter().map(|&i| qubits[i]).collect() };
        for op in &self.body {
            match op {
                BodyOp::Builtin {
                    gate,
                    params,
                    qubits: formal,
                } => {
                    let values = params
                        .iter()
                        .map(|e| e.eval_angle(args, stack).map(GateParam::Fixed))
                        .collect::<Result<Vec<_>, _>>()?;
                    sink((gate.build)(&values, &actual(formal)));
                }
                BodyOp::Call {
                    definition,
                    params,
                    qubits: formal,
                } => {
                    let values = params
                        .iter()
                        .map(|e| e.eval_angle(args, stack))
                        .collect::<Result<Vec<_>, _>>()?;
                    definition.instantiate(&values, &actual(formal), sink, stack)?;
                }
                BodyOp::Barrier(formal) => sink(GateInstruction::Barrier(actual(formal))),
            }
        }
        Ok(())
    }
}

// ──────────────────────────────── Calls ─────────────────────────────────

/// A call of a [`GateDefinition`]: the instruction a `name(args) qubits;`
/// statement becomes. It is re-emitted as that same statement (after the
/// declaration), never as its expanded body.
#[derive(Debug, Clone, PartialEq)]
pub struct CustomGate {
    definition: Arc<GateDefinition>,
    params: Vec<GateParam>,
    qubits: Vec<usize>,
}

impl CustomGate {
    /// A call with parameter and qubit counts matching `definition` (checked
    /// by the importer).
    pub(crate) fn new(
        definition: Arc<GateDefinition>,
        params: Vec<GateParam>,
        qubits: Vec<usize>,
    ) -> Self {
        debug_assert_eq!(params.len(), definition.num_params());
        debug_assert_eq!(qubits.len(), definition.num_qubits());
        CustomGate {
            definition,
            params,
            qubits,
        }
    }

    /// The called gate's name.
    pub fn name(&self) -> &str {
        self.definition.name()
    }

    /// The called gate's definition.
    pub fn definition(&self) -> &GateDefinition {
        &self.definition
    }

    /// The call's angle arguments, in order.
    pub fn params(&self) -> &[GateParam] {
        &self.params
    }

    /// The qubits the call acts on, in order.
    pub fn qubits(&self) -> &[usize] {
        &self.qubits
    }

    /// A call of the same declared gate with other arguments: `params` for
    /// its angles — fixed values, free parameters, or expressions stored in
    /// the circuit the call is pushed onto — and `qubits` for its qubits.
    ///
    /// A call whose angles are all fixed is checked now, through its whole
    /// body, as the importer checks a call. A call with free parameters is
    /// checked through its whole body whenever they are bound: by
    /// [`ParameterizedCircuit::assign_parameters`](crate::ParameterizedCircuit::assign_parameters)
    /// and by the exports that take parameter values.
    ///
    /// ```
    /// use polypus_circuit::{GateInstruction, Param, ParameterizedCircuit};
    ///
    /// let src = "OPENQASM 2.0;\ngate g(t) a { rz(t/2) a; }\nqreg q[1];\ng(0) q[0];\n";
    /// let imported = ParameterizedCircuit::from_qasm2(src).unwrap();
    /// let GateInstruction::Custom(call) = &imported.gates[0] else { unreachable!() };
    ///
    /// let mut qc = ParameterizedCircuit::new(2);
    /// qc.try_push(GateInstruction::Custom(call.with_arguments(vec![Param(0)], vec![1]).unwrap()))
    ///     .unwrap();
    /// assert_eq!(qc.num_params, 1);
    /// assert!(qc.to_qasm2_with_params(&[0.5]).unwrap().contains("g(0.500000000000) q[1];"));
    /// ```
    ///
    /// # Errors
    ///
    /// [`CircuitError::GateSignature`] if the numbers of angles or qubits
    /// differ from the declaration's; for fixed angles,
    /// [`CircuitError::NonFiniteParam`] or [`CircuitError::DivisionByZero`]
    /// if an angle, or an angle of the body, is not usable. Qubit indices and
    /// expressions are checked when the call is pushed onto a circuit.
    pub fn with_arguments(
        &self,
        params: Vec<GateParam>,
        qubits: Vec<usize>,
    ) -> Result<CustomGate, CircuitError> {
        let definition = &self.definition;
        if params.len() != definition.num_params() || qubits.len() != definition.num_qubits() {
            let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
            return Err(CircuitError::GateSignature {
                name: definition.name().to_string(),
                params: (count(definition.num_params()), count(params.len())),
                qubits: (count(definition.num_qubits()), count(qubits.len())),
            });
        }
        let call = CustomGate {
            definition: Arc::clone(definition),
            params,
            qubits,
        };
        if !call.has_free_angles() {
            call.check_bound()?;
        }
        Ok(call)
    }

    /// The same call with its parameters replaced (binding).
    pub(crate) fn with_params(&self, params: Vec<GateParam>) -> Self {
        CustomGate {
            definition: Arc::clone(&self.definition),
            params,
            qubits: self.qubits.clone(),
        }
    }

    /// The call with its angles replaced by `resolve(angle)`, checked through
    /// its body for them (binding a call with free angles).
    pub(crate) fn bind(
        &self,
        resolve: impl FnMut(&GateParam) -> Result<GateParam, CircuitError>,
    ) -> Result<CustomGate, CircuitError> {
        let bound = self.with_params(self.params.iter().map(resolve).collect::<Result<_, _>>()?);
        bound.check_bound()?;
        Ok(bound)
    }

    /// Whether an angle is a free parameter or an expression: the call has
    /// not been checked through its body yet.
    pub(crate) fn has_free_angles(&self) -> bool {
        self.params
            .iter()
            .any(|p| !matches!(p, GateParam::Fixed(_)))
    }

    /// Check a call whose angles are fixed through its whole body.
    pub(crate) fn check_bound(&self) -> Result<(), CircuitError> {
        let values = self
            .params
            .iter()
            .map(|p| match *p {
                GateParam::Fixed(v) if v.is_finite() => Ok(v),
                GateParam::Fixed(_) => Err(CircuitError::NonFiniteParam),
                GateParam::Param(index) => Err(CircuitError::ParamIndexOutOfBounds {
                    index,
                    num_params: 0,
                }),
                GateParam::Expr(_) => Err(CircuitError::UnknownExpression),
            })
            .collect::<Result<Vec<f64>, _>>()?;
        self.check_body(&values)
    }

    /// Check that every angle of the body is usable when the call's angles
    /// are `values`.
    pub(crate) fn check_body(&self, values: &[f64]) -> Result<(), CircuitError> {
        self.definition.validate(values).map_err(body_error)
    }

    /// Expand the call into built-in instructions — through every nested
    /// declared gate — with its parameters resolved against `params` (pass an
    /// empty slice for a concrete circuit).
    ///
    /// This is a lowering step for backends that need built-in gates (the
    /// native simulator, the QIR exporter); the circuit keeps the call.
    ///
    /// # Errors
    ///
    /// [`CircuitError::ParamIndexOutOfBounds`] / [`CircuitError::NonFiniteParam`]
    /// if a call parameter cannot be resolved,
    /// [`CircuitError::UnknownExpression`] if one is an expression (only its
    /// circuit can evaluate it: bind the circuit first), and
    /// [`CircuitError::NonFiniteParam`] or [`CircuitError::DivisionByZero`] if
    /// an angle of the body is not a usable value for these arguments.
    pub fn expand(&self, params: &[f64]) -> Result<Vec<GateInstruction>, CircuitError> {
        self.expand_with(params, &ExprArena::EMPTY, &mut Vec::new())
    }

    /// [`Self::expand`] for a call in a circuit whose expressions are
    /// `exprs`; `stack` is the expressions' scratch space.
    pub(crate) fn expand_with(
        &self,
        params: &[f64],
        exprs: &ExprArena,
        stack: &mut Vec<f64>,
    ) -> Result<Vec<GateInstruction>, CircuitError> {
        let args = self
            .params
            .iter()
            .map(|p| p.resolve(params, exprs, stack))
            .collect::<Result<Vec<f64>, _>>()?;
        // Not pre-sized with `expansion_size`: that also counts the nested
        // calls, so it can far exceed the number of instructions produced.
        let mut out = Vec::new();
        self.definition
            .instantiate(&args, &self.qubits, &mut |g| out.push(g), stack)
            .map_err(body_error)?;
        Ok(out)
    }
}

/// The error a call reports when an angle of its body is not usable.
fn body_error(e: EvalError) -> CircuitError {
    match e {
        EvalError::DivisionByZero { .. } => CircuitError::DivisionByZero,
        _ => CircuitError::NonFiniteParam,
    }
}
