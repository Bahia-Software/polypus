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
use crate::gate::{GateInstruction, GateParam};
use crate::qasm_import::BuiltinGate;
use std::fmt;
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

// ───────────────────────────── Expressions ──────────────────────────────

/// A unary function of the OpenQASM 2.0 expression grammar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Func {
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
}

impl Func {
    pub(crate) fn from_name(name: &str) -> Option<Func> {
        Some(match name {
            "sin" => Func::Sin,
            "cos" => Func::Cos,
            "tan" => Func::Tan,
            "exp" => Func::Exp,
            "ln" => Func::Ln,
            "sqrt" => Func::Sqrt,
            _ => return None,
        })
    }

    fn apply(self, v: f64) -> f64 {
        match self {
            Func::Sin => v.sin(),
            Func::Cos => v.cos(),
            Func::Tan => v.tan(),
            Func::Exp => v.exp(),
            Func::Ln => v.ln(),
            Func::Sqrt => v.sqrt(),
        }
    }
}

/// `+` / `-` between the terms of a [`Expr::Sum`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum AddOp {
    Add,
    Sub,
}

/// `*` / `/` between the factors of a [`Expr::Product`]; a division keeps its
/// source line for the division-by-zero error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum MulOp {
    Mul,
    Div { line: usize },
}

/// A parameter expression, as parsed. Top-level expressions are constant and
/// evaluated immediately; inside a gate body they may name the gate's formal
/// parameters ([`Expr::Param`]) and are evaluated at each call site.
///
/// Evaluation performs exactly the floating-point operations the source
/// spells, in the same order (left to right within a sum or a product), so a
/// constant expression evaluates bit-for-bit as it always has. Sums and
/// products are flat, not nested binary nodes: an arbitrarily long `a+b+c+…`
/// is evaluated by a loop, never by recursion as deep as the chain is long,
/// and every other form of nesting is bounded by the parser's depth limit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
    Num(f64),
    /// The gate's formal parameter at this position.
    Param(usize),
    Neg(Box<Expr>),
    /// `first (± term)*`.
    Sum(Box<Expr>, Vec<(AddOp, Expr)>),
    /// `first (*|/ factor)*`.
    Product(Box<Expr>, Vec<(MulOp, Expr)>),
    /// `base ^ exponent`.
    Pow(Box<Expr>, Box<Expr>),
    Func(Func, Box<Expr>),
}

/// Why an expression could not be evaluated to a usable angle.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum EvalError {
    /// A division by zero, at this source line.
    DivisionByZero { line: usize },
    /// The value is `NaN` or infinite (e.g. `ln(0)`), not a valid angle.
    NonFinite,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::DivisionByZero { .. } => {
                write!(f, "division by zero in parameter expression")
            }
            EvalError::NonFinite => write!(
                f,
                "parameter expression evaluated to a non-finite value (NaN or infinity)"
            ),
        }
    }
}

impl Expr {
    /// Evaluate with `args` bound to the formal parameters. The recursion is
    /// bounded by the parser's expression-depth limit.
    pub(crate) fn eval(&self, args: &[f64]) -> Result<f64, EvalError> {
        Ok(match self {
            Expr::Num(v) => *v,
            // In range: the parser only builds `Param(i)` for a declared name.
            Expr::Param(i) => args[*i],
            Expr::Neg(e) => -e.eval(args)?,
            Expr::Sum(first, rest) => {
                let mut v = first.eval(args)?;
                for (op, term) in rest {
                    let t = term.eval(args)?;
                    match op {
                        AddOp::Add => v += t,
                        AddOp::Sub => v -= t,
                    }
                }
                v
            }
            Expr::Product(first, rest) => {
                let mut v = first.eval(args)?;
                for (op, factor) in rest {
                    let f = factor.eval(args)?;
                    match op {
                        MulOp::Mul => v *= f,
                        MulOp::Div { line } => {
                            if f == 0.0 {
                                return Err(EvalError::DivisionByZero { line: *line });
                            }
                            v /= f;
                        }
                    }
                }
                v
            }
            Expr::Pow(base, exponent) => base.eval(args)?.powf(exponent.eval(args)?),
            Expr::Func(f, e) => f.apply(e.eval(args)?),
        })
    }

    /// [`Self::eval`], rejecting a non-finite result: the value must be a
    /// usable angle (contract C-2).
    pub(crate) fn eval_angle(&self, args: &[f64]) -> Result<f64, EvalError> {
        let value = self.eval(args)?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(EvalError::NonFinite)
        }
    }
}

// ───────────────────────────── Definitions ──────────────────────────────

/// One statement of a gate body, over the gate's formal arguments (qubits by
/// position in the declaration's qubit list).
#[derive(Debug, PartialEq)]
pub(crate) enum BodyOp {
    /// A built-in (`qelib1.inc`) gate.
    Builtin {
        gate: &'static BuiltinGate,
        params: Vec<Expr>,
        qubits: Vec<usize>,
    },
    /// A gate declared earlier in the same program.
    Call {
        definition: Arc<GateDefinition>,
        params: Vec<Expr>,
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
        for op in &self.body {
            match op {
                BodyOp::Builtin { params, .. } => {
                    for e in params {
                        e.eval_angle(args)?;
                    }
                }
                BodyOp::Call {
                    definition, params, ..
                } => {
                    let values = params
                        .iter()
                        .map(|e| e.eval_angle(args))
                        .collect::<Result<Vec<_>, _>>()?;
                    definition.validate(&values)?;
                }
                BodyOp::Barrier(_) => {}
            }
        }
        Ok(())
    }

    /// Instantiate the body for one call: `args` bound to the formal
    /// parameters, `qubits` to the formal qubits. Every built-in instruction of
    /// the (fully) expanded body is handed to `sink`, in order.
    pub(crate) fn instantiate(
        &self,
        args: &[f64],
        qubits: &[usize],
        sink: &mut impl FnMut(GateInstruction),
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
                        .map(|e| e.eval_angle(args).map(GateParam::Fixed))
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
                        .map(|e| e.eval_angle(args))
                        .collect::<Result<Vec<_>, _>>()?;
                    definition.instantiate(&values, &actual(formal), sink)?;
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

    /// The same call with its parameters replaced (binding).
    pub(crate) fn with_params(&self, params: Vec<GateParam>) -> Self {
        CustomGate {
            definition: Arc::clone(&self.definition),
            params,
            qubits: self.qubits.clone(),
        }
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
    /// if a call parameter cannot be resolved, and
    /// [`CircuitError::NonFiniteParam`] if an angle of the body evaluates to a
    /// non-finite value (including a division by zero) for these arguments.
    pub fn expand(&self, params: &[f64]) -> Result<Vec<GateInstruction>, CircuitError> {
        let args = self
            .params
            .iter()
            .map(|p| p.resolve(params))
            .collect::<Result<Vec<f64>, _>>()?;
        // Not pre-sized with `expansion_size`: that also counts the nested
        // calls, so it can far exceed the number of instructions produced.
        let mut out = Vec::new();
        self.definition
            .instantiate(&args, &self.qubits, &mut |g| out.push(g))
            .map_err(|_| CircuitError::NonFiniteParam)?;
        Ok(out)
    }
}
