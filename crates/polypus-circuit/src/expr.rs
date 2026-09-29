//! Angle expressions: gate angles computed from other angles.
//!
//! A circuit's angle can be an expression of its free parameters, such as
//! `2*beta` or `(x0 - pi)*(x1 - pi)`. [`ParamExpr`] builds one from Rust;
//! [`ParameterizedCircuit::add_expr`](crate::ParameterizedCircuit::add_expr)
//! validates it and stores it in the circuit, and gates refer to it through
//! [`GateParam::Expr`](crate::GateParam::Expr) and an [`ExprId`]. Binding
//! ([`ParameterizedCircuit::assign_parameters`](crate::ParameterizedCircuit::assign_parameters))
//! evaluates it.
//!
//! The same representation holds the angle expressions in the body of a
//! declared gate, over the gate's formal parameters. The two kinds of
//! reference are distinct types inside this crate: an expression over formal
//! parameters can never be evaluated against a circuit's free parameters, nor
//! the reverse.
//!
//! An expression is stored as a flat sequence of nodes in postfix order:
//! both operands of an operator come before it, the left one first.
//! Evaluation is a single loop over that sequence with a value stack. It
//! performs exactly the floating-point operations the expression spells, in
//! source order, with no reassociation, and it never recurses, however deep
//! the expression is. Cloning or dropping an expression is cloning or
//! dropping a flat vector.

use crate::error::CircuitError;
use std::fmt;
use std::ops;

/// Deepest nesting an expression may have, counted as its OpenQASM text nests:
/// one level per parenthesised subexpression, per unary minus, per function
/// argument and per exponent. A left-associative chain such as `a + b + c`
/// does not nest, however long it is.
///
/// The OpenQASM importers stop at this depth, so an expression that exceeds it
/// could not be imported again once exported.
pub(crate) const MAX_EXPR_DEPTH: usize = 64;

/// Most nodes (numbers, constants, references, operators and function calls)
/// one circuit expression may have.
pub(crate) const MAX_EXPR_NODES: usize = 1_000_000;

// ─────────────────────────── Vocabulary ───────────────────────────────────

/// A function of one argument that an expression can apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Function {
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Tangent.
    Tan,
    /// Inverse sine (`arcsin`; OpenQASM 3 only).
    Arcsin,
    /// Inverse cosine (`arccos`; OpenQASM 3 only).
    Arccos,
    /// Inverse tangent (`arctan`; OpenQASM 3 only).
    Arctan,
    /// The exponential function.
    Exp,
    /// The natural logarithm (`ln` in OpenQASM 2.0, `log` in OpenQASM 3).
    Ln,
    /// Square root.
    Sqrt,
}

impl Function {
    pub(crate) fn apply(self, v: f64) -> f64 {
        match self {
            Function::Sin => v.sin(),
            Function::Cos => v.cos(),
            Function::Tan => v.tan(),
            Function::Arcsin => v.asin(),
            Function::Arccos => v.acos(),
            Function::Arctan => v.atan(),
            Function::Exp => v.exp(),
            Function::Ln => v.ln(),
            Function::Sqrt => v.sqrt(),
        }
    }

    /// The function named `name` in an OpenQASM 2.0 expression.
    pub(crate) fn from_qasm2_name(name: &str) -> Option<Function> {
        Some(match name {
            "sin" => Function::Sin,
            "cos" => Function::Cos,
            "tan" => Function::Tan,
            "exp" => Function::Exp,
            "ln" => Function::Ln,
            "sqrt" => Function::Sqrt,
            _ => return None,
        })
    }
}

/// A named mathematical constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Constant {
    /// π (`pi`).
    Pi,
    /// τ = 2π (`tau`; OpenQASM 3 only).
    Tau,
    /// Euler's number e (`euler`; OpenQASM 3 only).
    Euler,
}

impl Constant {
    /// The constant's `f64` value.
    pub fn value(self) -> f64 {
        match self {
            Constant::Pi => std::f64::consts::PI,
            Constant::Tau => std::f64::consts::TAU,
            Constant::Euler => std::f64::consts::E,
        }
    }
}

/// A reference to a declared gate's formal parameter, by position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Formal(pub(crate) usize);

/// A reference to a circuit's free parameter, by index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Input(pub(crate) usize);

/// One node of an expression in postfix order, with references of kind `R`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Node<R> {
    /// A number. Never negative: `-1.5` is the negation of `1.5`, as it is
    /// written.
    Num(f64),
    Const(Constant),
    Ref(R),
    /// Unary minus of the previous operand.
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    /// The left operand raised to the right one.
    Pow,
    Call(Function),
}

// ─────────────────────────── Evaluation ───────────────────────────────────

/// Why an expression could not be evaluated to a usable angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvalError {
    /// A division by zero, by the node at this position.
    DivisionByZero { at: usize },
    /// The value (or a referenced value) is `NaN` or infinite.
    NonFinite,
    /// A reference to a value that was not supplied.
    Unbound { index: usize },
    /// Not a well-formed postfix sequence. The constructors never build one.
    Malformed,
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
            EvalError::Unbound { index } => write!(
                f,
                "parameter expression refers to parameter {index}, which has no value"
            ),
            EvalError::Malformed => write!(f, "malformed parameter expression"),
        }
    }
}

fn pop(stack: &mut Vec<f64>) -> Result<f64, EvalError> {
    stack.pop().ok_or(EvalError::Malformed)
}

/// Both operands of a binary operator, left first.
fn pop2(stack: &mut Vec<f64>) -> Result<(f64, f64), EvalError> {
    let rhs = pop(stack)?;
    Ok((pop(stack)?, rhs))
}

/// Evaluate `nodes`, taking each reference's value from `value_of`. `stack`
/// is scratch space, reused across calls to avoid an allocation per
/// evaluation. The result is not checked for finiteness: an intermediate or
/// final infinity is an ordinary `f64` here, and callers that need an angle
/// check the final value.
pub(crate) fn evaluate<R: Copy>(
    nodes: &[Node<R>],
    mut value_of: impl FnMut(R) -> Result<f64, EvalError>,
    stack: &mut Vec<f64>,
) -> Result<f64, EvalError> {
    stack.clear();
    for (at, node) in nodes.iter().enumerate() {
        let value = match *node {
            Node::Num(v) => v,
            Node::Const(c) => c.value(),
            Node::Ref(r) => value_of(r)?,
            Node::Neg => -pop(stack)?,
            Node::Call(f) => f.apply(pop(stack)?),
            Node::Add => {
                let (a, b) = pop2(stack)?;
                a + b
            }
            Node::Sub => {
                let (a, b) = pop2(stack)?;
                a - b
            }
            Node::Mul => {
                let (a, b) = pop2(stack)?;
                a * b
            }
            Node::Div => {
                let (a, b) = pop2(stack)?;
                if b == 0.0 {
                    return Err(EvalError::DivisionByZero { at });
                }
                a / b
            }
            Node::Pow => {
                let (a, b) = pop2(stack)?;
                a.powf(b)
            }
        };
        stack.push(value);
    }
    let value = pop(stack)?;
    if stack.is_empty() {
        Ok(value)
    } else {
        Err(EvalError::Malformed)
    }
}

// ───────────────────────────── Nesting ────────────────────────────────────

/// How tightly a subexpression's outermost operation binds, as it is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Class {
    /// A number, constant, reference or function call.
    Atom,
    /// `a ** b` (`a ^ b` in OpenQASM 2.0), right-associative.
    Pow,
    /// Unary minus.
    Neg,
    /// `*` and `/`, left-associative.
    Mul,
    /// `+` and `-`, left-associative.
    Add,
}

/// Where a subexpression appears in its parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Position {
    NegOperand,
    PowBase,
    PowExponent,
    MulLeft,
    MulRight,
    AddLeft,
    AddRight,
}

/// Whether a subexpression of class `child` must be parenthesised at
/// `position` for its text to parse back to the same expression. The
/// grammar, shared by both OpenQASM dialects here: `**` binds tighter than
/// unary minus, which binds tighter than `* /`, then `+ -`; `**` is
/// right-associative and its exponent may start with a unary minus; the other
/// binary operators are left-associative.
pub(crate) fn needs_parens(position: Position, child: Class) -> bool {
    match position {
        Position::AddLeft => false,
        Position::AddRight | Position::MulLeft => child == Class::Add,
        Position::MulRight | Position::NegOperand | Position::PowExponent => {
            matches!(child, Class::Mul | Class::Add)
        }
        Position::PowBase => child != Class::Atom,
    }
}

/// The nesting depth of `nodes` as written with the fewest parentheses (see
/// [`MAX_EXPR_DEPTH`]), or [`EvalError::Malformed`].
pub(crate) fn depth<R>(nodes: &[Node<R>]) -> Result<usize, EvalError> {
    // One entry per pending operand: its class and nesting depth.
    let mut stack: Vec<(Class, usize)> = Vec::new();
    let paren = |position, (class, depth): (Class, usize)| {
        depth + usize::from(needs_parens(position, class))
    };
    for node in nodes {
        let entry = match node {
            // Written with a leading minus, like a negation.
            Node::Num(v) if v.is_sign_negative() => (Class::Neg, 1),
            Node::Num(_) | Node::Const(_) | Node::Ref(_) => (Class::Atom, 0),
            Node::Neg => {
                let operand = stack.pop().ok_or(EvalError::Malformed)?;
                (Class::Neg, 1 + paren(Position::NegOperand, operand))
            }
            Node::Call(_) => {
                let (_, depth) = stack.pop().ok_or(EvalError::Malformed)?;
                (Class::Atom, 1 + depth)
            }
            Node::Add | Node::Sub | Node::Mul | Node::Div | Node::Pow => {
                let rhs = stack.pop().ok_or(EvalError::Malformed)?;
                let lhs = stack.pop().ok_or(EvalError::Malformed)?;
                match node {
                    Node::Add | Node::Sub => (
                        Class::Add,
                        paren(Position::AddLeft, lhs).max(paren(Position::AddRight, rhs)),
                    ),
                    Node::Mul | Node::Div => (
                        Class::Mul,
                        paren(Position::MulLeft, lhs).max(paren(Position::MulRight, rhs)),
                    ),
                    _ => (
                        Class::Pow,
                        paren(Position::PowBase, lhs).max(1 + paren(Position::PowExponent, rhs)),
                    ),
                }
            }
        };
        stack.push(entry);
    }
    match stack.as_slice() {
        [(_, depth)] => Ok(*depth),
        _ => Err(EvalError::Malformed),
    }
}

// ───────────────────────────── Printing ───────────────────────────────────

/// The OpenQASM dialect an expression is written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dialect {
    Qasm2,
    Qasm3,
}

/// Why an expression cannot be written in a dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unprintable {
    /// The dialect has no such function (`arcsin` in OpenQASM 2.0); its
    /// OpenQASM 3 name.
    Function(&'static str),
    /// A number that is not finite: an OpenQASM 2.0 literal too large for
    /// binary64, which no dialect can spell.
    NonFinite,
    /// A reference with no name, or a malformed sequence. The constructors
    /// never build one.
    Malformed,
}

impl fmt::Display for Unprintable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unprintable::Function(name) => write!(f, "it uses the function '{name}'"),
            Unprintable::NonFinite => write!(f, "it contains a number too large for binary64"),
            Unprintable::Malformed => write!(f, "malformed parameter expression"),
        }
    }
}

impl Function {
    /// The function's name in `dialect`, or `None` if the dialect lacks it.
    pub(crate) fn name(self, dialect: Dialect) -> Option<&'static str> {
        Some(match (self, dialect) {
            (Function::Sin, _) => "sin",
            (Function::Cos, _) => "cos",
            (Function::Tan, _) => "tan",
            (Function::Exp, _) => "exp",
            (Function::Sqrt, _) => "sqrt",
            (Function::Ln, Dialect::Qasm2) => "ln",
            (Function::Ln, Dialect::Qasm3) => "log",
            (Function::Arcsin, Dialect::Qasm3) => "arcsin",
            (Function::Arccos, Dialect::Qasm3) => "arccos",
            (Function::Arctan, Dialect::Qasm3) => "arctan",
            (Function::Arcsin | Function::Arccos | Function::Arctan, Dialect::Qasm2) => {
                return None
            }
        })
    }
}

/// The canonical text of a finite number: the shortest decimal that reads
/// back as exactly `value`, always with a decimal point (so it is a real
/// literal in both dialects), in positional form when `1e-5 <= |value| <
/// 1e16` and in scientific form (`1.5e-7`, `1.0e16`) otherwise. Negative zero
/// keeps its sign.
pub(crate) fn fmt_number(value: f64) -> String {
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return format!("{sign}0.0");
    }
    // `{:e}` prints the shortest digits that round-trip, as `d[.ddd]e<exp>`.
    let scientific = format!("{magnitude:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    if (-5..16).contains(&exponent) {
        if exponent < 0 {
            let zeros = "0".repeat((-exponent - 1) as usize);
            format!("{sign}0.{zeros}{digits}")
        } else {
            let integer_len = exponent as usize + 1;
            if digits.len() > integer_len {
                let (integer, fraction) = digits.split_at(integer_len);
                format!("{sign}{integer}.{fraction}")
            } else {
                let zeros = "0".repeat(integer_len - digits.len());
                format!("{sign}{digits}{zeros}.0")
            }
        }
    } else {
        let (first, rest) = digits.split_at(1);
        let rest = if rest.is_empty() { "0" } else { rest };
        format!("{sign}{first}.{rest}e{exponent}")
    }
}

/// How many nodes an importer reads back from `nodes` as [`print`] writes
/// them: a negative number is written with a leading minus, which reads back
/// as the negation of its magnitude.
pub(crate) fn printed_len<R>(nodes: &[Node<R>]) -> usize {
    let negative = nodes
        .iter()
        .filter(|node| matches!(node, Node::Num(v) if v.is_sign_negative()))
        .count();
    nodes.len() + negative
}

/// A kind of reference that indexes a list of names.
pub(crate) trait Reference: Copy {
    fn index(self) -> usize;
}

impl Reference for Formal {
    fn index(self) -> usize {
        self.0
    }
}

impl Reference for Input {
    fn index(self) -> usize {
        self.0
    }
}

/// Write `nodes` as `dialect` source text into `out`, with the fewest
/// parentheses that keep the expression as it is (see [`needs_parens`]), each
/// reference by its name in `names`. The walk is iterative, so no depth of
/// expression can exhaust the stack.
pub(crate) fn print<R: Reference, S: AsRef<str>>(
    nodes: &[Node<R>],
    dialect: Dialect,
    names: &[S],
    out: &mut String,
) -> Result<(), Unprintable> {
    const NONE: usize = usize::MAX;
    // The operand(s) of every node, found by replaying the postfix sequence.
    let mut operands = vec![(NONE, NONE); nodes.len()];
    let mut pending = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        match node {
            Node::Num(_) | Node::Const(_) | Node::Ref(_) => {}
            Node::Neg | Node::Call(_) => {
                operands[i].0 = pending.pop().ok_or(Unprintable::Malformed)?;
            }
            Node::Add | Node::Sub | Node::Mul | Node::Div | Node::Pow => {
                let rhs = pending.pop().ok_or(Unprintable::Malformed)?;
                let lhs = pending.pop().ok_or(Unprintable::Malformed)?;
                operands[i] = (lhs, rhs);
            }
        }
        pending.push(i);
    }
    let root = match pending.as_slice() {
        [root] => *root,
        _ => return Err(Unprintable::Malformed),
    };
    let class = |i: usize| match nodes[i] {
        Node::Num(v) if v.is_sign_negative() => Class::Neg,
        Node::Num(_) | Node::Const(_) | Node::Ref(_) | Node::Call(_) => Class::Atom,
        Node::Neg => Class::Neg,
        Node::Pow => Class::Pow,
        Node::Mul | Node::Div => Class::Mul,
        Node::Add | Node::Sub => Class::Add,
    };

    enum Work {
        Node(usize),
        Text(&'static str),
    }
    let mut work = vec![Work::Node(root)];
    let operand = |work: &mut Vec<Work>, child: usize, position: Position| {
        if needs_parens(position, class(child)) {
            work.push(Work::Text(")"));
            work.push(Work::Node(child));
            work.push(Work::Text("("));
        } else {
            work.push(Work::Node(child));
        }
    };
    while let Some(item) = work.pop() {
        let i = match item {
            Work::Text(text) => {
                out.push_str(text);
                continue;
            }
            Work::Node(i) => i,
        };
        let (lhs, rhs) = operands[i];
        match nodes[i] {
            Node::Num(v) if !v.is_finite() => return Err(Unprintable::NonFinite),
            Node::Num(v) => out.push_str(&fmt_number(v)),
            Node::Const(c) => match (c, dialect) {
                (Constant::Pi, _) => out.push_str("pi"),
                (Constant::Tau, Dialect::Qasm3) => out.push_str("tau"),
                (Constant::Euler, Dialect::Qasm3) => out.push_str("euler"),
                // OpenQASM 2.0 has no name for them: their exact value.
                (Constant::Tau | Constant::Euler, Dialect::Qasm2) => {
                    out.push_str(&fmt_number(c.value()))
                }
            },
            Node::Ref(r) => {
                let name = names.get(r.index()).ok_or(Unprintable::Malformed)?;
                out.push_str(name.as_ref());
            }
            Node::Neg => {
                out.push('-');
                operand(&mut work, lhs, Position::NegOperand);
            }
            Node::Call(f) => {
                let name = f
                    .name(dialect)
                    .ok_or(Unprintable::Function(f.name(Dialect::Qasm3).unwrap_or("?")))?;
                out.push_str(name);
                out.push('(');
                work.push(Work::Text(")"));
                work.push(Work::Node(lhs));
            }
            Node::Add | Node::Sub | Node::Mul | Node::Div | Node::Pow => {
                let (op, left, right) = match nodes[i] {
                    Node::Add => (" + ", Position::AddLeft, Position::AddRight),
                    Node::Sub => (" - ", Position::AddLeft, Position::AddRight),
                    Node::Mul => ("*", Position::MulLeft, Position::MulRight),
                    Node::Div => ("/", Position::MulLeft, Position::MulRight),
                    _ => (
                        if dialect == Dialect::Qasm2 { "^" } else { "**" },
                        Position::PowBase,
                        Position::PowExponent,
                    ),
                };
                operand(&mut work, rhs, right);
                work.push(Work::Text(op));
                operand(&mut work, lhs, left);
            }
        }
    }
    Ok(())
}

// ───────────────────────── Gate-body expressions ──────────────────────────

/// An expression over a declared gate's formal parameters: one angle of a
/// statement in the gate's body.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FormalExpr {
    nodes: Vec<Node<Formal>>,
}

impl FormalExpr {
    /// An expression the parser has built node by node, in postfix order.
    pub(crate) fn from_nodes(nodes: Vec<Node<Formal>>) -> Self {
        FormalExpr { nodes }
    }

    pub(crate) fn nodes(&self) -> &[Node<Formal>] {
        &self.nodes
    }

    /// Evaluate with `args` bound to the formal parameters, rejecting a
    /// non-finite result: the value must be a usable angle (contract C-2).
    pub(crate) fn eval_angle(&self, args: &[f64], stack: &mut Vec<f64>) -> Result<f64, EvalError> {
        let value = evaluate(
            &self.nodes,
            |Formal(i)| args.get(i).copied().ok_or(EvalError::Unbound { index: i }),
            stack,
        )?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(EvalError::NonFinite)
        }
    }
}

// ───────────────────────── Circuit expressions ────────────────────────────

/// An angle expression over a circuit's free parameters, built by value.
///
/// Numbers, [`Constant`]s and free parameters ([`ParamExpr::param`]) combine
/// with `+ - * /`, unary `-`, [`pow`](Self::pow) and the [`Function`]s. The
/// result goes into a circuit with
/// [`ParameterizedCircuit::add_expr`](crate::ParameterizedCircuit::add_expr),
/// which validates it and returns the [`GateParam`](crate::GateParam) to pass
/// to a gate:
///
/// ```
/// use polypus_circuit::{ParamExpr, ParameterizedCircuit};
///
/// let mut qc = ParameterizedCircuit::new(1);
/// let two_beta = qc.add_expr(2.0 * ParamExpr::param(0)).unwrap();
/// let qc = qc.rx(0, two_beta);
/// assert_eq!(qc.num_params, 1);
///
/// let bound = qc.assign_parameters(&[0.25]).unwrap();
/// assert_eq!(bound.gates, [polypus_circuit::GateInstruction::Rx { qubit: 0, theta: 0.5.into() }]);
/// ```
///
/// Operations are kept exactly as built: `(a + b) + c` and `a + (b + c)` are
/// different expressions and evaluate in their own order.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamExpr {
    nodes: Vec<Node<Input>>,
}

impl ParamExpr {
    /// The circuit's free parameter at `index`: the value bound to
    /// [`GateParam::Param(index)`](crate::GateParam::Param).
    pub fn param(index: usize) -> Self {
        ParamExpr {
            nodes: vec![Node::Ref(Input(index))],
        }
    }

    /// A named constant.
    pub fn constant(constant: Constant) -> Self {
        ParamExpr {
            nodes: vec![Node::Const(constant)],
        }
    }

    /// `self` raised to the power `exponent`.
    pub fn pow(self, exponent: impl Into<ParamExpr>) -> Self {
        self.binary(exponent.into(), Node::Pow)
    }

    /// `function(self)`.
    pub fn apply(mut self, function: Function) -> Self {
        self.nodes.push(Node::Call(function));
        self
    }

    fn binary(mut self, rhs: ParamExpr, op: Node<Input>) -> Self {
        self.nodes.extend(rhs.nodes);
        self.nodes.push(op);
        self
    }

    pub(crate) fn nodes(&self) -> &[Node<Input>] {
        &self.nodes
    }

    /// An expression a parser has built node by node, in postfix order.
    pub(crate) fn from_nodes(nodes: Vec<Node<Input>>) -> Self {
        ParamExpr { nodes }
    }
}

impl From<f64> for ParamExpr {
    /// A number. A negative one is stored as the negation of its magnitude,
    /// the way `-1.5` is written.
    fn from(value: f64) -> Self {
        let nodes = if value.is_sign_negative() {
            vec![Node::Num(-value), Node::Neg]
        } else {
            vec![Node::Num(value)]
        };
        ParamExpr { nodes }
    }
}

impl From<Constant> for ParamExpr {
    fn from(constant: Constant) -> Self {
        ParamExpr::constant(constant)
    }
}

impl ops::Neg for ParamExpr {
    type Output = ParamExpr;

    fn neg(mut self) -> ParamExpr {
        self.nodes.push(Node::Neg);
        self
    }
}

/// `ParamExpr ∘ impl Into<ParamExpr>` and `f64 ∘ ParamExpr` for each binary
/// operator.
macro_rules! binary_operator {
    ($trait:ident, $method:ident, $node:expr) => {
        impl<T: Into<ParamExpr>> ops::$trait<T> for ParamExpr {
            type Output = ParamExpr;

            fn $method(self, rhs: T) -> ParamExpr {
                self.binary(rhs.into(), $node)
            }
        }

        impl ops::$trait<ParamExpr> for f64 {
            type Output = ParamExpr;

            fn $method(self, rhs: ParamExpr) -> ParamExpr {
                ParamExpr::from(self).binary(rhs, $node)
            }
        }
    };
}

binary_operator!(Add, add, Node::Add);
binary_operator!(Sub, sub, Node::Sub);
binary_operator!(Mul, mul, Node::Mul);
binary_operator!(Div, div, Node::Div);

/// A reference to an expression stored in a circuit, the payload of
/// [`GateParam::Expr`](crate::GateParam::Expr).
///
/// Only the circuit that returned it from
/// [`ParameterizedCircuit::add_expr`](crate::ParameterizedCircuit::add_expr)
/// (or a clone of that circuit) holds the expression. An id also carries a
/// checksum of the expression's content, so a circuit that is handed an id it
/// did not issue rejects it with [`CircuitError::UnknownExpression`] — unless
/// it holds an identical expression at the same position, which then means the
/// same thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Aligned like the payloads of `Fixed` and `Param`, so all three variants of
// `GateParam` share one layout and copying an angle stays two aligned words;
// a 4-byte-aligned id made binding circuits without expressions measurably
// slower.
#[repr(align(8))]
pub struct ExprId {
    index: u32,
    check: u32,
}

/// Where one stored expression lives in the arena, and what it refers to.
#[derive(Debug, Clone, Copy)]
struct Stored {
    start: usize,
    len: usize,
    check: u32,
    /// The largest free-parameter index the expression refers to.
    max_input: usize,
}

/// A circuit's expressions: every node in one vector, each expression a
/// contiguous range of it.
#[derive(Debug, Clone, Default)]
pub(crate) struct ExprArena {
    nodes: Vec<Node<Input>>,
    exprs: Vec<Stored>,
}

/// The canonical form [`ExprArena::classify`] gives an expression.
pub(crate) enum Classified {
    /// A bare reference to a free parameter.
    Param(usize),
    /// No reference at all: a constant, whose value is known now.
    Constant,
    /// Anything else; refers to free parameters up to this index.
    Expr { max_input: usize },
}

/// FNV-1a over a canonical encoding of the nodes: stable across platforms and
/// runs, so ids stay deterministic.
fn checksum(nodes: &[Node<Input>]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    let mut feed = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u32::from(byte);
            hash = hash.wrapping_mul(0x0100_0193);
        }
    };
    for node in nodes {
        match *node {
            Node::Num(v) => {
                feed(&[0]);
                feed(&v.to_bits().to_le_bytes());
            }
            Node::Const(c) => feed(&[1, c as u8]),
            Node::Ref(Input(i)) => {
                feed(&[2]);
                feed(&(i as u64).to_le_bytes());
            }
            Node::Neg => feed(&[3]),
            Node::Add => feed(&[4]),
            Node::Sub => feed(&[5]),
            Node::Mul => feed(&[6]),
            Node::Div => feed(&[7]),
            Node::Pow => feed(&[8]),
            Node::Call(f) => feed(&[9, f as u8]),
        }
    }
    hash
}

/// Map an evaluation failure of a circuit expression to the error binding
/// reports. `num_params` is the number of values supplied.
pub(crate) fn binding_error(e: EvalError, num_params: usize) -> CircuitError {
    match e {
        EvalError::DivisionByZero { .. } => CircuitError::DivisionByZero,
        EvalError::NonFinite => CircuitError::NonFiniteParam,
        EvalError::Unbound { index } => CircuitError::ParamIndexOutOfBounds { index, num_params },
        EvalError::Malformed => CircuitError::InvalidExpression {
            reason: e.to_string(),
        },
    }
}

/// The value of an expression without free parameters, as an angle.
pub(crate) fn evaluate_constant(expr: &ParamExpr) -> Result<f64, CircuitError> {
    let value = evaluate(
        expr.nodes(),
        |Input(index)| Err(EvalError::Unbound { index }),
        &mut Vec::new(),
    )
    .map_err(|e| binding_error(e, 0))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CircuitError::NonFiniteParam)
    }
}

impl ExprArena {
    /// An arena holding nothing, for circuits that cannot hold expressions.
    pub(crate) const EMPTY: ExprArena = ExprArena {
        nodes: Vec::new(),
        exprs: Vec::new(),
    };

    /// Check that `expr` may be stored in a circuit — every number finite, at
    /// most [`MAX_EXPR_NODES`] nodes, nested at most [`MAX_EXPR_DEPTH`] levels
    /// — and say which canonical form it takes.
    pub(crate) fn classify(expr: &ParamExpr) -> Result<Classified, CircuitError> {
        let nodes = expr.nodes();
        if nodes.len() > MAX_EXPR_NODES {
            return Err(CircuitError::InvalidExpression {
                reason: format!(
                    "the expression has {} nodes, more than the {MAX_EXPR_NODES} allowed",
                    nodes.len()
                ),
            });
        }
        let mut max_input = None;
        for node in nodes {
            match *node {
                Node::Num(v) if !v.is_finite() => return Err(CircuitError::NonFiniteParam),
                Node::Ref(Input(i)) => max_input = Some(max_input.map_or(i, |m: usize| m.max(i))),
                _ => {}
            }
        }
        let depth = depth(nodes).map_err(|e| binding_error(e, 0))?;
        if depth > MAX_EXPR_DEPTH {
            return Err(CircuitError::InvalidExpression {
                reason: format!(
                    "the expression nests {depth} levels deep, more than the {MAX_EXPR_DEPTH} allowed"
                ),
            });
        }
        Ok(match (nodes, max_input) {
            ([Node::Ref(Input(i))], _) => Classified::Param(*i),
            (_, None) => Classified::Constant,
            (_, Some(max_input)) => Classified::Expr { max_input },
        })
    }

    /// Store a classified expression, returning its id.
    pub(crate) fn insert(
        &mut self,
        expr: &ParamExpr,
        max_input: usize,
    ) -> Result<ExprId, CircuitError> {
        let index =
            u32::try_from(self.exprs.len()).map_err(|_| CircuitError::InvalidExpression {
                reason: format!("a circuit holds at most {} expressions", u32::MAX),
            })?;
        let nodes = expr.nodes();
        let check = checksum(nodes);
        self.exprs.push(Stored {
            start: self.nodes.len(),
            len: nodes.len(),
            check,
            max_input,
        });
        self.nodes.extend_from_slice(nodes);
        Ok(ExprId { index, check })
    }

    fn stored(&self, id: ExprId) -> Option<&Stored> {
        let stored = self.exprs.get(usize::try_from(id.index).ok()?)?;
        (stored.check == id.check).then_some(stored)
    }

    /// The nodes of expression `id`, if this arena holds it.
    pub(crate) fn nodes(&self, id: ExprId) -> Option<&[Node<Input>]> {
        let stored = self.stored(id)?;
        self.nodes.get(stored.start..stored.start + stored.len)
    }

    /// The largest free-parameter index expression `id` refers to.
    pub(crate) fn max_input(&self, id: ExprId) -> Option<usize> {
        self.stored(id).map(|s| s.max_input)
    }

    /// Evaluate expression `id` with `params` bound to the free parameters,
    /// as an angle: every referenced value and the result must be finite.
    /// Intermediate values may be infinite — `1/exp(1000)` is `0.0`.
    pub(crate) fn evaluate(
        &self,
        id: ExprId,
        params: &[f64],
        stack: &mut Vec<f64>,
    ) -> Result<f64, CircuitError> {
        let nodes = self.nodes(id).ok_or(CircuitError::UnknownExpression)?;
        let value = evaluate(
            nodes,
            |Input(i)| match params.get(i) {
                Some(v) if v.is_finite() => Ok(*v),
                Some(_) => Err(EvalError::NonFinite),
                None => Err(EvalError::Unbound { index: i }),
            },
            stack,
        )
        .map_err(|e| binding_error(e, params.len()))?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(CircuitError::NonFiniteParam)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(expr: &ParamExpr, params: &[f64]) -> Result<f64, EvalError> {
        evaluate(
            expr.nodes(),
            |Input(i)| {
                params
                    .get(i)
                    .copied()
                    .ok_or(EvalError::Unbound { index: i })
            },
            &mut Vec::new(),
        )
    }

    fn x(i: usize) -> ParamExpr {
        ParamExpr::param(i)
    }

    #[test]
    fn evaluation_follows_the_written_order() {
        // (a + b) + c and a + (b + c) differ in binary64 for these values.
        let (a, b, c) = (1e16, -1e16, 1.0);
        let left = (x(0) + x(1)) + x(2);
        let right = x(0) + (x(1) + x(2));
        assert_eq!(eval(&left, &[a, b, c]), Ok((a + b) + c));
        assert_eq!(eval(&right, &[a, b, c]), Ok(a + (b + c)));
        assert_ne!((a + b) + c, a + (b + c));
    }

    #[test]
    fn negative_numbers_are_negations() {
        assert_eq!(ParamExpr::from(-1.5).nodes(), [Node::Num(1.5), Node::Neg]);
        let zero = eval(&ParamExpr::from(-0.0), &[]).unwrap();
        assert!(zero == 0.0 && zero.is_sign_negative());
    }

    #[test]
    fn gate_body_expressions_refer_to_formal_parameters() {
        use Node::*;
        let ln = FormalExpr::from_nodes(vec![Ref(Formal(0)), Call(Function::Ln)]);
        let stack = &mut Vec::new();
        assert_eq!(ln.eval_angle(&[1.0], stack), Ok(0.0));
        assert_eq!(ln.eval_angle(&[0.0], stack), Err(EvalError::NonFinite));
        let second = FormalExpr::from_nodes(vec![Ref(Formal(2))]);
        assert_eq!(
            second.eval_angle(&[1.0], stack),
            Err(EvalError::Unbound { index: 2 })
        );
    }

    #[test]
    fn division_by_zero_names_its_position() {
        let e = x(0) / (x(1) - x(1));
        assert_eq!(
            eval(&e, &[1.0, 3.0]),
            Err(EvalError::DivisionByZero { at: 4 })
        );
        // Negative zero is zero too.
        assert_eq!(
            eval(&(x(0) / -0.0), &[1.0]),
            Err(EvalError::DivisionByZero { at: 3 })
        );
    }

    #[test]
    fn malformed_sequences_are_errors_not_panics() {
        let stack = &mut Vec::new();
        let no_ref = |_: Input| Err(EvalError::Malformed);
        assert_eq!(
            evaluate::<Input>(&[], no_ref, stack),
            Err(EvalError::Malformed)
        );
        assert_eq!(
            evaluate::<Input>(&[Node::Add], no_ref, stack),
            Err(EvalError::Malformed)
        );
        assert_eq!(
            evaluate::<Input>(&[Node::Num(1.0), Node::Num(2.0)], no_ref, stack),
            Err(EvalError::Malformed)
        );
        assert_eq!(depth::<Input>(&[Node::Neg]), Err(EvalError::Malformed));
        assert_eq!(depth::<Input>(&[]), Err(EvalError::Malformed));
    }

    #[test]
    fn depth_counts_the_nesting_of_the_written_form() {
        let depth_of = |e: ParamExpr| depth(e.nodes()).unwrap();
        // A left-associative chain does not nest.
        let mut chain = x(0);
        for i in 1..1000 {
            chain = chain + x(i);
        }
        assert_eq!(depth_of(chain), 0);
        // `a + (b + c)` needs one pair of parentheses; `a*b + c` none.
        assert_eq!(depth_of(x(0) + (x(1) + x(2))), 1);
        assert_eq!(depth_of(x(0) * x(1) + x(2)), 0);
        assert_eq!(depth_of(x(0) * (x(1) + x(2))), 1);
        // Unary minus, function arguments and exponents nest one level each.
        assert_eq!(depth_of(-x(0)), 1);
        assert_eq!(depth_of(-(x(0) + x(1))), 2);
        assert_eq!(depth_of(x(0).apply(Function::Sin)), 1);
        assert_eq!(depth_of(x(0).pow(x(1))), 1);
        // `a ** b ** c` is right-associative: no parentheses, two exponents.
        assert_eq!(depth_of(x(0).pow(x(1).pow(x(2)))), 2);
        // `(a ** b) ** c` needs them around the base, whose exponent `b` then
        // sits two levels deep.
        assert_eq!(depth_of(x(0).pow(x(1)).pow(x(2))), 2);
        // `(-a) ** b`: the base is parenthesised and negated.
        assert_eq!(depth_of((-x(0)).pow(x(1))), 2);
    }

    #[test]
    fn classification_finds_the_canonical_form() {
        assert!(matches!(
            ExprArena::classify(&x(3)),
            Ok(Classified::Param(3))
        ));
        assert!(matches!(
            ExprArena::classify(&(ParamExpr::constant(Constant::Pi) / 2.0)),
            Ok(Classified::Constant)
        ));
        assert!(matches!(
            ExprArena::classify(&(x(1) * x(4))),
            Ok(Classified::Expr { max_input: 4 })
        ));
        assert_eq!(
            ExprArena::classify(&(x(0) * f64::INFINITY)).err(),
            Some(CircuitError::NonFiniteParam)
        );
        assert_eq!(
            ExprArena::classify(&(x(0) + f64::NAN)).err(),
            Some(CircuitError::NonFiniteParam)
        );
    }

    #[test]
    fn classification_bounds_depth_and_size() {
        let mut deep = x(0);
        for _ in 0..MAX_EXPR_DEPTH {
            deep = -deep;
        }
        assert!(ExprArena::classify(&deep).is_ok());
        assert!(matches!(
            ExprArena::classify(&-deep),
            Err(CircuitError::InvalidExpression { .. })
        ));
        let mut long = x(0);
        for _ in 0..MAX_EXPR_NODES / 2 {
            long = long + 1.0;
        }
        assert!(matches!(
            ExprArena::classify(&long),
            Err(CircuitError::InvalidExpression { .. })
        ));
    }

    #[test]
    fn ids_are_checked_against_the_content() {
        let mut arena = ExprArena::default();
        let e = x(0) * 2.0;
        let id = arena.insert(&e, 0).unwrap();
        assert_eq!(arena.nodes(id), Some(e.nodes()));

        // Another arena with different content at the same position rejects it.
        let mut other = ExprArena::default();
        other.insert(&(x(0) * 3.0), 0).unwrap();
        assert_eq!(other.nodes(id), None);
        // Identical content at the same position is the same expression.
        let mut same = ExprArena::default();
        same.insert(&e, 0).unwrap();
        assert_eq!(same.nodes(id), Some(e.nodes()));
        // Out of range.
        assert_eq!(ExprArena::EMPTY.nodes(id), None);
    }

    #[test]
    fn arena_evaluation_checks_references_and_result() {
        let mut arena = ExprArena::default();
        let stack = &mut Vec::new();
        let recip = arena.insert(&(1.0 / x(0)), 0).unwrap();
        assert_eq!(arena.evaluate(recip, &[4.0], stack), Ok(0.25));
        assert_eq!(
            arena.evaluate(recip, &[0.0], stack),
            Err(CircuitError::DivisionByZero)
        );
        // A non-finite value bound to a parameter is rejected even where the
        // result would be finite.
        assert_eq!(
            arena.evaluate(recip, &[f64::INFINITY], stack),
            Err(CircuitError::NonFiniteParam)
        );
        assert_eq!(
            arena.evaluate(recip, &[], stack),
            Err(CircuitError::ParamIndexOutOfBounds {
                index: 0,
                num_params: 0
            })
        );
        // A non-finite intermediate with a finite result is accepted...
        let damped = arena.insert(&(1.0 / x(0).apply(Function::Exp)), 0).unwrap();
        assert_eq!(arena.evaluate(damped, &[1000.0], stack), Ok(0.0));
        // ... a non-finite result is not.
        let grown = arena.insert(&x(0).apply(Function::Exp), 0).unwrap();
        assert_eq!(
            arena.evaluate(grown, &[1000.0], stack),
            Err(CircuitError::NonFiniteParam)
        );
    }
}
