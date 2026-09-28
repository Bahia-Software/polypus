//! Angle expressions, stored flat.
//!
//! The body of a declared gate computes its angles from the gate's formal
//! parameters ([`Formal`] references) with these expressions.
//!
//! An expression is a sequence of nodes in postfix order: both operands of an
//! operator come before it, the left one first. Evaluation is a single loop
//! over that sequence with a value stack. It performs exactly the
//! floating-point operations the expression spells, in source order, with no
//! reassociation, and it never recurses, however deep the expression is.
//! Cloning or dropping an expression is cloning or dropping a flat vector.

use std::fmt;

// ─────────────────────────── Vocabulary ───────────────────────────────────

/// A function of one argument that an expression can apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Function {
    Sin,
    Cos,
    Tan,
    Exp,
    /// The natural logarithm (`ln` in OpenQASM 2.0).
    Ln,
    Sqrt,
}

impl Function {
    pub(crate) fn apply(self, v: f64) -> f64 {
        match self {
            Function::Sin => v.sin(),
            Function::Cos => v.cos(),
            Function::Tan => v.tan(),
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
pub(crate) enum Constant {
    /// π (`pi`).
    Pi,
}

impl Constant {
    /// The constant's `f64` value.
    pub(crate) fn value(self) -> f64 {
        match self {
            Constant::Pi => std::f64::consts::PI,
        }
    }
}

/// A reference to a declared gate's formal parameter, by position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Formal(pub(crate) usize);

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
    /// The value is `NaN` or infinite.
    NonFinite,
    /// A reference to a value that was not supplied.
    Unbound { index: usize },
    /// Not a well-formed postfix sequence. The parser never builds one.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(nodes: &[Node<Formal>], args: &[f64]) -> Result<f64, EvalError> {
        evaluate(
            nodes,
            |Formal(i)| args.get(i).copied().ok_or(EvalError::Unbound { index: i }),
            &mut Vec::new(),
        )
    }

    #[test]
    fn evaluation_follows_the_written_order() {
        use Node::*;
        // (a + b) + c and a + (b + c) differ in binary64 for these values.
        let (a, b, c) = (1e16, -1e16, 1.0);
        let args = [a, b, c];
        let left = [Ref(Formal(0)), Ref(Formal(1)), Add, Ref(Formal(2)), Add];
        let right = [Ref(Formal(0)), Ref(Formal(1)), Ref(Formal(2)), Add, Add];
        assert_eq!(eval(&left, &args), Ok((a + b) + c));
        assert_eq!(eval(&right, &args), Ok(a + (b + c)));
        assert_ne!((a + b) + c, a + (b + c));
    }

    #[test]
    fn division_by_zero_names_its_position() {
        use Node::*;
        let nodes = [Num(1.0), Ref(Formal(0)), Ref(Formal(0)), Sub, Div];
        assert_eq!(
            eval(&nodes, &[3.0]),
            Err(EvalError::DivisionByZero { at: 4 })
        );
        // Negative zero is zero too.
        let nodes = [Num(1.0), Num(0.0), Neg, Div];
        assert_eq!(eval(&nodes, &[]), Err(EvalError::DivisionByZero { at: 3 }));
    }

    #[test]
    fn unbound_references_and_malformed_sequences_are_errors_not_panics() {
        use Node::*;
        assert_eq!(
            eval(&[Ref(Formal(2))], &[1.0]),
            Err(EvalError::Unbound { index: 2 })
        );
        assert_eq!(eval(&[], &[]), Err(EvalError::Malformed));
        assert_eq!(eval(&[Add], &[]), Err(EvalError::Malformed));
        assert_eq!(eval(&[Num(1.0), Num(2.0)], &[]), Err(EvalError::Malformed));
    }

    #[test]
    fn angles_must_be_finite() {
        use Node::*;
        let ln = FormalExpr::from_nodes(vec![Ref(Formal(0)), Call(Function::Ln)]);
        let stack = &mut Vec::new();
        assert_eq!(ln.eval_angle(&[1.0], stack), Ok(0.0));
        assert_eq!(ln.eval_angle(&[0.0], stack), Err(EvalError::NonFinite));
    }
}
