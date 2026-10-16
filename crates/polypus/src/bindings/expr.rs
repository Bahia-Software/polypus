//! Angle expressions built in Python: `polypus.ParamExpr`, the functions
//! `polypus.sin` … `polypus.sqrt` and the constants `polypus.pi`, `tau` and
//! `euler`.
//!
//! A thin wrapper over [`polypus_circuit::ParamExpr`]. The operators build the
//! native expression; a circuit builder stores it with
//! [`ParameterizedCircuit::add_expr`](polypus_circuit::ParameterizedCircuit::add_expr),
//! which validates it, so contract C-2 applies as for an imported expression.

use polypus_circuit::expr::{Constant, Function};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use super::circuit::Param;

/// An angle expression over a circuit's free parameters.
///
/// Built from `polypus.Param`, numbers and the constants `pi`, `tau` and
/// `euler` with `+ - * / **`, unary minus and the functions `sin`, `cos`,
/// `tan`, `arcsin`, `arccos`, `arctan`, `exp`, `log` and `sqrt`. Any angle
/// argument of a `polypus.Circuit` builder accepts one; binding evaluates it
/// exactly as written, without reassociation.
///
/// ```python
/// beta = polypus.Param(0)
/// qc = polypus.Circuit(1).rx(0, 2 * beta)
/// ```
#[pyclass(module = "polypus", name = "ParamExpr", frozen, skip_from_py_object)]
pub struct ParamExpr {
    pub(crate) inner: polypus_circuit::ParamExpr,
}

impl From<polypus_circuit::ParamExpr> for ParamExpr {
    fn from(inner: polypus_circuit::ParamExpr) -> Self {
        ParamExpr { inner }
    }
}

/// An operand of an expression: a number, a `Param` or a `ParamExpr`.
pub(crate) struct Operand(pub(crate) polypus_circuit::ParamExpr);

impl<'a, 'py> FromPyObject<'a, 'py> for Operand {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(p) = ob.extract::<PyRef<'_, Param>>() {
            return Ok(Operand(polypus_circuit::ParamExpr::param(p.index)));
        }
        if let Ok(e) = ob.extract::<PyRef<'_, ParamExpr>>() {
            return Ok(Operand(e.inner.clone()));
        }
        if let Ok(v) = ob.extract::<f64>() {
            return Ok(Operand(v.into()));
        }
        Err(PyTypeError::new_err(
            "operand must be a number, polypus.Param or polypus.ParamExpr",
        ))
    }
}

/// `base ** exponent`; the three-argument `pow` has no meaning here.
pub(crate) fn pow(
    base: polypus_circuit::ParamExpr,
    exponent: polypus_circuit::ParamExpr,
    modulo: Option<&Bound<'_, PyAny>>,
) -> PyResult<ParamExpr> {
    if modulo.is_some() {
        return Err(PyTypeError::new_err(
            "pow() with a modulus is not supported for parameter expressions",
        ));
    }
    Ok(base.pow(exponent).into())
}

#[pymethods]
impl ParamExpr {
    fn __add__(&self, rhs: Operand) -> Self {
        (self.inner.clone() + rhs.0).into()
    }

    fn __radd__(&self, lhs: Operand) -> Self {
        (lhs.0 + self.inner.clone()).into()
    }

    fn __sub__(&self, rhs: Operand) -> Self {
        (self.inner.clone() - rhs.0).into()
    }

    fn __rsub__(&self, lhs: Operand) -> Self {
        (lhs.0 - self.inner.clone()).into()
    }

    fn __mul__(&self, rhs: Operand) -> Self {
        (self.inner.clone() * rhs.0).into()
    }

    fn __rmul__(&self, lhs: Operand) -> Self {
        (lhs.0 * self.inner.clone()).into()
    }

    fn __truediv__(&self, rhs: Operand) -> Self {
        (self.inner.clone() / rhs.0).into()
    }

    fn __rtruediv__(&self, lhs: Operand) -> Self {
        (lhs.0 / self.inner.clone()).into()
    }

    fn __pow__(&self, exponent: Operand, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        pow(self.inner.clone(), exponent.0, modulo)
    }

    fn __rpow__(&self, base: Operand, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        pow(base.0, self.inner.clone(), modulo)
    }

    fn __neg__(&self) -> Self {
        (-self.inner.clone()).into()
    }

    /// Equal when built by the same operations on the same operands:
    /// `2 * a == 2 * a`, but `a + b != b + a`.
    fn __eq__(&self, other: PyRef<'_, Self>) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

macro_rules! functions {
    ($($name:ident, $py_name:literal => $function:expr, $doc:literal;)*) => {
        $(
            #[doc = $doc]
            #[pyfunction(name = $py_name)]
            fn $name(x: Operand) -> ParamExpr {
                x.0.apply($function).into()
            }
        )*

        fn add_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)*
            Ok(())
        }
    };
}

functions! {
    py_sin, "sin" => Function::Sin, "`sin(x)` as a parameter expression.";
    py_cos, "cos" => Function::Cos, "`cos(x)` as a parameter expression.";
    py_tan, "tan" => Function::Tan, "`tan(x)` as a parameter expression.";
    py_arcsin, "arcsin" => Function::Arcsin, "`arcsin(x)` as a parameter expression.";
    py_arccos, "arccos" => Function::Arccos, "`arccos(x)` as a parameter expression.";
    py_arctan, "arctan" => Function::Arctan, "`arctan(x)` as a parameter expression.";
    py_exp, "exp" => Function::Exp, "`exp(x)` as a parameter expression.";
    py_log, "log" => Function::Ln, "The natural logarithm `log(x)` as a parameter expression.";
    py_sqrt, "sqrt" => Function::Sqrt, "`sqrt(x)` as a parameter expression.";
}

/// Add `ParamExpr`, its functions and the constants to the `polypus` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ParamExpr>()?;
    add_functions(m)?;
    for (name, constant) in [
        ("pi", Constant::Pi),
        ("tau", Constant::Tau),
        ("euler", Constant::Euler),
    ] {
        m.add(
            name,
            ParamExpr::from(polypus_circuit::ParamExpr::constant(constant)),
        )?;
    }
    Ok(())
}
