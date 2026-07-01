//! OpenQASM 2.0 serialization.
//!
//! Emits the standard header (`OPENQASM 2.0; include "qelib1.inc";`), one
//! `qreg`/`creg` declaration pair, and one statement per instruction using the
//! standard `qelib1.inc` gate names. Angle values are written with 12 decimal
//! places.
//!
//! Note: `rzz`/`rxx` are part of Qiskit's `qelib1.inc` (and accepted by
//! `QuantumCircuit.from_qasm_str`). Strict parsers limited to the original
//! paper version of `qelib1.inc` may need Qiskit's
//! `qasm2.LEGACY_CUSTOM_INSTRUCTIONS` to recognise them.

use crate::error::CircuitError;
use crate::gate::{GateInstruction, GateParam};
use std::fmt::Write;

/// Format an angle with 12 decimal places (≥ 10 required for round-tripping
/// optimizer outputs without observable precision loss).
fn fmt_angle(value: f64) -> String {
    format!("{value:.12}")
}

/// Serialize a gate sequence to a complete OpenQASM 2.0 program.
///
/// `params` supplies values for any unresolved [`GateParam::Param`]; pass an
/// empty slice for fully concrete circuits.
pub(crate) fn write_qasm2(
    num_qubits: usize,
    num_clbits: usize,
    gates: &[GateInstruction],
    params: &[f64],
) -> Result<String, CircuitError> {
    let mut out = String::new();
    out.push_str("OPENQASM 2.0;\n");
    out.push_str("include \"qelib1.inc\";\n");
    if num_qubits > 0 {
        let _ = writeln!(out, "qreg q[{num_qubits}];");
    }
    if num_clbits > 0 {
        let _ = writeln!(out, "creg c[{num_clbits}];");
    }

    let angle = |p: &GateParam| -> Result<String, CircuitError> {
        Ok(fmt_angle(p.resolve(params)?))
    };

    for gate in gates {
        match gate {
            GateInstruction::H(q) => { let _ = writeln!(out, "h q[{q}];"); }
            GateInstruction::X(q) => { let _ = writeln!(out, "x q[{q}];"); }
            GateInstruction::Y(q) => { let _ = writeln!(out, "y q[{q}];"); }
            GateInstruction::Z(q) => { let _ = writeln!(out, "z q[{q}];"); }
            GateInstruction::S(q) => { let _ = writeln!(out, "s q[{q}];"); }
            GateInstruction::T(q) => { let _ = writeln!(out, "t q[{q}];"); }
            GateInstruction::Sdg(q) => { let _ = writeln!(out, "sdg q[{q}];"); }
            GateInstruction::Tdg(q) => { let _ = writeln!(out, "tdg q[{q}];"); }
            GateInstruction::Rx { qubit, theta } => {
                let _ = writeln!(out, "rx({}) q[{qubit}];", angle(theta)?);
            }
            GateInstruction::Ry { qubit, theta } => {
                let _ = writeln!(out, "ry({}) q[{qubit}];", angle(theta)?);
            }
            GateInstruction::Rz { qubit, theta } => {
                let _ = writeln!(out, "rz({}) q[{qubit}];", angle(theta)?);
            }
            GateInstruction::Cx(c, t) => { let _ = writeln!(out, "cx q[{c}],q[{t}];"); }
            GateInstruction::Cz(c, t) => { let _ = writeln!(out, "cz q[{c}],q[{t}];"); }
            GateInstruction::Rzz { q0, q1, theta } => {
                let _ = writeln!(out, "rzz({}) q[{q0}],q[{q1}];", angle(theta)?);
            }
            GateInstruction::Rxx { q0, q1, theta } => {
                let _ = writeln!(out, "rxx({}) q[{q0}],q[{q1}];", angle(theta)?);
            }
            GateInstruction::Cp { q0, q1, theta } => {
                let _ = writeln!(out, "cp({}) q[{q0}],q[{q1}];", angle(theta)?);
            }
            GateInstruction::U { qubit, theta, phi, lam } => {
                let _ = writeln!(
                    out,
                    "u3({},{},{}) q[{qubit}];",
                    angle(theta)?,
                    angle(phi)?,
                    angle(lam)?
                );
            }
            GateInstruction::Barrier(qubits) => {
                if qubits.is_empty() {
                    out.push_str("barrier q;\n");
                } else {
                    let args: Vec<String> = qubits.iter().map(|q| format!("q[{q}]")).collect();
                    let _ = writeln!(out, "barrier {};", args.join(","));
                }
            }
            GateInstruction::Measure { qubit, cbit } => {
                let _ = writeln!(out, "measure q[{qubit}] -> c[{cbit}];");
            }
            GateInstruction::MeasureAll => {
                if num_clbits == num_qubits {
                    out.push_str("measure q -> c;\n");
                } else {
                    // Register sizes differ (mixed Measure/MeasureAll usage):
                    // `measure q -> c;` would be invalid, expand per qubit.
                    for q in 0..num_qubits {
                        let _ = writeln!(out, "measure q[{q}] -> c[{q}];");
                    }
                }
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::fmt_angle;

    #[test]
    fn angles_have_at_least_ten_decimals() {
        assert_eq!(fmt_angle(0.5), "0.500000000000");
        assert_eq!(fmt_angle(-1.0), "-1.000000000000");
        assert_eq!(
            fmt_angle(std::f64::consts::FRAC_PI_2),
            "1.570796326795"
        );
    }
}
