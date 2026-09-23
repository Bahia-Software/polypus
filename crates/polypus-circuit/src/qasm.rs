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

use crate::custom_gate::GateDefinition;
use crate::error::CircuitError;
use crate::gate::{GateInstruction, GateParam};
use std::collections::HashMap;
use std::fmt::Write;

/// Format an angle with 12 decimal places (≥ 10 required for round-tripping
/// optimizer outputs without observable precision loss).
fn fmt_angle(value: f64) -> String {
    format!("{value:.12}")
}

/// Every gate declaration `gates` needs: the definitions of the declared gates
/// it calls and, transitively, of the declared gates their bodies call — each
/// once, in source order (a body only calls gates declared before it, so this
/// order also puts every declaration before its first use). Declarations no
/// instruction reaches are not re-emitted.
///
/// # Errors
///
/// [`CircuitError::ConflictingGateDefinitions`] if two *different* definitions
/// share a name (only possible when combining calls from separately imported
/// programs): one OpenQASM 2.0 program cannot declare both.
fn declared_gates(gates: &[GateInstruction]) -> Result<Vec<&GateDefinition>, CircuitError> {
    let mut found: HashMap<&str, &GateDefinition> = HashMap::new();
    let mut pending: Vec<&GateDefinition> = gates
        .iter()
        .filter_map(|g| match g {
            GateInstruction::Custom(call) => Some(call.definition()),
            _ => None,
        })
        .collect();
    while let Some(definition) = pending.pop() {
        if let Some(&seen) = found.get(definition.name()) {
            if !std::ptr::eq(seen, definition) && seen != definition {
                return Err(CircuitError::ConflictingGateDefinitions {
                    name: definition.name().to_string(),
                });
            }
            continue;
        }
        found.insert(definition.name(), definition);
        pending.extend(definition.callees().map(|callee| &**callee));
    }
    let mut ordered: Vec<&GateDefinition> = found.into_values().collect();
    ordered.sort_by(|a, b| a.ordinal().cmp(&b.ordinal()).then(a.name().cmp(b.name())));
    Ok(ordered)
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
    // The declarations of the gates the circuit calls (and of the gates those
    // call), verbatim and in their source order, before the registers — where
    // Qiskit's exporter puts them too.
    for definition in declared_gates(gates)? {
        out.push_str(definition.declaration());
        out.push('\n');
    }
    if num_qubits > 0 {
        let _ = writeln!(out, "qreg q[{num_qubits}];");
    }
    if num_clbits > 0 {
        let _ = writeln!(out, "creg c[{num_clbits}];");
    }

    let angle =
        |p: &GateParam| -> Result<String, CircuitError> { Ok(fmt_angle(p.resolve(params)?)) };

    for gate in gates {
        match gate {
            GateInstruction::H(q) => {
                let _ = writeln!(out, "h q[{q}];");
            }
            GateInstruction::X(q) => {
                let _ = writeln!(out, "x q[{q}];");
            }
            GateInstruction::Y(q) => {
                let _ = writeln!(out, "y q[{q}];");
            }
            GateInstruction::Z(q) => {
                let _ = writeln!(out, "z q[{q}];");
            }
            GateInstruction::S(q) => {
                let _ = writeln!(out, "s q[{q}];");
            }
            GateInstruction::T(q) => {
                let _ = writeln!(out, "t q[{q}];");
            }
            GateInstruction::Sdg(q) => {
                let _ = writeln!(out, "sdg q[{q}];");
            }
            GateInstruction::Tdg(q) => {
                let _ = writeln!(out, "tdg q[{q}];");
            }
            GateInstruction::Id(q) => {
                let _ = writeln!(out, "id q[{q}];");
            }
            GateInstruction::Rx { qubit, theta } => {
                let _ = writeln!(out, "rx({}) q[{qubit}];", angle(theta)?);
            }
            GateInstruction::Ry { qubit, theta } => {
                let _ = writeln!(out, "ry({}) q[{qubit}];", angle(theta)?);
            }
            GateInstruction::Rz { qubit, theta } => {
                let _ = writeln!(out, "rz({}) q[{qubit}];", angle(theta)?);
            }
            GateInstruction::Cx(c, t) => {
                let _ = writeln!(out, "cx q[{c}],q[{t}];");
            }
            GateInstruction::Cz(c, t) => {
                let _ = writeln!(out, "cz q[{c}],q[{t}];");
            }
            GateInstruction::Swap(q0, q1) => {
                let _ = writeln!(out, "swap q[{q0}],q[{q1}];");
            }
            GateInstruction::Rzz { q0, q1, theta } => {
                let _ = writeln!(out, "rzz({}) q[{q0}],q[{q1}];", angle(theta)?);
            }
            GateInstruction::Rxx { q0, q1, theta } => {
                let _ = writeln!(out, "rxx({}) q[{q0}],q[{q1}];", angle(theta)?);
            }
            GateInstruction::Cp { q0, q1, theta } => {
                let _ = writeln!(out, "cp({}) q[{q0}],q[{q1}];", angle(theta)?);
            }
            GateInstruction::U {
                qubit,
                theta,
                phi,
                lam,
            } => {
                let _ = writeln!(
                    out,
                    "u3({},{},{}) q[{qubit}];",
                    angle(theta)?,
                    angle(phi)?,
                    angle(lam)?
                );
            }
            GateInstruction::Sx(q) => {
                let _ = writeln!(out, "sx q[{q}];");
            }
            GateInstruction::Sxdg(q) => {
                let _ = writeln!(out, "sxdg q[{q}];");
            }
            GateInstruction::Cy(c, t) => {
                let _ = writeln!(out, "cy q[{c}],q[{t}];");
            }
            GateInstruction::Ch(c, t) => {
                let _ = writeln!(out, "ch q[{c}],q[{t}];");
            }
            GateInstruction::Csx(c, t) => {
                let _ = writeln!(out, "csx q[{c}],q[{t}];");
            }
            GateInstruction::Ccx(c0, c1, t) => {
                let _ = writeln!(out, "ccx q[{c0}],q[{c1}],q[{t}];");
            }
            GateInstruction::Cswap(c, t0, t1) => {
                let _ = writeln!(out, "cswap q[{c}],q[{t0}],q[{t1}];");
            }
            GateInstruction::Crx {
                control,
                target,
                theta,
            } => {
                let _ = writeln!(out, "crx({}) q[{control}],q[{target}];", angle(theta)?);
            }
            GateInstruction::Cry {
                control,
                target,
                theta,
            } => {
                let _ = writeln!(out, "cry({}) q[{control}],q[{target}];", angle(theta)?);
            }
            GateInstruction::Crz {
                control,
                target,
                theta,
            } => {
                let _ = writeln!(out, "crz({}) q[{control}],q[{target}];", angle(theta)?);
            }
            GateInstruction::Cu1 { q0, q1, theta } => {
                let _ = writeln!(out, "cu1({}) q[{q0}],q[{q1}];", angle(theta)?);
            }
            GateInstruction::Cu3 {
                control,
                target,
                theta,
                phi,
                lam,
            } => {
                let _ = writeln!(
                    out,
                    "cu3({},{},{}) q[{control}],q[{target}];",
                    angle(theta)?,
                    angle(phi)?,
                    angle(lam)?
                );
            }
            GateInstruction::Cu {
                control,
                target,
                theta,
                phi,
                lam,
                gamma,
            } => {
                let _ = writeln!(
                    out,
                    "cu({},{},{},{}) q[{control}],q[{target}];",
                    angle(theta)?,
                    angle(phi)?,
                    angle(lam)?,
                    angle(gamma)?
                );
            }
            GateInstruction::U0 { qubit, gamma } => {
                let _ = writeln!(out, "u0({}) q[{qubit}];", angle(gamma)?);
            }
            GateInstruction::Rccx(a, b, c) => {
                let _ = writeln!(out, "rccx q[{a}],q[{b}],q[{c}];");
            }
            GateInstruction::Rc3x(a, b, c, d) => {
                let _ = writeln!(out, "rc3x q[{a}],q[{b}],q[{c}],q[{d}];");
            }
            GateInstruction::C3x(a, b, c, d) => {
                let _ = writeln!(out, "c3x q[{a}],q[{b}],q[{c}],q[{d}];");
            }
            GateInstruction::C3sqrtx(a, b, c, d) => {
                let _ = writeln!(out, "c3sqrtx q[{a}],q[{b}],q[{c}],q[{d}];");
            }
            GateInstruction::C4x(a, b, c, d, e) => {
                let _ = writeln!(out, "c4x q[{a}],q[{b}],q[{c}],q[{d}],q[{e}];");
            }
            GateInstruction::Custom(call) => {
                let operands: Vec<String> =
                    call.qubits().iter().map(|q| format!("q[{q}]")).collect();
                if call.params().is_empty() {
                    let _ = writeln!(out, "{} {};", call.name(), operands.join(","));
                } else {
                    let angles = call
                        .params()
                        .iter()
                        .map(angle)
                        .collect::<Result<Vec<_>, _>>()?;
                    let _ = writeln!(
                        out,
                        "{}({}) {};",
                        call.name(),
                        angles.join(","),
                        operands.join(",")
                    );
                }
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
        assert_eq!(fmt_angle(std::f64::consts::FRAC_PI_2), "1.570796326795");
    }
}
