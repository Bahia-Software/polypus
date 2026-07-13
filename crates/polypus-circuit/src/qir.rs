//! QIR (Quantum Intermediate Representation) **Base Profile** serialization.
//!
//! Emits a self-contained LLVM IR module that conforms to the QIR Alliance
//! [Base Profile]: a single entry-point function, statically allocated qubits
//! and results, all quantum operations first and all measurements last, with
//! the measurement outcomes reported through the QIR output-recording runtime.
//! This is the most widely supported profile and the natural target for
//! Polypus circuits, which are straight-line gate sequences ending in
//! measurement.
//!
//! [Base Profile]: https://github.com/qir-alliance/qir-spec/blob/main/specification/under_development/profiles/Base_Profile.md
//!
//! ## Gate mapping
//!
//! Most instructions map one-to-one onto a standard QIS intrinsic
//! (`h`, `x`, `cnot`, `rz`, …). The QIR standard intrinsic set is smaller than
//! this crate's gate vocabulary, so a few gates are **decomposed** at emit
//! time (semantics-preserving up to an unobservable global phase):
//!
//! - `rzz(θ)` → `cnot; rz(θ); cnot`
//! - `rxx(θ)` → `h h; cnot; rz(θ); cnot; h h`
//! - `cp(θ)` → `rz(θ/2) q0; cnot; rz(−θ/2) q1; cnot; rz(θ/2) q1`
//! - `u3(θ,φ,λ)` → `rz(λ); ry(θ); rz(φ)` (ZYZ Euler decomposition)
//! - `barrier` is dropped (QIR has no barrier; it is only a scheduling hint).
//!
//! ## Angle encoding
//!
//! Angles are written as LLVM `double` hexadecimal literals (the raw IEEE-754
//! bits). LLVM only accepts decimal floating-point constants that are *exactly*
//! representable, so the hexadecimal form is the only encoding that round-trips
//! arbitrary optimizer outputs without precision loss or parser rejection.
//!
//! ## Measurement model
//!
//! Measurements are emitted after every unitary, regardless of where they
//! appear in the instruction sequence (Base Profile measurements are terminal:
//! a qubit is never operated on after measurement). Each `Measure { qubit,
//! cbit }` becomes an `mz(qubit, result=cbit)`; `MeasureAll` measures qubit `i`
//! into result `i`. The outcomes are then reported, in ascending result order,
//! as a single results array via `__quantum__rt__array_record_output` followed
//! by one `__quantum__rt__result_record_output` per measured result.

use crate::error::CircuitError;
use crate::gate::{GateInstruction, GateParam};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::io::Write as _;
use std::process::{Command, Stdio};

/// QIR major version reported in the module flags.
const QIR_MAJOR_VERSION: i32 = 1;
/// QIR minor version reported in the module flags.
const QIR_MINOR_VERSION: i32 = 0;

// Standard QIS / runtime intrinsic symbol names.
const H: &str = "__quantum__qis__h__body";
const X: &str = "__quantum__qis__x__body";
const Y: &str = "__quantum__qis__y__body";
const Z: &str = "__quantum__qis__z__body";
const S: &str = "__quantum__qis__s__body";
const S_ADJ: &str = "__quantum__qis__s__adj";
const T: &str = "__quantum__qis__t__body";
const T_ADJ: &str = "__quantum__qis__t__adj";
const RX: &str = "__quantum__qis__rx__body";
const RY: &str = "__quantum__qis__ry__body";
const RZ: &str = "__quantum__qis__rz__body";
const CNOT: &str = "__quantum__qis__cnot__body";
const CZ: &str = "__quantum__qis__cz__body";
const MZ: &str = "__quantum__qis__mz__body";

/// Format an `f64` as an LLVM IR `double` hexadecimal literal (the exact
/// IEEE-754 bit pattern, always accepted by the LLVM parser).
fn fmt_double(value: f64) -> String {
    format!("0x{:016X}", value.to_bits())
}

/// Render a statically allocated `%Qubit*` pointer for qubit index `i`.
fn qubit_ref(i: usize) -> String {
    if i == 0 {
        "%Qubit* null".to_string()
    } else {
        format!("%Qubit* inttoptr (i64 {i} to %Qubit*)")
    }
}

/// Render a statically allocated `%Result*` pointer for result index `i`.
fn result_ref(i: usize) -> String {
    if i == 0 {
        "%Result* null".to_string()
    } else {
        format!("%Result* inttoptr (i64 {i} to %Result*)")
    }
}

/// Accumulates the entry-block instructions and the (deduplicated) set of
/// intrinsic declarations they require.
struct QirWriter {
    body: String,
    decls: BTreeSet<String>,
}

impl QirWriter {
    fn new() -> Self {
        QirWriter {
            body: String::new(),
            decls: BTreeSet::new(),
        }
    }

    /// Single-qubit, no-angle intrinsic, e.g. `h`, `x`, `s__adj`.
    fn gate1(&mut self, intrinsic: &str, q: usize) {
        let _ = writeln!(self.body, "  call void @{intrinsic}({})", qubit_ref(q));
        self.decls
            .insert(format!("declare void @{intrinsic}(%Qubit*)"));
    }

    /// Single-qubit rotation intrinsic, e.g. `rx`, `ry`, `rz`.
    fn rot(&mut self, intrinsic: &str, theta: f64, q: usize) {
        let _ = writeln!(
            self.body,
            "  call void @{intrinsic}(double {}, {})",
            fmt_double(theta),
            qubit_ref(q)
        );
        self.decls
            .insert(format!("declare void @{intrinsic}(double, %Qubit*)"));
    }

    /// Two-qubit, no-angle intrinsic, e.g. `cnot`, `cz`.
    fn gate2(&mut self, intrinsic: &str, a: usize, b: usize) {
        let _ = writeln!(
            self.body,
            "  call void @{intrinsic}({}, {})",
            qubit_ref(a),
            qubit_ref(b)
        );
        self.decls
            .insert(format!("declare void @{intrinsic}(%Qubit*, %Qubit*)"));
    }
}

/// Serialize a gate sequence to a complete QIR Base Profile LLVM IR module.
///
/// `params` supplies values for any unresolved [`GateParam::Param`]; pass an
/// empty slice for fully concrete circuits. `num_clbits` is the size of the
/// implicit classical register (reported as `required_num_results`).
///
/// # Errors
///
/// - [`CircuitError::ParamIndexOutOfBounds`] if a gate references a parameter
///   index beyond `params` (only possible for manually assembled circuits).
/// - [`CircuitError::QubitAlreadyMeasured`] if the sequence violates the
///   terminal-measurement model (contract C-4). The Base Profile requires all
///   measurements last, so a gate after a measurement is rejected outright —
///   never silently reordered.
pub(crate) fn write_qir(
    num_qubits: usize,
    num_clbits: usize,
    gates: &[GateInstruction],
    params: &[f64],
) -> Result<String, CircuitError> {
    // Contract C-4: reject a gate acting on an already-measured qubit rather
    // than deferring/reordering it past the measurement below.
    if let Some(qubit) = crate::gate::terminal_measurement_violation(gates) {
        return Err(CircuitError::QubitAlreadyMeasured { qubit });
    }

    let angle = |p: &GateParam| -> Result<f64, CircuitError> { p.resolve(params) };

    let mut w = QirWriter::new();
    // Deferred measurements: result (classical bit) -> measured qubit.
    // A BTreeMap keeps results in ascending order and collapses any repeated
    // measurement of the same classical bit to its last assignment.
    let mut measurements: BTreeMap<usize, usize> = BTreeMap::new();

    for gate in gates {
        match gate {
            GateInstruction::H(q) => w.gate1(H, *q),
            GateInstruction::X(q) => w.gate1(X, *q),
            GateInstruction::Y(q) => w.gate1(Y, *q),
            GateInstruction::Z(q) => w.gate1(Z, *q),
            GateInstruction::S(q) => w.gate1(S, *q),
            GateInstruction::T(q) => w.gate1(T, *q),
            GateInstruction::Sdg(q) => w.gate1(S_ADJ, *q),
            GateInstruction::Tdg(q) => w.gate1(T_ADJ, *q),
            GateInstruction::Rx { qubit, theta } => w.rot(RX, angle(theta)?, *qubit),
            GateInstruction::Ry { qubit, theta } => w.rot(RY, angle(theta)?, *qubit),
            GateInstruction::Rz { qubit, theta } => w.rot(RZ, angle(theta)?, *qubit),
            GateInstruction::Cx(c, t) => w.gate2(CNOT, *c, *t),
            GateInstruction::Cz(c, t) => w.gate2(CZ, *c, *t),
            // rzz(θ) = cnot · rz(θ) · cnot
            GateInstruction::Rzz { q0, q1, theta } => {
                let t = angle(theta)?;
                w.gate2(CNOT, *q0, *q1);
                w.rot(RZ, t, *q1);
                w.gate2(CNOT, *q0, *q1);
            }
            // rxx(θ) = (h⊗h) · cnot · rz(θ) · cnot · (h⊗h)
            GateInstruction::Rxx { q0, q1, theta } => {
                let t = angle(theta)?;
                w.gate1(H, *q0);
                w.gate1(H, *q1);
                w.gate2(CNOT, *q0, *q1);
                w.rot(RZ, t, *q1);
                w.gate2(CNOT, *q0, *q1);
                w.gate1(H, *q0);
                w.gate1(H, *q1);
            }
            // cp(θ) = diag(1,1,1,e^{iθ}), up to global phase:
            //   rz(θ/2) q0; cnot; rz(−θ/2) q1; cnot; rz(θ/2) q1
            // (`cz; rz(θ); cz` is wrong: all-diagonal, cz² = I, collapses to
            // rz(θ) — see contract C-2 / audit item C3.)
            GateInstruction::Cp { q0, q1, theta } => {
                let t = angle(theta)?;
                w.rot(RZ, t / 2.0, *q0);
                w.gate2(CNOT, *q0, *q1);
                w.rot(RZ, -t / 2.0, *q1);
                w.gate2(CNOT, *q0, *q1);
                w.rot(RZ, t / 2.0, *q1);
            }
            // u3(θ,φ,λ) = rz(φ) · ry(θ) · rz(λ) up to global phase; applied
            // left-to-right that is rz(λ), ry(θ), rz(φ).
            GateInstruction::U {
                qubit,
                theta,
                phi,
                lam,
            } => {
                let (th, ph, la) = (angle(theta)?, angle(phi)?, angle(lam)?);
                w.rot(RZ, la, *qubit);
                w.rot(RY, th, *qubit);
                w.rot(RZ, ph, *qubit);
            }
            // Barriers are scheduling hints with no QIR representation.
            GateInstruction::Barrier(_) => {}
            GateInstruction::Measure { qubit, cbit } => {
                measurements.insert(*cbit, *qubit);
            }
            GateInstruction::MeasureAll => {
                for q in 0..num_qubits {
                    measurements.insert(q, q);
                }
            }
        }
    }

    // Measurements come after every unitary (Base Profile: terminal).
    let has_measurements = !measurements.is_empty();
    for (result, qubit) in &measurements {
        let _ = writeln!(
            w.body,
            "  call void @{MZ}({}, {})",
            qubit_ref(*qubit),
            result_ref(*result)
        );
    }
    if has_measurements {
        w.decls
            .insert(format!("declare void @{MZ}(%Qubit*, %Result*) #1"));

        // Report the outcomes as one results array, in ascending result order.
        let _ = writeln!(
            w.body,
            "  call void @__quantum__rt__array_record_output(i64 {}, i8* null)",
            measurements.len()
        );
        w.decls
            .insert("declare void @__quantum__rt__array_record_output(i64, i8*)".to_string());
        for result in measurements.keys() {
            let _ = writeln!(
                w.body,
                "  call void @__quantum__rt__result_record_output({}, i8* null)",
                result_ref(*result)
            );
        }
        w.decls
            .insert("declare void @__quantum__rt__result_record_output(%Result*, i8*)".to_string());
    }

    // ── Assemble the module ──────────────────────────────────────────────
    let mut out = String::new();
    out.push_str("; ModuleID = 'polypus'\n");
    out.push_str("source_filename = \"polypus\"\n\n");
    out.push_str("%Qubit = type opaque\n");
    out.push_str("%Result = type opaque\n\n");

    out.push_str("define void @main() #0 {\n");
    out.push_str("entry:\n");
    out.push_str(&w.body);
    out.push_str("  ret void\n");
    out.push_str("}\n\n");

    for decl in &w.decls {
        out.push_str(decl);
        out.push('\n');
    }
    out.push('\n');

    out.push_str(&format!(
        "attributes #0 = {{ \"entry_point\" \"output_labeling_schema\" \
         \"qir_profiles\"=\"base_profile\" \
         \"required_num_qubits\"=\"{num_qubits}\" \
         \"required_num_results\"=\"{num_clbits}\" }}\n"
    ));
    if has_measurements {
        out.push_str("attributes #1 = { \"irreversible\" }\n");
    }
    out.push('\n');

    out.push_str("!llvm.module.flags = !{!0, !1, !2, !3}\n\n");
    let _ = writeln!(
        out,
        "!0 = !{{i32 1, !\"qir_major_version\", i32 {QIR_MAJOR_VERSION}}}"
    );
    let _ = writeln!(
        out,
        "!1 = !{{i32 7, !\"qir_minor_version\", i32 {QIR_MINOR_VERSION}}}"
    );
    out.push_str("!2 = !{i32 1, !\"dynamic_qubit_management\", i1 false}\n");
    out.push_str("!3 = !{i32 1, !\"dynamic_result_management\", i1 false}\n");

    Ok(out)
}

/// Serialize a gate sequence to LLVM bitcode (`.bc`) by first emitting QIR
/// text and assembling it with `llvm-as`.
///
/// This keeps the crate dependency-free: no direct LLVM bindings are linked.
/// Instead, the standard LLVM assembler must be available on `PATH`.
pub(crate) fn write_qir_bitcode(
    num_qubits: usize,
    num_clbits: usize,
    gates: &[GateInstruction],
    params: &[f64],
) -> Result<Vec<u8>, CircuitError> {
    let ir = write_qir(num_qubits, num_clbits, gates, params)?;
    assemble_qir_bitcode_with("llvm-as", &ir)
}

fn assemble_qir_bitcode_with(tool: &str, ir: &str) -> Result<Vec<u8>, CircuitError> {
    let mut child = Command::new(tool)
        .args(["-o", "-", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                CircuitError::QirAssemblyToolNotFound {
                    tool: tool.to_string(),
                }
            } else {
                CircuitError::QirAssemblyFailed {
                    tool: tool.to_string(),
                    message: e.to_string(),
                }
            }
        })?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(ir.as_bytes())
            .map_err(|e| CircuitError::QirAssemblyFailed {
                tool: tool.to_string(),
                message: format!("failed to write QIR to stdin: {e}"),
            })?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| CircuitError::QirAssemblyFailed {
            tool: tool.to_string(),
            message: e.to_string(),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let suffix = if stderr.is_empty() {
            String::new()
        } else {
            format!(": {stderr}")
        };
        return Err(CircuitError::QirAssemblyFailed {
            tool: tool.to_string(),
            message: format!("non-zero exit status {}{suffix}", output.status),
        });
    }

    Ok(output.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_is_exact_ieee754_hex() {
        assert_eq!(fmt_double(1.0), "0x3FF0000000000000");
        assert_eq!(fmt_double(0.5), "0x3FE0000000000000");
        assert_eq!(fmt_double(0.0), "0x0000000000000000");
        assert_eq!(
            fmt_double(std::f64::consts::FRAC_PI_2),
            format!("0x{:016X}", std::f64::consts::FRAC_PI_2.to_bits())
        );
    }

    #[test]
    fn pointers_use_null_for_index_zero() {
        assert_eq!(qubit_ref(0), "%Qubit* null");
        assert_eq!(qubit_ref(2), "%Qubit* inttoptr (i64 2 to %Qubit*)");
        assert_eq!(result_ref(0), "%Result* null");
        assert_eq!(result_ref(1), "%Result* inttoptr (i64 1 to %Result*)");
    }

    #[test]
    fn bell_pair_has_expected_shape() {
        let gates = vec![
            GateInstruction::H(0),
            GateInstruction::Cx(0, 1),
            GateInstruction::MeasureAll,
        ];
        let ir = write_qir(2, 2, &gates, &[]).unwrap();

        assert!(ir.contains("define void @main() #0 {"));
        assert!(ir.contains("call void @__quantum__qis__h__body(%Qubit* null)"));
        assert!(ir.contains(
            "call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*))"
        ));
        assert!(ir.contains("call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)"));
        assert!(ir.contains("call void @__quantum__rt__array_record_output(i64 2, i8* null)"));
        assert!(ir.contains("\"required_num_qubits\"=\"2\""));
        assert!(ir.contains("\"required_num_results\"=\"2\""));
        assert!(ir.contains("attributes #1 = { \"irreversible\" }"));
        assert!(ir
            .trim_end()
            .ends_with("!3 = !{i32 1, !\"dynamic_result_management\", i1 false}"));
    }

    #[test]
    fn no_measurement_omits_recording_and_irreversible_attr() {
        let gates = vec![GateInstruction::H(0)];
        let ir = write_qir(1, 0, &gates, &[]).unwrap();
        assert!(!ir.contains("record_output"));
        assert!(!ir.contains("attributes #1"));
        assert!(!ir.contains("mz__body"));
        assert!(ir.contains("\"required_num_results\"=\"0\""));
    }

    #[test]
    fn missing_assembler_reports_tool_not_found() {
        let err = assemble_qir_bitcode_with(
            "polypus-llvm-as-does-not-exist",
            "define void @main() {\nentry:\n  ret void\n}\n",
        )
        .unwrap_err();
        assert!(matches!(err, CircuitError::QirAssemblyToolNotFound { .. }));
    }
}
