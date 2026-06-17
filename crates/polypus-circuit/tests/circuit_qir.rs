//! Integration tests for the QIR Base Profile export of polypus-circuit.
//!
//! These assert on the structure of the emitted LLVM IR (entry point, gate
//! intrinsics, gate decompositions, measurement recording and module flags).
//! The optional `qir_base_profile_module_is_valid_llvm` test additionally
//! parses the output with `llvm-as` when it is available on the host.

use polypus_circuit::{CircuitError, GateInstruction, Param, ParameterizedCircuit};

/// Collect just the `call void @...` instruction lines (without the leading
/// indentation), in order — handy for asserting on exact gate sequences.
fn call_lines(ir: &str) -> Vec<String> {
    ir.lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with("call void @"))
        .map(str::to_string)
        .collect()
}

#[test]
fn bell_pair_full_module() {
    let ir = ParameterizedCircuit::new(2)
        .h(0)
        .cx(0, 1)
        .measure_all()
        .to_qir_with_params(&[])
        .unwrap();

    // Header, opaque types and single entry point.
    assert!(ir.starts_with("; ModuleID = 'polypus'"));
    assert!(ir.contains("%Qubit = type opaque"));
    assert!(ir.contains("%Result = type opaque"));
    assert!(ir.contains("define void @main() #0 {\nentry:\n"));
    assert!(ir.contains("\n  ret void\n}\n"));

    // Gate calls then measurements then output recording, in order.
    assert_eq!(
        call_lines(&ir),
        vec![
            "call void @__quantum__qis__h__body(%Qubit* null)".to_string(),
            "call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*))".to_string(),
            "call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)".to_string(),
            "call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 1 to %Qubit*), %Result* inttoptr (i64 1 to %Result*))".to_string(),
            "call void @__quantum__rt__array_record_output(i64 2, i8* null)".to_string(),
            "call void @__quantum__rt__result_record_output(%Result* null, i8* null)".to_string(),
            "call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 1 to %Result*), i8* null)".to_string(),
        ]
    );

    // Entry-point attributes and module flags.
    assert!(ir.contains("\"qir_profiles\"=\"base_profile\""));
    assert!(ir.contains("\"required_num_qubits\"=\"2\""));
    assert!(ir.contains("\"required_num_results\"=\"2\""));
    assert!(ir.contains("attributes #1 = { \"irreversible\" }"));
    assert!(ir.contains("!llvm.module.flags = !{!0, !1, !2, !3}"));
    assert!(ir.contains("!0 = !{i32 1, !\"qir_major_version\", i32 1}"));
    assert!(ir.contains("!\"dynamic_qubit_management\", i1 false}"));
}

#[test]
fn rzz_is_decomposed_to_cnot_rz_cnot() {
    let ir = ParameterizedCircuit::new(2)
        .rzz(0, 1, 0.5)
        .to_qir_with_params(&[])
        .unwrap();

    let rz = format!("0x{:016X}", 0.5_f64.to_bits());
    assert_eq!(
        call_lines(&ir),
        vec![
            "call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*))".to_string(),
            format!("call void @__quantum__qis__rz__body(double {rz}, %Qubit* inttoptr (i64 1 to %Qubit*))"),
            "call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*))".to_string(),
        ]
    );
}

#[test]
fn rxx_is_decomposed_with_hadamard_basis_change() {
    let ir = ParameterizedCircuit::new(2)
        .rxx(0, 1, 0.25)
        .to_qir_with_params(&[])
        .unwrap();

    let lines = call_lines(&ir);
    // h h, cnot, rz, cnot, h h — seven operations, no measurement.
    assert_eq!(lines.len(), 7);
    assert_eq!(lines[0], "call void @__quantum__qis__h__body(%Qubit* null)");
    assert_eq!(
        lines[1],
        "call void @__quantum__qis__h__body(%Qubit* inttoptr (i64 1 to %Qubit*))"
    );
    assert!(lines[2].starts_with("call void @__quantum__qis__cnot__body"));
    assert!(lines[3].starts_with("call void @__quantum__qis__rz__body(double 0x"));
    assert!(lines[4].starts_with("call void @__quantum__qis__cnot__body"));
    assert_eq!(lines[5], "call void @__quantum__qis__h__body(%Qubit* null)");
    assert_eq!(
        lines[6],
        "call void @__quantum__qis__h__body(%Qubit* inttoptr (i64 1 to %Qubit*))"
    );
}

#[test]
fn u3_is_decomposed_zyz_in_application_order() {
    // u3(theta, phi, lam) applied left-to-right is rz(lam), ry(theta), rz(phi).
    let ir = ParameterizedCircuit::new(1)
        .u(0, 0.1, 0.2, 0.3)
        .to_qir_with_params(&[])
        .unwrap();

    let lam = format!("0x{:016X}", 0.3_f64.to_bits());
    let theta = format!("0x{:016X}", 0.1_f64.to_bits());
    let phi = format!("0x{:016X}", 0.2_f64.to_bits());
    assert_eq!(
        call_lines(&ir),
        vec![
            format!("call void @__quantum__qis__rz__body(double {lam}, %Qubit* null)"),
            format!("call void @__quantum__qis__ry__body(double {theta}, %Qubit* null)"),
            format!("call void @__quantum__qis__rz__body(double {phi}, %Qubit* null)"),
        ]
    );
}

#[test]
fn adjoint_gates_use_adj_specialization() {
    let ir = ParameterizedCircuit::new(1)
        .sdg(0)
        .tdg(0)
        .to_qir_with_params(&[])
        .unwrap();
    assert!(ir.contains("call void @__quantum__qis__s__adj(%Qubit* null)"));
    assert!(ir.contains("call void @__quantum__qis__t__adj(%Qubit* null)"));
    assert!(ir.contains("declare void @__quantum__qis__s__adj(%Qubit*)"));
}

#[test]
fn barrier_is_dropped() {
    let ir = ParameterizedCircuit::new(2)
        .h(0)
        .barrier()
        .barrier_on(&[0, 1])
        .cx(0, 1)
        .to_qir_with_params(&[])
        .unwrap();
    assert!(!ir.contains("barrier"));
    assert_eq!(call_lines(&ir).len(), 2); // only h and cnot
}

#[test]
fn no_measurement_has_no_recording_or_irreversible_attribute() {
    let ir = ParameterizedCircuit::new(1)
        .h(0)
        .to_qir_with_params(&[])
        .unwrap();
    assert!(!ir.contains("record_output"));
    assert!(!ir.contains("__quantum__qis__mz__body"));
    assert!(!ir.contains("attributes #1"));
    assert!(ir.contains("\"required_num_results\"=\"0\""));
}

#[test]
fn partial_measurement_records_only_measured_results() {
    // Measure qubit 2 -> c[0] and qubit 0 -> c[1]; classical register size 2.
    let ir = ParameterizedCircuit::new(3)
        .h(0)
        .measure(2, 0)
        .measure(0, 1)
        .to_qir_with_params(&[])
        .unwrap();

    // Results emitted in ascending result (classical-bit) index order.
    assert_eq!(
        call_lines(&ir),
        vec![
            "call void @__quantum__qis__h__body(%Qubit* null)".to_string(),
            "call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 2 to %Qubit*), %Result* null)".to_string(),
            "call void @__quantum__qis__mz__body(%Qubit* null, %Result* inttoptr (i64 1 to %Result*))".to_string(),
            "call void @__quantum__rt__array_record_output(i64 2, i8* null)".to_string(),
            "call void @__quantum__rt__result_record_output(%Result* null, i8* null)".to_string(),
            "call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 1 to %Result*), i8* null)".to_string(),
        ]
    );
    assert!(ir.contains("\"required_num_qubits\"=\"3\""));
    assert!(ir.contains("\"required_num_results\"=\"2\""));
}

#[test]
fn parameters_are_bound_into_intrinsic_angles() {
    let qc = ParameterizedCircuit::new(1).rx(0, Param(0)).rz(0, Param(1));
    let ir = qc.to_qir_with_params(&[0.7, 1.3]).unwrap();

    let rx = format!("0x{:016X}", 0.7_f64.to_bits());
    let rz = format!("0x{:016X}", 1.3_f64.to_bits());
    assert!(ir.contains(&format!(
        "call void @__quantum__qis__rx__body(double {rx}, %Qubit* null)"
    )));
    assert!(ir.contains(&format!(
        "call void @__quantum__qis__rz__body(double {rz}, %Qubit* null)"
    )));
}

#[test]
fn wrong_number_of_params_is_rejected() {
    let qc = ParameterizedCircuit::new(1).rx(0, Param(0));
    let err = qc.to_qir_with_params(&[]).unwrap_err();
    assert!(matches!(
        err,
        CircuitError::WrongNumberOfParams { expected: 1, got: 0 }
    ));
}

#[test]
fn declarations_are_emitted_once_and_sorted() {
    let ir = ParameterizedCircuit::new(2)
        .h(0)
        .h(1)
        .cx(0, 1)
        .cx(1, 0)
        .to_qir_with_params(&[])
        .unwrap();

    // Each distinct intrinsic is declared exactly once despite repeated use.
    assert_eq!(ir.matches("declare void @__quantum__qis__h__body").count(), 1);
    assert_eq!(
        ir.matches("declare void @__quantum__qis__cnot__body").count(),
        1
    );
}

/// Concrete circuits expose `to_qir()` directly (no params), mirroring
/// `to_qasm2()`.
#[test]
fn concrete_circuit_to_qir_matches_parameterized() {
    let pc = ParameterizedCircuit::new(1).rx(0, 0.5).measure(0, 0);
    let concrete = pc.assign_parameters(&[]).unwrap();
    assert_eq!(concrete.to_qir(), pc.to_qir_with_params(&[]).unwrap());
}

/// Manually constructed measurement still routes through `try_push`.
#[test]
fn measure_all_with_uneven_register_records_per_qubit() {
    let ir = ParameterizedCircuit::new(2)
        .h(0)
        .measure(0, 0)
        .measure(1, 1)
        .measure_all()
        .to_qir_with_params(&[])
        .unwrap();
    let _ = GateInstruction::MeasureAll; // keep import used regardless of feature flags
    assert!(ir.contains("\"required_num_results\"=\"2\""));
    assert!(ir.contains("call void @__quantum__rt__array_record_output(i64 2, i8* null)"));
}

/// When `llvm-as` is on PATH, prove the emitted module is well-formed LLVM IR
/// by assembling it to bitcode. Skipped (passes) when the tool is absent so
/// the suite stays dependency-free in CI without LLVM.
#[test]
fn qir_base_profile_module_is_valid_llvm() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let probe = Command::new("llvm-as").arg("--version").output();
    if probe.is_err() {
        eprintln!("skipping: llvm-as not found on PATH");
        return;
    }

    let ir = ParameterizedCircuit::new(3)
        .h(0)
        .rx(1, 0.5)
        .rzz(0, 1, 0.25)
        .rxx(1, 2, 0.75)
        .u(2, 0.1, 0.2, 0.3)
        .sdg(0)
        .cz(0, 2)
        .measure_all()
        .to_qir_with_params(&[])
        .unwrap();

    let mut child = Command::new("llvm-as")
        .args(["-o", "/dev/null", "-"])
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn llvm-as");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(ir.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "llvm-as rejected the emitted QIR:\n{}\n--- IR ---\n{ir}",
        String::from_utf8_lossy(&out.stderr)
    );
}
