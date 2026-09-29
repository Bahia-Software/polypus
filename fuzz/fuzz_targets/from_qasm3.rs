#![no_main]

use libfuzzer_sys::fuzz_target;
use polypus_circuit::{CircuitError, ParameterizedCircuit};

// `ParameterizedCircuit::from_qasm3` parses arbitrary, untrusted programs in
// the OpenQASM 3 profile. Whatever the bytes, it must only ever return
// `Ok(circuit)` or a `CircuitError` — never panic, never overflow the stack,
// and never work or allocate beyond its budgets. Arbitrary bytes are decoded
// lossily so the lexer also sees non-ASCII and replacement characters.
//
// Whatever it accepts must round-trip: the export imports again, and exporting
// that is byte-identical (the export is canonical, so it is a fixed point from
// the first step), parameter names included. The only export that may fail is
// one that would pass an importer budget — a register broadcast written out
// statement by statement can outgrow the program it came from.
//
// The circuit is then bound to a few finite values; binding may fail (division
// by zero, a non-finite angle), never panic. Circuits are not simulated.
fuzz_target!(|data: &[u8]| {
    let src = String::from_utf8_lossy(data);
    let Ok(circuit) = ParameterizedCircuit::from_qasm3(&src) else {
        return;
    };
    match circuit.to_qasm3() {
        Ok(exported) => {
            let reimported = ParameterizedCircuit::from_qasm3(&exported)
                .expect("the export must import again");
            assert_eq!(
                reimported.to_qasm3().unwrap(),
                exported,
                "export is not a fixed point"
            );
            assert_eq!(reimported.param_names(), circuit.param_names());
        }
        Err(CircuitError::ExportLimit { .. }) => {}
        Err(e) => panic!("an imported circuit must export: {e}"),
    }
    for value in [0.0, 0.5, -1.25] {
        let values = vec![value; circuit.num_params];
        if circuit.assign_parameters(&values).is_ok() {
            let _ = circuit.to_qasm3_with_params(&values);
            let _ = circuit.to_qasm2_with_params(&values);
        }
    }
});
