#![no_main]

use libfuzzer_sys::fuzz_target;
use polypus_circuit::ParameterizedCircuit;

// `ParameterizedCircuit::from_qasm2` parses arbitrary, untrusted OpenQASM 2.0.
// Whatever the bytes, it must only ever return `Ok(circuit)` or a
// `CircuitError` — never panic, never overflow the stack (Fase 1.1), and never
// allocate without bound (Fase 1.2). Arbitrary bytes are decoded lossily so the
// lexer also sees non-ASCII and replacement characters.
//
// Whatever it accepts must also round-trip (contract C-2): the export imports
// again and exporting that is byte-identical — the export is a fixed point,
// `gate` declarations and calls of declared gates included. (The circuits
// themselves need not be equal: angles are exported with 12 decimals, and
// declarations nothing uses are not re-emitted.)
fuzz_target!(|data: &[u8]| {
    let src = String::from_utf8_lossy(data);
    let Ok(circuit) = ParameterizedCircuit::from_qasm2(&src) else {
        return;
    };
    let exported = circuit
        .to_qasm2_with_params(&[])
        .expect("an imported circuit must export");
    let reimported =
        ParameterizedCircuit::from_qasm2(&exported).expect("the export must import again");
    assert_eq!(
        reimported.to_qasm2_with_params(&[]).unwrap(),
        exported,
        "export is not a fixed point"
    );
});
