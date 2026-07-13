#![no_main]

use libfuzzer_sys::fuzz_target;
use polypus_circuit::ParameterizedCircuit;

// `ParameterizedCircuit::from_qasm2` parses arbitrary, untrusted OpenQASM 2.0.
// Whatever the bytes, it must only ever return `Ok(circuit)` or a
// `CircuitError` — never panic, never overflow the stack (Fase 1.1), and never
// allocate without bound (Fase 1.2). Arbitrary bytes are decoded lossily so the
// lexer also sees non-ASCII and replacement characters.
fuzz_target!(|data: &[u8]| {
    let src = String::from_utf8_lossy(data);
    let _ = ParameterizedCircuit::from_qasm2(&src);
});
