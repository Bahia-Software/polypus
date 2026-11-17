//! What the four entry points share to prepare a run.

/// Why `run_quantum_circuit(..., infrastructure="qmio")` rejects a Qiskit
/// `QuantumCircuit`; `backend_compatibility` reports the same text.
pub(super) const QMIO_REJECTS_QISKIT: &str =
    "the 'qmio' infrastructure runs entirely in Rust (GIL-free) and cannot \
     serialize a Qiskit QuantumCircuit; pass a polypus.Circuit or an OpenQASM \
     2.0 string";

/// Why `qml.train` and `qml.predict` reject `infrastructure="qmio"`.
pub(super) const QMIO_REJECTS_QML: &str =
    "the 'qmio' infrastructure runs entirely in Rust (GIL-free) and cannot \
     serialize the Qiskit circuits of a QML model";

/// Why `run_quantum_circuit(..., backend="polypus")` rejects a Qiskit
/// `QuantumCircuit`; `backend_compatibility` reports the same text.
pub(super) const NATIVE_REJECTS_QISKIT: &str =
    "the native 'polypus' backend cannot execute a Qiskit QuantumCircuit; \
     pass a polypus.Circuit or an OpenQASM 2.0 string, or use backend=\"aer\"";
