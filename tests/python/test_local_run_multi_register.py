"""
Regression tests for issue #206 — Aer counts of circuits with several
``ClassicalRegister`` must reach the seam as flat bitstrings (contract C-3).

Qiskit's ``Result.get_counts()`` space-separates the key per register
(``"0 1 0"``), which ``validate_run_results`` rightly rejects. The direct
``Local.run_qcs`` cases need no Rust build; the ``polypus`` cases are marked
'integration' like ``test_local_run.py``.
"""

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm2


def _three_register_circuit(teleport: bool) -> QuantumCircuit:
    """Three one-bit registers, each measuring its own qubit. With
    ``teleport=False`` only ``x(0)`` is applied, so the outcome is fixed."""
    qr = QuantumRegister(3, "q")
    crz = ClassicalRegister(1, "crz")
    crx = ClassicalRegister(1, "crx")
    res = ClassicalRegister(1, "res")
    qc = QuantumCircuit(qr, crz, crx, res)
    qc.x(0)
    if teleport:
        qc.h(1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.h(0)
    qc.measure(0, crz[0])
    qc.measure(1, crx[0])
    qc.measure(2, res[0])
    return qc


def _assert_flat_bitstrings(counts: dict, width: int, shots: int) -> None:
    for key in counts:
        assert len(key) == width and set(key) <= {"0", "1"}, (
            f"non-bitstring key {key!r}"
        )
    assert sum(counts.values()) == shots


class TestLocalRunQcsMultiRegister:
    """``polypus_python.local.Local.run_qcs`` — the Python side of the seam."""

    def test_keys_are_flat_bitstrings(self):
        from polypus_python.local import Local

        shots = 200
        [counts] = Local().run_qcs(
            qcs=[_three_register_circuit(teleport=True)], shots=shots, seed=7
        )
        _assert_flat_bitstrings(counts, width=3, shots=shots)

    def test_register_concatenation_order(self):
        """``crz`` (qubit 0 → 1) is the first-declared register, so it holds
        clbit 0 — the rightmost character (C-3 little-endian)."""
        from polypus_python.local import Local

        shots = 50
        [counts] = Local().run_qcs(
            qcs=[_three_register_circuit(teleport=False)], shots=shots
        )
        assert counts == {"001": shots}


@pytest.mark.integration
class TestRunQuantumCircuitMultiRegister:
    """End to end through ``polypus.run_quantum_circuit`` and
    ``validate_run_results``."""

    def test_aer_accepts_multi_register_counts(self):
        import polypus

        shots = 100
        result = polypus.run_quantum_circuit(
            _three_register_circuit(teleport=True),
            shots=shots,
            infrastructure="local",
        )
        _assert_flat_bitstrings(result.counts[0], width=3, shots=shots)

    def test_key_format_matches_native_backend(self):
        """The same circuit gives the same key on Aer and on the native
        backend (which flattens QASM2 ``creg``s in declaration order)."""
        import polypus

        qc = _three_register_circuit(teleport=False)
        shots = 50
        aer = polypus.run_quantum_circuit(qc, shots=shots, infrastructure="local")
        native = polypus.run_quantum_circuit(
            polypus.Circuit.from_qasm2(qasm2.dumps(qc)),
            shots=shots,
            infrastructure="local",
            backend="polypus",
        )
        assert aer.counts[0] == native.counts[0] == {"001": shots}
