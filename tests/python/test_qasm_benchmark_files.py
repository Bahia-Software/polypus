"""
End to end on real benchmark files: parse with Polypus, re-emit, run on Aer.

The fixtures under ``data/qasm_benchmarks`` are unmodified circuits from
QASMBench and MQT Bench (see the README there). For each one:

- Polypus imports the file and re-emits it; Qiskit parses the original and the
  re-emitted text into circuits with the same instruction counts per name, the
  same size and the same depth — Polypus hands a backend the same program;
- both run on Aer (same transpilation, same seed) with identical counts;
- the native simulator reproduces Qiskit's statevector of the original.

Plus: every gate Qiskit's own exporter writes as a ``gate`` block (``ryy``,
``rzx``, ``ecr``, ``iswap``, ``xx_plus_yy``, multi-controlled gates, ...) goes
through the same checks, straight from ``qiskit.qasm2.dumps``.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data" / "qasm_benchmarks"
FIXTURES = sorted(DATA.rglob("*.qasm"))
SEED = 777
SHOTS = 4000


def _aer_counts(qasm):
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    qc = QuantumCircuit.from_qasm_str(qasm)
    if qc.num_clbits == 0:
        # A file with no measurement (some MQT Bench levels): measure every
        # qubit, on both sides alike, so there are counts to compare.
        qc.measure_all()
    qc = transpile(qc, sim, optimization_level=0)
    counts = sim.run(qc, shots=SHOTS, seed_simulator=SEED).result().get_counts()
    # Several classical registers print as space-separated groups; Polypus
    # flattens registers in declaration order, which yields the same bits.
    out = Counter()
    for key, value in counts.items():
        out[key.replace(" ", "")] += value
    return dict(out)


def _qiskit_statevector(qasm):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit.from_qasm_str(qasm)
    qc.remove_final_measurements()
    return Statevector(qc).data


def _assert_same_program(src):
    """Polypus re-emits `src` as the program Qiskit reads from `src`."""
    import polypus
    from qiskit import QuantumCircuit

    emitted = polypus.Circuit.from_qasm2(src).to_qasm2()
    original = QuantumCircuit.from_qasm_str(src)
    through_polypus = QuantumCircuit.from_qasm_str(emitted)
    assert through_polypus.count_ops() == original.count_ops()
    assert through_polypus.size() == original.size()
    assert through_polypus.depth() == original.depth()
    assert through_polypus.num_qubits == original.num_qubits
    return emitted


def test_fixtures_are_present():
    assert len(FIXTURES) >= 5, FIXTURES


@pytest.mark.parametrize("path", FIXTURES, ids=[p.name for p in FIXTURES])
class TestBenchmarkFile:
    def test_reemits_the_same_program(self, path):
        _assert_same_program(path.read_text())

    def test_reemitted_text_runs_on_aer_with_identical_counts(self, path):
        src = path.read_text()
        emitted = _assert_same_program(src)
        counts = _aer_counts(emitted)
        assert sum(counts.values()) == SHOTS
        assert counts == _aer_counts(src)

    def test_native_simulator_matches_qiskit(self, path):
        import polypus

        src = path.read_text()
        native = polypus.statevector(polypus.Circuit.from_qasm2(src))
        np.testing.assert_allclose(native, _qiskit_statevector(src), atol=1e-10)

    def test_reemission_is_a_fixed_point(self, path):
        import polypus

        once = polypus.Circuit.from_qasm2(path.read_text()).to_qasm2()
        assert polypus.Circuit.from_qasm2(once).to_qasm2() == once


# ─────────────────────────────────────────────────────────────────────────────
# Gates Qiskit's exporter writes as `gate` blocks
# ─────────────────────────────────────────────────────────────────────────────


def _declared_gate_circuits():
    """(id, circuit) pairs whose `qasm2.dumps` declares a non-qelib1 gate."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import (
        C3XGate,
        CCZGate,
        CSdgGate,
        CSGate,
        DCXGate,
        ECRGate,
        MCPhaseGate,
        RC3XGate,
        RGate,
        RYYGate,
        RZXGate,
        XXMinusYYGate,
        XXPlusYYGate,
        iSwapGate,
    )

    cases = [
        ("ryy", RYYGate(0.37), [2, 0]),
        ("rzx", RZXGate(-1.21), [1, 3]),
        ("ecr", ECRGate(), [3, 1]),
        ("iswap", iSwapGate(), [0, 2]),
        ("dcx", DCXGate(), [2, 1]),
        ("xx_plus_yy", XXPlusYYGate(0.37, -0.61), [1, 0]),
        ("xx_minus_yy", XXMinusYYGate(2.03, 0.4), [3, 2]),
        ("r", RGate(0.37, 1.1), [2]),
        ("ccz", CCZGate(), [3, 0, 2]),
        ("cs", CSGate(), [0, 3]),
        ("csdg", CSdgGate(), [2, 1]),
        ("rcccx", RC3XGate(), [0, 1, 2, 3]),
        ("mcx", C3XGate(), [3, 1, 0, 2]),
        ("mcphase", MCPhaseGate(0.73, 3), [1, 3, 0, 2]),
    ]
    out = []
    for name, gate, qubits in cases:
        qc = QuantumCircuit(4, 4)
        for q in range(4):
            qc.ry(0.3 + 0.4 * q, q)
            qc.rz(0.2 + 0.5 * q, q)
        qc.append(gate, qubits)
        qc.measure(range(4), range(4))
        out.append((name, qc))
    return out


_DECLARED = _declared_gate_circuits()


@pytest.mark.parametrize("qc", [c for _, c in _DECLARED], ids=[n for n, _ in _DECLARED])
def test_qiskit_declared_gates_survive_import_and_export(qc):
    import polypus
    from qiskit import qasm2

    src = qasm2.dumps(qc)
    assert "\ngate " in src, "the exporter no longer declares this gate"
    emitted = _assert_same_program(src)
    # The declarations come back verbatim, not expanded.
    for line in src.splitlines():
        if line.startswith("gate "):
            assert line in emitted.splitlines()
    assert _aer_counts(emitted) == _aer_counts(src)
    native = polypus.statevector(polypus.Circuit.from_qasm2(src))
    np.testing.assert_allclose(native, _qiskit_statevector(src), atol=1e-10)
