"""
Gate-by-gate equivalence of the Polypus OpenQASM path with Qiskit.

Every gate the importer accepts is checked four ways on the same QASM text:

1. Round-trip: a canonical statement is re-emitted byte-identically (one
   instruction, same name, same operand order -- nothing decomposed).
2. Aer: the text is run through Qiskit directly, and through
   ``polypus.Circuit.from_qasm2(...).to_qasm2()`` first; both go through the
   same Qiskit parse and Aer run with a fixed seed, and the counts must match.
   This is what the orchestration benchmark relies on: Polypus hands Aer the
   program Qiskit would have handed it.
3. Native: ``polypus.statevector`` of the imported circuit against
   ``qiskit.quantum_info.Statevector`` of the same text. This is the check
   that catches a control/target (or parameter-order) mistake *inside* the
   Polypus IR, which the text round-trip in (2) cannot see.
4. Builder: the Python builder method against Qiskit's own method.

Every gate runs on a generic input state (distinct rotations on every qubit)
with non-ascending operands and distinct angles, so swapped qubits or angles
change the result.
"""

import math

import numpy as np
import pytest

SEED = 1234
SHOTS = 4000
ANGLES = (0.37, -1.21, 2.03, 0.61)


def _fmt(values):
    return ",".join(f"{v:.12f}" for v in values)


# (name, qubit operands, number of angles, Qiskit builder, Polypus builder)
# Qiskit's builders take the angles first; Polypus's take the qubits first.
GATES = [
    ("h", (2,), 0, lambda qc: qc.h(2), lambda c: c.h(2)),
    ("x", (1,), 0, lambda qc: qc.x(1), lambda c: c.x(1)),
    ("y", (0,), 0, lambda qc: qc.y(0), lambda c: c.y(0)),
    ("z", (2,), 0, lambda qc: qc.z(2), lambda c: c.z(2)),
    ("s", (1,), 0, lambda qc: qc.s(1), lambda c: c.s(1)),
    ("t", (0,), 0, lambda qc: qc.t(0), lambda c: c.t(0)),
    ("sdg", (2,), 0, lambda qc: qc.sdg(2), lambda c: c.sdg(2)),
    ("tdg", (1,), 0, lambda qc: qc.tdg(1), lambda c: c.tdg(1)),
    ("id", (0,), 0, lambda qc: qc.id(0), lambda c: c.id(0)),
    ("sx", (2,), 0, lambda qc: qc.sx(2), lambda c: c.sx(2)),
    ("sxdg", (1,), 0, lambda qc: qc.sxdg(1), lambda c: c.sxdg(1)),
    ("rx", (0,), 1, lambda qc: qc.rx(ANGLES[0], 0), lambda c: c.rx(0, ANGLES[0])),
    ("ry", (2,), 1, lambda qc: qc.ry(ANGLES[0], 2), lambda c: c.ry(2, ANGLES[0])),
    ("rz", (1,), 1, lambda qc: qc.rz(ANGLES[0], 1), lambda c: c.rz(1, ANGLES[0])),
    (
        "u3",
        (0,),
        3,
        lambda qc: qc.append(_u3(*ANGLES[:3]), [0]),
        lambda c: c.u(0, *ANGLES[:3]),
    ),
    # The other spellings of the same family, each kept as written.
    ("p", (3,), 1, lambda qc: qc.p(ANGLES[0], 3), lambda c: c.p(3, ANGLES[0])),
    (
        "u1",
        (2,),
        1,
        lambda qc: qc.append(_u1(ANGLES[0]), [2]),
        lambda c: c.u1(2, ANGLES[0]),
    ),
    (
        "u2",
        (0,),
        2,
        lambda qc: qc.append(_u2(*ANGLES[:2]), [0]),
        lambda c: c.u2(0, *ANGLES[:2]),
    ),
    # Polypus's `u()` builder spells `u3`; the operator is the same.
    ("u", (4,), 3, lambda qc: qc.u(*ANGLES[:3], 4), lambda c: c.u(4, *ANGLES[:3])),
    ("cx", (2, 0), 0, lambda qc: qc.cx(2, 0), lambda c: c.cx(2, 0)),
    ("cz", (1, 2), 0, lambda qc: qc.cz(1, 2), lambda c: c.cz(1, 2)),
    ("cy", (2, 1), 0, lambda qc: qc.cy(2, 1), lambda c: c.cy(2, 1)),
    ("ch", (0, 2), 0, lambda qc: qc.ch(0, 2), lambda c: c.ch(0, 2)),
    ("csx", (2, 0), 0, lambda qc: qc.csx(2, 0), lambda c: c.csx(2, 0)),
    ("swap", (1, 0), 0, lambda qc: qc.swap(1, 0), lambda c: c.swap(1, 0)),
    ("ccx", (2, 0, 1), 0, lambda qc: qc.ccx(2, 0, 1), lambda c: c.ccx(2, 0, 1)),
    (
        "cswap",
        (1, 2, 0),
        0,
        lambda qc: qc.cswap(1, 2, 0),
        lambda c: c.cswap(1, 2, 0),
    ),
    (
        "rzz",
        (2, 1),
        1,
        lambda qc: qc.rzz(ANGLES[0], 2, 1),
        lambda c: c.rzz(2, 1, ANGLES[0]),
    ),
    (
        "rxx",
        (1, 0),
        1,
        lambda qc: qc.rxx(ANGLES[0], 1, 0),
        lambda c: c.rxx(1, 0, ANGLES[0]),
    ),
    (
        "cp",
        (2, 0),
        1,
        lambda qc: qc.cp(ANGLES[0], 2, 0),
        lambda c: c.cp(2, 0, ANGLES[0]),
    ),
    (
        "cu1",
        (2, 0),
        1,
        lambda qc: qc.append(_cu1(ANGLES[0]), [2, 0]),
        lambda c: c.cu1(2, 0, ANGLES[0]),
    ),
    (
        "crx",
        (1, 0),
        1,
        lambda qc: qc.crx(ANGLES[0], 1, 0),
        lambda c: c.crx(1, 0, ANGLES[0]),
    ),
    (
        "cry",
        (2, 1),
        1,
        lambda qc: qc.cry(ANGLES[0], 2, 1),
        lambda c: c.cry(2, 1, ANGLES[0]),
    ),
    (
        "crz",
        (0, 2),
        1,
        lambda qc: qc.crz(ANGLES[0], 0, 2),
        lambda c: c.crz(0, 2, ANGLES[0]),
    ),
    (
        "cu3",
        (2, 0),
        3,
        lambda qc: qc.append(_cu3(*ANGLES[:3]), [2, 0]),
        lambda c: c.cu3(2, 0, *ANGLES[:3]),
    ),
    (
        "cu",
        (1, 2),
        4,
        lambda qc: qc.cu(*ANGLES, 1, 2),
        lambda c: c.cu(1, 2, *ANGLES),
    ),
    # u0 is the identity: Qiskit's reference circuit gets nothing appended.
    ("u0", (4,), 1, lambda qc: None, lambda c: c.u0(4, 2.0)),
    ("rccx", (4, 0, 2), 0, lambda qc: qc.rccx(4, 0, 2), lambda c: c.rccx(4, 0, 2)),
    (
        "rc3x",
        (3, 1, 4, 0),
        0,
        lambda qc: qc.rcccx(3, 1, 4, 0),
        lambda c: c.rc3x(3, 1, 4, 0),
    ),
    (
        "c3x",
        (2, 4, 0, 3),
        0,
        lambda qc: qc.mcx([2, 4, 0], 3),
        lambda c: c.c3x(2, 4, 0, 3),
    ),
    (
        "c3sqrtx",
        (4, 3, 1, 2),
        0,
        lambda qc: qc.append(_c3sx(), [4, 3, 1, 2]),
        lambda c: c.c3sqrtx(4, 3, 1, 2),
    ),
    (
        "c4x",
        (1, 4, 0, 3, 2),
        0,
        lambda qc: qc.mcx([1, 4, 0, 3], 2),
        lambda c: c.c4x(1, 4, 0, 3, 2),
    ),
]

GATE_IDS = [g[0] for g in GATES]
N_QUBITS = 5


def _c3sx():
    from qiskit.circuit.library import C3SXGate

    return C3SXGate()


def _u3(theta, phi, lam):
    from qiskit.circuit.library import U3Gate

    return U3Gate(theta, phi, lam)


def _u1(lam):
    from qiskit.circuit.library import U1Gate

    return U1Gate(lam)


def _u2(phi, lam):
    from qiskit.circuit.library import U2Gate

    return U2Gate(phi, lam)


def _cu1(lam):
    from qiskit.circuit.library import CU1Gate

    return CU1Gate(lam)


def _cu3(theta, phi, lam):
    from qiskit.circuit.library import CU3Gate

    return CU3Gate(theta, phi, lam)


# Qiskit reads `u0(n)` as "idle for n identity slots" and rejects a non-integer
# n, so `u0` gets an integer argument.
_ANGLE_OVERRIDES = {"u0": (2.0,)}


def _statement(name, qubits, n_angles):
    values = _ANGLE_OVERRIDES.get(name, ANGLES[:n_angles])
    angles = f"({_fmt(values)})" if n_angles else ""
    operands = ",".join(f"q[{q}]" for q in qubits)
    return f"{name}{angles} {operands};"


def _prep_lines():
    """A generic, entangling-free input state: distinct rotations per qubit."""
    lines = []
    for q in range(N_QUBITS):
        lines.append(f"ry({0.3 + 0.4 * q:.12f}) q[{q}];")
        lines.append(f"rz({0.2 + 0.5 * q:.12f}) q[{q}];")
    return lines


def _program(statement, measure):
    body = _prep_lines() + [statement]
    head = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{N_QUBITS}];"]
    if measure:
        head.append(f"creg c[{N_QUBITS}];")
        body.append("measure q -> c;")
    return "\n".join(head + body) + "\n"


def _aer_counts(qasm):
    """Qiskit parse -> transpile (optimisation off) -> Aer, fixed seed."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    qc = transpile(QuantumCircuit.from_qasm_str(qasm), sim, optimization_level=0)
    return sim.run(qc, shots=SHOTS, seed_simulator=SEED).result().get_counts()


def _prep_qiskit():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(N_QUBITS)
    for q in range(N_QUBITS):
        qc.ry(0.3 + 0.4 * q, q)
        qc.rz(0.2 + 0.5 * q, q)
    return qc


def _prep_polypus():
    import polypus

    c = polypus.Circuit(N_QUBITS)
    for q in range(N_QUBITS):
        c = c.ry(q, 0.3 + 0.4 * q).rz(q, 0.2 + 0.5 * q)
    return c


@pytest.mark.parametrize(
    ("name", "qubits", "n_angles", "qk", "pp"), GATES, ids=GATE_IDS
)
class TestGateEquivalence:
    def test_statement_reemits_byte_identically(self, name, qubits, n_angles, qk, pp):
        import polypus

        src = _program(_statement(name, qubits, n_angles), measure=True)
        imported = polypus.Circuit.from_qasm2(src)
        assert len(imported) == len(_prep_lines()) + 2  # prep + gate + measure
        assert imported.to_qasm2() == src

    def test_aer_counts_match_qiskit(self, name, qubits, n_angles, qk, pp):
        import polypus

        src = _program(_statement(name, qubits, n_angles), measure=True)
        reemitted = polypus.Circuit.from_qasm2(src).to_qasm2()
        assert _aer_counts(reemitted) == _aer_counts(src)

    def test_count_ops_and_depth_match_qiskit(self, name, qubits, n_angles, qk, pp):
        import polypus
        from qiskit import QuantumCircuit

        src = _program(_statement(name, qubits, n_angles), measure=True)
        original = QuantumCircuit.from_qasm_str(src)
        through_polypus = QuantumCircuit.from_qasm_str(
            polypus.Circuit.from_qasm2(src).to_qasm2()
        )
        assert through_polypus.count_ops() == original.count_ops()
        assert through_polypus.depth() == original.depth()
        assert through_polypus.size() == original.size()

    def test_native_statevector_matches_qiskit(self, name, qubits, n_angles, qk, pp):
        import polypus
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        src = _program(_statement(name, qubits, n_angles), measure=False)
        native = polypus.statevector(polypus.Circuit.from_qasm2(src))
        reference = Statevector(QuantumCircuit.from_qasm_str(src)).data
        np.testing.assert_allclose(native, reference, atol=1e-10)

    def test_builder_matches_qiskit_builder(self, name, qubits, n_angles, qk, pp):
        import polypus
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        built = pp(_prep_polypus())
        native = polypus.statevector(built)
        reference_qc = _prep_qiskit()
        qk(reference_qc)
        reference = Statevector(reference_qc).data
        np.testing.assert_allclose(native, reference, atol=1e-10)
        # ... and the builder's export is the same program for Qiskit.
        exported = Statevector(QuantumCircuit.from_qasm_str(built.to_qasm2())).data
        np.testing.assert_allclose(exported, reference, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter binding: a bound parameterised circuit is the fixed-angle circuit
# ─────────────────────────────────────────────────────────────────────────────


def test_bound_parameterised_gates_match_fixed_angle_circuit():
    import polypus
    from polypus import Param

    values = [0.3, -1.2, 2.5, 0.7]
    parameterised = (
        polypus.Circuit(3)
        .crx(0, 1, Param(0))
        .cry(2, 0, Param(1))
        .crz(1, 2, Param(2))
        .cu1(2, 1, Param(3))
        .cu3(0, 2, Param(2), Param(0), Param(1))
        .cu(1, 0, Param(3), Param(2), Param(1), Param(0))
    )
    fixed = (
        polypus.Circuit(3)
        .crx(0, 1, values[0])
        .cry(2, 0, values[1])
        .crz(1, 2, values[2])
        .cu1(2, 1, values[3])
        .cu3(0, 2, values[2], values[0], values[1])
        .cu(1, 0, values[3], values[2], values[1], values[0])
    )
    assert parameterised.num_params == 4
    assert parameterised.to_qasm2(values) == fixed.to_qasm2()
    np.testing.assert_allclose(
        polypus.statevector(parameterised, values),
        polypus.statevector(fixed),
        atol=1e-12,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The real execution path: polypus.run_quantum_circuit on the local Aer backend
# ─────────────────────────────────────────────────────────────────────────────

# Gates Aer executes natively. `ch`, `u0` and the multi-qubit qelib1.inc gates
# (`rccx`, `rc3x`, `c3x`, `c3sqrtx`, `c4x`) are not in Aer's basis, and the
# local backend submits circuits to Aer untranspiled, so such a circuit does not
# run there today (a local-backend limitation, not an import/export one: the
# transpiled comparison above covers every gate).
_NOT_IN_AER_BASIS = {"ch", "u0", "rccx", "rc3x", "c3x", "c3sqrtx", "c4x"}
_AER_NATIVE = [g for g in GATES if g[0] not in _NOT_IN_AER_BASIS]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "qubits", "n_angles", "qk", "pp"),
    _AER_NATIVE,
    ids=[g[0] for g in _AER_NATIVE],
)
def test_run_quantum_circuit_matches_qiskit_on_aer(name, qubits, n_angles, qk, pp):
    import polypus
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    src = _program(_statement(name, qubits, n_angles), measure=True)
    result = polypus.run_quantum_circuit(
        polypus.Circuit.from_qasm2(src), shots=SHOTS, infrastructure="local", seed=SEED
    )
    polypus_counts = result.counts[0]
    qiskit_counts = (
        AerSimulator()
        .run(QuantumCircuit.from_qasm_str(src), shots=SHOTS, seed_simulator=SEED)
        .result()
        .get_counts()
    )
    assert sum(polypus_counts.values()) == SHOTS
    assert polypus_counts == qiskit_counts


# ─────────────────────────────────────────────────────────────────────────────
# Signature errors name the gate
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "qubits", "n_angles", "qk", "pp"), GATES, ids=GATE_IDS
)
def test_wrong_arity_and_parameter_count_name_the_gate(name, qubits, n_angles, qk, pp):
    import polypus

    too_many_qubits = tuple(range(len(qubits) + 1))
    src = f"OPENQASM 2.0;\nqreg q[8];\n{_statement(name, too_many_qubits, n_angles)}\n"
    with pytest.raises(
        ValueError,
        match=rf"line 3: gate '{name}' expects {len(qubits)} argument\(s\), found {len(qubits) + 1}",
    ):
        polypus.Circuit.from_qasm2(src)

    n_wrong = n_angles + 1
    angles = f"({','.join(['0.1'] * n_wrong)})"
    operands = ",".join(f"q[{q}]" for q in range(len(qubits)))
    src = f"OPENQASM 2.0;\nqreg q[5];\n{name}{angles} {operands};\n"
    with pytest.raises(
        ValueError,
        match=rf"line 3: gate '{name}' expects {n_angles} parameter\(s\), found {n_wrong}",
    ):
        polypus.Circuit.from_qasm2(src)


@pytest.mark.parametrize(
    ("name", "qubits", "n_angles", "qk", "pp"), GATES, ids=GATE_IDS
)
def test_out_of_range_qubit_is_rejected(name, qubits, n_angles, qk, pp):
    import polypus

    # A register exactly as wide as the gate; the last operand is one past it.
    width = len(qubits)
    operands = tuple(range(width - 1)) + (width,)
    src = f"OPENQASM 2.0;\nqreg q[{width}];\n{_statement(name, operands, n_angles)}\n"
    with pytest.raises(ValueError, match=rf"line 3: index {width} out of range"):
        polypus.Circuit.from_qasm2(src)


def test_angle_formatting_matches_the_exporter():
    """The helper renders angles exactly as the exporter does (12 decimals),
    which the byte-identity checks above depend on."""
    import polypus

    assert f"rx({math.pi:.12f}) q[0];" in polypus.Circuit(1).rx(0, math.pi).to_qasm2()
