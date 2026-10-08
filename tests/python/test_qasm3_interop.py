"""
Interoperability of the OpenQASM 3 profile with Qiskit, in both directions.

Supported (tested) versions: Qiskit 2.5.2 with qiskit-qasm3-import 0.6.0
(``requirements-dev.txt``). That Polypus reads Qiskit's output as Qiskit
does is a tested property of these versions and circuits, not a general
guarantee.

- Qiskit → Polypus: ``qasm3.dumps`` output — the pinned fixtures and a set
  generated here (QAOA, ``efficient_su2``, ``real_amplitudes``,
  ``zz_feature_map``, every vocabulary gate) — read by
  ``Circuit.from_qasm3``.
- Polypus → Qiskit: ``Circuit.to_qasm3`` read by ``qiskit.qasm3.loads``.

Parameters are paired by an explicit mapping — original parameter, the
identifier the exporter wrote for it, the Polypus index of that identifier —
never by position alone. Statevectors are compared with the phase factor the
profile predicts, never with one fitted to the result: both readers give
``U``, ``u2`` and ``u3`` Qiskit's matrices, so one text yields one statevector
(factor 1); against the original Qiskit circuit, the only difference is the
global phase ``qasm3.dumps`` does not write — the circuit's own, and that of
every gate definition it writes (``sxdg`` becomes ``s; h; s`` without its
e^{-iπ/4}) — computed from Qiskit's definitions. Counts and their bit order
(contract C-3) are tested apart, on basis states.
"""

import math

import numpy as np
import polypus
import pytest
import qasm3_fixtures
import qiskit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3
from qiskit.circuit import Parameter, ParameterVectorElement
from qiskit.circuit import library as lib
from qiskit.quantum_info import Statevector

ATOL = 1e-10
SHOTS = 64
RNG = np.random.default_rng(236)
BINDINGS = 3


def exported_name(param) -> str:
    """The identifier Qiskit 2.5.2's exporter writes for ``param``: a valid
    name as it is; the element ``v[i]`` of a ``ParameterVector`` named with
    identifier characters as ``_v_i_``."""
    if isinstance(param, ParameterVectorElement):
        return f"_{param.vector.name}_{param.index}_"
    return param.name


def polypus_values(circuit, mapping: dict, values: dict) -> list:
    """Order ``values`` (by original parameter) as ``circuit``'s parameters,
    through ``mapping`` (original parameter → exported identifier)."""
    names = circuit.param_names
    assert sorted(names) == sorted(mapping.values()), (names, mapping)
    ordered = [None] * len(names)
    for param, identifier in mapping.items():
        ordered[names.index(identifier)] = values[param]
    return ordered


def unitary_part(qc: QuantumCircuit) -> QuantumCircuit:
    return qc.remove_final_measurements(inplace=False)


# The gates Qiskit 2.5.2's exporter writes by name (`stdgates.inc` and `U`);
# it writes every other gate as a definition.
WRITTEN_BY_NAME = {
    *("p x y z h s sdg t tdg sx rx ry rz cx cy cz cp crx cry crz".split()),
    *("ch swap ccx cswap cu id u1 u2 u3 u barrier measure".split()),
}


def dropped_phase(qc: QuantumCircuit) -> float:
    """The global phase ``qasm3.dumps`` does not write for the bound
    circuit ``qc``: its own, and that of each definition it writes in place of
    a gate, recursively."""
    total = float(qc.global_phase)
    for instruction in qc.data:
        operation = instruction.operation
        if operation.name not in WRITTEN_BY_NAME:
            total += dropped_phase(operation.definition)
    return total


def draws(params) -> list:
    return [
        {
            p: float(v)
            for p, v in zip(params, RNG.uniform(-math.pi, math.pi, len(params)))
        }
        for _ in range(BINDINGS)
    ]


# ─────────────────────────────── The set ────────────────────────────────


def _prepared(n: int) -> QuantumCircuit:
    """A generic input state: distinct rotations on every qubit, so that a
    swapped operand or a wrong phase shows."""
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(0.4 + 0.3 * q, q)
        qc.rz(-0.7 + 0.5 * q, q)
    return qc


def qaoa(layers: int = 2, nodes: int = 4) -> QuantumCircuit:
    """QED-C-style QAOA MaxCut on a ring: ``rzz(-γ)`` per edge, ``rx(2β)``."""
    qc = QuantumCircuit(nodes)
    qc.h(range(nodes))
    for layer in range(layers):
        gamma = Parameter(f"gamma{layer}")
        beta = Parameter(f"beta{layer}")
        for q in range(nodes):
            qc.rzz(-gamma, q, (q + 1) % nodes)
        for q in range(nodes):
            qc.rx(2 * beta, q)
    qc.measure_all()
    return qc


def vocabulary() -> dict:
    """One circuit per Polypus instruction Qiskit has, on a prepared state,
    parameterised where the gate takes angles."""
    a, b, c, d = (Parameter(name) for name in ("a", "b", "c", "d"))
    gates = {
        "h": (1, lambda qc: qc.h(0)),
        "x": (1, lambda qc: qc.x(0)),
        "y": (1, lambda qc: qc.y(0)),
        "z": (1, lambda qc: qc.z(0)),
        "s": (1, lambda qc: qc.s(0)),
        "t": (1, lambda qc: qc.t(0)),
        "sdg": (1, lambda qc: qc.sdg(0)),
        "tdg": (1, lambda qc: qc.tdg(0)),
        "id": (1, lambda qc: qc.id(0)),
        "sx": (1, lambda qc: qc.sx(0)),
        "sxdg": (1, lambda qc: qc.sxdg(0)),
        "rx": (1, lambda qc: qc.rx(a, 0)),
        "ry": (1, lambda qc: qc.ry(a, 0)),
        "rz": (1, lambda qc: qc.rz(a * b, 0)),
        "p": (1, lambda qc: qc.p(a - 1.5, 0)),
        "u": (1, lambda qc: qc.u(a, b, c, 0)),
        "u1": (1, lambda qc: qc.append(lib.U1Gate(a), [0])),
        "u2": (1, lambda qc: qc.append(lib.U2Gate(a, b), [0])),
        "u3": (1, lambda qc: qc.append(lib.U3Gate(a, b, c), [0])),
        "cx": (2, lambda qc: qc.cx(1, 0)),
        "cy": (2, lambda qc: qc.cy(1, 0)),
        "cz": (2, lambda qc: qc.cz(1, 0)),
        "ch": (2, lambda qc: qc.ch(1, 0)),
        "csx": (2, lambda qc: qc.csx(1, 0)),
        "swap": (2, lambda qc: qc.swap(1, 0)),
        "cp": (2, lambda qc: qc.cp(a, 1, 0)),
        "crx": (2, lambda qc: qc.crx(a, 1, 0)),
        "cry": (2, lambda qc: qc.cry(a, 1, 0)),
        "crz": (2, lambda qc: qc.crz(a, 1, 0)),
        "cu": (2, lambda qc: qc.cu(a, b, c, d, 1, 0)),
        "cu1": (2, lambda qc: qc.append(lib.CU1Gate(a), [1, 0])),
        "cu3": (2, lambda qc: qc.append(lib.CU3Gate(a, b, c), [1, 0])),
        "rzz": (2, lambda qc: qc.rzz(a, 1, 0)),
        "rxx": (2, lambda qc: qc.rxx(2 * a, 1, 0)),
        "ccx": (3, lambda qc: qc.ccx(2, 0, 1)),
        "cswap": (3, lambda qc: qc.cswap(1, 2, 0)),
        "rccx": (3, lambda qc: qc.append(lib.RCCXGate(), [2, 0, 1])),
        "rc3x": (4, lambda qc: qc.append(lib.RC3XGate(), [3, 1, 0, 2])),
        "c3x": (4, lambda qc: qc.append(lib.C3XGate(), [2, 3, 0, 1])),
        "c3sqrtx": (4, lambda qc: qc.append(lib.C3SXGate(), [1, 3, 2, 0])),
        "c4x": (5, lambda qc: qc.append(lib.C4XGate(), [4, 0, 3, 1, 2])),
    }
    circuits = {}
    for name, (n, apply) in gates.items():
        qc = _prepared(n)
        apply(qc)
        circuits[name] = qc
    return circuits


def generated() -> dict:
    """The generated set: name → Qiskit circuit."""
    circuits = {
        "qaoa": qaoa(),
        "efficient_su2": lib.efficient_su2(3, reps=2),
        "efficient_su2_decomposed": lib.efficient_su2(3, reps=1).decompose(),
        "real_amplitudes": lib.real_amplitudes(3, reps=2),
        "zz_feature_map": lib.zz_feature_map(3, reps=2),
    }
    circuits.update({f"gate_{k}": v for k, v in vocabulary().items()})
    return circuits


GENERATED = generated()

# The pinned fixtures, with each input's original parameter spelled out: the
# names Qiskit wrote are not reversible (𝞫 and 𝞬 both became `___0_`).
FIXTURE_INPUTS = {
    "qaoa.qasm": {"𝞫[0]": "___0_", "𝞬[0]": "___0__0"},
    "zz_feature_map.qasm": {"x[0]": "_x_0_", "x[1]": "_x_1_"},
    "efficient_su2.qasm": {f"θ[{i}]": f"_θ_{i}_" for i in range(8)},
}


# ─────────────────────────────── Fixtures ───────────────────────────────


@pytest.mark.skipif(
    qiskit.__version__ != qasm3_fixtures.QISKIT_VERSION,
    reason=f"the fixtures are pinned for Qiskit {qasm3_fixtures.QISKIT_VERSION}",
)
@pytest.mark.parametrize("name", sorted(qasm3_fixtures.FIXTURES))
def test_the_pinned_fixtures_are_what_qiskit_writes(name):
    assert qasm3_fixtures.dumps(name) == qasm3_fixtures.pinned(name)


@pytest.mark.parametrize("name", sorted(qasm3_fixtures.FIXTURES))
def test_a_fixture_reads_as_qiskit_reads_it(name):
    text = qasm3_fixtures.pinned(name)
    original = qasm3_fixtures.FIXTURES[name]()
    by_name = {p.name: p for p in original.parameters}
    mapping = {by_name[orig]: ident for orig, ident in FIXTURE_INPUTS[name].items()}
    circuit = polypus.Circuit.from_qasm3(text)
    assert circuit.param_names == [
        line.split()[-1].rstrip(";")
        for line in text.splitlines()
        if line.startswith("input float[64] ")
    ]
    assert circuit.num_params == len(mapping)
    loaded = qasm3.loads(text)
    loaded_params = {p.name: p for p in loaded.parameters}
    for values in draws(list(mapping)):
        ours = polypus.statevector(circuit, polypus_values(circuit, mapping, values))
        theirs = Statevector(
            unitary_part(loaded).assign_parameters(
                {loaded_params[mapping[p]]: v for p, v in values.items()}
            )
        ).data
        np.testing.assert_allclose(ours, theirs, atol=ATOL)
        bound = unitary_part(original).assign_parameters(values)
        expected = Statevector(bound).data
        np.testing.assert_allclose(
            ours * np.exp(1j * dropped_phase(bound)), expected, atol=ATOL
        )


# ─────────────────────────── Qiskit → Polypus ───────────────────────────


@pytest.mark.parametrize("name", sorted(GENERATED))
def test_qiskit_writes_nothing_the_profile_rejects(name):
    """Qiskit's output for the generated set stays inside the profile: no
    `gphase`, no modifiers, no division of two integers (which the importer
    would reject, naming it)."""
    text = qasm3.dumps(GENERATED[name])
    assert "gphase" not in text and "@" not in text, text
    polypus.Circuit.from_qasm3(text)


@pytest.mark.parametrize("name", sorted(GENERATED))
def test_qiskit_output_reads_as_the_original_circuit(name):
    original = GENERATED[name]
    text = qasm3.dumps(original)
    mapping = {p: exported_name(p) for p in original.parameters}
    for identifier in mapping.values():
        assert f"input float[64] {identifier};" in text, (identifier, text)
    circuit = polypus.Circuit.from_qasm3(text)
    loaded = qasm3.loads(text)
    loaded_params = {p.name: p for p in loaded.parameters}
    for values in draws(list(mapping)) if mapping else [{}]:
        ours = polypus.statevector(circuit, polypus_values(circuit, mapping, values))
        theirs = Statevector(
            unitary_part(loaded).assign_parameters(
                {loaded_params[mapping[p]]: v for p, v in values.items()}
            )
        ).data
        np.testing.assert_allclose(ours, theirs, atol=ATOL)
        bound = unitary_part(original).assign_parameters(values)
        np.testing.assert_allclose(
            ours * np.exp(1j * dropped_phase(bound)),
            Statevector(bound).data,
            atol=ATOL,
        )


# ─────────────────────────── Polypus → Qiskit ───────────────────────────


def _polypus_set() -> dict:
    """Polypus circuits to export: the generated set read back (expressions
    and declared gates included), and builder circuits with every
    instruction Polypus writes through a definition."""
    circuits = {
        name: polypus.Circuit.from_qasm3(qasm3.dumps(qc))
        for name, qc in GENERATED.items()
    }
    circuits["builder_helpers"] = (
        polypus.Circuit(5)
        .h(0)
        .ry(1, 0.3)
        .rx(2, -0.8)
        .h(3)
        .ry(4, 1.1)
        .rzz(0, 1, polypus.Param(0))
        .rxx(1, 2, 0.5)
        .sxdg(2)
        .csx(3, 0)
        .cu1(0, 4, polypus.Param(1))
        .cu3(4, 3, 0.1, polypus.Param(0), -0.3)
        .u0(2, 1.0)
        .rccx(0, 1, 2)
        .rc3x(3, 2, 1, 0)
        .c3x(1, 2, 3, 4)
        .c3sqrtx(4, 3, 2, 1)
        .c4x(0, 1, 2, 3, 4)
        .u(1, 0.2, polypus.Param(1), 0.4)
        .u2(0, -0.2, 0.6)
    )
    # Names the export must rename: OpenQASM 3 keywords and a `stdgates.inc`
    # gate as OpenQASM 2.0 names, and a gate named like the register.
    circuits["renamed_from_qasm2"] = polypus.Circuit.from_qasm2(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        "gate input(t) box,delay { rx(t) box; cx box,delay; u(t,0,pi) delay; }\n"
        "qreg q[2];\nh q[0];\ninput(0.7) q[0],q[1];\n"
    )
    circuits["renamed_from_qasm3"] = polypus.Circuit.from_qasm3(
        "OPENQASM 3.0;\ninput float[64] c;\ngate q(x) a { U(x, 0, 0) a; }\n"
        "gate h a { q(0.3) a; U(pi/2, 0, pi) a; }\nqubit[2] r;\nh r[1];\nq(c) r[0];\n"
    )
    return circuits


POLYPUS_SET = _polypus_set()


def test_the_renamed_names_are_written():
    text = POLYPUS_SET["renamed_from_qasm2"].to_qasm3()
    assert "gate input_(t_) box_, delay_ {" in text
    text = POLYPUS_SET["renamed_from_qasm3"].to_qasm3()
    assert "input float[64] c;" in text and "gate h_ a {" in text
    assert "qubit[2] q_1;" in text
    assert not any(line.startswith("bit") for line in text.splitlines())


@pytest.mark.parametrize("name", sorted(POLYPUS_SET))
def test_polypus_output_reads_in_qiskit_as_the_same_circuit(name):
    circuit = POLYPUS_SET[name]
    text = circuit.to_qasm3()
    for identifier in circuit.param_names:
        assert f"input float[64] {identifier};" in text
    loaded = qasm3.loads(text)
    loaded_params = {p.name: p for p in loaded.parameters}
    # Polypus index → exported identifier → Qiskit parameter.
    assert sorted(loaded_params) == sorted(circuit.param_names)
    for _ in range(BINDINGS):
        values = [float(v) for v in RNG.uniform(-math.pi, math.pi, circuit.num_params)]
        ours = polypus.statevector(circuit, values)
        bound = unitary_part(loaded).assign_parameters(
            {loaded_params[circuit.param_names[i]]: v for i, v in enumerate(values)}
        )
        np.testing.assert_allclose(ours, Statevector(bound).data, atol=ATOL)


# ─────────────────────── Counts and bit order (C-3) ─────────────────────


def _aer_counts(qc: QuantumCircuit) -> dict:
    from qiskit_aer import AerSimulator

    counts = AerSimulator().run(qc, shots=SHOTS, seed_simulator=7).result().get_counts()
    return {key.replace(" ", ""): n for key, n in counts.items()}


def _native_counts(circuit) -> dict:
    [counts] = polypus.run_quantum_circuit(
        circuit, shots=SHOTS, infrastructure="local", backend="polypus"
    ).counts
    return counts


@pytest.mark.integration
def test_bits_keep_their_order_from_qiskit():
    """A basis state measured into two registers, the first declared before
    the qubits and filled out of order: clbit 0 (first register, first bit)
    is the rightmost character, as in Qiskit."""
    q = QuantumRegister(4, "q")
    first = ClassicalRegister(3, "first")
    second = ClassicalRegister(2, "second")
    qc = QuantumCircuit(first, q, second)
    qc.x(q[0])
    qc.x(q[3])
    qc.measure(q[3], first[0])
    qc.measure(q[1], first[2])
    qc.measure(q[0], second[1])
    qc.measure(q[2], second[0])
    text = qasm3.dumps(qc)
    # clbits 4..0: second[1] second[0] first[2] first[1] first[0] = 1 0 0 0 1
    # (first[1] is never measured).
    expected = {"10001": SHOTS}
    assert _aer_counts(qasm3.loads(text)) == expected
    assert _native_counts(polypus.Circuit.from_qasm3(text)) == expected


@pytest.mark.integration
def test_bits_keep_their_order_to_qiskit():
    circuit = polypus.Circuit(3).x(0).x(2).measure(2, 0).measure(0, 3).measure(1, 1)
    expected = {"1001": SHOTS}
    assert _native_counts(circuit) == expected
    assert _aer_counts(qasm3.loads(circuit.to_qasm3())) == expected
