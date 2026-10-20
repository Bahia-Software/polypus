"""
Lowering of instructions CUNQA does not convert, on the CUNQA backend.

CUNQA turns each Qiskit circuit into its own instruction format, and its
converter only takes the names in ``cunqa.circuit.ir.SUPPORTED_QISKIT_OPERATIONS``:
``ch``, ``u0``, ``rccx``, ``rc3x``, ``c3x``, ``c3sqrtx``, ``c4x`` and the calls of
declared ``gate`` blocks are in the Polypus vocabulary (C-2) but not in that
list. ``Cunqa.run_qcs`` therefore lowers a circuit with ``qiskit.transpile``
just before handing it to CUNQA, and only when it holds such an instruction,
as ``local.py`` does for Aer.

CUNQA is not installed in CI, so its modules are stubbed and the circuits
``run`` receives are inspected.
"""

import sys
import types

import pytest

# `SUPPORTED_QISKIT_OPERATIONS` of CUNQA 3.0.0 (`cunqa/circuit/ir.py`).
CUNQA_OPERATIONS = {
    "unitary", "ryy", "rz", "z", "p", "rxx", "rx", "cx", "id", "x", "sxdg", "u1",
    "ccy", "rzz", "rzx", "ry", "s", "cu", "crz", "ecr", "t", "ccx", "y", "cswap",
    "r", "sdg", "csx", "crx", "ccz", "u3", "u2", "u", "cp", "tdg", "sx", "cu1",
    "swap", "cy", "cry", "cz", "h", "cu3", "measure", "if_else", "barrier", "reset",
    "save_state", "set_statevector",
}  # fmt: skip

H = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\n'

# name -> (declarations, statement); non-ascending operands, so a swapped
# control and target changes the operator.
CASES = {
    "ch": ("", "ch q[3],q[1];"),
    "u0": ("", "u0(2) q[4];"),
    "rccx": ("", "rccx q[4],q[0],q[2];"),
    "rc3x": ("", "rc3x q[3],q[1],q[4],q[0];"),
    "c3x": ("", "c3x q[2],q[4],q[0],q[3];"),
    "c3sqrtx": ("", "c3sqrtx q[4],q[3],q[1],q[2];"),
    "c4x": ("", "c4x q[3],q[0],q[4],q[2],q[1];"),
    "declared_gate": (
        "gate mixer(t) a,b,c { ch a,b; rz(t) c; ccx c,b,a; }\n",
        "mixer(0.7) q[3],q[0],q[2];",
    ),
}


def _program(case, measure=False):
    declarations, statement = CASES[case]
    lines = [H + declarations.rstrip("\n")]
    if measure:
        lines.append("creg c[5];")
    for q in range(5):
        lines.append(f"ry({0.3 + 0.4 * q!r}) q[{q}];")
    lines.append(statement)
    if measure:
        lines.append("measure q -> c;")
    return "\n".join(lines) + "\n"


def _stub_cunqa(monkeypatch, operations):
    """Fake ``cunqa.*`` modules; ``operations=None`` stands for a CUNQA without
    ``cunqa.circuit.ir``. Returns the list the circuits sent to ``run`` land in."""
    sent = []

    def fake_run(circuit, qpu, **kwargs):
        sent.append(circuit)
        return types.SimpleNamespace(counts={"0" * circuit.num_clbits: kwargs["shots"]})

    qpu = types.ModuleType("cunqa.qpu")
    qpu.get_QPUs = lambda **k: [object() for _ in range(16)]
    qpu.qdrop = lambda *a, **k: None
    qpu.qraise = lambda *a, **k: "fam"
    qpu.run = fake_run
    qjob = types.ModuleType("cunqa.qjob")
    qjob.gather = lambda jobs: jobs

    modules = {"cunqa": types.ModuleType("cunqa"), "cunqa.qpu": qpu, "cunqa.qjob": qjob}
    if operations is not None:
        ir = types.ModuleType("cunqa.circuit.ir")
        ir.SUPPORTED_QISKIT_OPERATIONS = set(operations)
        modules["cunqa.circuit"] = types.ModuleType("cunqa.circuit")
        modules["cunqa.circuit.ir"] = ir
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    # A fresh import of the module under test, with the stubs in place.
    monkeypatch.delitem(sys.modules, "polypus_python.cunqa", raising=False)
    return sent


def _run_qcs(qcs):
    from polypus_python.cunqa import Cunqa

    return Cunqa().run_qcs(qcs=qcs, shots=8, family_id="fam")


def _names(qc):
    names = set(qc.count_ops())
    for instruction in qc.data:
        for block in getattr(instruction.operation, "blocks", ()):
            names |= _names(block)
    return names


@pytest.mark.parametrize(
    "as_string", [True, False], ids=["qasm_string", "quantum_circuit"]
)
@pytest.mark.parametrize("case", CASES)
def test_gate_cunqa_does_not_convert_is_lowered_to_an_equivalent_circuit(
    monkeypatch, case, as_string
):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    sent = _stub_cunqa(monkeypatch, CUNQA_OPERATIONS)
    src = _program(case)
    original = QuantumCircuit.from_qasm_str(src)
    assert not _names(original) <= CUNQA_OPERATIONS, "the case must need lowering"

    _run_qcs([src if as_string else original])

    (lowered,) = sent
    assert _names(lowered) <= CUNQA_OPERATIONS, sorted(
        _names(lowered) - CUNQA_OPERATIONS
    )
    # Without the full read-out the seal adds to an unmeasured circuit (C-3).
    unitary = lowered.remove_final_measurements(inplace=False)
    assert Operator(unitary).equiv(Operator(original)), case
    if not as_string:
        # The caller's circuit is never modified.
        assert original == QuantumCircuit.from_qasm_str(src)


def test_measurements_and_registers_survive_the_lowering(monkeypatch):
    sent = _stub_cunqa(monkeypatch, CUNQA_OPERATIONS)
    _run_qcs([_program("ch", measure=True)])

    (lowered,) = sent
    assert [(r.name, r.size) for r in lowered.cregs] == [("c", 5)]
    measures = [
        (lowered.find_bit(i.qubits[0]).index, lowered.find_bit(i.clbits[0]).index)
        for i in lowered.data
        if i.operation.name == "measure"
    ]
    assert measures == [(q, q) for q in range(5)]


def test_circuit_cunqa_converts_is_handed_over_as_it_is(monkeypatch):
    from qiskit import QuantumCircuit

    sent = _stub_cunqa(monkeypatch, CUNQA_OPERATIONS)
    qc = QuantumCircuit.from_qasm_str(
        H + "creg c[5];\nh q[0];\ncx q[0],q[3];\nccx q[3],q[0],q[1];\nbarrier q;\n"
        "measure q -> c;\n"
    )

    _run_qcs([qc])

    assert sent == [qc]
    assert sent[0] is qc, "neither copied nor transpiled"


def test_cunqa_without_the_operation_list_gets_the_circuit_unchanged(monkeypatch):
    from qiskit import QuantumCircuit

    sent = _stub_cunqa(monkeypatch, None)
    qc = QuantumCircuit.from_qasm_str(_program("ch", measure=True))

    _run_qcs([qc])

    assert sent[0] is qc
