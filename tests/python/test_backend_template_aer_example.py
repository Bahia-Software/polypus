"""
The Qiskit Aer fill that ``examples/python-backend-template/worker.py`` shows for
``execute_circuits`` runs as written (#642).

A registered backend receives each request's seed as any 64-bit value (contract
C-7), while Aer's ``seed_simulator`` is a signed 64-bit integer that rejects a
seed of ``2**63`` or more. The example masks the seed to 63 bits; this test
extracts the commented example from the worker, runs it on the seeds a provider
can receive, and checks that each one is accepted and replays.
"""

import pathlib
import textwrap

import pytest

aer = pytest.importorskip("qiskit_aer")

pytestmark = pytest.mark.integration

WORKER = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples"
    / "python-backend-template"
    / "worker.py"
)
START = "# A real fill looks like, e.g. (Qiskit Aer):"
BELL = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""


def _example():
    """The worker's commented Aer fill, as the body of
    ``execute_circuits(circuits, shots, seed)``."""
    lines = WORKER.read_text().splitlines()
    start = lines.index(START) + 1
    body = []
    for line in lines[start:]:
        if not line.startswith("#     "):
            break
        body.append(line[len("#     ") :])
    assert body and body[-1].startswith("return"), body
    source = "def execute_circuits(circuits, shots, seed):\n" + textwrap.indent(
        "\n".join(body), "    "
    )
    namespace = {}
    exec(compile(source, str(WORKER), "exec"), namespace)
    return namespace["execute_circuits"]


@pytest.mark.parametrize("seed", [0, 2**63 - 1, 2**63, 2**64 - 1, None])
def test_the_aer_example_takes_any_seed_a_registered_backend_receives(seed):
    execute_circuits = _example()
    circuits = [{"qasm": BELL, "n_qubits": 2}]
    counts = execute_circuits(circuits, 256, seed)
    assert len(counts) == 1 and sum(counts[0].values()) == 256
    assert set(counts[0]) <= {"00", "11"}
    if seed is not None:
        # The same seed replays the same counts.
        assert execute_circuits(circuits, 256, seed) == counts


def test_the_aer_example_hands_aer_the_seed_masked_to_63_bits():
    # The seed reaches Aer, and only its bit 63 is dropped: with bit 62 set
    # (so a narrower mask would show), s + 2**63 samples as s does, and s as
    # Aer sampling with seed s directly; another seed samples otherwise.
    from qiskit import QuantumCircuit

    execute_circuits = _example()
    circuits = [{"qasm": BELL, "n_qubits": 2}]
    seed = 2**62 + 5

    def direct(s):
        qc = QuantumCircuit.from_qasm_str(BELL)
        return (
            aer.AerSimulator(seed_simulator=s).run(qc, shots=256).result().get_counts()
        )

    assert direct(seed) != direct(5), "pick seeds Aer samples apart"
    assert execute_circuits(circuits, 256, seed) == [direct(seed)]
    assert execute_circuits(circuits, 256, seed + 2**63) == [direct(seed)]


def test_aer_still_rejects_an_unmasked_seed_of_two_to_the_63():
    # Why the example masks: if Aer ever accepts the full 64-bit range, the
    # mask stays harmless, but this premise of #642 no longer holds. Aer takes
    # the seed at construction and rejects it when it runs.
    from qiskit import QuantumCircuit

    qc = QuantumCircuit.from_qasm_str(BELL)
    with pytest.raises(TypeError):
        aer.AerSimulator(seed_simulator=2**63).run(qc, shots=8).result()
