"""
Shared machinery for the tests that compare the adjoint gradient with the
parameter-shift fallback (``test_qml_adjoint.py`` and the ``QuantumLayer`` case
of ``test_torch_quantum_layer.py``; #503).

There is no switch between the two: ``POLYPUS_MEM_BUDGET`` decides, so each
case runs in a child process twice — with a budget that holds two
statevectors, and with one that holds one. A plain module, not a test file:
imported as ``from qml_adjoint_helpers import ...``, like
``qml_native_helpers``.
"""

import json
import os
import subprocess
import sys
import textwrap

# A 4-qubit statevector is 16 · 2^4 = 256 bytes: 300 bytes hold one, not two;
# 200 hold none; 1 MiB holds plenty.
ONE_BUT_NOT_TWO = "300"
NOT_EVEN_ONE = "200"
PLENTY = "1M"

FALLBACK = "falls back to the parameter-shift rule"

# Exact either way; the sums run in another order, so the last bits differ.
TOLERANCE = 1e-9

SETUP = """
import json
import polypus
from polypus import qml

polypus.init_logger(level="info", console=True, timestamp=False)

model = qml.Model(
    qml.Encoder(4).angle_encoder("ry"),
    qml.Ansatz(4).hardware_efficient(1).conv("basic"),
    qml.Readout([qml.Z(1), qml.Z(0) @ qml.Z(2)], decision="argmax"),
)
rows = [[0.4, 1.1, 0.7, 2.0], [1.3, 0.2, 2.2, 0.9], [2.0, 1.7, 0.3, 1.2]]
params = [0.3 + 0.17 * k for k in range(model.num_params)]
"""


def run_child(code, budget):
    """Run ``SETUP`` + ``code`` with ``POLYPUS_MEM_BUDGET=budget``; return the
    completed process. ``code`` prints one ``RESULT <json>`` line."""
    env = dict(os.environ, POLYPUS_MEM_BUDGET=budget)
    return subprocess.run(
        [sys.executable, "-c", SETUP + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )


def child_result(process):
    assert process.returncode == 0, process.stdout + process.stderr
    lines = [line for line in process.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, process.stdout
    return json.loads(lines[0][len("RESULT ") :])


def fell_back(process):
    return FALLBACK in process.stdout + process.stderr


def assert_close(adjoint, shift, path="result"):
    if isinstance(adjoint, list):
        assert isinstance(shift, list) and len(adjoint) == len(shift), path
        for i, (a, s) in enumerate(zip(adjoint, shift)):
            assert_close(a, s, f"{path}[{i}]")
    else:
        assert abs(adjoint - shift) < TOLERANCE, f"{path}: {adjoint} vs {shift}"


def adjoint_and_shift(code):
    """``code``'s result with the adjoint, then with the shift fallback."""
    adjoint = run_child(code, PLENTY)
    shift = run_child(code, ONE_BUT_NOT_TWO)
    assert not fell_back(adjoint), adjoint.stdout
    assert fell_back(shift), "the 300-byte budget must take the shift rule"
    return child_result(adjoint), child_result(shift)
