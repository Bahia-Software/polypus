"""The exact gradients of native QML by adjoint differentiation (#503).

With ``exact=True`` on the native backend, ``qml.jacobian``, the backward pass
of ``polypus.torch.QuantumLayer`` and Adam's gradient in ``Model.fit`` come from
the adjoint: one differentiation per row (and observable), two statevectors
each, instead of two shifted circuits per trainable gate. When the memory
budget holds one statevector of the model's width but not two, they fall back to
the parameter shift, logged at ``info`` (contract C-11 (e)/(f); M2 of the
issue). There is no switch between the two: ``POLYPUS_MEM_BUDGET`` decides, so
each case runs in a child process twice — with a budget that holds two
statevectors, and with one that holds one — and the two exact results must
agree to rounding. That the second really took the shift rule is read off its
``info`` record.

Contract test: draft pending human sign-off.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

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


def _child(code, budget):
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


def _result(process):
    assert process.returncode == 0, process.stdout + process.stderr
    lines = [line for line in process.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, process.stdout
    return json.loads(lines[0][len("RESULT ") :])


def _fell_back(process):
    return FALLBACK in process.stdout + process.stderr


def _assert_close(adjoint, shift, path="result"):
    if isinstance(adjoint, list):
        assert isinstance(shift, list) and len(adjoint) == len(shift), path
        for i, (a, s) in enumerate(zip(adjoint, shift)):
            _assert_close(a, s, f"{path}[{i}]")
    else:
        assert abs(adjoint - shift) < TOLERANCE, f"{path}: {adjoint} vs {shift}"


def _adjoint_and_shift(code):
    """``code``'s result with the adjoint, then with the shift fallback."""
    adjoint = _child(code, PLENTY)
    shift = _child(code, ONE_BUT_NOT_TWO)
    assert not _fell_back(adjoint), adjoint.stdout
    assert _fell_back(shift), "the 300-byte budget must take the shift rule"
    return _result(adjoint), _result(shift)


JACOBIAN = """
by_params, by_inputs = qml.jacobian(model, rows, params, wrt="both", exact=True)
print("RESULT", json.dumps([by_params.tolist(), by_inputs.tolist()]))
"""


def test_qml_jacobian_is_the_same_by_the_adjoint_and_by_the_shift():
    adjoint, shift = _adjoint_and_shift(JACOBIAN)
    _assert_close(adjoint, shift)


def test_without_memory_for_one_statevector_the_error_is_the_exact_paths():
    process = _child(
        """
        try:
            qml.jacobian(model, rows, params, exact=True)
        except polypus.InsufficientMemoryError as e:
            print("RESULT", json.dumps(str(e)))
        """,
        NOT_EVEN_ONE,
    )
    message = _result(process)
    assert "4-qubit statevector" in message, message


def test_model_fit_with_adam_trains_the_same_by_the_adjoint_and_by_the_shift():
    # The same trajectory: every exact gradient is the same to rounding, so
    # Adam's steps, and where they lead, agree to far below the tolerance.
    adjoint, shift = _adjoint_and_shift(
        """
        result = model.fit(
            rows, [0, 1, 1], method=polypus.Adam(max_iters=6, patience=100),
            loss="categorical_cross_entropy", infrastructure="local",
            backend="polypus", exact=True, seed=5,
        )
        print("RESULT", json.dumps([result.best_fitness, list(result.best_params)]))
        """
    )
    _assert_close(adjoint, shift)


def test_quantum_layer_backward_is_the_same_by_the_adjoint_and_by_the_shift():
    pytest.importorskip("torch")
    adjoint, shift = _adjoint_and_shift(
        """
        import torch
        from polypus.torch import QuantumLayer

        layer = QuantumLayer(model, initial_params=params, exact=True, seed=3)
        x = torch.tensor(rows, dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.0, 0.75]], dtype=torch.float64)
        (layer(x) * weights).sum().backward()
        print("RESULT", json.dumps([layer.params.grad.tolist(), x.grad.tolist()]))
        """
    )
    _assert_close(adjoint, shift)
