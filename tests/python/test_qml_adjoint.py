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
``info`` record. The ``QuantumLayer`` case needs PyTorch, so it lives in
``test_torch_quantum_layer.py``.
"""

from qml_adjoint_helpers import (
    NOT_EVEN_ONE,
    adjoint_and_shift,
    assert_close,
    child_result,
    run_child,
)

JACOBIAN = """
by_params, by_inputs = qml.jacobian(model, rows, params, wrt="both", exact=True)
print("RESULT", json.dumps([by_params.tolist(), by_inputs.tolist()]))
"""


def test_qml_jacobian_is_the_same_by_the_adjoint_and_by_the_shift():
    adjoint, shift = adjoint_and_shift(JACOBIAN)
    assert_close(adjoint, shift)


def test_without_memory_for_one_statevector_the_error_is_the_exact_paths():
    process = run_child(
        """
        try:
            qml.jacobian(model, rows, params, exact=True)
        except polypus.InsufficientMemoryError as e:
            print("RESULT", json.dumps(str(e)))
        """,
        NOT_EVEN_ONE,
    )
    message = child_result(process)
    assert "4-qubit statevector" in message, message


def test_model_fit_with_adam_trains_the_same_by_the_adjoint_and_by_the_shift():
    # The same trajectory: every exact gradient is the same to rounding, so
    # Adam's steps, and where they lead, agree to far below the tolerance.
    adjoint, shift = adjoint_and_shift(
        """
        result = model.fit(
            rows, [0, 1, 1], method=polypus.Adam(max_iters=6, patience=100),
            loss="categorical_cross_entropy", infrastructure="local",
            backend="polypus", exact=True, seed=5,
        )
        print("RESULT", json.dumps([result.best_fitness, list(result.best_params)]))
        """
    )
    assert_close(adjoint, shift)
