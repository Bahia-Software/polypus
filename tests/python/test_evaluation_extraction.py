"""
Result-extraction failures on the evaluation path (issue #81 follow-up).

``run_and_evaluate`` is the single caller of ``polypus_python.expectation_values``
(used by ``polypus.train`` via ``VqcOracle`` and ``polypus.qml.train`` via
``QmlOracle``). When ``expectation_values`` *succeeds* but returns a value whose
shape isn't ``list[float]``, converting it Rust-side fails — that is a
``EvaluationError::Conversion``, surfaced as ``polypus.EvaluationError``, not the
bare ``TypeError`` PyO3's ``extract()`` emits and not a verbatim Python
exception (which is reserved for a genuinely *raised* callback/seam error).

This mirrors the ``test_seam_extraction.py`` tests for the ``run_qcs`` /
``connect_to_infrastructure`` seam. It exercises the shortest reachable path,
``polypus.train`` with ``infrastructure="local"`` (no SLURM), and monkeypatches
``expectation_values`` so the wrong shape is forced deterministically.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]


def _train(parametrized_circuit, simple_expectation_fn, run_id):
    import polypus

    return polypus.train(
        parametrized_circuit,
        polypus.DE(generations=2, population_size=4, tolerance=0.5),
        shots=64,
        n_qpus=1,
        dimensions=1,
        expectation_function=simple_expectation_fn,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id=run_id,
    )


def test_expectation_values_returning_scalar_is_evaluation_error(
    monkeypatch, parametrized_circuit, simple_expectation_fn
):
    # expectation_values succeeds but returns a bare float instead of
    # list[float]. The Rust-side extraction failure must be EvaluationError,
    # not the plain TypeError extract() would surface pre-fix.
    import polypus
    import polypus_python

    monkeypatch.setattr(polypus_python, "expectation_values", lambda *a, **k: 0.5)
    with pytest.raises(polypus.EvaluationError) as excinfo:
        _train(parametrized_circuit, simple_expectation_fn, "eval_extract_scalar")
    assert not isinstance(excinfo.value, TypeError), (
        "a wrong-shaped return value must not be reported as the plain "
        "TypeError that PyO3's extract() emits"
    )


def test_expectation_values_returning_none_is_evaluation_error(
    monkeypatch, parametrized_circuit, simple_expectation_fn
):
    import polypus
    import polypus_python

    monkeypatch.setattr(polypus_python, "expectation_values", lambda *a, **k: None)
    with pytest.raises(polypus.EvaluationError):
        _train(parametrized_circuit, simple_expectation_fn, "eval_extract_none")


def test_expectation_callback_exception_propagates_verbatim(
    monkeypatch, parametrized_circuit
):
    # Negative case: when the user's expectation_function *raises* (rather than
    # returning a wrong shape), that genuine Python exception must propagate
    # verbatim as itself, not be reclassified as polypus.EvaluationError.
    import polypus

    def exploding_expectation(_bitstring):
        raise ValueError("user callback blew up")

    with pytest.raises(ValueError, match="user callback blew up") as excinfo:
        _train(parametrized_circuit, exploding_expectation, "eval_callback_raises")
    assert not isinstance(excinfo.value, polypus.EvaluationError), (
        "a genuine raised callback exception must not be reclassified as "
        "polypus.EvaluationError"
    )
