"""
Cost-callback result handling on the evaluation path (issue #81 follow-up,
updated for the native cost-observable rewrite).

A Python ``expectation_function`` is wrapped in a ``PyCallbackObservable``, which
calls it once per unique bitstring and extracts an ``f64`` from each result.
Two failure modes are pinned here:

* the callback returns a **non-numeric** value — the per-bitstring ``f64``
  extraction fails, and that failure surfaces as the callback's own Python
  exception (a ``TypeError``) **verbatim**: never a ``pyo3_runtime.PanicException``
  and never a silently-poisoned result. Unlike the former batch
  ``polypus_python.expectation_values`` seam, there is no separate list-shape
  conversion step to reclassify — each result is extracted individually, so the
  natural Python error is carried through as-is.
* the callback **raises** — that genuine exception propagates verbatim as
  itself, not reclassified as ``polypus.EvaluationError``.

A non-finite (NaN/inf) numeric return is the one shape that becomes a typed
``polypus.EvaluationError`` (contract C-5); that is covered in
``test_oracle_contract.py``.

Exercises the shortest reachable path, ``polypus.train`` with
``infrastructure="local"`` (no SLURM).
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]


def _train(parametrized_circuit, expectation_fn, run_id):
    import polypus

    return polypus.train(
        parametrized_circuit,
        polypus.DE(generations=2, population_size=4, tolerance=0.5),
        shots=64,
        n_qpus=1,
        dimensions=1,
        expectation_function=expectation_fn,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id=run_id,
    )


def test_callback_returning_none_surfaces_typeerror_not_panic(parametrized_circuit):
    # The callback returns None, so the per-bitstring f64 extraction fails. The
    # failure must surface as the natural Python TypeError, never a Rust panic
    # and never a silent result.
    def none_expectation(_bitstring: str):
        return None

    try:
        _train(parametrized_circuit, none_expectation, "eval_extract_none")
    except BaseException as exc:  # noqa: BLE001 - we assert on the type below
        assert type(exc).__name__ != "PanicException", (
            "a wrong-typed callback result must not surface as a Rust panic"
        )
        assert isinstance(exc, TypeError)
    else:
        pytest.fail("expected the None return value to raise")


def test_callback_returning_non_numeric_surfaces_typeerror(parametrized_circuit):
    # A non-numeric (str) return likewise fails the f64 extraction and surfaces
    # as the callback's own TypeError.
    def str_expectation(_bitstring: str):
        return "not a number"

    with pytest.raises(TypeError):
        _train(parametrized_circuit, str_expectation, "eval_extract_str")


def test_expectation_callback_exception_propagates_verbatim(parametrized_circuit):
    # When the user's expectation_function *raises* (rather than returning a
    # wrong type), that genuine Python exception must propagate verbatim as
    # itself, not be reclassified as polypus.EvaluationError.
    import polypus

    def exploding_expectation(_bitstring):
        raise ValueError("user callback blew up")

    with pytest.raises(ValueError, match="user callback blew up") as excinfo:
        _train(parametrized_circuit, exploding_expectation, "eval_callback_raises")
    assert not isinstance(excinfo.value, polypus.EvaluationError), (
        "a genuine raised callback exception must not be reclassified as "
        "polypus.EvaluationError"
    )
