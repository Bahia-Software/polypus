"""
Result-extraction failures at the ``polypus_python`` seam (issue #81).

Contract C-1 re-raises a failure *thrown by the Python seam* verbatim (e.g.
``ValueError`` for an unknown infrastructure, ``TypeError`` for a bad kwarg).
But a seam call that *succeeds* and returns a value of the wrong shape is a
Rust-side data-conversion failure, not a seam exception, so it must surface as
``polypus.BackendError`` (mapped from ``BackendError::Conversion``), not as a
bare ``TypeError`` from PyO3's extraction.

These tests monkeypatch the seam (no SLURM / hardware needed), mirroring the
style of ``test_seam_contract.py``. They exercise the ``local`` path: a real
``connect_to_infrastructure("local")`` with a mocked ``run_qcs`` for the
run-result case, and a mocked ``connect_to_infrastructure`` for the handle case.
"""

import pytest


def _native_qc():
    import polypus

    return polypus.Circuit(1).h(0).measure_all()


def _run_local():
    import polypus

    return polypus.run_quantum_circuit(
        _native_qc(), shots=10, infrastructure="local", backend="aer"
    )


def test_run_qcs_returning_non_list_of_dicts_is_backend_error(monkeypatch):
    # run_qcs succeeds but returns list[str] instead of list[dict[str, int]].
    # The extraction failure is Rust-side, so it must be BackendError, not the
    # raw TypeError that C-1 reserves for a genuine bad-kwarg seam failure.
    import polypus
    import polypus_python

    monkeypatch.setattr(polypus_python, "run_qcs", lambda *a, **k: ["not", "dicts"])
    with pytest.raises(polypus.BackendError) as excinfo:
        _run_local()
    assert not isinstance(excinfo.value, TypeError), (
        "a wrong-shaped return value must not be reported as the plain "
        "TypeError that C-1 uses for a bad kwarg"
    )


def test_run_qcs_returning_none_is_backend_error(monkeypatch):
    import polypus
    import polypus_python

    monkeypatch.setattr(polypus_python, "run_qcs", lambda *a, **k: None)
    with pytest.raises(polypus.BackendError):
        _run_local()


def test_connect_returning_non_str_is_backend_error(monkeypatch):
    # connect_to_infrastructure succeeds but returns an int instead of a str
    # connection handle: again a Rust-side extraction failure => BackendError.
    import polypus
    import polypus_python

    monkeypatch.setattr(polypus_python, "connect_to_infrastructure", lambda *a, **k: 42)
    with pytest.raises(polypus.BackendError) as excinfo:
        _run_local()
    assert not isinstance(excinfo.value, TypeError)
