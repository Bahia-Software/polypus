"""
Enforcing test for contract C-1 (Rust → Python execution seam).

C-1 freezes the three ``polypus_python`` functions the Rust orchestration layer
calls and their documented failure modes. This test runs **without SLURM**: the
seam is exercised by monkeypatching ``polypus_python.run_qcs`` so a failure can
be forced deterministically.

It locks in the panic-safety guarantee introduced with the typed error
hierarchy: a failure crossing the seam surfaces as a proper Python exception,
never a ``pyo3_runtime.PanicException`` / interpreter crash, and the C-1 failure
types (``ValueError`` for an unknown infrastructure, ``TypeError`` for a bad
kwarg) are preserved because the Rust side re-raises the original exception
verbatim.

Note: this deliberately exercises the ``local`` path (real
``connect_to_infrastructure("local")`` + mocked ``run_qcs``). It does **not**
touch the CUNQA ``disconnect`` path, which still carries the separately-tracked
"known break" (reads ``slurm_job_id`` instead of ``family``, CONTRACTS.md C-1);
that defect is out of scope here and is not exercised or masked.
"""

import pytest


def _native_qc():
    import polypus

    return polypus.Circuit(1).h(0).measure_all()


def test_unknown_infrastructure_raises_value_error():
    # Rejected before any seam call; C-1 says ValueError, never a panic.
    import polypus

    with pytest.raises(ValueError):
        polypus.run_quantum_circuit(_native_qc(), shots=10, infrastructure="nope")


def test_seam_type_error_is_preserved(monkeypatch):
    # C-1: an unexpected/missing kwarg raises TypeError on the Python side. It
    # must reach the caller as TypeError, not a PanicException.
    import polypus
    import polypus_python

    def bad_kwarg(*_args, **_kwargs):
        raise TypeError("run_qcs() got an unexpected keyword argument 'bogus'")

    monkeypatch.setattr(polypus_python, "run_qcs", bad_kwarg)
    with pytest.raises(TypeError):
        polypus.run_quantum_circuit(
            _native_qc(), shots=10, infrastructure="local", backend="aer"
        )


def test_seam_runtime_failure_is_not_panic(monkeypatch):
    # A generic execution failure at the seam must surface as the original
    # Python exception (propagated verbatim), never a PanicException / abort.
    import polypus
    import polypus_python

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated backend execution failure")

    monkeypatch.setattr(polypus_python, "run_qcs", boom)
    with pytest.raises(RuntimeError, match="simulated backend execution failure"):
        polypus.run_quantum_circuit(
            _native_qc(), shots=10, infrastructure="local", backend="aer"
        )


def test_seam_failure_is_never_a_panic_exception(monkeypatch):
    """Whatever the seam raises, the caller never sees pyo3's PanicException."""
    import polypus
    import polypus_python

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(polypus_python, "run_qcs", boom)
    try:
        polypus.run_quantum_circuit(
            _native_qc(), shots=10, infrastructure="local", backend="aer"
        )
    except BaseException as exc:  # noqa: BLE001 - we assert on the type below
        assert type(exc).__name__ != "PanicException", (
            "a seam failure must not surface as a Rust panic"
        )
        assert isinstance(exc, RuntimeError)
    else:
        pytest.fail("expected the mocked seam failure to raise")
