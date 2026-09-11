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

The panic-safety tests deliberately exercise the ``local`` path (real
``connect_to_infrastructure("local")`` + mocked ``run_qcs``). The CUNQA
``disconnect`` path is covered separately below: it forwards the ``family``
handle to ``qdrop`` (CONTRACTS.md C-1). This used to be a "known break" — the
Python side read ``slurm_job_id`` (a key the Rust side never sends), so a
``KeyError`` fired before ``qdrop`` ran and the QPU allocation leaked; the test
below locks in the fix without needing a real ``cunqa`` install or SLURM.
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


def _install_fake_cunqa(monkeypatch, dropped):
    """Register a minimal fake ``cunqa`` package in ``sys.modules`` so the CUNQA
    seam imports without a real install or SLURM. ``qdrop`` records the family
    names it is handed; the rest are inert stubs to satisfy the module-level
    ``from cunqa.qpu import ...`` in ``polypus_python.cunqa``."""
    import sys
    import types

    qjob_mod = types.ModuleType("cunqa.qjob")
    qjob_mod.gather = lambda *a, **k: []
    qpu_mod = types.ModuleType("cunqa.qpu")
    qpu_mod.get_QPUs = lambda *a, **k: []
    qpu_mod.qraise = lambda *a, **k: "fam-1"
    qpu_mod.run = lambda *a, **k: None
    qpu_mod.qdrop = lambda *families, **k: dropped.extend(families)

    monkeypatch.setitem(sys.modules, "cunqa", types.ModuleType("cunqa"))
    monkeypatch.setitem(sys.modules, "cunqa.qjob", qjob_mod)
    monkeypatch.setitem(sys.modules, "cunqa.qpu", qpu_mod)
    # Force a fresh import so the fake ``qdrop`` is the one bound in the module.
    monkeypatch.delitem(sys.modules, "polypus_python.cunqa", raising=False)


def test_cunqa_disconnect_forwards_family_to_qdrop(monkeypatch):
    # C-1: the Rust side calls disconnect_from_infrastructure("cunqa",
    # family=<handle>). The Python side must forward that handle to CUNQA's
    # `qdrop` — regression guard for the historical break where it read
    # `slurm_job_id` (never sent) and `qdrop` was never reached.
    import polypus_python

    dropped = []
    _install_fake_cunqa(monkeypatch, dropped)

    polypus_python.disconnect_from_infrastructure("cunqa", family="fam-1")

    assert dropped == ["fam-1"], "the family handle must reach qdrop unchanged"
