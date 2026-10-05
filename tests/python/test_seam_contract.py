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
from qiskit.exceptions import QiskitError


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


# The exact kwargs the Rust local backend sends to `run_qcs` (C-1 table). Recorded
# in a fresh child so `POLYPUS_MEM_BUDGET` can be set without mutating this
# process's environment while Rust threads may read it.
_RECORD_LOCAL_KWARGS = """
import json
import polypus
import polypus_python

seen = {}

def record(infrastructure, **kwargs):
    seen["infrastructure"] = infrastructure
    seen.update({k: v for k, v in kwargs.items() if k != "qcs"})
    seen["keys"] = sorted(kwargs)
    return [{"0": kwargs["shots"]} for _ in kwargs["qcs"]]

polypus_python.run_qcs = record
polypus.run_quantum_circuit(
    polypus.Circuit(1).h(0).measure_all(), shots=10, infrastructure="local",
    backend="aer", seed=3)
print(json.dumps(seen))
"""

_LOCAL_REQUIRED_KWARGS = {
    "id",
    "backend",
    "qcs",
    "shots",
    "sim_method",
    "max_parallel_experiments",
    "seed",
}


def _local_kwargs(budget, tmp_path):
    import json
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env.pop("POLYPUS_MEM_BUDGET", None)
    if budget is not None:
        env["POLYPUS_MEM_BUDGET"] = budget
    env["XDG_CACHE_HOME"] = str(tmp_path)
    env["POLYPUS_NO_AUTOCALIBRATE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _RECORD_LOCAL_KWARGS],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_local_run_qcs_kwargs_carry_max_memory_mb_for_a_known_budget(tmp_path):
    # C-1: with a known budget the local backend adds `max_memory_mb` (the budget
    # in MiB) to the frozen kwarg set, and nothing else.
    seen = _local_kwargs("100M", tmp_path)
    assert seen["infrastructure"] == "local"
    assert set(seen["keys"]) == _LOCAL_REQUIRED_KWARGS | {"max_memory_mb"}
    assert seen["max_memory_mb"] == 100


def test_local_run_qcs_kwargs_follow_the_budget_source_when_unset(tmp_path):
    # Unset budget: detected (Linux: /proc/meminfo) => `max_memory_mb` is sent;
    # nothing detectable (macOS/Windows) => the 16 GiB fallback is a guess and is
    # never imposed on Aer, so the kwarg is absent. (A Linux host always detects,
    # so there the Fallback => absent mapping itself is pinned by the Rust unit
    # test `aer_max_memory_mb_is_the_known_budget_in_mib_and_never_zero`.)
    import os

    seen = _local_kwargs(None, tmp_path)
    detectable = os.path.exists("/proc/meminfo")
    if detectable:
        assert set(seen["keys"]) == _LOCAL_REQUIRED_KWARGS | {"max_memory_mb"}
        assert seen["max_memory_mb"] >= 1
    else:
        assert set(seen["keys"]) == _LOCAL_REQUIRED_KWARGS


# ── Qiskit exceptions join the polypus hierarchy (#218) ──────────────────────
#
# A seam exception whose class comes from a ``qiskit*`` module (anywhere in its
# MRO) reaches the caller as ``polypus.BackendError`` with the original chained
# as ``__cause__``; every other exception keeps being re-raised verbatim, and so
# does a Qiskit class that is also a ``ValueError``/``TypeError``/
# ``KeyboardInterrupt`` (C-1's typed failure modes win).


def _raise_from_seam(monkeypatch, exc):
    import polypus
    import polypus_python

    def failing(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(polypus_python, "run_qcs", failing)
    polypus.run_quantum_circuit(
        _native_qc(), shots=10, infrastructure="local", backend="aer"
    )


class UserQiskitError(QiskitError):
    """Defined outside qiskit: only its MRO says it is a Qiskit error."""


def _qiskit_errors():
    from qiskit.qasm2 import QASM2ParseError

    errors = [
        QiskitError("no counts"),
        QASM2ParseError("bad line"),
        UserQiskitError("subclassed"),
    ]
    try:
        from qiskit_aer import AerError
    except ImportError:
        pass
    else:
        errors.append(AerError("aer failed"))
    return errors


@pytest.mark.parametrize(
    "error", _qiskit_errors(), ids=lambda error: type(error).__name__
)
def test_seam_qiskit_error_becomes_backend_error(monkeypatch, error):
    import polypus

    with pytest.raises(polypus.BackendError) as info:
        _raise_from_seam(monkeypatch, error)
    exc = info.value
    assert isinstance(exc, polypus.PolypusError)
    assert type(exc) is polypus.BackendError
    # The Qiskit class (its fully qualified name, read from the class because
    # the module layout is Qiskit's and may move) and its message survive, and
    # the original is chained so its traceback is not lost.
    qualname = f"{type(error).__module__}.{type(error).__qualname__}"
    assert qualname in str(exc)
    assert str(error) in str(exc)
    assert exc.__cause__ is error


@pytest.mark.parametrize(
    "exc_type", [RuntimeError, TypeError, ValueError, ZeroDivisionError, KeyError]
)
def test_seam_non_qiskit_error_is_reraised_verbatim(monkeypatch, exc_type):
    import polypus

    error = exc_type("not from qiskit")
    with pytest.raises(exc_type) as info:
        _raise_from_seam(monkeypatch, error)
    assert info.value is error
    assert not isinstance(info.value, polypus.PolypusError)


@pytest.mark.parametrize("builtin", [ValueError, TypeError, KeyboardInterrupt])
def test_seam_qiskit_error_that_is_also_a_c1_type_is_preserved(monkeypatch, builtin):
    import polypus

    both = type("QiskitAnd" + builtin.__name__, (QiskitError, builtin), {})
    error = both("typed failure")
    with pytest.raises(builtin) as info:
        _raise_from_seam(monkeypatch, error)
    assert info.value is error
    assert not isinstance(info.value, polypus.PolypusError)


def test_real_aer_error_reaches_the_caller_as_polypus_error():
    """No monkeypatch: ``ch`` is outside Aer's basis (ENGINEERING §7), so Aer
    itself raises ``AerError('unknown instruction: ch')``."""
    import polypus

    pytest.importorskip("qiskit_aer")
    qc = polypus.Circuit(2).x(0).ch(0, 1).measure_all()
    with pytest.raises(polypus.PolypusError) as info:
        polypus.run_quantum_circuit(qc, shots=10, infrastructure="local", backend="aer")
    assert type(info.value) is polypus.BackendError
    assert "AerError" in str(info.value)
    assert type(info.value.__cause__).__name__ == "AerError"
    assert str(info.value.__cause__) in str(info.value)


def test_real_qasm_parse_error_reaches_the_caller_as_polypus_error():
    """No monkeypatch: the Aer path parses QASM with Qiskit, whose
    ``QASM2ParseError`` used to escape the polypus hierarchy."""
    import polypus
    from qiskit.qasm2 import QASM2ParseError

    with pytest.raises(polypus.BackendError) as info:
        polypus.run_quantum_circuit(
            "OPENQASM 2.0;\nqreg q[1];\nfoo q[0];\n",
            shots=10,
            infrastructure="local",
            backend="aer",
        )
    assert "QASM2ParseError" in str(info.value)
    assert isinstance(info.value.__cause__, QASM2ParseError)


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
