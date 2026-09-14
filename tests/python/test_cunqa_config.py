"""CUNQA provider configuration must be portable and honour the requested
resources.

CUNQA itself is an optional dependency that is not installed in CI or this
environment, so the ``cunqa.qjob`` / ``cunqa.qpu`` modules are stubbed to capture
exactly what :meth:`Cunqa.get_qpus` forwards to ``qraise``. These pin three
regressions of the pre-fix code: ``cores_per_qpu`` was read by the Rust backend
and passed across the seam but then *dropped* by ``get_qpus``; the vQPU backend
spec was a hardcoded absolute path to one developer's home; and the module
appended ``$HOME`` to ``sys.path`` unconditionally.
"""

import sys
import types

import pytest


@pytest.fixture
def stub_cunqa(monkeypatch):
    """Inject fake ``cunqa.*`` modules (so ``polypus_python.cunqa`` imports with
    no real CUNQA present) and capture the ``qraise`` call."""
    captured = {}

    def fake_qraise(n, t, **kwargs):
        captured["n"] = n
        captured["t"] = t
        captured["kwargs"] = kwargs
        return "fake-family"

    qpu = types.ModuleType("cunqa.qpu")
    qpu.get_QPUs = lambda **k: []
    qpu.qdrop = lambda *a, **k: None
    qpu.qraise = fake_qraise
    qpu.run = lambda *a, **k: None
    qjob = types.ModuleType("cunqa.qjob")
    qjob.gather = lambda jobs: []

    monkeypatch.setitem(sys.modules, "cunqa", types.ModuleType("cunqa"))
    monkeypatch.setitem(sys.modules, "cunqa.qpu", qpu)
    monkeypatch.setitem(sys.modules, "cunqa.qjob", qjob)
    # Force a fresh import of the module under test with the stubs in place.
    monkeypatch.delitem(sys.modules, "polypus_python.cunqa", raising=False)
    return captured


def test_get_qpus_forwards_cores_per_qpu(stub_cunqa, monkeypatch):
    monkeypatch.delenv("POLYPUS_CUNQA_BACKEND", raising=False)
    from polypus_python.cunqa import Cunqa

    Cunqa().get_qpus(n=2, t="1:00:00", n_nodes=1, family_name="fam", cores_per_qpu=4)

    kwargs = stub_cunqa["kwargs"]
    assert kwargs.get("cores") == 4, "cores_per_qpu must reach qraise, not be dropped"


def test_get_qpus_has_no_hardcoded_backend(stub_cunqa, monkeypatch):
    # With the env var unset, no backend is forwarded (CUNQA uses its default) --
    # and certainly not a hardcoded absolute path.
    monkeypatch.delenv("POLYPUS_CUNQA_BACKEND", raising=False)
    from polypus_python.cunqa import Cunqa

    Cunqa().get_qpus(n=1, t="1:00:00", n_nodes=1, family_name="fam", cores_per_qpu=1)
    assert "backend" not in stub_cunqa["kwargs"]


def test_get_qpus_backend_comes_from_env(stub_cunqa, monkeypatch):
    monkeypatch.setenv("POLYPUS_CUNQA_BACKEND", "/portable/backend.json")
    from polypus_python.cunqa import Cunqa

    Cunqa().get_qpus(n=1, t="1:00:00", n_nodes=1, family_name="fam", cores_per_qpu=1)
    assert stub_cunqa["kwargs"].get("backend") == "/portable/backend.json"


def test_no_hardcoded_personal_path_in_source():
    # Regression guard: the developer-specific absolute path must never return.
    import inspect

    import polypus_python.cunqa as cunqa_module

    source = inspect.getsource(cunqa_module)
    assert "/mnt/netapp2/Store_uni/home/empresa" not in source


def test_import_polypus_python_does_not_touch_home(monkeypatch):
    # The package __init__ used to `sys.path.append(os.getenv("HOME"))` on every
    # import (appending None when HOME was unset). Importing it fresh with HOME
    # cleared must succeed and must not add a stray entry to sys.path.
    monkeypatch.delenv("HOME", raising=False)
    for name in [m for m in sys.modules if m.startswith("polypus_python")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    import polypus_python  # noqa: F401

    assert None not in sys.path, "__init__ must not append os.getenv('HOME') (None)"
