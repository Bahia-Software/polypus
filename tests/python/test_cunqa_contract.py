"""The CUNQA seal must call CUNQA with names its real signatures accept.

``qraise`` has no ``**kwargs``, so a name it does not know raises ``TypeError``:
CUNQA renamed ``cores`` to ``cores_per_qpu`` in 2.4.0 and the seal kept sending
``cores`` (issue #204). The supported range is ``cunqa>=2.4,<4``.

No CUNQA is installed here. ``data/cunqa/signatures.json`` holds the parameter
lists of the functions the seal calls, extracted from the sources at the tags
2.3.0, 2.4.0, 3.0.0 and 3.1.0 (``data/cunqa/extract_signatures.py``). Each test
installs a fake ``cunqa`` whose functions carry the ``inspect.Signature`` of one
tag and bind every call against it, so this checks the call shapes the seal
sends, not CUNQA's behaviour; ``test_cunqa_run_flow.py`` drives the whole run
flow against a stateful fake built on the same signatures.
"""

import sys
import types

import pytest
from fake_cunqa import SIGNATURES, SUPPORTED_TAGS
from fake_cunqa import signature as _signature
from qiskit import QuantumCircuit

SUPPORTED = SUPPORTED_TAGS
UNSUPPORTED = ["2.3.0"]


def _checked(signature, result):
    def fake(*args, **kwargs):
        signature.bind(*args, **kwargs)
        return result(*args, **kwargs) if callable(result) else result

    return fake


@pytest.fixture
def install_cunqa(monkeypatch):
    """Install a fake ``cunqa`` with the signatures of ``tag`` and return the
    calls the seal made, as ``{function: [(args, kwargs), ...]}``."""

    def install(tag):
        sigs = {n: _signature(p) for n, p in SIGNATURES[tag].items()}
        calls = {}

        def recording(name, result):
            def record(*args, **kwargs):
                calls.setdefault(name, []).append((args, kwargs))
                return result

            return _checked(sigs[name], record)

        qpu = types.ModuleType("cunqa.qpu")
        qpu.qraise = recording("qraise", "family-1")
        qpu.qdrop = recording("qdrop", None)
        qpu.get_QPUs = recording("get_QPUs", ["qpu-0", "qpu-1"])
        qpu.run = recording("run", "qjob")
        qjob = types.ModuleType("cunqa.qjob")
        qjob.gather = recording(
            "gather", [types.SimpleNamespace(counts={"00": 4, "11": 4})] * 2
        )

        monkeypatch.setitem(sys.modules, "cunqa", types.ModuleType("cunqa"))
        monkeypatch.setitem(sys.modules, "cunqa.qpu", qpu)
        monkeypatch.setitem(sys.modules, "cunqa.qjob", qjob)
        monkeypatch.delitem(sys.modules, "polypus_python.cunqa", raising=False)
        monkeypatch.delenv("POLYPUS_CUNQA_BACKEND", raising=False)
        from polypus_python.cunqa import Cunqa

        return Cunqa(), calls

    return install


# The kwargs `CunqaBackend::raise_qpus` sends (contract C-1).
GET_QPUS = dict(n=2, t="1:00:00", n_nodes=1, family_name="fam", cores_per_qpu=4)


def test_every_tag_is_extracted():
    assert sorted(SIGNATURES) == sorted(SUPPORTED + UNSUPPORTED)


def test_cunqa_renamed_cores_in_2_4():
    # The fact behind the supported range, read off the extracted signatures.
    for tag in UNSUPPORTED:
        params = _signature(SIGNATURES[tag]["qraise"]).parameters
        assert "cores" in params and "cores_per_qpu" not in params
    for tag in SUPPORTED:
        params = _signature(SIGNATURES[tag]["qraise"]).parameters
        assert "cores_per_qpu" in params and "cores" not in params


@pytest.mark.parametrize("tag", SUPPORTED)
def test_get_qpus_binds_against_qraise(install_cunqa, tag):
    cunqa, calls = install_cunqa(tag)

    assert cunqa.get_qpus(**GET_QPUS) == "family-1"

    ((args, kwargs),) = calls["qraise"]
    assert args == (2, "1:00:00")
    assert kwargs["cores_per_qpu"] == 4
    assert "cores" not in kwargs


@pytest.mark.parametrize("tag", SUPPORTED)
def test_old_cores_name_is_rejected(tag):
    signature = _signature(SIGNATURES[tag]["qraise"])
    with pytest.raises(TypeError, match="cores"):
        signature.bind(2, "1:00:00", n_nodes=1, family="fam", cores=4)


@pytest.mark.parametrize("tag", UNSUPPORTED)
def test_2_3_is_unsupported_by_design(install_cunqa, tag):
    # 2.3.0 spells the argument `cores`, so the reservation now fails with
    # qraise's own TypeError. It is outside the supported range, not a bug.
    cunqa, _ = install_cunqa(tag)
    with pytest.raises(TypeError, match="cores_per_qpu"):
        cunqa.get_qpus(**GET_QPUS)


@pytest.mark.parametrize("tag", SUPPORTED)
def test_backend_env_binds_against_qraise(install_cunqa, monkeypatch, tag):
    cunqa, calls = install_cunqa(tag)
    monkeypatch.setenv("POLYPUS_CUNQA_BACKEND", "/portable/backend.json")

    cunqa.get_qpus(**GET_QPUS)

    assert calls["qraise"][0][1]["backend"] == "/portable/backend.json"


@pytest.mark.parametrize("tag", SUPPORTED)
def test_run_and_drop_bind_against_cunqa(install_cunqa, tag):
    cunqa, calls = install_cunqa(tag)
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    counts = cunqa.run_qcs(
        family_id="fam", qcs=[qc, qc], shots=8, sim_method="statevector", seed=7
    )
    cunqa.drop_qpus(family="fam")

    assert counts == [{"00": 4, "11": 4}] * 2
    assert calls["get_QPUs"][0][1] == {"co_located": True, "family": "fam"}
    assert [c[1]["seed"] for c in calls["run"]] == [7, 8]
    assert calls["run"][0][1]["method"] == "statevector"
    assert calls["qdrop"] == [(("fam",), {})]
