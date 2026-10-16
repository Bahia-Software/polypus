"""The SLURM walltime of the CUNQA virtual QPUs is a kwarg (issue #41).

Every entry point that can raise CUNQA QPUs takes ``walltime`` (``HH:MM:SS``,
default ``"10:00:00"``) and sends it across the C-1 seam as ``t``. A malformed
value is a ``ValueError`` at the edge, before ``connect_to_infrastructure`` (and
so ``qraise``) is reached. ``local`` and ``qmio`` ignore the kwarg.

No SLURM or CUNQA is needed: ``connect_to_infrastructure`` is replaced by a
recorder that stops the run once it has seen the kwargs.
"""

import pytest


class _Stop(Exception):
    """Raised by the recorder to end the run after the seam call."""


@pytest.fixture
def seam_calls(monkeypatch):
    import polypus_python

    calls = []

    def record(infrastructure, **kwargs):
        calls.append((infrastructure, kwargs))
        raise _Stop

    monkeypatch.setattr(polypus_python, "connect_to_infrastructure", record)
    return calls


def _circuit():
    import polypus

    return polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()


def _run(**kw):
    import polypus

    return polypus.run_quantum_circuit(
        polypus.Circuit(1).h(0).measure_all(), shots=8, infrastructure="cunqa", **kw
    )


def _train(**kw):
    import polypus

    return polypus.train(
        _circuit(),
        polypus.DE(generations=1, population_size=4),
        shots=8,
        n_qpus=1,
        dimensions=1,
        expectation_function=lambda b: 0.0,
        infrastructure="cunqa",
        nodes=1,
        cores_per_qpu=1,
        id="walltime",
        **kw,
    )


def _qml_model():
    from qiskit.circuit import ParameterVector, QuantumCircuit

    x = ParameterVector("x", 1)
    theta = ParameterVector("theta", 1)
    feature_map = QuantumCircuit(1)
    feature_map.ry(x[0], 0)
    ansatz = QuantumCircuit(1)
    ansatz.ry(theta[0], 0)
    return feature_map, ansatz


def _qml_train(**kw):
    import polypus

    feature_map, ansatz = _qml_model()
    return polypus.qml.train(
        feature_map,
        ansatz,
        [[0.1], [0.2]],
        polypus.DE(generations=1, population_size=4),
        shots=8,
        n_qpus=1,
        dimensions=1,
        expectation_function=lambda b: 0.0,
        infrastructure="cunqa",
        nodes=1,
        cores_per_qpu=1,
        id="walltime",
        **kw,
    )


def _qml_predict(**kw):
    import polypus

    feature_map, ansatz = _qml_model()
    return polypus.qml.predict(
        feature_map, ansatz, [[0.1]], [0.2], shots=8, infrastructure="cunqa", **kw
    )


ENTRY_POINTS = [_run, _train, _qml_train, _qml_predict]
IDS = ["run_quantum_circuit", "train", "qml.train", "qml.predict"]


@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=IDS)
def test_walltime_reaches_the_seam_as_t(seam_calls, entry):
    with pytest.raises(_Stop):
        entry(walltime="48:00:00")

    ((infrastructure, kwargs),) = seam_calls
    assert infrastructure == "cunqa"
    assert kwargs["t"] == "48:00:00"
    assert set(kwargs) == {"n", "t", "n_nodes", "family_name", "cores_per_qpu"}


@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=IDS)
def test_omitted_walltime_keeps_the_ten_hour_default(seam_calls, entry):
    with pytest.raises(_Stop):
        entry()

    assert seam_calls[0][1]["t"] == "10:00:00"


@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=IDS)
@pytest.mark.parametrize(
    "bad", ["00:00:00", "1:2:3", "1-12:00:00", "10:60:00", "10:00", "", "soon"]
)
def test_malformed_walltime_is_a_value_error_before_the_seam(seam_calls, entry, bad):
    with pytest.raises(ValueError, match="HH:MM:SS"):
        entry(walltime=bad)

    assert seam_calls == []


@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=IDS)
def test_non_string_walltime_is_a_type_error_before_the_seam(seam_calls, entry):
    with pytest.raises(TypeError):
        entry(walltime=36000)

    assert seam_calls == []


def test_local_ignores_walltime():
    import polypus

    result = polypus.run_quantum_circuit(
        polypus.Circuit(1).h(0).measure_all(),
        shots=16,
        infrastructure="local",
        walltime="garbage",
    )
    assert sum(result.merged_counts.values()) == 16
