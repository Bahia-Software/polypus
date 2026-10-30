"""Configuration errors are raised before any backend exists (issue #418).

With ``infrastructure="cunqa"``, creating the backend runs ``qraise``, which
requests a SLURM allocation. ``train`` and ``qml.train`` therefore check every
argument, the optimizer and the observable included, before they create it:
against ``fake_cunqa.FakeCunqa`` an invalid configuration fails with no CUNQA
call at all.

Every such check raises ``polypus.ConfigurationError``, a ``PolypusError`` that
is also a ``ValueError``.
"""

import pytest

pytest.importorskip("qiskit_aer")

import polypus  # noqa: E402
from fake_cunqa import FakeCunqa  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402


def _train(**kwargs):
    kwargs.setdefault("method", polypus.DE(generations=1, population_size=4))
    kwargs.setdefault("expectation_function", lambda bits: float(bits == "1"))
    kwargs.setdefault("infrastructure", "cunqa")
    return polypus.train(
        polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all(),
        shots=16,
        n_qpus=1,
        dimensions=1,
        nodes=1,
        cores_per_qpu=1,
        id="config_check",
        seed=3,
        **kwargs,
    )


def _qml_train(**kwargs):
    feature_map = QuantumCircuit(1)
    feature_map.ry(Parameter("x"), 0)
    ansatz = QuantumCircuit(1)
    ansatz.ry(Parameter("w"), 0)
    kwargs.setdefault("method", polypus.DE(generations=1, population_size=4))
    kwargs.setdefault("expectation_function", lambda bits: float(bits == "1"))
    kwargs.setdefault("infrastructure", "cunqa")
    return polypus.qml.train(
        feature_map,
        ansatz,
        [[0.1], [0.2]],
        shots=16,
        n_qpus=1,
        dimensions=1,
        nodes=1,
        cores_per_qpu=1,
        id="config_check",
        seed=3,
        **kwargs,
    )


ENTRY_POINTS = {"train": _train, "qml.train": _qml_train}


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
class TestNoAllocationForAnInvalidConfiguration:
    def test_a_population_too_small_requests_nothing(self, monkeypatch, entry_point):
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(polypus.ConfigurationError, match="population_size"):
            ENTRY_POINTS[entry_point](method=polypus.DE(population_size=3))
        assert fake.calls == []

    def test_an_invalid_observable_requests_nothing(self, monkeypatch, entry_point):
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(TypeError, match="expectation_function"):
            ENTRY_POINTS[entry_point](expectation_function=42)
        assert fake.calls == []

    def test_a_valid_configuration_reaches_cunqa(self, monkeypatch, entry_point):
        # The two tests above would pass if CUNQA were never reached at all.
        fake = FakeCunqa().install(monkeypatch)
        ENTRY_POINTS[entry_point]()
        assert fake.names()[0] == "qraise"


class TestConfigurationErrorClass:
    def test_is_a_polypus_error_and_a_value_error(self):
        assert issubclass(polypus.ConfigurationError, polypus.PolypusError)
        assert issubclass(polypus.ConfigurationError, ValueError)
        assert polypus.ConfigurationError.__module__ == "polypus"

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(
                lambda: _train(method=polypus.DE(population_size=3)), id="optimizer"
            ),
            pytest.param(lambda: _train(infrastructure="nowhere"), id="infrastructure"),
            pytest.param(lambda: _train(walltime="1-00:00:00"), id="walltime"),
            pytest.param(lambda: polypus.DE(strategy="best1bin"), id="de-strategy"),
            pytest.param(lambda: polypus.Adam(rule="sgd"), id="adam-rule"),
            pytest.param(
                lambda: polypus.run_quantum_circuit(
                    polypus.Circuit(1).h(0).measure_all(),
                    shots=0,
                    infrastructure="local",
                ),
                id="shots",
            ),
            pytest.param(
                lambda: _qml_train(infrastructure="local", backend="polypus"),
                id="qml-native-backend",
            ),
        ],
    )
    def test_configuration_checks_raise_it(self, monkeypatch, call):
        # `_train` defaults to CUNQA: a check that regressed past the backend
        # must reach the fake, never a real qraise.
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(polypus.ConfigurationError):
            call()
        assert fake.calls == []


class TestQmlTrainRejectsQmio:
    def test_like_qml_predict(self):
        # The QML model is a Qiskit circuit, which QMIO cannot serialize.
        with pytest.raises(
            polypus.ConfigurationError, match="cannot serialize the Qiskit circuits"
        ):
            _qml_train(infrastructure="qmio")


# ── The native QML API (`Model.fit`, `TrainedModel.predict`/`run`) ──────────

_UNSET = object()


def _fit(x=_UNSET, y=_UNSET, **kwargs):
    from qml_native_helpers import X, Y, make_model

    kwargs.setdefault("method", polypus.DE(generations=1, population_size=4))
    kwargs.setdefault("loss", "hinge")
    kwargs.setdefault("infrastructure", "cunqa")
    return make_model().fit(
        X if x is _UNSET else x,
        Y if y is _UNSET else y,
        shots=16,
        nodes=1,
        cores_per_qpu=1,
        id="config_check",
        seed=3,
        **kwargs,
    )


def _trained():
    from qml_native_helpers import make_model

    model = make_model()
    return model.bind([0.1] * model.num_params)


def _infer(entry_point, x=_UNSET, **kwargs):
    from qml_native_helpers import X

    kwargs.setdefault("infrastructure", "cunqa")
    return getattr(_trained(), entry_point)(
        X if x is _UNSET else x, shots=16, nodes=1, cores_per_qpu=1, **kwargs
    )


class TestNativeQmlChecksRaiseIt:
    """The native API's own checks raise `ConfigurationError` too, before any
    backend exists. The builders (`Encoder`, `Ansatz`, `Readout`, `Dataset`,
    `qml.load`) keep a plain `ValueError`, as `polypus.Circuit`'s do."""

    @pytest.mark.parametrize(
        ("call", "match"),
        [
            pytest.param(lambda: _fit(loss="l2"), "unknown loss", id="unknown-loss"),
            pytest.param(
                lambda: _fit(y=None), "needs labels", id="loss-without-labels"
            ),
            pytest.param(
                lambda: _fit(loss=None), "needs a loss", id="labels-without-loss"
            ),
            pytest.param(
                lambda: _fit(loss="expectation"),
                "does not use labels",
                id="expectation-with-labels",
            ),
            pytest.param(
                lambda: _fit(x=polypus.qml.Dataset([[0.1], [2.9]], [1, -1]), y=[1, -1]),
                "carries its own labels",
                id="dataset-and-y",
            ),
            pytest.param(lambda: _fit(batch_size=6), "batch_size", id="batch-size"),
            pytest.param(lambda: _fit(exact=True), "exact=True", id="fit-exact"),
            pytest.param(
                lambda: _fit(x=[[0.1, 0.2], [2.9, 2.8]], y=[1, -1]),
                None,
                id="fit-too-many-features",
            ),
            pytest.param(
                lambda: _fit(x=[[0.1], [2.9, 2.8]], y=[1, -1]),
                None,
                id="fit-ragged-rows",
            ),
            pytest.param(
                lambda: _fit(method=polypus.Rotosolve()),
                "polypus.Rotosolve",
                id="rotosolve",
            ),
            pytest.param(
                lambda: _fit(initial_point=[0.1]),
                "invalid initial_point",
                id="fit-initial-point",
            ),
            pytest.param(
                lambda: _fit(final_shots=0), "final_shots", id="fit-final-shots"
            ),
            pytest.param(
                lambda: _fit(infrastructure="local", exact=True, final_shots=4096),
                "final_shots does not apply with exact=True",
                id="fit-final-shots-exact",
            ),
            pytest.param(
                lambda: _infer("predict", exact=True), "exact=True", id="predict-exact"
            ),
            pytest.param(
                lambda: _infer("predict", x=[[0.1, 0.2]]),
                None,
                id="predict-too-many-features",
            ),
            pytest.param(
                lambda: _infer("run", infrastructure="qmio", seed=1),
                "seed is not supported",
                id="run-seed-on-qmio",
            ),
        ],
    )
    def test_before_any_cunqa_call(self, monkeypatch, call, match):
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(polypus.ConfigurationError, match=match):
            call()
        assert fake.calls == []

    def test_a_valid_native_configuration_reaches_cunqa(self, monkeypatch):
        # The checks above would pass if CUNQA were never reached at all.
        fake = FakeCunqa().install(monkeypatch)
        _fit()
        assert fake.names()[0] == "qraise"
        fake = FakeCunqa().install(monkeypatch)
        _infer("predict")
        assert fake.names()[0] == "qraise"
