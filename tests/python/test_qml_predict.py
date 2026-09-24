"""
``polypus.qml.predict``: batched inference for a ``qml.train`` model, one counts
dict per row from a single run.

The mocked-seam tests answer each circuit from its bound angles to check the row
order, the bound weights and the single backend call. The validation cases
(contract C-8) run with the seam forbidden. The Aer tests cover the manifest and
seed replay (contract C-7), shot conservation (contract C-3) and a trained model's
held-out accuracy.
"""

import math
import threading

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]


def _one_qubit_model():
    """``ry(x)`` feature map + ``ry(theta)`` ansatz; in a bound circuit the first
    angle is the feature and the second the weight."""
    from qiskit.circuit import ParameterVector, QuantumCircuit

    x = ParameterVector("x", 1)
    theta = ParameterVector("theta", 1)
    feature_map = QuantumCircuit(1)
    feature_map.ry(x[0], 0)
    ansatz = QuantumCircuit(1)
    ansatz.ry(theta[0], 0)
    return feature_map, ansatz


class _Seam:
    """``run_qcs`` stand-in recording batch sizes and answering
    ``counts_for(x, theta, shots)`` per circuit."""

    def __init__(self, counts_for):
        self._counts_for = counts_for
        self._lock = threading.Lock()
        self.batches = []

    def __call__(self, _infrastructure, **kwargs):
        qcs = kwargs["qcs"]
        with self._lock:
            self.batches.append(len(qcs))
        return [
            self._counts_for(
                float(qc.data[0].operation.params[0]),
                float(qc.data[1].operation.params[0]),
                kwargs["shots"],
            )
            for qc in qcs
        ]


def _predict(x, params, **kwargs):
    import polypus

    feature_map, ansatz = _one_qubit_model()
    kwargs.setdefault("shots", 64)
    kwargs.setdefault("infrastructure", "local")
    return polypus.qml.predict(feature_map, ansatz, x, params, **kwargs)


# ── Mocked seam ──────────────────────────────────────────────────────────────


def test_predict_returns_one_counts_dict_per_row_in_row_order(monkeypatch):
    import polypus_python

    seam = _Seam(lambda x, _theta, shots: {"1" if x > math.pi / 2 else "0": shots})
    monkeypatch.setattr(polypus_python, "run_qcs", seam)

    run = _predict([[0.1], [3.0], [0.2], [2.9], [2.8]], [0.0])
    assert run.counts == [{"0": 64}, {"1": 64}, {"0": 64}, {"1": 64}, {"1": 64}]
    assert seam.batches == [5], "the whole batch is one backend call"


def test_predict_binds_the_trained_weights_to_the_ansatz(monkeypatch):
    import polypus_python

    seam = _Seam(lambda _x, theta, shots: {"1" if theta > 1.0 else "0": shots})
    monkeypatch.setattr(polypus_python, "run_qcs", seam)

    assert _predict([[0.1], [3.0]], [2.0]).counts == [{"1": 64}, {"1": 64}]
    assert _predict([[0.1], [3.0]], [0.5]).counts == [{"0": 64}, {"0": 64}]


def test_predict_accepts_numpy_rows_and_weights(monkeypatch):
    import numpy as np
    import polypus_python

    seam = _Seam(
        lambda x, theta, shots: {"1" if x + theta > math.pi / 2 else "0": shots}
    )
    monkeypatch.setattr(polypus_python, "run_qcs", seam)

    run = _predict(np.array([[0.1], [3.0]]), np.array([0.2]))
    assert run.counts == [{"0": 64}, {"1": 64}]


# ── Validation (contract C-8): rejected with nothing executed ────────────────


@pytest.fixture
def forbid_execution(monkeypatch):
    import polypus_python

    def unreachable(*_args, **_kwargs):
        pytest.fail("a rejected qml.predict must not reach the backend")

    monkeypatch.setattr(polypus_python, "run_qcs", unreachable)


@pytest.mark.usefixtures("forbid_execution")
class TestQmlPredictValidation:
    def test_a_row_of_the_wrong_width_raises_naming_it(self):
        with pytest.raises(ValueError, match=r"x row 1 has 2 features.*expects 1"):
            _predict([[0.1], [0.2, 0.3]], [0.0])

    def test_too_many_weights_raise(self):
        with pytest.raises(ValueError, match=r"params has 2 values.*1 free parameters"):
            _predict([[0.1]], [0.0, 1.0])

    def test_too_few_weights_raise(self):
        with pytest.raises(ValueError, match=r"params has 0 values"):
            _predict([[0.1]], [])

    def test_a_non_finite_weight_raises(self):
        with pytest.raises(ValueError, match=r"params\[0\] is NaN"):
            _predict([[0.1]], [float("nan")])

    def test_an_empty_batch_raises(self):
        with pytest.raises(ValueError, match="at least one sample"):
            _predict([], [0.0])

    def test_the_native_backend_is_rejected(self):
        with pytest.raises(ValueError, match="native 'polypus' backend"):
            _predict([[0.1]], [0.0], backend="polypus")

    def test_qmio_is_rejected(self):
        with pytest.raises(ValueError, match="qmio"):
            _predict([[0.1]], [0.0], infrastructure="qmio")

    def test_zero_shots_are_rejected(self):
        with pytest.raises(ValueError):
            _predict([[0.1]], [0.0], shots=0)


# ── Real Aer ─────────────────────────────────────────────────────────────────


def test_predict_reports_a_replayable_manifest_and_conserves_shots():
    rows = [[0.3], [1.6], [2.9]]
    first = _predict(rows, [0.1], shots=200, seed=7)
    assert first.seed == 7
    assert first.id.startswith("predict_1_local_")
    assert (first.backend, first.infrastructure) == ("aer", "local")
    assert [sum(c.values()) for c in first.counts] == [200, 200, 200]
    assert _predict(rows, [0.1], shots=200, seed=first.seed).counts == first.counts
    assert isinstance(_predict(rows, [0.1], shots=200).seed, int)


def test_a_supervised_model_predicts_held_out_samples():
    import polypus

    def correct(bitstring, label):
        return float(int(bitstring) == label)

    feature_map, ansatz = _one_qubit_model()
    # Class 1 iff x > pi/2, with a margin.
    x_train = [[0.2], [2.6], [0.6], [3.0], [1.0], [2.2], [0.4], [2.8]]
    y_train = [0, 1, 0, 1, 0, 1, 0, 1]
    x_test = [[0.3], [2.7], [0.9], [2.3], [0.5], [2.9]]
    y_test = [0, 1, 0, 1, 0, 1]

    result = polypus.qml.train(
        feature_map,
        ansatz,
        x_train,
        polypus.DE(generations=15, population_size=10, tolerance=1e-9),
        shots=256,
        n_qpus=1,
        dimensions=len(ansatz.parameters),
        expectation_function=correct,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="qml_predict",
        seed=2024,
        y_train=y_train,
    )
    run = polypus.qml.predict(
        feature_map,
        ansatz,
        x_test,
        result.best_params,
        shots=256,
        infrastructure="local",
        seed=11,
    )
    predictions = [int(c.get("1", 0) > c.get("0", 0)) for c in run.counts]
    accuracy = sum(p == y for p, y in zip(predictions, y_test)) / len(y_test)
    assert accuracy >= 0.9, (
        f"held-out accuracy {accuracy} (best_params={result.best_params})"
    )
