"""
Supervised ``qml.train``: with ``y_train``, each sample is scored against its own
label, through ``f(bitstring, label)``, ``CachedCost(f)`` or ``SampleCost(g)``.

The mocked-seam tests answer each circuit from its own bound feature value, so
they can check which label each sample met and how the objective was called. The
Aer tests train a small separable dataset. Validation of ``y_train`` (contract
C-8) is in ``test_qml_train_validation.py``.
"""

import math
import threading

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]

# Interleaved classes, so a label shifted by one position meets the wrong class.
_X = [0.1, 3.0, 0.2, 2.9, 0.3, 2.8]
_Y = [0, 1, 0, 1, 0, 1]


def _one_qubit_model():
    """``ry(x)`` feature map + ``ry(theta)`` ansatz; in a bound circuit, the first
    gate's angle is the sample's feature."""
    from qiskit.circuit import ParameterVector, QuantumCircuit

    x = ParameterVector("x", 1)
    theta = ParameterVector("theta", 1)
    feature_map = QuantumCircuit(1)
    feature_map.ry(x[0], 0)
    ansatz = QuantumCircuit(1)
    ansatz.ry(theta[0], 0)
    return feature_map, ansatz


class _Seam:
    """``run_qcs`` stand-in answering ``counts_for(x, shots)`` per circuit. The
    default is a perfect classifier: all shots on ``"1"`` iff ``x > pi/2``."""

    def __init__(self, counts_for=None):
        self._counts_for = counts_for or (
            lambda x, shots: {"1" if x > math.pi / 2 else "0": shots}
        )
        self._lock = threading.Lock()
        self.calls = 0

    def __call__(self, _infrastructure, **kwargs):
        with self._lock:
            self.calls += 1
        shots = kwargs["shots"]
        return [
            self._counts_for(float(qc.data[0].operation.params[0]), shots)
            for qc in kwargs["qcs"]
        ]


def _train(expectation_function, y_train, seam=None, monkeypatch=None, **kwargs):
    import polypus
    import polypus_python

    if seam is not None:
        monkeypatch.setattr(polypus_python, "run_qcs", seam)
    feature_map, ansatz = _one_qubit_model()
    kwargs.setdefault(
        "method", polypus.DE(generations=2, population_size=4, tolerance=1e-12)
    )
    return polypus.qml.train(
        feature_map,
        ansatz,
        [[x] for x in kwargs.pop("x_train", _X)],
        kwargs.pop("method"),
        shots=kwargs.pop("shots", 64),
        n_qpus=1,
        dimensions=1,
        expectation_function=expectation_function,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="qml_supervised",
        seed=kwargs.pop("seed", 3),
        y_train=y_train,
        **kwargs,
    )


def _correct(bitstring, label):
    return float(int(bitstring) == label)


# ── Mocked seam: pairing, calling convention, dedup ─────────────────────────


def test_each_sample_is_scored_against_its_own_label(monkeypatch):
    result = _train(_correct, _Y, _Seam(), monkeypatch)
    assert result.best_fitness == 1.0
    assert result.fitness_history == [1.0] * result.iterations_run


def test_the_labels_really_drive_the_fitness(monkeypatch):
    result = _train(_correct, [1 - y for y in _Y], _Seam(), monkeypatch)
    assert result.best_fitness == 0.0


@pytest.mark.parametrize("dtype", ["int64", "uint8", "bool"])
def test_integer_labels_reach_the_cost_as_int_even_from_numpy(monkeypatch, dtype):
    import numpy as np

    seen = []

    def cost(bitstring, label):
        seen.append(type(label))
        return _correct(bitstring, label)

    # NumPy bools have no __index__ but must still arrive as the classes 0/1.
    result = _train(cost, np.array(_Y, dtype=dtype), _Seam(), monkeypatch)
    assert result.best_fitness == 1.0
    assert seen and set(seen) == {int}, f"{dtype} labels must arrive as Python ints"


def test_any_non_integer_label_makes_every_label_a_float(monkeypatch):
    seen = []

    def cost(bitstring, label):
        seen.append((type(label), label))
        return 0.0

    _train(cost, [0, 1.5, 0, 1, 0, 1], _Seam(), monkeypatch)
    assert {t for t, _ in seen} == {float}
    assert {v for _, v in seen} <= {0.0, 1.0, 1.5}


def test_a_labelled_cost_runs_once_per_label_and_bitstring_per_batch(monkeypatch):
    # Two labels x two outcomes: four distinct pairs per batch.
    seam = _Seam(lambda _x, shots: {"0": shots // 4, "1": shots - shots // 4})
    calls = []

    def cost(bitstring, label):
        calls.append((bitstring, label))
        return _correct(bitstring, label)

    result = _train(cost, _Y, seam, monkeypatch)
    # The local backend runs each candidate window in one call.
    assert len(calls) == 4 * seam.calls
    assert result.best_fitness == pytest.approx(0.5)


def test_cached_cost_evaluates_each_label_and_bitstring_once_per_run(monkeypatch):
    import polypus

    seam = _Seam(lambda _x, shots: {"0": shots // 4, "1": shots - shots // 4})
    calls = []

    def cost(bitstring, label):
        calls.append((bitstring, label))
        return _correct(bitstring, label)

    _train(polypus.CachedCost(cost), _Y, seam, monkeypatch)
    assert seam.calls > 1, "the run must span several batches"
    assert sorted(calls) == [("0", 0), ("0", 1), ("1", 0), ("1", 1)]


def test_sample_cost_sees_each_samples_distribution_and_label(monkeypatch):
    import polypus

    def counts_for(x, shots):
        ones = round(shots * math.sin(x / 2) ** 2)
        return {"1": ones, "0": shots - ones}

    seen = []

    def likelihood(counts, label):
        seen.append((dict(counts), label))
        return counts.get(str(label), 0) / sum(counts.values())

    result = _train(
        polypus.SampleCost(likelihood), _Y, _Seam(counts_for), monkeypatch, shots=64
    )
    expected = {(tuple(sorted(counts_for(x, 64).items())), y) for x, y in zip(_X, _Y)}
    assert {(tuple(c.items()), y) for c, y in seen} == expected
    assert all(list(c) == sorted(c) for c, _ in seen), "counts keys arrive sorted"
    assert all(type(y) is int for _, y in seen)
    # The mock ignores theta, so every candidate has the same fitness.
    assert result.best_fitness == pytest.approx(
        sum(counts_for(x, 64)[str(y)] / 64 for x, y in zip(_X, _Y)) / len(_X)
    )


def test_a_non_finite_sample_score_names_the_x_train_row(monkeypatch):
    import polypus

    labels = list(_Y)
    labels[2] = 1  # row 2's counts never show this label: probability 0

    def log_likelihood(counts, label):
        p = counts.get(str(label), 0) / sum(counts.values())
        return math.log(p) if p > 0 else -math.inf

    with pytest.raises(polypus.EvaluationError, match=r"x_train row 2\b"):
        _train(polypus.SampleCost(log_likelihood), labels, _Seam(), monkeypatch)


@pytest.mark.parametrize("wrap", ["callable", "sample_cost"])
def test_an_objective_exception_propagates_verbatim(monkeypatch, wrap):
    import polypus

    def explode(*_args):
        raise ValueError("user objective blew up")

    objective = explode if wrap == "callable" else polypus.SampleCost(explode)
    with pytest.raises(ValueError, match="user objective blew up") as excinfo:
        _train(objective, _Y, _Seam(), monkeypatch)
    assert not isinstance(excinfo.value, polypus.EvaluationError)


def test_an_unlabelled_cost_given_labels_fails_with_pythons_own_type_error(
    monkeypatch,
):
    with pytest.raises(TypeError, match="positional argument"):
        _train(lambda bitstring: 0.0, _Y, _Seam(), monkeypatch)


def test_y_train_none_is_the_unsupervised_path(monkeypatch):
    result = _train(
        lambda bitstring: float(bitstring == "1"), None, _Seam(), monkeypatch
    )
    assert result.best_fitness == pytest.approx(0.5)


# ── Real Aer: a separable dataset learns with labels ────────────────────────

# Class 1 iff x > pi/2, with a margin on both sides.
_SEPARABLE_X = [0.2, 2.6, 0.6, 3.0, 1.0, 2.2, 0.4, 2.8, 0.8, 2.4]
_SEPARABLE_Y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


def _predict(theta, xs, seed):
    """Majority read-out of the trained classifier, one run per sample."""
    import polypus

    feature_map, ansatz = _one_qubit_model()
    template = feature_map.compose(ansatz)
    template.measure_all()
    # By name: Qiskit orders parameters alphabetically (theta before x).
    (x_param,), (theta_param,) = feature_map.parameters, ansatz.parameters
    predictions = []
    for i, x in enumerate(xs):
        circuit = template.assign_parameters(
            {x_param: x, theta_param: theta[0]}, inplace=False
        )
        counts = polypus.run_quantum_circuit(
            circuit, shots=256, infrastructure="local", seed=seed + i
        ).counts[0]
        predictions.append(int(counts.get("1", 0) > counts.get("0", 0)))
    return predictions


def _log_likelihood(counts, label):
    p = counts.get(str(label), 0) / sum(counts.values())
    return math.log(max(p, 1e-3))


@pytest.mark.parametrize("objective", ["expected_accuracy", "log_likelihood"])
def test_labels_train_a_separable_dataset_to_high_accuracy(objective):
    import polypus

    expectation_function = (
        _correct
        if objective == "expected_accuracy"
        else polypus.SampleCost(_log_likelihood)
    )
    result = _train(
        expectation_function,
        _SEPARABLE_Y,
        x_train=_SEPARABLE_X,
        method=polypus.DE(generations=15, population_size=10, tolerance=1e-9),
        shots=256,
        seed=2024,
    )
    predictions = _predict(result.best_params, _SEPARABLE_X, seed=7)
    accuracy = sum(p == y for p, y in zip(predictions, _SEPARABLE_Y)) / len(
        _SEPARABLE_Y
    )
    assert accuracy >= 0.9, (
        f"{objective}: trained to accuracy {accuracy} "
        f"(best_fitness={result.best_fitness}, theta={result.best_params})"
    )
    if objective == "expected_accuracy":
        assert result.best_fitness >= 0.85


def test_a_seeded_supervised_run_is_reproducible():
    import polypus

    def run():
        return _train(
            polypus.SampleCost(_log_likelihood),
            _SEPARABLE_Y,
            x_train=_SEPARABLE_X,
            method=polypus.DE(generations=3, population_size=6, tolerance=1e-12),
            shots=128,
            seed=11,
        )

    first, second = run(), run()
    assert first.best_params == second.best_params
    assert first.best_fitness == second.best_fitness
    assert first.fitness_history == second.fitness_history
