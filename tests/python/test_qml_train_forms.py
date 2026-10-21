"""
What the native form brought to the Qiskit form of ``qml.train``: execution
defaults, an inferred ``dimensions``, a ``polypus.qml.Dataset`` in ``x_train``
(plan G1), ``batch_size``, and the rejection of native pieces mixed in (plan
G2). The Qiskit form's own behaviour is pinned by the existing QML tests, which
this change leaves untouched.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]

_X = [[0.1], [3.0], [0.2], [2.9]]
_Y = [0, 1, 0, 1]


def _circuits():
    from qiskit.circuit import ParameterVector, QuantumCircuit

    x = ParameterVector("x", 1)
    theta = ParameterVector("theta", 1)
    feature_map = QuantumCircuit(1)
    feature_map.ry(x[0], 0)
    ansatz = QuantumCircuit(1)
    ansatz.ry(theta[0], 0)
    return feature_map, ansatz


def _train(x_train, expectation_function, **kwargs):
    import polypus

    feature_map, ansatz = _circuits()
    return polypus.qml.train(
        feature_map,
        ansatz,
        x_train,
        kwargs.pop("method", None) or polypus.DE(generations=1, population_size=4),
        shots=kwargs.pop("shots", 32),
        expectation_function=expectation_function,
        seed=5,
        **kwargs,
    )


def _ones(bitstring):
    return float(bitstring == "1")


class TestDefaults:
    def test_only_the_data_and_the_method_are_needed(self):
        result = _train(_X, _ones)
        assert len(result.best_params) == 1

    def test_dimensions_is_inferred_and_checked(self):
        assert len(_train(_X, _ones, dimensions=1).best_params) == 1
        with pytest.raises(ValueError, match="does not match"):
            _train(_X, _ones, dimensions=2)

    def test_missing_expectation_function_is_a_type_error(self):
        import polypus

        feature_map, ansatz = _circuits()
        with pytest.raises(TypeError, match="expectation_function"):
            polypus.qml.train(feature_map, ansatz, _X, polypus.DE())

    def test_the_native_backend_still_needs_the_native_form(self):
        with pytest.raises(ValueError, match="native form"):
            _train(_X, _ones, backend="polypus")


class TestDatasetInTheQiskitForm:
    def test_labels_reach_the_objective_as_class_integers(self):
        from polypus.qml import Dataset

        seen = []

        def correct(bitstring, label):
            seen.append(label)
            return float(int(bitstring) == label)

        _train(Dataset(_X, _Y), correct)
        assert seen and all(type(label) is int for label in seen)
        assert set(seen) == {0, 1}

    def test_real_labels_reach_it_as_floats(self):
        from polypus.qml import Dataset

        seen = []

        def score(bitstring, label):
            seen.append(label)
            return -abs(int(bitstring) - label)

        _train(Dataset(_X, [0.0, 0.9, 0.1, 1.0]), score)
        assert seen and all(type(label) is float for label in seen)

    def test_an_unlabeled_dataset_is_unsupervised(self):
        from polypus.qml import Dataset

        assert len(_train(Dataset(_X), _ones).best_params) == 1

    def test_a_dataset_and_y_train_together(self):
        from polypus.qml import Dataset

        with pytest.raises(ValueError, match="carries its own labels"):
            _train(Dataset(_X, _Y), lambda b, y: 0.0, y_train=_Y)


class TestBatchSize:
    def test_minibatches_train(self):
        result = _train(_X, lambda b, y: float(int(b) == y), y_train=_Y, batch_size=2)
        assert len(result.best_params) == 1

    @pytest.mark.parametrize("size", [0, 4, 5])
    def test_batch_size_must_be_smaller_than_the_dataset(self, size):
        with pytest.raises(ValueError, match="batch_size"):
            _train(_X, _ones, batch_size=size)


class TestMixedForms:
    def test_an_objective_needs_the_native_form(self):
        from polypus.qml import Objective, Z

        with pytest.raises(TypeError, match="needs a native Encoder and Ansatz"):
            _train(_X, Objective([Z(0)], decision="sign", loss="hinge"), y_train=_Y)

    def test_exact_needs_the_native_form(self):
        with pytest.raises(ValueError, match="exact=True needs the native form"):
            _train(_X, _ones, exact=True)
