"""
qml.train row/dimension symmetry — upfront-validation tests (contract C-8).

These cover issue #79: ``qml.train`` used to zip each ``x_train`` row against
the feature-map parameters, so a row with too many features silently dropped
data and a row with too few left feature-map parameters unbound (surfacing
later as a cryptic Qiskit error inside the oracle). It also never validated
``dimensions`` against ``len(ansatz.parameters)``, unlike ``train``.

Both agreements are now enforced **upfront** with a clear ``ValueError`` before
any circuit is executed, so these tests never reach a real backend — the
failures are raised during argument validation, so no optimizer/backend mocking
is needed.

The label agreement (``y_train``: one finite number per row, and an objective
that can read labels) is checked upfront too. Those tests replace the execution
seam with one that fails if it is reached.
"""

import pytest


def _feature_map_ansatz():
    from qiskit.circuit.library import real_amplitudes, zz_feature_map

    # feature_dimension=2 ⇒ len(feature_map.parameters) == 2.
    feature_map = zz_feature_map(feature_dimension=2, reps=1)
    ansatz = real_amplitudes(num_qubits=2, reps=1)
    return feature_map, ansatz


def _qml_train(feature_map, ansatz, x_train, dimensions, **kwargs):
    import polypus

    kwargs.setdefault("expectation_function", lambda b: sum(int(c) for c in b) / len(b))
    return polypus.qml.train(
        feature_map,
        ansatz,
        x_train,
        polypus.DE(generations=3, population_size=6, tolerance=1e-12),
        shots=64,
        n_qpus=1,
        dimensions=dimensions,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="qml_validation",
        seed=7,
        **kwargs,
    )


@pytest.mark.integration
@pytest.mark.vqc
class TestQmlTrainRowDimensionSymmetry:
    def test_row_longer_than_feature_map_raises(self):
        feature_map, ansatz = _feature_map_ansatz()
        # feature_map expects 2 features; this row supplies 3.
        x_train = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        with pytest.raises(ValueError, match=r"row 0 has 3 features.*expects 2"):
            _qml_train(feature_map, ansatz, x_train, len(ansatz.parameters))

    def test_row_shorter_than_feature_map_raises(self):
        feature_map, ansatz = _feature_map_ansatz()
        # feature_map expects 2 features; this row supplies 1.
        x_train = [[0.1], [0.2]]
        with pytest.raises(ValueError, match=r"row 0 has 1 features.*expects 2"):
            _qml_train(feature_map, ansatz, x_train, len(ansatz.parameters))

    def test_dimensions_mismatch_raises(self):
        feature_map, ansatz = _feature_map_ansatz()
        # An otherwise-valid x_train (2 features per row) isolates the failure to
        # the dimensions/ansatz mismatch rather than the row width.
        x_train = [[0.1, 0.2], [0.3, 0.4]]
        wrong_dimensions = len(ansatz.parameters) + 1
        with pytest.raises(ValueError, match="does not match"):
            _qml_train(feature_map, ansatz, x_train, wrong_dimensions)


def _forbid_execution(monkeypatch):
    """Fail the test if anything reaches the backend."""
    import polypus_python

    def unreachable(*_args, **_kwargs):
        pytest.fail("a rejected y_train must not reach the backend")

    monkeypatch.setattr(polypus_python, "run_qcs", unreachable)


def _labelled_correct(bitstring, label):
    return float(bitstring.count("1") % 2 == label)


@pytest.mark.integration
@pytest.mark.vqc
class TestQmlTrainLabels:
    """y_train agreements (contract C-8), all rejected before anything runs."""

    _X = [[0.1, 0.2], [0.3, 0.4]]

    def _train(self, monkeypatch, y_train, expectation_function=_labelled_correct):
        _forbid_execution(monkeypatch)
        feature_map, ansatz = _feature_map_ansatz()
        return _qml_train(
            feature_map,
            ansatz,
            self._X,
            len(ansatz.parameters),
            expectation_function=expectation_function,
            y_train=y_train,
        )

    def test_fewer_labels_than_rows_raises(self, monkeypatch):
        with pytest.raises(
            ValueError, match=r"y_train has 1 labels, but x_train has 2 rows"
        ):
            self._train(monkeypatch, [0])

    def test_more_labels_than_rows_raises(self, monkeypatch):
        with pytest.raises(
            ValueError, match=r"y_train has 3 labels, but x_train has 2 rows"
        ):
            self._train(monkeypatch, [0, 1, 0])

    def test_a_string_label_raises_naming_its_index(self, monkeypatch):
        with pytest.raises(TypeError, match=r"y_train\[1\] is a str"):
            self._train(monkeypatch, [0, "right"])

    def test_one_hot_rows_raise(self, monkeypatch):
        with pytest.raises(TypeError, match=r"y_train\[0\] is a sequence"):
            self._train(monkeypatch, [[1, 0], [0, 1]])

    def test_a_column_vector_raises(self, monkeypatch):
        # NumPy would turn each length-1 row into a float without complaint.
        import numpy as np

        with pytest.raises(TypeError, match=r"ndarray of length 1.*ravel"):
            self._train(monkeypatch, np.array([[0], [1]]))

    def test_a_non_finite_label_raises_naming_its_index(self, monkeypatch):
        with pytest.raises(ValueError, match=r"y_train\[1\] is NaN"):
            self._train(monkeypatch, [0.5, float("nan")])

    def test_a_declarative_observable_cannot_read_labels(self, monkeypatch):
        import polypus

        qubo = polypus.Qubo(2, linear=[(0, 1.0)])
        with pytest.raises(TypeError, match="cannot read labels"):
            self._train(monkeypatch, [0, 1], expectation_function=qubo)

    def test_sample_cost_without_labels_raises(self, monkeypatch):
        import polypus

        _forbid_execution(monkeypatch)
        feature_map, ansatz = _feature_map_ansatz()
        with pytest.raises(TypeError, match=r"SampleCost.*y_train"):
            _qml_train(
                feature_map,
                ansatz,
                self._X,
                len(ansatz.parameters),
                expectation_function=polypus.SampleCost(lambda counts, label: 0.0),
            )

    def test_sample_cost_requires_a_callable(self):
        import polypus

        with pytest.raises(TypeError, match="SampleCost expects a callable"):
            polypus.SampleCost(42)

    def test_y_train_is_keyword_only(self, monkeypatch):
        import polypus

        _forbid_execution(monkeypatch)
        feature_map, ansatz = _feature_map_ansatz()
        # The 16 existing parameters positionally, plus y_train as a 17th.
        with pytest.raises(TypeError):
            polypus.qml.train(
                feature_map,
                ansatz,
                self._X,
                polypus.DE(generations=3, population_size=6),
                64,
                1,
                len(ansatz.parameters),
                _labelled_correct,
                "local",
                1,
                1,
                "qml_validation",
                "automatic",
                None,
                "aer",
                7,
                [0, 1],
            )
