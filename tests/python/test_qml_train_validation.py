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
"""

import pytest


def _feature_map_ansatz():
    from qiskit.circuit.library import real_amplitudes, zz_feature_map

    # feature_dimension=2 ⇒ len(feature_map.parameters) == 2.
    feature_map = zz_feature_map(feature_dimension=2, reps=1)
    ansatz = real_amplitudes(num_qubits=2, reps=1)
    return feature_map, ansatz


def _qml_train(feature_map, ansatz, x_train, dimensions):
    import polypus

    return polypus.qml.train(
        feature_map,
        ansatz,
        x_train,
        polypus.DE(generations=3, population_size=6, tolerance=1e-12),
        shots=64,
        n_qpus=1,
        dimensions=dimensions,
        expectation_function=lambda b: sum(int(c) for c in b) / len(b),
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="qml_validation",
        seed=7,
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
