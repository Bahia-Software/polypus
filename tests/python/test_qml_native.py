"""
The native pieces of ``polypus.qml``: ``Encoder``, ``Ansatz``, ``Objective``,
``Dataset`` and the Pauli readout helpers (``Z``/``X``/``Y``, ``PauliTerm``,
``Observable``).

These tests need no backend: they check construction, the closed option sets
(plan F6), the checks each builder call runs without the data (plan F1), the
loss and decision pairing (plan F2) and the unlabeled ``Dataset`` (plan F4).
"""

import math

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Pauli terms and observables
# ─────────────────────────────────────────────────────────────────────────────


class TestPauliTerms:
    def test_products_read_as_written(self):
        from polypus.qml import X, Y, Z

        assert repr(Z(0)) == "Z(0)"
        assert repr(Z(0) @ X(1) @ Y(3)) == "Z(0) @ X(1) @ Y(3)"

    @pytest.mark.parametrize("other", ["Z", "X"])
    def test_a_repeated_position_is_rejected_at_the_product(self, other):
        import polypus.qml as qml

        with pytest.raises(ValueError):
            qml.Z(0) @ getattr(qml, other)(0)

    def test_weighted_observable(self):
        from polypus.qml import Observable, Z

        observable = Observable([(0.5, Z(0)), (-1.5, Z(0) @ Z(1)), (2.0, [("x", 2)])])
        assert observable.num_terms == 3

    @pytest.mark.parametrize("coefficient", [math.nan, math.inf])
    def test_a_non_finite_coefficient_is_rejected(self, coefficient):
        from polypus.qml import Observable, Z

        with pytest.raises(ValueError):
            Observable([(coefficient, Z(0))])

    def test_an_unknown_pauli_is_rejected(self):
        from polypus.qml import Observable

        with pytest.raises(ValueError, match='"x", "y" or "z"'):
            Observable([(1.0, [("w", 0)])])

    def test_a_term_of_another_type_is_a_type_error(self):
        from polypus.qml import Observable

        with pytest.raises(TypeError, match="PauliTerm"):
            Observable([(1.0, "z0")])


# ─────────────────────────────────────────────────────────────────────────────
# Encoder and Ansatz
# ─────────────────────────────────────────────────────────────────────────────


class TestEncoder:
    def test_chaining_appends_to_the_same_object(self):
        from polypus.qml import Encoder

        encoder = Encoder(4)
        assert encoder.angle_encoder("ry").iqp_encoder("linear") is encoder
        assert (encoder.num_qubits, encoder.num_layers) == (4, 2)
        assert repr(encoder) == "Encoder(num_qubits=4, num_layers=2)"

    def test_every_encoder_is_available(self):
        from polypus.qml import Encoder

        Encoder(2).amplitude_encoder()
        Encoder(2).basis_encoder()
        Encoder(2).iqp_encoder()
        for axis in ["rx", "ry", "rz"]:
            Encoder(2).angle_encoder(axis)

    def test_zero_qubits_is_rejected(self):
        from polypus.qml import Encoder

        with pytest.raises(ValueError, match="at least one qubit"):
            Encoder(0)

    def test_an_unknown_axis_is_rejected_and_nothing_is_appended(self):
        from polypus.qml import Encoder

        encoder = Encoder(2)
        with pytest.raises(ValueError, match='"rx", "ry" or "rz"'):
            encoder.angle_encoder("rq")
        assert encoder.num_layers == 0

    def test_an_amplitude_encoder_after_another_layer_is_rejected_at_once(self):
        from polypus.qml import Encoder

        encoder = Encoder(2).basis_encoder()
        with pytest.raises(ValueError):
            encoder.amplitude_encoder()
        assert encoder.num_layers == 1


class TestAnsatz:
    def test_parameter_counts(self):
        from polypus.qml import Ansatz

        assert Ansatz(4).num_params == 0
        # A convolution block shares its parameters across every pair.
        assert Ansatz(4).conv("basic").num_params == 4
        assert Ansatz(4).conv("cartan").num_params == 3
        # RealAmplitudes on 2 qubits, 1 rep: 2 rotation layers of 2 Ry.
        assert Ansatz(2).real_amplitudes(1).num_params == 4
        assert Ansatz(2).hardware_efficient(1, rotations=["ry"]).num_params == 4

    def test_qcnn_with_data_reuploading(self):
        from polypus.qml import Ansatz

        ansatz = (
            Ansatz(4)
            .conv("basic", pairing="even_pairs")
            .pool("basic", keep="odd_positions")
            .angle_encoder("rx")
            .conv("cartan")
            .basis_encoder()
            .iqp_encoder("circular")
            .hardware_efficient(1, entangler="cz", final_rotation_layer=False)
        )
        assert ansatz.num_layers == 7
        assert repr(ansatz).startswith("Ansatz(num_qubits=4, num_layers=7, num_params=")

    def test_pooling_below_two_qubits_is_rejected_at_the_call(self):
        from polypus.qml import Ansatz

        ansatz = Ansatz(2).pool("basic")
        with pytest.raises(ValueError):
            ansatz.pool("basic")
        assert ansatz.num_layers == 1

    @pytest.mark.parametrize(
        ("method", "args", "valid"),
        [
            ("conv", ("dense",), '"basic" or "cartan"'),
            ("conv", ("basic", "pairs"), '"alternating"'),
            ("pool", ("cartan",), '"basic"'),
            ("pool", ("basic", "left"), '"odd_positions"'),
            ("hardware_efficient", (1, ["ry"], "cy"), '"cx" or "cz"'),
            ("hardware_efficient", (1, ["ry"], "cx", "ring"), '"full"'),
            ("hardware_efficient", (1, ["rq"]), '"rz"'),
        ],
    )
    def test_unknown_options_name_the_valid_ones(self, method, args, valid):
        from polypus.qml import Ansatz

        with pytest.raises(ValueError, match=valid):
            getattr(Ansatz(4), method)(*args)


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────


class TestObjective:
    def test_defaults(self):
        from polypus.qml import Objective, Z

        objective = Objective([Z(0)])
        assert objective.decision == "raw"
        assert objective.loss is None
        assert objective.threshold is None
        assert objective.num_observables == 1
        assert repr(objective) == (
            'Objective(num_observables=1, decision="raw", loss=None)'
        )

    def test_the_three_observable_forms(self):
        from polypus.qml import Objective, Observable, Z

        objective = Objective(
            [Z(0), [("z", 1)], Observable([(0.5, Z(2)), (0.5, Z(3))])],
            decision="argmax",
            loss="categorical_cross_entropy",
        )
        assert objective.num_observables == 3

    def test_threshold_is_required_by_threshold_only(self):
        from polypus.qml import Objective, Z

        objective = Objective([Z(0)], decision="threshold", threshold=0.2, loss="hinge")
        assert (objective.decision, objective.threshold) == ("threshold", 0.2)
        with pytest.raises(ValueError, match="requires a threshold"):
            Objective([Z(0)], decision="threshold")
        with pytest.raises(ValueError, match="only valid with"):
            Objective([Z(0)], decision="sign", threshold=0.2)

    @pytest.mark.parametrize(
        "loss",
        ["squared_error", "binary_cross_entropy", "hinge", "expectation", None],
    )
    def test_argmax_takes_only_the_categorical_loss(self, loss):
        from polypus.qml import Objective, Z

        with pytest.raises(ValueError, match="categorical_cross_entropy|Argmax"):
            Objective([Z(0), Z(1)], decision="argmax", loss=loss)

    @pytest.mark.parametrize("decision", ["raw", "sign"])
    def test_the_categorical_loss_takes_only_argmax(self, decision):
        from polypus.qml import Objective, Z

        with pytest.raises(ValueError):
            Objective([Z(0)], decision=decision, loss="categorical_cross_entropy")

    def test_argmax_needs_two_observables(self):
        from polypus.qml import Objective, Z

        with pytest.raises(ValueError):
            Objective([Z(0)], decision="argmax", loss="categorical_cross_entropy")

    def test_unknown_options_name_the_valid_ones(self):
        from polypus.qml import Objective, Z

        with pytest.raises(ValueError, match='"expectation"'):
            Objective([Z(0)], loss="l2")
        with pytest.raises(ValueError, match='"argmax"'):
            Objective([Z(0)], decision="max")

    def test_an_observable_of_another_type_is_a_type_error(self):
        from polypus.qml import Objective

        with pytest.raises(TypeError, match="PauliTerm"):
            Objective(["Z0"])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

_ROWS = [[0.0, 1.0], [0.5, 2.0], [1.0, 3.0], [0.2, 0.4]]


class TestDataset:
    def test_labeled(self):
        from polypus.qml import Dataset

        dataset = Dataset(_ROWS, [0, 1, 1, 0])
        assert (dataset.num_samples, dataset.num_features) == (4, 2)
        assert dataset.labeled
        assert dataset.labels == [0.0, 1.0, 1.0, 0.0]
        assert repr(dataset) == "Dataset(num_samples=4, num_features=2)"

    def test_unlabeled(self):
        from polypus.qml import Dataset

        dataset = Dataset(_ROWS)
        assert not dataset.labeled
        assert dataset.labels is None
        assert repr(dataset) == "Dataset(num_samples=4, num_features=2, unlabeled)"

    def test_numpy_input(self):
        import numpy as np
        from polypus.qml import Dataset

        dataset = Dataset(np.array(_ROWS), np.array([0, 1, 1, 0]))
        assert dataset.labels == [0.0, 1.0, 1.0, 0.0]

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            ([], []),
            ([[0.0, 1.0], [1.0]], [0, 1]),
            ([[0.0, math.nan]], [0]),
            ([[0.0], [1.0]], [0]),
            ([[0.0]], [math.inf]),
        ],
    )
    def test_invalid_data_is_rejected(self, x, y):
        from polypus.qml import Dataset

        with pytest.raises(ValueError):
            Dataset(x, y)

    @pytest.mark.parametrize("labeled", [True, False])
    def test_a_split_keeps_the_label_mark_and_replays_with_its_seed(self, labeled):
        from polypus.qml import Dataset

        rows = [[float(i)] for i in range(10)]
        dataset = Dataset(rows, list(range(10))) if labeled else Dataset(rows)
        train, test = dataset.train_test_split(0.3, seed=7)
        assert (train.num_samples, test.num_samples) == (7, 3)
        assert train.labeled == test.labeled == labeled
        again, _ = dataset.train_test_split(0.3, seed=7)
        assert again.feature_ranges() == train.feature_ranges()
        if labeled:
            assert again.labels == train.labels

    def test_split_without_a_seed_draws_one(self):
        from polypus.qml import Dataset

        train, test = Dataset([[float(i)] for i in range(10)]).train_test_split(0.5)
        assert (train.num_samples, test.num_samples) == (5, 5)

    @pytest.mark.parametrize("fraction", [0.0, 1.0])
    def test_a_split_needs_both_halves(self, fraction):
        from polypus.qml import Dataset

        with pytest.raises(ValueError):
            Dataset(_ROWS).train_test_split(fraction, seed=1)

    def test_a_frozen_scaler_replays_on_a_test_set(self):
        from polypus.qml import Dataset

        train = Dataset(_ROWS, [0, 1, 1, 0])
        ranges = train.feature_ranges()
        assert ranges == [(0.0, 1.0), (0.4, 3.0)]
        train.scale_features_to(0.0, math.pi)
        assert train.feature_ranges() == [(0.0, math.pi), (0.0, math.pi)]
        test = Dataset([[2.0, 0.4]], [1])
        test.scale_features_with(ranges, 0.0, math.pi)
        assert test.feature_ranges() == [(2 * math.pi, 2 * math.pi), (0.0, 0.0)]
        with pytest.raises(ValueError):
            test.scale_features_with(ranges[:1], 0.0, math.pi)


@pytest.mark.parametrize(
    "name", ["Encoder", "Ansatz", "Objective", "Dataset", "PauliTerm", "Observable"]
)
def test_classes_report_the_qml_module(name):
    import polypus.qml as qml

    assert getattr(qml, name).__module__ == "polypus.qml"
