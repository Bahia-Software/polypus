"""
The native pieces of ``polypus.qml`` and their checks (contract C-11): Pauli
terms and observables, ``Encoder``, ``Ansatz``, ``Readout``, ``Dataset`` and
``Model``, and the rejection of native pieces by the Qiskit form's
``qml.train``/``qml.predict``.

Nothing here trains or runs a circuit: these pin construction, the closed option
sets (plan F6), the checks each builder call and ``Model`` run without the data
(plans F1, G3), the unlabeled ``Dataset`` (plan F4) and the ``Dataset``'s split
and scaling arithmetic. Training is in ``test_qml_native_training.py``,
inference and saving in ``test_qml_native_inference.py``, and the semantics of
each layer and observable in ``test_qml_native_layers.py``.
"""

import math

import pytest
from qml_native_helpers import make_model

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
# Readout
# ─────────────────────────────────────────────────────────────────────────────


class TestReadout:
    def test_defaults(self):
        from polypus.qml import Readout, Z

        readout = Readout([Z(0)])
        assert readout.decision == "raw"
        assert readout.threshold is None
        assert readout.num_observables == 1
        assert repr(readout) == 'Readout(num_observables=1, decision="raw")'

    def test_the_three_observable_forms(self):
        from polypus.qml import Observable, Readout, Z

        readout = Readout(
            [Z(0), [("z", 1)], Observable([(0.5, Z(2)), (0.5, Z(3))])],
            decision="argmax",
        )
        assert readout.num_observables == 3

    def test_threshold_is_required_by_threshold_only(self):
        from polypus.qml import Readout, Z

        readout = Readout([Z(0)], decision="threshold", threshold=0.2)
        assert (readout.decision, readout.threshold) == ("threshold", 0.2)
        with pytest.raises(ValueError, match="requires a threshold"):
            Readout([Z(0)], decision="threshold")
        with pytest.raises(ValueError, match="only valid with"):
            Readout([Z(0)], decision="sign", threshold=0.2)

    def test_argmax_needs_two_observables(self):
        from polypus.qml import Readout, Z

        with pytest.raises(ValueError):
            Readout([Z(0)], decision="argmax")

    def test_an_unknown_decision_names_the_valid_ones(self):
        from polypus.qml import Readout, Z

        with pytest.raises(ValueError, match='"argmax"'):
            Readout([Z(0)], decision="max")

    def test_an_observable_of_another_type_is_a_type_error(self):
        from polypus.qml import Readout

        with pytest.raises(TypeError, match="PauliTerm"):
            Readout(["Z0"])


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


def _indexed(n=6):
    """``n`` samples whose first feature *is* the sample index, so a
    partition's ``feature_ranges()[0]`` reports which samples it received."""
    from polypus.qml import Dataset

    return Dataset(
        [[float(i), 1.0 + 0.5 * i] for i in range(n)],
        [1.0 if i % 2 else -1.0 for i in range(n)],
    )


class TestDatasetSplit:
    def test_the_test_partition_rounds_down(self):
        # floor(6 × 0.4) = 2 test samples, so 4 train, both at full width.
        train, test = _indexed().train_test_split(0.4, seed=7)
        assert (train.num_samples, test.num_samples) == (4, 2)
        assert train.num_features == test.num_features == 2

    def test_different_seeds_give_different_partitions(self):
        # A property of the deterministic shuffle, not a probabilistic hope.
        picks = {
            _indexed().train_test_split(1 / 3, seed=s)[1].feature_ranges()[0]
            for s in range(1, 9)
        }
        assert len(picks) > 1

    def test_the_test_sample_is_missing_from_train(self):
        # floor(6 × 0.2) = 1 test sample, so its index range is (i, i), and a
        # partition cannot also hold it in train.
        train, test = _indexed().train_test_split(0.2, seed=7)
        lo, hi = test.feature_ranges()[0]
        assert lo == hi and lo in {float(i) for i in range(6)}
        train_lo, train_hi = train.feature_ranges()[0]
        if lo == 0.0:
            assert train_lo == 1.0
        elif lo == 5.0:
            assert train_hi == 4.0

    @pytest.mark.parametrize("fraction", [-0.1, 1.5, math.nan])
    def test_the_fraction_lies_in_the_open_unit_interval(self, fraction):
        with pytest.raises(ValueError, match="test_fraction"):
            _indexed().train_test_split(fraction, seed=7)

    def test_a_partition_emptied_by_the_rounding(self):
        # floor(5 × 0.1) = 0: an in-range fraction can still empty a partition,
        # which would leave a problem averaging over no samples.
        with pytest.raises(ValueError, match="empty"):
            _indexed(n=5).train_test_split(0.1, seed=7)

    @pytest.mark.parametrize("fraction", [0.1, 0.5, 0.9])
    def test_one_sample_cannot_be_split(self, fraction):
        with pytest.raises(ValueError, match="empty"):
            _indexed(n=1).train_test_split(fraction, seed=7)


class TestDatasetScaling:
    @staticmethod
    def _ds(rows):
        from polypus.qml import Dataset

        return Dataset(rows)

    def test_ranges_are_per_feature(self):
        ds = self._ds([[0.0, 20.0], [10.0, 5.0], [4.0, 30.0]])
        assert ds.feature_ranges() == [(0.0, 10.0), (5.0, 30.0)]

    def test_scaling_spans_the_interval(self):
        ds = self._ds([[0.0, 20.0], [10.0, 5.0], [4.0, 30.0]])
        ds.scale_features_to(0.0, 1.0)
        assert ds.feature_ranges() == [(0.0, 1.0), (0.0, 1.0)]

    def test_scaling_a_scaled_set_is_the_identity(self):
        # 0 → 0, 5 → 0.5, 10 → 1 exactly; scaling again must not move anything.
        ds = self._ds([[0.0], [5.0], [10.0]])
        ds.scale_features_to(0.0, 1.0)
        before = ds.feature_ranges()
        ds.scale_features_to(0.0, 1.0)
        assert ds.feature_ranges() == before == [(0.0, 1.0)]

    def test_a_constant_feature_maps_to_lo(self):
        ds = self._ds([[7.0, 1.0], [7.0, 2.0]])
        ds.scale_features_to(3.0, 9.0)
        assert ds.feature_ranges() == [(3.0, 3.0), (3.0, 9.0)]

    def test_a_frozen_scaler_maps_beyond_its_range(self):
        train, test = self._ds([[0.0], [10.0]]), self._ds([[5.0], [15.0]])
        test.scale_features_with(train.feature_ranges(), 0.0, 1.0)
        assert test.feature_ranges() == [(0.5, 1.5)]

    def test_split_then_freeze_the_train_ranges(self):
        train, test = _indexed(n=8).train_test_split(0.25, seed=7)
        ranges = train.feature_ranges()
        train.scale_features_to(0.0, math.pi)
        test.scale_features_with(ranges, 0.0, math.pi)
        assert train.feature_ranges()[0] == (0.0, math.pi)
        test_lo, test_hi = test.feature_ranges()[0]
        assert -3.2 < test_lo <= test_hi < 6.3


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


class TestModel:
    def test_counts_and_repr(self):
        model = make_model(2)
        assert (model.num_qubits, model.num_params) == (2, 4)
        assert repr(model) == "Model(num_qubits=2, num_layers=2, num_params=4)"

    def test_the_pieces_are_copied(self):
        from polypus.qml import Ansatz

        ansatz = Ansatz(1).real_amplitudes(1)
        model = make_model(ansatz=ansatz)
        ansatz.real_amplitudes(1)
        assert model.num_params == 2

    def test_widths_must_match(self):
        from polypus.qml import Ansatz, Encoder

        with pytest.raises(ValueError, match="2 qubits but the Ansatz has 1"):
            make_model(
                encoder=Encoder(2).angle_encoder("ry"),
                ansatz=Ansatz(1).real_amplitudes(1),
            )

    def test_some_layer_must_load_the_data(self):
        from polypus.qml import Encoder

        with pytest.raises(ValueError, match="no layer loads the data"):
            make_model(encoder=Encoder(1))

    def test_data_reuploading_with_an_empty_encoder(self):
        from polypus.qml import Ansatz, Encoder

        model = make_model(
            encoder=Encoder(1), ansatz=Ansatz(1).angle_encoder("ry").real_amplitudes(1)
        )
        assert model.num_params == 2

    def test_the_model_needs_trainable_parameters(self):
        from polypus.qml import Ansatz

        with pytest.raises(ValueError):
            make_model(ansatz=Ansatz(1))

    def test_readout_positions_follow_pooling(self):
        from polypus.qml import Ansatz, Readout, Z

        # Pooling 2 qubits leaves one active: logical position 1 does not exist.
        with pytest.raises(ValueError):
            make_model(2, ansatz=Ansatz(2).pool("basic"), readout=Readout([Z(1)]))
        assert make_model(2, ansatz=Ansatz(2).pool("basic"), readout=Readout([Z(0)]))


@pytest.mark.parametrize(
    "name",
    [
        "Encoder",
        "Ansatz",
        "Readout",
        "Model",
        "TrainedModel",
        "ExactResult",
        "Dataset",
        "PauliTerm",
        "Observable",
    ],
)
def test_classes_report_the_qml_module(name):
    import polypus.qml as qml

    assert getattr(qml, name).__module__ == "polypus.qml"


# ─────────────────────────────────────────────────────────────────────────────
# qml.train / qml.predict run Qiskit circuits only
# ─────────────────────────────────────────────────────────────────────────────


def _qiskit_circuit():
    from qiskit.circuit import ParameterVector, QuantumCircuit

    qc = QuantumCircuit(1)
    qc.ry(ParameterVector("p", 1)[0], 0)
    return qc


@pytest.mark.parametrize("slot", ["feature_map", "ansatz"])
def test_qml_train_rejects_native_pieces(slot):
    import polypus
    from polypus.qml import Ansatz, Encoder

    pieces = {"feature_map": _qiskit_circuit(), "ansatz": _qiskit_circuit()}
    pieces[slot] = (
        Encoder(1).angle_encoder("ry") if slot == "feature_map" else Ansatz(1)
    )
    with pytest.raises(TypeError, match="qml.train takes Qiskit QuantumCircuits"):
        polypus.qml.train(
            pieces["feature_map"],
            pieces["ansatz"],
            [[0.1]],
            polypus.DE(generations=1, population_size=4),
            64,
            1,
            1,
            lambda bitstring: 0.0,
            "local",
            1,
            1,
            "native_rejected",
        )


def test_qml_predict_rejects_native_pieces():
    import polypus
    from polypus.qml import Ansatz, Encoder

    with pytest.raises(TypeError, match="qml.predict takes Qiskit QuantumCircuits"):
        polypus.qml.predict(
            Encoder(1).angle_encoder("ry"),
            Ansatz(1).real_amplitudes(1),
            [[0.1]],
            [0.0, 0.0],
            64,
            "local",
        )
