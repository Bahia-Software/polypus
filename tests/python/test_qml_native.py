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


# ─────────────────────────────────────────────────────────────────────────────
# qml.train in the native form
# ─────────────────────────────────────────────────────────────────────────────

# Two separable classes on one feature: ry(x) puts small x near |0> (⟨Z⟩ ≈ +1)
# and large x near |1> (⟨Z⟩ ≈ −1).
_X = [[0.1], [0.3], [2.9], [3.0], [0.2], [2.8]]
_Y = [1, 1, -1, -1, 1, -1]


def _pieces(num_qubits=1):
    from polypus.qml import Ansatz, Encoder

    return (
        Encoder(num_qubits).angle_encoder("ry"),
        Ansatz(num_qubits).real_amplitudes(1),
    )


def _classifier(**kwargs):
    from polypus.qml import Objective, Z

    return Objective([Z(0)], decision="sign", loss=kwargs.pop("loss", "hinge"))


def _native_train(method=None, x=_X, y=_Y, objective=None, **kwargs):
    import polypus

    encoder, ansatz = kwargs.pop("pieces", None) or _pieces()
    if y is not None:
        kwargs["y_train"] = y
    return polypus.qml.train(
        encoder,
        ansatz,
        x,
        method or polypus.Adam(max_iters=5, patience=10_000),
        expectation_function=objective or _classifier(),
        seed=kwargs.pop("seed", 3),
        **kwargs,
    )


@pytest.mark.integration
class TestNativeTraining:
    def test_supervised_run_with_the_defaults(self):
        result = _native_train()
        assert len(result.best_params) == 2  # RealAmplitudes(1) on one qubit
        assert result.iterations_run == 5
        assert math.isfinite(result.best_fitness)
        assert result.fitness_history[-1] == result.best_fitness

    def test_exact_mode_does_not_depend_on_shots(self):
        r1 = _native_train(exact=True, shots=16)
        r2 = _native_train(exact=True, shots=4096)
        assert r1.best_params == r2.best_params
        assert r1.best_fitness == r2.best_fitness

    def test_adam_climbs_in_exact_mode(self):
        import polypus

        result = _native_train(
            polypus.Adam(max_iters=60, learning_rate=0.1, patience=10_000), exact=True
        )
        # The hinge loss of a perfect margin-1 classifier is 0.
        assert result.best_fitness > -0.2

    def test_gradient_free_optimizers_train_too(self):
        import polypus

        result = _native_train(polypus.DE(generations=2, population_size=4))
        assert len(result.best_params) == 2

    def test_a_dataset_carries_its_labels(self):
        from polypus.qml import Dataset

        result = _native_train(x=Dataset(_X, _Y), y=None, exact=True)
        assert result.best_fitness == _native_train(exact=True).best_fitness

    def test_unsupervised_maximises_the_observable(self):
        import polypus
        from polypus.qml import Objective, Observable, Z

        # Maximise -⟨Z⟩ = -cos(x + φ): close samples can all be rotated near |1>.
        objective = Objective([Observable([(-1.0, Z(0))])])
        result = _native_train(
            polypus.Adam(max_iters=60, learning_rate=0.2, patience=10_000),
            x=[[0.1], [0.2], [0.3]],
            y=None,
            objective=objective,
            exact=True,
        )
        assert result.best_fitness > 0.5

    def test_minibatches_report_the_full_dataset(self):
        r = _native_train(batch_size=2, exact=True)
        assert len(r.best_params) == 2

    def test_explicit_aer_backend(self):
        result = _native_train(backend="aer", shots=64)
        assert len(result.best_params) == 2

    def test_data_reuploading_with_an_empty_encoder(self):
        from polypus.qml import Ansatz, Encoder

        pieces = (Encoder(1), Ansatz(1).angle_encoder("ry").real_amplitudes(1))
        assert len(_native_train(pieces=pieces).best_params) == 2

    def test_dimensions_is_inferred_and_checked(self):
        assert len(_native_train(dimensions=2).best_params) == 2
        with pytest.raises(ValueError, match="does not match"):
            _native_train(dimensions=3)


class TestNativeTrainingRejections:
    """Everything here is rejected before a backend exists."""

    def test_missing_expectation_function_is_a_type_error(self):
        import polypus

        encoder, ansatz = _pieces()
        with pytest.raises(TypeError, match="expectation_function"):
            polypus.qml.train(encoder, ansatz, _X, polypus.Adam(), y_train=_Y)

    def test_a_callable_needs_qiskit_circuits(self):
        with pytest.raises(TypeError, match="must be a polypus.qml.Objective"):
            _native_train(objective=lambda b, y: 0.0)

    def test_mixed_forms_are_rejected(self):
        import polypus
        from polypus.qml import Encoder
        from qiskit import QuantumCircuit

        with pytest.raises(ValueError, match="same form"):
            polypus.qml.train(
                Encoder(1).angle_encoder("ry"),
                QuantumCircuit(1),
                _X,
                polypus.Adam(),
                expectation_function=_classifier(),
                y_train=_Y,
            )

    def test_swapped_pieces_are_rejected(self):
        encoder, ansatz = _pieces()
        with pytest.raises(ValueError, match="same form"):
            _native_train(pieces=(ansatz, encoder))

    def test_widths_must_match(self):
        from polypus.qml import Ansatz, Encoder

        pieces = (Encoder(2).angle_encoder("ry"), Ansatz(1).real_amplitudes(1))
        with pytest.raises(ValueError, match="2 qubits but the Ansatz has 1"):
            _native_train(pieces=pieces)

    def test_a_model_that_never_loads_the_data(self):
        from polypus.qml import Ansatz, Encoder

        with pytest.raises(ValueError, match="no layer loads the data"):
            _native_train(pieces=(Encoder(1), Ansatz(1).real_amplitudes(1)))

    def test_labels_need_a_supervised_loss(self):
        from polypus.qml import Objective, Z

        with pytest.raises(ValueError, match="needs a loss"):
            _native_train(objective=Objective([Z(0)], decision="sign"))
        with pytest.raises(ValueError, match="does not use labels"):
            _native_train(objective=Objective([Z(0)], loss="expectation"))

    def test_a_supervised_loss_needs_labels(self):
        from polypus.qml import Dataset

        with pytest.raises(ValueError, match="needs labels"):
            _native_train(y=None)
        with pytest.raises(ValueError, match="needs labels"):
            _native_train(x=Dataset(_X), y=None)

    def test_a_dataset_and_y_train_together(self):
        from polypus.qml import Dataset

        with pytest.raises(ValueError, match="carries its own labels"):
            _native_train(x=Dataset(_X, _Y))

    def test_label_domain_is_checked(self):
        with pytest.raises(ValueError):
            _native_train(y=[0, 1, 0, 1, 0, 1])  # hinge needs ±1

    def test_too_many_features_for_the_encoder(self):
        with pytest.raises(ValueError):
            _native_train(x=[[0.1, 0.2]] * 6)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"backend": "aer"},
            {"infrastructure": "cunqa", "nodes": 1, "cores_per_qpu": 1},
        ],
    )
    def test_exact_needs_the_local_native_simulator(self, kwargs):
        with pytest.raises(ValueError, match="exact=True needs"):
            _native_train(exact=True, **kwargs)

    @pytest.mark.parametrize("size", [0, 6, 7])
    def test_batch_size_must_be_smaller_than_the_dataset(self, size):
        with pytest.raises(ValueError, match="batch_size"):
            _native_train(batch_size=size)

    def test_the_default_backend_is_the_native_simulator(self):
        # The native simulator is noiseless, so a noise model without a backend
        # is refused: proof that `backend=None` resolved to "polypus".
        with pytest.raises(ValueError, match="noise"):
            _native_train(noise_model=object())
