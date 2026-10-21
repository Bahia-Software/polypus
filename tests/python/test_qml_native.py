"""
The native API of ``polypus.qml`` (plan D4-R): ``Encoder``, ``Ansatz`` and
``Readout`` joined into a ``Model``, trained with ``Model.fit`` on rows or a
``Dataset``, plus the Pauli readout helpers (``Z``/``X``/``Y``, ``PauliTerm``,
``Observable``), and the rejection of these pieces by the Qiskit form's
``qml.train``/``qml.predict``.

Construction tests need no backend: the closed option sets (plan F6), the
checks each builder call and the ``Model`` run without the data (plans F1, G3)
and the unlabeled ``Dataset`` (plan F4). The ``fit`` tests train on the native
simulator, or are rejected before any backend exists (plans F2, F3, F5).
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


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


def _model(num_qubits=1, readout=None, encoder=None, ansatz=None):
    from polypus.qml import Ansatz, Encoder, Model, Readout, Z

    return Model(
        encoder or Encoder(num_qubits).angle_encoder("ry"),
        ansatz or Ansatz(num_qubits).real_amplitudes(1),
        readout or Readout([Z(0)], decision="sign"),
    )


class TestModel:
    def test_counts_and_repr(self):
        model = _model(2)
        assert (model.num_qubits, model.num_params) == (2, 4)
        assert repr(model) == "Model(num_qubits=2, num_layers=2, num_params=4)"

    def test_the_pieces_are_copied(self):
        from polypus.qml import Ansatz

        ansatz = Ansatz(1).real_amplitudes(1)
        model = _model(ansatz=ansatz)
        ansatz.real_amplitudes(1)
        assert model.num_params == 2

    def test_widths_must_match(self):
        from polypus.qml import Ansatz, Encoder

        with pytest.raises(ValueError, match="2 qubits but the Ansatz has 1"):
            _model(
                encoder=Encoder(2).angle_encoder("ry"),
                ansatz=Ansatz(1).real_amplitudes(1),
            )

    def test_some_layer_must_load_the_data(self):
        from polypus.qml import Encoder

        with pytest.raises(ValueError, match="no layer loads the data"):
            _model(encoder=Encoder(1))

    def test_data_reuploading_with_an_empty_encoder(self):
        from polypus.qml import Ansatz, Encoder

        model = _model(
            encoder=Encoder(1), ansatz=Ansatz(1).angle_encoder("ry").real_amplitudes(1)
        )
        assert model.num_params == 2

    def test_the_model_needs_trainable_parameters(self):
        from polypus.qml import Ansatz

        with pytest.raises(ValueError):
            _model(ansatz=Ansatz(1))

    def test_readout_positions_follow_pooling(self):
        from polypus.qml import Ansatz, Readout, Z

        # Pooling 2 qubits leaves one active: logical position 1 does not exist.
        with pytest.raises(ValueError):
            _model(2, ansatz=Ansatz(2).pool("basic"), readout=Readout([Z(1)]))
        assert _model(2, ansatz=Ansatz(2).pool("basic"), readout=Readout([Z(0)]))


# ─────────────────────────────────────────────────────────────────────────────
# Model.fit
# ─────────────────────────────────────────────────────────────────────────────

# Two separable classes on one feature: ry(x) puts small x near |0> (⟨Z⟩ ≈ +1)
# and large x near |1> (⟨Z⟩ ≈ −1).
_X = [[0.1], [0.3], [2.9], [3.0], [0.2], [2.8]]
_Y = [1, 1, -1, -1, 1, -1]


def _fit(model=None, x=_X, y=_Y, **kwargs):
    import polypus

    kwargs.setdefault("method", polypus.Adam(max_iters=5, patience=10_000))
    kwargs.setdefault("seed", 3)
    if y is not None:
        kwargs.setdefault("loss", "hinge")
    return (model or _model()).fit(x, y, **kwargs)


@pytest.mark.integration
class TestFit:
    def test_supervised_run_with_the_defaults(self):
        result = _fit()
        assert len(result.best_params) == 2
        assert result.iterations_run == 5
        assert math.isfinite(result.best_fitness)
        assert result.fitness_history[-1] == result.best_fitness

    def test_exact_mode_does_not_depend_on_shots(self):
        r1 = _fit(exact=True, shots=16)
        r2 = _fit(exact=True, shots=4096)
        assert r1.best_params == r2.best_params
        assert r1.best_fitness == r2.best_fitness

    def test_adam_climbs_in_exact_mode(self):
        import polypus

        result = _fit(
            method=polypus.Adam(max_iters=60, learning_rate=0.1, patience=10_000),
            exact=True,
        )
        # The hinge loss of a perfect margin-1 classifier is 0.
        assert result.best_fitness > -0.2

    def test_gradient_free_optimizers_train_too(self):
        import polypus

        result = _fit(method=polypus.DE(generations=2, population_size=4))
        assert len(result.best_params) == 2

    def test_a_dataset_carries_its_labels(self):
        from polypus.qml import Dataset

        result = _fit(x=Dataset(_X, _Y), y=None, loss="hinge", exact=True)
        assert result.best_fitness == _fit(exact=True).best_fitness

    def test_unsupervised_maximises_the_readout(self):
        import polypus
        from polypus.qml import Observable, Readout, Z

        # Maximise -⟨Z⟩ = -cos(x + φ): close samples can all be rotated near |1>.
        model = _model(readout=Readout([Observable([(-1.0, Z(0))])]))
        result = _fit(
            model,
            x=[[0.1], [0.2], [0.3]],
            y=None,
            method=polypus.Adam(max_iters=60, learning_rate=0.2, patience=10_000),
            exact=True,
        )
        assert result.best_fitness > 0.5

    def test_minibatches(self):
        assert len(_fit(batch_size=2, exact=True).best_params) == 2

    def test_explicit_aer_backend(self):
        assert len(_fit(backend="aer", shots=64).best_params) == 2

    def test_the_seed_replays_the_run(self):
        r1, r2 = _fit(seed=11), _fit(seed=11)
        assert r1.seed == r2.seed == 11
        assert r1.best_params == r2.best_params
        assert r1.fitness_history == r2.fitness_history


class TestFitRejections:
    """Everything here is rejected before a backend exists."""

    def test_method_is_required_and_keyword_only(self):
        import polypus

        with pytest.raises(TypeError):
            _model().fit(_X, _Y)
        with pytest.raises(TypeError):
            _model().fit(_X, _Y, polypus.Adam())

    def test_labels_need_a_supervised_loss(self):
        with pytest.raises(ValueError, match="needs a loss"):
            _fit(loss=None)
        with pytest.raises(ValueError, match="does not use labels"):
            _fit(loss="expectation")

    def test_a_supervised_loss_needs_labels(self):
        from polypus.qml import Dataset

        with pytest.raises(ValueError, match="needs labels"):
            _fit(y=None, loss="hinge")
        with pytest.raises(ValueError, match="needs labels"):
            _fit(x=Dataset(_X), y=None, loss="hinge")

    def test_a_dataset_and_y_together(self):
        from polypus.qml import Dataset

        with pytest.raises(ValueError, match="carries its own labels"):
            _fit(x=Dataset(_X, _Y))

    def test_an_unknown_loss_names_the_valid_ones(self):
        with pytest.raises(ValueError, match='"expectation"'):
            _fit(loss="l2")

    @pytest.mark.parametrize("loss", ["squared_error", "hinge", "expectation"])
    def test_argmax_takes_only_the_categorical_loss(self, loss):
        from polypus.qml import Readout, Z

        model = _model(2, readout=Readout([Z(0), Z(1)], decision="argmax"))
        y = None if loss == "expectation" else [0, 1, 0, 1, 0, 1]
        with pytest.raises(ValueError, match="CategoricalCrossEntropy"):
            _fit(model, x=[[0.1, 0.2]] * 6, y=y, loss=loss)

    def test_the_categorical_loss_takes_only_argmax(self):
        with pytest.raises(ValueError, match="CategoricalCrossEntropy"):
            _fit(y=[0, 1, 0, 1, 0, 1], loss="categorical_cross_entropy")

    def test_label_domain_is_checked(self):
        with pytest.raises(ValueError):
            _fit(y=[0, 1, 0, 1, 0, 1])  # hinge needs ±1

    def test_too_many_features_for_the_encoder(self):
        with pytest.raises(ValueError):
            _fit(x=[[0.1, 0.2]] * 6)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"backend": "aer"},
            {"infrastructure": "cunqa", "nodes": 1, "cores_per_qpu": 1},
        ],
    )
    def test_exact_needs_the_local_native_simulator(self, kwargs):
        with pytest.raises(ValueError, match="exact=True needs"):
            _fit(exact=True, **kwargs)

    @pytest.mark.parametrize("size", [0, 6, 7])
    def test_batch_size_must_be_smaller_than_the_dataset(self, size):
        with pytest.raises(ValueError, match="batch_size"):
            _fit(batch_size=size)

    def test_the_default_backend_is_the_native_simulator(self):
        # The native simulator is noiseless, so a noise model without a backend
        # is refused: proof that `backend=None` resolved to "polypus".
        with pytest.raises(ValueError, match="noise"):
            _fit(noise_model=object())

    def test_an_invalid_id_is_rejected(self):
        with pytest.raises(ValueError):
            _fit(id="no spaces")


# ─────────────────────────────────────────────────────────────────────────────
# Model.bind and TrainedModel
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trained():
    """A one-qubit classifier trained exactly: ⟨Z⟩ = cos(x + φ) with φ ≈ 0."""
    import polypus

    model = _model()
    result = model.fit(
        _X,
        _Y,
        method=polypus.Adam(max_iters=60, learning_rate=0.1, patience=10_000),
        loss="hinge",
        exact=True,
        seed=3,
    )
    return model.bind(result.best_params)


class TestBind:
    def test_bound_model(self):
        model = _model()
        trained = model.bind([0.1, 0.2])
        assert trained.params == [0.1, 0.2]
        assert trained.model.num_params == model.num_params == 2
        assert repr(trained) == "TrainedModel(num_qubits=1, num_params=2)"

    def test_the_model_is_unchanged(self):
        model = _model()
        model.bind([0.1, 0.2])
        model.bind([0.3, 0.4])
        assert model.num_params == 2

    def test_the_parameter_count_must_match(self):
        with pytest.raises(ValueError, match="params has 3 values"):
            _model().bind([0.1, 0.2, 0.3])

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_parameters_must_be_finite(self, value):
        with pytest.raises(ValueError, match=r"params\[1\]"):
            _model().bind([0.1, value])


@pytest.mark.integration
class TestInference:
    def test_predict_classifies_the_training_set(self, trained):
        labels = trained.predict(_X, exact=True)
        assert labels == _Y
        assert all(type(label) is int for label in labels)

    def test_sampled_predict_agrees(self, trained):
        assert trained.predict(_X, shots=2048, seed=5) == _Y

    def test_a_dataset_is_accepted_and_its_labels_unused(self, trained):
        from polypus.qml import Dataset

        assert trained.predict(Dataset(_X), exact=True) == _Y
        assert trained.predict(Dataset(_X, [0] * 6), exact=True) == _Y

    def test_run_returns_counts_per_row(self, trained):
        import polypus

        run = trained.run(_X, shots=128, seed=5)
        assert isinstance(run, polypus.RunResult)
        assert len(run.counts) == len(_X)
        assert all(sum(counts.values()) == 128 for counts in run.counts)
        assert run.merged_counts is None
        assert (run.seed, run.backend, run.infrastructure) == (5, "polypus", "local")

    def test_the_seed_replays_the_counts(self, trained):
        assert (
            trained.run(_X, shots=64, seed=9).counts
            == trained.run(_X, shots=64, seed=9).counts
        )

    def test_exact_run_returns_probabilities(self, trained):
        from polypus.qml import ExactResult

        run = trained.run(_X, exact=True)
        assert isinstance(run, ExactResult)
        assert len(run.probabilities) == len(_X)
        for row in run.probabilities:
            assert set(row) <= {"0", "1"}
            assert sum(row.values()) == pytest.approx(1.0)
        assert (run.backend, run.infrastructure) == ("polypus", "local")
        assert repr(run).startswith("ExactResult(rows=6,")

    def test_raw_decisions_are_the_expectations(self):
        from polypus.qml import Readout, Z

        trained = _model(readout=Readout([Z(0)])).bind([0.0, 0.0])
        values = trained.predict([[0.0], [math.pi / 2]], exact=True)
        assert all(type(value) is float for value in values)
        assert values == pytest.approx([1.0, 0.0], abs=1e-12)

    def test_argmax_decisions_are_class_indices(self):
        from polypus.qml import Ansatz, Encoder, Model, Readout, Z

        # Zero angles and CZ entanglers leave the basis state the encoder
        # prepared, so ⟨Z_k⟩ = −1 on every qubit set to 1 and the argmax picks
        # the qubit left at 0.
        model = Model(
            Encoder(2).basis_encoder(),
            Ansatz(2).hardware_efficient(1, rotations=["ry"], entangler="cz"),
            Readout([Z(0), Z(1)], decision="argmax"),
        )
        classes = model.bind([0.0] * model.num_params).predict(
            [[1.0, 0.0], [0.0, 1.0]], exact=True
        )
        assert classes == [1, 0]

    def test_explicit_aer_backend(self, trained):
        assert len(trained.run(_X, backend="aer", shots=32, seed=1).counts) == 6


class TestInferenceRejections:
    """Everything here is rejected before a backend exists."""

    def test_rows_must_fit_the_encoder(self, trained):
        with pytest.raises(ValueError):
            trained.predict([[0.1, 0.2]])

    def test_no_rows(self, trained):
        with pytest.raises(ValueError):
            trained.predict([])

    @pytest.mark.parametrize("method", ["predict", "run"])
    def test_exact_needs_the_local_native_simulator(self, trained, method):
        with pytest.raises(ValueError, match="exact=True needs"):
            getattr(trained, method)(_X, exact=True, backend="aer")

    def test_no_seed_on_qmio_hardware(self, trained):
        with pytest.raises(ValueError, match="seed is not supported"):
            trained.run(_X, infrastructure="qmio", seed=1)

    def test_execution_arguments_are_keyword_only(self, trained):
        with pytest.raises(TypeError):
            trained.predict(_X, 128)


# ─────────────────────────────────────────────────────────────────────────────
# TrainedModel.save and qml.load
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_round_trip_predicts_identically(self, trained, tmp_path):
        from polypus.qml import TrainedModel, load

        path = tmp_path / "model.json"
        trained.save(path)
        loaded = load(path)
        assert isinstance(loaded, TrainedModel)
        assert loaded.params == trained.params
        assert loaded.predict(_X, exact=True) == trained.predict(_X, exact=True)
        assert (
            loaded.run(_X, exact=True).probabilities
            == trained.run(_X, exact=True).probabilities
        )

    def test_floats_read_back_bit_for_bit(self, tmp_path):
        from polypus.qml import load

        params = [0.1 + 0.2, 1 / 3]
        _model().bind(params).save(str(tmp_path / "m.json"))
        assert load(str(tmp_path / "m.json")).params == params

    def test_the_file_is_versioned_json(self, trained, tmp_path):
        import json

        path = tmp_path / "model.json"
        trained.save(path)
        saved = json.loads(path.read_text())
        assert saved["format"] == "polypus.qml.TrainedModel"
        assert saved["version"] == 1
        assert saved["params"] == trained.params
        assert set(saved) == {"format", "version", "spec", "params"}

    def test_save_replaces_and_leaves_no_temporary_file(self, trained, tmp_path):
        from polypus.qml import load

        path = tmp_path / "model.json"
        _model().bind([0.0, 0.0]).save(path)
        trained.save(path)
        assert load(path).params == trained.params
        assert [p.name for p in tmp_path.iterdir()] == ["model.json"]

    def test_disk_failures_are_os_errors(self, trained, tmp_path):
        from polypus.qml import load

        with pytest.raises(FileNotFoundError, match="missing.json"):
            load(tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError):
            trained.save(tmp_path / "no_such_dir" / "model.json")
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        "edit",
        [
            lambda saved: "not json",
            lambda saved: json_dump({**saved, "format": "something.else"}),
            lambda saved: json_dump({**saved, "version": 2}),
            lambda saved: json_dump({**saved, "params": saved["params"][:-1]}),
            lambda saved: json_dump({k: v for k, v in saved.items() if k != "spec"}),
        ],
        ids=["not-json", "format", "version", "params", "no-spec"],
    )
    def test_a_corrupt_or_edited_file_is_a_value_error(self, trained, tmp_path, edit):
        import json

        from polypus.qml import load

        path = tmp_path / "model.json"
        trained.save(path)
        path.write_text(edit(json.loads(path.read_text())))
        with pytest.raises(ValueError, match="not a valid polypus.qml.TrainedModel"):
            load(path)


def json_dump(value):
    import json

    return json.dumps(value)


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
