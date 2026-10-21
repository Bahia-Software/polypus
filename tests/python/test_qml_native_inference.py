"""
Inference with native QML models (contract C-11): ``Model.bind``,
``TrainedModel.predict`` and ``run`` (``RunResult``/``ExactResult``), and
``TrainedModel.save``/``polypus.qml.load``.
"""

import math

import pytest
from qml_native_helpers import X, Y, make_model

# ─────────────────────────────────────────────────────────────────────────────
# Model.bind and TrainedModel
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trained():
    """A one-qubit classifier trained exactly: ⟨Z⟩ = cos(x + φ) with φ ≈ 0."""
    import polypus

    model = make_model()
    result = model.fit(
        X,
        Y,
        method=polypus.Adam(max_iters=60, learning_rate=0.1, patience=10_000),
        loss="hinge",
        exact=True,
        seed=3,
    )
    return model.bind(result.best_params)


class TestBind:
    def test_boundmodel(self):
        model = make_model()
        trained = model.bind([0.1, 0.2])
        assert trained.params == [0.1, 0.2]
        assert trained.model.num_params == model.num_params == 2
        assert repr(trained) == "TrainedModel(num_qubits=1, num_params=2)"

    def test_the_model_is_unchanged(self):
        model = make_model()
        model.bind([0.1, 0.2])
        model.bind([0.3, 0.4])
        assert model.num_params == 2

    def test_the_parameter_count_must_match(self):
        with pytest.raises(ValueError, match="params has 3 values"):
            make_model().bind([0.1, 0.2, 0.3])

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_parameters_must_be_finite(self, value):
        with pytest.raises(ValueError, match=r"params\[1\]"):
            make_model().bind([0.1, value])


@pytest.mark.integration
class TestInference:
    def test_predict_classifies_the_training_set(self, trained):
        labels = trained.predict(X, exact=True)
        assert labels == Y
        assert all(type(label) is int for label in labels)

    def test_sampled_predict_agrees(self, trained):
        assert trained.predict(X, shots=2048, seed=5) == Y

    def test_a_dataset_is_accepted_and_its_labels_unused(self, trained):
        from polypus.qml import Dataset

        assert trained.predict(Dataset(X), exact=True) == Y
        assert trained.predict(Dataset(X, [0] * 6), exact=True) == Y

    def test_run_returns_counts_per_row(self, trained):
        import polypus

        run = trained.run(X, shots=128, seed=5)
        assert isinstance(run, polypus.RunResult)
        assert len(run.counts) == len(X)
        assert all(sum(counts.values()) == 128 for counts in run.counts)
        assert run.merged_counts is None
        assert (run.seed, run.backend, run.infrastructure) == (5, "polypus", "local")

    def test_the_seed_replays_the_counts(self, trained):
        assert (
            trained.run(X, shots=64, seed=9).counts
            == trained.run(X, shots=64, seed=9).counts
        )

    def test_exact_run_returns_probabilities(self, trained):
        from polypus.qml import ExactResult

        run = trained.run(X, exact=True)
        assert isinstance(run, ExactResult)
        assert len(run.probabilities) == len(X)
        for row in run.probabilities:
            assert set(row) <= {"0", "1"}
            assert sum(row.values()) == pytest.approx(1.0)
        assert (run.backend, run.infrastructure) == ("polypus", "local")
        assert repr(run).startswith("ExactResult(rows=6,")

    def test_raw_decisions_are_the_expectations(self):
        from polypus.qml import Readout, Z

        trained = make_model(readout=Readout([Z(0)])).bind([0.0, 0.0])
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
        assert len(trained.run(X, backend="aer", shots=32, seed=1).counts) == 6


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
            getattr(trained, method)(X, exact=True, backend="aer")

    def test_no_seed_on_qmio_hardware(self, trained):
        with pytest.raises(ValueError, match="seed is not supported"):
            trained.run(X, infrastructure="qmio", seed=1)

    def test_execution_arguments_are_keyword_only(self, trained):
        with pytest.raises(TypeError):
            trained.predict(X, 128)


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
        assert loaded.predict(X, exact=True) == trained.predict(X, exact=True)
        assert (
            loaded.run(X, exact=True).probabilities
            == trained.run(X, exact=True).probabilities
        )

    def test_floats_read_back_bit_for_bit(self, tmp_path):
        from polypus.qml import load

        params = [0.1 + 0.2, 1 / 3]
        make_model().bind(params).save(str(tmp_path / "m.json"))
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
        make_model().bind([0.0, 0.0]).save(path)
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


# ─────────────────────────────────────────────────────────────────────────────
# Ported from the 71 branch: inference and saving of a trained classifier
# ─────────────────────────────────────────────────────────────────────────────

_CLUSTERS = [
    [0.3, 0.35],
    [0.4, 0.3],
    [0.35, 0.4],
    [2.8, 2.75],
    [2.9, 2.8],
    [2.75, 2.9],
]
_CLUSTER_LABELS = [-1, -1, -1, 1, 1, 1]
# New samples the model never trained on: one near each cluster.
_NEW_SAMPLES = [[0.32, 0.33], [2.85, 2.82]]


def _two_qubit_model():
    from polypus.qml import Ansatz, Encoder, Model, Readout, Z

    return Model(
        Encoder(2).angle_encoder("ry"),
        Ansatz(2).hardware_efficient(1),
        Readout([Z(0)], decision="sign"),
    )


@pytest.fixture(scope="module")
def clusters():
    """The two-qubit model trained exactly on the clusters, and its result."""
    import polypus

    model = _two_qubit_model()
    result = model.fit(
        _CLUSTERS,
        _CLUSTER_LABELS,
        method=polypus.DE(generations=30, population_size=16, tolerance=1e-9),
        loss="hinge",
        exact=True,
        seed=7,
    )
    return model, result


@pytest.mark.integration
class TestClassifierInference:
    def test_new_samples_get_their_cluster(self, clusters):
        model, result = clusters
        labels = model.bind(result.best_params).predict(_NEW_SAMPLES, exact=True)
        assert labels == [-1, 1]

    def test_sampled_labels_are_plus_or_minus_one(self, clusters):
        model, result = clusters
        for backend in ("polypus", "aer"):
            labels = model.bind(result.best_params).predict(
                _NEW_SAMPLES, shots=2048, seed=7, backend=backend
            )
            assert len(labels) == 2
            assert all(type(label) is int and label in (-1, 1) for label in labels)

    def test_exact_predictions_are_deterministic(self, clusters):
        model, result = clusters
        trained = model.bind(result.best_params)
        assert trained.predict(_NEW_SAMPLES, exact=True) == trained.predict(
            _NEW_SAMPLES, exact=True
        )

    def test_two_bindings_of_one_model_are_independent(self, clusters):
        # The model holds no trained state: binding it again, to anything,
        # changes nothing about an earlier binding.
        model, result = clusters
        trained = model.bind(result.best_params)
        before = trained.predict(_NEW_SAMPLES, exact=True)
        model.bind([0.0] * model.num_params)
        assert trained.predict(_NEW_SAMPLES, exact=True) == before
        assert trained.params == result.best_params

    def test_fitting_again_gives_the_same_result(self, clusters):
        import polypus

        model, result = clusters
        again = model.fit(
            _CLUSTERS,
            _CLUSTER_LABELS,
            method=polypus.DE(generations=30, population_size=16, tolerance=1e-9),
            loss="hinge",
            exact=True,
            seed=7,
        )
        assert again.best_params == result.best_params
        assert again.best_fitness == result.best_fitness


@pytest.mark.integration
class TestSavedClassifier:
    def test_parameters_survive_bit_for_bit(self, clusters, tmp_path):
        from polypus.qml import load

        model, result = clusters
        path = tmp_path / "model.json"
        model.bind(result.best_params).save(path)
        loaded = load(path)
        assert loaded.params == result.best_params
        assert loaded.predict(_NEW_SAMPLES, exact=True) == [-1, 1]

    def test_a_model_without_layers_is_rejected(self, tmp_path):
        import json

        from polypus.qml import load

        path = tmp_path / "model.json"
        _two_qubit_model().bind([0.0] * 8).save(path)
        saved = json.loads(path.read_text())
        saved["spec"]["layers"] = []
        path.write_text(json.dumps(saved))
        with pytest.raises(ValueError, match="no layers"):
            load(path)

    @pytest.mark.parametrize(
        ("corrupt", "message"),
        [
            # The readout is `sign` over one observable: empty the list.
            (
                lambda r: r.update(observables=[]),
                "incompatible with 0 observable",
            ),
            # `argmax` needs two observables; the saved readout has one.
            (lambda r: r.update(decision="Argmax"), "incompatible with 1 observable"),
            # A repeated position never survives `PauliString::new`, but
            # deserialisation bypasses it: ⟨Z₀Z₀⟩ would silently read +1.
            (
                lambda r: r["observables"][0].update(
                    terms=[[1.0, [[0, "Z"], [0, "Z"]]]]
                ),
                "more than one factor on position 0",
            ),
        ],
        ids=["no-observables", "argmax", "repeated-position"],
    )
    def test_a_tampered_readout_is_rejected(self, tmp_path, corrupt, message):
        # Revalidation covers the readout, not just the layers: such a file used
        # to load and fail only at the first prediction.
        import json

        from polypus.qml import load

        path = tmp_path / "model.json"
        _two_qubit_model().bind([0.0] * 8).save(path)
        saved = json.loads(path.read_text())
        corrupt(saved["spec"]["readout"])
        path.write_text(json.dumps(saved))
        with pytest.raises(ValueError, match=message):
            load(path)
