"""
``polypus.qml.Model.fit`` (contract C-11): supervised and unsupervised
training, exact mode, minibatches, seeds, the convergence curve, and every
rejection, all before a backend exists.
"""

import math

import pytest
from qml_native_helpers import X, Y, make_model

# ─────────────────────────────────────────────────────────────────────────────
# Model.fit
# ─────────────────────────────────────────────────────────────────────────────


def _fit(model=None, x=X, y=Y, **kwargs):
    import polypus

    kwargs.setdefault("method", polypus.Adam(max_iters=5, patience=10_000))
    kwargs.setdefault("seed", 3)
    if y is not None:
        kwargs.setdefault("loss", "hinge")
    return (model or make_model()).fit(x, y, **kwargs)


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

        result = _fit(x=Dataset(X, Y), y=None, loss="hinge", exact=True)
        assert result.best_fitness == _fit(exact=True).best_fitness

    def test_unsupervised_maximises_the_readout(self):
        import polypus
        from polypus.qml import Observable, Readout, Z

        # Maximise -⟨Z⟩ = -cos(x + φ): close samples can all be rotated near |1>.
        model = make_model(readout=Readout([Observable([(-1.0, Z(0))])]))
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
            make_model().fit(X, Y)
        with pytest.raises(TypeError):
            make_model().fit(X, Y, polypus.Adam())

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
            _fit(x=Dataset(X), y=None, loss="hinge")

    def test_a_dataset_and_y_together(self):
        from polypus.qml import Dataset

        with pytest.raises(ValueError, match="carries its own labels"):
            _fit(x=Dataset(X, Y))

    def test_an_unknown_loss_names_the_valid_ones(self):
        with pytest.raises(ValueError, match='"expectation"'):
            _fit(loss="l2")

    @pytest.mark.parametrize("loss", ["squared_error", "hinge", "expectation"])
    def test_argmax_takes_only_the_categorical_loss(self, loss):
        from polypus.qml import Readout, Z

        model = make_model(2, readout=Readout([Z(0), Z(1)], decision="argmax"))
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

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"shots": 0}, "shots"),
            ({"n_qpus": 0}, "n_qpus"),
            ({"infrastructure": "cunqa", "nodes": 0}, "nodes"),
        ],
    )
    def test_the_shared_execution_guards_apply(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            _fit(**kwargs)

    def test_an_invalid_id_is_rejected(self):
        with pytest.raises(ValueError):
            _fit(id="no spaces")


# ─────────────────────────────────────────────────────────────────────────────
# Ported from the 71 branch: a two-qubit model on two separable clusters
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


def _two_qubit_model():
    """Angle encoder + hardware-efficient block (8 parameters), ⟨Z₀⟩ by sign."""
    from polypus.qml import Ansatz, Encoder, Model, Readout, Z

    return Model(
        Encoder(2).angle_encoder("ry"),
        Ansatz(2).hardware_efficient(1),
        Readout([Z(0)], decision="sign"),
    )


def _clusters_fit(model=None, x=_CLUSTERS, y=_CLUSTER_LABELS, **kwargs):
    import polypus

    kwargs.setdefault(
        "method", polypus.DE(generations=30, population_size=16, tolerance=1e-9)
    )
    kwargs.setdefault("loss", "hinge")
    kwargs.setdefault("seed", 7)
    return (model or _two_qubit_model()).fit(x, y, **kwargs)


def _assert_history(result, ends_on_best=True):
    """One best-so-far value per iteration run, never decreasing (C-5)."""
    history = result.fitness_history
    assert isinstance(history, list) and all(isinstance(f, float) for f in history)
    assert len(history) == result.iterations_run
    assert all(b >= a for a, b in zip(history, history[1:]))
    if ends_on_best:
        assert history[-1] == result.best_fitness


@pytest.mark.integration
class TestSeeds:
    def test_the_same_seed_replays_a_sampled_run(self):
        r1, r2 = _clusters_fit(shots=1024), _clusters_fit(shots=1024)
        assert r1.seed == r2.seed == 7
        assert r1.best_params == r2.best_params
        assert r1.best_fitness == r2.best_fitness
        assert r1.iterations_run == r2.iterations_run
        assert r1.fitness_history == r2.fitness_history

    def test_no_seed_differs(self):
        r1, r2 = _clusters_fit(seed=None), _clusters_fit(seed=None)
        assert r1.seed != r2.seed
        assert r1.best_params != r2.best_params

    def test_the_id_is_a_prefix(self):
        result = _clusters_fit(id="clusters", exact=True)
        assert result.id.startswith("clusters_") and result.id != "clusters"


@pytest.mark.integration
class TestNativeAndAerAgree:
    # The same optimizer seed but different shot RNGs (the native backend's and
    # Aer's), so the two DE trajectories diverge after the first generation. On
    # separable data both still reach near-zero hinge loss: the gap is ~0.01 at
    # 2048 shots, and 0.1 is a safe, non-flaky bound.
    FITNESS_TOL = 0.1

    def test_comparable_fitness(self):
        import polypus

        def run(backend):
            return _clusters_fit(
                method=polypus.DE(generations=100, population_size=20, tolerance=1e-6),
                shots=2048,
                backend=backend,
            )

        native, aer = run("polypus"), run("aer")
        assert abs(native.best_fitness - aer.best_fitness) < self.FITNESS_TOL, (
            native.best_fitness,
            aer.best_fitness,
        )


@pytest.mark.integration
class TestExactMode:
    def test_bit_identical_across_runs(self):
        r1, r2 = _clusters_fit(exact=True), _clusters_fit(exact=True)
        assert r1.best_params == r2.best_params
        assert r1.best_fitness == r2.best_fitness
        assert r1.iterations_run == r2.iterations_run
        assert r1.converged == r2.converged

    def test_shots_never_enter_the_fitness(self):
        low = _clusters_fit(exact=True, shots=32, seed=11)
        high = _clusters_fit(exact=True, shots=1_000_000, seed=11)
        assert low.best_params == high.best_params
        assert low.best_fitness == high.best_fitness


# Six samples: four separable ones plus a contradictory pair (identical features,
# opposite labels). The pair alone costs hinge loss exactly 2 for every θ
# (max(0, 1−e) + max(0, 1+e) = 2 for e = ⟨Z₀⟩ ∈ [−1, 1]), so the best full-set
# fitness is bounded at −2/6 ≈ −0.333. A minibatch without the pair looks
# separable (≈ 0) and one with it scores on a −2/3 scale; either leaking into
# `best_fitness` would land far from −0.333.
_HARD = [
    [0.30, 0.35],
    [0.40, 0.30],
    [2.80, 2.75],
    [2.90, 2.80],
    [1.55, 1.60],
    [1.55, 1.60],
]
_HARD_LABELS = [-1, -1, 1, 1, -1, 1]


@pytest.mark.integration
class TestMinibatches:
    @staticmethod
    def _de(batch_size, seed=7):
        import polypus

        return _clusters_fit(
            x=_HARD,
            y=_HARD_LABELS,
            method=polypus.DE(generations=120, population_size=24, tolerance=1e-12),
            exact=True,
            batch_size=batch_size,
            seed=seed,
        )

    def test_a_dataset_larger_than_the_batch_trains(self):
        result = self._de(3)
        assert len(result.best_params) == 8
        assert math.isfinite(result.best_fitness)

    def test_the_reported_fitness_is_the_full_set(self):
        full, mb = self._de(None), self._de(3)
        assert -0.5 < full.best_fitness < -0.30
        assert mb.best_fitness < -0.20
        assert abs(mb.best_fitness - full.best_fitness) < 0.15

    def test_reproducible_for_a_seed(self):
        r1, r2 = self._de(3), self._de(3)
        assert r1.best_params == r2.best_params
        assert r1.best_fitness == r2.best_fitness

    @pytest.mark.parametrize("seed", [11, 140])
    def test_a_cancelling_minibatch_cannot_stop_adam(self, seed):
        # On the 71 branch Adam stopped when the *minibatch* gradient was small,
        # and a batch of just the contradictory pair has a gradient of exactly
        # zero: seeds 11 and 140 stopped there, far from the optimum. Adam now
        # tracks its best on the full set and stops on that (C-5), so the same
        # seeds reach the optimum the full-batch run reaches.
        import polypus

        def run(batch_size):
            return _clusters_fit(
                x=_HARD,
                y=_HARD_LABELS,
                method=polypus.Adam(max_iters=60),
                exact=True,
                batch_size=batch_size,
                seed=seed,
            )

        full, mb = run(None), run(2)
        assert -0.40 < full.best_fitness < -0.30
        assert -0.40 < mb.best_fitness < -0.30
        _assert_history(mb)


@pytest.mark.integration
class TestEveryLayerTrains:
    """Each layer reaches training from Python and the separable clusters are
    learned (hinge loss near zero)."""

    def test_real_amplitudes(self):
        from polypus.qml import Ansatz, Encoder, Model, Readout, Z

        model = Model(
            Encoder(2).angle_encoder("ry"),
            Ansatz(2).real_amplitudes(2),
            Readout([Z(0)], decision="sign"),
        )
        result = _clusters_fit(model, exact=True)
        assert len(result.best_params) == 6
        assert result.best_fitness > -0.2

    def test_iqp_encoder(self):
        from polypus.qml import Ansatz, Encoder, Model, Readout, Z

        model = Model(
            Encoder(4).iqp_encoder(),
            Ansatz(4).hardware_efficient(1),
            Readout([Z(0)], decision="sign"),
        )
        result = _clusters_fit(model, x=[row * 2 for row in _CLUSTERS], exact=True)
        assert len(result.best_params) == 16
        assert math.isfinite(result.best_fitness)

    def test_qcnn(self):
        # conv → pool → conv over 4 features: 4 + 3 + 4 shared parameters.
        import polypus
        from polypus.qml import Ansatz, Encoder, Model, Readout, Z

        model = Model(
            Encoder(4).angle_encoder("ry"),
            Ansatz(4).conv("basic").pool("basic").conv("basic", "even_pairs"),
            Readout([Z(0)], decision="sign"),
        )
        result = _clusters_fit(
            model,
            x=[row * 2 for row in _CLUSTERS],
            method=polypus.DE(generations=40, population_size=16, tolerance=1e-9),
            exact=True,
        )
        assert len(result.best_params) == 11
        assert result.best_fitness > -0.2

    def test_qcnn_is_reproducible(self):
        from polypus.qml import Ansatz, Encoder, Model, Readout, Z

        def run():
            model = Model(
                Encoder(4).angle_encoder("ry"),
                Ansatz(4).conv("cartan").pool("basic", "odd_positions"),
                Readout([Z(0)], decision="sign"),
            )
            return _clusters_fit(
                model, x=[row * 2 for row in _CLUSTERS], exact=True, seed=13
            )

        a, b = run(), run()
        assert a.best_params == b.best_params
        assert a.best_fitness == b.best_fitness

    def test_weighted_observable(self):
        import polypus
        from polypus.qml import Ansatz, Encoder, Model, Observable, Readout, Z

        model = Model(
            Encoder(2).angle_encoder("ry"),
            Ansatz(2).hardware_efficient(0, rotations=["ry"]),
            Readout([Observable([(0.5, Z(0)), (0.5, Z(0) @ Z(1))])], decision="sign"),
        )
        result = _clusters_fit(
            model, method=polypus.DE(population_size=6, generations=4), exact=True
        )
        assert len(result.best_params) == 2
        assert math.isfinite(result.best_fitness)


@pytest.mark.integration
class TestFitnessHistory:
    def test_de(self):
        result = _clusters_fit(exact=True)
        _assert_history(result)
        assert len(result.fitness_history) > 1
        assert result.fitness_history[-1] > result.fitness_history[0]

    def test_adam_tracks_its_best_not_its_iterate(self):
        # Gradient ascent lets the iterate oscillate; the curve is the best so far.
        import polypus

        result = _clusters_fit(
            method=polypus.Adam(max_iters=15, learning_rate=0.1, patience=10_000),
            exact=True,
        )
        assert result.iterations_run == 15
        _assert_history(result)

    def test_minibatched_de_keeps_its_estimates(self):
        # DE's curve holds the minibatch estimates it steered by, while
        # `best_fitness` is re-scored on the full set: the shape still holds, but
        # the last entry need not equal `best_fitness` (C-5's one exception).
        result = _clusters_fit(exact=True, batch_size=2, seed=5)
        _assert_history(result, ends_on_best=False)

    def test_an_invalid_method_propagates_its_error(self):
        with pytest.raises(TypeError, match="method must be an instance"):
            _clusters_fit(method="not-an-optimizer", exact=True)
