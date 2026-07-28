"""
Native ``polypus.qml.train`` — public-API end-to-end tests (phase 4, contracts
C-7 and C-8).

These exercise the pure-Rust QML path added in phase 4: ``polypus.qml.train``
now dispatches on its first argument, so a native ``polypus.qml.Model`` +
``polypus.qml.Dataset`` train through Rust (``NativeQmlOracle``) on any simulated
backend, while a Qiskit feature map keeps the original Qiskit/Aer path.

Three groups:

1. **Reproducibility (C-7).** On ``backend="polypus"`` a fixed seed reproduces
   the whole outcome byte-for-byte; an omitted seed varies it. The native
   backend is fully seeded by Polypus, so no mocking is needed.
2. **Argument validation.** Kwargs that belong to the Qiskit path
   (``x_train``/``expectation_function``) are rejected with a native Model +
   Dataset; a ``dimensions`` that disagrees with the compiled model is rejected;
   and every strict string parser (axis / decision / loss) rejects an
   unrecognised value with a clear ``ValueError``.
3. **Native vs Aer.** The same model + dataset + hyperparameters train to a
   comparable fitness on ``backend="polypus"`` and ``backend="aer"`` — the two
   differ only by shot noise, so their best fitnesses stay close (see the
   tolerance note on the test).
"""

import pytest


def _dataset():
    """Two well-separated clusters in ``[0, π]`` — trivially separable, so a
    short DE run reaches near-optimal hinge loss on both backends."""
    import polypus

    x = [
        [0.3, 0.35],
        [0.4, 0.3],
        [0.35, 0.4],
        [2.8, 2.75],
        [2.9, 2.8],
        [2.75, 2.9],
    ]
    y = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    return polypus.qml.Dataset(x, y)


def _model():
    """A fresh 2-qubit angle-encoder + hardware-efficient model reading ⟨Z₀⟩
    with a sign decision (8 trainable parameters). Built fresh each call because
    the builder is consumed on ``train``'s Rust side."""
    import polypus

    return (
        polypus.qml.Model(2)
        .angle_encoder(axis="ry")
        .hardware_efficient(reps=1)
        .readout(observables=[[("z", 0)]], decision="sign")
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reproducibility on the native backend (C-7)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vqc
class TestNativeQmlTrainSeed:
    @staticmethod
    def _train(seed):
        import polypus

        return polypus.qml.train(
            _model(),
            _dataset(),
            method=polypus.DE(generations=30, population_size=16, tolerance=1e-9),
            loss="hinge",
            shots=1024,
            infrastructure="local",
            backend="polypus",
            id="qml_native_seed",
            seed=seed,
        )

    def test_same_seed_reproduces_outcome(self):
        r1 = self._train(seed=7)
        r2 = self._train(seed=7)
        assert r1.seed == 7 and r2.seed == 7
        # The native path is byte-reproducible: the whole outcome matches.
        assert r1.best_params == r2.best_params
        assert r1.best_fitness == r2.best_fitness
        assert r1.iterations_run == r2.iterations_run
        assert r1.converged == r2.converged

    def test_no_seed_differs_across_calls(self):
        r1 = self._train(seed=None)
        r2 = self._train(seed=None)
        # Each unseeded run draws a fresh entropy seed → different trajectory.
        assert r1.seed != r2.seed
        assert r1.best_params != r2.best_params


# ─────────────────────────────────────────────────────────────────────────────
# 2. Argument validation (native path + strict parsers)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vqc
class TestNativeQmlTrainValidation:
    def test_x_train_rejected_with_model_and_dataset(self):
        import polypus

        with pytest.raises(ValueError, match="x_train is not accepted"):
            polypus.qml.train(
                _model(),
                _dataset(),
                x_train=[[0.1, 0.2]],
                method=polypus.DE(generations=2, population_size=4),
                loss="hinge",
                backend="polypus",
                id="qml_native_xtrain",
                seed=7,
            )

    def test_expectation_function_rejected_with_model_and_dataset(self):
        import polypus

        with pytest.raises(ValueError, match="expectation_function is not accepted"):
            polypus.qml.train(
                _model(),
                _dataset(),
                method=polypus.DE(generations=2, population_size=4),
                expectation_function=lambda b: 0.0,
                loss="hinge",
                backend="polypus",
                id="qml_native_ef",
                seed=7,
            )

    def test_loss_required_on_native_path(self):
        import polypus

        with pytest.raises(ValueError, match="loss is required"):
            polypus.qml.train(
                _model(),
                _dataset(),
                method=polypus.DE(generations=2, population_size=4),
                backend="polypus",
                id="qml_native_noloss",
                seed=7,
            )

    def test_dimensions_mismatch_rejected(self):
        import polypus

        # The compiled model has 8 trainable parameters; 9 must be rejected.
        with pytest.raises(ValueError, match="does not match"):
            polypus.qml.train(
                _model(),
                _dataset(),
                method=polypus.DE(generations=2, population_size=4),
                dimensions=9,
                loss="hinge",
                backend="polypus",
                id="qml_native_dims",
                seed=7,
            )

    def test_unknown_axis_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="unknown rotation axis"):
            polypus.qml.Model(2).angle_encoder(axis="xy")

    def test_unknown_decision_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="unknown decision"):
            polypus.qml.Model(2).readout(observables=[[("z", 0)]], decision="bogus")

    def test_unknown_loss_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="unknown loss"):
            polypus.qml.train(
                _model(),
                _dataset(),
                method=polypus.DE(generations=2, population_size=4),
                loss="bogus",
                backend="polypus",
                id="qml_native_badloss",
                seed=7,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Native vs Aer equivalence
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vqc
class TestNativeVsAerQmlTrain:
    # Tolerance rationale (mirroring TestNativeVsAerEquivalence in
    # test_backend_selection.py): with the same optimizer seed but different shot
    # RNGs (the native backend's in-process RNG vs Aer's seed_simulator), the two
    # DE trajectories diverge after the first generation, so we cannot expect
    # identical parameters. But on well-separated data both converge to near-zero
    # hinge loss, so their best fitnesses stay within shot-noise range of each
    # other. Empirically the gap is ~0.01 for shots=2048; 0.1 is a safe,
    # non-flaky bound (~10×).
    FITNESS_TOL = 0.1

    @staticmethod
    def _train(backend):
        import polypus

        return polypus.qml.train(
            _model(),
            _dataset(),
            method=polypus.DE(generations=100, population_size=20, tolerance=1e-6),
            loss="hinge",
            shots=2048,
            infrastructure="local",
            backend=backend,
            id="qml_native_vs_aer",
            seed=7,
        )

    def test_native_and_aer_reach_comparable_fitness(self):
        native = self._train("polypus")
        aer = self._train("aer")
        assert abs(native.best_fitness - aer.best_fitness) < self.FITNESS_TOL, (
            f"native={native.best_fitness:.4f} aer={aer.best_fitness:.4f} "
            f"differ by more than {self.FITNESS_TOL}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Exact mode (exact=True) — design doc §17
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vqc
class TestNativeQmlTrainExact:
    """The shot-free exact path: no sampling, so two runs with the same
    configuration are byte-identical, and the result does not depend on the
    seed. Only the native local + native-backend combination is accepted."""

    @staticmethod
    def _train(seed, backend="polypus", shots=1024):
        import polypus

        return polypus.qml.train(
            _model(),
            _dataset(),
            method=polypus.DE(generations=30, population_size=16, tolerance=1e-9),
            loss="hinge",
            shots=shots,
            infrastructure="local",
            backend=backend,
            id="qml_native_exact",
            seed=seed,
            exact=True,
        )

    def test_exact_trains_and_is_bit_identical_across_runs(self):
        r1 = self._train(seed=7)
        r2 = self._train(seed=7)
        # Exact mode draws no shot noise → the whole outcome matches exactly.
        assert r1.best_params == r2.best_params
        assert r1.best_fitness == r2.best_fitness
        assert r1.iterations_run == r2.iterations_run
        assert r1.converged == r2.converged

    def test_exact_is_independent_of_seed(self):
        # The optimizer seed still drives DE's search, but the *fitness* is
        # exact, so the same seed obviously reproduces; here we assert the
        # stronger property that shots never enter the fitness: two shot counts
        # that would give very different sampling noise yield the identical
        # best fitness for the same optimizer seed.
        r_low = self._train(seed=11, shots=32)
        r_high = self._train(seed=11, shots=1_000_000)
        assert r_low.best_params == r_high.best_params
        assert r_low.best_fitness == r_high.best_fitness

    def test_exact_rejected_on_aer_backend(self):
        import polypus

        with pytest.raises(ValueError, match="exact"):
            polypus.qml.train(
                _model(),
                _dataset(),
                method=polypus.DE(generations=2, population_size=4),
                loss="hinge",
                infrastructure="local",
                backend="aer",
                id="qml_exact_aer",
                seed=7,
                exact=True,
            )

    def test_exact_rejected_on_qiskit_path(self):
        import numpy as np
        import polypus
        from qiskit.circuit.library import real_amplitudes, zz_feature_map

        feature_map = zz_feature_map(feature_dimension=2, reps=1)
        ansatz = real_amplitudes(num_qubits=2, reps=1)
        with pytest.raises(ValueError, match="exact"):
            polypus.qml.train(
                feature_map,
                ansatz,
                np.zeros((2, 2)),
                polypus.DE(generations=2, population_size=4),
                dimensions=len(ansatz.parameters),
                expectation_function=lambda b: sum(int(c) for c in b) / len(b),
                infrastructure="local",
                backend="aer",
                id="qml_exact_qiskit",
                seed=7,
                exact=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Deterministic minibatching (batch_size=N) — design doc §17
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vqc
class TestNativeQmlTrainMinibatch:
    """`batch_size=N` scores a deterministic N-sample subset per optimizer
    evaluation, but the reported `best_fitness` is recomputed **once** against
    the whole dataset when the run ends — so it stays a genuine full-dataset
    value, never the last iteration's minibatch estimate (design doc §17)."""

    @staticmethod
    def _hard_dataset():
        """Six samples: four trivially separable ones plus a **contradictory**
        pair (identical features, opposite labels). The pair alone contributes
        hinge loss exactly 2 for *every* θ (``max(0,1−e)+max(0,1+e)=2`` for
        ``e=⟨Z₀⟩∈[−1,1]``), so the best achievable **full-dataset** fitness is
        bounded at ``−2/6 ≈ −0.333`` — no θ can beat it. A minibatch that omits
        the pair looks separable (fitness ≈ 0); a 3-sample minibatch that
        includes it scores on a ``−2/3`` scale. Either would leak a value far
        from −0.333 into ``best_fitness`` if the full-dataset recompute were
        missing, which is exactly what these tests catch."""
        import polypus

        x = [
            [0.30, 0.35],
            [0.40, 0.30],  # class −1, easy
            [2.80, 2.75],
            [2.90, 2.80],  # class +1, easy
            [1.55, 1.60],
            [1.55, 1.60],  # contradictory pair (identical features)
        ]
        y = [-1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
        return polypus.qml.Dataset(x, y)

    @classmethod
    def _train(cls, *, batch_size, seed, generations=120, population=24):
        # Exact mode: no shot noise, so the run is fully deterministic and the
        # full-dataset recompute is an exact number — the cleanest setting to
        # assert the reported fitness against the analytic −1/3 bound.
        import polypus

        return polypus.qml.train(
            _model(),
            cls._hard_dataset(),
            method=polypus.DE(
                generations=generations, population_size=population, tolerance=1e-12
            ),
            loss="hinge",
            infrastructure="local",
            backend="polypus",
            id="qml_native_minibatch",
            seed=seed,
            exact=True,
            batch_size=batch_size,
        )

    def test_trains_on_dataset_larger_than_batch(self):
        # 6 samples, batch_size 3 → a genuine subset each evaluation.
        import math

        r = self._train(batch_size=3, seed=7)
        assert len(r.best_params) == 8
        assert math.isfinite(r.best_fitness)

    def test_reported_fitness_is_full_dataset_not_minibatch(self):
        # Full-batch reference (the trusted non-minibatch path) and a minibatch
        # run over the *same* problem.
        full = self._train(batch_size=None, seed=7)
        mb = self._train(batch_size=3, seed=7)

        # The full-batch optimum sits just below the analytic −1/3 bound.
        assert -0.5 < full.best_fitness < -0.30

        # The minibatch run's reported fitness is a real full-dataset value:
        #   • clearly on the full scale, never a rosy ≈ 0 (a separable subset)
        #     nor a ≈ −2/3 three-sample value; and
        #   • essentially the same optimum the full-batch run reached.
        # Both would be violated if best_fitness were the last minibatch value
        # instead of the full-dataset recompute.
        assert mb.best_fitness < -0.20
        assert abs(mb.best_fitness - full.best_fitness) < 0.15

    def test_reproducible_for_fixed_seed(self):
        # Exact mode + fixed seed/batch_size → byte-identical outcome, including
        # the deterministic minibatch selection (C-7).
        r1 = self._train(batch_size=3, seed=7)
        r2 = self._train(batch_size=3, seed=7)
        assert r1.best_params == r2.best_params
        assert r1.best_fitness == r2.best_fitness

    def test_batch_size_zero_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="batch_size"):
            polypus.qml.train(
                _model(),
                self._hard_dataset(),
                method=polypus.DE(generations=2, population_size=4),
                loss="hinge",
                infrastructure="local",
                backend="polypus",
                id="qml_mb_zero",
                seed=7,
                batch_size=0,
            )

    def test_batch_size_equal_to_dataset_rejected(self):
        # A batch as large as the dataset is just the non-minibatch path.
        import polypus

        with pytest.raises(ValueError, match="batch_size"):
            polypus.qml.train(
                _model(),
                self._hard_dataset(),  # 6 samples
                method=polypus.DE(generations=2, population_size=4),
                loss="hinge",
                infrastructure="local",
                backend="polypus",
                id="qml_mb_full",
                seed=7,
                batch_size=6,
            )

    def test_batch_size_rejected_on_qiskit_path(self):
        import numpy as np
        import polypus
        from qiskit.circuit.library import real_amplitudes, zz_feature_map

        feature_map = zz_feature_map(feature_dimension=2, reps=1)
        ansatz = real_amplitudes(num_qubits=2, reps=1)
        with pytest.raises(ValueError, match="batch_size"):
            polypus.qml.train(
                feature_map,
                ansatz,
                np.zeros((2, 2)),
                polypus.DE(generations=2, population_size=4),
                dimensions=len(ansatz.parameters),
                expectation_function=lambda b: sum(int(c) for c in b) / len(b),
                infrastructure="local",
                backend="aer",
                id="qml_mb_qiskit",
                seed=7,
                batch_size=1,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model serialization: TrainedModel save / load (design doc §17)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vqc
class TestTrainedModelSaveLoad:
    def test_save_load_round_trips_theta(self, tmp_path):
        """Train a small model, wrap it with its best parameters, save it,
        reload it, and confirm θ survives byte-for-byte."""
        import polypus

        result = polypus.qml.train(
            _model(),
            _dataset(),
            method=polypus.DE(generations=20, population_size=12, tolerance=1e-9),
            loss="hinge",
            shots=1024,
            infrastructure="local",
            backend="polypus",
            id="qml_trained_save",
            seed=7,
        )

        # A fresh builder + the same dataset the model was compiled against.
        trained = polypus.qml.TrainedModel(_model(), _dataset(), result.best_params)
        assert trained.theta == result.best_params

        path = str(tmp_path / "model.json")
        trained.save(path)
        loaded = polypus.qml.TrainedModel.load(path)

        # `float_roundtrip` guarantees exact equality, not just approximate.
        assert loaded.theta == result.best_params

    def test_predict_from_counts_applies_decision(self):
        # ⟨Z₀⟩ over "00" (width 2) = +1 → sign decision → +1; "01" → −1.
        import polypus

        trained = polypus.qml.TrainedModel(_model(), _dataset(), [0.0] * 8)
        assert trained.predict_from_counts({"00": 10}) == 1.0
        assert trained.predict_from_counts({"01": 10}) == -1.0

    def test_load_rejects_corrupt_file(self, tmp_path):
        # A file whose spec no longer compiles (empty layers) must fail loading
        # with a ValueError — recompilation revalidates, never a silent accept.
        # The "model" wrapper matches TrainedModel's real wire shape
        # ({"model": {"spec", "num_features"}, "theta"}); a bare {"spec": ...}
        # at the top level fails for the wrong reason (a missing "model" field,
        # not the recompiled-and-rejected spec this test means to exercise) —
        # confirmed by checking the exact error message below.
        import polypus

        path = tmp_path / "corrupt.json"
        path.write_text(
            '{"model": {"spec": {"num_qubits": 2, "layers": [], "readout": null}, '
            '"num_features": 2}, "theta": []}'
        )
        with pytest.raises(ValueError, match="no layers"):
            polypus.qml.TrainedModel.load(str(path))


# ─────────────────────────────────────────────────────────────────────────────
# 6. End-to-end inference: TrainedModel.predict(X, ...) (design doc §17)
# ─────────────────────────────────────────────────────────────────────────────


def _trained_model():
    """Train the small angle-encoder model to separation and wrap it as a
    `TrainedModel` ready for inference. Trained on the exact path so the θ is
    deterministic run to run."""
    import polypus

    result = polypus.qml.train(
        _model(),
        _dataset(),
        method=polypus.DE(generations=30, population_size=16, tolerance=1e-9),
        loss="hinge",
        infrastructure="local",
        backend="polypus",
        id="qml_predict_train",
        seed=7,
        exact=True,
    )
    return polypus.qml.TrainedModel(_model(), _dataset(), result.best_params)


# New samples the model never trained on: one near each cluster of `_dataset()`.
_NEW_SAMPLES = [[0.32, 0.33], [2.85, 2.82]]


@pytest.mark.integration
@pytest.mark.vqc
class TestTrainedModelPredict:
    """`TrainedModel.predict(X, ...)` — bind new samples to θ, run them on a
    backend, and apply the readout decision, all in one call."""

    def test_predict_native_backend_returns_finite_sign_labels(self):
        # A `Sign` readout: every prediction is exactly ±1 and finite.
        trained = _trained_model()
        preds = trained.predict(
            _NEW_SAMPLES,
            shots=2048,
            infrastructure="local",
            backend="polypus",
            id="qml_predict_native",
            seed=7,
        )
        assert isinstance(preds, list)
        assert len(preds) == len(_NEW_SAMPLES)
        assert all(isinstance(p, float) and p in (-1.0, 1.0) for p in preds)

    def test_predict_exact_is_deterministic(self):
        # Exact mode draws no shot noise → two identical calls are byte-identical.
        trained = _trained_model()
        kwargs = dict(
            infrastructure="local",
            backend="polypus",
            id="qml_predict_exact",
            exact=True,
        )
        a = trained.predict(_NEW_SAMPLES, **kwargs)
        b = trained.predict(_NEW_SAMPLES, **kwargs)
        assert a == b

    def test_predict_exact_rejected_on_aer_backend(self):
        # `exact=True` reuses the same guard `qml.train` applies: aer is rejected.
        trained = _trained_model()
        with pytest.raises(ValueError, match="exact"):
            trained.predict(
                _NEW_SAMPLES,
                infrastructure="local",
                backend="aer",
                id="qml_predict_exact_aer",
                exact=True,
            )

    def test_predict_wrong_feature_count_raises_value_error(self):
        # A sample with 3 features where the model compiled for 2: `bind`'s
        # `FeatureCountMismatch` surfaces as a clean ValueError, never a panic.
        trained = _trained_model()
        with pytest.raises(ValueError):
            trained.predict(
                [[0.1, 0.2, 0.3]],
                infrastructure="local",
                backend="polypus",
                id="qml_predict_bad_features",
                seed=7,
            )

    def test_predict_aer_backend_with_shots_runs(self):
        # The non-exact shot path on aer must also work end to end (not required
        # to be bit-identical — just finite ±1 labels in the right shape).
        trained = _trained_model()
        preds = trained.predict(
            _NEW_SAMPLES,
            shots=2048,
            infrastructure="local",
            backend="aer",
            id="qml_predict_aer",
            seed=7,
        )
        assert len(preds) == len(_NEW_SAMPLES)
        assert all(isinstance(p, float) and p in (-1.0, 1.0) for p in preds)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Builder coverage: every layer and every ansatz knob reachable from Python
# ─────────────────────────────────────────────────────────────────────────────
#
# These assert *semantics*, not just "the call does not raise", and they do it
# without running an optimizer, using two properties of the exact inference path:
#
#   • `TrainedModel.predict(..., exact=True)` on a `decision="raw"` readout is a
#     deterministic function of the circuit — the exact ⟨Z₀⟩ of the bound state,
#     no shot noise. Two spellings that build the *same* circuit agree bit for
#     bit; two that build different circuits disagree. (`Raw` rather than `Sign`
#     precisely because a ±1 sign could coincide for two different circuits.)
#   • `bind` validates θ's length, so `predict` raises unless `len(theta)` equals
#     the compiled model's parameter count — an exact, optimizer-free probe of
#     the θ count each ansatz configuration reserves.


def _raw_model(num_qubits, ansatz):
    """An angle-encoder model whose variational block is appended by `ansatz`,
    reading ⟨Z₀⟩ out **unchanged** (`decision="raw"`)."""
    import polypus

    model = polypus.qml.Model(num_qubits).angle_encoder(axis="ry")
    model = ansatz(model)
    return model.readout(observables=[[("z", 0)]], decision="raw")


def _exact_expectation(model, theta, sample=(0.7, 1.1), dataset=None):
    """The exact ⟨Z₀⟩ of `model` bound to `theta` on one sample. Deterministic
    (no sampling), and raises `ValueError` if `theta` has the wrong length."""
    import polypus

    trained = polypus.qml.TrainedModel(model, dataset or _dataset(), list(theta))
    return trained.predict(
        [list(sample)],
        infrastructure="local",
        backend="polypus",
        id="qml_builder_raw",
        exact=True,
    )[0]


def _num_params(model, dataset=None, limit=16):
    """The compiled model's θ count, found by the only length `predict` accepts.

    `bind` rejects every other length with a `ValueError`, so the accepted one is
    unique and this is an exact measurement, not an estimate. `limit` bounds the
    search — every configuration tested here stays well under it."""
    accepted = [
        n for n in range(limit + 1) if _accepts_theta_length(model, n, dataset=dataset)
    ]
    assert len(accepted) == 1, f"expected exactly one accepted θ length, got {accepted}"
    return accepted[0]


def _accepts_theta_length(model, n, dataset=None):
    try:
        _exact_expectation(model, [0.1] * n, dataset=dataset)
        return True
    except ValueError:
        return False


@pytest.mark.integration
@pytest.mark.vqc
class TestHardwareEfficientConfiguration:
    """`hardware_efficient` exposes all five fields of the Rust struct; its
    defaults are exactly the `TwoLocal` defaults it had before the kwargs."""

    # 3 qubits, reps=1, default [ry, rz] → 2 axes × 3 qubits × (1 + 1) blocks.
    THETA_3Q = [0.1 * (i + 1) for i in range(12)]

    def test_defaults_are_unchanged_by_the_new_kwargs(self):
        # Passing every default explicitly must build the *same* circuit as
        # passing none — the guarantee that no existing caller changed behaviour.
        implicit = _raw_model(3, lambda m: m.hardware_efficient(reps=1))
        explicit = _raw_model(
            3,
            lambda m: m.hardware_efficient(
                reps=1,
                rotations=["ry", "rz"],
                entangler="cx",
                entanglement="linear",
                final_rotation_layer=True,
            ),
        )
        a = _exact_expectation(implicit, self.THETA_3Q)
        b = _exact_expectation(explicit, self.THETA_3Q)
        assert a == b

    def test_default_theta_count_is_axes_times_qubits_times_blocks(self):
        assert _num_params(_raw_model(3, lambda m: m.hardware_efficient(reps=1))) == 12
        # The 2-qubit model the rest of this file trains: 2 × 2 × 2 = 8.
        assert _num_params(_raw_model(2, lambda m: m.hardware_efficient(reps=1))) == 8

    def test_rotations_list_sets_the_theta_count(self):
        # One axis instead of two halves the count; three axes multiply it by 3/2.
        one = _raw_model(2, lambda m: m.hardware_efficient(reps=1, rotations=["ry"]))
        three = _raw_model(
            2, lambda m: m.hardware_efficient(reps=1, rotations=["rx", "ry", "rz"])
        )
        assert _num_params(one) == 4
        assert _num_params(three) == 12

    def test_final_rotation_layer_false_drops_one_rotation_block(self):
        # blocks = reps + final_rotation_layer → 2 × 2 × 1 = 4 instead of 8.
        model = _raw_model(
            2, lambda m: m.hardware_efficient(reps=1, final_rotation_layer=False)
        )
        assert _num_params(model) == 4

    def test_entanglement_patterns_build_different_circuits(self):
        # On 3 qubits the three patterns select different pair sets — linear
        # {(0,1),(1,2)}, circular adds (2,0), full has (0,2) instead — so the same
        # θ yields three different exact expectations. The θ count is identical
        # for all three (entanglers consume no θ), which `_num_params` confirms.
        values = {}
        for pattern in ("linear", "circular", "full"):
            model = _raw_model(
                3, lambda m, p=pattern: m.hardware_efficient(reps=1, entanglement=p)
            )
            assert _num_params(model) == 12
            values[pattern] = _exact_expectation(model, self.THETA_3Q)
        assert len(set(values.values())) == 3, (
            f"entanglement patterns collapsed to the same circuit: {values}"
        )

    def test_entangler_cz_builds_a_different_circuit_than_cx(self):
        cx = _raw_model(3, lambda m: m.hardware_efficient(reps=1, entangler="cx"))
        cz = _raw_model(3, lambda m: m.hardware_efficient(reps=1, entangler="cz"))
        assert _exact_expectation(cx, self.THETA_3Q) != _exact_expectation(
            cz, self.THETA_3Q
        )

    def test_real_amplitudes_is_the_single_ry_preset(self):
        # The preset must equal the explicit spelling of its four fixed fields,
        # bit for bit, and differ from the [ry, rz] default (which needs 8 θ).
        preset = _raw_model(2, lambda m: m.real_amplitudes(reps=1))
        explicit = _raw_model(
            2,
            lambda m: m.hardware_efficient(
                reps=1,
                rotations=["ry"],
                entangler="cx",
                entanglement="linear",
                final_rotation_layer=True,
            ),
        )
        theta = [0.3, 0.7, 1.1, 1.5]
        assert _num_params(preset) == 4
        assert _exact_expectation(preset, theta) == _exact_expectation(explicit, theta)

    def test_real_amplitudes_trains_end_to_end(self):
        import math

        import polypus

        model = (
            polypus.qml.Model(2)
            .angle_encoder(axis="ry")
            .real_amplitudes(reps=2)
            .readout(observables=[[("z", 0)]], decision="sign")
        )
        result = polypus.qml.train(
            model,
            _dataset(),
            method=polypus.DE(generations=30, population_size=16, tolerance=1e-9),
            loss="hinge",
            infrastructure="local",
            backend="polypus",
            id="qml_real_amplitudes",
            seed=7,
            exact=True,
        )
        # reps=2 → 1 axis × 2 qubits × 3 blocks = 6 trainable parameters, and the
        # well-separated dataset is reachable: near-zero hinge loss.
        assert len(result.best_params) == 6
        assert math.isfinite(result.best_fitness)
        assert result.best_fitness > -0.2

    def test_unknown_entangler_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="unknown entangler"):
            polypus.qml.Model(2).hardware_efficient(reps=1, entangler="cy")

    def test_unknown_entanglement_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="unknown entanglement"):
            polypus.qml.Model(2).hardware_efficient(reps=1, entanglement="mesh")

    def test_unknown_rotation_axis_in_list_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="unknown rotation axis"):
            polypus.qml.Model(2).hardware_efficient(reps=1, rotations=["ry", "rw"])
