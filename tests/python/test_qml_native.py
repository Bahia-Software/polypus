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

    def test_gradient_norm_early_stop_can_fire_on_a_minibatch(self):
        """The minibatch / gradient-norm early-stopping interaction, and what
        `patience` does to it (design doc §17, and the note beside C-5/C-7 in
        `CONTRACTS.md`).

        The trigger is structural, not shot noise. `_hard_dataset`'s last two
        samples are contradictory — identical features, opposite labels — so for a
        hinge loss their gradient contributions cancel term by term and a
        `batch_size=2` minibatch that draws exactly that pair has a gradient of
        **exactly** zero. `seed=11` is such a case on the first gradient call.
        (`crates/polypus/src/evaluation/exact_native_qml_oracle.rs` proves the two
        norms directly: 0 for the pair against > 0.2 for the full dataset.)

        Under the old one-iteration rule — which `patience=1` still selects, so
        the behaviour stays pinned and cannot regress unnoticed — the optimizer
        stopped on iteration 1 reporting `converged=True`, having barely moved
        from its random initialization. With the `patience=3` default the same
        seed no longer stops at all: it spends all 60 iterations and lands on the
        analytic optimum near −1/3, matching the run without `batch_size`.

        The reported `best_fitness` was honest either way — the full-dataset
        recompute guarantees C-5 — so what a spurious stop cost was the
        optimization itself and the meaning of `converged`, not the number."""
        import polypus

        def run(batch_size, tolerance=0.01, patience=3):
            return polypus.qml.train(
                _model(),
                self._hard_dataset(),
                method=polypus.Adam(
                    max_iters=60, tolerance=tolerance, patience=patience
                ),
                loss="hinge",
                infrastructure="local",
                backend="polypus",
                id="qml_mb_early_stop",
                seed=11,
                exact=True,
                batch_size=batch_size,
            )

        full = run(batch_size=None)
        assert not full.converged and full.iterations_run == 60
        # Near the analytic −1/3 bound the contradictory pair imposes.
        assert -0.40 < full.best_fitness < -0.30

        # `patience=1` is the pre-fix rule: the cancelling minibatch on the very
        # first gradient call ends the run there, at essentially the random init.
        legacy = run(batch_size=2, patience=1)
        assert legacy.converged and legacy.iterations_run == 1
        assert legacy.best_fitness < -1.0

        # A tighter tolerance was never the mitigation: the minibatch norm is
        # exactly zero, so it falls below any threshold whatsoever.
        for tolerance in (1e-6, 1e-12):
            tight = run(batch_size=2, tolerance=tolerance, patience=1)
            assert tight.converged and tight.iterations_run == 1
            assert tight.best_fitness == legacy.best_fitness

        # `patience=3` (the default) is: three *consecutive* sub-tolerance
        # iterations are needed, and iteration 2 draws a different minibatch, so
        # this seed no longer stops early — it reaches the same optimum the
        # full-batch run does.
        mb = run(batch_size=2)
        assert not mb.converged and mb.iterations_run == 60
        assert -0.40 < mb.best_fitness < -0.30
        assert abs(mb.best_fitness - full.best_fitness) < 0.05

    def test_gradient_norm_early_stop_affects_qng_identically(self):
        # QNG shares the same convergence rule, so both the old failure and the
        # `patience` fix behave identically on the same minibatch — the
        # interaction is a property of the convergence check, not of one
        # optimizer.
        import polypus

        def run(batch_size, patience=3):
            return polypus.qml.train(
                _model(),
                self._hard_dataset(),
                method=polypus.QNG(
                    variance_function=lambda *_: 0.25,
                    max_iters=60,
                    tolerance=0.01,
                    patience=patience,
                ),
                loss="hinge",
                infrastructure="local",
                backend="polypus",
                id="qml_mb_early_stop_qng",
                seed=11,
                exact=True,
                batch_size=batch_size,
            )

        assert not run(batch_size=None).converged

        legacy = run(batch_size=2, patience=1)
        assert legacy.converged and legacy.iterations_run == 1

        mb = run(batch_size=2)
        assert not mb.converged and mb.iterations_run == 60
        assert -0.40 < mb.best_fitness < -0.30

    def test_patience_is_a_probabilistic_mitigation_not_a_guarantee(self):
        """`patience` makes the spurious stop unlikely, not impossible — and that
        limit is demonstrated here rather than merely conceded in the docs.

        Nothing prevents the cancelling minibatch from being redrawn `patience`
        times in a row. `seed=140` is such a case, constructed rather than found
        by luck: the oracle's `MinibatchConfig` counter is shared by
        `evaluate_batch` and `gradient_batch` (one object serves both facets of
        the `Arc`), so gradient calls land on the **even** counter values — the
        gradient of iteration `n` uses `call_index = 2(n−1)`. Replaying
        `minibatch_indices`' SplitMix64 shuffle outside the crate and searching
        for seeds whose calls `2(n−1)`, `2n`, `2(n+1)` all draw the contradictory
        pair yields `seed=140` at `n = 6`, so the streak completes on iteration 8.

        Measured over 200 seeds on this dataset with `batch_size=2`, spurious
        stops fall from 194 at `patience=1` to 39 at `2` and 4 at `3` — a ~50×
        reduction with four survivors, of which this is one. So `converged=True`
        under minibatching is far more trustworthy than before, but still not a
        guarantee: `best_fitness` remains the number to read."""
        import polypus

        def run(batch_size, opt="adam"):
            method = (
                polypus.Adam(max_iters=60, tolerance=0.01, patience=3)
                if opt == "adam"
                else polypus.QNG(
                    variance_function=lambda *_: 0.25,
                    max_iters=60,
                    tolerance=0.01,
                    patience=3,
                )
            )
            return polypus.qml.train(
                _model(),
                self._hard_dataset(),
                method=method,
                loss="hinge",
                infrastructure="local",
                backend="polypus",
                id="qml_mb_patience_survivor",
                seed=140,
                exact=True,
                batch_size=batch_size,
            )

        # The stop still fires with the default patience, on the predicted
        # iteration, and it is genuinely spurious: ≈ −0.94 where the same
        # configuration without `batch_size` reaches ≈ −0.34.
        for opt in ("adam", "qng"):
            mb = run(batch_size=2, opt=opt)
            assert mb.converged, f"{opt}: expected the survivor case to stop"
            assert mb.iterations_run == 8, f"{opt}: {mb.iterations_run}"
            assert mb.best_fitness < -0.5, f"{opt}: {mb.best_fitness}"

        full = run(batch_size=None)
        assert not full.converged and full.iterations_run == 60
        assert -0.40 < full.best_fitness < -0.30

    def test_early_stop_does_not_fire_without_cancelling_structure(self):
        # The scope of the interaction: on a separable dataset with no
        # contradictory samples, minibatch gradient norms stay well above the
        # default tolerance, and no seed in this sweep stops early — under either
        # rule. Asserted at `patience=1` because that is the *strict* case: if the
        # single-iteration rule never fires here, the three-consecutive rule
        # cannot either, so this pins the scope of the interaction independently
        # of the mitigation. The risk tracks *cancellation between samples of one
        # minibatch*, not minibatching in general.
        import polypus

        for seed in range(1, 11):
            r = polypus.qml.train(
                _model(),
                _dataset(),  # 6 well-separated samples, no contradictory pair
                method=polypus.Adam(max_iters=40, tolerance=0.01, patience=1),
                loss="hinge",
                infrastructure="local",
                backend="polypus",
                id="qml_mb_no_early_stop",
                seed=seed,
                exact=True,
                batch_size=2,
            )
            assert r.iterations_run > 3, (
                f"seed {seed} stopped after {r.iterations_run} iterations on a "
                "dataset with no cancelling structure"
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


def _exact_expectation(model, theta, sample=None, dataset=None):
    """The exact ⟨Z₀⟩ of `model` bound to `theta` on one sample. Deterministic
    (no sampling), and raises `ValueError` if `theta` has the wrong length.

    `sample=None` builds one of the dataset's own width, so a caller who only
    cares about θ cannot accidentally trip the feature-count check instead."""
    import polypus

    dataset = dataset or _dataset()
    if sample is None:
        sample = [0.1 * (i + 1) for i in range(dataset.num_features)]
    trained = polypus.qml.TrainedModel(model, dataset, list(theta))
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


# ─────────────────────────────────────────────────────────────────────────────
# 8. The layers that had no Python sugar: amplitude / IQP encoders, conv, pool
# ─────────────────────────────────────────────────────────────────────────────


def _dataset_4f():
    """Two well-separated clusters in ``[0, π]`` over **four** features — the
    width the QCNN and the IQP entanglement test need (see the note on
    `TestIqpEncoderLayer`)."""
    import polypus

    x = [
        [0.30, 0.35, 0.25, 0.30],
        [0.40, 0.30, 0.35, 0.25],
        [0.35, 0.40, 0.30, 0.35],
        [2.80, 2.75, 2.85, 2.80],
        [2.90, 2.80, 2.75, 2.85],
        [2.75, 2.90, 2.80, 2.75],
    ]
    y = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    return polypus.qml.Dataset(x, y)


_SAMPLE_4F = (0.7, 1.1, 0.4, 1.3)


@pytest.mark.integration
@pytest.mark.vqc
class TestAmplitudeEncoderLayer:
    def test_encoding_is_scale_invariant(self):
        # The characteristic property of amplitude encoding: the sample is
        # normalized before it becomes amplitudes, so two positive multiples of
        # the same vector prepare the *identical* state and must predict the same
        # value bit for bit. An angle encoder has no such invariance — the
        # companion assertion below pins that difference down.
        amplitude = _amplitude_raw_model(2, reps=1)
        theta = [0.1 * (i + 1) for i in range(8)]
        a = _exact_expectation(amplitude, theta, sample=(0.3, 0.4))
        b = _exact_expectation(amplitude, theta, sample=(0.6, 0.8))
        assert a == b

    def test_angle_encoder_is_not_scale_invariant(self):
        # Guards the test above against passing for a trivial reason (e.g. a
        # readout that ignores the encoder entirely).
        angle = _raw_model(2, lambda m: m.hardware_efficient(reps=1))
        theta = [0.1 * (i + 1) for i in range(8)]
        assert _exact_expectation(angle, theta, sample=(0.3, 0.4)) != (
            _exact_expectation(angle, theta, sample=(0.6, 0.8))
        )

    def test_reserves_no_parameters_of_its_own(self):
        # An encoder consumes no θ: the count is the ansatz's alone (2 × 2 × 2).
        assert _num_params(_amplitude_raw_model(2, reps=1)) == 8

    def test_rejected_when_not_the_first_layer(self):
        # `compile` owns this rule; from Python it must surface as a ValueError.
        import polypus

        model = (
            polypus.qml.Model(2)
            .hardware_efficient(reps=1)
            .amplitude_encoder()
            .readout(observables=[[("z", 0)]], decision="sign")
        )
        with pytest.raises(ValueError, match="first"):
            polypus.qml.TrainedModel(model, _dataset(), [0.0] * 8)


def _amplitude_raw_model(num_qubits, reps):
    """An amplitude-encoder model reading ⟨Z₀⟩ raw."""
    import polypus

    return (
        polypus.qml.Model(num_qubits)
        .amplitude_encoder()
        .hardware_efficient(reps=reps)
        .readout(observables=[[("z", 0)]], decision="raw")
    )


@pytest.mark.integration
@pytest.mark.vqc
class TestIqpEncoderLayer:
    """The IQP encoder's `Rzz` gates are diagonal (they commute) and symmetric in
    their two qubits, so two patterns selecting the same *unordered* pairs build
    the same unitary regardless of order. That is why this class needs **four**
    features: at n = 3, `circular` = {01,12,20} and `full` = {01,02,12} are the
    same unordered set and genuinely coincide, while at n = 4 `full` has six pairs
    against `circular`'s four and all three patterns differ."""

    THETA_4Q = [0.1 * (i + 1) for i in range(16)]

    def test_entanglement_patterns_build_different_circuits(self):
        values = {}
        for pattern in ("linear", "circular", "full"):
            model = _iqp_raw_model(4, pattern)
            assert _num_params(model, dataset=_dataset_4f()) == 16
            values[pattern] = _exact_expectation(
                model, self.THETA_4Q, sample=_SAMPLE_4F, dataset=_dataset_4f()
            )
        assert len(set(values.values())) == 3, (
            f"IQP entanglement patterns collapsed to the same circuit: {values}"
        )

    def test_default_entanglement_is_full(self):
        # `IqpEncoder::new`'s connectivity — the original ZZFeatureMap.
        default = _iqp_raw_model(4, None)
        full = _iqp_raw_model(4, "full")
        args = dict(sample=_SAMPLE_4F, dataset=_dataset_4f())
        assert _exact_expectation(default, self.THETA_4Q, **args) == (
            _exact_expectation(full, self.THETA_4Q, **args)
        )

    def test_trains_end_to_end(self):
        import math

        import polypus

        model = (
            polypus.qml.Model(4)
            .iqp_encoder()
            .hardware_efficient(reps=1)
            .readout(observables=[[("z", 0)]], decision="sign")
        )
        result = polypus.qml.train(
            model,
            _dataset_4f(),
            method=polypus.DE(generations=30, population_size=16, tolerance=1e-9),
            loss="hinge",
            infrastructure="local",
            backend="polypus",
            id="qml_iqp_train",
            seed=7,
            exact=True,
        )
        assert len(result.best_params) == 16
        assert math.isfinite(result.best_fitness)

    def test_unknown_entanglement_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="unknown entanglement"):
            polypus.qml.Model(4).iqp_encoder(entanglement="star")


def _iqp_raw_model(num_qubits, entanglement):
    """An IQP-encoder model reading ⟨Z₀⟩ raw. `entanglement=None` leaves the
    kwarg off entirely, exercising the default."""
    import polypus

    model = polypus.qml.Model(num_qubits)
    model = (
        model.iqp_encoder()
        if entanglement is None
        else model.iqp_encoder(entanglement=entanglement)
    )
    return model.hardware_efficient(reps=1).readout(
        observables=[[("z", 0)]], decision="raw"
    )


@pytest.mark.integration
@pytest.mark.vqc
class TestConvAndPoolLayers:
    def test_conv_theta_count_is_the_block_width_only(self):
        # Parameter sharing: the block reserves its θ once per *layer*, so the
        # count is fixed by the block and independent of the qubit count —
        # 4 for "basic", 3 for "cartan", on 2 qubits and on 4 alike.
        for num_qubits in (2, 4):
            assert _num_params(_conv_raw_model(num_qubits, "basic")) == 4
            assert _num_params(_conv_raw_model(num_qubits, "cartan")) == 3

    def test_conv_pairings_build_different_circuits(self):
        # On 4 qubits: even {(0,1),(2,3)}, odd {(1,2)}, alternating = even + odd.
        theta = [0.3, 0.7, 1.1, 1.5]
        values = {
            pairing: _exact_expectation(
                _conv_raw_model(4, "basic", pairing=pairing), theta
            )
            for pairing in ("even_pairs", "odd_pairs", "alternating")
        }
        assert len(set(values.values())) == 3, (
            f"conv pairings collapsed to the same circuit: {values}"
        )

    def test_pool_theta_count_is_the_block_width_only(self):
        for num_qubits in (2, 4):
            assert _num_params(_pool_raw_model(num_qubits, keep=None)) == 3

    def test_pool_shrinks_the_active_set(self):
        # 4 qubits pooled in adjacent pairs leave 2 active, so a readout at
        # logical position 1 still resolves and position 2 no longer does.
        import polypus

        assert _num_params(_pool_raw_model(4, keep=None, position=1)) == 3
        with pytest.raises(ValueError, match="logical qubit 2"):
            polypus.qml.TrainedModel(
                _pool_raw_model(4, keep=None, position=2), _dataset(), [0.0] * 3
            )

    def test_pool_keep_rule_selects_which_qubit_survives(self):
        # Logical position 0 of the pooled set is physical qubit 0 under
        # "even_positions" and physical qubit 1 under "odd_positions", and the
        # block's discarded/retained roles swap with it — so the same θ reads a
        # different expectation.
        theta = [0.3, 0.7, 1.1]
        even = _exact_expectation(_pool_raw_model(4, keep="even_positions"), theta)
        odd = _exact_expectation(_pool_raw_model(4, keep="odd_positions"), theta)
        assert even != odd

    def test_qcnn_trains_end_to_end_from_python(self):
        # conv → pool → conv over 4 features: the whole QCNN stack reached from
        # Python, trained through `polypus.qml.train` on the native exact path.
        # (Rust already covers the layers' semantics; what is new here is that
        # the Python spelling arrives and trains.)
        import math

        import polypus

        model = (
            polypus.qml.Model(4)
            .angle_encoder(axis="ry")
            .conv(block="basic")
            .pool(block="basic")
            .conv(block="basic", pairing="even_pairs")
            .readout(observables=[[("z", 0)]], decision="sign")
        )
        result = polypus.qml.train(
            model,
            _dataset_4f(),
            method=polypus.DE(generations=40, population_size=16, tolerance=1e-9),
            loss="hinge",
            infrastructure="local",
            backend="polypus",
            id="qml_qcnn_train",
            seed=7,
            exact=True,
        )
        # 4 (conv) + 3 (pool) + 4 (conv) shared parameters — no per-pair blow-up.
        assert len(result.best_params) == 11
        assert math.isfinite(result.best_fitness)
        # The clusters are well separated, so the QCNN reaches near-zero hinge.
        assert result.best_fitness > -0.2

    def test_qcnn_reproducible_for_fixed_seed(self):
        # Same guarantee as every other native run (C-7), now via conv/pool.
        import polypus

        def run():
            model = (
                polypus.qml.Model(4)
                .angle_encoder(axis="ry")
                .conv(block="cartan")
                .pool(block="basic", keep="odd_positions")
                .readout(observables=[[("z", 0)]], decision="sign")
            )
            return polypus.qml.train(
                model,
                _dataset_4f(),
                method=polypus.DE(generations=20, population_size=12, tolerance=1e-12),
                loss="hinge",
                infrastructure="local",
                backend="polypus",
                id="qml_qcnn_seed",
                seed=13,
                exact=True,
            )

        a, b = run(), run()
        assert a.best_params == b.best_params
        assert a.best_fitness == b.best_fitness

    def test_unknown_conv_block_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="unknown conv block"):
            polypus.qml.Model(4).conv(block="fancy")

    def test_unknown_pairing_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="unknown pairing"):
            polypus.qml.Model(4).conv(block="basic", pairing="every_pair")

    def test_unknown_pool_block_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="unknown pool block"):
            polypus.qml.Model(4).pool(block="max")

    def test_unknown_keep_rule_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="unknown keep rule"):
            polypus.qml.Model(4).pool(block="basic", keep="first_half")


def _conv_raw_model(num_qubits, block, pairing=None):
    """An angle-encoder + single conv-layer model reading ⟨Z₀⟩ raw."""
    import polypus

    model = polypus.qml.Model(num_qubits).angle_encoder(axis="ry")
    model = (
        model.conv(block=block)
        if pairing is None
        else model.conv(block=block, pairing=pairing)
    )
    return model.readout(observables=[[("z", 0)]], decision="raw")


# ─────────────────────────────────────────────────────────────────────────────
# 9. TrainedModel.predict_from_probabilities — the exact mirror of ...from_counts
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vqc
class TestPredictFromProbabilities:
    """The readout reads counts only through their relative frequencies, so a
    probability dict and a counts dict describing the **same** distribution must
    give the same prediction. These pick distributions where that correspondence
    is exact in binary floating point, so the assertions are `==`, not `approx`."""

    @staticmethod
    def _raw_trained():
        # `decision="raw"` so the comparison is over a continuous expectation
        # rather than a ±1 label two different values could share. θ is irrelevant
        # here: neither entry point runs a circuit.
        import polypus

        model = _raw_model(2, lambda m: m.hardware_efficient(reps=1))
        return polypus.qml.TrainedModel(model, _dataset(), [0.0] * 8)

    @staticmethod
    def _sign_trained():
        import polypus

        return polypus.qml.TrainedModel(_model(), _dataset(), [0.0] * 8)

    def test_matches_counts_on_the_same_distribution(self):
        # Little-endian keys (C-3): qubit 0 is the last character, so "00" reads
        # ⟨Z₀⟩ = +1 and "01" reads −1. A 3:1 split gives 0.75 − 0.25 = 0.5 both
        # ways — exactly representable, hence the equality.
        trained = self._raw_trained()
        from_probs = trained.predict_from_probabilities({"00": 0.75, "01": 0.25})
        from_counts = trained.predict_from_counts({"00": 3, "01": 1})
        assert from_probs == 0.5
        assert from_probs == from_counts

    def test_deterministic_extremes(self):
        # An all-mass distribution reproduces the pure-state readings the counts
        # entry point gives for a single key.
        trained = self._raw_trained()
        assert trained.predict_from_probabilities({"00": 1.0, "01": 0.0}) == 1.0
        assert trained.predict_from_probabilities({"00": 0.0, "01": 1.0}) == -1.0
        assert trained.predict_from_probabilities({"00": 0.5, "01": 0.5}) == 0.0

    def test_applies_the_sign_decision_like_counts(self):
        # The decision rule is applied to the expectation identically on both
        # entry points, including the `e >= 0 → +1` tie convention.
        trained = self._sign_trained()
        cases = [
            ({"00": 0.75, "01": 0.25}, {"00": 3, "01": 1}, 1.0),
            ({"00": 0.25, "01": 0.75}, {"00": 1, "01": 3}, -1.0),
            ({"00": 0.5, "01": 0.5}, {"00": 1, "01": 1}, 1.0),
        ]
        for probs, counts, expected in cases:
            assert trained.predict_from_probabilities(probs) == expected
            assert trained.predict_from_counts(counts) == expected

    def test_probabilities_over_the_full_register_agree_with_counts(self):
        # All four basis states present, with a distribution whose ⟨Z₀⟩ is exact:
        # p(q0=0) = 0.125 + 0.375 = 0.5 → ⟨Z₀⟩ = 0.
        trained = self._raw_trained()
        probs = {"00": 0.125, "01": 0.25, "10": 0.375, "11": 0.25}
        counts = {"00": 1, "01": 2, "10": 3, "11": 2}
        assert trained.predict_from_probabilities(probs) == 0.0
        assert trained.predict_from_probabilities(probs) == trained.predict_from_counts(
            counts
        )

    def test_empty_probabilities_rejected(self):
        trained = self._raw_trained()
        with pytest.raises(ValueError):
            trained.predict_from_probabilities({})

    def test_inconsistent_key_widths_rejected(self):
        # Same C-3 width rule the counts path enforces — a clean ValueError,
        # never a panic across the FFI boundary.
        trained = self._raw_trained()
        with pytest.raises(ValueError, match="width|counts"):
            trained.predict_from_probabilities({"00": 0.5, "010": 0.5})


# ─────────────────────────────────────────────────────────────────────────────
# 10. Dataset: train/test split and feature scaling
# ─────────────────────────────────────────────────────────────────────────────
#
# `Dataset` exposes no row accessor, so these read the data back through
# `feature_ranges()` — which is exactly enough to pin the scaling arithmetic down
# to exact float equality, and (for a 1- or 2-sample partition) to identify the
# selected samples outright.


@pytest.mark.integration
@pytest.mark.vqc
class TestDatasetSplit:
    @staticmethod
    def _indexed(n=6):
        """`n` samples whose first feature *is* the sample index, so a partition's
        `feature_ranges()[0]` reports which samples it received."""
        import polypus

        x = [[float(i), 1.0 + 0.5 * i] for i in range(n)]
        y = [1.0 if i % 2 else -1.0 for i in range(n)]
        return polypus.qml.Dataset(x, y)

    def test_sizes_and_test_partition_rounds_down(self):
        # floor(6 × 0.4) = 2 test, so 4 train — the documented rounding rule.
        train, test = self._indexed().train_test_split(0.4, seed=7)
        assert (train.num_samples, test.num_samples) == (4, 2)
        # Both keep the full feature width.
        assert train.num_features == test.num_features == 2

    def test_fixed_seed_reproduces_the_partition(self):
        a_train, a_test = self._indexed().train_test_split(1 / 3, seed=7)
        b_train, b_test = self._indexed().train_test_split(1 / 3, seed=7)
        # The index feature's range identifies the two selected test samples.
        assert a_test.feature_ranges() == b_test.feature_ranges()
        assert a_train.feature_ranges() == b_train.feature_ranges()

    def test_different_seeds_give_different_partitions(self):
        # SplitMix64 is deterministic, so this is a property of the shuffle, not
        # a probabilistic hope: across a handful of seeds the 2-sample test
        # partition cannot be the same one every time.
        picks = {
            self._indexed().train_test_split(1 / 3, seed=s)[1].feature_ranges()[0]
            for s in range(1, 9)
        }
        assert len(picks) > 1

    def test_single_sample_test_partition_is_excluded_from_train(self):
        # floor(6 × 0.2) = 1 test sample, so its `feature_ranges()[0]` is
        # (i, i) — the exact index. Splitting is a partition, so that index must
        # be missing from the 5-sample train set; with the index as feature 0 the
        # train range proves it whenever the drawn index is an endpoint, and
        # bounds it otherwise.
        train, test = self._indexed().train_test_split(0.2, seed=7)
        assert test.num_samples == 1 and train.num_samples == 5
        lo, hi = test.feature_ranges()[0]
        assert lo == hi and lo in {float(i) for i in range(6)}
        train_lo, train_hi = train.feature_ranges()[0]
        if lo == 0.0:
            assert train_lo == 1.0
        elif lo == 5.0:
            assert train_hi == 4.0

    def test_unseeded_split_still_splits(self):
        # `seed=None` draws OS entropy: the partition is not reproducible, but it
        # is still a valid split of the right sizes.
        train, test = self._indexed().train_test_split(1 / 3)
        assert (train.num_samples, test.num_samples) == (4, 2)

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5, float("nan")])
    def test_test_fraction_outside_the_open_unit_interval_rejected(self, fraction):
        # Either endpoint would leave a partition empty; NaN fails every
        # comparison and is rejected by the same guard.
        with pytest.raises(ValueError, match="test_fraction|fraction"):
            self._indexed().train_test_split(fraction, seed=7)

    def test_train_partition_trains(self):
        # The seam's real job: a split partition is a usable Dataset.
        import math

        import polypus

        train, _ = self._indexed(n=8).train_test_split(0.25, seed=7)
        result = polypus.qml.train(
            _model(),
            train,
            method=polypus.DE(generations=10, population_size=8, tolerance=1e-9),
            loss="hinge",
            infrastructure="local",
            backend="polypus",
            id="qml_split_train",
            seed=7,
            exact=True,
        )
        assert len(result.best_params) == 8
        assert math.isfinite(result.best_fitness)


@pytest.mark.integration
@pytest.mark.vqc
class TestDatasetScaling:
    @staticmethod
    def _ds(rows):
        import polypus

        return polypus.qml.Dataset(rows, [1.0] * len(rows))

    def test_feature_ranges_reports_per_feature_min_max(self):
        ds = self._ds([[0.0, 20.0], [10.0, 5.0], [4.0, 30.0]])
        assert ds.feature_ranges() == [(0.0, 10.0), (5.0, 30.0)]

    def test_scale_features_to_maps_each_feature_onto_the_interval(self):
        # In place, and every non-constant feature ends spanning exactly [lo, hi].
        ds = self._ds([[0.0, 20.0], [10.0, 5.0], [4.0, 30.0]])
        ds.scale_features_to(0.0, 1.0)
        assert ds.feature_ranges() == [(0.0, 1.0), (0.0, 1.0)]

    def test_scale_features_to_is_exact_on_known_values(self):
        # (v − min)/span × (hi − lo) + lo, with values chosen so the arithmetic is
        # exact in binary floating point: 0 → 0, 5 → 0.5, 10 → 1.
        ds = self._ds([[0.0], [5.0], [10.0]])
        ds.scale_features_to(0.0, 1.0)
        # After scaling the range is [0, 1]; scaling again is then the identity,
        # which pins the midpoint too: a mid value of anything but 0.5 would move.
        before = ds.feature_ranges()
        ds.scale_features_to(0.0, 1.0)
        assert ds.feature_ranges() == before == [(0.0, 1.0)]

    def test_constant_feature_maps_to_lo(self):
        # A constant column has no range to normalize against.
        ds = self._ds([[7.0, 1.0], [7.0, 2.0]])
        ds.scale_features_to(3.0, 9.0)
        assert ds.feature_ranges() == [(3.0, 3.0), (3.0, 9.0)]

    def test_scale_features_with_replays_a_frozen_scaler(self):
        # The train set's ranges, replayed on a test set: test values beyond the
        # frozen range land *outside* [lo, hi] — the intended behaviour, and the
        # observable difference from `scale_features_to`, which would have
        # rescaled the test set onto [0, 1] using its own min/max.
        train = self._ds([[0.0], [10.0]])
        test = self._ds([[5.0], [15.0]])
        ranges = train.feature_ranges()
        assert ranges == [(0.0, 10.0)]
        test.scale_features_with(ranges, 0.0, 1.0)
        assert test.feature_ranges() == [(0.5, 1.5)]

    def test_scale_features_with_wrong_range_count_rejected(self):
        ds = self._ds([[0.0, 1.0], [2.0, 3.0]])
        with pytest.raises(ValueError, match="feature"):
            ds.scale_features_with([(0.0, 1.0)], 0.0, 1.0)

    def test_split_then_freeze_train_ranges_onto_test(self):
        # The workflow the two methods exist for, end to end.
        ds = TestDatasetSplit._indexed(n=8)
        train, test = ds.train_test_split(0.25, seed=7)
        ranges = train.feature_ranges()
        train.scale_features_to(0.0, 3.141592653589793)
        test.scale_features_with(ranges, 0.0, 3.141592653589793)
        # The train set now spans exactly [0, π]; the test set is on the *same*
        # scale, so its values stay within a sane band around it rather than
        # being independently stretched onto [0, π].
        assert train.feature_ranges()[0] == (0.0, 3.141592653589793)
        test_lo, test_hi = test.feature_ranges()[0]
        assert -3.2 < test_lo <= test_hi < 6.3


def _pool_raw_model(num_qubits, keep, position=0):
    """An angle-encoder + single pool-layer model reading ⟨Z at `position`⟩ raw,
    where `position` indexes the **surviving** active qubits."""
    import polypus

    model = polypus.qml.Model(num_qubits).angle_encoder(axis="ry")
    model = (
        model.pool(block="basic")
        if keep is None
        else model.pool(block="basic", keep=keep)
    )
    return model.readout(observables=[[("z", position)]], decision="raw")


# ─────────────────────────────────────────────────────────────────────────────
# 12. `polypus.qml.Observable` — weighted multi-term readout observables (§17)
# ─────────────────────────────────────────────────────────────────────────────
#
# `readout` used to force one Pauli string with coefficient 1.0 per observable;
# `polypus.qml.Observable([(c, term), …])` now spells the weighted sum
# `Σ cᵢ·Pᵢ` the Rust `Observable` always supported. The type is **additive**: the
# bare `[("z", 0)]` form is unchanged, and the two may be mixed in one call.
#
# The exactness these tests lean on is the same as section 7's: a `decision="raw"`
# readout under `predict(..., exact=True)` returns the observable's exact
# expectation, no shot noise — so the weighted sum can be checked against a value
# computed analytically rather than merely asserted finite.


def _product_state_model(observables, decision="raw"):
    """A 2-qubit model whose bound state is an exactly-known **product** state.

    `angle_encoder(ry)` puts `Ry(x_j)` on qubit `j`; the ansatz is a single
    rotation block of one `Ry` per qubit and **no entangler** (`reps=0` leaves
    the entangling loop empty, `final_rotation_layer=True` keeps the block). So
    the state is `Ry(x₀+θ₀)|0⟩ ⊗ Ry(x₁+θ₁)|0⟩` and every Pauli-`Z` expectation
    is a closed form: ⟨Z_j⟩ = cos(x_j + θ_j), and ⟨Z₀Z₁⟩ = ⟨Z₀⟩·⟨Z₁⟩ because the
    state factorises. That is what `_expected_z` below computes.
    """
    import polypus

    return (
        polypus.qml.Model(2)
        .angle_encoder(axis="ry")
        .hardware_efficient(reps=0, rotations=["ry"], final_rotation_layer=True)
        .readout(observables=observables, decision=decision)
    )


def _expected_z(sample, theta, positions):
    """The analytic ⟨Π_j Z_j⟩ of `_product_state_model`'s state over `positions`."""
    import math

    value = 1.0
    for j in positions:
        value *= math.cos(sample[j] + theta[j])
    return value


_OBS_THETA = [0.7, 1.3]
_OBS_SAMPLE = [0.4, 0.9]


def _obs_predict(observables, sample=None, decision="raw"):
    """The exact prediction of `_product_state_model(observables)` on one sample."""
    import polypus

    dataset = _dataset()
    trained = polypus.qml.TrainedModel(
        _product_state_model(observables, decision=decision), dataset, list(_OBS_THETA)
    )
    return trained.predict(
        [list(sample or _OBS_SAMPLE)],
        infrastructure="local",
        backend="polypus",
        id="qml_observable",
        exact=True,
    )[0]


@pytest.mark.integration
@pytest.mark.vqc
class TestObservableIsAdditive:
    """The bare form keeps working exactly as before, and its explicit
    `Observable` spelling builds the identical observable."""

    def test_bare_form_and_explicit_single_term_agree_bit_for_bit(self):
        import polypus

        bare = _obs_predict([[("z", 0)]])
        explicit = _obs_predict([polypus.qml.Observable([(1.0, [("z", 0)])])])
        # Not merely close: the same circuit and the same coefficient, so the
        # exact expectation must be the identical float.
        assert bare == explicit
        assert bare == _expected_z(_OBS_SAMPLE, _OBS_THETA, [0])

    def test_repr_reports_the_term_count(self):
        import polypus

        assert (
            repr(
                polypus.qml.Observable([(0.5, [("z", 0)]), (1.5, [("z", 0), ("z", 1)])])
            )
            == "Observable(num_terms=2)"
        )


@pytest.mark.integration
@pytest.mark.vqc
class TestObservableWeightedSum:
    """A genuine `Σ cᵢ·Pᵢ`, checked against the analytic value — not just
    "the call does not raise"."""

    def test_weighted_sum_matches_the_hand_computed_expectation(self):
        import polypus

        observable = polypus.qml.Observable(
            [(0.5, [("z", 0)]), (1.5, [("z", 0), ("z", 1)])]
        )
        got = _obs_predict([observable])
        expected = 0.5 * _expected_z(_OBS_SAMPLE, _OBS_THETA, [0]) + 1.5 * _expected_z(
            _OBS_SAMPLE, _OBS_THETA, [0, 1]
        )
        assert got == pytest.approx(expected, abs=1e-12)
        # And it is genuinely a sum: neither term alone reproduces it.
        assert got != pytest.approx(_obs_predict([[("z", 0)]]), abs=1e-9)

    def test_coefficients_scale_the_expectation_linearly(self):
        # Doubling every coefficient doubles ⟨O⟩ — the defining property of the
        # weighted sum, and impossible to express with the bare form.
        import polypus

        single = _obs_predict([polypus.qml.Observable([(1.0, [("z", 0)])])])
        doubled = _obs_predict([polypus.qml.Observable([(2.0, [("z", 0)])])])
        assert doubled == pytest.approx(2.0 * single, abs=1e-12)

    def test_weighted_observable_trains_end_to_end(self):
        # The weighted observable travels the whole native training path, not
        # just inference: compile → QmlProblem → oracle → optimizer.
        import math

        import polypus

        model = _product_state_model(
            [polypus.qml.Observable([(0.5, [("z", 0)]), (0.5, [("z", 0), ("z", 1)])])],
            decision="sign",
        )
        result = polypus.qml.train(
            model,
            _dataset(),
            method=polypus.DE(population_size=6, generations=4),
            loss="hinge",
            infrastructure="local",
            backend="polypus",
            id="qml_observable_train",
            seed=7,
            exact=True,
        )
        assert len(result.best_params) == 2
        assert math.isfinite(result.best_fitness)


@pytest.mark.integration
@pytest.mark.vqc
class TestObservableMixedWithBareFormInArgmax:
    """Both spellings in the *same* `readout` call, on a multiclass `argmax`."""

    # Class 0 is the bare ⟨Z₀⟩; class 1 is the weighted sum. The two samples are
    # chosen so each class wins once — a test where one class always won would
    # pass even if the other observable were silently dropped.
    CASES = [([0.4, 0.9], 0), ([2.9, 0.2], 1)]

    def test_argmax_picks_the_class_with_the_larger_expectation(self):
        import polypus

        observables = [
            [("z", 0)],
            polypus.qml.Observable([(0.5, [("z", 0)]), (1.5, [("z", 0), ("z", 1)])]),
        ]
        for sample, expected_class in self.CASES:
            z0 = _expected_z(sample, _OBS_THETA, [0])
            z0z1 = _expected_z(sample, _OBS_THETA, [0, 1])
            scores = [z0, 0.5 * z0 + 1.5 * z0z1]
            # The hand-computed winner, so the assertion below is pinned to the
            # analytic values and not to whatever the model happens to return.
            assert max(range(2), key=lambda i: scores[i]) == expected_class
            got = _obs_predict(observables, sample=sample, decision="argmax")
            assert got == float(expected_class)


@pytest.mark.integration
@pytest.mark.vqc
class TestObservableValidation:
    """Every rejection is a clean Python exception, never a panic."""

    @pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_coefficient_rejected(self, coefficient):
        import polypus

        with pytest.raises(ValueError, match="non-finite coefficient"):
            polypus.qml.Observable([(coefficient, [("z", 0)])])

    def test_non_finite_coefficient_reported_with_its_term_index(self):
        import polypus

        with pytest.raises(ValueError, match="term 1"):
            polypus.qml.Observable(
                [(1.0, [("z", 0)]), (float("nan"), [("z", 0), ("z", 1)])]
            )

    def test_unknown_pauli_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="unknown Pauli"):
            polypus.qml.Observable([(1.0, [("w", 0)])])

    def test_repeated_position_within_one_term_rejected(self):
        import polypus

        with pytest.raises(ValueError, match="position"):
            polypus.qml.Observable([(1.0, [("z", 0), ("x", 0)])])

    def test_readout_rejects_an_element_that_is_neither_form(self):
        import polypus

        model = polypus.qml.Model(2).angle_encoder(axis="ry").real_amplitudes(reps=1)
        with pytest.raises(TypeError, match="polypus.qml.Observable"):
            model.readout(observables=[42], decision="raw")


# ─────────────────────────────────────────────────────────────────────────────
# 13. `QmlTrainResult` — the native path returns its trained model (§17)
# ─────────────────────────────────────────────────────────────────────────────
#
# `qml.train` used to hand back a bare `TrainResult` on both paths, which knows
# nothing about the `Model`/`Dataset` it trained — so predicting meant rebuilding
# `TrainedModel(model, dataset, result.best_params)` by hand, passing back in the
# two objects the call already had. The native path now returns a
# `QmlTrainResult`: the same six fields plus `trained_model`, built eagerly at the
# end of training. The Qiskit path is unchanged and still returns `TrainResult`
# (contract C-7) — the return type follows the path, like the kwargs already do.


def _native_train(seed=7, **kwargs):
    """One short native run on the exact path, so θ is deterministic."""
    import polypus

    return polypus.qml.train(
        _model(),
        _dataset(),
        method=polypus.DE(generations=20, population_size=12, tolerance=1e-9),
        loss="hinge",
        infrastructure="local",
        backend="polypus",
        id="qml_train_result",
        seed=seed,
        exact=True,
        **kwargs,
    )


@pytest.mark.integration
@pytest.mark.vqc
class TestQmlTrainResultNativePath:
    def test_native_path_returns_a_qml_train_result(self):
        import polypus

        result = _native_train()
        assert isinstance(result, polypus.qml.QmlTrainResult)
        # Two independent types by design (no inheritance), so a `TrainResult`
        # this is not — pinning that keeps a future `extends = TrainResult`
        # from being introduced silently.
        assert not isinstance(result, polypus.TrainResult)
        assert type(result).__name__ == "QmlTrainResult"

    def test_carries_the_six_train_result_fields(self):
        result = _native_train()
        assert result.seed == 7
        assert isinstance(result.best_params, list)
        assert len(result.best_params) == 8  # the 8 θ of `_model()`
        assert all(isinstance(p, float) for p in result.best_params)
        assert isinstance(result.best_fitness, float)
        # Exact hinge fitness is `−mean(loss)`, so finite and non-positive.
        assert result.best_fitness <= 0.0
        assert 1 <= result.iterations_run <= 20
        assert isinstance(result.converged, bool)
        # The effective id is the `id` prefix plus a UUID v4 suffix (#75).
        assert result.id.startswith("qml_train_result_")
        assert result.id != "qml_train_result"

    def test_fields_match_a_train_result_from_the_same_run(self):
        # The upgrade must copy the outcome verbatim, not recompute it: the same
        # seed on the exact path is byte-reproducible, so every field of two runs
        # agrees — and `trained_model.theta` is exactly `best_params`.
        a = _native_train()
        b = _native_train()
        assert a.best_params == b.best_params
        assert a.best_fitness == b.best_fitness
        assert a.iterations_run == b.iterations_run
        assert a.converged == b.converged
        assert a.seed == b.seed
        assert a.trained_model.theta == a.best_params

    def test_repr_mentions_the_outcome_and_the_trained_model(self):
        result = _native_train()
        text = repr(result)
        assert text.startswith("QmlTrainResult(")
        for field in ("best_fitness=", "iterations_run=", "converged=", "seed="):
            assert field in text
        assert "trained_model=TrainedModel(num_theta=8)" in text

    def test_trained_model_is_reused_not_rebuilt_per_access(self):
        # The attribute is a stored object, built once: two reads are the same
        # instance, so `predict`-ing twice does not recompile the model.
        result = _native_train()
        assert result.trained_model is result.trained_model


@pytest.mark.integration
@pytest.mark.vqc
class TestQmlTrainResultTrainedModelEquivalence:
    """`result.trained_model` must be *the same* trained model a caller would
    have assembled by hand — identical predictions, not merely "it runs"."""

    @staticmethod
    def _manual(result):
        import polypus

        return polypus.qml.TrainedModel(_model(), _dataset(), result.best_params)

    def test_exact_predictions_are_bit_identical_to_the_manual_wrapper(self):
        result = _native_train()
        kwargs = dict(
            infrastructure="local",
            backend="polypus",
            id="qml_train_result_predict",
            exact=True,
        )
        auto = result.trained_model.predict(_NEW_SAMPLES, **kwargs)
        manual = self._manual(result).predict(_NEW_SAMPLES, **kwargs)
        # Exact mode has no shot noise: equality is exact, not approximate.
        assert auto == manual

    def test_seeded_shot_predictions_are_bit_identical_to_the_manual_wrapper(self):
        # The shot path with a fixed seed is byte-reproducible too, so the two
        # wrappers must agree there as well — the same compiled model and θ.
        result = _native_train()
        kwargs = dict(
            shots=2048,
            infrastructure="local",
            backend="polypus",
            id="qml_train_result_predict_shots",
            seed=11,
        )
        auto = result.trained_model.predict(_NEW_SAMPLES, **kwargs)
        manual = self._manual(result).predict(_NEW_SAMPLES, **kwargs)
        assert auto == manual

    def test_predict_from_counts_agrees_with_the_manual_wrapper(self):
        # The lower-level readout entry too, so the equivalence is not specific
        # to `predict`'s backend plumbing.
        result = _native_train()
        manual = self._manual(result)
        for counts in ({"00": 10}, {"01": 10}, {"00": 7, "11": 3}):
            assert result.trained_model.predict_from_counts(
                counts
            ) == manual.predict_from_counts(counts)

    def test_trained_model_round_trips_through_save_load(self, tmp_path):
        # It is a fully-fledged `TrainedModel`: serialization works unchanged.
        import polypus

        result = _native_train()
        path = str(tmp_path / "auto.json")
        result.trained_model.save(path)
        assert polypus.qml.TrainedModel.load(path).theta == result.best_params

    def test_minibatched_run_also_returns_a_usable_trained_model(self):
        # `batch_size` reaches the same single return point, and the θ it wraps is
        # the recomputed-fitness `best_params`, not a minibatch artefact.
        import polypus

        result = _native_train(seed=5, batch_size=2)
        assert isinstance(result, polypus.qml.QmlTrainResult)
        assert result.trained_model.theta == result.best_params

    def test_failed_run_propagates_the_original_error(self):
        # An `Err` out of the optimizer dispatch must surface unchanged — the
        # upgrade to `QmlTrainResult` never runs, and never masks the cause.
        import polypus

        with pytest.raises(TypeError, match="method must be an instance"):
            polypus.qml.train(
                _model(),
                _dataset(),
                method="not-an-optimizer",
                loss="hinge",
                infrastructure="local",
                backend="polypus",
                id="qml_train_result_bad_method",
                seed=7,
                exact=True,
            )


@pytest.mark.integration
@pytest.mark.vqc
class TestQiskitPathStillReturnsTrainResult:
    """The Qiskit path has no `Model`/`Dataset` to wrap, so it keeps returning a
    plain `TrainResult`. Pinned explicitly so a future change cannot quietly
    merge the two paths' return shapes."""

    def test_qiskit_path_returns_a_plain_train_result(self):
        import numpy as np
        import polypus
        from qiskit.circuit.library import real_amplitudes, zz_feature_map

        feature_map = zz_feature_map(feature_dimension=2, reps=1)
        ansatz = real_amplitudes(num_qubits=2, reps=1)
        result = polypus.qml.train(
            feature_map,
            ansatz,
            np.array([[0.3, 0.35], [2.8, 2.75]]),
            polypus.DE(generations=2, population_size=4),
            dimensions=len(ansatz.parameters),
            expectation_function=lambda b: sum(int(c) for c in b) / len(b),
            shots=256,
            infrastructure="local",
            backend="aer",
            id="qml_train_result_qiskit",
            seed=7,
        )
        assert isinstance(result, polypus.TrainResult)
        assert type(result).__name__ == "TrainResult"
        assert not hasattr(result, "trained_model")
        # And the generic `polypus.train` is untouched as well.
        assert not isinstance(result, polypus.qml.QmlTrainResult)
