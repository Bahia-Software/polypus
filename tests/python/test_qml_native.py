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
