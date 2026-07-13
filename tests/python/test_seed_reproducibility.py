"""
Seed reproducibility & run manifest — public-API end-to-end tests (contract C-7).

These exercise issue #34's acceptance criteria from Python:

* ``run_quantum_circuit(..., seed=...)`` on the native backend: same seed ⇒
  identical counts; no seed ⇒ different counts; the effective seed and manifest
  are returned; a seed with any non-native backend is rejected.
* ``train`` / ``qml.train`` seed: same seed ⇒ identical outcome; no seed ⇒
  different outcome; the outcome now exposes fitness / iterations / convergence
  / seed, and the ``DE``/``PSO``/``QNG`` ``seed`` field feeds the precedence.

The native backend is fully seeded by Polypus, so its reproducibility tests need
no mocking. ``qml.train`` runs on Qiskit/Aer (native rejected) and Aer's own shot
sampling is not seeded by Polypus, so that test mocks ``polypus_python.run_qcs``
to make the oracle deterministic and isolate the optimizer RNG the seed controls.
"""

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# run_quantum_circuit — native backend is seeded; other backends reject a seed
# ─────────────────────────────────────────────────────────────────────────────


def _uniform3():
    """3-qubit uniform superposition: eight outcomes, so two independent
    samplings differ with overwhelming probability (non-flaky inequality)."""
    import polypus

    return polypus.Circuit(3).h(0).h(1).h(2).measure_all()


@pytest.mark.integration
class TestRunQuantumCircuitSeed:
    def test_same_seed_reproduces_counts(self):
        import polypus

        qc = _uniform3()
        r1 = polypus.run_quantum_circuit(
            qc, shots=2000, infrastructure="local", backend="polypus", seed=42
        )
        r2 = polypus.run_quantum_circuit(
            qc, shots=2000, infrastructure="local", backend="polypus", seed=42
        )
        assert r1.seed == 42 and r2.seed == 42
        assert r1.counts == r2.counts

    def test_no_seed_differs_across_calls(self):
        import polypus

        qc = _uniform3()
        r1 = polypus.run_quantum_circuit(
            qc, shots=2000, infrastructure="local", backend="polypus"
        )
        r2 = polypus.run_quantum_circuit(
            qc, shots=2000, infrastructure="local", backend="polypus"
        )
        # An entropy seed is generated, reported, and fresh on every call.
        assert isinstance(r1.seed, int) and isinstance(r2.seed, int)
        assert r1.seed != r2.seed
        assert r1.counts != r2.counts

    def test_manifest_fields(self):
        import polypus

        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        r = polypus.run_quantum_circuit(
            qc, shots=100, infrastructure="local", backend="polypus", seed=7
        )
        assert r.seed == 7
        assert r.backend == "polypus"
        assert r.infrastructure == "local"
        assert r.id == "run_1_local"
        assert isinstance(r.counts, list) and len(r.counts) == 1

    def test_seed_rejected_for_aer(self):
        import polypus

        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        with pytest.raises(ValueError, match="seed is only supported"):
            polypus.run_quantum_circuit(
                qc, shots=100, infrastructure="local", backend="aer", seed=1
            )

    def test_seed_rejected_for_qmio(self):
        import polypus

        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        with pytest.raises(ValueError, match="seed is only supported"):
            polypus.run_quantum_circuit(
                qc, shots=100, infrastructure="qmio", seed=1
            )

    def test_aer_without_seed_reports_none(self):
        import polypus

        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        r = polypus.run_quantum_circuit(qc, shots=100, infrastructure="local", backend="aer")
        assert r.seed is None
        assert r.backend == "aer"


# ─────────────────────────────────────────────────────────────────────────────
# DE / PSO / QNG `seed` field (no backend needed)
# ─────────────────────────────────────────────────────────────────────────────


class TestOptimizerSeedField:
    def test_defaults_to_none(self):
        import polypus

        assert polypus.DE().seed is None
        assert polypus.PSO().seed is None
        assert polypus.QNG(lambda theta, a: 0.5).seed is None

    def test_getter_setter_roundtrip(self):
        import polypus

        de = polypus.DE(seed=5)
        assert de.seed == 5
        de.seed = 8
        assert de.seed == 8


# ─────────────────────────────────────────────────────────────────────────────
# train — native backend is fully reproducible from a single seed
# ─────────────────────────────────────────────────────────────────────────────


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


@pytest.mark.integration
@pytest.mark.vqc
class TestTrainSeed:
    @staticmethod
    def _train(seed=None, method=None, ident="train_seed"):
        import polypus

        qc = polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()
        return polypus.train(
            qc,
            method or polypus.DE(generations=5, population_size=8, tolerance=1e-12),
            shots=256,
            n_qpus=1,
            dimensions=1,
            expectation_function=_all_ones,
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id=ident,
            backend="polypus",
            seed=seed,
        )

    def test_same_seed_reproduces_outcome(self):
        r1 = self._train(seed=2024)
        r2 = self._train(seed=2024)
        assert r1.seed == 2024 and r2.seed == 2024
        assert r1.best_params == r2.best_params
        assert r1.best_fitness == r2.best_fitness
        assert r1.iterations_run == r2.iterations_run
        assert r1.converged == r2.converged
        # The full outcome is exposed now, not just the parameters (C-7).
        assert isinstance(r1.best_fitness, float)
        assert isinstance(r1.iterations_run, int)
        assert isinstance(r1.converged, bool)

    def test_no_seed_differs_across_calls(self):
        r1 = self._train(seed=None, ident="train_seed_none")
        r2 = self._train(seed=None, ident="train_seed_none")
        assert r1.seed != r2.seed
        assert r1.best_params != r2.best_params

    def test_optimizer_field_seed_is_used(self):
        import polypus

        # No kwarg seed → the seed pinned on the optimizer object drives the run.
        m1 = polypus.DE(generations=5, population_size=8, tolerance=1e-12, seed=99)
        m2 = polypus.DE(generations=5, population_size=8, tolerance=1e-12, seed=99)
        r1 = self._train(seed=None, method=m1, ident="train_field_seed")
        r2 = self._train(seed=None, method=m2, ident="train_field_seed")
        assert r1.seed == 99 and r2.seed == 99
        assert r1.best_params == r2.best_params

    def test_kwarg_seed_overrides_field(self):
        import polypus

        # The explicit kwarg wins over the optimizer object's field.
        m = polypus.DE(generations=5, population_size=8, tolerance=1e-12, seed=1)
        r = self._train(seed=777, method=m, ident="train_override")
        assert r.seed == 777


# ─────────────────────────────────────────────────────────────────────────────
# qml.train — optimizer RNG is seeded; Aer sampling isn't, so mock the backend
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vqc
class TestQmlTrainSeed:
    @staticmethod
    def _patch_deterministic_backend(monkeypatch):
        import polypus_python

        # Fixed counts regardless of the circuit ⇒ a deterministic oracle, so the
        # only remaining randomness is the optimizer RNG that qml.train's seed
        # controls (Aer's own shot sampling is not seeded by Polypus; C-7).
        def fake_run_qcs(infrastructure, **kwargs):
            return [{"1": kwargs["shots"]} for _ in kwargs["qcs"]]

        monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)

    @staticmethod
    def _qml_train(seed):
        import numpy as np
        import polypus
        from qiskit.circuit.library import real_amplitudes, zz_feature_map

        feature_map = zz_feature_map(feature_dimension=2, reps=1)
        ansatz = real_amplitudes(num_qubits=2, reps=1)
        x_train = np.zeros((2, 2))
        return polypus.qml.train(
            feature_map,
            ansatz,
            x_train,
            polypus.DE(generations=3, population_size=6, tolerance=1e-12),
            shots=64,
            n_qpus=1,
            dimensions=len(ansatz.parameters),
            expectation_function=lambda b: sum(int(c) for c in b) / len(b),
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id="qml_seed",
            seed=seed,
        )

    def test_same_seed_reproduces_outcome(self, monkeypatch):
        self._patch_deterministic_backend(monkeypatch)
        r1 = self._qml_train(123)
        r2 = self._qml_train(123)
        assert r1.seed == 123 and r2.seed == 123
        assert r1.best_params == r2.best_params
        assert r1.best_fitness == r2.best_fitness

    def test_no_seed_differs_across_calls(self, monkeypatch):
        self._patch_deterministic_backend(monkeypatch)
        r1 = self._qml_train(None)
        r2 = self._qml_train(None)
        assert r1.seed != r2.seed
        assert r1.best_params != r2.best_params
