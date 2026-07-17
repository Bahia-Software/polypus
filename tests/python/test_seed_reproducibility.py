"""
Seed reproducibility & run manifest — public-API end-to-end tests (contract C-7).

These exercise issue #34 / #56's acceptance criteria from Python:

* ``run_quantum_circuit(..., seed=...)`` on every *simulated* backend (native
  ``"polypus"`` and Aer, both under ``infrastructure="local"``): same seed ⇒
  identical counts; no seed ⇒ different counts; the effective seed and manifest
  are returned. A seed is rejected for ``infrastructure="qmio"`` (real hardware).
* ``train`` / ``qml.train`` seed: same seed ⇒ identical outcome; no seed ⇒
  different outcome; the outcome now exposes fitness / iterations / convergence
  / seed, and the ``DE``/``PSO``/``QNG`` ``seed`` field feeds the precedence.

CUNQA's simulated QPUs (``infrastructure="cunqa"``) attempt the same kind of
seed forwarding as Aer (`crates/polypus/src/infrastructure/cunqa.rs` mirrors
`local.rs`), but this is unverified, not just untested: the `cunqa` package
isn't installed anywhere in this environment/CI, and reading the actual
CESGA-Quantum-Spain/cunqa source at the version README.md pins turned up API
mismatches predating this seed work (wrong import path, wrong kwarg names, a
`QPU.run()` method that may not exist at that version) — see the CUNQA
integration follow-up. There is no dedicated test here for the same reason
QMIO's `real_qpu_smoke` test is marked `ignored`: no live infrastructure.

The native backend is fully seeded by Polypus, so its reproducibility tests need
no mocking; Aer is genuinely seeded too (via `seed_simulator`, forwarded across
the C-1 seam), so its tests run for real as well. ``qml.train`` additionally
mocks ``polypus_python.run_qcs`` to make the oracle's expectation values fixed
regardless of the (still-seeded) Aer sampling underneath, isolating the
assertions to the optimizer RNG that ``seed`` controls.
"""

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# run_quantum_circuit — native and Aer are seeded; qmio (real hardware) rejects
# ─────────────────────────────────────────────────────────────────────────────


def _uniform3():
    """3-qubit uniform superposition: eight outcomes, so two independent
    samplings differ with overwhelming probability (non-flaky inequality)."""
    import polypus

    return polypus.Circuit(3).h(0).h(1).h(2).measure_all()


@pytest.mark.integration
class TestRunQuantumCircuitSeed:
    @pytest.mark.parametrize("backend_name", ["polypus", "aer"])
    def test_same_seed_reproduces_counts(self, backend_name):
        import polypus

        qc = _uniform3()
        r1 = polypus.run_quantum_circuit(
            qc, shots=2000, infrastructure="local", backend=backend_name, seed=42
        )
        r2 = polypus.run_quantum_circuit(
            qc, shots=2000, infrastructure="local", backend=backend_name, seed=42
        )
        assert r1.seed == 42 and r2.seed == 42
        assert r1.counts == r2.counts

    @pytest.mark.parametrize("backend_name", ["polypus", "aer"])
    def test_no_seed_differs_across_calls(self, backend_name):
        import polypus

        qc = _uniform3()
        r1 = polypus.run_quantum_circuit(
            qc, shots=2000, infrastructure="local", backend=backend_name
        )
        r2 = polypus.run_quantum_circuit(
            qc, shots=2000, infrastructure="local", backend=backend_name
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
        # The id keeps the human-readable `run_{n_qpus}_{infra}_` prefix but is
        # suffixed with a per-call UUID for uniqueness (see test_id_is_unique),
        # so only the stable prefix is asserted here.
        assert r.id.startswith("run_1_local_")
        assert isinstance(r.counts, list) and len(r.counts) == 1

    def test_id_is_unique_across_identical_calls(self):
        # Regression for #45: the auto-generated run id must be unique per call.
        # Two runs with identical arguments used to collide on `run_{n}_{infra}`,
        # which names SLURM families/allocations, temp files and log streams.
        import polypus

        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        kwargs = dict(shots=100, infrastructure="local", backend="polypus", seed=7)
        r1 = polypus.run_quantum_circuit(qc, **kwargs)
        r2 = polypus.run_quantum_circuit(qc, **kwargs)
        assert r1.id.startswith("run_1_local_")
        assert r2.id.startswith("run_1_local_")
        assert r1.id != r2.id

    def test_seed_rejected_for_qmio(self):
        import polypus

        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        with pytest.raises(ValueError, match="seed is not supported"):
            polypus.run_quantum_circuit(qc, shots=100, infrastructure="qmio", seed=1)

    def test_aer_without_seed_reports_entropy_seed(self):
        import polypus

        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        r = polypus.run_quantum_circuit(
            qc, shots=100, infrastructure="local", backend="aer"
        )
        assert isinstance(r.seed, int)
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

    def test_id_is_unique_across_identical_calls(self):
        # Regression for #75: two training runs with byte-identical arguments —
        # crucially the same id="..." — must not share an effective id. The id
        # names SLURM families/allocations, temp files and log streams, so a
        # collision could make two concurrent runs clobber one another. The
        # supplied string is kept as a prefix and a UUID v4 is appended, so the
        # returned TrainResult.id differs from what was passed in.
        r1 = self._train(seed=7, ident="train_id")
        r2 = self._train(seed=7, ident="train_id")
        assert r1.id.startswith("train_id_")
        assert r2.id.startswith("train_id_")
        assert r1.id != r2.id


# ─────────────────────────────────────────────────────────────────────────────
# qml.train — optimizer RNG and Aer sampling are both seeded; mock the backend
# to isolate the optimizer RNG assertions from Aer's own (also-seeded) noise
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vqc
class TestQmlTrainSeed:
    @staticmethod
    def _patch_deterministic_backend(monkeypatch):
        import polypus_python

        # Fixed counts regardless of the circuit ⇒ a deterministic oracle, so
        # these assertions isolate the optimizer RNG that qml.train's seed
        # controls, independent of Aer's own (now also seeded, C-7) sampling.
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

    def test_id_is_unique_across_identical_calls(self, monkeypatch):
        # Regression for #75, qml.train side: two runs with byte-identical
        # arguments — including the same id="qml_seed" — must not share an
        # effective id (which names the SLURM family/allocation, temp files and
        # log streams). The supplied string is kept as a prefix; a UUID v4 is
        # appended for uniqueness.
        self._patch_deterministic_backend(monkeypatch)
        r1 = self._qml_train(7)
        r2 = self._qml_train(7)
        assert r1.id.startswith("qml_seed_")
        assert r2.id.startswith("qml_seed_")
        assert r1.id != r2.id
