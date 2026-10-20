"""
``polypus.CMAES``: CMA-ES with optional IPOP restarts (contract C-5).

The option tests need no backend. The batch tests run the real
``polypus.train`` / ``polypus.qml.train`` loops on Aer with
``polypus_python.run_qcs`` replaced by a fake that records every batch the
optimizer sends and answers with fixed counts. Seeded reproducibility is in
``test_seed_reproducibility.py`` (contract C-7) and the non-finite fitness
cases in ``test_oracle_contract.py``. The last class runs a seeded QAOA MaxCut
on the native simulator end to end.
"""

import math
import re

import pytest

_ANGLE = re.compile(r"^ry\(([^)]*)\) q\[0\];$", re.MULTILINE)


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


def _train(method, *, seed=11, ident="cmaes", dimensions=1):
    import polypus

    qc = polypus.Circuit(dimensions)
    for q in range(dimensions):
        qc.ry(q, polypus.Param(q))
    qc.measure_all()
    return polypus.train(
        qc,
        method,
        shots=64,
        n_qpus=1,
        dimensions=dimensions,
        expectation_function=_all_ones,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id=ident,
        backend="aer",
        seed=seed,
    )


def _qml_train(method, *, seed=11):
    import numpy as np
    import polypus
    from qiskit.circuit.library import real_amplitudes, zz_feature_map

    feature_map = zz_feature_map(feature_dimension=2, reps=1)
    ansatz = real_amplitudes(num_qubits=2, reps=1)
    return polypus.qml.train(
        feature_map,
        ansatz,
        np.zeros((2, 2)),
        method,
        shots=64,
        n_qpus=1,
        dimensions=len(ansatz.parameters),
        expectation_function=lambda b: sum(int(c) for c in b) / len(b),
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="cmaes_qml",
        seed=seed,
    )


def _record_batches(monkeypatch):
    """Replace the local seam with a fake that records each batch of circuits
    (OpenQASM strings) and returns the same counts for every circuit, so every
    candidate gets the same fitness."""
    import polypus_python

    batches = []

    def fake_run_qcs(infrastructure, **kwargs):
        batches.append(list(kwargs["qcs"]))
        return [{"1": kwargs["shots"]} for _ in kwargs["qcs"]]

    monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)
    return batches


def _runs(sizes):
    """Consecutive equal batch sizes as ``(size, count)`` pairs: one per run."""
    runs = []
    for size in sizes:
        if runs and runs[-1][0] == size:
            runs[-1] = (size, runs[-1][1] + 1)
        else:
            runs.append((size, 1))
    return runs


class TestCmaesOptions:
    def test_defaults(self):
        import polypus

        cmaes = polypus.CMAES()
        assert cmaes.generations == 100
        assert cmaes.population_size is None
        assert cmaes.sigma0 == math.pi / 2
        assert cmaes.tolerance == pytest.approx(0.01)
        assert cmaes.patience == 20
        assert cmaes.restarts == 0
        assert cmaes.seed is None

    def test_keywords_and_positions(self):
        import polypus

        by_name = polypus.CMAES(
            generations=30,
            population_size=8,
            sigma0=0.5,
            tolerance=1e-3,
            patience=7,
            restarts=2,
            seed=5,
        )
        by_position = polypus.CMAES(30, 8, 0.5, 1e-3, 7, 2, 5)
        for cmaes in (by_name, by_position):
            assert (
                cmaes.generations,
                cmaes.population_size,
                cmaes.sigma0,
                cmaes.tolerance,
                cmaes.patience,
                cmaes.restarts,
                cmaes.seed,
            ) == (30, 8, 0.5, 1e-3, 7, 2, 5)

    def test_every_field_is_settable(self):
        import polypus

        cmaes = polypus.CMAES(population_size=8)
        cmaes.generations = 12
        cmaes.population_size = None
        cmaes.sigma0 = 1.0
        cmaes.tolerance = 0.5
        cmaes.patience = 3
        cmaes.restarts = 4
        cmaes.seed = 9
        assert (
            cmaes.generations,
            cmaes.population_size,
            cmaes.sigma0,
            cmaes.tolerance,
            cmaes.patience,
            cmaes.restarts,
            cmaes.seed,
        ) == (12, None, 1.0, 0.5, 3, 4, 9)

    @pytest.mark.parametrize(
        "kwargs", [{"generations": -1}, {"population_size": -1}, {"restarts": -1}]
    )
    def test_negative_counts_are_rejected_at_construction(self, kwargs):
        import polypus

        with pytest.raises(OverflowError):
            polypus.CMAES(**kwargs)

    def test_a_non_method_names_cmaes_in_its_type_error(self):
        with pytest.raises(TypeError, match="polypus.CMAES"):
            _train(object())


@pytest.mark.integration
@pytest.mark.vqc
class TestCmaesBatches:
    """What CMA-ES sends to the backend: one batch per generation and no
    initial batch, doubling at each IPOP restart, every angle wrapped into
    ``[0, 2π)``, and nothing at all for an invalid configuration."""

    def test_without_restarts_one_batch_per_generation(self, monkeypatch):
        import polypus

        batches = _record_batches(monkeypatch)
        # All fitness values are equal, so only a patience longer than the run
        # keeps the stagnation stop from ending it early.
        result = _train(polypus.CMAES(generations=6, patience=10_000))
        # 4 + floor(3 ln 1) = 4 candidates for one parameter.
        assert [len(b) for b in batches] == [4] * 6
        assert result.iterations_run == 6
        assert not result.converged
        assert len(result.fitness_history) == result.iterations_run
        assert result.fitness_history[-1] == result.best_fitness

    def test_population_size_sets_the_batch_size(self, monkeypatch):
        import polypus

        batches = _record_batches(monkeypatch)
        _train(polypus.CMAES(generations=3, population_size=7, patience=10_000))
        assert [len(b) for b in batches] == [7, 7, 7]

    def test_default_population_follows_the_dimensions(self, monkeypatch):
        import polypus

        batches = _record_batches(monkeypatch)
        _train(polypus.CMAES(generations=2, patience=10_000), dimensions=3)
        # 4 + floor(3 ln 3) = 7.
        assert [len(b) for b in batches] == [7, 7]

    def test_restarts_double_the_population_then_stop(self, monkeypatch):
        import polypus

        batches = _record_batches(monkeypatch)
        # On the flat landscape every run stagnates as soon as it has a full
        # window: patience 2 gives 3 generations per run.
        result = _train(
            polypus.CMAES(generations=100, tolerance=1e-9, patience=2, restarts=2)
        )
        assert _runs([len(b) for b in batches]) == [(4, 3), (8, 3), (16, 3)]
        assert result.converged
        assert result.iterations_run == 9
        assert len(result.fitness_history) == 9

    def test_restarts_share_the_generation_budget(self, monkeypatch):
        import polypus

        batches = _record_batches(monkeypatch)
        result = _train(
            polypus.CMAES(generations=5, tolerance=1e-9, patience=2, restarts=3)
        )
        assert _runs([len(b) for b in batches]) == [(4, 3), (8, 2)]
        assert not result.converged
        assert result.iterations_run == 5

    def test_every_angle_reaches_the_backend_wrapped(self, monkeypatch):
        import polypus

        batches = _record_batches(monkeypatch)
        # The widest step spreads the unwrapped samples over several turns.
        result = _train(
            polypus.CMAES(generations=5, sigma0=2 * math.pi, patience=10_000)
        )
        angles = [
            float(a) for batch in batches for qc in batch for a in _ANGLE.findall(qc)
        ]
        assert len(angles) == 4 * 5
        assert all(0.0 <= a < 2 * math.pi for a in angles), angles
        assert 0.0 <= result.best_params[0] < 2 * math.pi

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"sigma0": 0.0}, "sigma0"),
            ({"sigma0": -1.0}, "sigma0"),
            ({"sigma0": 7.0}, "sigma0"),
            ({"sigma0": float("nan")}, "sigma0"),
            ({"sigma0": float("inf")}, "sigma0"),
            ({"population_size": 0}, "population_size"),
            ({"population_size": 1}, "population_size"),
            ({"generations": 0}, "generations"),
            ({"population_size": 8, "restarts": 40}, "restarts"),
        ],
    )
    def test_invalid_configuration_raises_before_any_batch(
        self, monkeypatch, kwargs, message
    ):
        import polypus

        batches = _record_batches(monkeypatch)
        with pytest.raises(ValueError, match=message):
            _train(polypus.CMAES(**kwargs), ident="cmaes_invalid")
        assert batches == []

    def test_qml_train_runs_cmaes(self, monkeypatch):
        import polypus

        batches = _record_batches(monkeypatch)
        result = _qml_train(polypus.CMAES(generations=3, patience=10_000))
        assert result.iterations_run == 3
        assert len(result.best_params) == 4
        assert all(0.0 <= p < 2 * math.pi for p in result.best_params)
        assert batches

    def test_qml_train_rejects_an_invalid_configuration_before_any_batch(
        self, monkeypatch
    ):
        import polypus

        batches = _record_batches(monkeypatch)
        with pytest.raises(ValueError, match="sigma0"):
            _qml_train(polypus.CMAES(sigma0=-1.0))
        assert batches == []


def _ring_cut(bitstring):
    """MaxCut value of a bitstring on the 6-node ring (maximum 6)."""
    n = len(bitstring)
    return float(sum(bitstring[i] != bitstring[(i + 1) % n] for i in range(n)))


def _qaoa_ring(nodes, layers, measure):
    """QAOA MaxCut on a ring: ``rzz(γ)`` per edge, ``rx(2β)`` per qubit;
    parameters ``[γ_0, β_0, γ_1, β_1, ...]``."""
    import polypus

    qc = polypus.Circuit(nodes)
    for q in range(nodes):
        qc.h(q)
    for layer in range(layers):
        gamma = polypus.Param(2 * layer)
        beta = polypus.Param(2 * layer + 1)
        for q in range(nodes):
            qc.rzz(q, (q + 1) % nodes, gamma)
        for q in range(nodes):
            qc.rx(q, 2 * beta)
    if measure:
        qc.measure_all()
    return qc


@pytest.mark.integration
@pytest.mark.vqc
class TestCmaesQaoaMaxCut:
    """A seeded QAOA MaxCut through ``polypus.train`` on the native simulator:
    CMA-ES must find parameters well beyond random guessing. The check uses
    the exact expected cut of the returned parameters, from the statevector,
    so shot noise in ``best_fitness`` cannot pass it."""

    NODES = 6
    LAYERS = 2

    def _expected_cut(self, params):
        import numpy as np
        import polypus

        state = polypus.statevector(
            _qaoa_ring(self.NODES, self.LAYERS, measure=False), params
        )
        probabilities = np.abs(state) ** 2
        cuts = [
            _ring_cut(format(k, f"0{self.NODES}b")) for k in range(len(probabilities))
        ]
        return float(np.dot(probabilities, cuts))

    def test_seeded_qaoa_maxcut_converges(self):
        import polypus

        result = polypus.train(
            _qaoa_ring(self.NODES, self.LAYERS, measure=True),
            polypus.CMAES(generations=60, restarts=1),
            shots=512,
            n_qpus=2,
            dimensions=2 * self.LAYERS,
            expectation_function=_ring_cut,
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id="cmaes_qaoa",
            backend="polypus",
            seed=2026,
        )
        assert result.seed == 2026
        assert len(result.best_params) == 2 * self.LAYERS
        assert all(0.0 <= p < 2 * math.pi for p in result.best_params)
        assert result.fitness_history[-1] == result.best_fitness
        # A uniformly random cut averages 3 of the 6 edges (ratio 0.5); p = 1
        # QAOA reaches 0.75 and p = 2 about 0.83. With 512 shots, seeds 0-29
        # gave 0.77 to 0.83 here; the margin absorbs another platform's
        # trajectory (CMA-ES is reproducible per platform only).
        ratio = self._expected_cut(result.best_params) / self.NODES
        assert ratio > 0.74, ratio
