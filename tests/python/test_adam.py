"""
``polypus.Adam``: gradient ascent with Kingma and Ba's Adam (contract C-5).

Adam takes a central finite difference of the fitness: each iteration sends
one batch of ``2·dimensions + 1`` circuits, ``θ`` and then ``θ ± h·e_i``. The
option tests need no backend. The batch tests run the real loops on Aer with
``polypus_python.run_qcs`` replaced by a fake that records every batch and
answers with fixed counts. Seeded reproducibility is in
``test_seed_reproducibility.py`` (contract C-7) and the non-finite fitness cases
in ``test_oracle_contract.py``. The last class trains a one-qubit rotation on
the native simulator end to end.
"""

import math
import re

import pytest

_ANGLE = re.compile(r"^ry\(([^)]*)\) q\[0\];$", re.MULTILINE)


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


def _train(method, *, seed=11, ident="adam", dimensions=1, **kwargs):
    import polypus

    qc = polypus.Circuit(dimensions)
    for q in range(dimensions):
        qc.ry(q, polypus.Param(q))
    qc.measure_all()
    options = {"shots": 64, "backend": "aer"}
    options.update(kwargs)
    return polypus.train(
        qc,
        method,
        n_qpus=1,
        dimensions=dimensions,
        expectation_function=_all_ones,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id=ident,
        seed=seed,
        **options,
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
        id="adam_qml",
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


class TestAdamOptions:
    def test_defaults(self):
        import polypus

        adam = polypus.Adam()
        assert adam.max_iters == 100
        assert adam.learning_rate == pytest.approx(0.05)
        assert adam.beta1 == pytest.approx(0.9)
        assert adam.beta2 == pytest.approx(0.999)
        assert adam.epsilon == pytest.approx(1e-8)
        assert adam.bounds == (-math.pi, math.pi)
        assert adam.tolerance == pytest.approx(0.01)
        assert adam.patience == 20
        assert adam.finite_difference_step == pytest.approx(0.1)
        assert adam.seed is None

    def test_keywords_and_positions(self):
        import polypus

        values = (30, 0.2, 0.8, 0.99, 1e-6, (-1.0, 2.0), 1e-3, 7, 0.05, 5)
        by_name = polypus.Adam(
            max_iters=30,
            learning_rate=0.2,
            beta1=0.8,
            beta2=0.99,
            epsilon=1e-6,
            bounds=(-1.0, 2.0),
            tolerance=1e-3,
            patience=7,
            finite_difference_step=0.05,
            seed=5,
        )
        by_position = polypus.Adam(*values)
        for adam in (by_name, by_position):
            assert (
                adam.max_iters,
                adam.learning_rate,
                adam.beta1,
                adam.beta2,
                adam.epsilon,
                adam.bounds,
                adam.tolerance,
                adam.patience,
                adam.finite_difference_step,
                adam.seed,
            ) == values

    def test_every_field_is_settable(self):
        import polypus

        adam = polypus.Adam()
        adam.max_iters = 12
        adam.learning_rate = 0.3
        adam.beta1 = 0.5
        adam.beta2 = 0.9
        adam.epsilon = 1e-4
        adam.bounds = (0.0, 1.0)
        adam.tolerance = 0.5
        adam.patience = 3
        adam.finite_difference_step = 0.2
        adam.seed = 9
        assert (
            adam.max_iters,
            adam.learning_rate,
            adam.beta1,
            adam.beta2,
            adam.epsilon,
            adam.bounds,
            adam.tolerance,
            adam.patience,
            adam.finite_difference_step,
            adam.seed,
        ) == (12, 0.3, 0.5, 0.9, 1e-4, (0.0, 1.0), 0.5, 3, 0.2, 9)

    @pytest.mark.parametrize("kwargs", [{"max_iters": -1}, {"patience": -1}])
    def test_negative_counts_are_rejected_at_construction(self, kwargs):
        import polypus

        with pytest.raises(OverflowError):
            polypus.Adam(**kwargs)

    def test_a_non_method_names_adam_in_its_type_error(self):
        with pytest.raises(TypeError, match="polypus.Adam"):
            _train(object())


@pytest.mark.integration
@pytest.mark.vqc
class TestAdamBatches:
    """What Adam sends to the backend: per iteration, one batch of ``θ`` and
    the finite-difference stencil around it, and nothing at all for an invalid
    configuration."""

    def test_each_iteration_is_one_batch_of_theta_and_its_stencil(self, monkeypatch):
        import polypus

        batches = _record_batches(monkeypatch)
        # All fitness values are equal, so only a patience longer than the run
        # keeps the stagnation stop from ending it early.
        result = _train(polypus.Adam(max_iters=4, patience=10_000), dimensions=3)
        assert [len(b) for b in batches] == [7] * 4
        assert result.iterations_run == 4
        assert not result.converged
        assert len(result.fitness_history) == result.iterations_run
        assert result.fitness_history[-1] == result.best_fitness

    def test_theta_comes_first_and_the_stencil_is_centred_on_it(self, monkeypatch):
        import polypus

        batches = _record_batches(monkeypatch)
        _train(polypus.Adam(max_iters=2, finite_difference_step=0.25, patience=10_000))
        angles = [[float(a) for qc in b for a in _ANGLE.findall(qc)] for b in batches]
        # Each iteration's batch is θ, then θ + h and θ − h around it.
        assert len(angles) == 2
        for theta, plus, minus in angles:
            assert plus - minus == pytest.approx(0.5)
            assert (plus + minus) / 2 == pytest.approx(theta)

    def test_a_flat_fitness_stagnates_and_converges(self, monkeypatch):
        import polypus

        _record_batches(monkeypatch)
        result = _train(polypus.Adam(max_iters=100, tolerance=1e-9, patience=2))
        assert result.converged
        assert result.iterations_run == 3

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_iters": 0}, "max_iters"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"learning_rate": float("nan")}, "learning_rate"),
            ({"beta1": 1.0}, "beta1"),
            ({"beta2": -0.1}, "beta2"),
            ({"epsilon": 0.0}, "epsilon"),
            ({"bounds": (1.0, 1.0)}, "bound"),
            ({"bounds": (0.0, float("inf"))}, "bound"),
            ({"finite_difference_step": 0.0}, "finite_difference_step"),
            ({"finite_difference_step": float("nan")}, "finite_difference_step"),
        ],
    )
    def test_invalid_configuration_raises_before_any_batch(
        self, monkeypatch, kwargs, message
    ):
        import polypus

        batches = _record_batches(monkeypatch)
        with pytest.raises(ValueError, match=message):
            _train(polypus.Adam(**kwargs), ident="adam_invalid")
        assert batches == []

    def test_qml_train_runs_adam(self, monkeypatch):
        import polypus

        batches = _record_batches(monkeypatch)
        result = _qml_train(polypus.Adam(max_iters=3, patience=10_000))
        assert result.iterations_run == 3
        assert len(result.best_params) == 4
        assert len(result.fitness_history) == 3
        assert batches

    def test_qml_train_rejects_an_invalid_configuration_before_any_batch(
        self, monkeypatch
    ):
        import polypus

        batches = _record_batches(monkeypatch)
        with pytest.raises(ValueError, match="learning_rate"):
            _qml_train(polypus.Adam(learning_rate=-1.0))
        assert batches == []


@pytest.mark.integration
@pytest.mark.vqc
class TestAdamClimbs:
    def test_seeded_rotation_reaches_the_all_ones_state(self):
        """``P(1) = sin²(θ/2)`` on one qubit: Adam climbs to ``θ ≈ ±π``."""
        import polypus

        result = _train(
            polypus.Adam(max_iters=150, learning_rate=0.1, patience=10_000),
            seed=2024,
            ident="adam_climb",
            shots=4096,
            backend="polypus",
        )
        assert result.best_fitness > 0.95
        assert math.cos(result.best_params[0]) < -0.9
