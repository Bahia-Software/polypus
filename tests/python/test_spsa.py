"""
``polypus.SPSA``: Spall's simultaneous perturbation stochastic approximation
(contract C-5).

SPSA starts with one calibration batch of 50 circuits, 25 pairs around the
starting point, then sends one batch per iteration: ``2·resamplings`` perturbed
circuits, headed by ``θ`` itself on the first iteration and every
``eval_interval`` iterations after it, and after the last step one circuit
measures the final ``θ``. The option tests need no
backend. The batch tests run the real loops on Aer with
``polypus_python.run_qcs`` replaced by a fake that records every batch and
answers with fixed counts. Seeded reproducibility is in
``test_seed_reproducibility.py`` (contract C-7) and the non-finite fitness cases
in ``test_oracle_contract.py``. The last class trains a one-qubit rotation on
the native simulator end to end.
"""

import math
import re

import pytest

_ANGLE = re.compile(r"^ry\(([^)]*)\) q\[(\d+)\];$", re.MULTILINE)


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


def _train(method, *, seed=11, ident="spsa", dimensions=1, **kwargs):
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
        id="spsa_qml",
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


def _candidate(qasm):
    """The candidate a recorded ``train`` circuit was bound to."""
    angles = {int(q): float(a) for a, q in _ANGLE.findall(qasm)}
    return [angles[q] for q in range(len(angles))]


class TestSPSAOptions:
    def test_defaults(self):
        import polypus

        spsa = polypus.SPSA()
        assert spsa.max_iters == 300
        assert spsa.resamplings == 1
        assert spsa.perturbation == pytest.approx(0.1)
        assert spsa.target_step == pytest.approx(0.63)
        assert spsa.alpha == pytest.approx(0.602)
        assert spsa.gamma == pytest.approx(0.101)
        assert spsa.eval_interval == 20
        assert spsa.seed is None

    def test_keywords_and_positions(self):
        import polypus

        values = (30, 8, 0.2, 1.2, 1.0, 0.25, 5, 7)
        by_name = polypus.SPSA(
            max_iters=30,
            resamplings=8,
            perturbation=0.2,
            target_step=1.2,
            alpha=1.0,
            gamma=0.25,
            eval_interval=5,
            seed=7,
        )
        by_position = polypus.SPSA(*values)
        for spsa in (by_name, by_position):
            assert (
                spsa.max_iters,
                spsa.resamplings,
                spsa.perturbation,
                spsa.target_step,
                spsa.alpha,
                spsa.gamma,
                spsa.eval_interval,
                spsa.seed,
            ) == values

    def test_every_field_is_settable(self):
        import polypus

        spsa = polypus.SPSA()
        spsa.max_iters = 12
        spsa.resamplings = 4
        spsa.perturbation = 0.3
        spsa.target_step = 0.5
        spsa.alpha = 0.9
        spsa.gamma = 0.2
        spsa.eval_interval = 3
        spsa.seed = 9
        assert (
            spsa.max_iters,
            spsa.resamplings,
            spsa.perturbation,
            spsa.target_step,
            spsa.alpha,
            spsa.gamma,
            spsa.eval_interval,
            spsa.seed,
        ) == (12, 4, 0.3, 0.5, 0.9, 0.2, 3, 9)

    @pytest.mark.parametrize(
        "kwargs", [{"max_iters": -1}, {"resamplings": -1}, {"eval_interval": -1}]
    )
    def test_negative_counts_are_rejected_at_construction(self, kwargs):
        import polypus

        with pytest.raises(OverflowError):
            polypus.SPSA(**kwargs)

    def test_a_non_method_names_spsa_in_its_type_error(self):
        with pytest.raises(TypeError, match="polypus.SPSA"):
            _train(object())


@pytest.mark.integration
@pytest.mark.vqc
class TestSPSABatches:
    """What SPSA sends to the backend: one calibration batch, then per
    iteration one batch of perturbed pairs, headed by ``θ`` when it is
    measured, then the final ``θ`` alone, and nothing at all for an invalid
    configuration."""

    @pytest.mark.parametrize(
        ("resamplings", "sizes"),
        [
            # Iterations 1, 3 and 5 (the first and every 2nd after it)
            # measure θ, then the final θ is one circuit.
            (1, [50, 3, 2, 3, 2, 3, 1]),
            (3, [50, 7, 6, 7, 6, 7, 1]),
        ],
    )
    def test_calibration_then_one_batch_per_iteration(
        self, monkeypatch, resamplings, sizes
    ):
        import polypus

        batches = _record_batches(monkeypatch)
        method = polypus.SPSA(max_iters=5, resamplings=resamplings, eval_interval=2)
        result = _train(method, dimensions=3)
        assert [len(b) for b in batches] == sizes
        assert result.iterations_run == 5
        assert not result.converged
        assert len(result.fitness_history) == result.iterations_run
        assert result.fitness_history[-1] == result.best_fitness

    def test_the_final_theta_is_measured_whatever_the_interval(self, monkeypatch):
        import polypus

        batches = _record_batches(monkeypatch)
        _train(polypus.SPSA(max_iters=4), dimensions=2)
        # eval_interval 20: only the first iteration measures θ, and the
        # final θ is measured after the last step.
        assert [len(b) for b in batches] == [50, 3, 2, 2, 2, 1]

    def test_pairs_are_centred_on_theta_with_the_perturbation(self, monkeypatch):
        import polypus

        batches = _record_batches(monkeypatch)
        _train(
            polypus.SPSA(max_iters=2, perturbation=0.25, eval_interval=1),
            dimensions=2,
        )
        calibration, first, second, final = (
            [_candidate(qc) for qc in batch] for batch in batches
        )
        # The fake answers the same counts everywhere: the calibration finds a
        # flat landscape, every estimate is zero, and θ stays at θ₀.
        theta = first[0]
        assert second[0] == pytest.approx(theta, abs=1e-12)
        assert len(final) == 1
        assert final[0] == pytest.approx(theta, abs=1e-12)
        # c₁ = c, so the calibration and the first iteration perturb by c;
        # c₂ = c / 2^gamma.
        for pairs, c in [
            (calibration, 0.25),
            (first[1:], 0.25),
            (second[1:], 0.25 / 2**0.101),
        ]:
            for plus, minus in zip(pairs[::2], pairs[1::2]):
                assert [(a + b) / 2 for a, b in zip(plus, minus)] == pytest.approx(
                    theta, abs=1e-12
                )
                assert [abs(a - t) for a, t in zip(plus, theta)] == pytest.approx(
                    [c, c], abs=1e-12
                )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_iters": 0}, "max_iters"),
            ({"resamplings": 0}, "resamplings"),
            ({"perturbation": 0.0}, "perturbation"),
            ({"perturbation": float("nan")}, "perturbation"),
            ({"target_step": -1.0}, "target_step"),
            ({"target_step": float("inf")}, "target_step"),
            ({"alpha": 0.5}, "alpha"),
            ({"alpha": 1.1}, "alpha"),
            ({"gamma": 0.0}, "gamma"),
            ({"gamma": 0.2}, "gamma"),
            ({"eval_interval": 0}, "eval_interval"),
        ],
    )
    def test_invalid_configuration_raises_before_any_batch(
        self, monkeypatch, kwargs, message
    ):
        import polypus

        batches = _record_batches(monkeypatch)
        with pytest.raises(ValueError, match=message):
            _train(polypus.SPSA(**kwargs), ident="spsa_invalid")
        assert batches == []

    @pytest.mark.parametrize("resamplings", [1, 2])
    def test_qml_train_runs_spsa(self, monkeypatch, resamplings):
        import polypus

        batches = _record_batches(monkeypatch)
        result = _qml_train(polypus.SPSA(max_iters=3, resamplings=resamplings))
        assert result.iterations_run == 3
        assert len(result.best_params) == 4
        assert len(result.fitness_history) == 3
        assert not result.converged
        # The 50 calibration circuits, then 2K per iteration, θ on the first,
        # and the final θ, each on the 2 rows of x_train. (The QML oracle may
        # split a batch into windows, so only the total is fixed.)
        assert sum(len(b) for b in batches) == (50 + 3 * 2 * resamplings + 2) * 2

    def test_qml_train_rejects_an_invalid_configuration_before_any_batch(
        self, monkeypatch
    ):
        import polypus

        batches = _record_batches(monkeypatch)
        with pytest.raises(ValueError, match="gamma"):
            _qml_train(polypus.SPSA(alpha=0.6, gamma=0.1))
        assert batches == []


@pytest.mark.integration
@pytest.mark.vqc
class TestSPSAClimbs:
    @pytest.mark.parametrize("resamplings", [1, 4])
    def test_seeded_rotation_reaches_the_all_ones_state(self, resamplings):
        """``P(1) = sin²(θ/2)`` on one qubit: SPSA climbs to ``θ ≈ π``."""
        import polypus

        result = _train(
            polypus.SPSA(max_iters=100, resamplings=resamplings),
            seed=2024,
            ident="spsa_climb",
            shots=4096,
            backend="polypus",
        )
        assert result.best_fitness > 0.95
        assert math.cos(result.best_params[0]) < -0.9
        assert result.iterations_run == 100
        assert not result.converged
