"""
VQC training tests — verify that polypus.train works end-to-end with each
optimizer (DE, PSO, QNG) using a minimal 1-qubit parametrized circuit
and the local AerSimulator backend.

These tests are marked with both 'integration' and 'vqc'. They are intentionally
kept fast by using very small generation/iteration counts (≤ 3).

Skip all VQC tests:
    pytest -m "not vqc"

Skip all integration tests (including VQC):
    pytest -m "not integration"
"""

import math

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]

# Shared training parameters for all VQC tests — kept minimal for speed.
_SHOTS = 256
_N_QPUS = 1
_DIMENSIONS = 1
_NODES = 1
_CORES_PER_QPU = 1


class TestTrainDE:
    def test_train_returns_list(self, parametrized_circuit, simple_expectation_fn):
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.DE(generations=2, population_size=4, tolerance=0.5),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_de",
        )
        assert isinstance(result.best_params, list), (
            f"Expected list of parameters, got {type(result.best_params)}"
        )
        # The richer result also carries fitness / iterations / convergence /
        # seed (contract C-7), not just the parameters.
        assert isinstance(result.best_fitness, float)
        assert isinstance(result.iterations_run, int)
        assert isinstance(result.converged, bool)
        assert isinstance(result.seed, int)

    def test_train_result_length(self, parametrized_circuit, simple_expectation_fn):
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.DE(generations=2, population_size=4, tolerance=0.5),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_de_len",
        )
        assert len(result.best_params) == _DIMENSIONS

    def test_train_result_contains_floats(
        self, parametrized_circuit, simple_expectation_fn
    ):
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.DE(generations=2, population_size=4, tolerance=0.5),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_de_floats",
        )
        for val in result.best_params:
            assert isinstance(val, float), f"Expected float parameter, got {type(val)}"


class TestTrainPSO:
    def test_train_returns_list(self, parametrized_circuit, simple_expectation_fn):
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.PSO(generations=2, population_size=4, bounds=(0.0, math.pi)),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_pso",
        )
        assert isinstance(result.best_params, list)

    def test_train_result_length(self, parametrized_circuit, simple_expectation_fn):
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.PSO(generations=2, population_size=4, bounds=(0.0, math.pi)),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_pso_len",
        )
        assert len(result.best_params) == _DIMENSIONS


class TestTrainQNG:
    def test_train_returns_list(
        self, parametrized_circuit, simple_expectation_fn, simple_variance_fn
    ):
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.QNG(
                variance_function=simple_variance_fn,
                max_iters=3,
                bounds=(0.0, math.pi),
                learning_rate=0.1,
            ),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_qng",
        )
        assert isinstance(result.best_params, list)

    def test_train_result_length(
        self, parametrized_circuit, simple_expectation_fn, simple_variance_fn
    ):
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.QNG(
                variance_function=simple_variance_fn,
                max_iters=3,
                bounds=(0.0, math.pi),
            ),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_qng_len",
        )
        assert len(result.best_params) == _DIMENSIONS


class TestTrainAdam:
    def test_train_returns_list(self, parametrized_circuit, simple_expectation_fn):
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.Adam(max_iters=3, learning_rate=0.1, bounds=(0.0, math.pi)),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_adam",
        )
        assert isinstance(result.best_params, list)

    def test_train_result_length(self, parametrized_circuit, simple_expectation_fn):
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.Adam(max_iters=3, bounds=(0.0, math.pi)),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_adam_len",
        )
        assert len(result.best_params) == _DIMENSIONS


class TestTrainFitnessHistory:
    """`result.fitness_history` is the convergence curve of the run: one
    best-fitness-so-far value per generation/iteration actually executed
    (contract C-5). Checked on `polypus.train` — the generic entry point — for a
    gradient-free (DE) and a gradient (Adam) optimizer, since each optimizer
    tracks its own incumbent best."""

    @staticmethod
    def _assert_history(result):
        assert isinstance(result.fitness_history, list)
        assert all(isinstance(f, float) for f in result.fitness_history)
        # One entry per iteration actually run, ending on the reported best.
        assert len(result.fitness_history) == result.iterations_run
        assert result.fitness_history[-1] == result.best_fitness
        # Non-decreasing: every entry is the running best, never the fitness of
        # that iteration's current candidate.
        assert all(
            b >= a
            for a, b in zip(result.fitness_history, result.fitness_history[1:])
        )

    def test_de_reports_one_fitness_per_generation(
        self, parametrized_circuit, simple_expectation_fn
    ):
        import polypus

        result = polypus.train(
            parametrized_circuit,
            # A tight tolerance keeps the population from collapsing early, so
            # all 4 generations run and the curve has 4 points to compare.
            polypus.DE(generations=4, population_size=4, tolerance=1e-9),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_de_history",
        )
        assert result.iterations_run == 4
        self._assert_history(result)

    def test_adam_reports_one_fitness_per_iteration(
        self, parametrized_circuit, simple_expectation_fn
    ):
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.Adam(max_iters=4, learning_rate=0.1, bounds=(0.0, math.pi)),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_adam_history",
        )
        self._assert_history(result)

    def test_repr_summarises_the_history_by_length(
        self, parametrized_circuit, simple_expectation_fn
    ):
        # The repr reports the curve's length, not its contents: one float per
        # iteration would dominate the line on a long run.
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.DE(generations=3, population_size=4, tolerance=1e-9),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_de_history_repr",
        )
        assert "fitness_history=<3 values>" in repr(result)


class TestTrainInvalidMethod:
    def test_invalid_method_raises_type_error(
        self, parametrized_circuit, simple_expectation_fn
    ):
        import polypus

        with pytest.raises(TypeError):
            polypus.train(
                parametrized_circuit,
                "not_a_valid_method",
                shots=_SHOTS,
                n_qpus=_N_QPUS,
                dimensions=_DIMENSIONS,
                expectation_function=simple_expectation_fn,
                infrastructure="local",
                nodes=_NODES,
                cores_per_qpu=_CORES_PER_QPU,
                id="test_invalid",
            )


class TestTrainInvalidConfig:
    """Invalid optimizer configuration must cross the FFI seam as a ValueError,
    not panic. DE population_size < 4 and PSO empty bounds previously panicked
    inside the Rust optimizer loops; they now return a typed OptimizerError that
    the binding maps to PyValueError."""

    def test_de_population_below_four_raises_value_error(
        self, parametrized_circuit, simple_expectation_fn
    ):
        import polypus

        with pytest.raises(ValueError):
            polypus.train(
                parametrized_circuit,
                polypus.DE(generations=2, population_size=1, tolerance=0.5),
                shots=_SHOTS,
                n_qpus=_N_QPUS,
                dimensions=_DIMENSIONS,
                expectation_function=simple_expectation_fn,
                infrastructure="local",
                nodes=_NODES,
                cores_per_qpu=_CORES_PER_QPU,
                id="test_de_bad_pop",
            )

    def test_pso_empty_bounds_raises_value_error(
        self, parametrized_circuit, simple_expectation_fn
    ):
        import polypus

        with pytest.raises(ValueError):
            polypus.train(
                parametrized_circuit,
                polypus.PSO(generations=2, population_size=4, bounds=(1.0, 1.0)),
                shots=_SHOTS,
                n_qpus=_N_QPUS,
                dimensions=_DIMENSIONS,
                expectation_function=simple_expectation_fn,
                infrastructure="local",
                nodes=_NODES,
                cores_per_qpu=_CORES_PER_QPU,
                id="test_pso_bad_bounds",
            )
