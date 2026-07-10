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

import pytest
import math


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
        assert isinstance(result, list), f"Expected list of parameters, got {type(result)}"

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
        assert len(result) == _DIMENSIONS

    def test_train_result_contains_floats(self, parametrized_circuit, simple_expectation_fn):
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
        for val in result:
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
        assert isinstance(result, list)

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
        assert len(result) == _DIMENSIONS


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
        assert isinstance(result, list)

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
        assert len(result) == _DIMENSIONS


class TestTrainInvalidMethod:
    def test_invalid_method_raises_type_error(self, parametrized_circuit, simple_expectation_fn):
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
