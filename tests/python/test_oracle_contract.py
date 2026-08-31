"""
Enforcing test for contract C-5 (optimizer ↔ oracle) at the Rust↔Python seam.

C-5 requires ``EvaluationOracle::evaluate_batch`` to return exactly
``candidates.len()`` **finite** ``f64`` values. The single choke point that
reduces measurement counts to fitness lives in
``crates/polypus/src/evaluation/mod.rs`` (``run_and_evaluate``), which validates
the batch (``WrongLength`` / ``NonFinite``) before returning across the FFI.

Since the native cost-observable rewrite, that reduction is done by a
``CostObservable`` — a native ``polypus.Qubo`` / ``polypus.Ising``, or a
``PyCallbackObservable`` wrapping a Python callable — each of which emits exactly
one value per candidate. A **wrong-length** batch is therefore no longer
reachable through the public API (the old failure required the former
``polypus_python.expectation_values`` seam to return a mis-sized list); the
length half of C-5 is now a defensive check whose error→exception mapping is
pinned by the Rust unit tests in ``crates/polypus/src/evaluation/error.rs``.

The end-to-end failure mode a user can still trigger is a **non-finite** value
(a cost callback returning NaN/inf). This test drives it through
``polypus.train`` and locks in that it surfaces as a typed
``polypus.EvaluationError`` — never a ``pyo3_runtime.PanicException`` and never a
silently-poisoned result.
"""

import math

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]

# Shared training parameters — kept minimal for speed, matching test_vqc.py.
_SHOTS = 256
_N_QPUS = 1
_DIMENSIONS = 1
_NODES = 1
_CORES_PER_QPU = 1
_POPULATION = 4


class TestOracleContractC5:
    def test_non_finite_value_raises_evaluation_error(self, parametrized_circuit):
        """A NaN expectation value → typed EvaluationError, not a silent result.

        The cost callback returns NaN for every bitstring, so a non-finite value
        flows through the real ``PyCallbackObservable`` aggregation exactly as it
        would in production; ``run_and_evaluate``'s C-5 check must reject it.
        """
        import polypus

        def nan_expectation_fn(_bitstring: str) -> float:
            return float("nan")

        with pytest.raises(polypus.EvaluationError):
            polypus.train(
                parametrized_circuit,
                polypus.DE(generations=2, population_size=_POPULATION, tolerance=0.5),
                shots=_SHOTS,
                n_qpus=_N_QPUS,
                dimensions=_DIMENSIONS,
                expectation_function=nan_expectation_fn,
                infrastructure="local",
                nodes=_NODES,
                cores_per_qpu=_CORES_PER_QPU,
                id="test_oracle_non_finite",
            )

    def test_non_finite_value_is_never_a_panic_exception(self, parametrized_circuit):
        """Whatever the non-finite value does, the caller never sees PanicException.

        Same assertion style as test_seam_contract.py's
        ``test_seam_failure_is_never_a_panic_exception``: a NaN would otherwise
        silently poison the pure-Rust optimizer, so it must surface as a typed
        ``polypus.EvaluationError``, never an uncatchable Rust panic across the FFI.
        """
        import polypus

        def nan_expectation_fn(_bitstring: str) -> float:
            return float("nan")

        try:
            polypus.train(
                parametrized_circuit,
                polypus.DE(generations=2, population_size=_POPULATION, tolerance=0.5),
                shots=_SHOTS,
                n_qpus=_N_QPUS,
                dimensions=_DIMENSIONS,
                expectation_function=nan_expectation_fn,
                infrastructure="local",
                nodes=_NODES,
                cores_per_qpu=_CORES_PER_QPU,
                id="test_oracle_non_finite_no_panic",
            )
        except BaseException as exc:  # noqa: BLE001 - we assert on the type below
            assert type(exc).__name__ != "PanicException", (
                "a non-finite expectation value must not surface as a Rust panic"
            )
            assert isinstance(exc, polypus.EvaluationError)
        else:
            pytest.fail("expected the NaN expectation value to raise")

    def test_correct_length_finite_path_still_trains(
        self, parametrized_circuit, simple_expectation_fn
    ):
        """Regression guard: the valid path (correct length, finite values) still
        trains successfully — the C-5 validation must not break test_vqc.py."""
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.DE(generations=2, population_size=_POPULATION, tolerance=0.5),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=simple_expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id="test_oracle_valid_path",
        )
        assert isinstance(result.best_params, list)
        assert len(result.best_params) == _DIMENSIONS
        assert isinstance(result.best_fitness, float)
        assert math.isfinite(result.best_fitness)
