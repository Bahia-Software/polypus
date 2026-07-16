"""
Enforcing test for contract C-5 (optimizer ↔ oracle) at the Rust↔Python seam.

C-5 requires ``EvaluationOracle::evaluate_batch`` to return exactly
``candidates.len()`` **finite** ``f64`` values, and states explicitly that
"Python-backed oracles must validate length before returning across the FFI".
The single choke point that calls ``polypus_python.expectation_values`` lives in
``crates/polypus/src/evaluation/mod.rs`` (``run_and_evaluate``); this test drives
both failure modes through the public ``polypus.train`` entry point and locks in
that they surface as a typed ``polypus.EvaluationError`` — never a
``pyo3_runtime.PanicException`` and never a silently-poisoned result:

* a **short** return list would otherwise index out of bounds inside the
  pure-Rust optimizer → panic → uncatchable ``PanicException`` across the FFI
  (the failure mode #35 / PR #58 eliminated elsewhere);
* a **non-finite** return value would otherwise silently poison the optimizer
  and let ``train`` return a bogus ``TrainResult`` with no error at all.

It mirrors the structure and panic-safety style of ``test_seam_contract.py``
(C-1): monkeypatch the ``polypus_python`` seam to force the failure, then assert
on the surfaced exception type.
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
    def test_wrong_length_raises_evaluation_error(
        self, parametrized_circuit, simple_expectation_fn, monkeypatch
    ):
        """A short expectation_values list → typed EvaluationError, never a panic.

        Monkeypatch the seam (same pattern as test_seam_contract.py's run_qcs
        patches) to return one *fewer* value than circuits submitted, so the
        length mismatch is triggered regardless of how the backend batches.
        """
        import polypus
        import polypus_python

        def short_expectation_values(array_counts, _bitstring_to_obj):
            # Deliberately drop one value: length != len(array_counts).
            return [0.0] * (len(array_counts) - 1)

        monkeypatch.setattr(
            polypus_python, "expectation_values", short_expectation_values
        )

        with pytest.raises(polypus.EvaluationError):
            polypus.train(
                parametrized_circuit,
                polypus.DE(generations=2, population_size=_POPULATION, tolerance=0.5),
                shots=_SHOTS,
                n_qpus=_N_QPUS,
                dimensions=_DIMENSIONS,
                expectation_function=simple_expectation_fn,
                infrastructure="local",
                nodes=_NODES,
                cores_per_qpu=_CORES_PER_QPU,
                id="test_oracle_wrong_length",
            )

    def test_wrong_length_is_never_a_panic_exception(
        self, parametrized_circuit, simple_expectation_fn, monkeypatch
    ):
        """Whatever the length mismatch does, the caller never sees PanicException.

        Same assertion style as test_seam_contract.py's
        test_seam_failure_is_never_a_panic_exception.
        """
        import polypus
        import polypus_python

        def short_expectation_values(array_counts, _bitstring_to_obj):
            return [0.0] * (len(array_counts) - 1)

        monkeypatch.setattr(
            polypus_python, "expectation_values", short_expectation_values
        )

        try:
            polypus.train(
                parametrized_circuit,
                polypus.DE(generations=2, population_size=_POPULATION, tolerance=0.5),
                shots=_SHOTS,
                n_qpus=_N_QPUS,
                dimensions=_DIMENSIONS,
                expectation_function=simple_expectation_fn,
                infrastructure="local",
                nodes=_NODES,
                cores_per_qpu=_CORES_PER_QPU,
                id="test_oracle_wrong_length_no_panic",
            )
        except BaseException as exc:  # noqa: BLE001 - we assert on the type below
            assert type(exc).__name__ != "PanicException", (
                "an oracle length violation must not surface as a Rust panic"
            )
            assert isinstance(exc, polypus.EvaluationError)
        else:
            pytest.fail("expected the short expectation_values list to raise")

    def test_non_finite_value_raises_evaluation_error(self, parametrized_circuit):
        """A NaN expectation value → typed EvaluationError, not a silent result.

        Deliberately does NOT monkeypatch the seam: the expectation_function
        returns NaN for every bitstring so the NaN flows through the *real*
        polypus_python.qaoa_utils.expectation_values averaging code, exactly as
        it would in production.
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

    def test_correct_length_finite_path_still_trains(
        self, parametrized_circuit, simple_expectation_fn
    ):
        """Regression guard: the valid path (correct length, finite values) still
        trains successfully — the added validation must not break test_vqc.py."""
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
