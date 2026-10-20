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
silently-poisoned result. Each test runs every DE ``strategy``: the adaptive
ones evaluate their batches in their own loop.
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
_STRATEGIES = ["rand1bin", "shade", "lshade"]


class TestOracleContractC5:
    @pytest.mark.parametrize("strategy", _STRATEGIES)
    def test_non_finite_value_raises_evaluation_error(
        self, parametrized_circuit, strategy
    ):
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
                polypus.DE(
                    generations=2,
                    population_size=_POPULATION,
                    tolerance=0.5,
                    strategy=strategy,
                ),
                shots=_SHOTS,
                n_qpus=_N_QPUS,
                dimensions=_DIMENSIONS,
                expectation_function=nan_expectation_fn,
                infrastructure="local",
                nodes=_NODES,
                cores_per_qpu=_CORES_PER_QPU,
                id="test_oracle_non_finite",
            )

    @pytest.mark.parametrize("strategy", _STRATEGIES)
    def test_non_finite_value_is_never_a_panic_exception(
        self, parametrized_circuit, strategy
    ):
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
                polypus.DE(
                    generations=2,
                    population_size=_POPULATION,
                    tolerance=0.5,
                    strategy=strategy,
                ),
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

    @pytest.mark.parametrize("strategy", _STRATEGIES)
    def test_correct_length_finite_path_still_trains(
        self, parametrized_circuit, simple_expectation_fn, strategy
    ):
        """Regression guard: the valid path (correct length, finite values) still
        trains successfully — the C-5 validation must not break test_vqc.py."""
        import polypus

        result = polypus.train(
            parametrized_circuit,
            polypus.DE(
                generations=2,
                population_size=_POPULATION,
                tolerance=0.5,
                strategy=strategy,
            ),
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


class TestOracleContractC5Cmaes:
    """The same guarantees for ``polypus.CMAES``, which evaluates its batches
    in its own loop and, with restarts, sends larger batches later on."""

    @staticmethod
    def _train(parametrized_circuit, expectation_fn, ident, **kwargs):
        import polypus

        return polypus.train(
            parametrized_circuit,
            polypus.CMAES(generations=4, tolerance=0.5, patience=1, **kwargs),
            shots=_SHOTS,
            n_qpus=_N_QPUS,
            dimensions=_DIMENSIONS,
            expectation_function=expectation_fn,
            infrastructure="local",
            nodes=_NODES,
            cores_per_qpu=_CORES_PER_QPU,
            id=ident,
        )

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_value_raises_evaluation_error(
        self, parametrized_circuit, value
    ):
        import polypus

        try:
            self._train(parametrized_circuit, lambda _b: value, "cmaes_non_finite")
        except BaseException as exc:  # noqa: BLE001 - the class is the assertion
            assert type(exc).__name__ != "PanicException"
            assert isinstance(exc, polypus.EvaluationError), exc
        else:
            pytest.fail("expected the non-finite expectation value to raise")

    def test_non_finite_value_after_a_restart_raises_evaluation_error(
        self, parametrized_circuit, monkeypatch
    ):
        """The first run sees only finite values and stagnates, so CMA-ES
        restarts with twice the population; the first larger batch measures a
        bitstring whose value is NaN."""
        import polypus
        import polypus_python

        sizes = []

        def fake_run_qcs(infrastructure, **kwargs):
            sizes.append(len(kwargs["qcs"]))
            # λ = 4 for one parameter; the restart's batches hold 8 circuits.
            bitstring = "1" if len(kwargs["qcs"]) > 4 else "0"
            return [{bitstring: kwargs["shots"]} for _ in kwargs["qcs"]]

        monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)
        with pytest.raises(polypus.EvaluationError):
            self._train(
                parametrized_circuit,
                lambda b: float("nan") if b == "1" else 1.0,
                "cmaes_late_nan",
                restarts=2,
            )
        assert sizes == [4, 4, 8], sizes

    def test_short_seam_result_is_a_typed_error(
        self, parametrized_circuit, simple_expectation_fn, monkeypatch
    ):
        """A backend that returns one counts map too few is a typed
        ``polypus.PolypusError``, never a panic and never a result."""
        import polypus
        import polypus_python

        def short_run_qcs(infrastructure, **kwargs):
            return [{"1": kwargs["shots"]} for _ in kwargs["qcs"]][:-1]

        monkeypatch.setattr(polypus_python, "run_qcs", short_run_qcs)
        with pytest.raises(polypus.PolypusError):
            self._train(parametrized_circuit, simple_expectation_fn, "cmaes_short")

    def test_valid_path_trains(self, parametrized_circuit, simple_expectation_fn):
        result = self._train(
            parametrized_circuit, simple_expectation_fn, "cmaes_valid", restarts=1
        )
        assert len(result.best_params) == _DIMENSIONS
        assert 0.0 <= result.best_params[0] < 2 * math.pi
        assert math.isfinite(result.best_fitness)
        assert len(result.fitness_history) == result.iterations_run
        assert result.fitness_history[-1] == result.best_fitness
