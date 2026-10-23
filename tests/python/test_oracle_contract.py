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


class TestOracleContractC5Adam:
    """The same guarantees for ``polypus.Adam``: each iteration's batch of ``θ``
    and its finite-difference stencil is an evaluation-oracle batch, so a bad
    value in it, ``θ``'s own included, must surface as a typed error."""

    @staticmethod
    def _train(parametrized_circuit, expectation_fn, ident, **kwargs):
        import polypus

        return polypus.train(
            parametrized_circuit,
            polypus.Adam(max_iters=4, tolerance=0.5, patience=1, **kwargs),
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
            self._train(parametrized_circuit, lambda _b: value, "adam_non_finite")
        except BaseException as exc:  # noqa: BLE001 - the class is the assertion
            assert type(exc).__name__ != "PanicException"
            assert isinstance(exc, polypus.EvaluationError), exc
        else:
            pytest.fail("expected the non-finite expectation value to raise")

    def test_non_finite_value_in_the_update_raises_evaluation_error(
        self, parametrized_circuit, monkeypatch
    ):
        """The first iteration sees only finite values; in the second batch,
        the updated ``θ`` (its first circuit) measures a bitstring whose value
        is NaN while its stencil stays finite."""
        import polypus
        import polypus_python

        sizes = []

        def fake_run_qcs(infrastructure, **kwargs):
            sizes.append(len(kwargs["qcs"]))
            # Each batch holds θ and the 2 stencil circuits of one parameter.
            update = len(sizes) == 2
            return [
                {"1" if update and i == 0 else "0": kwargs["shots"]}
                for i in range(len(kwargs["qcs"]))
            ]

        monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)
        with pytest.raises(polypus.EvaluationError):
            self._train(
                parametrized_circuit,
                lambda b: float("nan") if b == "1" else 1.0,
                "adam_late_nan",
            )
        assert sizes == [3, 3], sizes

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
            self._train(parametrized_circuit, simple_expectation_fn, "adam_short")

    def test_valid_path_trains(self, parametrized_circuit, simple_expectation_fn):
        result = self._train(parametrized_circuit, simple_expectation_fn, "adam_valid")
        assert len(result.best_params) == _DIMENSIONS
        assert math.isfinite(result.best_fitness)
        assert len(result.fitness_history) == result.iterations_run
        assert result.fitness_history[-1] == result.best_fitness


class TestOracleContractC5SPSA:
    """The same guarantees for ``polypus.SPSA``: its calibration batch and each
    iteration's batch of perturbed pairs, with ``θ`` heading it when measured,
    are evaluation-oracle batches, so a bad value in any of them must surface
    as a typed error."""

    @staticmethod
    def _train(parametrized_circuit, expectation_fn, ident, **kwargs):
        import polypus

        return polypus.train(
            parametrized_circuit,
            polypus.SPSA(max_iters=4, eval_interval=2, **kwargs),
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
            self._train(parametrized_circuit, lambda _b: value, "spsa_non_finite")
        except BaseException as exc:  # noqa: BLE001 - the class is the assertion
            assert type(exc).__name__ != "PanicException"
            assert isinstance(exc, polypus.EvaluationError), exc
        else:
            pytest.fail("expected the non-finite expectation value to raise")

    def test_non_finite_value_of_theta_raises_evaluation_error(
        self, parametrized_circuit, monkeypatch
    ):
        """The calibration batch sees only finite values; in the second batch,
        ``θ`` (its first circuit) measures a bitstring whose value is NaN while
        its perturbed pair stays finite."""
        import polypus
        import polypus_python

        sizes = []

        def fake_run_qcs(infrastructure, **kwargs):
            sizes.append(len(kwargs["qcs"]))
            theta = len(sizes) == 2
            return [
                {"1" if theta and i == 0 else "0": kwargs["shots"]}
                for i in range(len(kwargs["qcs"]))
            ]

        monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)
        with pytest.raises(polypus.EvaluationError):
            self._train(
                parametrized_circuit,
                lambda b: float("nan") if b == "1" else 1.0,
                "spsa_theta_nan",
            )
        # The 50 calibration circuits, then θ and one perturbed pair.
        assert sizes == [50, 3], sizes

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
            self._train(parametrized_circuit, simple_expectation_fn, "spsa_short")

    def test_valid_path_trains(self, parametrized_circuit, simple_expectation_fn):
        result = self._train(parametrized_circuit, simple_expectation_fn, "spsa_valid")
        assert len(result.best_params) == _DIMENSIONS
        assert math.isfinite(result.best_fitness)
        assert result.iterations_run == 4
        assert not result.converged
        assert len(result.fitness_history) == result.iterations_run
        assert result.fitness_history[-1] == result.best_fitness


class TestOracleContractC5Rotosolve:
    """The same guarantees for ``polypus.Rotosolve``: each step's batch, the
    two neighbours of ``θ`` along one parameter with ``θ`` heading it when it
    is measured, is an evaluation-oracle batch, so a bad value in any of them
    must surface as a typed error."""

    @staticmethod
    def _train(parametrized_circuit, expectation_fn, ident, **kwargs):
        import polypus

        return polypus.train(
            parametrized_circuit,
            polypus.Rotosolve(sweeps=4, reevaluate_every=2, **kwargs),
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
            self._train(parametrized_circuit, lambda _b: value, "roto_non_finite")
        except BaseException as exc:  # noqa: BLE001 - the class is the assertion
            assert type(exc).__name__ != "PanicException"
            assert isinstance(exc, polypus.EvaluationError), exc
        else:
            pytest.fail("expected the non-finite expectation value to raise")

    def test_non_finite_value_of_theta_raises_evaluation_error(
        self, parametrized_circuit, monkeypatch
    ):
        """The first two steps see only finite values; the third measures ``θ``
        again (its first circuit), on a bitstring whose value is NaN while its
        two neighbours stay finite."""
        import polypus
        import polypus_python

        sizes = []

        def fake_run_qcs(infrastructure, **kwargs):
            sizes.append(len(kwargs["qcs"]))
            theta = len(sizes) == 3
            return [
                {"1" if theta and i == 0 else "0": kwargs["shots"]}
                for i in range(len(kwargs["qcs"]))
            ]

        monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)
        with pytest.raises(polypus.EvaluationError):
            self._train(
                parametrized_circuit,
                lambda b: float("nan") if b == "1" else 1.0,
                "roto_theta_nan",
                # A flat landscape would otherwise stop after two sweeps.
                tolerance=0,
            )
        # θ and its two neighbours, the two neighbours alone, then θ again.
        assert sizes == [3, 2, 3], sizes

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
            self._train(parametrized_circuit, simple_expectation_fn, "roto_short")

    def test_valid_path_trains(self, parametrized_circuit, simple_expectation_fn):
        result = self._train(
            parametrized_circuit, simple_expectation_fn, "roto_valid", tolerance=0
        )
        assert len(result.best_params) == _DIMENSIONS
        assert math.isfinite(result.best_fitness)
        assert result.iterations_run == 4
        assert not result.converged
        assert len(result.fitness_history) == result.iterations_run
        assert result.fitness_history[-1] == result.best_fitness


class TestOracleContractC5External:
    """The same guarantees for ``polypus.External``: each ``ask()`` is an
    evaluation-oracle batch, so a bad value in any of them must surface as a
    typed error, and the library is not told it."""

    @staticmethod
    def _train(parametrized_circuit, expectation_fn, ident, optimizer=None):
        import polypus
        from ask_tell_helpers import RandomSearch

        if optimizer is None:
            optimizer = RandomSearch(_DIMENSIONS, sizes=(3,))
        return polypus.train(
            parametrized_circuit,
            polypus.External(optimizer, 12),
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
        from ask_tell_helpers import RandomSearch

        optimizer = RandomSearch(_DIMENSIONS, sizes=(3,))
        try:
            self._train(
                parametrized_circuit, lambda _b: value, "ext_non_finite", optimizer
            )
        except BaseException as exc:  # noqa: BLE001 - the class is the assertion
            assert type(exc).__name__ != "PanicException"
            assert isinstance(exc, polypus.EvaluationError), exc
        else:
            pytest.fail("expected the non-finite expectation value to raise")
        assert len(optimizer.asked) == 1
        assert optimizer.told == []

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
            self._train(parametrized_circuit, simple_expectation_fn, "ext_short")

    def test_valid_path_trains(self, parametrized_circuit, simple_expectation_fn):
        result = self._train(parametrized_circuit, simple_expectation_fn, "ext_valid")
        assert len(result.best_params) == _DIMENSIONS
        assert math.isfinite(result.best_fitness)
        assert result.iterations_run == 4
        assert not result.converged
        assert len(result.fitness_history) == result.iterations_run
        assert result.fitness_history[-1] == result.best_fitness
