"""
The adapters of ``polypus.external``: Optuna, Nevergrad and pycma through
``polypus.External`` (issue #399).

The first part runs on stand-ins for each library's public ask/tell API, so it
needs none of the three: what each adapter asks, how it maps candidates to
rows, the sign of the values it tells, the library's stop signal, the batch
it re-offers and the checks of its arguments. Importing ``polypus.external``
must not import any of the libraries.

The second part trains through each real library and is skipped when it is
not installed (``pytest.importorskip``), as in CI, where none of them is: the
values reach the minimising libraries negated, checked on what their ``tell``
received.
"""

import math
import re
import subprocess
import sys

import numpy as np
import pytest
from polypus.external import from_nevergrad, from_optuna, from_pycma

DIMENSIONS = 2
SHOTS = 64


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


def _train(optimizer, max_evaluations, *, seed=5):
    """A seeded run on the native simulator, the third qubit in |+> so that
    values differ from candidate to candidate."""
    import polypus

    qc = polypus.Circuit(3)
    qc.ry(0, polypus.Param(0))
    qc.ry(1, polypus.Param(1))
    qc.h(2)
    qc.measure_all()
    return polypus.train(
        qc,
        polypus.External(optimizer, max_evaluations),
        shots=SHOTS,
        n_qpus=1,
        dimensions=DIMENSIONS,
        expectation_function=_all_ones,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="external_adapter",
        backend="polypus",
        seed=seed,
    )


class Recording:
    """Wraps an ask/tell optimizer and keeps every ``(candidates, values)``
    pair Polypus told it."""

    def __init__(self, inner):
        self.inner = inner
        self.told = []

    def ask(self):
        return self.inner.ask()

    def tell(self, candidates, values):
        self.told.append((candidates, values))
        self.inner.tell(candidates, values)

    def values(self):
        return [v for _, values in self.told for v in values]


def test_importing_the_adapters_imports_none_of_the_libraries():
    code = (
        "import sys, polypus, polypus.external\n"
        "assert polypus.external.from_optuna\n"
        "print(sorted({'optuna', 'nevergrad', 'cma'} & set(sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_the_module_is_reachable_from_polypus():
    import polypus
    import polypus.external

    assert polypus.external is sys.modules["polypus.external"]
    assert "external" in polypus.__all__
    assert polypus.external.__all__ == [
        "AskTell",
        "from_optuna",
        "from_nevergrad",
        "from_pycma",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Stand-ins for the libraries' public APIs
# ─────────────────────────────────────────────────────────────────────────────


class _Direction:
    def __init__(self, name):
        self.name = name


class FakeTrial:
    def __init__(self, number, fixed):
        self.number = number
        self.fixed = fixed
        self.params = {}

    def suggest_float(self, name, low, high):
        value = self.fixed.get(name, low + (high - low) * ((self.number + 1) / 10.0))
        self.params[name] = value
        return value


class FakeStudy:
    """``ask``, ``tell``, ``enqueue_trial`` and ``directions``, as Optuna's
    ``Study`` documents them."""

    def __init__(self, *directions):
        self.directions = [_Direction(d) for d in directions or ("MINIMIZE",)]
        self.trials = []
        self.queue = []
        self.told = []

    def ask(self):
        fixed = self.queue.pop(0) if self.queue else {}
        trial = FakeTrial(len(self.trials), fixed)
        self.trials.append(trial)
        return trial

    def tell(self, trial, value):
        if trial.number in getattr(self, "refuse", ()):
            raise LookupError(f"trial {trial.number}")
        self.told.append((trial.number, value))

    def enqueue_trial(self, params):
        self.queue.append(dict(params))


class FakeCandidate:
    def __init__(self, value):
        self.value = value


class FakeNevergrad:
    """``ask``, ``tell``, ``suggest``, ``num_workers``, ``budget``,
    ``num_ask`` and ``dimension``, as Nevergrad's optimizers document them."""

    def __init__(self, dimension=DIMENSIONS, budget=None, num_workers=3):
        self.dimension = dimension
        self.budget = budget
        self.num_workers = num_workers
        self.num_ask = 0
        self.suggested = []
        self.told = []

    def ask(self):
        if self.suggested:
            value = np.asarray(self.suggested.pop(0))
        else:
            value = np.full(self.dimension, 0.1 * (self.num_ask + 1))
        self.num_ask += 1
        return FakeCandidate(value)

    def tell(self, candidate, loss):
        if len(self.told) in getattr(self, "refuse", ()):
            raise LookupError(f"tell {len(self.told)}")
        self.told.append((candidate, loss))

    def suggest(self, *args):
        self.suggested.append(args[0])


class FakeES:
    """``ask``, ``tell`` and ``stop``, as pycma's ``CMAEvolutionStrategy``
    documents them; it stops after ``generations``."""

    def __init__(self, generations=3, popsize=4):
        self.generations = generations
        self.popsize = popsize
        self.asked = []
        self.told = []

    def stop(self):
        return (
            {"maxiter": self.generations} if len(self.told) >= self.generations else {}
        )

    def ask(self):
        g = len(self.asked)
        solutions = [np.array([0.1 * g, 0.2 * i]) for i in range(self.popsize)]
        self.asked.append(solutions)
        return solutions

    def tell(self, solutions, values):
        if getattr(self, "refuse", False):
            raise LookupError("generation")
        self.told.append((solutions, values))


class TestOptunaAdapter:
    @pytest.mark.parametrize(("direction", "sign"), [("MINIMIZE", -1), ("MAXIMIZE", 1)])
    def test_batches_names_bounds_and_the_sign_told(self, direction, sign):
        study = FakeStudy(direction)
        adapter = from_optuna(study, 3, 2, bounds=(-1.0, 1.0))
        batch = adapter.ask()
        assert len(batch) == 2 and all(len(row) == 3 for row in batch)
        assert all(-1.0 <= x <= 1.0 for row in batch for x in row)
        assert list(study.trials[0].params) == ["x0", "x1", "x2"]
        adapter.tell(batch, [0.25, 0.5])
        assert study.told == [(0, sign * 0.25), (1, sign * 0.5)]

    def test_the_default_bounds_are_zero_to_two_pi(self):
        study = FakeStudy()
        row = from_optuna(study, 1, 1).ask()[0]
        assert row == pytest.approx([2 * math.pi / 10.0])

    def test_a_batch_not_told_is_offered_again(self):
        study = FakeStudy()
        adapter = from_optuna(study, 2, 3)
        first = adapter.ask()
        assert adapter.ask() == first
        assert len(study.trials) == 3
        adapter.tell(first, [1.0, 2.0, 3.0])
        assert adapter.ask() != first
        assert len(study.trials) == 6

    def test_a_tell_that_fails_partway_keeps_the_rest_pending(self):
        study = FakeStudy()
        study.refuse = {1}
        adapter = from_optuna(study, 2, 3)
        batch = adapter.ask()
        with pytest.raises(LookupError, match="trial 1"):
            adapter.tell(batch, [1.0, 2.0, 3.0])
        assert study.told == [(0, -1.0)]
        assert adapter.ask() == batch[1:], "the two untold trials, offered again"
        study.refuse = set()
        adapter.tell(batch[1:], [2.0, 3.0])
        assert study.told == [(0, -1.0), (1, -2.0), (2, -3.0)]
        assert len(study.trials) == 3

    def test_a_tell_with_no_batch_pending_raises(self):
        adapter = from_optuna(FakeStudy(), 2, 2)
        with pytest.raises(RuntimeError, match="no batch pending"):
            adapter.tell([[0.0, 0.0]], [1.0])
        batch = adapter.ask()
        with pytest.raises(ValueError, match="1 values for a batch of 2"):
            adapter.tell(batch, [1.0])
        adapter.tell(batch, [1.0, 2.0])
        with pytest.raises(RuntimeError, match="no batch pending"):
            adapter.tell(batch, [1.0, 2.0])

    def test_the_initial_point_is_the_first_trial(self):
        study = FakeStudy()
        adapter = from_optuna(study, 2, 2, initial_point=[0.5, np.float64(1.5)])
        batch = adapter.ask()
        assert batch[0] == [0.5, 1.5]
        assert batch[1] != [0.5, 1.5]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"dimensions": 0}, "dimensions must be an integer >= 1, got 0"),
            ({"batch_size": 2.0}, "batch_size must be an integer >= 1, got 2.0"),
            ({"batch_size": True}, "batch_size must be an integer >= 1, got True"),
            ({"bounds": (1.0, 1.0)}, "bounds must be finite with low < high"),
            ({"bounds": (0.0, math.inf)}, "bounds must be finite with low < high"),
            ({"bounds": 3}, "bounds must be two numbers (low, high), got 3"),
            ({"initial_point": [0.5]}, "must have one value per dimension (2), got 1"),
            ({"initial_point": [0.5, math.nan]}, "every value must be finite"),
            ({"initial_point": [0.5, 7.0]}, "must lie within bounds"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, message):
        arguments = {"dimensions": 2, "batch_size": 2, **kwargs}
        study = FakeStudy()
        with pytest.raises(ValueError, match=re.escape(message)):
            from_optuna(study, **arguments)
        assert study.queue == []

    def test_a_study_with_several_objectives_raises(self):
        with pytest.raises(ValueError, match="single-objective study, got one with 2"):
            from_optuna(FakeStudy("MINIMIZE", "MAXIMIZE"), 2, 2)
        with pytest.raises(ValueError, match="maximizes or minimizes, got NOT_SET"):
            from_optuna(FakeStudy("NOT_SET"), 2, 2)

    def test_a_run_through_the_bridge(self):
        study = FakeStudy()
        recording = Recording(from_optuna(study, DIMENSIONS, 3))
        result = _train(recording, 9)
        assert result.iterations_run == 3
        assert not result.converged, "Optuna never ends: the budget did"
        assert [value for _, value in study.told] == [-v for v in recording.values()]


class TestNevergradAdapter:
    def test_batches_of_num_workers_told_negated(self):
        optimizer = FakeNevergrad(num_workers=3)
        adapter = from_nevergrad(optimizer)
        batch = adapter.ask()
        assert batch == [[0.1, 0.1], [0.2, 0.2], [0.30000000000000004] * 2]
        adapter.tell(batch, [1.0, 2.0, 3.0])
        assert [loss for _, loss in optimizer.told] == [-1.0, -2.0, -3.0]

    def test_the_budget_ends_the_run_and_cuts_the_last_batch(self):
        optimizer = FakeNevergrad(budget=5)
        adapter = from_nevergrad(optimizer, batch_size=2)
        sizes = []
        while batch := adapter.ask():
            sizes.append(len(batch))
            adapter.tell(batch, [0.0] * len(batch))
        assert sizes == [2, 2, 1]
        assert optimizer.num_ask == 5

    def test_a_batch_not_told_is_offered_again(self):
        optimizer = FakeNevergrad()
        adapter = from_nevergrad(optimizer, batch_size=2)
        first = adapter.ask()
        assert adapter.ask() == first
        assert optimizer.num_ask == 2

    def test_a_tell_that_fails_partway_keeps_the_rest_pending(self):
        optimizer = FakeNevergrad(num_workers=3)
        optimizer.refuse = {2}
        adapter = from_nevergrad(optimizer)
        batch = adapter.ask()
        with pytest.raises(LookupError, match="tell 2"):
            adapter.tell(batch, [1.0, 2.0, 3.0])
        assert [loss for _, loss in optimizer.told] == [-1.0, -2.0]
        assert adapter.ask() == batch[2:]
        optimizer.refuse = set()
        adapter.tell(batch[2:], [3.0])
        assert [loss for _, loss in optimizer.told] == [-1.0, -2.0, -3.0]
        assert optimizer.num_ask == 3

    def test_a_tell_with_no_batch_pending_raises(self):
        adapter = from_nevergrad(FakeNevergrad(), batch_size=2)
        with pytest.raises(RuntimeError, match="no batch pending"):
            adapter.tell([[0.0, 0.0]], [1.0])
        batch = adapter.ask()
        with pytest.raises(ValueError, match="3 values for a batch of 2"):
            adapter.tell(batch, [1.0, 2.0, 3.0])

    def test_the_initial_point_is_suggested(self):
        optimizer = FakeNevergrad()
        adapter = from_nevergrad(optimizer, batch_size=2, initial_point=(0.5, 1.5))
        assert adapter.ask()[0] == [0.5, 1.5]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"batch_size": 0}, "batch_size must be an integer >= 1, got 0"),
            ({"initial_point": [0.5, 0.5, 0.5]}, "one value per dimension"),
            ({"initial_point": [0.5, math.inf]}, "every value must be finite"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, message):
        optimizer = FakeNevergrad()
        with pytest.raises(ValueError, match=message):
            from_nevergrad(optimizer, **kwargs)
        assert optimizer.suggested == []

    def test_a_candidate_that_is_not_flat_raises(self):
        optimizer = FakeNevergrad()
        optimizer.suggest(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="flat sequence of numbers"):
            from_nevergrad(optimizer, batch_size=1).ask()

    def test_a_run_through_the_bridge(self):
        optimizer = FakeNevergrad(budget=8, num_workers=3)
        recording = Recording(from_nevergrad(optimizer))
        result = _train(recording, 100)
        assert [len(c) for c, _ in recording.told] == [3, 3, 2]
        assert result.converged, "the budget of the library ended the run"
        assert [loss for _, loss in optimizer.told] == [-v for v in recording.values()]


class TestPycmaAdapter:
    def test_a_generation_per_batch_told_negated_with_the_same_solutions(self):
        es = FakeES(popsize=3)
        adapter = from_pycma(es)
        batch = adapter.ask()
        assert batch == [[0.0, 0.0], [0.0, 0.2], [0.0, 0.4]]
        adapter.tell(batch, [1.0, 2.0, 3.0])
        solutions, values = es.told[0]
        assert solutions is es.asked[0]
        assert values == [-1.0, -2.0, -3.0]

    def test_stop_ends_the_run(self):
        es = FakeES(generations=2)
        adapter = from_pycma(es)
        for _ in range(2):
            adapter.tell(adapter.ask(), [0.0] * 4)
        assert adapter.ask() == []
        assert len(es.asked) == 2

    def test_a_batch_not_told_is_offered_again(self):
        es = FakeES()
        adapter = from_pycma(es)
        assert adapter.ask() == adapter.ask()
        assert len(es.asked) == 1

    def test_a_tell_that_fails_keeps_the_generation_pending(self):
        es = FakeES(popsize=2)
        es.refuse = True
        adapter = from_pycma(es)
        batch = adapter.ask()
        with pytest.raises(LookupError, match="generation"):
            adapter.tell(batch, [1.0, 2.0])
        assert adapter.ask() == batch
        assert len(es.asked) == 1
        es.refuse = False
        adapter.tell(batch, [1.0, 2.0])
        assert es.told[0][1] == [-1.0, -2.0]

    def test_a_tell_with_no_batch_pending_raises(self):
        adapter = from_pycma(FakeES(popsize=2))
        with pytest.raises(RuntimeError, match="no batch pending"):
            adapter.tell([[0.0, 0.0]], [1.0])
        batch = adapter.ask()
        with pytest.raises(ValueError, match="1 values for a batch of 2"):
            adapter.tell(batch, [1.0])

    def test_a_run_through_the_bridge(self):
        es = FakeES(generations=3, popsize=4)
        recording = Recording(from_pycma(es))
        result = _train(recording, 100)
        assert result.converged
        assert result.iterations_run == 3
        assert [v for _, values in es.told for v in values] == [
            -v for v in recording.values()
        ]


# ─────────────────────────────────────────────────────────────────────────────
# The real libraries, when installed
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestOptuna:
    @pytest.mark.parametrize("direction", ["minimize", "maximize"])
    def test_tpe_through_the_bridge(self, direction):
        optuna = pytest.importorskip("optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=1, constant_liar=True),
        )
        recording = Recording(
            from_optuna(study, DIMENSIONS, 4, initial_point=[math.pi, math.pi])
        )
        result = _train(recording, 24)
        assert result.iterations_run == 6
        assert len(study.trials) == 24
        sign = -1.0 if direction == "minimize" else 1.0
        # Each trial was told its value, negated when the study minimises.
        assert [t.value for t in study.trials] == [sign * v for v in recording.values()]
        rows = [row for candidates, _ in recording.told for row in candidates]
        assert [[t.params["x0"], t.params["x1"]] for t in study.trials] == rows
        assert rows[0] == [math.pi, math.pi]
        # Polypus's best is the study's best, up to the sign.
        assert result.best_fitness == sign * study.best_value
        assert result.best_params == [study.best_params["x0"], study.best_params["x1"]]

    def test_the_same_seeds_reproduce_the_run(self):
        optuna = pytest.importorskip("optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def run():
            study = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=7, constant_liar=True)
            )
            return _train(from_optuna(study, DIMENSIONS, 3), 15, seed=11)

        first, second = run(), run()
        assert first.best_params == second.best_params
        assert first.fitness_history == second.fitness_history


@pytest.mark.integration
class TestNevergrad:
    def test_ngopt_through_the_bridge(self):
        ng = pytest.importorskip("nevergrad")
        parametrization = ng.p.Array(shape=(DIMENSIONS,)).set_bounds(0.0, 2 * math.pi)
        parametrization.random_state = np.random.RandomState(3)
        optimizer = ng.optimizers.NGOpt(
            parametrization=parametrization, budget=20, num_workers=4
        )
        received = []
        tell = optimizer.tell

        def recording_tell(candidate, loss, *args, **kwargs):
            received.append(loss)
            return tell(candidate, loss, *args, **kwargs)

        optimizer.tell = recording_tell
        recording = Recording(
            from_nevergrad(optimizer, initial_point=[math.pi, math.pi])
        )
        result = _train(recording, 1000)
        # The library's budget of 20 ended the run, in batches of num_workers.
        assert [len(c) for c, _ in recording.told] == [4] * 5
        assert result.converged
        assert optimizer.num_tell == 20
        assert received == [-v for v in recording.values()]
        assert recording.told[0][0][0] == [math.pi, math.pi]
        best = max(recording.values())
        assert result.best_fitness == best


@pytest.mark.integration
class TestPycma:
    def test_cma_es_through_the_bridge(self):
        cma = pytest.importorskip("cma")
        es = cma.CMAEvolutionStrategy(
            [1.0, 1.0],
            0.5,
            {"seed": 3, "popsize": 6, "maxfevals": 30, "verbose": -9},
        )
        received = []
        tell = es.tell

        def recording_tell(solutions, function_values, *args, **kwargs):
            received.extend(function_values)
            return tell(solutions, function_values, *args, **kwargs)

        es.tell = recording_tell
        recording = Recording(from_pycma(es))
        result = _train(recording, 1000)
        # One generation of six per batch, until es.stop() reports maxfevals.
        sizes = [len(c) for c, _ in recording.told]
        assert set(sizes) == {6}
        assert sum(sizes) == es.countevals >= 30
        assert "maxfevals" in es.stop()
        assert result.converged
        assert received == [-v for v in recording.values()]
        assert result.best_fitness == max(recording.values())
