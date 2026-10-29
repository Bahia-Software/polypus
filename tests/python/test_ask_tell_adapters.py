"""
The adapters of ``polypus.ask_tell``: Optuna, Nevergrad and pycma through
``polypus.External`` (issue #399).

The first part runs on stand-ins for each library's public ask/tell API, so it
needs none of the three: what each adapter asks, how it maps candidates to
rows, the sign of the values it tells, the library's stop signal, the batch
it re-offers, the checks of its arguments and of what it is told, and
Optuna's ``close()``. Importing ``polypus.ask_tell`` must not import any of
the libraries, and the pip extras ask for the versions CI tests.

The second part trains through each real library and is skipped when it is
not installed, except under ``POLYPUS_REQUIRE_EXTERNAL_LIBRARIES=1``, where it
fails instead: CI installs the three libraries, pinned, on its latest Python
and sets it. The values reach the minimising libraries negated, checked on
what their ``tell`` received.
"""

import importlib
import math
import os
import re
import subprocess
import sys
import types
from importlib.metadata import PackageNotFoundError, requires
from pathlib import Path

import numpy as np
import polypus
import pytest
from polypus.ask_tell import from_nevergrad, from_optuna, from_pycma

DIMENSIONS = 2
SHOTS = 64
# CI installs the three libraries on its latest Python and sets this to 1, so
# a missing one fails the tests that train through it instead of skipping.
REQUIRE_LIBRARIES = "POLYPUS_REQUIRE_EXTERNAL_LIBRARIES"
CI_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
# A batch of two rows told back otherwise than ask() handed it out.
OTHER_CANDIDATES = pytest.mark.parametrize(
    "other",
    [
        lambda batch: [batch[1], batch[0]],
        lambda batch: [batch[0], [x + 1.0 for x in batch[1]]],
        lambda batch: [batch[0]],
        lambda batch: [batch[0], ["a", "b"]],
    ],
    ids=["reordered", "another-row", "one-row-missing", "not-numbers"],
)


def _library(name):
    """The library ``name``, imported. The test skips when it is not
    installed, or fails under ``POLYPUS_REQUIRE_EXTERNAL_LIBRARIES=1``."""
    if os.environ.get(REQUIRE_LIBRARIES) == "1":
        return importlib.import_module(name)
    return pytest.importorskip(name)


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
        "import sys, polypus, polypus.ask_tell\n"
        "assert polypus.ask_tell.from_optuna\n"
        "print(sorted({'optuna', 'nevergrad', 'cma'} & set(sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_the_module_is_reachable_from_polypus():
    import polypus
    import polypus.ask_tell

    assert polypus.ask_tell is sys.modules["polypus.ask_tell"]
    assert "ask_tell" in polypus.__all__
    assert polypus.ask_tell.__all__ == [
        "AskTell",
        "OptunaAskTell",
        "from_optuna",
        "from_nevergrad",
        "from_pycma",
    ]


def test_the_extras_ask_for_the_versions_ci_tests():
    # pyproject.toml's floors are the pins of CI's adapter tests: raising a pin
    # without its floor would let pip install a version no test ran on.
    try:
        requirements = requires("polypus-quantum") or []
    except PackageNotFoundError:
        pytest.skip("the polypus-quantum distribution is not installed")
    if not CI_WORKFLOW.is_file():
        pytest.skip(f"no {CI_WORKFLOW}: tests run outside the repository")
    pins = dict(
        re.findall(r"\b(optuna|nevergrad|cma)==([\w.]+)", CI_WORKFLOW.read_text())
    )
    floors = {}
    for requirement in requirements:
        extra = re.search(r"""extra\s*==\s*["']([\w-]+)["']""", requirement)
        if extra:
            name = re.match(r"[\w.-]+", requirement).group(0)
            floor = re.search(r">=\s*([\w.]+)", requirement)
            floors[extra.group(1)] = (name, floor and floor.group(1))
    assert sorted(pins) == ["cma", "nevergrad", "optuna"], pins
    assert floors == {name: (name, pin) for name, pin in pins.items()}


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
        self.failed = []

    def ask(self):
        fixed = self.queue.pop(0) if self.queue else {}
        trial = FakeTrial(len(self.trials), fixed)
        self.trials.append(trial)
        return trial

    def tell(self, trial, values=None, state=None):
        if trial.number in getattr(self, "refuse", ()):
            raise LookupError(f"trial {trial.number}")
        if state is None:
            self.told.append((trial.number, values))
        else:
            self.failed.append((trial.number, state))

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

    @OTHER_CANDIDATES
    def test_a_tell_with_other_candidates_raises(self, other):
        study = FakeStudy()
        adapter = from_optuna(study, 2, 2)
        batch = adapter.ask()
        with pytest.raises(ValueError, match="other candidates than the batch"):
            adapter.tell(other(batch), [1.0, 2.0])
        assert study.told == []
        adapter.tell(batch, [1.0, 2.0])
        assert study.told == [(0, -1.0), (1, -2.0)]

    @pytest.fixture
    def trial_state(self, monkeypatch):
        """A stand-in for ``optuna.trial.TrialState``, which ``close()``
        imports."""
        trial = types.ModuleType("optuna.trial")
        trial.TrialState = types.SimpleNamespace(FAIL="FAIL")
        optuna = types.ModuleType("optuna")
        optuna.trial = trial
        monkeypatch.setitem(sys.modules, "optuna", optuna)
        monkeypatch.setitem(sys.modules, "optuna.trial", trial)

    @pytest.mark.usefixtures("trial_state")
    def test_close_fails_the_trials_of_a_batch_not_told(self):
        study = FakeStudy()
        adapter = from_optuna(study, 2, 3)
        adapter.close()
        assert study.failed == [], "no batch handed out, nothing to fail"
        batch = adapter.ask()
        adapter.close()
        assert study.failed == [(0, "FAIL"), (1, "FAIL"), (2, "FAIL")]
        assert study.told == []
        with pytest.raises(RuntimeError, match="no batch pending"):
            adapter.tell(batch, [1.0, 2.0, 3.0])
        # The adapter goes on with new trials.
        assert adapter.ask() != batch
        assert len(study.trials) == 6
        adapter.close()
        adapter.close()
        assert [number for number, _ in study.failed] == [0, 1, 2, 3, 4, 5]

    @pytest.mark.usefixtures("trial_state")
    def test_close_after_a_tell_fails_nothing(self):
        study = FakeStudy()
        adapter = from_optuna(study, 2, 2)
        adapter.tell(adapter.ask(), [1.0, 2.0])
        adapter.close()
        assert study.failed == []

    @pytest.mark.usefixtures("trial_state")
    def test_a_close_that_fails_partway_keeps_the_rest_pending(self):
        study = FakeStudy()
        study.refuse = {1}
        adapter = from_optuna(study, 2, 3)
        batch = adapter.ask()
        with pytest.raises(LookupError, match="trial 1"):
            adapter.close()
        assert study.failed == [(0, "FAIL")]
        assert adapter.ask() == batch[1:]
        study.refuse = set()
        adapter.close()
        assert study.failed == [(0, "FAIL"), (1, "FAIL"), (2, "FAIL")]

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
            ({"bounds": (1.0, 1.0)}, "bounds must be finite with low < high"),
            ({"bounds": (0.0, math.inf)}, "bounds must be finite with low < high"),
            ({"bounds": 3}, "bounds must be two numbers (low, high), got 3"),
            ({"initial_point": ["a", 0.5]}, "must be 2 numbers, got ['a', 0.5]"),
            ({"initial_point": [0.5]}, "must have one value per dimension (2), got 1"),
            ({"initial_point": [0.5, math.nan]}, "every value must be finite"),
            ({"initial_point": [0.5, 7.0]}, "must lie within bounds"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, message):
        arguments = {"dimensions": 2, "batch_size": 2, **kwargs}
        study = FakeStudy()
        with pytest.raises(polypus.ConfigurationError, match=re.escape(message)):
            from_optuna(study, **arguments)
        assert study.queue == []

    @pytest.mark.parametrize("name", ["dimensions", "batch_size"])
    @pytest.mark.parametrize("value", [2.0, True, "2", None])
    def test_a_count_that_is_not_an_integer_is_a_type_error(self, name, value):
        # As polypus.External's max_evaluations, and an integer argument's.
        arguments = {"dimensions": 2, "batch_size": 2, name: value}
        study = FakeStudy()
        message = f"{name} must be an integer >= 1, got {value!r}"
        with pytest.raises(TypeError, match=re.escape(message)):
            from_optuna(study, **arguments)
        assert study.queue == []

    def test_a_study_with_several_objectives_raises(self):
        with pytest.raises(
            polypus.ConfigurationError, match="single-objective study, got one with 2"
        ):
            from_optuna(FakeStudy("MINIMIZE", "MAXIMIZE"), 2, 2)
        with pytest.raises(
            polypus.ConfigurationError, match="maximizes or minimizes, got NOT_SET"
        ):
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

    @OTHER_CANDIDATES
    def test_a_tell_with_other_candidates_raises(self, other):
        optimizer = FakeNevergrad()
        adapter = from_nevergrad(optimizer, batch_size=2)
        batch = adapter.ask()
        with pytest.raises(ValueError, match="other candidates than the batch"):
            adapter.tell(other(batch), [1.0, 2.0])
        assert optimizer.told == []
        adapter.tell(batch, [1.0, 2.0])
        assert [loss for _, loss in optimizer.told] == [-1.0, -2.0]

    def test_the_initial_point_is_suggested(self):
        optimizer = FakeNevergrad()
        adapter = from_nevergrad(optimizer, batch_size=2, initial_point=(0.5, 1.5))
        (suggested,) = optimizer.suggested
        # As an array of floats, the value of an ``Array`` parametrization.
        assert isinstance(suggested, np.ndarray) and suggested.dtype == float
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
        with pytest.raises(polypus.ConfigurationError, match=message):
            from_nevergrad(optimizer, **kwargs)

    @pytest.mark.parametrize("value", [2.0, True])
    def test_a_batch_size_that_is_not_an_integer_is_a_type_error(self, value):
        message = f"batch_size must be an integer >= 1, got {value!r}"
        with pytest.raises(TypeError, match=re.escape(message)):
            from_nevergrad(FakeNevergrad(), batch_size=value)

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
        first = adapter.ask()
        assert adapter.ask() == first
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

    @OTHER_CANDIDATES
    def test_a_tell_with_other_candidates_raises(self, other):
        es = FakeES(popsize=2)
        adapter = from_pycma(es)
        batch = adapter.ask()
        with pytest.raises(ValueError, match="other candidates than the batch"):
            adapter.tell(other(batch), [1.0, 2.0])
        assert es.told == []
        # The same rows as a NumPy array are the same candidates.
        adapter.tell(np.array(batch), [1.0, 2.0])
        assert es.told[0][1] == [-1.0, -2.0]

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
        optuna = _library("optuna")
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
        optuna = _library("optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def run():
            study = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=7, constant_liar=True)
            )
            return _train(from_optuna(study, DIMENSIONS, 3), 15, seed=11)

        first, second = run(), run()
        assert first.best_params == second.best_params
        assert first.fitness_history == second.fitness_history

    def test_close_leaves_no_trial_running(self):
        optuna = _library("optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        state = optuna.trial.TrialState
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=2, constant_liar=True)
        )
        adapter = from_optuna(study, DIMENSIONS, 4)
        # The budget leaves the third batch untold: its trials stay RUNNING.
        _train(adapter, 10)
        assert [t.state for t in study.trials] == [state.COMPLETE] * 8 + [
            state.RUNNING
        ] * 4
        adapter.close()
        assert [t.state for t in study.trials] == [state.COMPLETE] * 8 + [
            state.FAIL
        ] * 4
        # The adapter goes on with new trials, which the study completes.
        _train(adapter, 4)
        assert [t.state for t in study.trials[12:]] == [state.COMPLETE] * 4
        adapter.close()
        assert len(study.trials) == 16


@pytest.mark.integration
class TestNevergrad:
    def test_ngopt_through_the_bridge(self):
        ng = _library("nevergrad")
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
        cma = _library("cma")
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
