"""
``polypus.External`` (issue #399): an external ask/tell optimizer run through
Polypus's scheduler.

Each ``ask()`` is one oracle batch; its values go back through ``tell()``. The
run stops at an empty ``ask()`` or before the batch that would take it past
``max_evaluations``. These tests drive the bridge with the deterministic
``RandomSearch`` of ``ask_tell_helpers.py``; the adapters for Optuna,
Nevergrad and pycma are in ``test_ask_tell_adapters.py``.

The configuration tests need no backend. ``initial_point`` is refused before
anything runs: with ``infrastructure="cunqa"``, before
``connect_to_infrastructure`` (and so ``qraise``) is reached. The batch tests
run the real training loops on Aer with ``polypus_python.run_qcs`` replaced by
a fake that scores each circuit by its parameters and records it, so they
check what was evaluated and told; one runs over a fake CUNQA family to check
the QPUs each batch is spread over. The C-5 tests train on the native
simulator over several seeds. Seeded reproducibility is in
``test_seed_reproducibility.py`` and the final batch of ``final_shots`` in
``test_final_shots.py``.
"""

import gc
import itertools
import math
import re
import threading
import warnings
import weakref

import numpy as np
import pytest
from ask_tell_helpers import RandomSearch

DIMENSIONS = 2
SHOTS = 64

_RY = re.compile(r"^ry\(([^)]*)\) q\[(\d+)\];$", re.MULTILINE)


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


def _circuit():
    import polypus

    qc = polypus.Circuit(DIMENSIONS)
    for q in range(DIMENSIONS):
        qc.ry(q, polypus.Param(q))
    qc.measure_all()
    return qc


def _external(optimizer, max_evaluations=100):
    import polypus

    return polypus.External(optimizer, max_evaluations)


def _train(method, *, infrastructure="local", backend="aer", **kwargs):
    import polypus

    return polypus.train(
        _circuit(),
        method,
        shots=SHOTS,
        n_qpus=kwargs.pop("n_qpus", 1),
        dimensions=DIMENSIONS,
        expectation_function=_all_ones,
        infrastructure=infrastructure,
        nodes=1,
        cores_per_qpu=1,
        id="external",
        backend=backend,
        seed=kwargs.pop("seed", 11),
        **kwargs,
    )


ROWS = [[0.3], [1.1], [2.0]]


def _qml_train(method, *, infrastructure="local", **kwargs):
    """Three samples, so each candidate is three circuits."""
    import polypus
    from qiskit.circuit import ParameterVector, QuantumCircuit

    x = ParameterVector("x", 1)
    theta = ParameterVector("theta", DIMENSIONS)
    feature_map = QuantumCircuit(1)
    feature_map.rx(x[0], 0)
    ansatz = QuantumCircuit(1)
    ansatz.ry(theta[0], 0)
    ansatz.rz(theta[1], 0)
    return polypus.qml.train(
        feature_map,
        ansatz,
        ROWS,
        method,
        shots=SHOTS,
        n_qpus=1,
        dimensions=DIMENSIONS,
        expectation_function=_all_ones,
        infrastructure=infrastructure,
        nodes=1,
        cores_per_qpu=1,
        id="external_qml",
        seed=11,
        **kwargs,
    )


# The runner and the circuits each candidate is.
ENTRY_POINTS = {"train": (_train, 1), "qml.train": (_qml_train, len(ROWS))}


def _probability(candidate):
    a, b = candidate
    return (1.0 + math.sin(3.0 * a + 1.3) * math.cos(2.0 * b - 0.4)) / 2.0


def _record(monkeypatch):
    """Replace the local seam with a fake that records each batch's size and
    measures all ones on ``round(shots · p)`` shots, ``p`` a function of the
    circuit's ``ry`` angles (a ``train`` circuit is OpenQASM, whose angles are
    exported at full precision). ``qml.train`` circuits are Qiskit circuits:
    all ones on half the shots."""
    import polypus_python

    sizes = []

    def fake_run_qcs(infrastructure, **kwargs):
        shots, results = kwargs["shots"], []
        sizes.append(len(kwargs["qcs"]))
        for qc in kwargs["qcs"]:
            if isinstance(qc, str):
                angles = {int(q): float(a) for a, q in _RY.findall(qc)}
                ones = round(shots * _probability([angles[0], angles[1]]))
                results.append({"11": ones, "00": shots - ones})
            else:
                results.append({"1": shots // 2, "0": shots - shots // 2})
        return [{k: v for k, v in r.items() if v} for r in results]

    monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)
    return sizes


class _Stop(Exception):
    """Raised by the recorder to end the run at the CUNQA seam."""


@pytest.fixture
def seam_calls(monkeypatch):
    """Record every ``connect_to_infrastructure`` call (the one that raises
    CUNQA QPUs) and stop the run there."""
    import polypus_python

    calls = []

    def record(infrastructure, **kwargs):
        calls.append((infrastructure, kwargs))
        raise _Stop

    monkeypatch.setattr(polypus_python, "connect_to_infrastructure", record)
    return calls


# ─────────────────────────────────────────────────────────────────────────────
# The object
# ─────────────────────────────────────────────────────────────────────────────


class TestTheObject:
    def test_the_fields_and_setters(self):
        import polypus

        first, second = RandomSearch(DIMENSIONS), RandomSearch(DIMENSIONS)
        method = polypus.External(first, 30)
        assert method.optimizer is first
        assert method.max_evaluations == 30
        method.optimizer = second
        method.max_evaluations = np.int64(7)
        assert method.optimizer is second
        assert method.max_evaluations == 7
        assert type(method).__module__ == "polypus"
        by_name = polypus.External(optimizer=first, max_evaluations=5)
        assert (by_name.optimizer, by_name.max_evaluations) == (first, 5)
        assert not hasattr(method, "seed"), "the library owns its seed (C-7)"

    def test_both_arguments_are_required(self):
        import polypus

        with pytest.raises(TypeError):
            polypus.External(RandomSearch(DIMENSIONS))
        with pytest.raises(TypeError):
            polypus.External()

    @pytest.mark.parametrize(
        ("value", "shown", "error"),
        [
            (0, "0", ValueError),
            (-1, "-1", ValueError),
            # Not an integer: a TypeError, as for an integer argument.
            (2.5, "2.5", TypeError),
            (10.0, "10.0", TypeError),
            (True, "True", TypeError),
            ("10", "'10'", TypeError),
            (None, "None", TypeError),
            ([10], "[10]", TypeError),
        ],
    )
    def test_max_evaluations_must_be_a_positive_integer(self, value, shown, error):
        import polypus

        message = f"max_evaluations must be an integer >= 1, got {shown}"
        with pytest.raises(error, match=re.escape(message)) as raised:
            polypus.External(RandomSearch(DIMENSIONS), value)
        method = polypus.External(RandomSearch(DIMENSIONS), 3)
        with pytest.raises(error, match=re.escape(message)):
            method.max_evaluations = value
        assert method.max_evaluations == 3
        # Only the out-of-range integers are configuration errors.
        assert isinstance(raised.value, polypus.ConfigurationError) == (
            error is ValueError
        )

    def test_max_evaluations_fits_in_64_bits(self):
        import polypus

        assert polypus.External(RandomSearch(1), 2**64 - 1).max_evaluations == 2**64 - 1
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"max_evaluations must be at most {2**64 - 1}, got {2**64}"
            ),
        ):
            polypus.External(RandomSearch(1), 2**64)

    @pytest.mark.parametrize("raised", [LookupError, KeyboardInterrupt])
    def test_an_exception_from_index_propagates_unchanged(self, raised):
        import polypus

        class _Index:
            def __index__(self):
                raise raised("from __index__")

        with pytest.raises(raised, match="from __index__"):
            polypus.External(RandomSearch(1), _Index())

    def test_the_optimizer_needs_callable_ask_and_tell(self):
        import polypus

        class AskOnly:
            def ask(self):
                return []

        class TellNotCallable:
            tell = 3

            def ask(self):
                return []

        for optimizer, missing in [
            (object(), "ask"),
            (AskOnly(), "tell"),
            (TellNotCallable(), "tell"),
        ]:
            with pytest.raises(TypeError, match=rf"has no callable {missing}\(\)"):
                polypus.External(optimizer, 10)
        method = polypus.External(RandomSearch(1), 10)
        with pytest.raises(TypeError, match="callable ask"):
            method.optimizer = object()
        assert isinstance(method.optimizer, RandomSearch)

    def test_an_unknown_method_names_external(self):
        with pytest.raises(TypeError, match="or polypus.External"):
            _train(object())

    def test_a_cycle_through_the_optimizer_is_collected(self):
        """An optimizer that holds its own ``External`` makes a cycle, which
        the garbage collector frees: ``External`` traverses its optimizer and
        clears it."""
        optimizer = RandomSearch(DIMENSIONS)
        optimizer.external = _external(optimizer, 10)
        assert gc.is_tracked(optimizer.external)
        assert gc.get_referents(optimizer.external) == [optimizer]
        gone = weakref.ref(optimizer)
        del optimizer
        gc.collect()
        assert gone() is None


# ─────────────────────────────────────────────────────────────────────────────
# Before anything runs
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("entry", ENTRY_POINTS)
class TestBeforeAnythingRuns:
    @pytest.mark.parametrize("point", [[0.5, 0.5], [0.5], [float("nan"), 0.0]])
    def test_an_initial_point_raises(self, seam_calls, entry, point):
        run = ENTRY_POINTS[entry][0]
        optimizer = RandomSearch(DIMENSIONS)
        with pytest.raises(ValueError) as raised:
            run(_external(optimizer), infrastructure="cunqa", initial_point=point)
        message = str(raised.value)
        assert message.startswith(
            "polypus.External takes no initial_point: the external optimizer chooses "
            "its own start. Give the starting point to the library, or to the "
            "initial_point of the polypus.ask_tell adapter"
        ), message
        assert seam_calls == []
        assert optimizer.asked == []

    def test_a_valid_external_gets_to_the_seam_before_any_ask(self, seam_calls, entry):
        # The control: the backend, and so the CUNQA family, comes first.
        run = ENTRY_POINTS[entry][0]
        optimizer = RandomSearch(DIMENSIONS)
        with pytest.raises(_Stop):
            run(_external(optimizer), infrastructure="cunqa")
        assert len(seam_calls) == 1
        assert optimizer.asked == []


# ─────────────────────────────────────────────────────────────────────────────
# Batches
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.vqc
class TestBatches:
    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_each_ask_is_one_batch_and_its_values_are_told(self, monkeypatch, entry):
        run, circuits = ENTRY_POINTS[entry]
        sizes = _record(monkeypatch)
        optimizer = RandomSearch(DIMENSIONS, sizes=(3, 1, 4), batches=4)
        result = run(_external(optimizer, 100))
        asked = [len(batch) for batch in optimizer.asked]
        assert asked == [3, 1, 4, 3]
        assert sizes == [circuits * n for n in asked]
        # tell gets every row asked, in order, and one value per row.
        assert [candidates for candidates, _ in optimizer.told] == optimizer.asked
        for candidates, values in optimizer.told:
            assert len(values) == len(candidates)
            assert all(isinstance(v, float) for v in values)
            assert all(isinstance(x, float) for row in candidates for x in row)
        assert result.converged
        assert result.iterations_run == 4
        assert len(result.fitness_history) == 4

    def test_the_values_told_are_the_measured_fitness(self, monkeypatch):
        _record(monkeypatch)
        optimizer = RandomSearch(DIMENSIONS, sizes=(5,), batches=2)
        _train(_external(optimizer))
        for candidates, values in optimizer.told:
            for candidate, value in zip(candidates, values):
                ones = round(SHOTS * _probability(candidate))
                assert value == ones / SHOTS

    def test_the_budget_is_never_exceeded(self, monkeypatch):
        # Three batches of three make nine; a fourth would make twelve.
        sizes = _record(monkeypatch)
        optimizer = RandomSearch(DIMENSIONS, sizes=(3,))
        result = _train(_external(optimizer, 10))
        assert sizes == [3, 3, 3]
        assert len(optimizer.asked) == 4, "the fourth batch was asked"
        assert len(optimizer.told) == 3, "and neither evaluated nor told"
        assert not result.converged
        assert result.iterations_run == 3
        assert len(result.fitness_history) == 3

    def test_a_budget_spent_exactly_stops_without_another_ask(self, monkeypatch):
        sizes = _record(monkeypatch)
        optimizer = RandomSearch(DIMENSIONS, sizes=(3,))
        result = _train(_external(optimizer, 9))
        assert sizes == [3, 3, 3]
        assert len(optimizer.asked) == 3
        assert not result.converged

    @pytest.mark.parametrize("end", [[], None], ids=["empty", "None"])
    def test_an_empty_ask_ends_the_run(self, monkeypatch, end):
        sizes = _record(monkeypatch)
        optimizer = RandomSearch(DIMENSIONS, sizes=(2,), batches=2, end=end)
        result = _train(_external(optimizer, 1000))
        assert sizes == [2, 2]
        assert result.converged
        assert result.iterations_run == 2

    def test_a_numpy_batch_is_accepted(self, monkeypatch):
        sizes = _record(monkeypatch)

        class NumpyRows(RandomSearch):
            def ask(self):
                batch = super().ask()
                return np.asarray(batch) if batch else batch

        optimizer = NumpyRows(DIMENSIONS, sizes=(3,), batches=2)
        result = _train(_external(optimizer))
        assert sizes == [3, 3]
        assert [c for c, _ in optimizer.told] == optimizer.asked
        assert result.converged

    @pytest.mark.parametrize(
        ("batch", "kind"),
        [
            (np.array([[1 + 2j, 0.5]]), "a complex number; every value must be real"),
            (np.array([[True, False]]), "a bool, not a number"),
        ],
        ids=["complex", "bool"],
    )
    def test_a_numpy_complex_or_bool_batch_raises(self, monkeypatch, batch, kind):
        # Rejected before float() would drop the imaginary part with a
        # ComplexWarning (an error here) or read a bool as 1.0.
        sizes = _record(monkeypatch)

        class Fixed:
            told = []

            def ask(self):
                return batch

            def tell(self, candidates, values):
                self.told.append(values)

        optimizer = Fixed()
        with warnings.catch_warnings():
            # `numpy.exceptions` is NumPy 1.25 and later.
            exceptions = getattr(np, "exceptions", np)
            warnings.simplefilter("error", exceptions.ComplexWarning)
            with pytest.raises(ValueError) as raised:
                _train(_external(optimizer))
        # The value's repr is NumPy's, which differs between its versions.
        message = str(raised.value)
        assert message.startswith(
            "invalid candidate in row 0 of batch 0 from the external optimizer's "
            "ask() (both 0-based): its value at index 0 is "
        ), message
        assert message.endswith(f", {kind}"), message
        assert sizes == []
        assert optimizer.told == []

    def test_an_endless_batch_is_read_one_row_past_the_budget(self, monkeypatch):
        sizes = _record(monkeypatch)

        class Endless(RandomSearch):
            read = 0

            def rows(self):
                for _ in itertools.count():
                    self.read += 1
                    yield [0.5, 0.5]

            def ask(self):
                return self.rows() if self.told else super().ask()

        # After a batch of 3, 7 of the budget of 10 are left: the second
        # batch is read up to 8 rows, which do not fit, and is not evaluated.
        optimizer = Endless(DIMENSIONS, sizes=(3,))
        result = _train(_external(optimizer, 10))
        assert sizes == [3]
        assert optimizer.read == 8
        assert len(optimizer.told) == 1
        assert result.iterations_run == 1
        assert not result.converged

    def test_an_endless_first_batch_raises(self, monkeypatch):
        sizes = _record(monkeypatch)

        class Endless(RandomSearch):
            def ask(self):
                return itertools.repeat([0.5, 0.5])

        with pytest.raises(
            ValueError,
            match=re.escape(
                "the external optimizer's first batch has more than max_evaluations "
                "(5) candidates"
            ),
        ):
            _train(_external(Endless(DIMENSIONS), 5))
        assert sizes == []

    def test_nothing_on_the_first_ask_raises(self, monkeypatch):
        sizes = _record(monkeypatch)
        optimizer = RandomSearch(DIMENSIONS, batches=0)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "external optimizer: ask() proposed no candidates on its first call, "
                "so nothing was evaluated"
            ),
        ):
            _train(_external(optimizer))
        assert sizes == []

    def test_a_first_batch_larger_than_the_budget_raises(self, monkeypatch):
        sizes = _record(monkeypatch)
        optimizer = RandomSearch(DIMENSIONS, sizes=(6,))
        with pytest.raises(
            ValueError,
            match=re.escape(
                "invalid max_evaluations: the external optimizer's first batch has "
                "more than max_evaluations (5) candidates; nothing was evaluated"
            ),
        ):
            _train(_external(optimizer, 5))
        assert sizes == []
        assert optimizer.told == []

    @pytest.mark.parametrize(
        ("row", "reason"),
        [
            ([0.5], "it has 1 values, expected one per dimension (2)"),
            (
                [0.5, 0.5, 0.5],
                "it has more than 2 values, expected one per dimension (2)",
            ),
            # An endless row is read only one value past the dimensions.
            (
                itertools.repeat(0.5),
                "it has more than 2 values, expected one per dimension (2)",
            ),
            ([0.5, float("nan")], "its value at index 1 is NaN; every value must be"),
            ([float("inf"), 0.5], "its value at index 0 is inf; every value must be"),
            ([0.5, "x"], "its value at index 1 is 'x', not a number"),
            ([0.5, None], "its value at index 1 is None, not a number"),
            ([True, 0.5], "its value at index 0 is True, a bool, not a number"),
            (
                [0.5, 1 + 2j],
                "its value at index 1 is (1+2j), a complex number; every value must "
                "be real",
            ),
            ([0.5, 10**400], "its value at index 1 is 1000000000"),
            (7, "it is 7, not a sequence of numbers"),
        ],
        ids=[
            "short",
            "long",
            "endless",
            "nan",
            "inf",
            "str",
            "None",
            "bool",
            "complex",
            "huge",
            "scalar",
        ],
    )
    def test_a_bad_row_raises_and_its_batch_is_not_evaluated(
        self, monkeypatch, row, reason
    ):
        sizes = _record(monkeypatch)

        class BadSecondBatch(RandomSearch):
            def ask(self):
                batch = super().ask()
                if len(self.asked) == 2:
                    batch[1] = row
                return batch

        optimizer = BadSecondBatch(DIMENSIONS, sizes=(3,))
        with pytest.raises(ValueError) as raised:
            _train(_external(optimizer))
        assert str(raised.value).startswith(
            "invalid candidate in row 1 of batch 1 from the external optimizer's "
            f"ask() (both 0-based): {reason}"
        ), str(raised.value)
        assert sizes == [3]
        assert len(optimizer.told) == 1

    def test_a_batch_that_is_not_rows_raises(self, monkeypatch):
        sizes = _record(monkeypatch)

        class Scalar(RandomSearch):
            def ask(self):
                return 3.5 if self.asked else super().ask()

        with pytest.raises(
            ValueError,
            match=re.escape(
                "the external optimizer's ask() must return None or an iterable of "
                "rows, got 3.5 for batch 1 (0-based)"
            ),
        ):
            _train(_external(Scalar(DIMENSIONS, sizes=(2,))))
        assert sizes == [2]

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    @pytest.mark.parametrize("call", ["ask", "tell"])
    @pytest.mark.parametrize("raised", [LookupError, KeyboardInterrupt])
    def test_an_exception_from_ask_or_tell_keeps_its_class(
        self, monkeypatch, entry, call, raised
    ):
        run = ENTRY_POINTS[entry][0]
        sizes = _record(monkeypatch)

        class Failing(RandomSearch):
            def ask(self):
                if call == "ask" and len(self.asked) == 1:
                    raise raised("from ask")
                return super().ask()

            def tell(self, candidates, values):
                super().tell(candidates, values)
                if call == "tell" and len(self.told) == 2:
                    raise raised("from tell")

        optimizer = Failing(DIMENSIONS, sizes=(2,))
        with pytest.raises(raised, match=f"from {call}") as caught:
            run(_external(optimizer))
        assert type(caught.value) is raised
        # The run stopped there: one batch before a failing ask, two before a
        # failing second tell.
        assert len(sizes) == (1 if call == "ask" else 2)

    def test_a_failing_cost_is_not_told(self, monkeypatch):
        # A failure in a batch ends the run before tell: the library is never
        # told the oracle's placeholder values. The cost fails once the first
        # batch has been told.
        _record(monkeypatch)
        told_once = []

        def failing(bitstring):
            if told_once:
                raise RuntimeError("cost failed")
            return _all_ones(bitstring)

        class FailAfterTheFirstTell(RandomSearch):
            def tell(self, candidates, values):
                super().tell(candidates, values)
                told_once.append(True)

        import polypus

        optimizer = FailAfterTheFirstTell(DIMENSIONS, sizes=(4,))
        with pytest.raises(RuntimeError, match="cost failed"):
            polypus.train(
                _circuit(),
                _external(optimizer),
                shots=SHOTS,
                n_qpus=1,
                dimensions=DIMENSIONS,
                expectation_function=failing,
                infrastructure="local",
                nodes=1,
                cores_per_qpu=1,
                id="external_cost",
                backend="aer",
                seed=11,
            )
        assert len(optimizer.asked) == 2
        assert len(optimizer.told) == 1

    def test_ask_and_tell_run_on_the_calling_thread(self, monkeypatch):
        _record(monkeypatch)
        threads = set()

        class Recording(RandomSearch):
            def ask(self):
                threads.add(threading.get_ident())
                return super().ask()

            def tell(self, candidates, values):
                threads.add(threading.get_ident())
                super().tell(candidates, values)

        _train(_external(Recording(DIMENSIONS, sizes=(2,), batches=3)))
        assert threads == {threading.get_ident()}

    def test_a_second_run_continues_the_library(self, monkeypatch):
        # The object is used, not copied: its state carries over, and the
        # budget counts each run on its own.
        sizes = _record(monkeypatch)
        optimizer = RandomSearch(DIMENSIONS, sizes=(3,))
        method = _external(optimizer, 6)
        first = _train(method)
        second = _train(method)
        assert sizes == [3, 3, 3, 3]
        assert len(optimizer.asked) == 4
        assert first.iterations_run == second.iterations_run == 2
        assert first.best_params in optimizer.asked[0] + optimizer.asked[1]
        assert second.best_params in optimizer.asked[2] + optimizer.asked[3]

    def test_each_batch_is_spread_over_the_qpus(self, monkeypatch):
        # Over a fake CUNQA family of two QPUs, a batch of five is three waves
        # (QPUs 0 and 1, 0 and 1, then 0) and the next batch starts again at
        # QPU 0: one ask, one batch.
        pytest.importorskip("qiskit_aer")
        from fake_cunqa import FakeCunqa

        fake = FakeCunqa().install(monkeypatch)
        optimizer = RandomSearch(DIMENSIONS, sizes=(5, 3), batches=2)
        result = _train(_external(optimizer), infrastructure="cunqa", n_qpus=2)
        assert [run["qpu"].index for run in fake.runs] == [0, 1, 0, 1, 0, 0, 1, 0]
        assert fake.names().count("gather") == 3 + 2
        assert all(run["qpu"].family == result.id for run in fake.runs)
        assert result.iterations_run == 2


# ─────────────────────────────────────────────────────────────────────────────
# Contract C-5 on the native simulator
# ─────────────────────────────────────────────────────────────────────────────


def _native_train(method, seed):
    """A seeded run on the native simulator. The third qubit is in |+>, so no
    value is 0 or 1 and the values differ from candidate to candidate."""
    import polypus

    qc = polypus.Circuit(3)
    qc.ry(0, polypus.Param(0))
    qc.ry(1, polypus.Param(1))
    qc.h(2)
    qc.measure_all()
    return polypus.train(
        qc,
        method,
        shots=SHOTS,
        n_qpus=1,
        dimensions=DIMENSIONS,
        expectation_function=_all_ones,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="external_native",
        backend="polypus",
        seed=seed,
    )


@pytest.mark.integration
@pytest.mark.vqc
class TestContractC5:
    """The outcome is what the bridge measured, checked against what ``tell``
    received, which is every candidate evaluated and its value."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("batches", [None, 3], ids=["budget", "empty ask"])
    def test_the_outcome_is_the_best_measured(self, seed, batches):
        optimizer = RandomSearch(
            DIMENSIONS, sizes=(3, 5, 2), seed=seed, batches=batches
        )
        result = _native_train(_external(optimizer, 20), seed)
        told = optimizer.told
        # 3 + 5 + 2 + 3 + 5 + 2 = 20: the budget is spent exactly.
        assert len(told) == (6 if batches is None else 3)
        assert result.iterations_run == len(told)
        assert result.converged is (batches is not None)
        # best: the first candidate with the highest value.
        scored = [
            (v, c) for candidates, values in told for c, v in zip(candidates, values)
        ]
        top = max(v for v, _ in scored)
        first = next(c for v, c in scored if v == top)
        assert result.best_fitness == top
        assert result.best_params == first
        # One history entry per batch: the best measured up to it.
        running, history = -math.inf, []
        for _, values in told:
            running = max(running, *values)
            history.append(running)
        assert result.fitness_history == history
        assert result.fitness_history[-1] == result.best_fitness
        assert 0.0 < result.best_fitness < 1.0
        assert result.seed == seed
