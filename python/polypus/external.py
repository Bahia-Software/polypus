"""Adapters that run Optuna, Nevergrad and pycma through ``polypus.External``.

``polypus.External(optimizer, max_evaluations)`` runs any object with two
methods, ``ask()`` and ``tell(candidates, values)``, in Polypus's convention:
``ask()`` returns the next batch of rows (``None`` or an empty batch when the
optimizer is done), and ``tell`` receives the rows of that batch and their
fitness, to be MAXIMISED. The functions here wrap the public ask/tell API of
three libraries in that shape:

- :func:`from_optuna` wraps an Optuna study (for example TPE, Bayesian
  optimisation), asking ``batch_size`` trials per batch and telling each its
  value, negated when the study minimises.
- :func:`from_nevergrad` wraps a Nevergrad optimizer, which minimises: it is
  told ``-value``, and its ``budget``, when set, ends the run.
- :func:`from_pycma` wraps a pycma ``CMAEvolutionStrategy``, which minimises:
  it is told ``-value``, and its ``stop()`` ends the run.

None of the three libraries is a dependency of Polypus. The adapters only call
methods of the object you built with the library you installed, so this module
imports none of them.

Each adapter re-offers a batch it handed out and was not told about, instead of
asking the library again: the bridge leaves a batch untold when it would take
the run past ``max_evaluations``, and a later run with the same object starts
with it. A ``tell`` that the library fails partway keeps the candidates it did
not tell, to be offered again; a ``tell`` with no batch handed out raises
``RuntimeError``. The library keeps its state between runs, its seed included: seed it
where you build it (contract C-7). Its start is its own too, so
``polypus.External`` refuses ``initial_point``; :func:`from_optuna` and
:func:`from_nevergrad` take one here, and pycma's is the ``x0`` of
``CMAEvolutionStrategy``.
"""

import math
import operator
from typing import Any, Iterable, Optional, Protocol, Sequence

__all__ = ["AskTell", "from_optuna", "from_nevergrad", "from_pycma"]


class AskTell(Protocol):
    """What ``polypus.External`` runs: ``ask()`` returns the next batch of
    rows (a list of lists, a 2-D NumPy array), each one value per circuit
    parameter, and ``None`` or an empty batch ends the run;
    ``tell(candidates, values)`` receives that batch, as a
    ``list[list[float]]``, and its fitness to maximise, as a ``list[float]``.
    Its return value is ignored.
    """

    def ask(self) -> Optional[Iterable[Iterable[float]]]: ...

    def tell(self, candidates: list[list[float]], values: list[float], /) -> object: ...


def _positive_int(value: Any, name: str) -> int:
    """``value`` as an ``int >= 1``; a ``bool`` or a non-integer is a
    ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    try:
        number = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}") from None
    if number < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return number


def _bounds(bounds: Any) -> tuple[float, float]:
    """``(low, high)``, both finite and ``low < high``."""
    try:
        low, high = (float(b) for b in bounds)
    except (TypeError, ValueError):
        raise ValueError(
            f"bounds must be two numbers (low, high), got {bounds!r}"
        ) from None
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError(
            f"bounds must be finite with low < high, got ({low!r}, {high!r})"
        )
    return low, high


def _point(
    point: Any, dimensions: int, bounds: Optional[tuple[float, float]] = None
) -> list[float]:
    """``point`` as ``dimensions`` finite floats, within ``bounds`` when given."""
    try:
        values = [float(x) for x in point]
    except (TypeError, ValueError):
        raise ValueError(
            f"invalid initial_point: must be {dimensions} numbers, got {point!r}"
        ) from None
    if len(values) != dimensions:
        raise ValueError(
            f"invalid initial_point: must have one value per dimension "
            f"({dimensions}), got {len(values)}"
        )
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(
                f"invalid initial_point: every value must be finite, got {value} "
                f"at index {index}"
            )
        if bounds is not None and not bounds[0] <= value <= bounds[1]:
            raise ValueError(
                f"invalid initial_point: every value must lie within bounds "
                f"[{bounds[0]}, {bounds[1]}], got {value} at index {index}"
            )
    return values


def _check_told(pending: Optional[list[Any]], values: list[float]) -> list[Any]:
    """The candidates awaiting ``values``: ``RuntimeError`` when no batch was
    handed out, ``ValueError`` when ``values`` does not have one per candidate."""
    if pending is None:
        raise RuntimeError(
            "tell() was called with no batch pending: each batch that ask() hands "
            "out is told once"
        )
    if len(values) != len(pending):
        raise ValueError(
            f"tell() got {len(values)} values for a batch of {len(pending)} candidates"
        )
    return pending


def _row(value: Any) -> list[float]:
    """A candidate's value as a flat list of floats: a 1-D NumPy array or a
    sequence of numbers, otherwise ``ValueError``."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        raise ValueError(
            "a candidate of the library must be a flat sequence of numbers, one "
            f"per circuit parameter, got {value!r}"
        ) from None


class _Optuna:
    def __init__(
        self,
        study: Any,
        dimensions: int,
        batch_size: int,
        bounds: tuple[float, float],
        initial_point: Optional[Sequence[float]],
    ) -> None:
        self.study = study
        self.dimensions = _positive_int(dimensions, "dimensions")
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.low, self.high = _bounds(bounds)
        directions = list(study.directions)
        if len(directions) != 1:
            raise ValueError(
                "from_optuna needs a single-objective study, got one with "
                f"{len(directions)} directions"
            )
        name = getattr(directions[0], "name", str(directions[0]))
        if name not in ("MAXIMIZE", "MINIMIZE"):
            raise ValueError(
                f"from_optuna needs a study that maximizes or minimizes, got {name}"
            )
        # Polypus maximises: a study that minimises is told -value.
        self.sign = 1.0 if name == "MAXIMIZE" else -1.0
        self.pending: Optional[list[Any]] = None
        self.rows: list[list[float]] = []
        if initial_point is not None:
            point = _point(initial_point, self.dimensions, (self.low, self.high))
            study.enqueue_trial({self._name(i): x for i, x in enumerate(point)})

    @staticmethod
    def _name(index: int) -> str:
        return f"x{index}"

    def ask(self) -> list[list[float]]:
        if self.pending is None:
            trials = [self.study.ask() for _ in range(self.batch_size)]
            self.rows = [
                [
                    trial.suggest_float(self._name(i), self.low, self.high)
                    for i in range(self.dimensions)
                ]
                for trial in trials
            ]
            self.pending = trials
        return [list(row) for row in self.rows]

    def tell(self, candidates: list[list[float]], values: list[float]) -> None:
        trials = _check_told(self.pending, values)
        # One at a time, so a tell that raises keeps the rest pending.
        for value in values:
            self.study.tell(trials[0], self.sign * value)
            del trials[0], self.rows[0]
        self.pending = None


class _Nevergrad:
    def __init__(
        self,
        optimizer: Any,
        batch_size: Optional[int],
        initial_point: Optional[Sequence[float]],
    ) -> None:
        self.optimizer = optimizer
        if batch_size is None:
            batch_size = optimizer.num_workers
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.pending: Optional[list[Any]] = None
        self.rows: list[list[float]] = []
        if initial_point is not None:
            point = _point(initial_point, optimizer.dimension)
            optimizer.suggest(point)

    def ask(self) -> list[list[float]]:
        if self.pending is None:
            size = self.batch_size
            budget = self.optimizer.budget
            if budget is not None:
                # Nevergrad asks on past its budget; the budget ends the run.
                size = min(size, budget - self.optimizer.num_ask)
                if size <= 0:
                    return []
            candidates = [self.optimizer.ask() for _ in range(size)]
            self.rows = [_row(candidate.value) for candidate in candidates]
            self.pending = candidates
        return [list(row) for row in self.rows]

    def tell(self, candidates: list[list[float]], values: list[float]) -> None:
        asked = _check_told(self.pending, values)
        # Nevergrad minimises. One at a time, so a tell that raises keeps the
        # rest pending.
        for value in values:
            self.optimizer.tell(asked[0], -value)
            del asked[0], self.rows[0]
        self.pending = None


class _Pycma:
    def __init__(self, es: Any) -> None:
        self.es = es
        self.pending: Optional[list[Any]] = None

    def ask(self) -> list[list[float]]:
        if self.pending is None:
            if self.es.stop():
                return []
            self.pending = self.es.ask()
        return [_row(x) for x in self.pending]

    def tell(self, candidates: list[list[float]], values: list[float]) -> None:
        solutions = _check_told(self.pending, values)
        # pycma minimises, and is told a whole generation at once: one that
        # raises leaves the generation pending.
        self.es.tell(solutions, [-value for value in values])
        self.pending = None


def from_optuna(
    study: Any,
    dimensions: int,
    batch_size: int,
    bounds: tuple[float, float] = (0.0, 2 * math.pi),
    initial_point: Optional[Sequence[float]] = None,
) -> AskTell:
    """An Optuna study as an ask/tell optimizer for ``polypus.External``.

    Each batch asks the study for ``batch_size`` trials (``study.ask()``) and
    suggests ``dimensions`` floats in each, named ``x0``, ``x1``, … with
    ``trial.suggest_float`` within ``bounds`` (default ``(0, 2π)``), so
    ``study.best_params`` names them that way. Each trial is told its value
    with ``study.tell``, negated when the study minimises
    (``optuna.create_study(direction="minimize")``, Optuna's default): give
    the study ``direction="maximize"`` and the values reach it as Polypus
    measured them. A study with several objectives is a ``ValueError``.

    Optuna never says it is done, so ``max_evaluations`` ends the run: make it
    a multiple of ``batch_size``, or the last ``max_evaluations % batch_size``
    evaluations are not spent. The trials of that untold remainder stay in
    the study as ``RUNNING`` trials, which TPE with ``constant_liar=True``
    keeps penalising, until a later run with the same adapter tells them. The sampler is the study's: ``TPESampler`` for
    Bayesian optimisation, with ``constant_liar=True`` when ``batch_size > 1``
    so the trials of one batch differ, and ``seed`` to fix its candidates.

    ``initial_point`` is enqueued as the first trial (``study.enqueue_trial``):
    ``dimensions`` finite numbers within ``bounds``, otherwise ``ValueError``.
    Each call enqueues it again.
    """
    return _Optuna(study, dimensions, batch_size, bounds, initial_point)


def from_nevergrad(
    optimizer: Any,
    batch_size: Optional[int] = None,
    initial_point: Optional[Sequence[float]] = None,
) -> AskTell:
    """A Nevergrad optimizer as an ask/tell optimizer for ``polypus.External``.

    Build the optimizer over an array of the circuit's parameters, such as
    ``ng.optimizers.NGOpt(parametrization=dimensions, budget=…, num_workers=…)``
    or ``parametrization=ng.p.Array(shape=(dimensions,)).set_bounds(0, 2π)``.
    Each batch is ``batch_size`` candidates (default: the optimizer's
    ``num_workers``), and each is told ``-value``: Nevergrad minimises.

    The optimizer's ``budget``, when set, ends the run once that many
    candidates have been asked, the last batch cut to fit; without one,
    ``max_evaluations`` does. ``initial_point`` is suggested to the optimizer
    (``optimizer.suggest``), so its first candidate is that point: one finite
    number per parameter, otherwise ``ValueError``. Seed the optimizer through
    its parametrization (``parametrization.random_state``).
    """
    return _Nevergrad(optimizer, batch_size, initial_point)


def from_pycma(es: Any) -> AskTell:
    """A pycma ``CMAEvolutionStrategy`` as an ask/tell optimizer for
    ``polypus.External``.

    Each batch is one generation (``es.ask()``, ``popsize`` candidates), told
    back with ``es.tell`` as ``-value``: pycma minimises. The run ends when
    ``es.stop()`` reports a termination criterion (``maxfevals``,
    ``maxiter``, ``tolfun``, …), or at ``max_evaluations``. The start is the
    ``x0`` of ``cma.CMAEvolutionStrategy(x0, sigma0, options)``, and the seed
    its ``seed`` option.
    """
    return _Pycma(es)
