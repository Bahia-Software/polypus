"""
The ``callback`` of ``polypus.train``, ``polypus.qml.train`` and
``polypus.qml.Model.fit`` (issue #500, contract C-5).

``callback`` is keyword-only. One that is neither callable nor ``None`` raises
``TypeError`` before anything runs: with ``infrastructure="cunqa"``, before
``connect_to_infrastructure`` (and so ``qraise``) is reached, which a recorder
checks without SLURM or CUNQA.

A valid callback is called once per iteration of the optimizer with a
read-only ``polypus.TrainProgress``. ``train`` and ``qml.train`` run here on
Aer with ``polypus_python.run_qcs`` replaced by a fake that scores each circuit
by its parameters and records every circuit with its shots, so a test can
check how many circuits were sent when the callback was called and that none
is sent after a stop; ``Model.fit`` runs on the native simulator. Returning
``True`` stops the run, which returns what it has reached; ``None`` and
``False`` go on; anything else is a ``TypeError``; an exception propagates as
itself. A callback that never stops leaves a seeded result unchanged, in this
interpreter and in a fresh one.
"""

import inspect
import json
import math
import re
import subprocess
import sys

import numpy as np
import pytest

DIMENSIONS = 2
SHOTS = 64
FINAL_SHOTS = 4096
OPTIMIZERS = [
    "rand1bin",
    "shade",
    "lshade",
    "pso",
    "qng",
    "cmaes",
    "adam",
    "spsa",
    "rotosolve",
    "external",
]

_RY = re.compile(r"^ry\(([^)]*)\) q\[(\d+)\];$", re.MULTILINE)


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


def _method(name, dimensions=DIMENSIONS):
    """A small configuration of the optimizer ``name``, with no early stop
    within its budget and at least three iterations. ``"external"`` is a fresh
    seeded random search."""
    import polypus
    from ask_tell_helpers import RandomSearch

    if name == "pso":
        return polypus.PSO(generations=4, population_size=6, tolerance=1e-12)
    if name == "qng":
        return polypus.QNG(lambda theta, a: 1.0, max_iters=4)
    if name == "cmaes":
        return polypus.CMAES(generations=4, tolerance=1e-12, patience=100)
    if name == "adam":
        return polypus.Adam(max_iters=4, patience=100)
    if name == "spsa":
        return polypus.SPSA(max_iters=4)
    if name == "rotosolve":
        return polypus.Rotosolve(sweeps=2, tolerance=0.0)
    if name == "external":
        return polypus.External(RandomSearch(dimensions, sizes=(4,), seed=3), 16)
    return polypus.DE(
        generations=4, population_size=6, strategy=name, tolerance=1e-12, patience=100
    )


def _circuit():
    import polypus

    qc = polypus.Circuit(DIMENSIONS)
    for q in range(DIMENSIONS):
        qc.ry(q, polypus.Param(q))
    qc.measure_all()
    return qc


def _train(method, *, infrastructure="local", backend="aer", **kwargs):
    import polypus

    return polypus.train(
        _circuit(),
        method,
        shots=SHOTS,
        n_qpus=1,
        dimensions=DIMENSIONS,
        expectation_function=_all_ones,
        infrastructure=infrastructure,
        nodes=1,
        cores_per_qpu=1,
        id="callback",
        backend=backend,
        seed=11,
        **kwargs,
    )


def _qml_train(method, *, infrastructure="local", **kwargs):
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
        # One sample, so each candidate is one circuit.
        [[0.3]],
        method,
        shots=SHOTS,
        n_qpus=1,
        dimensions=DIMENSIONS,
        expectation_function=_all_ones,
        infrastructure=infrastructure,
        nodes=1,
        cores_per_qpu=1,
        id="callback_qml",
        seed=11,
        **kwargs,
    )


def _model():
    """Two qubits, ``ry`` encodings and ``⟨Z₀⟩``: every optimizer, Rotosolve
    included, can train it without labels."""
    from polypus.qml import Ansatz, Encoder, Model, Readout, Z

    return Model(
        Encoder(2).angle_encoder("ry"), Ansatz(2).hardware_efficient(1), Readout([Z(0)])
    )


def _fit(method, *, infrastructure="local", **kwargs):
    kwargs.setdefault("seed", 11)
    return _model().fit(
        [[0.1, 0.2], [0.4, 0.3]], method=method, infrastructure=infrastructure, **kwargs
    )


def _vqc_candidate(qasm):
    angles = {int(q): float(a) for a, q in _RY.findall(qasm)}
    return [angles[q] for q in range(DIMENSIONS)]


def _qml_candidate(circuit):
    angles = {
        inst.operation.name: float(inst.operation.params[0])
        for inst in circuit.data
        if inst.operation.name in ("ry", "rz")
    }
    return [angles["ry"], angles["rz"]]


# The runner, how to read a recorded circuit's candidate, and its width.
RECORDED = {
    "train": (_train, _vqc_candidate, DIMENSIONS),
    "qml.train": (_qml_train, _qml_candidate, 1),
}


def _probability(candidate):
    a, b = candidate
    return (1.0 + math.sin(3.0 * a + 1.3) * math.cos(2.0 * b - 0.4)) / 2.0


def _fitness(candidate):
    """The fitness the fake gives ``candidate`` with ``SHOTS`` shots."""
    return round(SHOTS * _probability(candidate)) / SHOTS


@pytest.fixture
def sent(monkeypatch):
    """Replace the local seam with a fake that measures all ones on
    ``round(shots · p)`` shots, ``p`` a fixed function of the circuit's
    candidate, and record ``(entry, candidate, shots)`` for every circuit in
    the order sent. The test sets ``sent.entry`` before running."""
    import polypus_python

    class Sent(list):
        entry = "train"

    record = Sent()

    def fake_run_qcs(infrastructure, **kwargs):
        _, candidate, width = RECORDED[record.entry]
        shots, results = kwargs["shots"], []
        for qc in kwargs["qcs"]:
            params = candidate(qc)
            ones = round(shots * _probability(params))
            record.append((params, shots))
            results.append({"1" * width: ones, "0" * width: shots - ones})
        return [{k: v for k, v in r.items() if v} for r in results]

    monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)
    return record


class _Watcher:
    """A callback that keeps every record, with the circuits sent by then,
    and returns ``True`` on iteration ``stop_at``."""

    def __init__(self, sent=None, stop_at=None, stop_value=True):
        self.sent = sent
        self.stop_at = stop_at
        self.stop_value = stop_value
        self.seen = []
        self.circuits = []

    def __call__(self, progress):
        self.seen.append(progress)
        if self.sent is not None:
            self.circuits.append(len(self.sent))
        if progress.iteration == self.stop_at:
            return self.stop_value
        return None


def _assert_same_result(a, b):
    assert a.best_params == b.best_params
    assert a.best_fitness == b.best_fitness
    assert a.fitness_history == b.fitness_history
    assert a.iterations_run == b.iterations_run
    assert a.converged == b.converged
    assert a.final_params == b.final_params
    assert a.final_fitness == b.final_fitness
    assert a.seed == b.seed


# ─────────────────────────────────────────────────────────────────────────────
# Validation, before anything runs
# ─────────────────────────────────────────────────────────────────────────────


class _Stop(Exception):
    """Raised by the recorder to end the run at the CUNQA seam."""


@pytest.fixture
def seam_calls(monkeypatch):
    """Record every ``connect_to_infrastructure`` call (the one that raises
    CUNQA QPUs) and stop the run there."""
    import polypus_python

    calls = []

    def record(infrastructure, **kwargs):
        calls.append(infrastructure)
        raise _Stop

    monkeypatch.setattr(polypus_python, "connect_to_infrastructure", record)
    return calls


ENTRY_POINTS = {"train": _train, "qml.train": _qml_train, "Model.fit": _fit}


@pytest.mark.parametrize("entry", ENTRY_POINTS)
class TestValidation:
    @pytest.mark.parametrize(
        ("value", "shown"),
        [
            (1, "1 (of type int)"),
            (True, "True (of type bool)"),
            ("checkpoint.json", "'checkpoint.json' (of type str)"),
            ([print], "[<built-in function print>] (of type list)"),
        ],
    )
    def test_a_callback_that_is_not_callable_raises_before_anything_runs(
        self, seam_calls, entry, value, shown
    ):
        import polypus

        with pytest.raises(
            TypeError,
            match=re.escape(f"callback must be a callable or None, got {shown}"),
        ):
            ENTRY_POINTS[entry](polypus.DE(), infrastructure="cunqa", callback=value)
        assert seam_calls == []

    @pytest.mark.parametrize("value", [None, print, lambda progress: None])
    def test_a_valid_callback_gets_to_the_seam(self, seam_calls, entry, value):
        # The control for the test above.
        import polypus

        with pytest.raises(_Stop):
            ENTRY_POINTS[entry](polypus.DE(), infrastructure="cunqa", callback=value)
        assert seam_calls == ["cunqa"]

    def test_callback_is_keyword_only(self, entry):
        import polypus

        function = {
            "train": polypus.train,
            "qml.train": polypus.qml.train,
            "Model.fit": polypus.qml.Model.fit,
        }[entry]
        parameter = inspect.signature(function).parameters["callback"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_final_shots_is_checked_before_the_callback(seam_calls, entry):
    import polypus

    with pytest.raises(ValueError, match="final_shots must be an integer >= 1"):
        ENTRY_POINTS[entry](
            polypus.DE(), infrastructure="cunqa", final_shots=0, callback=1
        )
    assert seam_calls == []


# ─────────────────────────────────────────────────────────────────────────────
# One call per iteration
# ─────────────────────────────────────────────────────────────────────────────


def _assert_watched(result, watcher):
    """The calls ``watcher`` recorded are one per iteration, each with the
    entry its iteration left in the history."""
    seen = watcher.seen
    assert [p.iteration for p in seen] == list(range(1, result.iterations_run + 1))
    assert [p.best_fitness for p in seen] == result.fitness_history
    assert seen[-1].best_params == result.best_params
    assert all(isinstance(p.best_params, list) for p in seen)
    assert all(
        later.best_fitness >= earlier.best_fitness
        for earlier, later in zip(seen, seen[1:])
    )
    evaluations = [p.evaluations for p in seen]
    assert evaluations[0] > 0
    assert all(b > a for a, b in zip(evaluations, evaluations[1:]))


@pytest.mark.integration
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("entry", ["train", "qml.train"])
def test_called_once_per_iteration_with_the_circuits_scored(sent, entry, optimizer):
    import polypus

    sent.entry = entry
    run = RECORDED[entry][0]
    watcher = _Watcher(sent)
    result = run(_method(optimizer), callback=watcher, final_shots=FINAL_SHOTS)
    assert result.iterations_run >= 3
    _assert_watched(result, watcher)
    for progress in watcher.seen:
        assert isinstance(progress, polypus.TrainProgress)
        # One circuit per candidate: QNG's variance_function sends none.
        assert progress.evaluations == watcher.circuits[progress.iteration - 1]
    # The final batch comes after the last call and is not counted.
    assert all(shots == SHOTS for _, shots in sent[: watcher.circuits[-1]])
    assert all(shots == FINAL_SHOTS for _, shots in sent[watcher.circuits[-1] :])
    assert result.final_params is not None


@pytest.mark.integration
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_model_fit_calls_it_once_per_iteration(optimizer):
    dimensions = _model().num_params
    watcher = _Watcher()
    result = _fit(_method(optimizer, dimensions), callback=watcher)
    assert result.iterations_run >= 3
    _assert_watched(result, watcher)


@pytest.mark.integration
@pytest.mark.parametrize("optimizer", ["rand1bin", "adam"])
def test_model_fit_with_minibatches_is_not_called_for_the_rescore(optimizer):
    # A gradient-free optimizer's best_fitness is re-scored on every sample
    # once it has stopped; the callback sees the minibatch estimates of the
    # history and is not called for the re-score. Adam tracks its best on
    # every sample, so its history ends at best_fitness.
    dimensions = _model().num_params
    watcher = _Watcher()
    result = _model().fit(
        [[0.1, 0.2], [0.4, 0.3], [2.9, 2.8], [0.2, 0.1]],
        method=_method(optimizer, dimensions),
        batch_size=2,
        seed=11,
        callback=watcher,
    )
    assert len(watcher.seen) == result.iterations_run
    assert [p.best_fitness for p in watcher.seen] == result.fitness_history
    if optimizer == "adam":
        assert result.fitness_history[-1] == result.best_fitness


@pytest.mark.integration
def test_de_evaluations_count_the_initial_population_and_each_generation():
    import polypus

    watcher = _Watcher()
    method = polypus.DE(generations=5, population_size=6, tolerance=1e-12, patience=100)
    result = _fit(method, callback=watcher)
    assert [p.evaluations for p in watcher.seen] == [
        6 * (g + 1) for g in range(1, result.iterations_run + 1)
    ]


@pytest.mark.integration
def test_adam_evaluations_count_theta_but_not_its_parameter_shift():
    # Model.fit gives Adam the exact gradient, which measures θ in the batch
    # of its shifted circuits: θ counts, the shifted circuits do not, and the
    # last iteration also measures the θ its step reached.
    import polypus

    watcher = _Watcher()
    _fit(polypus.Adam(max_iters=4, patience=100), callback=watcher)
    assert [p.evaluations for p in watcher.seen] == [1, 2, 3, 5]


@pytest.mark.integration
def test_the_record_is_read_only_and_its_params_a_copy():
    import polypus

    def vandal(progress):
        progress.best_params[0] = 1e6
        progress.best_params.clear()
        with pytest.raises(AttributeError):
            progress.best_fitness = 1e6
        with pytest.raises(AttributeError):
            progress.iteration = 0

    method = polypus.DE(generations=4, population_size=6, tolerance=1e-12, patience=100)
    plain = _train(method, backend="polypus")
    vandalised = _train(method, backend="polypus", callback=vandal)
    _assert_same_result(plain, vandalised)

    watcher = _Watcher()
    _train(method, backend="polypus", callback=watcher)
    assert re.fullmatch(
        r"TrainProgress\(iteration=1, best_fitness=\S+, evaluations=12, "
        r"best_params=\[\S+, \S+\]\)",
        repr(watcher.seen[0]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stopping
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.parametrize("stop_at", [1, 2])
@pytest.mark.parametrize("stop_value", [True, np.bool_(True)])
@pytest.mark.parametrize("optimizer", ["rand1bin", "adam"])
@pytest.mark.parametrize("entry", ["train", "qml.train"])
def test_true_stops_the_run_with_what_it_has_reached(
    sent, entry, optimizer, stop_value, stop_at
):
    sent.entry = entry
    watcher = _Watcher(sent, stop_at=stop_at, stop_value=stop_value)
    result = RECORDED[entry][0](_method(optimizer), callback=watcher)
    assert result.iterations_run == stop_at
    assert len(result.fitness_history) == stop_at
    assert not result.converged
    assert result.fitness_history[-1] == result.best_fitness
    # best_fitness is the value measured for best_params (C-5).
    assert result.best_fitness == _fitness(result.best_params)
    assert watcher.seen[-1].best_params == result.best_params
    assert len(watcher.seen) == stop_at
    # Nothing is sent after the stop.
    assert watcher.circuits[-1] == len(sent)


@pytest.mark.integration
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_model_fit_stops_on_true(optimizer):
    dimensions = _model().num_params
    watcher = _Watcher(stop_at=2)
    result = _fit(_method(optimizer, dimensions), callback=watcher)
    assert result.iterations_run == 2
    assert len(result.fitness_history) == 2
    assert not result.converged
    assert result.fitness_history[-1] == result.best_fitness
    assert watcher.seen[-1].best_params == result.best_params


@pytest.mark.integration
@pytest.mark.parametrize("entry", ["train", "qml.train"])
def test_a_stop_still_sends_the_final_batch(sent, entry):
    sent.entry = entry
    watcher = _Watcher(sent, stop_at=2)
    result = RECORDED[entry][0](
        _method("rand1bin"), callback=watcher, final_shots=FINAL_SHOTS
    )
    assert result.iterations_run == 2
    final = sent[watcher.circuits[-1] :]
    assert final, "the final batch was sent"
    assert all(shots == FINAL_SHOTS for _, shots in final)
    assert final[0][0] == result.best_params
    assert result.final_params is not None
    assert result.final_fitness is not None


@pytest.mark.integration
@pytest.mark.parametrize("optimizer", ["rand1bin", "adam"])
def test_model_fit_still_sends_the_final_batch_after_a_stop(optimizer):
    # Model.fit takes final_shots too: a stop asked for by the callback still
    # re-evaluates the best candidates, and the optimizer's fields are the
    # ones the callback saw last.
    dimensions = _model().num_params
    watcher = _Watcher(stop_at=2)
    result = _fit(
        _method(optimizer, dimensions), callback=watcher, final_shots=FINAL_SHOTS
    )
    assert result.iterations_run == 2
    assert not result.converged
    assert watcher.seen[-1].best_params == result.best_params
    assert watcher.seen[-1].best_fitness == result.best_fitness
    assert result.final_params is not None
    assert result.final_fitness is not None
    plain = _fit(_method(optimizer, dimensions), final_shots=FINAL_SHOTS)
    assert plain.iterations_run > 2


@pytest.mark.integration
@pytest.mark.parametrize("entry", ["train", "qml.train"])
def test_false_and_none_go_on(sent, entry):
    sent.entry = entry
    plain = RECORDED[entry][0](_method("rand1bin"))
    for value in (False, None, np.bool_(False)):
        result = RECORDED[entry][0](_method("rand1bin"), callback=lambda p: value)
        _assert_same_result(plain, result)


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────


class _CheckpointError(Exception):
    pass


@pytest.mark.integration
@pytest.mark.parametrize("raised", [_CheckpointError, KeyboardInterrupt])
@pytest.mark.parametrize("entry", ["train", "qml.train"])
def test_an_exception_propagates_as_itself_without_a_final_batch(sent, entry, raised):
    sent.entry = entry
    calls = []

    def failing(progress):
        calls.append(len(sent))
        if progress.iteration == 2:
            raise raised("disk full")

    with pytest.raises(raised, match="disk full") as info:
        RECORDED[entry][0](
            _method("rand1bin"), callback=failing, final_shots=FINAL_SHOTS
        )
    assert type(info.value) is raised
    assert len(calls) == 2, "no call after the exception"
    assert calls[-1] == len(sent), "no circuit after the exception"
    assert all(shots == SHOTS for _, shots in sent), "no final batch"


@pytest.mark.integration
def test_model_fit_propagates_an_exception_as_itself_without_a_final_batch(
    monkeypatch,
):
    # On the local Aer seam, with a fake that records the shots of every batch
    # and measures "00" on every shot: nothing is sent after the exception,
    # and no batch has final_shots.
    import polypus
    import polypus_python

    shots = []

    def fake_run_qcs(infrastructure, **kwargs):
        shots.append(kwargs["shots"])
        return [{"00": kwargs["shots"]} for _ in kwargs["qcs"]]

    monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)
    calls = []

    def failing(progress):
        calls.append(len(shots))
        if progress.iteration == 2:
            raise _CheckpointError("disk full")

    with pytest.raises(_CheckpointError, match="disk full") as info:
        _fit(
            polypus.Adam(max_iters=4, patience=100),
            backend="aer",
            shots=SHOTS,
            final_shots=FINAL_SHOTS,
            callback=failing,
        )
    assert type(info.value) is _CheckpointError
    assert len(calls) == 2, "no call after the exception"
    assert calls[-1] == len(shots), "no batch after the exception"
    assert shots, "the run sent its batches through the fake"
    assert all(s == SHOTS for s in shots), "no final batch"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("returned", "shown"),
    [
        (1, "1 (of type int)"),
        (0, "0 (of type int)"),
        (1.0, "1.0 (of type float)"),
        ("stop", "'stop' (of type str)"),
        ([True], "[True] (of type list)"),
        (np.float64(1.0), "np.float64(1.0) (of type float64)"),
        (np.int64(1), "np.int64(1) (of type int64)"),
    ],
)
@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_any_other_return_value_is_a_type_error(sent, entry, returned, shown):
    import polypus

    sent.entry = "train" if entry == "Model.fit" else entry
    calls = []

    def callback(progress):
        calls.append(progress.iteration)
        return returned

    if isinstance(returned, np.generic) and not repr(returned).startswith("np."):
        # NumPy 1 prints scalars without the `np.` prefix.
        shown = f"{returned!r} (of type {type(returned).__name__})"
    method = polypus.DE(generations=4, population_size=6, tolerance=1e-12, patience=100)
    with pytest.raises(TypeError, match=re.escape(f"it returned {shown}")):
        ENTRY_POINTS[entry](method, callback=callback)
    assert calls == [1], "the first return value is the error"


# ─────────────────────────────────────────────────────────────────────────────
# A callback that never stops changes nothing
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_a_callback_that_never_stops_leaves_a_seeded_result_unchanged(optimizer):
    plain = _train(_method(optimizer), backend="polypus")
    watched = _train(_method(optimizer), backend="polypus", callback=lambda p: None)
    _assert_same_result(plain, watched)

    dimensions = _model().num_params
    plain = _fit(_method(optimizer, dimensions))
    watched = _fit(_method(optimizer, dimensions), callback=lambda p: False)
    _assert_same_result(plain, watched)


# A seeded native run, with and without a callback that never stops, printed
# as exact float hex: a fresh interpreter replays the run without a callback.
_REPLAY = """
import json, sys
import polypus

P = polypus.Param
circuit = polypus.Circuit(2).ry(0, P(0)).ry(1, P(1)).cx(0, 1).measure_all()


def run(**kwargs):
    result = polypus.train(
        circuit,
        polypus.DE(generations=6, population_size=8, tolerance=1e-12, patience=50),
        shots=256, n_qpus=1, dimensions=2,
        expectation_function=lambda b: float(b.count("1")),
        infrastructure="local", nodes=1, cores_per_qpu=1, id="replay",
        backend="polypus", seed=5, **kwargs,
    )
    return {
        "best_params": [x.hex() for x in result.best_params],
        "best_fitness": result.best_fitness.hex(),
        "fitness_history": [x.hex() for x in result.fitness_history],
        "iterations_run": result.iterations_run,
        "converged": result.converged,
    }


kwargs = {"callback": lambda progress: None} if sys.argv[1] == "watched" else {}
print(json.dumps(run(**kwargs)))
"""


@pytest.mark.integration
def test_a_watched_run_replays_an_unwatched_one_in_a_fresh_interpreter():
    def replay(mode):
        proc = subprocess.run(
            [sys.executable, "-c", _REPLAY, mode],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)

    assert replay("watched") == replay("plain")
