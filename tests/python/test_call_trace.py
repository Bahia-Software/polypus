"""
The ``trace`` argument of ``polypus.train`` and ``polypus.run_quantum_circuit``
(contract C-12).

``trace`` is keyword-only and defaults to ``None``. A path writes the run's
call trace to a new JSON Lines file: the ``run`` header, one line per planner
batch, backend wave, optimizer iteration and call to QNG's metric, and the
``end`` line with the run's status. In either function, a value that is not
a path, or a ``bytes`` path, raises ``TypeError`` before anything runs: with
``infrastructure="cunqa"``, before ``connect_to_infrastructure`` is reached,
which a recorder checks without SLURM or CUNQA.

The trace never changes a seeded result, with or without a ``callback``. Its
``end`` line follows the run's outcome, an error before the first batch
included: the recorder also stops a run of either function as creating its
backend raises CUNQA's QPUs, before the run in ``train`` and inside it in
``run_quantum_circuit``. A path where something already exists, or in a
missing directory, is left as it is and the run goes on without a trace. The
runs are small seeded runs on the native backend (``call_trace.train``), and
a circuit run whose one batch is traced with its waves, on one QPU and on
several.
"""

import inspect
import sys
import time

import polypus
import pytest
from call_trace import (
    FINAL_SHOTS,
    OPTIMIZERS,
    Trace,
    check_trace,
    method,
    read_trace,
    result_bits,
    train,
)


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


# The two functions that take `trace`.
TRACED_FUNCTIONS = pytest.mark.parametrize(
    "function",
    [polypus.train, polypus.run_quantum_circuit],
    ids=["train", "run_quantum_circuit"],
)


def _on_cunqa(function, trace):
    """A small run of ``function`` on CUNQA with ``trace``: a training run of
    ``call_trace.train`` or a one-qubit circuit run.

    Creating the backend raises the QPUs: ``train`` creates it before the
    run, ``run_quantum_circuit`` as part of it, with the GIL released."""
    if function is polypus.train:
        return train(
            method("rand1bin"), infrastructure="cunqa", backend="aer", trace=trace
        )
    qc = polypus.Circuit(1).h(0).measure_all()
    return polypus.run_quantum_circuit(qc, 8, "cunqa", backend="aer", trace=trace)


@TRACED_FUNCTIONS
def test_trace_is_a_keyword_only_argument_that_defaults_to_none(function):
    parameter = inspect.signature(function).parameters["trace"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@TRACED_FUNCTIONS
@pytest.mark.parametrize("value", [1, 2.5, ["t.jsonl"]])
def test_a_trace_that_is_not_a_path_raises_before_anything_runs(
    seam_calls, tmp_path, monkeypatch, function, value
):
    monkeypatch.chdir(tmp_path)
    # The message is the conversion's own (os.fspath's) on every Python
    # version; PyO3 names the argument in a note, and notes need Python >= 3.11
    # (docs/ENGINEERING.md §9).
    with pytest.raises(TypeError, match=r"os\.PathLike") as excinfo:
        _on_cunqa(function, trace=value)
    if sys.version_info >= (3, 11):
        assert excinfo.value.__notes__ == ["while processing 'trace'"]
    assert seam_calls == []
    assert list(tmp_path.iterdir()) == []


class _BytesPath:
    """An ``os.PathLike`` whose path is ``bytes``."""

    def __fspath__(self):
        return b"t.jsonl"


@TRACED_FUNCTIONS
@pytest.mark.parametrize(
    "value", [b"t.jsonl", _BytesPath()], ids=["bytes", "pathlike-bytes"]
)
def test_a_bytes_trace_path_raises_before_anything_runs(
    seam_calls, tmp_path, monkeypatch, function, value
):
    # Only a str path is taken: a bytes path, given as is or by an
    # os.PathLike, is not.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError, match=r"'bytes' object") as excinfo:
        _on_cunqa(function, trace=value)
    if sys.version_info >= (3, 11):
        assert excinfo.value.__notes__ == ["while processing 'trace'"]
    assert seam_calls == []
    assert list(tmp_path.iterdir()) == []


def test_trace_header_fields(tmp_path):
    path = tmp_path / "run.jsonl"
    before = time.time_ns()
    result = train(method("spsa"), trace=path)
    after = time.time_ns()
    trace = read_trace(path)
    check_trace(trace)
    head = trace.header
    assert head == {
        "v": 1,
        "ev": "run",
        "flow": "train",
        "id": result.id,
        "polypus": polypus.__version__,
        "method": "spsa",
        "dims": 2,
        "backend": "polypus",
        "infrastructure": "local",
        "n_qpus": 1,
        "shots": 64,
        "seed": 11,
        "t0_unix_ns": head["t0_unix_ns"],
    }
    assert before <= head["t0_unix_ns"] <= after
    assert trace.end["status"] == "ok"


@pytest.mark.parametrize("n_qpus", [1, 4])
def test_run_quantum_circuit_trace(tmp_path, n_qpus):
    qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
    run = dict(
        shots=1000, infrastructure="local", n_qpus=n_qpus, backend="polypus", seed=5
    )
    plain = polypus.run_quantum_circuit(qc, **run)
    traced = polypus.run_quantum_circuit(qc, **run, trace=tmp_path / "rqc.jsonl")
    assert traced.counts == plain.counts
    trace = read_trace(tmp_path / "rqc.jsonl")
    check_trace(trace)
    head = trace.header
    assert (
        head["flow"],
        head["method"],
        head["dims"],
        head["n_qpus"],
        head["shots"],
        head["seed"],
    ) == ("run_circuit", None, None, n_qpus, 1000, 5)
    assert head["id"].startswith(f"run_{n_qpus}_local_")
    assert (head["id"], head["backend"], head["infrastructure"]) == (
        traced.id,
        "polypus",
        "local",
    )
    assert trace.end["status"] == "ok"
    [batch] = trace.events("batch")
    assert (batch["k"], batch["op"], batch["n"], batch["shots"], batch["ok"]) == (
        0,
        "execute_replicas",
        1,
        1000,
        True,
    )
    waves = trace.events("wave")
    if n_qpus == 1:
        assert [(w["op"], w["n"], w["shots"]) for w in waves] == [
            ("run_circuits", 1, 1000)
        ]
    else:
        assert {w["op"] for w in waves} == {"run_shots_distributed"}
        replicas = [shots for w in waves for shots in w["replicas"]]
        assert (len(replicas), sum(replicas)) == (4, 1000)


@pytest.mark.integration
@pytest.mark.parametrize("name", OPTIMIZERS)
def test_trace_does_not_change_results(tmp_path, name):
    plain = train(method(name), final_shots=FINAL_SHOTS)
    traced = train(
        method(name), final_shots=FINAL_SHOTS, trace=tmp_path / "traced.jsonl"
    )
    watched = train(
        method(name), final_shots=FINAL_SHOTS, callback=lambda progress: None
    )
    both = train(
        method(name),
        final_shots=FINAL_SHOTS,
        callback=lambda progress: None,
        trace=tmp_path / "both.jsonl",
    )
    assert result_bits(traced) == result_bits(plain)
    assert result_bits(both) == result_bits(watched) == result_bits(plain)
    for path in ("traced.jsonl", "both.jsonl"):
        trace = read_trace(tmp_path / path)
        check_trace(trace)
        assert trace.header["method"] == name
        assert len(trace.events("iter")) == plain.iterations_run
        assert trace.events("batch")[-1]["shots"] == FINAL_SHOTS


def test_qng_metric_events_match_variance_calls(tmp_path):
    calls, progress = [], []

    def variance(theta, a):
        calls.append(a)
        return 1.0

    def watch(p):
        progress.append((p.iteration, p.evaluations, p.best_fitness))

    train(
        polypus.QNG(variance, max_iters=3), callback=watch, trace=tmp_path / "qng.jsonl"
    )
    trace = read_trace(tmp_path / "qng.jsonl")
    check_trace(trace)
    assert [m["n"] for m in trace.events("metric")] == [2, 2, 2]
    assert calls == [0, 1] * 3
    assert [(i["i"], i["evals"], i["best"]) for i in trace.events("iter")] == progress


@pytest.mark.parametrize(
    ("case", "status"),
    [("ok", "ok"), ("error", "error"), ("interrupt", "cancelled")],
)
def test_trace_status_follows_the_outcome(tmp_path, case, status):
    path = tmp_path / f"{case}.jsonl"
    if case == "ok":
        train(method("rand1bin"), trace=path)
    elif case == "error":

        def broken(bits):
            raise ValueError("broken cost")

        with pytest.raises(ValueError, match="broken cost"):
            train(method("rand1bin"), expectation_function=broken, trace=path)
    else:

        def interrupt(progress):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            train(method("rand1bin"), callback=interrupt, trace=path)
    trace = read_trace(path)
    check_trace(trace)
    assert trace.end["status"] == status


@TRACED_FUNCTIONS
def test_an_error_before_the_first_batch_closes_the_trace_with_error(
    seam_calls, tmp_path, function
):
    # The run stops where creating the backend raises CUNQA's QPUs: before the
    # run in `train`, and inside it in `run_quantum_circuit`, whose trace
    # learns of the failure from the run's own error.
    path = tmp_path / "early.jsonl"
    with pytest.raises(_Stop):
        _on_cunqa(function, trace=path)
    assert seam_calls == ["cunqa"]
    trace = read_trace(path)
    check_trace(trace)
    flow = "train" if function is polypus.train else "run_circuit"
    assert [line["ev"] for line in trace.lines] == ["run", "end"]
    assert (trace.header["flow"], trace.end["status"], trace.end["batches"]) == (
        flow,
        "error",
        0,
    )


def test_same_trace_path_twice_keeps_the_first_file(tmp_path):
    path = tmp_path / "once.jsonl"
    first = train(method("spsa"), trace=path)
    kept = path.read_bytes()
    second = train(method("spsa"), trace=path)
    assert path.read_bytes() == kept
    assert result_bits(second) == result_bits(first)


def test_a_trace_in_a_missing_directory_runs_without_a_trace(tmp_path):
    missing = tmp_path / "missing"
    traced = train(method("spsa"), trace=missing / "t.jsonl")
    assert result_bits(traced) == result_bits(train(method("spsa")))
    assert not missing.exists()


def test_reader_stops_at_last_complete_line(tmp_path):
    path = tmp_path / "full.jsonl"
    train(method("spsa"), trace=path)
    data = path.read_bytes()
    lines = data.split(b"\n")[:-1]
    cut = tmp_path / "cut.jsonl"
    cut.write_bytes(data[: len(data) - 1 - len(lines[-1]) // 2])
    trace = read_trace(cut)
    assert not trace.complete
    assert len(trace.lines) == len(lines) - 1
    assert trace.lines[0]["ev"] == "run"
    cut.write_bytes(data[: len(lines[0]) + 1] + b'{"ev": broken}\n' + lines[1] + b"\n")
    assert [line["ev"] for line in read_trace(cut).lines] == ["run"]
    for prefix in (b"", data[:10]):
        cut.write_bytes(prefix)
        assert read_trace(cut) == Trace(lines=[], complete=False)
