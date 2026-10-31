"""
The ``trace`` argument of ``polypus.train`` (contract C-12).

``trace`` is keyword-only and defaults to ``None``. A path writes the run's
call trace to a new JSON Lines file: the ``run`` header, one line per planner
batch, backend wave, optimizer iteration and call to QNG's metric, and the
``end`` line with the run's status. A value that is not a path, or a
``bytes`` path, raises ``TypeError`` before anything runs: with
``infrastructure="cunqa"``, before ``connect_to_infrastructure`` is reached,
which a recorder checks without SLURM or CUNQA.

The trace never changes a seeded result, with or without a ``callback``. Its
``end`` line follows the run's outcome, an error before the first batch
included. A path where something already exists, or in a missing directory,
is left as it is and the run goes on without a trace. The runs are small
seeded runs on the native backend (``call_trace.train``).
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


def test_trace_is_a_keyword_only_argument_that_defaults_to_none():
    parameter = inspect.signature(polypus.train).parameters["trace"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("value", [1, 2.5, ["t.jsonl"]])
def test_a_trace_that_is_not_a_path_raises_before_anything_runs(
    seam_calls, tmp_path, monkeypatch, value
):
    monkeypatch.chdir(tmp_path)
    # The message is the conversion's own (os.fspath's) on every Python
    # version; PyO3 names the argument in a note, and notes need Python >= 3.11
    # (docs/ENGINEERING.md §9).
    with pytest.raises(TypeError, match=r"os\.PathLike") as excinfo:
        train(method("rand1bin"), infrastructure="cunqa", backend="aer", trace=value)
    if sys.version_info >= (3, 11):
        assert excinfo.value.__notes__ == ["while processing 'trace'"]
    assert seam_calls == []
    assert list(tmp_path.iterdir()) == []


class _BytesPath:
    """An ``os.PathLike`` whose path is ``bytes``."""

    def __fspath__(self):
        return b"t.jsonl"


@pytest.mark.parametrize(
    "value", [b"t.jsonl", _BytesPath()], ids=["bytes", "pathlike-bytes"]
)
def test_a_bytes_trace_path_raises_before_anything_runs(
    seam_calls, tmp_path, monkeypatch, value
):
    # Only a str path is taken: a bytes path, given as is or by an
    # os.PathLike, is not.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError, match=r"'bytes' object") as excinfo:
        train(method("rand1bin"), infrastructure="cunqa", backend="aer", trace=value)
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


def test_an_error_before_the_first_batch_closes_the_trace_with_error(
    seam_calls, tmp_path
):
    path = tmp_path / "early.jsonl"
    with pytest.raises(_Stop):
        train(method("rand1bin"), infrastructure="cunqa", backend="aer", trace=path)
    trace = read_trace(path)
    check_trace(trace)
    assert [line["ev"] for line in trace.lines] == ["run", "end"]
    assert (trace.end["status"], trace.end["batches"]) == ("error", 0)


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
