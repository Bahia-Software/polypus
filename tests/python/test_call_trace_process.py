"""
The call trace of ``polypus.train`` and ``polypus.run_quantum_circuit``
(contract C-12) seen from outside the process that writes it.

Each case runs in a fresh interpreter with the logger on the console at the
``warn`` level: a path where something already exists, a file or a symbolic
link, is left as it is with one warning naming it, and the run goes on
without a trace; a new trace file gets mode ``0640``; and a run killed with
``SIGKILL`` leaves a trace that reads up to its last whole line, without an
``end`` line.

Without ``trace=``, ``POLYPUS_CALL_TRACE`` names a directory that gets one
trace per run, ``<id>-<pid>-<n>.jsonl`` after the run's effective id, for
``train`` and ``run_quantum_circuit`` alike, with the id cut to a safe file
name, so the traces of processes that share the directory and a pid stay
apart; an empty variable is off, and a directory that does not exist is never
created.
``POLYPUS_CALL_TRACE_MAX_MB`` caps every trace, and an invalid value is
warned about once and ignored.
"""

import itertools
import json
import os
import re
import select
import signal
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from call_trace import check_trace, read_trace

PRELUDE = f"""\
import json, os, sys
import polypus
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from call_trace import method, train
polypus.init_logger(level="warn", console=True, timestamp=False)
"""


def _env(tmp_path, **env):
    """The environment of a child: this one without any ``POLYPUS_CALL_TRACE*``
    variable, with auto-calibration off and a cache of its own, and each of
    ``env`` set, or removed if it is ``None``."""
    child = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("POLYPUS_CALL_TRACE")
    }
    child["POLYPUS_NO_AUTOCALIBRATE"] = "1"
    child["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    for key, value in env.items():
        if value is None:
            child.pop(key, None)
        else:
            child[key] = str(value)
    return child


def _child(code, tmp_path, *, cwd=None, **env):
    """Run ``PRELUDE`` and then ``code`` in a fresh interpreter."""
    return subprocess.run(
        [sys.executable, "-c", PRELUDE + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=cwd,
        env=_env(tmp_path, **env),
    )


def _ok(result):
    """``result``, once it is known to have exited with status 0."""
    assert result.returncode == 0, (
        f"exit status {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def test_an_existing_path_or_symlink_is_left_alone_with_one_warning(tmp_path):
    taken = tmp_path / "taken.jsonl"
    taken.write_bytes(b"keep\n")
    target = tmp_path / "target.txt"
    target.write_bytes(b"target\n")
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    out = _ok(
        _child(
            f"""
            for path in ({str(taken)!r}, {str(link)!r}):
                print("RESULT", train(method("spsa"), trace=path).best_fitness)
            """,
            tmp_path,
        )
    )
    text = out.stdout + out.stderr
    assert text.count("RESULT") == 2
    assert text.count("call trace off") == 2
    assert text.count(str(taken)) == 1 and text.count(str(link)) == 1
    assert taken.read_bytes() == b"keep\n" and target.read_bytes() == b"target\n"
    assert link.is_symlink()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_trace_file_mode_is_0640(tmp_path):
    path = tmp_path / "mode.jsonl"
    _ok(
        _child(f'os.umask(0o022)\ntrain(method("spsa"), trace={str(path)!r})', tmp_path)
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL")
def test_killed_run_leaves_a_readable_trace(tmp_path):
    path = tmp_path / "killed.jsonl"
    code = PRELUDE + textwrap.dedent(
        f"""
        def ready(progress):
            if progress.iteration == 50:
                print("READY", flush=True)
        train(polypus.SPSA(max_iters=1_000_000), callback=ready, trace={str(path)!r})
        """
    )
    # The child's log lines share its stdout with READY, and so does anything
    # it writes to stderr: read the pipe's bytes as they come, up to READY.
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_env(tmp_path),
    )
    try:
        output, deadline = b"", time.monotonic() + 120
        while b"\nREADY\n" not in b"\n" + output:
            timeout = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([proc.stdout], [], [], timeout)
            chunk = os.read(proc.stdout.fileno(), 65536) if ready else b""
            assert chunk, f"no READY from the child:\n{output.decode(errors='replace')}"
            output += chunk
        proc.send_signal(signal.SIGKILL)
    finally:
        # A child that never got READY is killed too, so no run outlives the test.
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=30)
    trace = read_trace(path)
    assert not trace.complete
    assert trace.lines[0]["ev"] == "run"
    assert all(line["ev"] != "end" for line in trace.lines)


def test_env_mode_writes_one_file_per_run(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    out = _ok(
        _child(
            """
        first = train(method("spsa"), id="envrun")
        second = train(method("spsa"), id="envrun")
        third = polypus.run_quantum_circuit(polypus.Circuit(1).h(0).measure_all(), 10, "local", backend="polypus", seed=1)
        print(json.dumps({"pid": os.getpid(), "ids": [first.id, second.id, third.id]}))
        """,
            tmp_path,
            POLYPUS_CALL_TRACE=str(traces),
        )
    )
    info = json.loads(out.stdout.splitlines()[-1])
    pid, ids = info["pid"], info["ids"]
    # Each file is named after its run's effective id, the one in its header:
    # `run_quantum_circuit`'s (`run_1_local_<uuid>`) as `train`'s.
    expected = {f"{run_id}-{pid}-{n}.jsonl": run_id for n, run_id in enumerate(ids)}
    assert {path.name for path in traces.iterdir()} == set(expected)
    for name, run_id in expected.items():
        trace = read_trace(traces / name)
        check_trace(trace)
        assert trace.header["id"] == run_id, name


def test_a_file_named_after_the_pid_alone_does_not_stop_the_trace(tmp_path):
    # Processes that write to one directory can share a pid: in two
    # containers, on two hosts, or once the system hands a finished process's
    # pid on. A file that one of them left, or that another user put there, at
    # a name made of nothing but this pid and a count (`run-<pid>-<n>.jsonl`)
    # leaves the run's own trace alone, since its name holds the run's id.
    traces = tmp_path / "traces"
    traces.mkdir()
    out = _ok(
        _child(
            """
        planted = os.path.join(os.environ["POLYPUS_CALL_TRACE"], f"run-{os.getpid()}-0.jsonl")
        with open(planted, "x", encoding="utf-8") as file:
            file.write("planted\\n")
        result = polypus.run_quantum_circuit(polypus.Circuit(1).h(0).measure_all(), 8, "local", backend="polypus", seed=1)
        print(json.dumps({"pid": os.getpid(), "id": result.id}))
        """,
            tmp_path,
            POLYPUS_CALL_TRACE=str(traces),
        )
    )
    assert "call trace off" not in out.stdout + out.stderr
    info = json.loads(out.stdout.splitlines()[-1])
    pid, run_id = info["pid"], info["id"]
    planted, own = f"run-{pid}-0.jsonl", f"{run_id}-{pid}-0.jsonl"
    assert {path.name for path in traces.iterdir()} == {planted, own}
    assert (traces / planted).read_bytes() == b"planted\n"
    trace = read_trace(traces / own)
    check_trace(trace)
    assert trace.header["id"] == run_id


def test_runs_with_the_same_pid_and_count_get_different_names(tmp_path):
    # Two processes that write to one directory, in two containers or on two
    # hosts, can have the same pid, and each counts its traces from 0. Here
    # two processes each run the same circuit with the same seed: with the pid
    # taken out, their names still differ, as each holds its run's effective
    # id.
    traces = tmp_path / "traces"
    traces.mkdir()
    runs = []
    for _ in range(2):
        before = set(traces.iterdir())
        out = _ok(
            _child(
                """
            result = polypus.run_quantum_circuit(polypus.Circuit(1).h(0).measure_all(), 8, "local", backend="polypus", seed=1)
            print(json.dumps({"pid": os.getpid(), "id": result.id}))
            """,
                tmp_path,
                POLYPUS_CALL_TRACE=str(traces),
            )
        )
        info = json.loads(out.stdout.splitlines()[-1])
        [path] = set(traces.iterdir()) - before
        suffix = f"-{info['pid']}-0.jsonl"
        assert path.name.endswith(suffix), path.name
        runs.append((path.name.removesuffix(suffix), info["id"]))
    (first, first_id), (second, second_id) = runs
    assert first != second, f"both names are {first}-<pid>-0.jsonl"
    assert (first, second) == (first_id, second_id)


def test_the_trace_argument_wins_over_the_variable(tmp_path):
    traces, explicit = tmp_path / "traces", tmp_path / "explicit.jsonl"
    traces.mkdir()
    _ok(
        _child(
            f'train(method("spsa"), trace={str(explicit)!r})',
            tmp_path,
            POLYPUS_CALL_TRACE=str(traces),
        )
    )
    check_trace(read_trace(explicit))
    assert list(traces.iterdir()) == []


def test_env_mode_creates_no_directory(tmp_path):
    missing = tmp_path / "missing"
    out = _ok(
        _child(
            'train(method("spsa")); print("DONE")',
            tmp_path,
            POLYPUS_CALL_TRACE=str(missing),
        )
    )
    assert "DONE" in out.stdout
    assert (out.stdout + out.stderr).count("call trace off") == 1
    assert not missing.exists()


def test_an_empty_variable_is_off(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    out = _ok(
        _child('train(method("spsa"))', tmp_path, cwd=work, POLYPUS_CALL_TRACE="")
    )
    assert list(work.iterdir()) == []
    assert "call trace" not in out.stdout + out.stderr


@pytest.mark.parametrize("bad", ["../../x", "a" * 300, "naïve", ""])
def test_trace_id_is_sanitized(tmp_path, bad):
    # C-9 rejects each of these ids before a trace is opened, so the name rule
    # itself, hostile ids included, is pinned by the Rust unit test
    # `trace_file_name_is_sanitized`. Here: the rejected run leaves no trace
    # file, and an accepted id long enough to be cut gets the cut name. The
    # directory is nested so that a file that climbed out of it, as `../../x`
    # would, still lands under `tmp_path`, where the search below finds it.
    traces = tmp_path / "outer" / "inner" / "traces"
    traces.mkdir(parents=True)
    out = _ok(
        _child(
            f"""
        try:
            train(method("spsa"), id={bad!r})
        except polypus.ConfigurationError as err:
            print("REJECTED", err)
        long = train(method("spsa"), id="a" * 64)
        print(json.dumps({{"pid": os.getpid(), "id": long.id}}))
        """,
            tmp_path,
            POLYPUS_CALL_TRACE=str(traces),
        )
    )
    assert "REJECTED" in out.stdout, "C-9 rejects the id before anything runs"
    info = json.loads(out.stdout.splitlines()[-1])
    [path] = traces.iterdir()
    assert sorted(tmp_path.rglob("*.jsonl")) == [path]
    assert re.fullmatch(
        r"a{64}_[0-9a-f-]{35}-[0-9a-f]{8}-%d-0\.jsonl" % info["pid"], path.name
    )
    assert read_trace(path).header["id"] == info["id"]


def test_max_mb_caps_the_trace(tmp_path):
    path = tmp_path / "capped.jsonl"
    _ok(
        _child(
            f"train(polypus.SPSA(max_iters=6000), trace={str(path)!r})",
            tmp_path,
            POLYPUS_CALL_TRACE_MAX_MB="1",
        )
    )
    data = path.read_bytes()
    end_line = data.rstrip(b"\n").rsplit(b"\n", 1)[1]
    assert len(data) <= (1 << 20) + len(end_line) + 1
    trace = read_trace(path)
    check_trace(trace)
    assert (trace.end["status"], trace.end["truncated"]) == ("ok", True)


@pytest.mark.parametrize("raw", ["abc", "0"])
def test_an_invalid_max_mb_is_warned_about_once_and_ignored(tmp_path, raw):
    out = _ok(
        _child(
            f"""
        for name in ("a.jsonl", "b.jsonl"):
            train(method("spsa"), trace=os.path.join({str(tmp_path)!r}, name))
        """,
            tmp_path,
            POLYPUS_CALL_TRACE_MAX_MB=raw,
        )
    )
    assert (out.stdout + out.stderr).count(
        f'ignoring POLYPUS_CALL_TRACE_MAX_MB="{raw}"'
    ) == 1
    # Ignored, not applied as a cap of its own: a capped trace is complete too,
    # since its header and `end` line are always written, so it is the events
    # that tell the two apart.
    for name in ("a.jsonl", "b.jsonl"):
        trace = read_trace(tmp_path / name)
        check_trace(trace)
        assert trace.end["truncated"] is False and trace.events("batch"), name


@pytest.mark.integration
def test_native_trace_independent_of_budget(tmp_path):
    batches, waves = {}, {}
    for n_qpus, budget, threads in itertools.product(
        (1, 4), ("512K", "64G"), ("1", "4")
    ):
        traces = tmp_path / f"{n_qpus}-{budget}-{threads}"
        traces.mkdir()
        _ok(
            _child(
                f"""
            qc = polypus.Circuit(12)
            for q in range(12):
                qc.ry(q, polypus.Param(q))
            qc.measure_all()
            train(polypus.DE(generations=2, population_size=16, tolerance=0.0), qc=qc, dimensions=12, n_qpus={n_qpus})
            """,
                tmp_path,
                POLYPUS_CALL_TRACE=str(traces),
                POLYPUS_MEM_BUDGET=budget,
                RAYON_NUM_THREADS=threads,
            )
        )
        [path] = traces.iterdir()
        trace = read_trace(path)
        check_trace(trace)
        key = (n_qpus, budget, threads)
        batches[key] = [
            (b["k"], b["op"], b["n"], b["shots"], b["ok"])
            for b in trace.events("batch")
        ]
        waves[key] = len(trace.events("wave"))
    first = batches[(1, "64G", "1")]
    assert [b[2] for b in first] == [16, 16, 16]
    assert all(value == first for value in batches.values()), (
        "batch events are identical (C-7)"
    )
    for (_, budget, _), count in waves.items():
        if budget == "64G":
            assert count == 3
        else:
            assert count > 3, (
                "512K holds 8 statevectors of 12 qubits, so batches of 16 split"
            )


def test_trace_contains_only_schema_fields(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    canary = "canary-6d1f2b"
    _ok(
        _child(
            """
        train(method("spsa"), initial_point=[0.1234567891, -0.9876543219], final_shots=128)
        polypus.run_quantum_circuit(polypus.Circuit(2).h(0).cx(0, 1).measure_all(), 64, "local", n_qpus=2, backend="polypus", seed=3)
        """,
            tmp_path,
            POLYPUS_CALL_TRACE=str(traces),
            POLYPUS_TRACE_CANARY=canary,
        )
    )
    paths = sorted(traces.iterdir())
    assert len(paths) == 2
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for needle in (
            canary,
            "0.1234567891",
            "0.9876543219",
            "OPENQASM",
            "qreg",
            "measure",
            '"00"',
            '"11"',
            str(tmp_path),
        ):
            assert needle not in text, needle
        check_trace(read_trace(path))  # exact key sets for every event
