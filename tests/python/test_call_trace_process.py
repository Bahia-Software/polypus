"""
The call trace of ``polypus.train`` (contract C-12) seen from outside the
process that writes it.

Each case runs in a fresh interpreter with the logger on the console at the
``warn`` level: a path where something already exists, a file or a symbolic
link, is left as it is with one warning naming it, and the run goes on
without a trace; a new trace file gets mode ``0640``; and a run killed with
``SIGKILL`` leaves a trace that reads up to its last whole line, without an
``end`` line.
"""

import os
import select
import signal
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from call_trace import read_trace

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
