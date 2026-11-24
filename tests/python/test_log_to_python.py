"""
``polypus.log_to_python()``: Rust log records reach Python's ``logging``
(issue #458).

Only one log sink can be installed per process, and ``test_logging.py``
installs ``init_logger``'s in the test runner itself, so every test here runs
in a child interpreter of its own. Each child configures the ``polypus``
logger with a handler that keeps the records it gets, and prints what the
test checks as one ``RESULT <json>`` line.

* records carry the logger name ``polypus.<crate>.<module>``, the Rust source
  location and the message already formatted;
* the level snapshot is refreshed at the start of every call, so a level
  changed between two calls takes effect on the second one;
* ``init_logger`` and ``log_to_python`` share one install: whichever comes
  second is a no-op with a ``UserWarning``;
* ``logging.disable`` and a handler that raises are honoured as for any
  Python logger;
* a record emitted while the interpreter shuts down never panics: an error
  goes to stderr as one fixed line.
"""

import json
import os
import re
import signal
import subprocess
import sys

import pytest

pytest.importorskip("polypus")

# Far above what any child needs (a few seconds): only a hang reaches it.
_TIMEOUT_S = 120.0

# A child's common start: a handler on the `polypus` logger that keeps every
# record, and a one-generation-per-record DE training on the native backend.
_PRELUDE = r"""
import json, logging, warnings
import polypus


class Keep(logging.Handler):
    def __init__(self):
        super().__init__(logging.NOTSET)
        self.records = []

    def emit(self, record):
        self.records.append(record)


keep = Keep()
top = logging.getLogger("polypus")
top.addHandler(keep)


def train(run_id):
    polypus.train(
        polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all(),
        polypus.DE(generations=5, population_size=4, patience=2),
        shots=8,
        n_qpus=1,
        dimensions=1,
        expectation_function=lambda b: float(b == "1"),
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id=run_id,
        backend="polypus",
        seed=1,
    )


def result(value):
    print("RESULT " + json.dumps(value), flush=True)
"""


def _run_child(source: str, **env: str) -> subprocess.CompletedProcess:
    """Run ``source`` in a fresh interpreter and return it once it has exited."""
    full_env = dict(os.environ, POLYPUS_NO_AUTOCALIBRATE="1", **env)
    try:
        return subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            env=full_env,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"the child hung (killed after {_TIMEOUT_S:.0f} s)\n"
            f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        )


def _assert_exited_cleanly(child: subprocess.CompletedProcess) -> None:
    """No signal, a normal exit status, and no panic anywhere."""
    output = f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}"
    if child.returncode < 0:
        pytest.fail(
            f"the child was killed by {signal.Signals(-child.returncode).name}\n{output}"
        )
    assert child.returncode == 0, output
    for marker in ("panicked", "PanicException"):
        assert marker not in child.stderr, output
        assert marker not in child.stdout, output


def _result(source: str, **env: str):
    """The value the child printed on its ``RESULT`` line."""
    child = _run_child(source, **env)
    _assert_exited_cleanly(child)
    lines = [line for line in child.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}"
    return json.loads(lines[0][len("RESULT ") :])


def test_records_reach_python_logging_with_their_rust_location():
    value = _result(
        _PRELUDE
        + r"""
top.setLevel(logging.DEBUG)
assert polypus.log_to_python() is None
train("bridge")
result([
    {
        "name": r.name,
        "levelno": r.levelno,
        "pathname": r.pathname,
        "lineno": r.lineno,
        "message": r.getMessage(),
        "msg": r.msg,
        "args": list(r.args),
    }
    for r in keep.records
])
"""
    )
    generations = [
        r
        for r in value
        if r["levelno"] == 10
        and r["name"].startswith("polypus.optimizers")
        and "Generation" in r["message"]
    ]
    assert generations, value
    for record in generations:
        assert record["name"].startswith("polypus.optimizers."), record
        assert record["pathname"].endswith(".rs"), record
        assert "polypus-optimizers" in record["pathname"], record
        assert record["lineno"] > 0, record
    # Every record is handed over already formatted, with no `%` arguments.
    for record in value:
        assert record["args"] == [] and record["msg"] == record["message"], record
        assert record["name"] == "polypus" or record["name"].startswith("polypus."), (
            record
        )
    # The binding's own records land under `polypus.bindings`.
    assert any(r["name"] == "polypus.bindings" for r in value), value


def test_a_level_set_between_two_calls_applies_to_the_second():
    value = _result(
        _PRELUDE
        + r"""
top.setLevel(logging.INFO)
polypus.log_to_python()
train("at-info")
at_info = [r.levelno for r in keep.records]
keep.records.clear()
top.setLevel(logging.DEBUG)
train("at-debug")
result({"info": at_info, "debug": [r.levelno for r in keep.records]})
"""
    )
    assert value["info"], "INFO records arrive at INFO"
    assert min(value["info"]) >= 20, value
    assert 10 in value["debug"], value


@pytest.mark.parametrize(
    "first, second",
    [("log_to_python", "init_logger"), ("init_logger", "log_to_python")],
)
def test_the_second_install_is_a_noop_with_a_warning(tmp_path, first, second):
    log = tmp_path / "polypus.log"
    calls = {
        "log_to_python": "polypus.log_to_python()",
        "init_logger": f"polypus.init_logger(file={str(log)!r})",
    }
    value = _result(
        _PRELUDE
        + f"""
first = {calls[first]}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    second = {calls[second]}
result({{
    "second": second,
    "warnings": [
        (w.category.__name__, str(w.message)) for w in caught
    ],
}})
"""
    )
    assert value["second"] is None, value
    assert [category for category, _ in value["warnings"]] == ["UserWarning"], value
    assert f"ignoring this {second} call" in value["warnings"][0][1], value
    if first == "log_to_python":
        assert not log.exists(), "the ignored init_logger must not create its file"


def test_log_to_python_twice_is_a_noop_with_a_warning():
    value = _result(
        _PRELUDE
        + r"""
top.setLevel(logging.DEBUG)
polypus.log_to_python()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    second = polypus.log_to_python()
train("still-bridged")
result({
    "second": second,
    "warnings": [w.category.__name__ for w in caught],
    "records": len(keep.records),
})
"""
    )
    assert value["second"] is None, value
    assert value["warnings"] == ["UserWarning"], value
    assert value["records"] > 0, "the first install keeps working"


def test_logging_disable_and_a_raising_handler_are_honoured():
    value = _result(
        _PRELUDE
        + r"""
top.setLevel(logging.DEBUG)
polypus.log_to_python()
logging.disable(logging.CRITICAL)
train("disabled")
disabled = len(keep.records)
logging.disable(logging.NOTSET)


class Raises(logging.Handler):
    def emit(self, record):
        raise RuntimeError("handler failure")


raising = Raises()
top.addHandler(raising)
train("raising")
result({"disabled": disabled, "after": len(keep.records)})
"""
    )
    assert value["disabled"] == 0, value
    # The handler's exception is reported by `Handler.handleError` and does
    # not stop the run, nor the other handlers.
    assert value["after"] > 0, value


# The main thread exits while a daemon thread's training run is in flight, as in
# `test_interpreter_shutdown.py` (see its docstring for why the 20-qubit QNG
# wave and the `__del__` of an object in `sys.modules` are needed). With the
# bridge installed, the first attach the run makes after the interpreter has
# started finalizing is the `variance_function` call: it fails, and the run
# records that failure with an `ERROR` record, which the bridge can no longer
# hand to Python. It goes to stderr as one fixed line; the DEBUG records the
# run emits around it are dropped.
_SHUTDOWN_CHILD = r"""
import logging, os, resource, sys, threading, time
import polypus


class Stdout(logging.Handler):
    def __init__(self):
        super().__init__(logging.NOTSET)
        self.write = os.write

    def emit(self, record):
        self.write(1, ("PY " + record.levelname + " " + record.name + "\n").encode())


top = logging.getLogger("polypus")
top.setLevel(logging.DEBUG)
top.addHandler(Stdout())
polypus.log_to_python()

qc = polypus.Circuit(20)
for _ in range(20):
    for q in range(20):
        qc = qc.ry(q, polypus.Param(0)).cx(q, (q + 1) % 20)
qc = qc.measure_all()

started = threading.Event()


def train():
    started.set()
    try:
        polypus.train(
            qc,
            polypus.QNG(variance_function=lambda theta, a: 1.0, max_iters=2),
            shots=8,
            n_qpus=1,
            dimensions=1,
            expectation_function=lambda bitstring: 0.0,
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id="shutdown",
            backend="polypus",
            seed=1,
        )
        os.write(1, b"RUN RETURNED\n")
    except BaseException as exc:
        os.write(1, b"RUN RAISED " + type(exc).__name__.encode() + b"\n")


threading.Thread(target=train, daemon=True).start()
started.wait()
time.sleep(0.3)


class AwaitTheRun:
    def __init__(self):
        self.getrusage, self.who = resource.getrusage, resource.RUSAGE_SELF
        self.sleep, self.clock, self.write = time.sleep, time.monotonic, os.write

    def cpu(self):
        usage = self.getrusage(self.who)
        return usage.ru_utime + usage.ru_stime

    def __del__(self):
        start, last, idle = self.clock(), self.cpu(), 0.0
        while idle < 1.0 and self.clock() - start < 60:
            self.sleep(0.1)
            now = self.cpu()
            idle = idle + 0.1 if now - last < 0.01 else 0.0
            last = now
        self.write(1, b"FINALIZER SETTLED\n")


sys.modules["_polypus_await_the_run"] = AwaitTheRun()
"""


@pytest.mark.integration
@pytest.mark.skipif(
    sys.platform == "win32", reason="the child reads CPU time with `resource`"
)
def test_a_record_at_interpreter_shutdown_goes_to_stderr_without_a_panic():
    child = _run_child(_SHUTDOWN_CHILD, RAYON_NUM_THREADS="1")
    _assert_exited_cleanly(child)
    output = f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}"
    # The bridge was live before the shutdown: the run's DEBUG records reached
    # the Python handler.
    assert "PY DEBUG polypus." in child.stdout, output
    # The interpreter really was shutting down while the run went on, and the
    # run never returned to Python.
    assert "FINALIZER SETTLED" in child.stdout, output
    assert "RUN " not in child.stdout, output
    # The failure recorded at shutdown is on stderr, as the fixed line.
    fixed = re.compile(
        r"^polypus: ERROR polypus(\.\w+)+: run shutdown\S*: .*interpreter is unavailable",
        re.MULTILINE,
    )
    assert fixed.search(child.stderr), output
    # Less severe records are dropped, not written.
    assert not re.search(r"^polypus: (INFO|DEBUG)", child.stderr, re.MULTILINE), output
