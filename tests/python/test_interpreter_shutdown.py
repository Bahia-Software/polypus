"""
A run that is still going when the interpreter shuts down ends without a panic
(issues #267 and #269).

Polypus runs a training or circuit run with the thread *detached* from the
interpreter (``py.detach``), and code inside it attaches again where it needs
Python: a user callback, the Aer seam, binding a Qiskit circuit. Each of those
attaches is a fresh one, so it is checked against the interpreter's state. Once
the interpreter has started finalizing, ``Py_IsInitialized()`` is already 0, and
PyO3's ``Python::attach`` then either panics (when it is the process's first
fresh attach), parks the thread for good (a hang), or attaches to the finalizing
interpreter. Every such site now attaches through ``attach_or`` and returns a
typed error instead, which the FFI edge raises as ``KeyboardInterrupt``
(contract C-1, failure modes); see ``docs/ENGINEERING.md`` §9.

Both tests run a child interpreter and keep it finalizing while the run hits one
of those sites, then check that the child exits normally (no abort, no hang) and
that nothing panicked. Keeping it finalizing needs code that runs *after*
``Py_IsInitialized()`` becomes 0: an ``atexit`` handler runs too early, and a
``__del__`` on a global of ``__main__`` does not run at all while a daemon
thread is alive (measured on CPython 3.13). An object stored in ``sys.modules``
is released while the interpreter clears that dict, late in finalization, so its
``__del__`` is the hook.

* ``test_a_training_run_on_a_daemon_thread_survives_shutdown`` is the scenario
  of issue #269: ``polypus.train`` on a ``daemon=True`` thread while the main
  thread exits. QNG on the native backend makes it fail without the fix: its
  first wave (a 20-qubit simulation, pure Rust) is still running when the main
  thread exits, so the first fresh attach of the process is the
  ``variance_function`` call that QNG makes after it. Without the fix that
  attach panics; with it, the oracle records the failure and the run winds down.
  The interrupt guard that runs between the wave and that call already attached
  safely, which is why a callback reached through the planner would not show the
  difference. The test can catch a regression only while no fresh attach happens
  in the process before the first wave ends: if one is added (a poll of the
  guard before the first wave, say), the attach at shutdown parks the thread
  instead of panicking, and this test can no longer fail.
* ``test_a_run_started_while_finalizing_aborts_as_keyboard_interrupt`` starts
  ``polypus.run_quantum_circuit`` on the Aer backend from that ``__del__``
  itself, so the run's first attach (the planner asking the backend how wide its
  circuits are) happens with the interpreter finalizing. It pins the exception
  class the run fails with. A ``run_quantum_circuit`` on a daemon thread cannot
  show the difference: on Aer it attaches as soon as it starts, before the main
  thread can exit, and on the native backend only the interrupt guard attaches,
  which was already safe.

What a daemon thread does after the run is CPython's business: with or without
the fix, CPython parks the thread when it next tries to take the GIL, and the
process exits. Without the fix, the first test panics on Python 3.11 and 3.13
(measured), and so does the second.

The child runs the native simulation single-threaded (``RAYON_NUM_THREADS=1``),
as in ``test_interrupt.py``, so its first wave lasts long enough on any runner.
"""

import os
import signal
import subprocess
import sys

import pytest

pytest.importorskip("polypus")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32", reason="the children read CPU time with `resource`"
    ),
]

# Far above what either child needs (a few seconds): only a hang reaches it.
_TIMEOUT_S = 120.0

# The main thread exits while the daemon thread's first QNG wave is running.
# `started` is set just before `polypus.train` is called; the setup before its
# GIL-free run takes milliseconds, and the wave takes over a second
# single-threaded, so the 0.3 s sleep lands inside the wave. `AwaitTheRun` then
# keeps the interpreter finalizing until the process has stopped using CPU,
# that is, until the run has reached `variance_function` and the daemon thread
# is parked, or it gives up after 60 s.
_DAEMON_TRAINING_CHILD = r"""
import os, resource, sys, threading, time
import polypus

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

# The run is started from a `__del__` that runs while the interpreter finalizes.
_FINALIZING_RUN_CHILD = r"""
import os, sys
import polypus

qc = polypus.Circuit(1).x(0).measure_all()


class RunWhileFinalizing:
    def __init__(self):
        self.run, self.write = polypus.run_quantum_circuit, os.write

    def __del__(self):
        try:
            self.run(qc, shots=8, infrastructure="local", backend="aer", seed=1)
            self.write(1, b"RUN RETURNED\n")
        except BaseException as exc:
            name, message = type(exc).__name__, str(exc)
            self.write(1, ("RUN RAISED " + name + ": " + message + "\n").encode())


sys.modules["_polypus_run_while_finalizing"] = RunWhileFinalizing()
"""


def _run_child(source: str) -> subprocess.CompletedProcess:
    """Run `source` in a fresh interpreter and return it once it has exited."""
    env = dict(os.environ, POLYPUS_NO_AUTOCALIBRATE="1", RAYON_NUM_THREADS="1")
    try:
        return subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"the child hung at interpreter shutdown (killed after {_TIMEOUT_S:.0f} s)\n"
            f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        )


def _assert_exited_cleanly(child: subprocess.CompletedProcess) -> None:
    """No abort or other signal, a normal exit status, and no panic anywhere."""
    output = f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}"
    if child.returncode < 0:
        pytest.fail(
            f"the child was killed by {signal.Signals(-child.returncode).name}\n{output}"
        )
    assert child.returncode == 0, output
    for marker in ("panicked", "PanicException"):
        assert marker not in child.stderr, output
        assert marker not in child.stdout, output


def test_a_training_run_on_a_daemon_thread_survives_shutdown():
    child = _run_child(_DAEMON_TRAINING_CHILD)
    _assert_exited_cleanly(child)
    # The finalizer ran, so the interpreter really was shutting down while the
    # run went on, and the run never returned to Python: it was still in flight.
    assert "FINALIZER SETTLED" in child.stdout, child.stdout
    assert "RUN " not in child.stdout, child.stdout


def test_a_run_started_while_finalizing_aborts_as_keyboard_interrupt():
    child = _run_child(_FINALIZING_RUN_CHILD)
    _assert_exited_cleanly(child)
    # The run cannot attach, so it is aborted the way an interrupted run is.
    assert "RUN RAISED KeyboardInterrupt: the run was aborted: " in child.stdout, (
        child.stdout + child.stderr
    )
