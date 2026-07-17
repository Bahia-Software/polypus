"""
Ctrl+C responsiveness and GIL release during long-running calls
(issues #36 and #73).

Acceptance criterion: a ``KeyboardInterrupt`` takes effect *promptly* while
``polypus.train`` (native backend) or ``polypus.qml.train`` (Qiskit/Aer
backend) is running, rather than being ignored until the whole optimization
finishes.

``run_quantum_circuit`` (issue #73) is covered here too: it releases the GIL
around the whole run and calls ``py.check_signals()`` at the per-circuit /
pre-result-conversion boundary in both orchestration variants
(``AlgorithmSingleRun`` for ``n_qpus == 1``, ``DistributeByShotsRun`` for
``n_qpus > 1``). Two properties are asserted: a pending SIGINT surfaces as a
``KeyboardInterrupt`` (not ``COMPLETED``) when the run returns, and another
Python thread makes real progress while a native run is in flight (proving the
GIL is actually released). Unlike ``train``, ``run_quantum_circuit`` has no
per-shot/per-gate signal checks (out of scope for #73), so the interrupt is
honored when the run *reaches* that boundary — i.e. when the run completes.
Interrupt latency is therefore bounded by the run's own duration, which the two
SIGINT children keep independent of the runner's core count by pinning
``RAYON_NUM_THREADS=1`` (forcing the native statevector sim single-threaded, see
``polypus-sim`` `parallel` kernels): without it a circuit sized to ~1.3s on a
many-core dev box balloons past ``_INTERRUPT_DEADLINE_S`` on a 2-4 core CI
runner. The circuits are sized to ~1.3s single-threaded — long enough that the
run is still in flight when SIGINT arrives ``_DELAY_BEFORE_SIGINT_S`` after
READY, short enough to complete well inside ``_INTERRUPT_DEADLINE_S`` even on a
slower CI core.

The GIL-release test instead uses a small (sub-``parallel_threshold``) circuit
that never engages the rayon pool at all, so the native run occupies a single
core and leaves the others free for the counter thread — the only configuration
in which a CPU-bound Python thread can be *observed* to make progress, and thus
prove the GIL was released, even on a 2-core runner.

Why both entry points: `train`'s `VqcOracle` and `qml.train`'s `QmlOracle`
share `run_and_evaluate`, but reach it through different paths — a native,
GIL-free simulation loop for `train`, and Tokio `spawn_blocking` workers
(`allow_threads` around a `block_on`, needed to avoid the deadlock documented
in `qml_oracle.rs`) for `qml.train`. Only the calling (main) thread can have a
pending SIGINT turned into `KeyboardInterrupt` (`PyErr_CheckSignals` is a
no-op off the main thread), so `qml.train` additionally checks signals once on
the main thread after its workers join. Both paths need their own proof.

Why a subprocess rather than an in-process background thread:

* Isolation — the SIGINT goes to the child, so a mishandled signal can never
  abort the pytest session itself.
* No hang on regression — the child is given an effectively unbounded training
  budget (so a *completed* run would take far longer than the interrupt window),
  and the parent enforces a hard wall-clock timeout: if the child does not
  respond to SIGINT, the parent kills it and fails with a clear message instead
  of blocking forever.
* The child's `train`/`qml.train` runs on *its* main thread, which is where
  CPython delivers signals — matching real interactive use.

Manual verification (complementary to these automated tests):

    $ python -c "
    import polypus
    qc = polypus.Circuit(16)
    for q in range(16): qc = qc.h(q)
    for i in range(4): qc = qc.ry(i, polypus.Param(i))
    for q in range(15): qc = qc.cx(q, q + 1)
    qc = qc.measure_all()
    polypus.train(qc, polypus.DE(generations=1500, population_size=8),
                  shots=1024, n_qpus=1, dimensions=4,
                  expectation_function=lambda b: float(b.count('1')),
                  infrastructure='local', nodes=1, cores_per_qpu=1,
                  id='manual', backend='polypus')
    "
    # ...then press Ctrl+C: training must stop within about a second with a
    # KeyboardInterrupt traceback, not run for the full ~minute-long budget.
    # Swap in polypus.qml.train(...) (see TestQmlTrainSeed in
    # test_seed_reproducibility.py for its call shape) to check the QML path
    # the same way.
"""

import os
import select
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.integration

# Generous margins (the true interrupt latency is well under a second): these
# bounds only need to be far below the "completed run" budget below to prove
# "promptly", not pin down an exact latency — avoids flakiness from CI jitter.
_READY_TIMEOUT_S = 60.0  # child import + one warm-up generation on a cold CI runner
_DELAY_BEFORE_SIGINT_S = 0.5  # let the GIL-free optimization get going
_INTERRUPT_DEADLINE_S = 5.0  # hard ceiling; real value <1s, full run far longer

# The child trains on the native backend with a budget whose *completed* run
# takes far longer than the interrupt window (~a minute), so any prompt exit
# proves the interrupt was honored. A one-off warm-up run forces the
# `polypus_python` / qiskit import (done lazily on the first evaluation) *before*
# the timed window, and READY is printed only once training has started, so the
# measured elapsed reflects interrupt latency and not process/import warm-up.
#
# Sizing note: the 16-qubit circuit makes each generation's simulation the
# dominant cost (~tens of ms), so a *completed* run takes far longer than 20s,
# while the number of generations is small enough that the post-interrupt
# wind-down (the optimizer's own per-generation bookkeeping, with evaluation
# short-circuited) stays in the millisecond range. Ctrl+C is captured within
# one generation; what follows is negligible.
_NATIVE_CHILD = r"""
import sys, time
import polypus

qc = polypus.Circuit(16)
for q in range(16):
    qc = qc.h(q)
for i in range(4):
    qc = qc.ry(i, polypus.Param(i))
for q in range(15):
    qc = qc.cx(q, q + 1)
qc = qc.measure_all()

def expect(bitstring):
    return float(bitstring.count("1"))

# Warm up the lazy import path (and JIT-free code paths) with a 1-generation run.
polypus.train(
    qc, polypus.DE(generations=1, population_size=8, tolerance=1e-12),
    shots=1024, n_qpus=1, dimensions=4, expectation_function=expect,
    infrastructure="local", nodes=1, cores_per_qpu=1, id="warmup", backend="polypus",
)

print("READY", flush=True)
start = time.time()
try:
    polypus.train(
        qc,
        polypus.DE(generations=1500, population_size=8, tolerance=1e-12),
        shots=1024,
        n_qpus=1,
        dimensions=4,
        expectation_function=expect,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="interrupt_test",
        backend="polypus",
    )
    print("COMPLETED", flush=True)
except KeyboardInterrupt:
    print(f"KEYBOARDINTERRUPT {time.time() - start:.3f}", flush=True)
except BaseException as exc:  # e.g. a PanicException from a swallowed error
    print(f"OTHER {type(exc).__name__}", flush=True)
    sys.exit(1)
"""

# Same shape as `_NATIVE_CHILD`, but exercising `QmlOracle` (Qiskit/Aer, native
# backend rejected — see `docs/CONTRACTS.md` C-7) instead of `VqcOracle`. Each
# generation here costs ~10ms (measured: 4 training circuits x population_size=6
# through Aer), so 100000 generations is far beyond anything the interrupt
# window (well under a second) could let complete, without needing a slow or
# large circuit.
_QML_CHILD = r"""
import sys, time
import numpy as np
import polypus
from qiskit.circuit.library import real_amplitudes, zz_feature_map

feature_map = zz_feature_map(feature_dimension=2, reps=1)
ansatz = real_amplitudes(num_qubits=2, reps=1)
x_train = np.zeros((4, 2))

def expect(bitstring):
    return sum(int(c) for c in bitstring) / len(bitstring)

# Warm up the lazy import path (qiskit/Aer) with a 1-generation run.
polypus.qml.train(
    feature_map, ansatz, x_train,
    polypus.DE(generations=1, population_size=6, tolerance=1e-12),
    shots=64, n_qpus=1, dimensions=len(ansatz.parameters),
    expectation_function=expect,
    infrastructure="local", nodes=1, cores_per_qpu=1, id="warmup",
)

print("READY", flush=True)
start = time.time()
try:
    polypus.qml.train(
        feature_map, ansatz, x_train,
        polypus.DE(generations=100000, population_size=6, tolerance=1e-12),
        shots=64, n_qpus=1, dimensions=len(ansatz.parameters),
        expectation_function=expect,
        infrastructure="local", nodes=1, cores_per_qpu=1, id="qml_interrupt_test",
    )
    print("COMPLETED", flush=True)
except KeyboardInterrupt:
    print(f"KEYBOARDINTERRUPT {time.time() - start:.3f}", flush=True)
except BaseException as exc:  # e.g. a PanicException from a swallowed error
    print(f"OTHER {type(exc).__name__}", flush=True)
    sys.exit(1)
"""


def _assert_responds_to_sigint_promptly(child_code, *, failure_hint, env=None):
    """Run `child_code` in a subprocess, SIGINT it mid-run, and assert a real
    `KeyboardInterrupt` was raised well before a completed run could finish.
    Shared by the training and run_quantum_circuit variants below — they differ
    only in which entry point/backend the child script exercises (and, for
    run_quantum_circuit, an `env` that pins the sim single-threaded)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        # Wait for the child to reach the training loop before interrupting.
        ready, _, _ = select.select([proc.stdout], [], [], _READY_TIMEOUT_S)
        if not ready:
            raise AssertionError("child did not start training within the timeout")
        first_line = proc.stdout.readline().strip()
        assert first_line == "READY", (
            f"unexpected child startup output {first_line!r}; "
            f"stderr:\n{proc.stderr.read()}"
        )

        time.sleep(_DELAY_BEFORE_SIGINT_S)
        proc.send_signal(signal.SIGINT)

        try:
            out, err = proc.communicate(timeout=_INTERRUPT_DEADLINE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            pytest.fail(
                f"{failure_hint} did not respond to SIGINT within "
                f"{_INTERRUPT_DEADLINE_S}s.\nstdout:\n{out}\nstderr:\n{err}"
            )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    # A real KeyboardInterrupt must have been raised — not a PanicException
    # (which is what swallowing the signal with `.expect()` would produce) and
    # not "COMPLETED" (the run must not have finished the huge budget).
    assert "KEYBOARDINTERRUPT" in out, (
        f"expected a KeyboardInterrupt; got stdout:\n{out}\nstderr:\n{err}"
    )
    elapsed = float(out.split("KEYBOARDINTERRUPT", 1)[1].split()[0])
    assert elapsed < _INTERRUPT_DEADLINE_S, (
        f"interrupt was honored but not promptly: {elapsed:.2f}s "
        f"(a full run would take far longer)"
    )


def test_native_training_responds_to_sigint_promptly():
    _assert_responds_to_sigint_promptly(
        _NATIVE_CHILD,
        failure_hint=(
            "native-backend training (VqcOracle) — the GIL is likely held "
            "end-to-end or check_signals is missing"
        ),
    )


def test_qml_training_responds_to_sigint_promptly():
    _assert_responds_to_sigint_promptly(
        _QML_CHILD,
        failure_hint=(
            "qml.train (QmlOracle) — the main-thread check_signals after the "
            "worker join is likely missing or the GIL is not released around "
            "the optimizer"
        ),
    )


# --------------------------------------------------------------------------- #
# run_quantum_circuit (issue #73)
# --------------------------------------------------------------------------- #

# Env pinning the native sim single-threaded in the SIGINT children (see module
# docstring and the `parallel` kernels in polypus-sim): makes each run's
# duration independent of the runner's core count. Inherit the rest of the
# environment so the child still finds the installed `polypus`/`polypus_python`.
_SINGLE_THREADED_ENV = {**os.environ, "RAYON_NUM_THREADS": "1"}

# Shared circuit builder used by every child below.
_MAKE_CIRCUIT_SRC = r"""
def make(n, reps):
    qc = polypus.Circuit(n)
    for q in range(n):
        qc = qc.h(q)
    for _ in range(reps):
        for q in range(n - 1):
            qc = qc.cx(q, q + 1)
        for q in range(n):
            qc = qc.rx(q, 0.3)
    return qc.measure_all()
"""

# Child template for the SIGINT tests, run with `_SINGLE_THREADED_ENV`. A cheap
# warm-up forces the lazy `polypus_python` import before the timed window;
# `READY` is printed only once that is done, so the measured elapsed reflects
# the native run itself. Because the interrupt is only honored at the
# per-circuit / pre-result-conversion boundary (i.e. when the run completes — no
# per-shot checks), the circuit is sized to complete in ~1.3s single-threaded:
# still running when SIGINT arrives `_DELAY_BEFORE_SIGINT_S` after READY, done
# well inside `_INTERRUPT_DEADLINE_S`. `__N_QPUS__` selects the variant:
# 1 -> AlgorithmSingleRun, >1 -> DistributeByShotsRun (which duplicates the
# circuit across QPUs, so its per-QPU `reps` is scaled down to keep the total
# run time comparable).
_RUN_CHILD_TEMPLATE = (
    r"""
import sys, time
import polypus
"""
    + _MAKE_CIRCUIT_SRC
    + r"""
# Cheap warm-up: pay the lazy import cost outside the timed window.
polypus.run_quantum_circuit(
    make(2, 1), shots=64, infrastructure="local",
    n_qpus=__N_QPUS__, backend="polypus",
)

qc = make(__N__, __REPS__)
print("READY", flush=True)
start = time.time()
try:
    polypus.run_quantum_circuit(
        qc, shots=4096, infrastructure="local",
        n_qpus=__N_QPUS__, backend="polypus",
    )
    print("COMPLETED", flush=True)
except KeyboardInterrupt:
    print(f"KEYBOARDINTERRUPT {time.time() - start:.3f}", flush=True)
except BaseException as exc:  # e.g. a PanicException from a swallowed signal
    print(f"OTHER {type(exc).__name__}", flush=True)
    sys.exit(1)
"""
)


def _run_child(*, n_qpus, n, reps):
    """Build a `run_quantum_circuit` SIGINT child source for the given variant.

    Uses token replacement rather than `str.format` because the template's
    f-string (`{time.time() ...}`) contains literal braces."""
    return (
        _RUN_CHILD_TEMPLATE.replace("__N_QPUS__", str(n_qpus))
        .replace("__N__", str(n))
        .replace("__REPS__", str(reps))
    )


def test_run_quantum_circuit_single_responds_to_sigint_promptly():
    # n_qpus == 1 -> AlgorithmSingleRun. n=18/reps=100 is ~1.3s single-threaded.
    _assert_responds_to_sigint_promptly(
        _run_child(n_qpus=1, n=18, reps=100),
        failure_hint=(
            "run_quantum_circuit (AlgorithmSingleRun) — the GIL is likely held "
            "for the whole run or check_signals is missing before result "
            "conversion"
        ),
        env=_SINGLE_THREADED_ENV,
    )


def test_run_quantum_circuit_distributed_responds_to_sigint_promptly():
    # n_qpus > 1 -> DistributeByShotsRun. 4 QPUs x n=18/reps=25 is ~1.3s total.
    _assert_responds_to_sigint_promptly(
        _run_child(n_qpus=4, n=18, reps=25),
        failure_hint=(
            "run_quantum_circuit (DistributeByShotsRun) — the GIL is likely "
            "held for the whole run or check_signals is missing before the "
            "merge/result conversion"
        ),
        env=_SINGLE_THREADED_ENV,
    )


# GIL-release proof: while a native `run_quantum_circuit` call is in flight, a
# background Python thread spins a tight counter loop. If the GIL were held for
# the whole run the thread is starved and advances only by the incidental amount
# the call's Python entry/exit stages allow (measured ~1.4e5, stable across core
# counts); with the GIL released it advances by tens of millions (measured
# ~3.6e7 on 32 cores, ~4e7 on 2) over the run. The threshold sits ~35x above the
# held-GIL value and ~7x below the released-GIL value, so it separates them
# cleanly with wide margin on either side, independent of the runner's cores.
#
# Two design points make this robust (see module docstring):
#   * A subprocess, not an in-process thread, so the measurement starts from a
#     clean interpreter with no rayon pool initialised by earlier tests.
#   * An 11-qubit circuit — below polypus-sim's `parallel_threshold` (12) — so
#     the run stays on a single thread and never engages the rayon pool. A
#     parallel (>=12q) run instead saturates every core with rayon workers and
#     starves the counter thread *even when the GIL is free*, which would make
#     the observation depend on spare cores the CI runner may not have. Depth
#     comes from `n_qpus` (sequential per-QPU circuits), not qubit count, so the
#     circuit stays cheap to build.
_MIN_COUNTER_PROGRESS = 5_000_000

_GIL_RELEASE_CHILD = (
    r"""
import threading
import polypus
"""
    + _MAKE_CIRCUIT_SRC
    + r"""
# Warm up the lazy import path outside the measured window.
polypus.run_quantum_circuit(
    make(2, 1), shots=64, infrastructure="local", n_qpus=1, backend="polypus"
)

qc = make(11, 1500)  # sub-parallel-threshold; ~1.3-2.6s across 32..2 cores
counter = 0
stop = threading.Event()

def spin():
    global counter
    while not stop.is_set():
        counter += 1

worker = threading.Thread(target=spin)
worker.start()
try:
    before = counter
    # n_qpus=20 -> 20 sequential single-thread circuits (DistributeByShotsRun),
    # enough wall-clock for the counter to accumulate a decisive lead.
    polypus.run_quantum_circuit(
        qc, shots=4096, infrastructure="local", n_qpus=20, backend="polypus"
    )
    advanced = counter - before
finally:
    stop.set()
    worker.join()
print(f"ADVANCED {advanced}", flush=True)
"""
)


def test_run_quantum_circuit_releases_gil_for_other_threads():
    proc = subprocess.run(
        [sys.executable, "-c", _GIL_RELEASE_CHILD],
        capture_output=True,
        text=True,
        timeout=_READY_TIMEOUT_S,
    )
    assert proc.returncode == 0 and "ADVANCED" in proc.stdout, (
        f"GIL-release child did not complete cleanly.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    advanced = int(proc.stdout.split("ADVANCED", 1)[1].split()[0])
    assert advanced > _MIN_COUNTER_PROGRESS, (
        f"background thread advanced only {advanced} counts during the native "
        f"run — the GIL was likely not released around run_quantum_circuit"
    )
