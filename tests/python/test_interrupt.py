"""
Ctrl+C responsiveness and GIL release during long-running calls
(issues #36, #73, #86 and #110).

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

The GIL-release tests instead use a small (sub-``parallel_threshold``) circuit
that never engages the rayon pool at all, so the native run occupies a single
core and leaves the others free for the counter thread — the only configuration
in which a CPU-bound Python thread can be *observed* to make progress, and thus
prove the GIL was released, even on a 2-core runner.

``polypus.statevector`` (issue #86) gets the same GIL-release proof at the end
of this module, plus two SIGINT tests. Unlike ``run_quantum_circuit``, it *is*
interruptible mid-run (issue #110): ``polypus-sim``'s gate loop polls an
injected, Python-agnostic cancellation hook, and ``statevector`` supplies one
that reacquires the GIL and calls ``py.check_signals()``, so a pending Ctrl+C
stops the simulation part-way through the gate sequence instead of waiting for
it to finish. This matters because the qubit ceiling
(``polypus_sim::MAX_QUBITS``, covered in ``test_statevector.py``) bounds a run's
*memory*, not its wall-clock time: cost scales with gates × ``2^n`` and nothing
bounds the gate count. The two tests split along that line —
``test_statevector_responds_to_sigint_mid_simulation`` proves the gate loop
itself is cut short (by comparing against a completed run of the same circuit),
while ``test_statevector_responds_to_sigint_promptly`` covers the remaining
post-run boundary, which is all a run dominated by ``Statevector::new``'s
``2^n`` allocation can offer. See ``docs/ENGINEERING.md`` §3.

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
# window (well under a second) could let complete on iteration count alone —
# but this landscape is degenerate enough (all-zero `x_train`, a 4-parameter
# ansatz) that DE's default `patience=20` early-stops it in ~40 generations
# (measured), well under `_DELAY_BEFORE_SIGINT_S`. `patience` is set far above
# `generations` below so only the interrupt (never convergence) can end the run.
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
        polypus.DE(
            generations=100000, population_size=6, tolerance=1e-12, patience=200000
        ),
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


def _assert_responds_to_sigint_promptly(
    child_code, *, failure_hint, env=None, delay=None, deadline=None
):
    """Run `child_code` in a subprocess, SIGINT it mid-run, and assert a real
    `KeyboardInterrupt` was raised well before a completed run could finish.
    Shared by the training, run_quantum_circuit and statevector variants below
    — they differ only in which entry point/backend the child script exercises
    (and, for run_quantum_circuit, an `env` that pins the sim single-threaded).
    `delay`/`deadline` default to the module-wide `_DELAY_BEFORE_SIGINT_S` /
    `_INTERRUPT_DEADLINE_S`; the statevector variants override `delay` because
    their whole run lasts a few seconds (rather than the others' generation/shot
    budgets), so the SIGINT has to arrive sooner — see their tests for why.

    Returns `(elapsed, ready_fields)`: the interrupt latency the child measured
    itself, and whatever it printed after `READY` on its first line (the mid-run
    statevector test uses that channel to carry a completed-run baseline it
    measured in the same process, on the same machine)."""
    delay = _DELAY_BEFORE_SIGINT_S if delay is None else delay
    deadline = _INTERRUPT_DEADLINE_S if deadline is None else deadline
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
        ready_fields = first_line.split()
        assert ready_fields[:1] == ["READY"], (
            f"unexpected child startup output {first_line!r}; "
            f"stderr:\n{proc.stderr.read()}"
        )

        time.sleep(delay)
        proc.send_signal(signal.SIGINT)

        try:
            out, err = proc.communicate(timeout=deadline)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            pytest.fail(
                f"{failure_hint} did not respond to SIGINT within "
                f"{deadline}s.\nstdout:\n{out}\nstderr:\n{err}"
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
    assert elapsed < deadline, (
        f"interrupt was honored but not promptly: {elapsed:.2f}s "
        f"(a full run would take far longer)"
    )
    return elapsed, ready_fields[1:]


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
# 1 -> AlgorithmSingleRun, >1 -> DistributeByShotsRun. The latter's
# single-evolution fast path evolves the shared circuit once (not once per QPU),
# so both variants use the same `reps` to keep the total run time comparable.
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
    # n_qpus > 1 -> DistributeByShotsRun. Its single-evolution fast path evolves
    # the shared circuit once (not once per QPU), so this uses the same
    # n=18/reps=100 as the single-run variant (~1.3s single-threaded) rather than
    # a per-QPU-scaled reps — otherwise the run finishes inside the SIGINT window.
    _assert_responds_to_sigint_promptly(
        _run_child(n_qpus=4, n=18, reps=100),
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
# the call's Python entry/exit stages allow (measured ~1.4e5); with the GIL
# released it keeps advancing at close to its free-running rate. Rather than a
# brittle absolute count, the child calibrates that free-running rate (via a
# time.sleep window, which releases the GIL) and reports the *ratio* of counts
# achieved during the native run to counts expected at the free rate over the
# same wall-clock. GIL released -> ratio near 1.0 (>=0.5 even under single-core
# contention with the native thread); GIL held -> ratio ~0. The threshold sits
# far below the released value and far above the held value, independent of the
# runner's CPU speed or the native run's duration.
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
_MIN_GIL_RELEASE_RATIO = 0.2

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

qc = make(11, 1500)  # sub-parallel-threshold native run (~1.3-2.6s across 32..2 cores)
counter = 0
stop = threading.Event()

def spin():
    global counter
    while not stop.is_set():
        counter += 1

worker = threading.Thread(target=spin)
worker.start()
try:
    import time

    # Calibrate the worker's free-running spin rate: time.sleep releases the
    # GIL, so the worker runs unobstructed during this window. Comparing the
    # native run against this per-machine rate makes the test robust to CPU
    # speed and native-run duration -- an absolute count threshold was not,
    # and caused spurious CI failures on fast/loaded runners.
    c0 = counter
    time.sleep(0.2)
    free_rate = (counter - c0) / 0.2  # counts/sec with the GIL free

    # n_qpus=20 -> 20 sequential single-thread circuits (DistributeByShotsRun),
    # enough wall-clock for the counter to accumulate a decisive lead.
    before = counter
    t0 = time.perf_counter()
    polypus.run_quantum_circuit(
        qc, shots=4096, infrastructure="local", n_qpus=20, backend="polypus"
    )
    dt = time.perf_counter() - t0
    advanced = counter - before
finally:
    stop.set()
    worker.join()

# Fraction of its free-running rate the worker sustained *during* the native
# run. GIL released -> worker keeps running (ratio near 1.0, or ~0.5 under
# single-core contention with the native thread); GIL held -> worker starved
# (ratio ~0).
expected = free_rate * dt
ratio = (advanced / expected) if expected > 0 else 0.0
print(f"ADVANCED {advanced} RATE {free_rate:.0f} DT {dt:.4f} RATIO {ratio:.4f}", flush=True)
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
    fields = proc.stdout.split("ADVANCED", 1)[1].split()
    advanced = int(fields[0])
    ratio = float(fields[fields.index("RATIO") + 1])
    assert ratio > _MIN_GIL_RELEASE_RATIO, (
        f"background thread sustained only {ratio:.3f} of its free-running rate "
        f"during the native run (advanced {advanced} counts) — the GIL was "
        f"likely not released around run_quantum_circuit\nstdout:\n{proc.stdout}"
    )


# --------------------------------------------------------------------------- #
# statevector (issue #86)
# --------------------------------------------------------------------------- #

# `polypus.statevector` releases the GIL around the simulation only; parameter
# binding (O(gates)) and the amplitude conversion stay on the GIL side. Proven
# with the same calibrated-ratio design as the `run_quantum_circuit` child above.
#
# Sizing differs, though. `statevector` has no `shots`/`n_qpus` knob to buy
# wall-clock with, and circuit *construction* cost grows superlinearly with the
# gate count while simulation cost grows linearly, so a single
# sub-`parallel_threshold` circuit long enough to measure takes far longer to
# build than to simulate (42k gates: ~0.8s to build, ~0.09s to run). The child
# therefore simulates one 11-qubit circuit `_SV_REPEATS` times in sequence — the
# same "many sequential single-thread runs" trick the run_quantum_circuit child
# gets from `n_qpus=20` — for ~1.4s of native work in total, with the qubit count
# below polypus-sim's `parallel_threshold` (12) so the run never engages the
# rayon pool and leaves the other cores free for the counter thread (see the note
# above).
#
# Measured on the reference machine: ratio 0.97 with the GIL released, 0.05 with
# `allow_threads` reverted (the residual comes from the interpreter's switch
# interval at each of the `_SV_REPEATS` call boundaries, so it shrinks as the
# per-call simulation grows — hence few, long calls rather than many short ones).
# `_MIN_GIL_RELEASE_RATIO` (0.2) sits ~4x above the regressed value.
_SV_REPEATS = 15

_SV_GIL_RELEASE_CHILD = (
    r"""
import threading
import time
import polypus
"""
    + _MAKE_CIRCUIT_SRC
    + r"""
# Warm up the import/build paths outside the measured window.
polypus.statevector(make(2, 1))

qc = make(11, 2000)  # sub-parallel-threshold; ~0.09s per simulation
counter = 0
stop = threading.Event()

def spin():
    global counter
    while not stop.is_set():
        counter += 1

worker = threading.Thread(target=spin)
worker.start()
try:
    # Calibrate the worker's free-running spin rate (time.sleep releases the
    # GIL), then compare against it — robust to CPU speed, as above.
    c0 = counter
    time.sleep(0.2)
    free_rate = (counter - c0) / 0.2

    before = counter
    t0 = time.perf_counter()
    for _ in range(__REPEATS__):
        polypus.statevector(qc)
    dt = time.perf_counter() - t0
    advanced = counter - before
finally:
    stop.set()
    worker.join()

expected = free_rate * dt
ratio = (advanced / expected) if expected > 0 else 0.0
print(f"ADVANCED {advanced} RATE {free_rate:.0f} DT {dt:.4f} RATIO {ratio:.4f}", flush=True)
""".replace("__REPEATS__", str(_SV_REPEATS))
)


def test_statevector_releases_gil_for_other_threads():
    proc = subprocess.run(
        [sys.executable, "-c", _SV_GIL_RELEASE_CHILD],
        capture_output=True,
        text=True,
        timeout=_READY_TIMEOUT_S,
    )
    assert proc.returncode == 0 and "ADVANCED" in proc.stdout, (
        f"GIL-release child did not complete cleanly.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    fields = proc.stdout.split("ADVANCED", 1)[1].split()
    advanced = int(fields[0])
    ratio = float(fields[fields.index("RATIO") + 1])
    assert ratio > _MIN_GIL_RELEASE_RATIO, (
        f"background thread sustained only {ratio:.3f} of its free-running rate "
        f"during {_SV_REPEATS} native simulations (advanced {advanced} counts) — "
        f"the GIL was likely not released around statevector\n"
        f"stdout:\n{proc.stdout}"
    )


# `statevector` also calls `py.check_signals()` the moment the GIL is reacquired,
# before handing the amplitudes to NumPy (see docs/ENGINEERING.md §3/§4), so a
# Ctrl+C that arrived while the GIL-free simulation was running surfaces as a
# `KeyboardInterrupt` out of the call itself — never swallowed into a
# `PanicException`, and never ignored so the call reports a completed run.
#
# This is the *post-run* boundary, and since #110 it is no longer the only one:
# the gate loop is polled mid-run too, which
# `test_statevector_responds_to_sigint_mid_simulation` below proves. The two
# tests are complementary rather than redundant, and this one's sizing is exactly
# why: at 27 qubits the run's cost is `Statevector::new`'s `2^n` allocation — a
# single `vec![]` with nowhere to put a checkpoint — so this child has *only* the
# post-run boundary to be caught by, and keeps guarding it.
#
# What this test can and cannot pin down. It used to size the child so that the
# *list* conversion dominated (26 qubits: ~3.8s completed, interrupt honored at
# ~1.5s with the check in place and ~3.3s without it) and put the deadline
# between those two numbers, which made the check's presence directly observable.
# That gap is gone now that the amplitudes cross the seam as a NumPy array: the
# Rust buffer is *moved* into the array rather than converted element by element,
# so there is no expensive post-check phase left. Re-measured on the reference
# machine (26q, full Hadamard layer): completed 1.52s, interrupt latency 1.52s —
# and, with `check_signals()` temporarily removed and the extension rebuilt,
# 1.52s again (CPython delivers the pending SIGINT at the very next bytecode
# after the call returns, which is indistinguishable in wall-clock). No timing
# threshold can separate the two cases any more, and neither can observing
# whether the statement after the call ran (checked: it does not, either way).
#
# So this now guards the same property as the `run_quantum_circuit` children
# above — a pending SIGINT is honored as a real `KeyboardInterrupt` when the run
# reaches the check boundary, i.e. when it completes — and the deadline is a hang
# guard sized above the full run, not a discriminator. The `check_signals()` call
# stays because it is the documented boundary and keeps the interrupt ahead of
# whatever result-building work may be added later; that part is enforced by
# review and by ENGINEERING.md §3, no longer by a wall-clock assertion. The
# discriminating-timing role this test lost is now carried by the mid-run test
# below, which recovers it by comparing against a completed run of its own
# circuit instead of against a fixed deadline.
#
# Sizing: 27 qubits with a *single* gate. The finding that motivated dropping the
# Hadamard layer is that near the ceiling the run's cost is `Statevector::new`'s
# `2^n` allocation, not the gates — so one gate is enough to make the call last
# ~0.85s (measured; 1.07s with RAYON_NUM_THREADS=1, i.e. essentially
# core-count independent, since first-touch of the 2 GiB buffer is single-threaded
# either way). That is comfortably longer than the 0.1s `delay`, so the SIGINT
# lands while the simulation is genuinely in flight, and ~5x under the module's
# `_INTERRUPT_DEADLINE_S` (5.0s) even on a slower runner — which is why, unlike
# the `run_quantum_circuit` children, this one needs neither a bespoke deadline
# nor `_SINGLE_THREADED_ENV`. A full Hadamard layer instead swings the run from
# 1.5s (32 cores) to 3.9s (single-threaded), for no added signal.
_SV_LARGE_N = 27
_SV_DELAY_BEFORE_SIGINT_S = 0.1

_SV_INTERRUPT_CHILD = r"""
import sys, time
import polypus

# Warm up the lazy import path outside the timed window.
polypus.statevector(polypus.Circuit(2).h(0))

qc = polypus.Circuit(__N__).h(0)
print("READY", flush=True)
start = time.time()
try:
    polypus.statevector(qc)
    print("COMPLETED", flush=True)
except KeyboardInterrupt:
    print(f"KEYBOARDINTERRUPT {time.time() - start:.3f}", flush=True)
except BaseException as exc:  # e.g. a PanicException from a swallowed signal
    print(f"OTHER {type(exc).__name__}", flush=True)
    sys.exit(1)
""".replace("__N__", str(_SV_LARGE_N))


def test_statevector_responds_to_sigint_promptly():
    _assert_responds_to_sigint_promptly(
        _SV_INTERRUPT_CHILD,
        failure_hint=(
            "statevector — the SIGINT was swallowed instead of surfacing as a "
            "KeyboardInterrupt (py.check_signals() after the GIL is reacquired "
            "is likely missing, or the GIL is never released)"
        ),
        delay=_SV_DELAY_BEFORE_SIGINT_S,
    )


# Mid-run interruption (issue #110): the SIGINT must cut the *gate loop* short,
# not merely be honored once the simulation finishes anyway. Since #110,
# `polypus-sim`'s loop polls a cancellation hook that `statevector` fills with a
# GIL-reacquiring `py.check_signals()`, and a cancelled run raises the pending
# `KeyboardInterrupt` verbatim (never the `ValueError` that mapping the new
# `SimError::Cancelled` through `PyValueError` would have produced, and never a
# `PanicException`).
#
# How this discriminates, where `test_statevector_responds_to_sigint_promptly`
# above no longer can: the child times a *completed* run of the same circuit
# before printing READY (which doubles as the warm-up), and the parent asserts
# the interrupted run came in under `_SV_MID_RUN_MAX_FRACTION` of it. Both
# numbers come from the same process on the same machine, so the comparison
# survives any CI speed — a slow runner scales both — where an absolute deadline
# would only prove the call returned before a hang guard. Without a mid-run
# checkpoint the ratio is ~1.0 (the interrupt lands when the run ends); with one
# it is the SIGINT delay plus one ~25ms checkpoint over the full run's duration
# (measured on the reference machine: 0.31s against 2.75s completed, ratio 0.11,
# i.e. ~4.5x inside the 0.5 threshold).
#
# Sizing, the inverse of the test above: 20 qubits (`2^20` amplitudes = 16 MiB,
# so `Statevector::new`'s allocation is noise) with 1580 gates, which puts *all*
# the ~2.8s single-threaded run time in the gate loop — the one place a hook can
# live. `_SINGLE_THREADED_ENV` keeps that duration core-count independent, as for
# the run_quantum_circuit children; 20 qubits is above polypus-sim's
# `parallel_threshold`, so without it the run time would vary with the runner's
# core count. Building the circuit is free by comparison (<1ms measured), which
# is what makes the elapsed comparison a statement about simulation alone.
_SV_MID_RUN_N = 20
_SV_MID_RUN_REPS = 40
_SV_MID_RUN_DELAY_BEFORE_SIGINT_S = 0.3
_SV_MID_RUN_MAX_FRACTION = 0.5

_SV_MID_RUN_CHILD = (
    r"""
import sys, time
import polypus
"""
    + _MAKE_CIRCUIT_SRC
    + r"""
qc = make(__N__, __REPS__)

# Time a full run of this very circuit: it is both the warm-up (paying the lazy
# import cost outside the timed window) and the baseline the parent compares the
# interrupted run against.
t0 = time.time()
polypus.statevector(qc)
completed = time.time() - t0

print(f"READY {completed:.3f}", flush=True)
start = time.time()
try:
    polypus.statevector(qc)
    print("COMPLETED", flush=True)
except KeyboardInterrupt:
    print(f"KEYBOARDINTERRUPT {time.time() - start:.3f}", flush=True)
except BaseException as exc:  # e.g. a ValueError from a downgraded cancellation
    print(f"OTHER {type(exc).__name__}", flush=True)
    sys.exit(1)
""".replace("__N__", str(_SV_MID_RUN_N)).replace("__REPS__", str(_SV_MID_RUN_REPS))
)


def test_statevector_responds_to_sigint_mid_simulation():
    elapsed, ready_fields = _assert_responds_to_sigint_promptly(
        _SV_MID_RUN_CHILD,
        failure_hint=(
            "statevector — the simulation was not interrupted mid-run (the "
            "cancellation hook passed to polypus-sim's gate loop is likely "
            "missing, never polled, or its check_signals() never reacquires "
            "the GIL)"
        ),
        env=_SINGLE_THREADED_ENV,
        delay=_SV_MID_RUN_DELAY_BEFORE_SIGINT_S,
    )
    completed = float(ready_fields[0])
    assert elapsed < completed * _SV_MID_RUN_MAX_FRACTION, (
        f"the KeyboardInterrupt was raised after {elapsed:.2f}s, but a full run "
        f"of the same circuit takes {completed:.2f}s — the interrupt was honored "
        f"when the simulation *finished*, not mid-run: the gate loop's "
        f"cancellation hook is not stopping it"
    )
