"""
Ctrl+C responsiveness during training (issue #36).

Acceptance criterion: a ``KeyboardInterrupt`` takes effect *promptly* while
``polypus.train`` (native backend) or ``polypus.qml.train`` (Qiskit/Aer
backend) is running, rather than being ignored until the whole optimization
finishes.

Why both entry points: `train`'s `VqcOracle` and `qml.train`'s `QmlOracle`
share `run_and_evaluate`, but reach it through different paths — a native,
GIL-free simulation loop for `train`, and Tokio `spawn_blocking` workers
(`allow_threads` around a `block_on`, needed to avoid the deadlock documented
in `qml_oracle.rs`) for `qml.train`. Only the calling (main) thread can have a
pending SIGINT turned into `KeyboardInterrupt` (`PyErr_CheckSignals` is a
no-op off the main thread), so `qml.train` additionally checks signals once on
the main thread after its workers join. Both paths need their own proof.

A third test covers the native minibatched path (`polypus.qml.train` with a
`Model` + `Dataset` and `batch_size=`), whose end-of-run **full-dataset fitness
recompute** (design doc §17) is the one heavy step the optimizer loop above does
not cover. That recompute is a single, indivisible native call: a
`check_signals` *after* it cannot preempt it mid-flight, so — unlike the loop
tests above — it is not measurable by SIGINT latency (a fixed and an unfixed
build both finish the whole recompute before honouring the signal, at the same
moment). What the fix actually guarantees, and what that test asserts, is the
other half of ENGINEERING §3: the recompute runs **GIL-free**
(`py.allow_threads`). A background Python thread ticking every 50 ms stays live
throughout a multi-second recompute when the GIL is released, but is frozen for
the whole recompute when it is held — the pre-fix behaviour. That liveness is
the observable, backend-independent signature of the GIL being released.

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

# Sizing for the minibatch-recompute GIL-freedom child (last test). The child
# runs to completion (no SIGINT), so these only bound the heartbeat's freeze
# detection, not an interrupt window.
_HEARTBEAT_INTERVAL_S = 0.05  # background-thread tick period
_MAX_LIVE_GAP_S = 1.0  # a live thread's max gap is ~_HEARTBEAT_INTERVAL_S; a
#                        frozen one's is ~the whole recompute (seconds) — 1.0s
#                        sits far above the former and far below the latter.
_MIN_RECOMPUTE_S = 1.0  # the recompute must be long enough that a freeze is
#                         unmistakable; the child self-fails (resize hint) below.
_RECOMPUTE_CHILD_TIMEOUT_S = 120.0  # import + warm-up + a multi-second recompute,
#                                     generous for a cold/slow CI runner.

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

# Native minibatched path (`Model` + `Dataset` + `batch_size`), the only path on
# which `dispatch_optimizer` runs a final full-dataset recompute (design doc
# §17). This child measures GIL-freedom of that recompute, not SIGINT latency
# (see the module docstring for why the recompute — a single native call — is not
# SIGINT-testable): a background thread times itself while the main thread runs
# the recompute; if the GIL is held the thread freezes for the whole recompute.
#
# Sizing note: the minibatch loop is deliberately trivial — `batch_size=1`,
# `generations=1`, `population_size=8`, so ~16 one-sample circuit evaluations,
# milliseconds total — while a single full-dataset recompute scores all
# `_SAMPLES` circuits at `_QUBITS` qubits. 12 qubits + 1400 samples puts that one
# recompute in the multi-second range (≈6s on the reference dev box), so a frozen
# heartbeat's gap (≈the recompute) and a live heartbeat's gap (≈50 ms) differ by
# ~two orders of magnitude — `_MAX_LIVE_GAP_S = 1.0` separates them with a wide
# margin either way. The child self-fails with a resize hint if the recompute
# ever falls below `_MIN_RECOMPUTE_S` (e.g. on far faster hardware).
_QML_MINIBATCH_RECOMPUTE_CHILD = f"""
import sys, time, threading
import polypus

_QUBITS = 12
_SAMPLES = 1400
_HEARTBEAT_INTERVAL_S = {_HEARTBEAT_INTERVAL_S!r}
_MAX_LIVE_GAP_S = {_MAX_LIVE_GAP_S!r}
_MIN_RECOMPUTE_S = {_MIN_RECOMPUTE_S!r}


def _model():
    return (
        polypus.qml.Model(_QUBITS)
        .angle_encoder(axis="ry")
        .hardware_efficient(reps=1)
        .readout(observables=[[("z", 0)]], decision="sign")
    )


def _dataset(n):
    # Distinct feature rows so the encoder does real per-sample work; the labels
    # only need to be valid (the recompute's timing, not its value, is measured).
    x = [[0.3 + 0.0001 * i + 0.01 * j for j in range(_QUBITS)] for i in range(n)]
    y = [1.0 if i % 2 else -1.0 for i in range(n)]
    return polypus.qml.Dataset(x, y)


# Warm up the lazy import/setup path with a tiny run down the *same* native
# minibatch path, so the timed run below reflects only the full-dataset recompute
# and not one-off process/import warm-up.
polypus.qml.train(
    _model(), _dataset(4),
    method=polypus.DE(generations=1, population_size=8, tolerance=1e-12),
    loss="hinge", infrastructure="local", backend="polypus",
    id="warmup", seed=7, exact=True, batch_size=1,
)

# A background Python thread that timestamps itself every _HEARTBEAT_INTERVAL_S.
# It can only run when the main thread is not holding the GIL, so the largest gap
# between its stamps reports the longest stretch the main thread held the GIL.
_stamps = []
_stop = False


def _heartbeat():
    while not _stop:
        _stamps.append(time.time())
        time.sleep(_HEARTBEAT_INTERVAL_S)


_beat = threading.Thread(target=_heartbeat, daemon=True)
_beat.start()
time.sleep(0.2)  # let the heartbeat establish its baseline cadence first

_start = time.time()
polypus.qml.train(
    _model(), _dataset(_SAMPLES),
    method=polypus.DE(generations=1, population_size=8, tolerance=1e-12),
    loss="hinge", infrastructure="local", backend="polypus",
    id="recompute_gil", seed=7, exact=True, batch_size=1,
)
_recompute_dur = time.time() - _start
_stop = True
_beat.join()

_gaps = [_stamps[i + 1] - _stamps[i] for i in range(len(_stamps) - 1)]
_max_gap = max(_gaps) if _gaps else float("inf")

# Sizing sanity: if the recompute was too short, a frozen heartbeat would be
# indistinguishable from a live one — the test would be meaningless. Fail loudly
# with a resize hint rather than pass vacuously.
if _recompute_dur < _MIN_RECOMPUTE_S:
    print(f"UNSIZED recompute_dur={{_recompute_dur:.3f}} max_gap={{_max_gap:.3f}}", flush=True)
    sys.exit(2)

if _max_gap < _MAX_LIVE_GAP_S:
    print(f"PASS recompute_dur={{_recompute_dur:.3f}} max_gap={{_max_gap:.3f}}", flush=True)
    sys.exit(0)
print(f"FAIL recompute_dur={{_recompute_dur:.3f}} max_gap={{_max_gap:.3f}}", flush=True)
sys.exit(1)
"""


def _assert_responds_to_sigint_promptly(child_code, *, failure_hint):
    """Run `child_code` in a subprocess, SIGINT it mid-training, and assert a
    real `KeyboardInterrupt` was raised well before a completed run could
    finish. Shared by the native and QML variants below — they differ only in
    which entry point/backend the child script exercises."""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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


def _assert_recompute_releases_gil(child_code):
    """Run `child_code` (a self-checking GIL-freedom probe) to completion and
    assert it passed. The child freezes a background thread if the final
    full-dataset recompute holds the GIL, so a non-zero exit means the recompute
    ran under the GIL — the pre-fix behaviour this guards against."""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = proc.communicate(timeout=_RECOMPUTE_CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        pytest.fail(
            "the minibatch-recompute child did not finish within "
            f"{_RECOMPUTE_CHILD_TIMEOUT_S}s — a frozen heartbeat cannot explain "
            f"this (it would still finish the recompute), so this is a hang.\n"
            f"stdout:\n{out}\nstderr:\n{err}"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    if "UNSIZED" in out:
        pytest.fail(
            "the full-dataset recompute was too short for this hardware to make "
            "a frozen heartbeat detectable — raise _SAMPLES in the child.\n"
            f"stdout:\n{out}\nstderr:\n{err}"
        )
    assert proc.returncode == 0 and "PASS" in out, (
        "the full-dataset recompute held the GIL: a background Python thread was "
        "frozen for ~the whole recompute (the pre-fix behaviour — the recompute "
        "must run under py.allow_threads, ENGINEERING §3).\n"
        f"stdout:\n{out}\nstderr:\n{err}"
    )


def test_minibatch_recompute_runs_gil_free():
    _assert_recompute_releases_gil(_QML_MINIBATCH_RECOMPUTE_CHILD)
