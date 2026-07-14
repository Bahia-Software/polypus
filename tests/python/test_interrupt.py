"""
Ctrl+C responsiveness during training (issue #36).

Acceptance criterion: a ``KeyboardInterrupt`` takes effect *promptly* while
``polypus.train`` (native backend) or ``polypus.qml.train`` (Qiskit/Aer
backend) is running, rather than being ignored until the whole optimization
finishes.

Why both entry points: `train`'s `VqcOracle` and `qml.train`'s `QmlOracle`
share `run_and_expect`, but reach it through different paths — a native,
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
