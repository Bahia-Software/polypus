"""
Bounded per-candidate concurrency in ``QmlOracle`` (issue #85).

``QmlOracle::try_evaluate`` used to hand Tokio's blocking pool **one
``spawn_blocking`` task per candidate**, so a DE/PSO ``population_size`` in the
hundreds created hundreds of simultaneously-blocked OS threads — for no gain,
since the GIL serialises the actual Qiskit/Aer work. It now dispatches
candidates in chunks of at most ``2 * available_parallelism()`` (see the
``QmlOracle`` doc comment in
``crates/polypus/src/evaluation/qml_oracle.rs``), so the number of candidates in
flight no longer scales with ``population_size``.

This is the Python half of the proof; the chunked-dispatch mechanism itself has
Rust unit tests next to it (``dispatch_bounded`` in ``qml_oracle.rs``). Here the
real ``polypus.qml.train`` runs with a population in the hundreds while the C-1
``polypus_python.run_qcs`` seam is monkeypatched — the same technique
``test_seam_contract.py`` uses — into a probe that counts how many candidates
are inside the oracle's per-candidate work simultaneously.

Why that seam is the observation point:

* It is reached exactly once per candidate evaluation here (one training row,
  so one backend call per candidate), from the worker thread that owns that
  candidate — so concurrent calls mean concurrent candidates.
* Its ``time.sleep`` releases the GIL, so genuinely concurrent workers overlap
  and are *observable*; without a GIL-releasing dwell the count would only
  reflect GIL serialisation, not the dispatch bound.
* It keeps the test fast and deterministic: no Aer simulation, no shot noise.

Before the fix the observed peak reached the full population; it must now stay
at or below the derived bound.
"""

import os
import threading
import time

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]

# A population "in the hundreds", per the issue. Deliberately not a multiple of
# any plausible chunk size, so a dispatch bug in the final partial chunk shows up
# in the evaluation count asserted below.
_POPULATION = 200

# How long each faked backend call dwells with the GIL released. Long enough for
# a whole chunk to overlap (thread hand-off is microseconds), short enough that
# the whole run stays a couple of seconds.
_DWELL_S = 0.003

# The bound the oracle derives: CONCURRENCY_MULTIPLIER (2) x available
# parallelism. `os.cpu_count()` is an upper bound for Rust's
# `available_parallelism()`, which additionally respects cgroup quotas and
# affinity masks — so this is a ceiling on the real limit, never below it.
_MULTIPLIER = 2
_EXPECTED_BOUND = _MULTIPLIER * (os.cpu_count() or 1)


class _ConcurrencyProbe:
    """Records the peak number of simultaneous backend calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak = 0
        self.calls = 0

    def __enter__(self):
        with self._lock:
            self._in_flight += 1
            self.calls += 1
            self.peak = max(self.peak, self._in_flight)
        return self

    def __exit__(self, *_exc):
        with self._lock:
            self._in_flight -= 1
        return False


def _probing_seam(probe):
    """A ``polypus_python.run_qcs`` stand-in that records concurrency.

    Returns one counts dict per submitted circuit, conserving ``shots`` and
    using a bitstring of the circuit's own classical width (contract C-3), so the
    Rust side and the user ``expectation_function`` see a well-formed payload.
    """

    def run_qcs(_infrastructure, **kwargs):
        qcs = kwargs["qcs"]
        shots = kwargs["shots"]
        with probe:
            time.sleep(_DWELL_S)
            return [{"0" * (qc.num_clbits or qc.num_qubits): shots} for qc in qcs]

    return run_qcs


def test_qml_train_bounds_candidates_in_flight(monkeypatch):
    import numpy as np
    import polypus
    import polypus_python
    from qiskit.circuit.library import real_amplitudes, zz_feature_map

    if _EXPECTED_BOUND >= _POPULATION:
        pytest.skip(
            f"{os.cpu_count()} cores put the derived bound at {_EXPECTED_BOUND}, "
            f"which a population of {_POPULATION} cannot discriminate against"
        )

    feature_map = zz_feature_map(feature_dimension=2, reps=1)
    ansatz = real_amplitudes(num_qubits=2, reps=1)
    # A single training row keeps the seam call count equal to the candidate
    # count: one candidate evaluation == one backend call.
    x_train = np.zeros((1, 2))

    probe = _ConcurrencyProbe()
    monkeypatch.setattr(polypus_python, "run_qcs", _probing_seam(probe))

    result = polypus.qml.train(
        feature_map,
        ansatz,
        x_train,
        polypus.DE(generations=1, population_size=_POPULATION, tolerance=1e-12),
        shots=64,
        n_qpus=1,
        dimensions=len(ansatz.parameters),
        expectation_function=lambda b: sum(int(c) for c in b) / len(b),
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="qml_concurrency",
        seed=11,
    )

    # The run really happened (and returned the C-7 manifest, unchanged).
    assert len(result.best_params) == len(ansatz.parameters)
    assert probe.calls >= _POPULATION, (
        f"only {probe.calls} candidate evaluations reached the backend for a "
        f"population of {_POPULATION} — candidates were dropped"
    )
    # Every batch evaluates the whole population exactly once, so the total is a
    # multiple of it. `_POPULATION` is not a multiple of any plausible chunk
    # size, so this also catches a partial final chunk being skipped or
    # dispatched twice.
    assert probe.calls % _POPULATION == 0, (
        f"{probe.calls} candidate evaluations is not a whole number of "
        f"populations of {_POPULATION} — chunked dispatch dropped or duplicated "
        f"candidates"
    )

    assert probe.peak <= _EXPECTED_BOUND, (
        f"{probe.peak} candidates were in flight at once; the bound for "
        f"{os.cpu_count()} cores is {_EXPECTED_BOUND}. A population of "
        f"{_POPULATION} must not translate into per-candidate concurrency"
    )
    # Guard against the test "passing" because dispatch became fully sequential:
    # the fix bounds concurrency, it does not remove it.
    assert probe.peak > 1, (
        "no two candidates were ever in flight together — concurrency was lost, "
        "not bounded"
    )
