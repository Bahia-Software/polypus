"""
``QmlOracle`` bounds its in-flight circuit batch to a window of candidates.

``QmlOracle::try_evaluate`` used to hand Tokio's blocking pool **one
``spawn_blocking`` task per candidate**, so a DE/PSO ``population_size`` in the
hundreds created hundreds of simultaneously-blocked OS threads — for no gain,
since the GIL serialises the actual Qiskit/Aer work (issue #85). That was replaced
by binding every ``candidate × training-circuit`` pair into one flat batch handed
to the ``Planner`` in one go.

That flat batch is now itself bounded (issue #146): binding the whole
``candidates × training-circuit`` product up front holds
``population × n_train`` ``BoundCircuit``s live at once (15 000 for a 300×50 run),
and the ``Planner``'s wave cap does not help — ``LocalBackend`` reports
``max_concurrency == usize::MAX``, so it would run the whole batch in one wave.
So the oracle now binds the population in **windows of a few candidates per core**
(``CONCURRENCY_MULTIPLIER × available_parallelism()``), submitting and dropping
each window before building the next. The population therefore deliberately does
**not** reach the backend as one giant call — but it is still *batched*, not the
pre-#85 one-call-per-candidate dispatch.

This is the Python half of the proof. The real ``polypus.qml.train`` runs with a
population in the hundreds while the C-1 ``polypus_python.run_qcs`` seam is
monkeypatched — the same technique ``test_seam_contract.py`` uses — into a probe
that records how many circuits each backend call carries. The result-level proof
(the migration is byte-identical) lives in the determinism gate's ``qml_train``
vector; here we pin the *batching shape*: bounded windows, not one giant batch and
not per-candidate.
"""

import os
import threading

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]

# A population "in the hundreds", per the original issue. Deliberately not a
# multiple of any plausible chunk size, so a batching bug (a dropped or duplicated
# partial batch) shows up in the circuit counts asserted below.
_POPULATION = 200


class _BatchProbe:
    """Records the circuits carried by each backend call."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.total_circuits = 0
        self.max_circuits_in_a_call = 0

    def record(self, n):
        with self._lock:
            self.calls += 1
            self.total_circuits += n
            self.max_circuits_in_a_call = max(self.max_circuits_in_a_call, n)


def _probing_seam(probe):
    """A ``polypus_python.run_qcs`` stand-in that records the batch shape.

    Returns one counts dict per submitted circuit, conserving ``shots`` and using
    a bitstring of the circuit's own classical width (contract C-3), so the Rust
    side and the user ``expectation_function`` see a well-formed payload.
    """

    def run_qcs(_infrastructure, **kwargs):
        qcs = kwargs["qcs"]
        shots = kwargs["shots"]
        probe.record(len(qcs))
        return [{"0" * (qc.num_clbits or qc.num_qubits): shots} for qc in qcs]

    return run_qcs


def _candidate_window():
    """The oracle's in-flight window, in candidates.

    Mirrors ``candidate_window_size`` in ``crates/polypus-evaluation/src/qml_oracle.rs``
    (``CONCURRENCY_MULTIPLIER × available_parallelism()``). Uses the CPU-affinity
    count on Linux — what Rust's ``std::thread::available_parallelism`` reports —
    falling back to the CPU count elsewhere, so the expected window is derived from
    the runner, never hardcoded.
    """
    try:
        parallelism = len(os.sched_getaffinity(0))
    except AttributeError:  # sched_getaffinity is Linux-only
        parallelism = os.cpu_count() or 1
    _concurrency_multiplier = 2
    return parallelism * _concurrency_multiplier


def test_qml_train_bounds_the_population_into_windows(monkeypatch):
    import numpy as np
    import polypus
    import polypus_python
    from qiskit.circuit.library import real_amplitudes, zz_feature_map

    feature_map = zz_feature_map(feature_dimension=2, reps=1)
    ansatz = real_amplitudes(num_qubits=2, reps=1)
    # A single training row keeps one circuit per candidate, so a call carrying
    # the whole population is exactly `_POPULATION` circuits.
    x_train = np.zeros((1, 2))

    probe = _BatchProbe()
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
    assert probe.calls >= 1, "the training run never reached the backend"

    # One training row means one circuit per candidate, so a window of candidates
    # reaches the backend as that many circuits. The largest call is one full window
    # (or the whole population, on a machine whose window already covers it) — never
    # the eager whole-population batch this fix replaced.
    expected_max = min(_candidate_window(), _POPULATION)
    assert probe.max_circuits_in_a_call == expected_max, (
        f"the largest backend call carried {probe.max_circuits_in_a_call} circuits; "
        f"the oracle must bound each call to one window of {expected_max}"
    )
    # …and that is genuinely batched, not the pre-#85 one-circuit-per-call dispatch.
    assert probe.max_circuits_in_a_call > 1, (
        "each backend call carried a single circuit — that is per-candidate dispatch"
    )
    # Correctness: every candidate's circuit reaches the backend, and each
    # evaluation submits exactly one population — no drops, no duplication. (DE
    # runs a few evaluation phases: the initial population and one per generation.)
    assert probe.calls < _POPULATION, (
        f"the backend was called {probe.calls} times for a population of "
        f"{_POPULATION} — that is per-candidate dispatch, not batched"
    )
    assert probe.total_circuits % _POPULATION == 0, (
        f"{probe.total_circuits} circuits is not a whole number of populations of "
        f"{_POPULATION} — a batch was dropped or duplicated"
    )
