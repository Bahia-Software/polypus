"""
QML evaluation-path error classification (issue #81, Change 2).

``EvaluationError::Runtime`` (Tokio runtime construction, or a ``spawn_blocking``
worker panic surfaced as a ``JoinError``) is a Rust-side infrastructure failure
and now surfaces as ``polypus.EvaluationError`` rather than a bare
``RuntimeError`` — pinned by the Rust unit test in
``crates/polypus/src/evaluation/error.rs`` (forcing those OS-level conditions
deterministically from Python is neither viable nor portable; see the PR).

This test guards the *other* side of that change: a genuine Python exception
raised by the user's ``expectation_function`` callback must still propagate
**verbatim** (as itself), never be reclassified as ``polypus.EvaluationError``.
It runs on the ``local`` path with a mocked backend, so no SLURM/hardware.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]


def _patch_deterministic_backend(monkeypatch):
    import polypus_python

    def fake_run_qcs(infrastructure, **kwargs):
        return [{"1": kwargs["shots"]} for _ in kwargs["qcs"]]

    monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)


def test_qml_callback_exception_propagates_verbatim(monkeypatch):
    import numpy as np
    import polypus
    from qiskit.circuit.library import real_amplitudes, zz_feature_map

    _patch_deterministic_backend(monkeypatch)

    def exploding_expectation(_bitstring):
        raise ValueError("user callback blew up")

    feature_map = zz_feature_map(feature_dimension=2, reps=1)
    ansatz = real_amplitudes(num_qubits=2, reps=1)
    x_train = np.zeros((2, 2))

    # The user's callback raises ValueError; contract C-1 / ENGINEERING.md §9
    # require it to reach the caller as that same ValueError, not wrapped in
    # polypus.EvaluationError.
    with pytest.raises(ValueError, match="user callback blew up") as excinfo:
        polypus.qml.train(
            feature_map,
            ansatz,
            x_train,
            polypus.DE(generations=2, population_size=4, tolerance=0.5),
            shots=64,
            n_qpus=1,
            dimensions=len(ansatz.parameters),
            expectation_function=exploding_expectation,
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id="qml_verbatim_error",
        )
    assert not isinstance(excinfo.value, polypus.EvaluationError), (
        "a genuine Python callback exception must not be reclassified as "
        "polypus.EvaluationError"
    )
