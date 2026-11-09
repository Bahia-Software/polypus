"""Shared pytest fixtures for polypus tests."""

import os

# Auto-calibration (issue #176) runs once per process, on the first native
# statevector run (#351). In the test-runner process that would measure and write
# the *real* user cache from whichever test runs first, making the suite
# non-hermetic and order-dependent. Disable it here so the parent process is
# deterministic; the auto-calibration behaviour itself is exercised in isolated
# child interpreters by ``test_autocalibrate.py`` (which set the env explicitly
# per case).
os.environ.setdefault("POLYPUS_NO_AUTOCALIBRATE", "1")

import pytest  # noqa: E402
from qiskit.circuit import ParameterVector, QuantumCircuit  # noqa: E402

# POLYPUS_ALLOWED_SKIPS: a skip not on the list fails the run (CI sets it).
from skip_guard import (  # noqa: E402, F401
    pytest_collectreport,
    pytest_runtest_logreport,
    pytest_sessionfinish,
    pytest_sessionstart,
    pytest_terminal_summary,
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cunqa: needs a real CUNQA install and SLURM (opt-in: POLYPUS_CUNQA_REAL=1)",
    )


@pytest.fixture
def bell_circuit() -> QuantumCircuit:
    """A simple 2-qubit Bell circuit with measurements."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture
def parametrized_circuit() -> QuantumCircuit:
    """A 1-qubit RY circuit with a single trainable parameter."""
    theta = ParameterVector("θ", 1)
    qc = QuantumCircuit(1)
    qc.ry(theta[0], 0)
    qc.measure_all()
    return qc


@pytest.fixture
def simple_expectation_fn():
    """
    Minimal expectation function: fn(bitstring: str) -> float.
    Passed to polypus.train as the per-bitstring objective. Returns 1.0 for
    the all-ones state, 0.0 otherwise — drives optimisers toward θ = π.
    """

    def _fn(bitstring: str) -> float:
        return float(all(b == "1" for b in bitstring))

    return _fn


@pytest.fixture
def simple_variance_fn():
    """
    Minimal variance function for QNG: fn(theta, a) -> float.
    Returns 0.5 (constant) — sufficient to verify QNG plumbing without
    requiring a real quantum Fisher information matrix computation.
    """

    def _fn(theta: list, a: int) -> float:
        return 0.5

    return _fn
