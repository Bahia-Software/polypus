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

# POLYPUS_TEST_LOG_TO_PYTHON=1 runs the whole suite with the Python log bridge
# installed (issue #458): `polypus.log_to_python()` with the `polypus` logger at
# DEBUG and a handler that formats every record and drops it, so every Rust log
# record of every test goes through Python. It checks that no path deadlocks or
# fails with the bridge on; CI runs it on one leg (`log-bridge` in ci.yml).
# Child interpreters that tests start do not load this file, so they run
# without the bridge.
LOG_TO_PYTHON = os.environ.get("POLYPUS_TEST_LOG_TO_PYTHON") == "1"


def _install_log_to_python():
    import logging

    import polypus

    class FormatAndDrop(logging.Handler):
        def emit(self, record):
            self.format(record)

    handler = FormatAndDrop(logging.NOTSET)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(threadName)s "
            "%(pathname)s:%(lineno)d %(message)s"
        )
    )
    top = logging.getLogger("polypus")
    top.setLevel(logging.DEBUG)
    top.addHandler(handler)
    polypus.log_to_python()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cunqa: needs a real CUNQA install and SLURM (opt-in: POLYPUS_CUNQA_REAL=1)",
    )
    if LOG_TO_PYTHON:
        _install_log_to_python()


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
