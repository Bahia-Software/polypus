"""Shared pytest fixtures for polypus tests."""

import pytest
from qiskit.circuit import QuantumCircuit, ParameterVector


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
