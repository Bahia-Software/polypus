"""
Cross-validation of the pure-Rust ``polypus-sim`` statevector backend against
Qiskit's reference simulator (``qiskit.quantum_info.Statevector``).

``polypus.statevector(circuit, params)`` runs the native Rust backend directly
on a ``polypus.Circuit`` (no OpenQASM round-trip). Here we build the *same*
circuit independently with Qiskit gate calls and assert the amplitudes match to
1e-10. The randomized test exercises the whole gate set over many circuits.

Both backends use the little-endian convention (qubit 0 is the least-significant
bit), so the amplitude arrays are compared directly.

The two tests at the end cover the *qubit ceiling* at the seam (issue #86): the
dense statevector backend caps circuits at ``polypus_sim::MAX_QUBITS``
(30 ≈ 16 GiB) and must reject anything larger with a ``ValueError`` before
allocating — the Python-visible half of the guard already tested in Rust
(``crates/polypus-sim/tests/scalability.rs``).
"""

import math
import random
import time

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

PI = math.pi


def _compare(polypus_amps, qc, atol=1e-10):
    """Assert polypus amplitudes equal Qiskit's reference statevector."""
    got = np.asarray(polypus_amps, dtype=complex)
    ref = Statevector(qc).data
    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
    assert np.allclose(got, ref, atol=atol), (
        f"\npolypus = {np.round(got, 4)}\nqiskit  = {np.round(ref, 4)}"
    )


def test_bell_matches_qiskit():
    import polypus

    p = polypus.Circuit(2).h(0).cx(0, 1)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    _compare(polypus.statevector(p), qc)


def test_ghz_matches_qiskit():
    import polypus

    p = polypus.Circuit(3).h(0).cx(0, 1).cx(1, 2)
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    _compare(polypus.statevector(p), qc)


def test_single_qubit_gate_zoo():
    import polypus

    p = (
        polypus.Circuit(1)
        .h(0)
        .t(0)
        .s(0)
        .rx(0, 0.3)
        .ry(0, -1.1)
        .rz(0, 0.7)
        .u(0, 0.5, 1.2, -0.4)
    )
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)
    qc.s(0)
    qc.rx(0.3, 0)
    qc.ry(-1.1, 0)
    qc.rz(0.7, 0)
    qc.u(0.5, 1.2, -0.4, 0)
    _compare(polypus.statevector(p), qc)


def test_two_qubit_interactions():
    import polypus

    p = polypus.Circuit(2).h(0).h(1).rzz(0, 1, 0.9).rxx(0, 1, -0.6).cz(0, 1)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.rzz(0.9, 0, 1)
    qc.rxx(-0.6, 0, 1)
    qc.cz(0, 1)
    _compare(polypus.statevector(p), qc)


def test_parameterized_binding_matches():
    import polypus

    # Param(0) is reused to confirm shared bindings resolve identically.
    p = (
        polypus.Circuit(3)
        .h(0)
        .rzz(0, 1, polypus.Param(0))
        .rx(2, polypus.Param(1))
        .ry(1, polypus.Param(0))
    )
    vals = [0.31, 1.27]
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.rzz(vals[0], 0, 1)
    qc.rx(vals[1], 2)
    qc.ry(vals[0], 1)
    _compare(polypus.statevector(p, vals), qc)


def test_qaoa_layer_matches():
    import polypus

    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    beta, gamma = 0.8, 0.4

    p = polypus.Circuit(4)
    for q in range(4):
        p = p.h(q)
    for a, b in edges:
        p = p.rzz(a, b, gamma)
    for q in range(4):
        p = p.rx(q, beta)

    qc = QuantumCircuit(4)
    for q in range(4):
        qc.h(q)
    for a, b in edges:
        qc.rzz(gamma, a, b)
    for q in range(4):
        qc.rx(beta, q)

    _compare(polypus.statevector(p), qc)


def _distinct_pair(n, rng):
    a = rng.randrange(n)
    b = rng.randrange(n)
    while b == a:
        b = rng.randrange(n)
    return a, b


def _apply_random_gate(p, qc, n, rng):
    """Apply one identical random gate to the polypus and Qiskit circuits."""
    single = ["h", "x", "y", "z", "s", "t", "sdg", "tdg"]
    rot = ["rx", "ry", "rz"]
    two = ["cx", "cz"]
    two_param = ["rzz", "rxx"]
    pool = single + rot + ["u"]
    if n >= 2:
        pool += two + two_param

    g = rng.choice(pool)
    if g in single:
        q = rng.randrange(n)
        p = getattr(p, g)(q)
        getattr(qc, g)(q)
    elif g in rot:
        q = rng.randrange(n)
        th = rng.uniform(-PI, PI)
        p = getattr(p, g)(q, th)
        getattr(qc, g)(th, q)
    elif g == "u":
        q = rng.randrange(n)
        a, b, c = (rng.uniform(-PI, PI) for _ in range(3))
        p = p.u(q, a, b, c)
        qc.u(a, b, c, q)
    elif g in two:
        a, b = _distinct_pair(n, rng)
        p = getattr(p, g)(a, b)
        getattr(qc, g)(a, b)
    else:  # two_param
        a, b = _distinct_pair(n, rng)
        th = rng.uniform(-PI, PI)
        p = getattr(p, g)(a, b, th)
        getattr(qc, g)(th, a, b)
    return p


def test_random_circuits_match_qiskit():
    import polypus

    rng = random.Random(2024)
    for _ in range(25):
        n = rng.randint(1, 5)
        depth = rng.randint(5, 30)
        p = polypus.Circuit(n)
        qc = QuantumCircuit(n)
        for _ in range(depth):
            p = _apply_random_gate(p, qc, n, rng)
        _compare(polypus.statevector(p), qc)


# ``polypus_sim::MAX_QUBITS``. Not exposed to Python (the constant belongs to the
# simulator, not to the seam), so it is mirrored here — keep both in sync.
_MAX_QUBITS = 30


def test_qubit_ceiling_is_documented_at_the_seam():
    """The ceiling has to be discoverable from Python, not only from the Rust
    source: `help(polypus.statevector)` must state it."""
    import polypus

    doc = polypus.statevector.__doc__ or ""
    assert "MAX_QUBITS" in doc and str(_MAX_QUBITS) in doc, (
        f"statevector.__doc__ must document the qubit ceiling; got:\n{doc}"
    )


# Comfortably above the ceiling: 31 is the first rejected value, 40 would need
# 16 TiB and 64 exceeds addressable memory entirely — an unguarded run would not
# merely be slow, it would take the interpreter down with it.
@pytest.mark.parametrize("num_qubits", [_MAX_QUBITS + 1, 40, 64])
def test_rejects_circuits_above_the_qubit_ceiling(num_qubits):
    """`statevector` must refuse a circuit larger than the backend's ceiling
    with a `ValueError` (contract C-1's failure mode), raised *before* the
    `2^n` allocation is attempted."""
    import polypus

    qc = polypus.Circuit(num_qubits).h(0)

    start = time.perf_counter()
    with pytest.raises(ValueError) as excinfo:
        polypus.statevector(qc)
    elapsed = time.perf_counter() - start

    message = str(excinfo.value)
    assert str(num_qubits) in message and str(_MAX_QUBITS) in message, (
        f"the error must name the requested count and the limit; got: {message!r}"
    )
    assert "qubit" in message.lower(), f"unexpected error message: {message!r}"
    # The guard is a single comparison, so the call returns essentially
    # instantly. A slow failure would mean allocation was attempted first.
    assert elapsed < 1.0, (
        f"rejection took {elapsed:.3f}s — the guard must fire before any "
        f"2^{num_qubits} allocation"
    )
