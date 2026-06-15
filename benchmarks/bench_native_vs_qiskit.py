"""
Benchmark: native polypus.Circuit vs Qiskit QuantumCircuit through polypus.train.

Measures:
1. Per-candidate binding cost (the GIL-bound step the native path eliminates):
   Qiskit assign_parameters vs native to_qasm2 (binding+serialization in Rust).
2. End-to-end DE training wall-clock on the QAOA MaxCut ansatz (4 qubits).

Usage:
    python benchmarks/bench_native_vs_qiskit.py
"""

import math
import statistics
import time

import polypus
from qiskit.circuit import ParameterVector, QuantumCircuit

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
N_QUBITS = 4


def qiskit_qaoa():
    params = ParameterVector("p", 2)
    qc = QuantumCircuit(N_QUBITS)
    for q in range(N_QUBITS):
        qc.h(q)
    for a, b in EDGES:
        qc.rzz(params[0], a, b)
    for q in range(N_QUBITS):
        qc.rx(params[1], q)
    qc.measure_all()
    return qc


def native_qaoa():
    qc = polypus.Circuit(N_QUBITS)
    for q in range(N_QUBITS):
        qc.h(q)
    for a, b in EDGES:
        qc.rzz(a, b, polypus.Param(0))
    for q in range(N_QUBITS):
        qc.rx(q, polypus.Param(1))
    qc.measure_all()
    return qc


def bench(fn, n=2000, repeat=5):
    """Best-of-`repeat` mean microseconds per call over `n` calls."""
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for i in range(n):
            fn(i)
        dt = (time.perf_counter() - t0) / n * 1e6
        best = min(best, dt)
    return best


def main():
    qk, nv = qiskit_qaoa(), native_qaoa()

    # 1. Binding microbenchmark
    us_qiskit = bench(lambda i: qk.assign_parameters([0.001 * i, 0.002 * i], inplace=False))
    us_native = bench(lambda i: nv.to_qasm2([0.001 * i, 0.002 * i]))
    print(f"binding (QAOA 4q, 2 params, mean of best round, n=2000):")
    print(f"  qiskit assign_parameters : {us_qiskit:8.1f} us/candidate")
    print(f"  native  bind+QASM        : {us_native:8.1f} us/candidate")
    print(f"  speedup                  : {us_qiskit / us_native:8.1f}x")

    # 2. End-to-end DE training (identical settings, fresh circuits)
    def train_once(qc, run_id):
        t0 = time.perf_counter()
        polypus.train(
            qc,
            polypus.DE(generations=10, population_size=20, tolerance=1e-12),
            shots=512,
            n_qpus=1,
            dimensions=2,
            expectation_function=lambda b: float(b.count("1")),
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id=run_id,
        )
        return time.perf_counter() - t0

    for label, qc_factory in (("qiskit", qiskit_qaoa), ("native", native_qaoa)):
        times = [train_once(qc_factory(), f"bench_{label}_{i}") for i in range(3)]
        print(f"train DE 10x20 ({label:6s})    : {min(times):.2f}s best, "
              f"{statistics.mean(times):.2f}s mean of 3")


if __name__ == "__main__":
    main()
