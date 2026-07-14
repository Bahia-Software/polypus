"""
Benchmark: batched vs per-circuit Aer submission through polypus.train.

The local backend submits each evaluation batch (population) in ONE
AerSimulator.run call (max_parallel_experiments=0 → parallel C++ execution,
GIL released). This script quantifies the gain against the legacy behaviour
(one Aer call per circuit), emulated via monkeypatching.

Usage:
    python benchmarks/bench_batching.py
"""

import statistics
import time

import polypus
import polypus_python.local as local_mod

POPULATION = 32
GENERATIONS = 10
N_QUBITS = 8
SHOTS = 1024
REPEATS = 3


def ansatz():
    qc = polypus.Circuit(N_QUBITS)
    for q in range(N_QUBITS):
        qc.h(q)
    for q in range(N_QUBITS - 1):
        qc.rzz(q, q + 1, polypus.Param(0))
    for q in range(N_QUBITS):
        qc.rx(q, polypus.Param(1))
    qc.measure_all()
    return qc


def train_once(run_id):
    t0 = time.perf_counter()
    polypus.train(
        ansatz(),
        polypus.DE(
            generations=GENERATIONS, population_size=POPULATION, tolerance=1e-12
        ),
        shots=SHOTS,
        n_qpus=1,
        dimensions=2,
        expectation_function=lambda b: float(b.count("1")),
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id=run_id,
    )
    return time.perf_counter() - t0


class UnbatchedSpy(local_mod.AerSimulator):
    """Emulate the legacy path: one Aer call per circuit."""

    def run(self, circuits, **kwargs):
        if isinstance(circuits, list) and len(circuits) > 1:
            jobs = [super(UnbatchedSpy, self).run([qc], **kwargs) for qc in circuits]
            return _MergedJob(jobs)
        return super().run(circuits, **kwargs)


class _MergedJob:
    def __init__(self, jobs):
        self.jobs = jobs

    def result(self):
        return _MergedResult([j.result() for j in self.jobs])


class _MergedResult:
    def __init__(self, results):
        self.results = results

    def get_counts(self, i):
        return self.results[i].get_counts(0)


def main():
    real_sim = local_mod.AerSimulator
    print(
        f"DE {GENERATIONS}x{POPULATION}, {N_QUBITS} qubits, {SHOTS} shots, "
        f"best of {REPEATS}:"
    )

    results = {}
    for label, sim_cls in (
        ("batched (current)", real_sim),
        ("per-circuit (legacy)", UnbatchedSpy),
    ):
        local_mod.AerSimulator = sim_cls
        try:
            times = [train_once(f"bench_{label[:7]}_{i}") for i in range(REPEATS)]
        finally:
            local_mod.AerSimulator = real_sim
        results[label] = min(times)
        print(
            f"  {label:22s}: {min(times):6.2f}s best, "
            f"{statistics.mean(times):6.2f}s mean"
        )

    speedup = results["per-circuit (legacy)"] / results["batched (current)"]
    print(f"  speedup               : {speedup:6.1f}x")


if __name__ == "__main__":
    main()
