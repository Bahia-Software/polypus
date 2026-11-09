# ── teleport_qiskit.py ──────────────────────────────────────────────────────
# Quantum teleportation on QmioBackend: the SAME circuit in Qiskit and in polypus,
# to compare the TIME each library takes (same work → fair comparison).
# Run:  python examples/hpc/teleport_qiskit.py
# ─────────────────────────────────────────────────────────────────
#
# 6-qubit circuit routed by hand for QMIO's topology (chain 3-4-5):
#   q5  state to teleport  |ψ⟩ = Rx(θ)|0⟩
#   q4  Alice's Bell qubit
#   q3  Bob's qubit   ← receives |ψ⟩
# The deferred Z correction (logical cz q5-q3) needs a SWAP(5,4) because q5 and q3
# are not adjacent. Deferred measurement → standard gates only (no classical if).
#
# Expected result with θ=π/3:  P(Bob=|1⟩) = sin²(π/6) ≈ 0.25
# ────────────────────────────────────────────────────────────────────────────

import math
import time

import polypus
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qmiotools.integrations.qiskitqmio import QmioBackend

THETA = math.pi / 3
SHOTS = 1024

# ── 1. Qiskit circuit ───────────────────────────────────────────────────────


# A single builder: builds the SAME circuit in both libraries (guarantees they
# are identical). The only API difference is the argument order of rx
# (Qiskit: rx(theta, q); polypus: rx(q, theta)), adapted with the `rx` parameter.
def build_teleport(qc, rx):
    rx(5, THETA)  # state to teleport  |ψ⟩ = Rx(θ)|0⟩
    qc.h(4)
    qc.cx(4, 3)  # Bell pair (Alice q4 — Bob q3)
    qc.cx(5, 4)
    qc.h(5)  # Bell measurement basis
    qc.cx(4, 3)  # deferred X correction
    # Deferred Z correction cz(5,3): 5 and 3 not adjacent → SWAP(5,4).
    # cx(4,5) = H(4)H(5) cx(5,4) H(4)H(5);  SWAP(5,4) = cx(5,4) cx(4,5) cx(5,4)
    qc.cx(5, 4)
    qc.h(4)
    qc.h(5)
    qc.cx(5, 4)
    qc.h(4)
    qc.h(5)
    qc.cx(5, 4)
    qc.cz(4, 3)  # Z correction (q5's state already in position 4)
    qc.cx(5, 4)
    qc.h(4)
    qc.h(5)
    qc.cx(5, 4)
    qc.h(4)
    qc.h(5)
    qc.cx(5, 4)  # undo SWAP
    qc.measure_all()
    return qc


qc_qk = QuantumCircuit(QuantumRegister(6, "q"))
build_teleport(qc_qk, lambda qb, th: qc_qk.rx(th, qb))

# ── 2. polypus circuit ──────────────────────────────────────────────────────

qc_poly = polypus.Circuit(6)
build_teleport(qc_poly, lambda qb, th: qc_poly.rx(qb, th))

# ── 3. Run on QmioBackend ───────────────────────────────────────────────────

if __name__ == "__main__":
    p1_theory = math.sin(THETA / 2) ** 2  # P(Bob=|1⟩) = sin²(π/6) = 0.25
    BOB = 3  # the teleported state ends up in q3
    TOL = 0.10  # margin for sampling + QPU noise
    print(f"P(Bob=|1⟩) expected: {p1_theory:.4f}\n")

    def p_bob(counts):
        # little-endian with the MSB on the left → qubit i = character i from the right
        total = sum(counts.values())
        return sum(v for k, v in counts.items() if k[::-1][BOB] == "1") / total

    def report(tag, counts, elapsed):
        p1 = p_bob(counts)
        err = abs(p1 - p1_theory)
        status = "CORRECT" if err <= TOL else "INCORRECT"
        print(
            f"[{tag}] P(Bob=|1⟩)={p1:.4f}  error={err:.4f}  "
            f"time={elapsed:.3f}s  => {status}"
        )

    backend = QmioBackend()

    # Qiskit: opt_level=0 + identity layout → does NOT re-route, it only translates
    # to native gates; it runs EXACTLY the same circuit as polypus.
    print("[Qiskit] Sending the circuit to Qmio...")
    t0 = time.time()
    qc_qk_t = transpile(
        qc_qk, backend, optimization_level=0, initial_layout=list(range(6))
    )
    t1 = time.time()
    job_qk = backend.run(qc_qk_t, shots=SHOTS)
    t1_run = time.time()
    counts_qk = job_qk.result().get_counts()
    t_qk = time.time() - t0

    # polypus: the same end-to-end path (serialise + send + collect).
    print("[polypus] Sending the circuit to Qmio...")
    t0 = time.time()
    result_poly = polypus.run_quantum_circuit(
        qc_poly, shots=SHOTS, infrastructure="qmio"
    )
    t_poly = time.time() - t0
    counts_poly = result_poly.counts[0]

    print()
    report("Qiskit ", counts_qk, t_qk)
    report("polypus", counts_poly, t_poly)
    print("\n── Times ──")
    print(f"  Qiskit : {t_qk:.3f}s")
    print(f"  Qiskit run: {t1_run - t1:.3f}s")
    print(f"  polypus run: {t_poly:.3f}s")
    if t_poly > 0:
        print(f"  ratio Qiskit/polypus: {t_qk / t_poly:.2f}x")
