import math

import polypus

THETA = math.pi / 3
qc_poly = polypus.Circuit(6)
qc_poly.rx(5, THETA)  # state to teleport
qc_poly.h(4)  # Bell pair
qc_poly.cx(4, 3)  # Bell pair: h(4)+cx(4,3) == Qiskit's h(1)+cx(1,2).
# NOTE: cz(4,3) does NOT entangle (with q3=|0> the CZ is the identity).
qc_poly.cx(5, 4)  # Bell measurement basis
qc_poly.h(5)
qc_poly.cx(4, 3)  # deferred X correction

# Deferred Z correction: cz(5,3). 5 and 3 are not adjacent (5 only connects to 4),
# so we route with SWAP(5,4). QMIO's directionality in this region:
#   edge 5-4: only 5->4  (cx(5,4) is valid; cx(4,5) does NOT exist)
#   edge 4-3: only 4->3  (cx(4,3) is valid)
# The cx(4,5) the SWAP needs is built by reversing a cx(5,4) with H:
#   cx(4,5) = H(4) H(5) cx(5,4) H(4) H(5)
# SWAP(5,4) = cx(5,4); cx(4,5); cx(5,4)
qc_poly.cx(5, 4)
qc_poly.h(4)
qc_poly.h(5)
qc_poly.cx(5, 4)
qc_poly.h(4)
qc_poly.h(5)  # = cx(4,5)
qc_poly.cx(5, 4)

qc_poly.cz(4, 3)  # CZ(logical qubit 5, 3): qubit 5's state is now in position 4

# undo SWAP(5,4)
qc_poly.cx(5, 4)
qc_poly.h(4)
qc_poly.h(5)
qc_poly.cx(5, 4)
qc_poly.h(4)
qc_poly.h(5)  # = cx(4,5)
qc_poly.cx(5, 4)

qc_poly.measure_all()

# ── Run on QMIO and check ────────────────────────────────────────────────────
SHOTS = 1000
BOB = 3  # the teleported state ends up in qubit 3
p1_theory = math.sin(THETA / 2) ** 2  # expected P(Bob=|1>) = sin^2(pi/6) = 0.25

print(qc_poly)
result = polypus.run_quantum_circuit(qc_poly, shots=SHOTS, infrastructure="qmio")

# One QPU, so `counts` holds a single {bitstring: count} dict.
counts = result.counts[0]
total = sum(counts.values())

# Polypus bit order: little-endian with the most significant bit on the left,
# so qubit i is character i counting from the right.
p1_bob = sum(c for bits, c in counts.items() if bits[::-1][BOB] == "1") / total

print(f"\ncounts              : {counts}")
print(f"P(Bob=|1>) measured : {p1_bob:.4f}")
print(f"P(Bob=|1>) expected : {p1_theory:.4f}")
print(f"absolute error      : {abs(p1_bob - p1_theory):.4f}")

# Margin for the sampling noise (~0.014 with 1000 shots) and the QPU noise.
TOL = 0.10
if abs(p1_bob - p1_theory) <= TOL:
    print(f"\n=> Teleportation CORRECT   (|delta| <= {TOL})")
else:
    print(f"\n=> Teleportation INCORRECT (|delta| > {TOL})")
