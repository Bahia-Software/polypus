import math

import polypus

THETA = math.pi / 3
qc_poly = polypus.Circuit(6)
qc_poly.rx(5, THETA)  # estado a teleportar
qc_poly.h(4)  # par Bell
qc_poly.cx(4, 3)  # par Bell: h(4)+cx(4,3) == h(1)+cx(1,2) de Qiskit.
# OJO: cz(4,3) NO entrelaza (con q3=|0> la CZ es identidad).
qc_poly.cx(5, 4)  # base de medida Bell
qc_poly.h(5)
qc_poly.cx(4, 3)  # corrección X diferida

# corrección Z diferida: cz(5,3). 5 y 3 no son adyacentes (5 solo conecta con 4),
# así que enrutamos con SWAP(5,4). Direccionalidad de QMIO en esta zona:
#   arista 5-4: solo 5->4  (cx(5,4) válido; cx(4,5) NO existe)
#   arista 4-3: solo 4->3  (cx(4,3) válido)
# El cx(4,5) que necesita el SWAP se construye invirtiendo un cx(5,4) con H:
#   cx(4,5) = H(4) H(5) cx(5,4) H(4) H(5)
# SWAP(5,4) = cx(5,4); cx(4,5); cx(5,4)
qc_poly.cx(5, 4)
qc_poly.h(4)
qc_poly.h(5)
qc_poly.cx(5, 4)
qc_poly.h(4)
qc_poly.h(5)  # = cx(4,5)
qc_poly.cx(5, 4)

qc_poly.cz(4, 3)  # CZ(qubit-5-lógico, 3): el estado del 5 ya está en la pos. 4

# deshacer SWAP(5,4)
qc_poly.cx(5, 4)
qc_poly.h(4)
qc_poly.h(5)
qc_poly.cx(5, 4)
qc_poly.h(4)
qc_poly.h(5)  # = cx(4,5)
qc_poly.cx(5, 4)

qc_poly.measure_all()

# ── Ejecución en Qmio y verificación ─────────────────────────────────────────
SHOTS = 1000
BOB = 3  # el estado teleportado queda en el qubit 3
p1_theory = math.sin(THETA / 2) ** 2  # P(Bob=|1>) esperado = sin^2(pi/6) = 0.25

print(qc_poly)
job = polypus.run_quantum_circuit(qc_poly, shots=SHOTS, infrastructure="qmio")

# n_qpus=1 -> lista con un único dict {bitstring: cuentas}; n_qpus>1 -> dict.
counts = job[0] if isinstance(job, list) else job
total = sum(counts.values())

# Convención de bits de polypus: little-endian con el bit más significativo a la
# izquierda, así que el qubit i es el carácter i contando desde la derecha.
p1_bob = sum(c for bits, c in counts.items() if bits[::-1][BOB] == "1") / total

print(f"\ncounts             : {counts}")
print(f"P(Bob=|1>) medido  : {p1_bob:.4f}")
print(f"P(Bob=|1>) teórico : {p1_theory:.4f}")
print(f"error absoluto     : {abs(p1_bob - p1_theory):.4f}")

# Margen para el ruido de muestreo (~0.014 con 1000 shots) y el ruido del QPU.
TOL = 0.10
if abs(p1_bob - p1_theory) <= TOL:
    print(f"\n=> Teleportacion CORRECTA  (|delta| <= {TOL})")
else:
    print(f"\n=> Teleportacion INCORRECTA (|delta| > {TOL})")
