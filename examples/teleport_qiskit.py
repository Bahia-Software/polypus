# ── teleport_qiskit.py ──────────────────────────────────────────────────────
# Teleportación cuántica en QmioBackend: MISMO circuito en Qiskit y en polypus
# para comparar el TIEMPO de cada librería (mismo trabajo → comparación justa).
# Lanzar:  python examples/teleport_qiskit.py
# ─────────────────────────────────────────────────────────────────
#
# Circuito de 6 qubits enrutado a mano para la topología de QMIO (cadena 3-4-5):
#   q5  estado a teleportar  |ψ⟩ = Rx(θ)|0⟩
#   q4  qubit Bell de Alice
#   q3  qubit de Bob   ← recibe |ψ⟩
# La corrección Z diferida (cz lógico q5-q3) necesita un SWAP(5,4) porque q5 y q3
# no son adyacentes. Medida diferida → solo puertas estándar (sin if clásico).
#
# Resultado esperado con θ=π/3:  P(Bob=|1⟩) = sin²(π/6) ≈ 0.25
# ────────────────────────────────────────────────────────────────────────────

import math
import time

import polypus
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qmiotools.integrations.qiskitqmio import QmioBackend

THETA = math.pi / 3
SHOTS = 1024

# ── 1. Circuito Qiskit ───────────────────────────────────────────────────────

# Builder único: construye el MISMO circuito en ambas librerías (garantiza que
# son idénticos). La única diferencia de API es el orden de argumentos de rx
# (Qiskit: rx(theta, q); polypus: rx(q, theta)), adaptado con el parámetro `rx`.
def build_teleport(qc, rx):
    rx(5, THETA)                          # estado a teleportar  |ψ⟩ = Rx(θ)|0⟩
    qc.h(4); qc.cx(4, 3)                  # par Bell (Alice q4 — Bob q3)
    qc.cx(5, 4); qc.h(5)                  # base de medida de Bell
    qc.cx(4, 3)                           # corrección X diferida
    # corrección Z diferida cz(5,3): 5 y 3 no adyacentes → SWAP(5,4).
    # cx(4,5) = H(4)H(5) cx(5,4) H(4)H(5);  SWAP(5,4) = cx(5,4) cx(4,5) cx(5,4)
    qc.cx(5, 4); qc.h(4); qc.h(5); qc.cx(5, 4); qc.h(4); qc.h(5); qc.cx(5, 4)
    qc.cz(4, 3)                           # corrección Z (estado de q5 ya en pos. 4)
    qc.cx(5, 4); qc.h(4); qc.h(5); qc.cx(5, 4); qc.h(4); qc.h(5); qc.cx(5, 4)  # deshacer SWAP
    qc.measure_all()
    return qc


qc_qk = QuantumCircuit(QuantumRegister(6, "q"))
build_teleport(qc_qk, lambda qb, th: qc_qk.rx(th, qb))

# ── 2. Circuito polypus ──────────────────────────────────────────────────────

qc_poly = polypus.Circuit(6)
build_teleport(qc_poly, lambda qb, th: qc_poly.rx(qb, th))

# ── 3. Ejecución en QmioBackend ──────────────────────────────────────────────

if __name__ == "__main__":
    p1_theory = math.sin(THETA / 2) ** 2   # P(Bob=|1⟩) = sin²(π/6) = 0.25
    BOB       = 3                           # el estado teleportado queda en q3
    TOL       = 0.10                         # margen ruido muestreo + QPU
    print(f"P(Bob=|1⟩) teórico: {p1_theory:.4f}\n")

    def p_bob(counts):
        # little-endian con MSB a la izquierda → qubit i = carácter i por la derecha
        total = sum(counts.values())
        return sum(v for k, v in counts.items() if k[::-1][BOB] == "1") / total

    def report(tag, counts, elapsed):
        p1     = p_bob(counts)
        err    = abs(p1 - p1_theory)
        estado = "CORRECTA" if err <= TOL else "INCORRECTA"
        print(f"[{tag}] P(Bob=|1⟩)={p1:.4f}  error={err:.4f}  "
              f"tiempo={elapsed:.3f}s  => {estado}")

    backend = QmioBackend()

    # Qiskit: opt_level=0 + layout identidad → NO re-enruta, solo traduce a
    # puertas nativas; corre EXACTAMENTE el mismo circuito que polypus.
    print("[Qiskit] Enviando circuito a Qmio...")
    t0        = time.time()
    qc_qk_t   = transpile(qc_qk, backend, optimization_level=0,
                          initial_layout=list(range(6)))
    t1 = time.time()
    job_qk    = backend.run(qc_qk_t, shots=SHOTS)
    t1_run = time.time()
    counts_qk = job_qk.result().get_counts()
    t_qk      = time.time() - t0

    # polypus: misma ruta de extremo a extremo (serializa + envía + recoge).
    print("[polypus] Enviando circuito a Qmio...")
    t0          = time.time()
    result_poly = polypus.run_quantum_circuit(qc_poly, shots=SHOTS, infrastructure="qmio")
    t_poly      = time.time() - t0
    counts_poly = result_poly[0] if isinstance(result_poly, list) else result_poly

    print()
    report("Qiskit ", counts_qk, t_qk)
    report("polypus", counts_poly, t_poly)
    print("\n── Tiempos ──")
    print(f"  Qiskit : {t_qk:.3f}s")
    print(f"  Qiskit run: {t1_run - t1:.3f}s")
    print(f"  polypus run: {t_poly:.3f}s")
    if t_poly > 0:
        print(f"  ratio Qiskit/polypus: {t_qk / t_poly:.2f}x")
