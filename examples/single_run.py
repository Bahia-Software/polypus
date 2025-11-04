import polypus
from qiskit import QuantumCircuit
import time

NUM_QUBITS = 25
NUM_LAYERS = 30
NUM_SHOTS = 10000
qc = QuantumCircuit(NUM_QUBITS)
for _ in range(NUM_LAYERS):
    qc.h([i for i in range(NUM_QUBITS)])
    for i in range(NUM_QUBITS - 1):
        qc.crx(0.5, i, i + 1)
        qc.cx(i, i + 1)
qc.measure_all()

# Single run on local
print("Running single run local")
tic1 = time.time()
result1 = polypus.run_quantum_circuit(qc, shots=NUM_SHOTS, infrastructure="local", n_qpus=1)
tac1 = time.time()
time.sleep(5)

# Single run on Cunqa
print("Running single run cunqa")
tic2 = time.time()
result2 = polypus.run_quantum_circuit(qc, shots=NUM_SHOTS, infrastructure="cunqa", n_qpus=1)
tac2 = time.time()
time.sleep(5)

# Distribute by shots on Cunqa
print("Running distribute by shots cunqa")
tic3 = time.time()
result3 = polypus.run_quantum_circuit(qc, shots=NUM_SHOTS, infrastructure="cunqa", n_qpus=10)
tac3 = time.time()

# Results
print(" ------------------ Results ------------------ ")
print("Single run local result:")
print("Time taken (s):", tac1 - tic1)

print("Single run cunqa result:")
print("Time taken (s):", tac2 - tic2)

print("Distribute by shots cunqa result:")
print("Time taken (s):", tac3 - tic3)