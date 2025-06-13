import polypus
from qiskit.circuit.library import QFT

# QFT definition
n_qubits = 5
qft_circuit = QFT(n_qubits, inverse=False).decompose()
qft_circuit.measure_all()

# 1. Single Run Local
print("Calling Polypus Local with 1 qpu")
result = polypus.run_quantum_circuit(qft_circuit, shots=1000, infraestructure="local")
print(result)

# 2. Single Run Local with distribute by shots
print("Calling Polypus Local with 10 qpus")
result = polypus.run_quantum_circuit(qft_circuit, shots=10000, infraestructure="local", n_qpus=10)
print(result)

# 3. Single Run QMIO
print("Calling Polypus CUNQA with 1 qpu")
result = polypus.run_quantum_circuit(qft_circuit, shots=1000, infraestructure="qmio")
print(result)

# 4. Distribute by shots
print("Calling Polypus Single Run for 10 QPUs")
result = polypus.run_quantum_circuit(qft_circuit, shots=10000, infraestructure="qmio", n_qpus=10)
print(result)