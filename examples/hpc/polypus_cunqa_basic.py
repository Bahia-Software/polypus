import polypus
from qiskit.synthesis import synth_qft_full

polypus.init_logger(run_name="polypus_cunqa_basic")

# QFT definition
n_qubits = 5
qft_circuit = synth_qft_full(n_qubits)
qft_circuit.measure_all()

# 1. Single Run Local
print("Calling Polypus Local with 1 qpu")
result = polypus.run_quantum_circuit(qft_circuit, shots=1000, infrastructure="local")
print(result)

# 2. Single Run Local with distribute by shots
print("Calling Polypus Local with 10 qpus")
result = polypus.run_quantum_circuit(
    qft_circuit, shots=10000, infrastructure="local", n_qpus=10
)
print(result)

# 3. Single Run CUNQA
print("Calling Polypus CUNQA with 1 qpu")
result = polypus.run_quantum_circuit(qft_circuit, shots=1000, infrastructure="cunqa")
print(result)

# 4. Distribute by shots
print("Calling Polypus CUNQA with 10 qpus")
result = polypus.run_quantum_circuit(
    qft_circuit, shots=10000, infrastructure="cunqa", n_qpus=10
)
print(result)
