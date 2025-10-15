import polypus
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

result = polypus.run_quantum_circuit(qc, shots=1024, infrastructure="local", n_qpus=1)

print(result)
