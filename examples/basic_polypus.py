import time
import polypus
import matplotlib.pyplot as plt
from qiskit.circuit.library import QFT

# Backend
N_QPUS = 2
N_SHOTS = 10**4

# Run the circuit with different numbers of shots
n_qubits_values = [10,11,12,13,14]
polypus_times = []

for n_qubits in n_qubits_values:
    # Quantum Circuit
    qft_circuit = QFT(n_qubits, inverse=False).decompose()
    qft_circuit.measure_all()
    
    # Run
    tic = time.time()
    result = polypus.run_quantum_circuit(qft_circuit, shots=N_SHOTS, n_qpus = N_QPUS, method="distribute_by_shots")
    polypus_running_time = time.time() - tic
    polypus_times.append(polypus_running_time)

# Plot the results
plt.plot(n_qubits_values, polypus_times, label=f'Polypus ({N_QPUS} QPUs)', marker='o')
plt.xlabel('Number of Qubits')
plt.ylabel('Running Time (s)')
plt.legend()
plt.title(f'Running Time ({N_SHOTS} shots QFT Circuit)')
plt.savefig('examples/running_polypus_qft.png')
