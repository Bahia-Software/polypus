import time

import polypus

tic = time.time()
circuit = polypus.Circuit(1).x(0).measure_all()
tac = time.time()
print(f"Time to create circuit: {tac - tic} seconds")
print(circuit)
tic = time.time()
result = polypus.run_quantum_circuit(circuit, shots=10000, infrastructure="qmio")
tac = time.time()
print(result.counts[0])
print(f"Execution time: {tac - tic} seconds")
