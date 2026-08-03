import polypus
import time


tic = time.time()
bell = polypus.Circuit(1).x(0).measure_all()
tac = time.time()
print(f"Time to create circuit: {tac - tic} seconds")
print(bell)
tic = time.time()
counts = polypus.run_quantum_circuit(bell, shots=10000, infrastructure="qmio")
tac = time.time()
print(counts)
print(f"Execution time: {tac - tic} seconds")