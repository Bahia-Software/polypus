import matplotlib.pyplot as plt
import numpy as np

# from qmiotools.integrations.qiskitqmio import FakeQmio
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qmiotools.integrations.qiskitqmio import QmioBackend


def myqft_inverse(n, swap_qubits=True):
    """Apply the Inverse Quantum Fourier Transform in big-endian convention."""
    circuit = QuantumCircuit(n)
    # Apply Hadamard and controlled phase gates starting from the MSB
    for j in range(n - 1, -1, -1):
        circuit.h(j)
        for k in range(j - 1, -1, -1):
            circuit.cp(-np.pi / 2 ** (j - k), j, k)
        circuit.barrier()
    # Swap qubits to correct the order
    if swap_qubits:
        for j in range(n // 2):
            circuit.swap(j, n - j - 1)

    circuit.measure_all()
    return circuit


def run_experiments(min_qubits, max_qubits):
    for n in range(min_qubits, max_qubits + 1):
        qft = myqft_inverse(n)
        print(f"Running QFT Inverse for {n} qubits")

        backend = QmioBackend()

        qft_t = transpile(qft, backend, basis_gates=["h", "rz", "cx"])

        # Print depth and width of the transpiled circuit
        print(f"Transpiled circuit depth: {qft_t.depth()}")

        # tic = time.time()
        # job = backend.run(qft_t, shots=1024)
        # tac = time.time()
        # print(f"Execution time Qiskit: {tac - tic} seconds")

        # # Save to csv method|n_qubits|time
        # results = open(f"qft_results.csv", "a+")
        # results.write(f"qiskit|{n}|{tac - tic}\n")
        # results.close()

        # qasm_str = dumps(qft_t)
        # qct_pol = polypus.Circuit.from_qasm2(qasm_str)
        # tic = time.time()
        # job = polypus.run_quantum_circuit(qct_pol, shots=1024, infrastructure="qmio")
        # tac = time.time()
        # print(f"Execution time polypus: {tac - tic} seconds")

        # # Save to csv method|n_qubits|time
        # results = open(f"qft_results.csv", "a+")
        # results.write(f"polypus|{n}|{tac - tic}\n")
        # results.close()

        # time.sleep(1)  # Sleep for 1 second to avoid overwhelming the backend


if __name__ == "__main__":
    # Create csv with header method|n_qubits|time
    results = open("qft_results.csv", "w")
    results.write("method|n_qubits|time\n")
    results.close()

    run_experiments(3, 17)

    # Create plot using the csv and save fig
    df = pd.read_csv("qft_results.csv", sep="|")
    plt.figure(figsize=(10, 6))
    for method in df["method"].unique():
        subset = df[df["method"] == method]
        plt.plot(subset["n_qubits"], subset["time"], marker="o", label=method)
    plt.title("Execution Time of Inverse QFT on Qmio")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.grid()
    plt.savefig("qft_execution_time.png")


# qft = myqft_inverse(12)

# print(qft)

# backend = QmioBackend()

# qft_t = transpile(qft, backend, basis_gates=["h", "rz", "cx"])

# # qft_t.draw("mpl", fold=-1, filename="qft.png")


# # To openQASM
# qasm_str = dumps(qft_t)
# # print(qasm_str)

# tic = time.time()
# job = backend.run(qft_t, shots=1024)
# tac = time.time()

# result = job.result().get_counts()

# print(result)
# print(f"Execution time Qiskit: {tac - tic} seconds")


# qct_pol = polypus.Circuit.from_qasm2(qasm_str)

# print(qct_pol)

# tic = time.time()
# job = polypus.run_quantum_circuit(qct_pol, shots=1024, infrastructure="qmio")
# tac = time.time()
# print(f"Execution time polypus: {tac - tic} seconds")
# counts_poly = job[0] if isinstance(job, list) else job

# print(counts_poly)
