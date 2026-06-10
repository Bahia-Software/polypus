import polypus, numpy as np
from qiskit.circuit.library import zz_feature_map, real_amplitudes

feature_map = zz_feature_map(feature_dimension=4, reps=2)
ansatz = real_amplitudes(num_qubits=4, reps=2)

X_train = np.random.rand(10, 4)  # Dummy training data (10 samples, 4 features)
# Loss: maximise the fraction of |1⟩ outcomes across all measured qubits
my_loss = lambda bitstring: sum(int(b) for b in bitstring) / len(bitstring)


result = polypus.qml.train(
    feature_map, ansatz, X_train,
    polypus.PSO(generations=50, population_size=20, bounds=(0, np.pi)),
    shots=1024, n_qpus=4, dimensions=len(ansatz.parameters),
    expectation_function=my_loss,
    infrastructure="local", nodes=1, cores_per_qpu=2, id="qml_run",
)

print(result)