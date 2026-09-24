"""Supervised QML: train a two-class classifier with ``polypus.qml.train``.

Each row of ``X_train`` is scored against its label in ``y_train``. The
prediction is read from qubit 0, the right-most bit of a Qiskit bitstring.
"""

import numpy as np
import polypus
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import real_amplitudes

# Install the logger sink so the internal Rust log records are written to a file.
# NOTE: optimizer progress is logged at DEBUG level, and the default build
# compiles those out (feature "info-logs" => log/max_level_info). To capture it,
# build with debug logging enabled:
#   maturin develop --no-default-features --features "extension-module,debug-logs"
polypus.init_logger(level="info", file="logs/basic_qml.log")

# Two separable blobs in [0, pi]^2: class 0 near (0.6, 0.6), class 1 near (2.5, 2.5).
rng = np.random.default_rng(7)
n_per_class = 12
X = np.vstack(
    [
        rng.normal(0.6, 0.25, size=(n_per_class, 2)),
        rng.normal(2.5, 0.25, size=(n_per_class, 2)),
    ]
).clip(0.0, np.pi)
y = np.repeat([0, 1], n_per_class)
order = rng.permutation(len(y))
X, y = X[order], y[order]
X_train, y_train = X[:16], y[:16]
X_test, y_test = X[16:], y[16:]

# Angle encoding (one RY rotation per feature) + a hardware-efficient ansatz.
x = ParameterVector("x", 2)
feature_map = QuantumCircuit(2)
for qubit in range(2):
    feature_map.ry(x[qubit], qubit)
ansatz = real_amplitudes(num_qubits=2, reps=1)


def predicted_class(bitstring):
    return int(bitstring[-1])  # qubit 0 is the right-most bit


# The fitness is then the expected accuracy on the training set.
def correct(bitstring, label):
    return float(predicted_class(bitstring) == label)


# For a loss that is not linear in the outcome probabilities, such as the
# log-likelihood, score each sample's whole counts dict instead:
#
#     def log_likelihood(counts, label):
#         hits = sum(n for bits, n in counts.items() if predicted_class(bits) == label)
#         return math.log(max(hits / sum(counts.values()), 1e-6))
#
#     expectation_function=polypus.SampleCost(log_likelihood)

result = polypus.qml.train(
    feature_map,
    ansatz,
    X_train,
    polypus.DE(generations=30, population_size=12),
    shots=512,
    n_qpus=1,
    dimensions=len(ansatz.parameters),
    expectation_function=correct,
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
    id="qml_run",
    seed=42,
    y_train=y_train,
)
print(result)

# Inference: bind a sample and the trained weights, then take the majority vote.
# `best_params` follows the order of `ansatz.parameters`.
template = feature_map.compose(ansatz)
template.measure_all()
weights = dict(zip(ansatz.parameters, result.best_params))


def predict(sample, seed):
    circuit = template.assign_parameters({**dict(zip(x, sample)), **weights})
    counts = polypus.run_quantum_circuit(
        circuit, shots=512, infrastructure="local", seed=seed
    ).counts[0]
    votes = np.zeros(2)
    for bits, n in counts.items():
        votes[predicted_class(bits)] += n
    return int(votes.argmax())


predictions = [predict(sample, seed=i) for i, sample in enumerate(X_test)]
print(f"test accuracy: {np.mean(np.array(predictions) == y_test):.2f}")
