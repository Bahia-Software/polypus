# Quantum machine learning

`polypus.qml.train()` trains a data-encoding circuit: a Qiskit `feature_map` encodes each row of `x_train` (one sample per row, `len(feature_map.parameters)` values each), an `ansatz` holds the trainable weights, and the optimizer maximises the mean per-sample score over the training set. It takes the same optimizers and parameters as `train()`.

Pass `y_train` (one label per row) to train a supervised model: each sample is then scored against **its own label**. Here a binary classifier reads out the parity of the measured bits:

```python
from qiskit.circuit.library import real_amplitudes, zz_feature_map

feature_map = zz_feature_map(feature_dimension=4, reps=2)
ansatz = real_amplitudes(num_qubits=4, reps=2)


def correct(bitstring, label):  # 1.0 when the read-out matches the label
    return float(bitstring.count("1") % 2 == label)


result = polypus.qml.train(
    feature_map,
    ansatz,
    X_train,  # shape (n_samples, 4)
    polypus.DE(generations=50, population_size=20),
    shots=1024,
    n_qpus=1,
    dimensions=len(ansatz.parameters),
    expectation_function=correct,  # fitness = expected training accuracy
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
    id="qml_classifier",
    y_train=y_train,  # shape (n_samples,): class indices
)
```

## Cost functions

With `y_train`, `expectation_function` takes one of three forms:

| `expectation_function` | Scores each sample by |
|---|---|
| `f(bitstring, label) -> float` | the count-weighted mean of `f` over its shots; `f` runs once per distinct `(label, bitstring)` of each batch |
| `polypus.CachedCost(f)` | the same, memoised across generations (`f` must be pure) |
| `polypus.SampleCost(g)` | `g(counts, label) -> float` on its whole `{bitstring: count}` dict, for losses that are not linear in the outcome probabilities |

For example, a `polypus.SampleCost` can maximise the log-likelihood, i.e. minimise the cross-entropy:

```python
import math


def log_likelihood(counts, label):
    hits = sum(n for bits, n in counts.items() if bits.count("1") % 2 == label)
    # Clip before the log: every score must be finite.
    return math.log(max(hits / sum(counts.values()), 1e-6))


expectation_function = polypus.SampleCost(log_likelihood)
```

Integer labels (Python or NumPy ints and bools) reach your function as `int`; if any label is not an integer, all of them arrive as `float` (regression targets). `y_train` is validated before anything runs: a length other than the number of rows, a `NaN`/`inf`, a string or a one-hot row is rejected with the offending index. Without `y_train`, one `expectation_function(bitstring)` is shared by every sample, as before.

## Prediction

To evaluate a trained model, `polypus.qml.predict()` builds the circuits training ran and executes every sample in one scheduled run: one backend batch and, on CUNQA, one allocation. Its `RunResult.counts` holds one dict per row, in row order; map them to classes with the objective's read-out:

```python
run = polypus.qml.predict(
    feature_map,
    ansatz,
    X_test,
    result.best_params,
    shots=1024,
    infrastructure="local",
    seed=7,
)


def majority_parity(counts):
    odd = sum(n for bits, n in counts.items() if bits.count("1") % 2)
    return int(odd > sum(counts.values()) - odd)


predictions = [majority_parity(counts) for counts in run.counts]
```

With `n_qpus > 1` the samples are spread over the QPUs; one sample's shots are never split. See [`examples/basic_qml.py`](../../examples/basic_qml.py) for a complete classifier, from training to held-out accuracy.
