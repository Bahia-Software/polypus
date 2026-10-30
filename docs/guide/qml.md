# Quantum machine learning

Polypus trains quantum models in two ways:

- **Your Qiskit circuits** with `polypus.qml.train()`: you bring a feature map, an ansatz and a cost that scores the measured bitstrings. Polypus runs the circuits; their meaning is yours.
- **Native models** with `polypus.qml.Model`: Polypus builds the model from encoders, trainable layers and a readout, so it also knows how to differentiate it exactly, predict with it and save it.

| | `qml.train` (Qiskit circuits) | `qml.Model` (native) |
|---|---|---|
| The model | Any feature map and ansatz you build in Qiskit | Encoders, a hardware-efficient ansatz, QCNN convolution and pooling |
| Scoring | Your function of the measured bitstrings (or counts) | A readout of Pauli observables and a loss (hinge, cross-entropy, squared error, …) |
| Adam's gradient | Finite differences of the fitness | Exact parameter-shift gradient |
| Exact probabilities instead of shots | No | Yes (`exact=True`, local native simulator) |
| Minibatches | No | Yes (`batch_size`) |
| Prediction | Counts per row (`qml.predict`); you map them to classes | Decisions per row (`TrainedModel.predict`), or counts |
| Saving a trained model | Save `best_params` yourself | `TrainedModel.save` / `qml.load` |

Both run on the same backends and infrastructures and take the same optimizers. Pass native pieces to `qml.train` and it raises `TypeError`.

## Qiskit circuits: `qml.train`

`polypus.qml.train()` trains a data-encoding circuit: a Qiskit `feature_map` encodes each row of `x_train` (one sample per row, `len(feature_map.parameters)` values each), an `ansatz` holds the trainable weights, and the optimizer maximises the mean per-sample score over the training set. It takes the same optimizers and parameters as `train()`, `polypus.External` included: there each candidate the library asks for runs one circuit per training sample, and `max_evaluations` counts candidates, not circuits.

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

Like `train()`, it takes a keyword-only `callback`, called once per iteration with a `polypus.TrainProgress`; returning `True` stops the run with what it has reached (see [Watching and stopping a run](training.md#watching-and-stopping-a-run)). Its `evaluations` count candidates, each one circuit per training sample.

### Cost functions

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

`polypus.Rotosolve` needs a fitness that averages a cost per bitstring, so it does not train with a `SampleCost`: `qml.train` raises `polypus.ConfigurationError` (a `ValueError`) before anything runs. It trains the ansatz only, so the feature map's gates are free; see [Rotosolve](training.md#rotosolve).

Integer labels (Python or NumPy ints and bools) reach your function as `int`; if any label is not an integer, all of them arrive as `float` (regression targets). `y_train` is validated before anything runs: a length other than the number of rows, a `NaN`/`inf`, a string or a one-hot row is rejected with the offending index. Without `y_train`, one `expectation_function(bitstring)` is shared by every sample, as before.

### Prediction

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

## Native models: `qml.Model`

A native model has three pieces: an **`Encoder`** that loads each sample, an **`Ansatz`** with the trainable layers, and a **`Readout`** that turns the measurement into a prediction. `Model.fit` trains it and returns a `TrainResult`, as `train()` does; `bind` fixes the trained parameters for inference.

```python
import numpy as np
import polypus
from polypus import qml

# Two classes on one feature: small angles are +1, large ones -1.
X = np.array([[0.1], [0.3], [0.2], [2.9], [3.0], [2.8]])
y = np.array([1, 1, 1, -1, -1, -1])

model = qml.Model(
    qml.Encoder(1).angle_encoder("ry"),
    qml.Ansatz(1).real_amplitudes(1),
    qml.Readout([qml.Z(0)], decision="sign"),
)
result = model.fit(
    X, y, method=polypus.Adam(max_iters=60, learning_rate=0.1), loss="hinge", seed=3
)

trained = model.bind(result.best_params)
labels = trained.predict(np.array([[0.15], [2.95]]))  # [1, -1]
```

[`examples/qml_iris_demo.py`](../../examples/qml_iris_demo.py) trains one on the Iris flowers end to end, and [`examples/qml_native_showcase.py`](../../examples/qml_native_showcase.py) runs every encoder, layer, optimizer, readout, decision and loss in twenty checked scenarios: classification, regression, an unsupervised ground state and data re-uploading among them, SPSA on sampled results, Rotosolve and the models it refuses, an ask/tell optimizer of your own through `polypus.External`, and a trained model run on Aer with a noise model.

Everything after `y` is keyword-only. The execution arguments have defaults (`shots=1024`, `infrastructure="local"`, `n_qpus=1`, `nodes=1`, `cores_per_qpu=2`, `id="qml"`), and without `backend` the native simulator runs the model locally (on CUNQA, whose QPUs run Aer, it is `"aer"`).

### Building a model

`Encoder(num_qubits)` and `Ansatz(num_qubits)` declare their width, which must match. Each method appends a layer and returns the same object, so calls chain. A layer that cannot work is rejected where you write it: pooling below two qubits raises `ValueError` at that `.pool(...)`, and the object keeps its previous layers.

| Layer | Method | Trainable parameters |
|---|---|---|
| Angle encoding: feature `j` as a rotation of qubit `j` | `angle_encoder("rx" \| "ry" \| "rz")` | none |
| Amplitude encoding: up to `2^n` features in the amplitudes (must come first) | `amplitude_encoder()` (`Encoder` only) | none |
| Basis encoding: `X` on qubit `j` when feature `j` is `1.0` | `basis_encoder()` | none |
| IQP (`ZZFeatureMap`) encoding | `iqp_encoder(entanglement="full")` | none |
| Hardware-efficient block (Qiskit's `TwoLocal`) | `hardware_efficient(reps, rotations=["ry", "rz"], entangler="cx", entanglement="linear", final_rotation_layer=True)` (`Ansatz` only) | one per rotation gate |
| `RealAmplitudes` | `real_amplitudes(reps)` (`Ansatz` only) | one per rotation gate |
| QCNN convolution, one block shared by every pair | `conv("basic" \| "cartan", pairing="alternating")` (`Ansatz` only) | 4 (`"basic"`) or 3 (`"cartan"`) |
| QCNN pooling: each pair of active qubits becomes one | `pool("basic", keep="even_positions")` (`Ansatz` only) | 3 |

The `Ansatz` may also re-encode the data between its trainable layers (data re-uploading) with `angle_encoder`, `basis_encoder` and `iqp_encoder`. `Model(encoder, ansatz, readout)` copies the pieces, so building on them afterwards does not change it; it checks that some layer loads the data, that the layers have trainable parameters and that the readout fits the qubits left at the end, and reports `num_params`.

### The readout

`Readout(observables, decision="raw", threshold=None)` measures Pauli observables and turns their expectations into a prediction. Write an observable as a product, `qml.Z(0)` or `qml.Z(0) @ qml.Z(1)` (also `X` and `Y`), or as a weighted sum, `qml.Observable([(0.5, qml.Z(0)), (-1.0, qml.Z(0) @ qml.Z(1))])`. Positions count the qubits still active, so after a pooling `Z(0)` is the first survivor.

| `decision` | Prediction | Observables |
|---|---|---|
| `"raw"` | the expectation `⟨O₀⟩` itself (`float`) | one (others are ignored) |
| `"sign"` | `+1` if `⟨O₀⟩ ≥ 0`, else `-1` | one |
| `"threshold"` | `+1` if `⟨O₀⟩ ≥ threshold`, else `-1` | one |
| `"argmax"` | the index of the largest `⟨O_k⟩`: the class | one per class, at least two |

`X` and `Y` observables are measured by rotating those qubits first; the readout may mix them as long as each qubit is measured in one basis throughout.

### Training

`model.fit(x, y=None, *, method, loss=None, ...)` takes rows of features (a list or a NumPy array) or a `qml.Dataset`, and any optimizer: `polypus.Adam` gets the exact gradient, and `DE`, `PSO`, `QNG`, `CMAES`, `SPSA` and `polypus.External` work as in `train()`. `polypus.Rotosolve` needs a fitness that is a sinusoid of each parameter, so it takes only `loss="expectation"` and models whose trainable parameters each sit in one rotation: a `conv` or `pool` layer over more than one pair shares its parameters, and `fit` raises `polypus.ConfigurationError` for it before anything runs.

| `loss` | Labels | Scores `⟨O₀⟩` (or all `⟨O_k⟩`) as |
|---|---|---|
| `"hinge"` | `+1` / `-1` | `max(0, 1 − y·⟨O₀⟩)` |
| `"binary_cross_entropy"` | `0` / `1` | cross-entropy with `p = (1 + ⟨O₀⟩) / 2` |
| `"squared_error"` | any number | `(⟨O₀⟩ − y)²` |
| `"categorical_cross_entropy"` | class index `0 … k−1` | softmax cross-entropy over the `k` observables (the loss for `"argmax"`) |
| `"expectation"` | none | `−⟨O₀⟩`: maximises the readout |

With labels, `loss` is required. Without them (`y=None`), the run maximises the readout's first observable, and a supervised loss raises `polypus.ConfigurationError`. To minimise a cost Hamiltonian, write it as an `Observable` with negated coefficients:

```python
cost = qml.Readout([qml.Observable([(-1.0, qml.Z(0) @ qml.Z(1)), (-0.5, qml.Z(1))])])
model = qml.Model(
    qml.Encoder(2).angle_encoder("ry"), qml.Ansatz(2).hardware_efficient(1), cost
)
result = model.fit(X_unlabeled, method=polypus.Adam(max_iters=100))
```

Five more options are keyword-only:

- `exact=True` computes each circuit's probabilities exactly instead of sampling, so `shots` no longer matters and the run has no shot noise. It needs the local native simulator, and memory grows as `2^n` per circuit.
- `batch_size=b` scores each evaluation on `b` samples drawn afresh from the seed (`1 ≤ b < number of samples`). Adam uses the batch for its gradient and keeps its best result on the whole set; the other optimizers report `best_fitness` re-scored on the whole set at the end.
- `initial_point` starts the optimizer from `num_params` parameters you choose instead of random ones, as in `train()` (see [Starting point](training.md#starting-point)): the `best_params` of an earlier run, or the `params` of a saved model to keep training it (see [Saving](#saving)).
- `final_shots=n` re-evaluates the best candidates once the optimizer has stopped, with `n` shots per circuit, as in `train()` (see [Final re-evaluation](training.md#final-re-evaluation)): `result.final_params` is the one with the best value and `result.final_fitness` that value. The final batch scores every training sample, also with `batch_size`, and the run before it is the one without `final_shots`. `exact=True` uses no shots, so `fit` refuses `final_shots` with it.
- `callback=f` calls `f` once per iteration with a `polypus.TrainProgress`, as in `train()`: returning `True` stops the run, which returns what it has reached, and the `final_shots` batch is still sent (see [Watching and stopping a run](training.md#watching-and-stopping-a-run)). With `batch_size`, the `best_fitness` it sees is the minibatch estimate, except for Adam, whose best is on the whole set. With Adam, `evaluations` counts the point each gradient batch measures, not the gradient's shifted circuits.

**Cost of the exact gradient.** One gradient runs `samples × (1 + 2 × occurrences)` circuits, where `occurrences` counts the trainable gates. A parameter that several gates share costs two circuits per gate: a basic convolution over 4 qubits has 4 parameters on 3 pairs, so 12 occurrences and 25 circuits per sample instead of 9. The `1` is the parameters' own circuits, so an Adam iteration is that one batch: Adam reads the iteration's fitness from it rather than running them again. With `batch_size`, the gradient runs on the minibatch and Adam scores the parameters on the whole set apart, one more batch of `samples` circuits.

### Prediction

`model.bind(params)` returns a `TrainedModel` with those parameters; the `Model` itself does not change, so it can carry several (from different seeds, say). Both methods take rows or a `Dataset`, and the execution arguments of `fit`.

- `trained.predict(x)` returns one decision per row: an `int` for `"sign"`, `"threshold"` and `"argmax"`, a `float` for `"raw"`.
- `trained.run(x)` returns the counts behind them: a `RunResult` with one dict per row (`merged_counts` is `None`). With `exact=True` it returns an `ExactResult`, whose `probabilities` holds one dict per row.

### Inspecting the circuit

`model.circuit(x)` returns the circuit the model runs for one sample `x`, a `polypus.Circuit` to draw or export like any other (see [Drawing](circuits.md#drawing) and [Export](circuits.md#export)). The features are fixed, the trainable angles are the parameters `theta_0`, `theta_1`, … in the order of `best_params`, and the readout's basis change comes right before the measurement:

```python
from polypus import qml

model = qml.Model(
    qml.Encoder(2).angle_encoder("ry"),
    qml.Ansatz(2).real_amplitudes(1),
    qml.Readout([qml.Y(0)], decision="sign"),
)
qc = model.circuit([0.3, 1.2])
print(qc)
print(qc.param_names)  # ['theta_0', 'theta_1', 'theta_2', 'theta_3']
```

```text
     ┌─────────┐ ┌─────────────┐       ┌─────────────┐ ┌─────┐ ┌───┐ ┌─┐
q0: ─┤ Ry(0.3) ├─┤ Ry(theta_0) ├───■───┤ Ry(theta_2) ├─┤ S†  ├─┤ H ├─┤M├─────
     └─────────┘ └─────────────┘   │   └─────────────┘ └─────┘ └───┘ └╥┘
     ┌─────────┐ ┌─────────────┐ ┌─┴─┐ ┌─────────────┐                ║  ┌─┐
q1: ─┤ Ry(1.2) ├─┤ Ry(theta_1) ├─┤ X ├─┤ Ry(theta_3) ├────────────────╫──┤M├─
     └─────────┘ └─────────────┘ └───┘ └─────────────┘                ║  └╥┘
                                                                      ║   ║
c2: ══════════════════════════════════════════════════════════════════╩═══╩══
                                                                      0   1
```

`S†` then `H` rotates `Y` onto `Z`, which is what the measurement reads (`X` needs only the `H`). `model.circuit(x, params)` fixes the trainable angles too, and `trained.circuit(x)` does so with the trained model's own parameters: it is exactly the circuit `trained.run([x])` executes, so `polypus.run_quantum_circuit(trained.circuit(x), shots, "local", backend="polypus", seed=s)` gives the counts of `trained.run([x], shots=shots, seed=s)`. `x` is checked as a row of `fit` is: a feature count the encoders cannot carry or a non-finite value raises `ValueError`.

### Your own training loop: `qml.expval` and `qml.jacobian`

`fit` runs the whole optimization. To write the loop yourself, with a loss of your own or another framework's optimizer, two functions give you the model's raw values and their derivatives, as NumPy arrays (`float64`, one row per row of `x`):

- `qml.expval(model, x, params)` returns the expectation of **every** readout observable, shape `(rows, observables)`, whatever the decision: one column per class for `"argmax"`.
- `qml.jacobian(model, x, params, wrt="params")` returns their derivatives with respect to the parameters, shape `(rows, observables, num_params)`. `wrt="inputs"` differentiates with respect to the features instead, shape `(rows, observables, features)`, and `wrt="both"` returns the two arrays as a tuple.

Both take `params` as a list or an array of `num_params` values and the execution arguments of `run` (with `exact=True`, the exact values; with shots, estimates). The Jacobian is that of the expectations, not of a loss, so the chain rule is yours. For a squared error:

```python
params = np.zeros(model.num_params)
for _ in range(40):
    values = qml.expval(model, X, params, exact=True)[:, 0]
    by_params = qml.jacobian(model, X, params, exact=True)[:, 0, :]
    gradient = 2 * (values - y) @ by_params / len(y)  # d mean((⟨O₀⟩ − y)²) / dθ
    params = params - 0.3 * gradient
```

[`examples/qml_expval_jacobian_loop.py`](../../examples/qml_expval_jacobian_loop.py) runs this loop to the end and checks that the loss goes down.

- **Inputs.** The derivative with respect to the features covers angle encoders, re-uploaded ones included, and IQP encoders (for `Rzz(x_i·x_j)` it applies the product rule). Amplitude and basis encoders have no derivative in the data: `wrt="inputs"` or `"both"` with one raises `polypus.ConfigurationError`; `wrt="params"` works with any model.
- **Cost.** `expval` runs one circuit per row. `jacobian` runs two per shifted gate per row (`+π/2` and `−π/2`): every gate that reads a parameter (a parameter shared by several gates, as in `conv` and `pool`, counts once per gate), every encoder gate that reads the row for `"inputs"`, both for `"both"`. A `real_amplitudes(1)` over 2 qubits has 4 parameters, so 8 circuits per row.
- **Execution.** A call opens one backend for all its rows (one allocation on CUNQA) and runs them in blocks sized from the cores and the backend, so memory stays bounded however many rows you pass; Ctrl+C stops it between blocks. The result does not depend on the blocks: a call on all the rows equals the calls on pieces of them, put together.
- **Seeds.** The functions return bare arrays, so the seed drawn when you omit `seed` is only logged: pass `seed=` to repeat a sampled call exactly. On Aer and CUNQA, a seeded call repeats on the same machine and configuration.

### Data

`qml.Dataset(x, y=None)` checks the rows once (non-empty, equal width, finite) and keeps the labels with them. It splits reproducibly and scales features:

```python
data = qml.Dataset(X, y)
train, test = data.train_test_split(0.25, seed=7)
ranges = train.feature_ranges()
train.scale_features_to(0.0, np.pi)  # angle encoding works best on [0, π]
test.scale_features_with(ranges, 0.0, np.pi)  # the same scaling, frozen on train

result = model.fit(train, method=polypus.Adam(), loss="hinge")
labels = model.bind(result.best_params).predict(test)
```

A `Dataset` built without labels stays unlabeled through splits, and trains only with `loss="expectation"`. With a `Dataset`, do not pass `y`.

### Saving

```python
trained.save("classifier.json")
trained = qml.load("classifier.json")
```

The file is JSON: the model and its parameters, with a format name and a version. Loading checks it as `Model` and `bind` would, so a damaged or edited file raises `ValueError`, and it never runs code. The parameters read back exactly, so a loaded model predicts as the saved one did. `save` replaces any file at that path.

A loaded model can keep training from where it stopped: its `model` is the `Model`, and its `params` are the starting point.

```python
trained = qml.load("classifier.json")
more = trained.model.fit(
    X, y, method=polypus.Adam(), loss="hinge", initial_point=trained.params
)
trained = trained.model.bind(more.best_params)
```
