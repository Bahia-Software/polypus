# ADR 0007 — Native QML, with its own API beside `qml.train`

- **Status:** Proposed (2026-10)
- **Contracts:** [C-5 · Optimizer ↔ oracle](../CONTRACTS.md#c-5--optimizer--oracle-contract-polypus-optimizers), [C-7 · Seeding & run manifest](../CONTRACTS.md#c-7--seeding--run-manifest-python-entry-points), [C-8 · `qml.train`](../CONTRACTS.md#c-8--qmltrain-rowdimensionlabel-symmetry-python-entry-point), [C-11 · Native QML](../CONTRACTS.md#c-11--native-qml-polypus-qml-problem--oracle-and-polypusqmlmodel)
- **Supersedes:** —
- **Issue / PRs:** #71 (the `polypus-qml` crate and the native API); reopens [ADR 0002](0002-qml-supervised-labels.md) on its own criterion

## Context

`polypus.qml.train` trains Qiskit circuits: a feature map and an ansatz,
composed and pre-bound per sample, scored by the caller's cost (a callable,
`CachedCost`, `SampleCost`, `Qubo`/`Ising`). The circuit is opaque to Polypus,
and so is the meaning of each bitstring. ADR 0002 added labels to that form
through Python objectives, and named the condition for revisiting it: *"a
native (pure-Rust, `Portable`) supervised reducer"*, evaluated without Python.

The `polypus-qml` crate is that reducer, and more: Polypus builds the model
itself (encoders, a hardware-efficient ansatz, QCNN convolution and pooling), so
it knows what the circuit measures and how a measurement becomes a prediction
and a loss. That knowledge is what buys the native form's capabilities:

- an **exact parameter-shift gradient** for Adam (C-11 (e)), where the Qiskit
  form can only take finite differences of an opaque fitness;
- an **exact mode** that reads probabilities from the native simulator instead
  of sampling;
- **logical readout positions** that follow pooling;
- **inference with the model's own decision** and a **save format** that
  rebuilds the model.

## Decision

1. **The native form is its own API**, `polypus.qml.Model`, not a second form of
   `qml.train`:

   ```python
   model = polypus.qml.Model(
       polypus.qml.Encoder(4).angle_encoder("ry"),
       polypus.qml.Ansatz(4).hardware_efficient(2),
       polypus.qml.Readout([polypus.qml.Z(0)], decision="sign"),
   )
   result = model.fit(X, y, method=polypus.Adam(), loss="hinge")
   trained = model.bind(result.best_params)
   labels = trained.predict(X_test)
   trained.save("model.json")
   trained = polypus.qml.load("model.json")
   ```

   `qml.train` and `qml.predict` stay the Qiskit form, as before, and reject
   native pieces with a `TypeError`.
2. **The model holds no trained state.** `fit` returns the usual `TrainResult`;
   `bind` gives an immutable `TrainedModel`, which predicts, runs and saves. One
   `Model` can carry several bindings (different seeds, different data).
3. **The readout belongs to the model, the loss to training.** `Readout`
   (observables, decision, threshold) is a piece of the `Model`; `loss` is an
   argument of `fit`. With labels a supervised loss must be named; without them
   the run maximises the readout (the `expectation` loss), and the two are never
   mixed silently.
4. **Everything below the entry points is shared.** The native form trains
   through the same `QmlOracle` (a `QmlSource::Native` beside
   `QmlSource::Qiskit`), the same scheduler and planner, the same optimizers and
   the same minibatch rules. Adam, the `GradientOracle`, the exact mode of the
   native backend and the minibatches serve both forms.
5. **One save format for Rust and Python**: the crate's `TrainedModel`, as
   versioned JSON, revalidated on load.

## Alternatives considered

1. **One `qml.train` for both forms**, choosing the form by the type of each
   argument (the precedent of `train`, which takes a `polypus.Circuit` or a
   Qiskit `QuantumCircuit`). Implemented first, then reverted. `train` can do
   this because only the circuit differs between its forms: both run a circuit
   and score its counts with the same cost. In QML the forms differ in what the
   model *is*, not in how it runs, so every argument took two meanings
   (`expectation_function` a callable or a model readout; labels from `y_train`
   or from a `Dataset`; a loss with rules that exist in one form only), each
   needed rules for mixing the forms, defaults had to depend on the form
   (`backend`), the signature outgrew the linter's argument limit, and inference
   could not apply a decision without the model it came from. Making the forms
   converge would mean either reducing the native model to an opaque circuit,
   losing what it is for, or constraining Qiskit circuits until Polypus could
   read them.
2. **`model.train(...)` beside `qml.train(...)`** with the same verb (the
   earlier design of this branch): two `train`s with different signatures, and
   no clear answer to "which one?". `fit` names the native path apart.
3. **A stateful model** (`fit` stores the parameters, as scikit-learn and
   Qiskit ML's `VQC` do): convenient, but re-fitting overwrites, a `predict`
   before `fit` needs a "not fitted" state, and one model cannot hold two
   results. `bind` keeps the convenience without mutable state, as PennyLane's
   QNodes and Qiskit's QNNs keep weights explicit.
4. **Python reducers only** (keep ADR 0002's design): no exact gradient, no
   exact mode, a Python call per sample.

## Consequences

- Users choose by what they have: their own Qiskit circuits and cost
  (`qml.train`), or a model Polypus builds (`qml.Model`). The guide documents
  both side by side.
- Each signature has only its own arguments; no `Union` types for the forms,
  no cross-form rules.
- A new capability lands where it makes sense: one built on knowing the model
  (gradients, decisions, saving) in the native API, one on running circuits in
  the shared layers, where both forms get it.
- The execution arguments repeat across `train`, `qml.train`, `qml.predict`,
  `Model.fit` and `TrainedModel.predict`/`run`. Grouping them is left to a
  later, library-wide change.

## Reopening criteria

- A way to give the Qiskit form the native capabilities without constraining its
  circuits (for example, importing a Qiskit circuit as a layer of a native
  model), which would let the forms meet on the native side.
- A user-written training loop (PennyLane style, or a PyTorch layer) needing the
  model's expectations and Jacobian as primitives.
