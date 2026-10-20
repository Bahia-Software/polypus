# ADR 0002 — Supervised labels for `qml.train`

- **Status:** Accepted (2026-09)
- **Contract:** [C-8 · qml.train row/dimension/label symmetry](../CONTRACTS.md#c-8--qmltrain-rowdimensionlabel-symmetry-python-entry-point)
- **Supersedes:** —

## Context

`polypus.qml.train` binds each `x_train` row into one training circuit and, for
every candidate `θ`, averages one shared `expectation_function(bitstring) -> float`
over the samples. The callback never learns which sample produced the counts, so
only label-free objectives could be expressed.

Supervised classifiers worked around it with a *label register*: `⌈log₂ K⌉`
extra qubits the ansatz never touches, set to `|y⟩` by the feature map, and a
cost that read the label back from the bitstring. That costs qubits, limits
losses to ones linear in the outcome probabilities, and breaks on real hardware,
where readout errors flip the label bits too.

## Decision

`qml.train` takes an optional, keyword-only `y_train`, one label per `x_train`
row. With it, each sample is scored against its own label, and the fitness is the
mean of the per-sample scores (maximised, as before). `expectation_function` is
then one of:

| `expectation_function` | with `y_train` |
|---|---|
| callable | `f(bitstring, label) -> float`: a per-shot score, averaged per sample weighted by counts |
| `polypus.CachedCost(f)` | the same, memoised across generations by `(label, bitstring)` |
| `polypus.SampleCost(g)` (new) | `g(counts, label) -> float` over the sample's whole `{bitstring: count}` distribution, so non-linear losses (log-likelihood, squared error of ⟨Z⟩) can be expressed |
| `polypus.Qubo` / `polypus.Ising` | `TypeError`: a declarative observable cannot read labels |

Without `y_train` nothing changes, and a `SampleCost` there is a `TypeError`.

If every label is an integer (Python or NumPy ints and bools) they reach the
objective as `int`; otherwise all are `float` (regression). Contract C-8 checks,
before anything runs, the count, that each label is a single finite number, and
that the objective can read labels.

Labels meet counts in the QML oracle, not in the `Planner`. The supervised branch
runs each window through `Planner::execute` and pairs circuit `i` of the
candidate-major window with `labels[i % n_train]`. The planner's reducer
(`Planner::evaluate` with a `CostObservable`) is position-free by contract, so a
reducer carrying labels would mis-pair them as soon as a planner reduced a batch
wave by wave. The unsupervised branch calls `Planner::evaluate` as before.

In code: `polypus-evaluation` gains `SupervisedObjective`, `Label`,
`PyLabelledCost` and `PySampleCost`, and `QmlOracle` holds a
`QmlObjective::{Unsupervised, Supervised}`.

## Consequences

- Without `y_train` the code path is unchanged; the determinism gate is
  byte-identical.
- `polypus-backend` and `polypus-observable`, the pyo3-free contracts, are
  untouched.
- The per-shot objective keeps `PyCallbackObservable`'s design: one Python call
  per unique `(label, bitstring)` pair of a batch, in one GIL section, and the
  aggregation in Rust. `SampleCost` makes one call per `(candidate, sample)`, in
  one GIL section per batch.
- Supervised results do not depend on `HashMap` iteration order: aggregation and
  the dicts passed to Python follow sorted-bitstring order.
- A `NaN`/`inf` score is `EvaluationError::NonFiniteScore`, naming the `x_train`
  row.
- Supervised reduction always runs in the caller's process (no reduce-at-source).

## Alternatives considered

1. **Labels through `Planner::evaluate`**, as a `CostObservable` adapter or a
   label field on `CircuitTask`. The adapter relies on the planner calling the
   reducer once, in task order, which its contract does not promise; the task
   field changes the pyo3-free backend contract for a Python-only feature.
2. **Keep the label register**, for the reasons in *Context*.
3. **Labels as arbitrary Python objects** (strings, one-hot rows): deduplication
   would depend on Python equality (`1 == 1.0 == True`) and hashability. Class
   names are encoded as integers first, as in scikit-learn.
4. **Always `float` labels**: a class-indexed cost (`weights[label]`) would fail.
5. **Guessing the cost's arity** with `inspect.signature`: unavailable for many
   callables. A one-argument cost given labels fails at the first batch with
   Python's own `TypeError`.

## Reopening criteria

Revisit when a native (pure-Rust, `Portable`) supervised reducer is needed, for
example a parity or cross-entropy read-out evaluated without Python, possibly at
the data source. That needs per-task labels in the pyo3-free planner contract,
with its own ADR.
