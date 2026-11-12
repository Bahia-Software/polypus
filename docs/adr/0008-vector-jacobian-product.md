# ADR 0008 — `qml.vjp`: the vector-Jacobian product in one adjoint run per row

- **Status:** Proposed (2026-11)
- **Contracts:** [C-7 · Seeding & run manifest](../CONTRACTS.md#c-7--seeding--run-manifest-python-entry-points), [C-11 · Native QML](../CONTRACTS.md#c-11--native-qml-polypus-qml-problem--oracle-and-polypusqmlmodel)
- **Supersedes:** —
- **Issue / PRs:** #559; builds on #502 (`polypus.torch.QuantumLayer`) and #503 (adjoint differentiation)

## Context

The backward pass of `polypus.torch.QuantumLayer` (#502) asked `qml.jacobian`
for the whole Jacobian, `(rows, K, num_params)` for `K` readout observables,
and contracted it with the incoming gradient `v` in NumPy. In exact mode the
Jacobian comes from adjoint differentiation (#503): one run per row **and per
observable**, `rows · K` runs of two statevectors each. PyTorch only needs the
contraction, and

```text
Σ_k v_k · ∂⟨O_k⟩/∂θ = ∂⟨Σ_k v_k · O_k⟩/∂θ
```

is the derivative of one diagonal observable: one adjoint run per row would
do, whatever `K`. A classifier with an `argmax` readout of `K` classes paid `K`
times what it needed on every training step. With shots there is nothing to
gain: the shift rule's circuits do not depend on the observables.

## Decision

1. **A new public primitive, `qml.vjp`,** beside `qml.jacobian`, which does not
   change (signature, behaviour, stubs):

   ```python
   qml.vjp(model, x, params, cotangent, *, wrt="params", <qml.jacobian's execution arguments>)
   ```

2. **It returns the VJP in the mathematical sense,** what the layer's
   `einsum`s computed: `wrt="params"` → `float64` `(num_params,)`, **summed
   over the rows**; `"inputs"` → `(rows, features)`; `"both"` → the tuple, each
   identical to its separate call. C-contiguous; `wrt` as in `qml.jacobian`.
3. **The cotangent is checked for its shape, and its values are read as
   `x`'s are.** It must be exactly `(rows, observables)` and real (a
   `polypus.ConfigurationError` otherwise, checked right after `params`), each
   value a real number as `x` takes it, a `bool` included (a string, `None` or
   another object is the `TypeError` `x` raises); a `NaN` or an infinity is
   **accepted and propagates** as it does through the contraction of
   `qml.jacobian`'s result.
4. **Exactly, one adjoint run per row;** with shots (and in the fallback when
   the memory budget holds one statevector but not two), `qml.jacobian`'s
   circuits, each block's Jacobian contracted with the block's cotangent: with
   the same `seed` the result is the contraction of `qml.jacobian`'s, to
   rounding. Only the exact adjoint path uses the combined observable.
5. **A combined observable that cannot be weighed safely** — a non-finite
   value in the cotangent, or a weight bound past `f64::MAX / 4`, so that
   neither the weight `Σ_k v_k · O_k` nor the derivatives overflow — is not
   handed to the simulator: the native one refuses non-finite weights
   (`SimError::NonFiniteWeight`), and near `f64::MAX` the derivatives' sums
   could overflow where the contraction does not. The binding tells by
   `DiagonalObservable::weight_bound()` (`Σ |c|`, a bound on every eigenvalue);
   the simulator's sweep back and each derivative stay within that bound, and
   the factor 4 leaves headroom for the sums over the gates that share a
   parameter and the chain-rule factors. **The row** with such an observable
   takes `qml.jacobian`'s adjoint runs, one per observable, and contracts
   them, so the cotangent reaches the product exactly as through the
   contraction; the other rows keep one run each. The route is decided **per
   row, not per block**, because the blocks follow the host's cores: decided
   per block, one row's `NaN` changed the last bits of the rows that shared
   its block, differently on hosts with different core counts. The crate
   gives each row's bound before any circuit is bound
   (`JacobianBlock::vjp_weight_bounds`), so each row's runs are built once
   (decided during the implementation of #559; the overflow and the margin
   found in its review, the per-row route in the PR's review).
6. **`QuantumLayer.backward` calls `qml.vjp`,** once per backward pass, with
   the incoming gradient as the cotangent, the same `wrt` as before and the same
   derived seeds (`[s, n, 1]`), and no longer `qml.jacobian`.
7. **In `polypus-qml`,** the product is a `Vjp` with **one row per sample**
   (`Jacobian::vjp`, and by the adjoint `JacobianBlock::vjp_runs` /
   `vjp_from_adjoint`, with `vjp_weight_bounds` to choose each sample's
   route); the binding sums the parameter rows in row order, so the sum does
   not depend on the blocks. `Jacobian` and `Vjp` share one private
   bookkeeping (where a gate's derivative goes in a row), so the two cannot
   drift apart.

## Alternatives considered

- **A `cotangent=` argument on `qml.jacobian`.** Rejected: one function whose
  return shape and meaning change with an optional argument (a Jacobian without
  it, a product with it) is harder to type and to document, and `qml.jacobian`
  would change for every caller.
- **Return the product per row** (`(rows, num_params)` for `"params"`).
  Rejected for the Python API: it is not the VJP a reverse-mode framework asks
  for, and every caller would sum it. The crate keeps the rows (decision 7),
  where they cost nothing and keep the sum independent of the blocks.
- **Reject a non-finite cotangent.** Rejected: the layer's backward pass
  propagated it through `einsum`, and a backward pass is not where to decide
  what a `NaN` gradient means; the caller's framework (PyTorch's anomaly
  detection, say) does.
- **For a non-finite cotangent in exact mode, the shifted circuits** (the
  memory fallback) instead of the per-observable adjoint runs (decision 5):
  simpler, but `2 · shifts` circuits per row instead of `K` runs; and the
  per-observable runs give the very Jacobian `qml.jacobian(exact=True)` gives.

## Consequences

- An exact backward pass of the layer costs `rows` adjoint runs instead of
  `rows · K` (`benchmarks/bench_vjp.py` measures the ratio as `K` grows); a
  sampled one is unchanged.
- The layer's gradients are the same to rounding, not bit for bit: the adjoint
  sums over one combined observable rather than per observable.
- `qml.vjp` is one more entry point with a `seed` (C-7) and one more check in
  C-11's order (the cotangent, after `params`).
- The cotangent takes the values `x` takes, no more: a string, `None` or
  another object is a `TypeError`, as in `x`, rather than what
  `numpy.asarray(..., "float64")` would make of it (`"0.5"` read as `0.5`,
  `None` as `NaN`). A complex cotangent is refused, rather than converted
  with its imaginary part dropped.

## Reopening criteria

- A caller needs the per-row parameter product from Python (a per-sample
  gradient, for differential privacy, say): expose the crate's rows.
- The native simulator learns to weigh non-finite observables: decision 5's
  detour can go.
- A sampled backend gains a cheaper estimator of the product than the
  contracted shift Jacobian.
