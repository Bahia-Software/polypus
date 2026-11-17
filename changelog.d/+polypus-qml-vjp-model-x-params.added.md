`polypus.qml.vjp(model, x, params, cotangent, *, wrt="params", ...)`, the
vector-Jacobian product of `qml.jacobian` with a `(rows, observables)`
cotangent, what a reverse-mode backward pass needs: `(num_params,)` summed
over the rows for `wrt="params"`, `(rows, features)` for `"inputs"`, the
tuple for `"both"`. With `exact=True` it takes one adjoint run per row
whatever the number of observables (`qml.jacobian` takes one per row and
observable), so the exact backward pass of `polypus.torch.QuantumLayer`,
which now calls it, is up to `K` times cheaper for a `K`-observable readout;
with shots it runs `qml.jacobian`'s circuits and gives its contraction for
the same `seed`. The layer's gradients move in the last bits, no more:
exactly, they come from one combined observable instead of one run per
observable, and with shots the parameter product is summed in row order.
The cotangent's shape is checked right after `params` (a
`polypus.ConfigurationError`, as is a complex one) and its values are read
as `x`'s are (a string or `None` is the same `TypeError`); a `NaN` or an
infinity in it propagates as through the contraction, and changes only its
own row's route. `qml.jacobian` does not change. In `polypus-qml`:
`Jacobian::vjp`, `JacobianBlock::vjp_runs`/`num_vjp_runs`/
`vjp_weight_bounds`/`vjp_from_adjoint` (and
`ExpectationJacobian`'s), `Vjp` and `DiagonalObservable::weight_bound`.
`benchmarks/bench_vjp.py` measures it (#559, ADR 0008).
