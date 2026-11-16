`polypus.qml.expval(model, x, params, *, ...)` and
`polypus.qml.jacobian(model, x, params, *, wrt="params", ...)` for a training
loop of your own over a native `Model`: the expectation of every readout
observable per row as a `float64` NumPy array `(rows, observables)`, and its
exact parameter-shift Jacobian with respect to the parameters
(`(rows, observables, num_params)`), the inputs of angle and IQP encoders
(`(rows, observables, features)`) or both (`wrt="both"`, a tuple). They take
`TrainedModel.run`'s execution arguments, check everything before a backend
exists, and run the rows in blocks over one backend per call. In
`polypus-qml`: `CompiledModel::jacobian` with `ExpectationJacobian`,
`JacobianBlock`, `Jacobian` and `JacobianWrt`,
`CompiledModel::expectations_from_counts`/`expectations_from_probabilities`
and `num_observables`, and `QmlError::NoInputGradient`. In
`polypus-evaluation`: `window_size` and `candidate_window_size`, the
`QmlOracle`'s window policy, now public, which also sizes the primitives'
blocks. See `examples/qml_expval_jacobian_loop.py`.
