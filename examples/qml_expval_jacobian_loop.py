"""A training loop of your own over a native ``polypus.qml.Model``.

``Model.fit`` hands the whole optimization to Polypus. This script writes the
loop by hand instead, from the two loss-free primitives:

- ``qml.expval(model, x, params)``: every readout observable's expectation per
  row, shape ``(rows, observables)``;
- ``qml.jacobian(model, x, params)``: their derivatives by the parameter-shift
  rule, shape ``(rows, observables, num_params)``, or with respect to the
  inputs (``wrt="inputs"``), or both (``wrt="both"``).

The loss is yours, so is its chain rule: here a squared error
``mean((⟨Z₀⟩ − y)²)``, whose gradient is ``2·mean((⟨Z₀⟩ − y)·∂⟨Z₀⟩/∂θ)``, fed to
plain gradient descent in NumPy. With ``exact=True`` there is no shot noise and
the start is seeded, so the run is deterministic.

Run it directly::

    python examples/qml_expval_jacobian_loop.py

It prints the loss before and after, and fails if it did not go down.
"""

import time

import numpy as np
import numpy.typing as npt
from polypus import qml

# Two features per row; the target is +1 for small angles and -1 for large.
X = np.array([[0.2, 0.1], [0.4, 0.3], [0.1, 0.5], [2.7, 2.9], [3.0, 2.6], [2.8, 3.1]])
Y = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

STEPS = 40
LEARNING_RATE = 0.3


def loss(values: npt.NDArray[np.float64]) -> float:
    """Squared error of the first observable against ``Y``."""
    return float(np.mean((values[:, 0] - Y) ** 2))


def main() -> None:
    model = qml.Model(
        qml.Encoder(2).angle_encoder("ry"),
        qml.Ansatz(2).real_amplitudes(1),
        qml.Readout([qml.Z(0)]),
    )
    params = np.random.default_rng(7).uniform(-np.pi, np.pi, model.num_params)

    start = time.perf_counter()
    initial = loss(qml.expval(model, X, params, exact=True))
    for _ in range(STEPS):
        values = qml.expval(model, X, params, exact=True)
        by_params = qml.jacobian(model, X, params, exact=True)
        # Chain rule: ∂loss/∂θ = 2·mean((⟨Z₀⟩ − y) · ∂⟨Z₀⟩/∂θ).
        gradient = 2 * (values[:, 0] - Y) @ by_params[:, 0, :] / len(Y)
        params = params - LEARNING_RATE * gradient
    final = loss(qml.expval(model, X, params, exact=True))
    elapsed = time.perf_counter() - start

    # How the trained expectation moves with each input feature of each row.
    _, by_inputs = qml.jacobian(model, X, params, wrt="both", exact=True)

    print(f"loss: {initial:.4f} -> {final:.4f} in {STEPS} steps ({elapsed:.2f} s)")
    print("expectations:", np.round(qml.expval(model, X, params, exact=True)[:, 0], 3))
    print("d<Z0>/dx per row:", np.round(by_inputs[:, 0, :], 3).tolist())
    assert final < initial, f"the loss did not go down: {initial} -> {final}"


if __name__ == "__main__":
    main()
