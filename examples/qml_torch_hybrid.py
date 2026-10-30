"""A hybrid quantum-classical network in PyTorch over a native ``qml.Model``.

``polypus.torch.QuantumLayer`` turns a native model into a ``torch.nn.Module``:
its forward pass is ``qml.expval`` (every readout observable, one column each)
and its backward pass ``qml.jacobian``, so it sits in a ``torch.nn.Sequential``
between classical layers and trains with any ``torch.optim`` optimizer.

Here a ``Linear`` layer maps four features onto the angles of two qubits, the
quantum layer reads ``⟨Z₀⟩`` and ``⟨Z₁⟩``, and a second ``Linear`` turns them
into one logit, trained with Adam on a binary cross-entropy. The classical
layers stay ``float32``: the quantum layer converts at its boundary. With
``exact=True`` there is no shot noise and ``torch.manual_seed`` fixes the start,
so the run is deterministic.

Needs PyTorch: ``pip install "polypus-quantum[torch]"``. Run it directly::

    python examples/qml_torch_hybrid.py

It prints the loss and accuracy before and after, and fails if they did not
improve.
"""

import time

import torch
from polypus import qml
from polypus.torch import QuantumLayer

# Two classes in four features: the first two are large for class 1, the last
# two for class 0.
X = torch.tensor(
    [
        [0.9, 0.8, 0.1, 0.0],
        [1.0, 0.6, 0.2, 0.1],
        [0.8, 0.9, 0.0, 0.2],
        [0.7, 1.0, 0.1, 0.1],
        [0.1, 0.2, 0.9, 1.0],
        [0.0, 0.1, 1.0, 0.8],
        [0.2, 0.0, 0.8, 0.9],
        [0.1, 0.1, 0.7, 1.0],
    ]
)
Y = torch.tensor([[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])

STEPS = 30
LEARNING_RATE = 0.1


def evaluate(net: torch.nn.Module, loss_fn: torch.nn.Module) -> tuple[float, float]:
    """The loss and the accuracy on ``X``, without a gradient."""
    with torch.no_grad():
        logits = net(X)
    accuracy = ((logits > 0).float() == Y).float().mean().item()
    return loss_fn(logits, Y).item(), accuracy


def main() -> None:
    torch.manual_seed(3)
    model = qml.Model(
        qml.Encoder(2).angle_encoder("ry"),
        qml.Ansatz(2).real_amplitudes(1),
        qml.Readout([qml.Z(0), qml.Z(1)]),
    )
    quantum = QuantumLayer(model, exact=True)
    net = torch.nn.Sequential(
        torch.nn.Linear(4, 2),
        quantum,
        torch.nn.Linear(quantum.num_observables, 1),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    start = time.perf_counter()
    initial_loss, initial_accuracy = evaluate(net, loss_fn)
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = loss_fn(net(X), Y)
        # One qml.expval call forward, one qml.jacobian call backward.
        loss.backward()
        optimizer.step()
    final_loss, final_accuracy = evaluate(net, loss_fn)
    elapsed = time.perf_counter() - start

    print(
        f"loss: {initial_loss:.4f} -> {final_loss:.4f}, accuracy: "
        f"{initial_accuracy:.2f} -> {final_accuracy:.2f} in {STEPS} steps "
        f"({elapsed:.2f} s)"
    )
    print("quantum parameters:", [round(p, 3) for p in quantum.params.tolist()])
    assert final_loss < initial_loss, (
        f"the loss did not go down: {initial_loss} -> {final_loss}"
    )
    assert final_accuracy > initial_accuracy or final_accuracy == 1.0, (
        f"the accuracy did not improve: {initial_accuracy} -> {final_accuracy}"
    )


if __name__ == "__main__":
    main()
