"""
Shared builders for the native QML tests (``test_qml_native*.py``).

A plain module, not a test file: imported as ``from qml_native_helpers import
...``, like ``fake_cunqa``.
"""

# Two separable classes on one feature: ry(x) puts small x near |0> (⟨Z⟩ ≈ +1)
# and large x near |1> (⟨Z⟩ ≈ −1).
X = [[0.1], [0.3], [2.9], [3.0], [0.2], [2.8]]
Y = [1, 1, -1, -1, 1, -1]


def make_model(num_qubits=1, readout=None, encoder=None, ansatz=None):
    """An angle-encoder + RealAmplitudes model reading ``⟨Z₀⟩`` with a sign
    decision, each piece replaceable."""
    from polypus.qml import Ansatz, Encoder, Model, Readout, Z

    return Model(
        encoder or Encoder(num_qubits).angle_encoder("ry"),
        ansatz or Ansatz(num_qubits).real_amplitudes(1),
        readout or Readout([Z(0)], decision="sign"),
    )
