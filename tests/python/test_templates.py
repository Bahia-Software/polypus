"""
Cross-validation of the Python-exposed QFT template
(``polypus.circuits.templates.qft``) against Qiskit's *supported* synthesiser
``qiskit.synthesis.qft.synth_qft_full`` — and against two Qiskit-independent
analytic identities.

The Rust tests in ``crates/polypus-circuit/tests/templates.rs`` already pin the
template's algebra against a hand-built reference. This file covers a different
seam: that the **public Python API** (`qft(num_qubits, inverse, swaps)`) still
maps 1:1 onto `synth_qft_full(num_qubits, do_swaps=swaps, inverse=inverse)` for
every ``inverse × swaps`` combination. A drift in the binding signature, the
big-endian/swap convention, or the little-endian statevector readout would slip
past the Rust tests but is caught here.

``synth_qft_full`` is used deliberately in place of the deprecated
``qiskit.circuit.library.QFT`` class (removed in Qiskit 3.0). Reference
amplitudes come from Terra's exact ``Statevector`` (no Aer dependency, no
sampling). Both backends use the little-endian convention, so amplitude arrays
compare directly — same as ``test_statevector.py``.
"""

import itertools

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit.quantum_info import Statevector
from qiskit.synthesis.qft import synth_qft_full

import polypus

COMBOS = list(itertools.product([False, True], [False, True]))  # inverse × swaps


@pytest.mark.parametrize("inverse, swaps", COMBOS)
@pytest.mark.parametrize("n", [1, 4, 6])
def test_template_matches_qiskit_synth(n, inverse, swaps):
    """`polypus.circuits.templates.qft` == Qiskit `synth_qft_full`, statevector
    to statevector, for every inverse × swaps combination."""
    p_sv = np.asarray(
        polypus.statevector(polypus.circuits.templates.qft(n, inverse, swaps)),
        dtype=complex,
    )
    ref = Statevector(synth_qft_full(n, do_swaps=swaps, inverse=inverse)).data
    assert p_sv.shape == ref.shape, f"shape {p_sv.shape} != {ref.shape}"
    assert np.allclose(p_sv, ref, atol=1e-10), (
        f"QFT statevectors disagree (n={n}, inverse={inverse}, swaps={swaps})\n"
        f"max error = {np.max(np.abs(p_sv - ref)):.2e}"
    )


@pytest.mark.parametrize("n", [1, 3, 6])
def test_qft_of_zero_is_uniform_superposition(n):
    """Qiskit-independent: QFT|0…0⟩ has every amplitude equal to 1/√(2ⁿ)."""
    sv = np.asarray(polypus.statevector(polypus.circuits.templates.qft(n)))
    assert np.allclose(np.abs(sv), 1.0 / np.sqrt(2**n), atol=1e-10)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_inverse_undoes_forward_on_every_basis_state(n):
    """Qiskit-independent: QFT† · QFT = I on every computational-basis input.

    The composite circuit is assembled by splicing the template's *own* QASM
    (forward then inverse) behind an X-prepared basis state |k⟩, so nothing is
    reimplemented by hand; the result must return exactly |k⟩."""
    fwd = polypus.circuits.templates.qft(n, False, True).to_qasm2()
    inv = polypus.circuits.templates.qft(n, True, True).to_qasm2()

    skip = ("OPENQASM", "include", "qreg", "creg")
    header = [ln for ln in fwd.splitlines() if ln.startswith(("OPENQASM", "include", "qreg"))]

    def body(qasm):
        return [ln for ln in qasm.splitlines() if ln.strip() and not ln.startswith(skip)]

    fbody, ibody = body(fwd), body(inv)
    for k in range(2**n):
        xprep = [f"x q[{i}];" for i in range(n) if (k >> i) & 1]
        circ = polypus.Circuit.from_qasm2("\n".join(header + xprep + fbody + ibody))
        sv = np.asarray(polypus.statevector(circ))
        expected = np.zeros(2**n, dtype=complex)
        expected[k] = 1.0
        assert np.allclose(sv, expected, atol=1e-10), (
            f"QFT†·QFT != I for n={n}, basis state |{k}⟩; "
            f"max error {np.max(np.abs(sv - expected)):.2e}"
        )
