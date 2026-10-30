"""
``polypus.circuits.templates`` — the circuit templates the extension ships.

The ``qft`` and ``shor_order_finding`` docstrings carry usage examples; they
are executed here so the examples keep working as the entry points they call
evolve.
"""

import re

import pytest


def _python_example(doc):
    """The body of the single ```python block in a docstring."""
    blocks = re.findall(r"```python\n(.*?)```", doc or "", re.S)
    assert len(blocks) == 1, f"expected one ```python example, found {len(blocks)}"
    return blocks[0]


class TestQftDocstringExample:
    def test_example_runs_and_conserves_shots(self):
        import polypus

        namespace = {}
        exec(_python_example(polypus.circuits.templates.qft.__doc__), namespace)
        result = namespace["result"]
        assert isinstance(result, polypus.RunResult)
        assert sum(namespace["counts"].values()) == 1024


class TestQftSize:
    def test_beyond_the_largest_qft_raises_value_error(self):
        import polypus

        with pytest.raises(ValueError, match="at most 2827 qubits, got 2828"):
            polypus.circuits.templates.qft(2828)


class TestShorOrderFinding:
    def test_docstring_example_runs_and_conserves_shots(self):
        import polypus

        namespace = {}
        exec(
            _python_example(polypus.circuits.templates.shor_order_finding.__doc__),
            namespace,
        )
        assert sum(namespace["counts"].values()) == 1024

    def test_size_for_15(self):
        from polypus.circuits.templates import shor_order_finding

        assert (
            repr(shor_order_finding(7, 15))
            == "Circuit(num_qubits=18, num_params=0, gates=8561)"
        )

    def test_counting_register_peaks_at_multiples_of_64(self):
        import numpy as np
        import polypus
        from polypus.circuits.templates import shor_order_finding

        probs = np.abs(polypus.statevector(shor_order_finding(7, 15))) ** 2
        marginal = np.bincount(
            np.arange(probs.size) & 0xFF, weights=probs, minlength=256
        )
        np.testing.assert_allclose(marginal[[0, 64, 128, 192]], 0.25, atol=1e-9)

    # The limits the docs state: 24-bit moduli build, and to_qasm3 exports
    # 16-bit ones but not 17-bit ones.
    def test_the_largest_modulus_builds(self):
        from polypus.circuits.templates import shor_order_finding

        assert (
            repr(shor_order_finding(2, 2**24 - 3))
            == "Circuit(num_qubits=98, num_params=0, gates=3716641)"
        )

    def test_to_qasm3_exports_up_to_16_bit_moduli(self):
        from polypus.circuits.templates import shor_order_finding

        assert shor_order_finding(2, 2**16 - 1).to_qasm3().startswith("OPENQASM 3")
        with pytest.raises(ValueError, match="MAX_SOURCE_BYTES"):
            shor_order_finding(2, 2**16 + 1).to_qasm3()

    @pytest.mark.parametrize(
        "args", [(2, 2), (2, 2**24 + 1), (0, 15), (6, 15), (7, 15, 0), (7, 15, 9)]
    )
    def test_invalid_arguments_raise_value_error(self, args):
        from polypus.circuits.templates import shor_order_finding

        with pytest.raises(ValueError):
            shor_order_finding(*args)

    @pytest.mark.parametrize("args", [(-7, 15), (7, -15), (7, 2**64), (7, 15, -1)])
    def test_out_of_range_integers_raise_overflow_error(self, args):
        from polypus.circuits.templates import shor_order_finding

        with pytest.raises(OverflowError):
            shor_order_finding(*args)
