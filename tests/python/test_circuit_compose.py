import numpy as np
import polypus
import pytest
from polypus.circuits.templates import qft


class TestCompose:
    def test_maps_qubits_and_returns_self(self):
        qc = polypus.Circuit(3)
        assert qc.compose(polypus.Circuit(2).h(0).cx(0, 1), qubits=[2, 0]) is qc
        assert "cx q[2],q[0];" in qc.to_qasm2()

    def test_default_map_is_the_leading_qubits(self):
        qc = polypus.Circuit(3).compose(polypus.Circuit(2).x(1))
        assert "x q[1];" in qc.to_qasm2()

    def test_a_template_on_a_subregister_matches_building_it_in_place(self):
        composed = polypus.Circuit(3).x(0).compose(qft(2), qubits=[1, 2])
        manual = polypus.Circuit(3).x(0).h(2).cp(1, 2, np.pi / 2).h(1).swap(1, 2)
        np.testing.assert_allclose(
            polypus.statevector(composed), polypus.statevector(manual), atol=1e-12
        )

    def test_compose_with_itself_appends_a_copy(self):
        qc = polypus.Circuit(1).x(0)
        qc.compose(qc)
        assert repr(qc) == "Circuit(num_qubits=1, num_params=0, gates=2)"

    @pytest.mark.parametrize("qubits", [[0], [0, 3], [1, 1]])
    def test_an_invalid_map_raises_value_error(self, qubits):
        with pytest.raises(ValueError):
            polypus.Circuit(3).compose(polypus.Circuit(2).cx(0, 1), qubits=qubits)

    def test_a_negative_qubit_raises_overflow_error(self):
        with pytest.raises(OverflowError):
            polypus.Circuit(2).compose(polypus.Circuit(1).x(0), qubits=[-1])

    def test_a_non_circuit_raises_type_error(self):
        with pytest.raises(TypeError):
            polypus.Circuit(2).compose("h q[0];")

    def test_failed_compose_leaves_the_circuit_unchanged(self):
        qc = polypus.Circuit(2).h(0).measure(1, 0)
        before = qc.to_qasm2()
        with pytest.raises(ValueError, match="measured"):
            qc.compose(polypus.Circuit(2).x(0).x(1), qubits=[0, 1])
        assert qc.to_qasm2() == before

    def test_composing_a_circuit_mid_call_raises_runtime_error(self):
        qc = polypus.Circuit(1)

        # `qc.x` holds `qc` while it reads its argument's `__index__`.
        class ComposesFrom:
            def __index__(self):
                polypus.Circuit(1).compose(qc)
                return 0

        class ComposesInto:
            def __index__(self):
                qc.compose(polypus.Circuit(1))
                return 0

        for qubit in (ComposesFrom(), ComposesInto()):
            with pytest.raises(RuntimeError):
                qc.x(qubit)
        assert repr(qc) == "Circuit(num_qubits=1, num_params=0, gates=0)"

    def test_docstring_example_runs(self):
        doc = polypus.Circuit.compose.__doc__
        exec(doc.split("```python\n", 1)[1].split("```", 1)[0], {})
