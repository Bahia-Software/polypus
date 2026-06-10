"""
Native circuit path tests — verify that polypus.Circuit (pure-Rust circuits)
works end-to-end through the same entry points as Qiskit circuits:

- polypus.Circuit / polypus.Param construction and QASM export
- polypus.run_quantum_circuit with Circuit, QASM string, and Qiskit inputs
- polypus.train(DE/PSO/QNG) with a native parameterized circuit
- statistical equivalence between the native and Qiskit paths
- early validation errors (unbound params, dimension mismatch)
"""

import math

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Construction + QASM export (no backend required)
# ─────────────────────────────────────────────────────────────────────────────


class TestCircuitConstruction:
    def test_builder_chains(self):
        import polypus
        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        assert qc.num_qubits == 2
        assert qc.num_params == 0
        assert len(qc) == 3

    def test_param_tracking(self):
        import polypus
        qc = polypus.Circuit(2).rx(0, polypus.Param(0)).rzz(0, 1, polypus.Param(1))
        assert qc.num_params == 2

    def test_fixed_angles_accept_ints_and_floats(self):
        import polypus
        qc = polypus.Circuit(1).rx(0, 1).ry(0, 0.5)
        assert qc.num_params == 0

    def test_to_qasm2_header_and_gates(self):
        import polypus
        qasm = polypus.Circuit(2).h(0).cx(0, 1).measure_all().to_qasm2()
        assert qasm.startswith('OPENQASM 2.0;\ninclude "qelib1.inc";')
        assert "h q[0];" in qasm
        assert "cx q[0],q[1];" in qasm
        assert "measure q -> c;" in qasm

    def test_to_qasm2_binds_params(self):
        import polypus
        qc = polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()
        qasm = qc.to_qasm2([math.pi])
        assert f"ry({math.pi:.12f}) q[0];" in qasm

    def test_qiskit_parses_generated_qasm(self):
        from qiskit import QuantumCircuit
        import polypus
        qasm = (
            polypus.Circuit(3)
            .h(0).h(1).h(2)
            .rzz(0, 1, polypus.Param(0))
            .rx(2, polypus.Param(1))
            .measure_all()
            .to_qasm2([0.4, 0.8])
        )
        qc = QuantumCircuit.from_qasm_str(qasm)
        assert qc.num_qubits == 3
        assert qc.num_clbits == 3

    def test_qubit_out_of_range_raises_valueerror(self):
        import polypus
        with pytest.raises(ValueError, match="out of range"):
            polypus.Circuit(2).h(5)

    def test_identical_qubits_raises_valueerror(self):
        import polypus
        with pytest.raises(ValueError, match="distinct"):
            polypus.Circuit(2).cx(1, 1)

    def test_wrong_param_count_raises_valueerror(self):
        import polypus
        qc = polypus.Circuit(1).rx(0, polypus.Param(0))
        with pytest.raises(ValueError, match="wrong number of parameter"):
            qc.to_qasm2([0.1, 0.2])

    def test_bad_angle_type_raises_typeerror(self):
        import polypus
        with pytest.raises(TypeError):
            polypus.Circuit(1).rx(0, "not-an-angle")

    def test_param_repr_and_index(self):
        import polypus
        p = polypus.Param(3)
        assert p.index == 3
        assert repr(p) == "Param(3)"


# ─────────────────────────────────────────────────────────────────────────────
# run_quantum_circuit with native inputs
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def native_bell_circuit():
    import polypus
    return polypus.Circuit(2).h(0).cx(0, 1).measure_all()


@pytest.mark.integration
class TestRunNativeCircuit:
    def test_native_single_qpu(self, native_bell_circuit):
        import polypus
        result = polypus.run_quantum_circuit(
            native_bell_circuit, shots=1000, infrastructure="local"
        )
        assert isinstance(result, list) and len(result) == 1
        counts = result[0]
        assert set(counts.keys()).issubset({"00", "11"})
        assert sum(counts.values()) == 1000

    def test_native_distributed(self, native_bell_circuit):
        import polypus
        result = polypus.run_quantum_circuit(
            native_bell_circuit, shots=400, infrastructure="local", n_qpus=4
        )
        assert isinstance(result, dict)
        assert set(result.keys()).issubset({"00", "11"})
        assert sum(result.values()) == 400

    def test_qasm_string_input(self, native_bell_circuit):
        import polypus
        qasm = native_bell_circuit.to_qasm2()
        result = polypus.run_quantum_circuit(qasm, shots=500, infrastructure="local")
        counts = result[0]
        assert set(counts.keys()).issubset({"00", "11"})
        assert sum(counts.values()) == 500

    def test_unbound_circuit_raises(self):
        import polypus
        qc = polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()
        with pytest.raises(ValueError, match="unbound parameters"):
            polypus.run_quantum_circuit(qc, shots=100, infrastructure="local")

    def test_native_matches_qiskit_distribution(self, native_bell_circuit, bell_circuit):
        """Same Bell circuit, both paths: distributions must agree (~50/50)."""
        import polypus
        shots = 4000
        native = polypus.run_quantum_circuit(
            native_bell_circuit, shots=shots, infrastructure="local"
        )[0]
        qiskit = polypus.run_quantum_circuit(
            bell_circuit, shots=shots, infrastructure="local"
        )[0]
        for key in ("00", "11"):
            p_native = native.get(key, 0) / shots
            p_qiskit = qiskit.get(key, 0) / shots
            assert abs(p_native - p_qiskit) < 0.08, (
                f"{key}: native={p_native:.3f} vs qiskit={p_qiskit:.3f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# train() with native circuits
# ─────────────────────────────────────────────────────────────────────────────

_SHOTS = 256
_TRAIN_KW = dict(
    shots=_SHOTS,
    n_qpus=1,
    dimensions=1,
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
)


@pytest.fixture
def native_parametrized_circuit():
    """Native equivalent of the Qiskit `parametrized_circuit` fixture."""
    import polypus
    return polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()


@pytest.mark.integration
@pytest.mark.vqc
class TestTrainNativeCircuit:
    def test_train_de(self, native_parametrized_circuit, simple_expectation_fn):
        import polypus
        result = polypus.train(
            native_parametrized_circuit,
            polypus.DE(generations=2, population_size=4, tolerance=0.5),
            expectation_function=simple_expectation_fn,
            id="test_native_de",
            **_TRAIN_KW,
        )
        assert isinstance(result, list) and len(result) == 1
        assert all(isinstance(v, float) for v in result)

    def test_train_pso(self, native_parametrized_circuit, simple_expectation_fn):
        import polypus
        result = polypus.train(
            native_parametrized_circuit,
            polypus.PSO(generations=2, population_size=4, bounds=(0.0, math.pi)),
            expectation_function=simple_expectation_fn,
            id="test_native_pso",
            **_TRAIN_KW,
        )
        assert isinstance(result, list) and len(result) == 1

    def test_train_qng(self, native_parametrized_circuit, simple_expectation_fn, simple_variance_fn):
        import polypus
        result = polypus.train(
            native_parametrized_circuit,
            polypus.QNG(simple_variance_fn, max_iters=2, bounds=(0.0, math.pi)),
            expectation_function=simple_expectation_fn,
            id="test_native_qng",
            **_TRAIN_KW,
        )
        assert isinstance(result, list) and len(result) == 1

    def test_dimension_mismatch_raises_before_training(self, native_parametrized_circuit, simple_expectation_fn):
        import polypus
        kwargs = dict(_TRAIN_KW)
        kwargs["dimensions"] = 3  # circuit has 1 free param
        with pytest.raises(ValueError, match="does not match"):
            polypus.train(
                native_parametrized_circuit,
                polypus.DE(generations=2, population_size=4),
                expectation_function=simple_expectation_fn,
                id="test_native_dim_mismatch",
                **kwargs,
            )

    def test_native_de_converges_like_qiskit(
        self, native_parametrized_circuit, parametrized_circuit, simple_expectation_fn
    ):
        """RY(θ) with all-ones objective drives θ → π on both paths."""
        import polypus
        method = lambda: polypus.DE(generations=10, population_size=10, tolerance=1e-9)
        kwargs = dict(_TRAIN_KW)
        kwargs["shots"] = 512

        theta_native = polypus.train(
            native_parametrized_circuit, method(),
            expectation_function=simple_expectation_fn,
            id="test_native_conv", **kwargs,
        )[0]
        theta_qiskit = polypus.train(
            parametrized_circuit, method(),
            expectation_function=simple_expectation_fn,
            id="test_qiskit_conv", **kwargs,
        )[0]

        # Both must approach θ = π (probability of |1⟩ maximised).
        p1 = math.sin(theta_native / 2) ** 2
        p1_qiskit = math.sin(theta_qiskit / 2) ** 2
        assert p1 > 0.9, f"native path did not converge: θ={theta_native:.3f}, p1={p1:.3f}"
        assert p1_qiskit > 0.9, f"qiskit path did not converge: θ={theta_qiskit:.3f}"
