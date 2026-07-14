"""
QASM 2.0 import (round-trip) and Aer batching tests.

- polypus.Circuit.from_qasm2: inverse of to_qasm2, including Qiskit interop
  (qasm2.dumps output) and parse errors with line info.
- Batching: polypus.train on the local backend must submit each evaluation
  batch in a single AerSimulator.run call (list of circuits), not one call
  per circuit.
"""

import math

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Circuit.from_qasm2 (no backend required)
# ─────────────────────────────────────────────────────────────────────────────


class TestQasmImport:
    def test_roundtrip_is_byte_stable(self):
        import polypus

        qc = (
            polypus.Circuit(3)
            .h(0)
            .h(1)
            .h(2)
            .rzz(0, 1, polypus.Param(0))
            .rx(2, polypus.Param(1))
            .u(0, 0.1, 0.2, 0.3)
            .barrier()
            .measure_all()
        )
        qasm1 = qc.to_qasm2([0.4, 0.8])
        imported = polypus.Circuit.from_qasm2(qasm1)
        assert imported.to_qasm2() == qasm1

    def test_import_is_fully_concrete(self):
        import polypus

        qasm = polypus.Circuit(1).ry(0, polypus.Param(0)).to_qasm2([math.pi])
        imported = polypus.Circuit.from_qasm2(qasm)
        assert imported.num_params == 0
        assert f"ry({math.pi:.12f})" in imported.to_qasm2()

    def test_imports_qiskit_dumps_output(self):
        import polypus
        from qiskit import QuantumCircuit, qasm2

        qk = QuantumCircuit(2)
        qk.h(0)
        qk.cx(0, 1)
        qk.rzz(0.4, 0, 1)
        qk.measure_all()
        imported = polypus.Circuit.from_qasm2(qasm2.dumps(qk))
        assert imported.num_qubits == 2
        assert imported.num_clbits == 2
        # Re-exported QASM must be Qiskit-parseable again.
        back = QuantumCircuit.from_qasm_str(imported.to_qasm2())
        assert back.num_qubits == 2

    def test_imported_circuit_extends_with_builder(self):
        import polypus

        qasm = polypus.Circuit(2).h(0).cx(0, 1).to_qasm2()
        qc = polypus.Circuit.from_qasm2(qasm).rz(1, 0.5).measure_all()
        out = qc.to_qasm2()
        assert "rz(0.500000000000) q[1];" in out
        assert out.endswith("measure q -> c;\n")

    def test_parse_error_carries_line_number(self):
        import polypus

        src = "OPENQASM 2.0;\nqreg q[2];\nccz q[0],q[1];\n"
        with pytest.raises(ValueError, match=r"line 3.*unsupported gate 'ccz'"):
            polypus.Circuit.from_qasm2(src)

    def test_rejects_qasm3(self):
        import polypus

        with pytest.raises(ValueError, match="version 2.0"):
            polypus.Circuit.from_qasm2("OPENQASM 3.0;\n")

    def test_rejects_out_of_range_index(self):
        import polypus

        with pytest.raises(ValueError, match="out of range"):
            polypus.Circuit.from_qasm2("OPENQASM 2.0;\nqreg q[2];\nh q[7];\n")


@pytest.mark.integration
class TestRunImportedCircuit:
    def test_imported_circuit_runs_like_original(self):
        """from_qasm2(Qiskit Bell) executes with the expected distribution."""
        import polypus
        from qiskit import QuantumCircuit, qasm2

        qk = QuantumCircuit(2)
        qk.h(0)
        qk.cx(0, 1)
        qk.measure_all()
        imported = polypus.Circuit.from_qasm2(qasm2.dumps(qk))
        result = polypus.run_quantum_circuit(
            imported, shots=1000, infrastructure="local"
        )
        counts = result[0]
        assert set(counts.keys()).issubset({"00", "11"})
        assert sum(counts.values()) == 1000
        assert abs(counts.get("00", 0) - 500) < 150


# ─────────────────────────────────────────────────────────────────────────────
# Aer batching — every evaluation batch must be one simulator call
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestAerBatching:
    def test_train_submits_population_in_single_aer_calls(self, monkeypatch):
        """DE with population P must call AerSimulator.run with P circuits at
        once (local backend has no QPU cap), never one circuit per call."""
        import polypus
        import polypus_python.local as local_mod

        calls = []  # batch size of every AerSimulator.run invocation
        real_sim = local_mod.AerSimulator

        class SpySimulator(real_sim):
            def run(self, circuits, **kwargs):
                calls.append(len(circuits) if isinstance(circuits, list) else 1)
                return super().run(circuits, **kwargs)

        monkeypatch.setattr(local_mod, "AerSimulator", SpySimulator)

        population, generations = 8, 2
        qc = polypus.Circuit(2).ry(0, polypus.Param(0)).cx(0, 1).measure_all()
        polypus.train(
            qc,
            polypus.DE(
                generations=generations, population_size=population, tolerance=1e-12
            ),
            shots=128,
            n_qpus=1,
            dimensions=1,
            expectation_function=lambda b: float(b.count("1")),
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id="batching_spy",
        )

        assert calls, "spy never saw an AerSimulator.run call"
        # Each evaluation batch (initial population + one per generation)
        # must arrive as ONE call with the whole population.
        assert all(size == population for size in calls), calls
        assert len(calls) <= generations + 1, calls
