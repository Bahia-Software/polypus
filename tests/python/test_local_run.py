"""
Integration tests — run real quantum circuits using the local AerSimulator backend.

These tests require qiskit-aer to be installed. They are marked with the
'integration' pytest mark and can be skipped in CI with:

    pytest -m "not integration"
"""

import pytest


pytestmark = pytest.mark.integration


class TestRunQuantumCircuitSingleQpu:
    """n_qpus=1 goes through AlgorithmSingleRun which returns the raw runner
    output: a list with one counts dict."""

    def test_returns_list(self, bell_circuit):
        import polypus
        result = polypus.run_quantum_circuit(bell_circuit, shots=100, infrastructure="local")
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_returns_one_element(self, bell_circuit):
        import polypus
        result = polypus.run_quantum_circuit(bell_circuit, shots=100, infrastructure="local")
        assert len(result) == 1

    def test_element_is_dict(self, bell_circuit):
        import polypus
        result = polypus.run_quantum_circuit(bell_circuit, shots=100, infrastructure="local")
        assert isinstance(result[0], dict)

    def test_only_bell_states(self, bell_circuit):
        """A Bell circuit can only produce '00' or '11'."""
        import polypus
        result = polypus.run_quantum_circuit(bell_circuit, shots=1000, infrastructure="local")
        assert set(result[0].keys()).issubset({"00", "11"}), (
            f"Unexpected bitstrings in Bell result: {result[0].keys()}"
        )

    def test_total_shots(self, bell_circuit):
        import polypus
        shots = 512
        result = polypus.run_quantum_circuit(bell_circuit, shots=shots, infrastructure="local")
        assert sum(result[0].values()) == shots

    def test_both_bell_states_observed(self, bell_circuit):
        """With 1000 shots both '00' and '11' should appear."""
        import polypus
        result = polypus.run_quantum_circuit(bell_circuit, shots=1000, infrastructure="local")
        counts = result[0]
        assert "00" in counts and "11" in counts


class TestRunQuantumCircuitMultipleQpus:
    """n_qpus>1 goes through DistributeByShotsRun which merges partial results
    into a single counts dict."""

    def test_distributed_returns_dict(self, bell_circuit):
        import polypus
        result = polypus.run_quantum_circuit(
            bell_circuit, shots=400, infrastructure="local", n_qpus=4
        )
        assert isinstance(result, dict), f"Expected merged dict for n_qpus>1, got {type(result)}"

    def test_distributed_only_bell_states(self, bell_circuit):
        import polypus
        result = polypus.run_quantum_circuit(
            bell_circuit, shots=400, infrastructure="local", n_qpus=4
        )
        assert set(result.keys()).issubset({"00", "11"}), (
            f"Unexpected bitstrings in distributed Bell result: {result.keys()}"
        )

    def test_distributed_total_shots(self, bell_circuit):
        import polypus
        shots = 400
        result = polypus.run_quantum_circuit(
            bell_circuit, shots=shots, infrastructure="local", n_qpus=4
        )
        assert sum(result.values()) == shots

    def test_distributed_total_shots_not_divisible(self, bell_circuit):
        """Contract C-3: shots % n_qpus != 0 must still conserve the total.
        1000 / 3 leaves a remainder of 1; the old `shots /= n_qpus` ran 999."""
        import polypus
        shots = 1000
        result = polypus.run_quantum_circuit(
            bell_circuit, shots=shots, infrastructure="local", n_qpus=3
        )
        assert sum(result.values()) == shots

    def test_distributed_total_shots_fewer_than_qpus(self, bell_circuit):
        """Contract C-3 degenerate case: shots < n_qpus. 5 shots on 8 QPUs must
        run exactly 5 shots (one-per-QPU on the first 5), not 0 (5 // 8 == 0)."""
        import polypus
        shots = 5
        result = polypus.run_quantum_circuit(
            bell_circuit, shots=shots, infrastructure="local", n_qpus=8
        )
        assert sum(result.values()) == shots
