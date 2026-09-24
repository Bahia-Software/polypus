"""
Integration tests — run real quantum circuits on the local infrastructure
(AerSimulator unless a test selects the native backend).

These tests require qiskit-aer to be installed. They are marked with the
'integration' pytest mark and can be skipped in CI with:

    pytest -m "not integration"
"""

import pytest

pytestmark = pytest.mark.integration


class TestRunQuantumCircuitSingleQpu:
    """n_qpus=1 goes through AlgorithmSingleRun. The counts payload (a list with
    one counts dict) is exposed as ``RunResult.counts`` (contract C-7)."""

    def test_returns_list(self, bell_circuit):
        import polypus

        result = polypus.run_quantum_circuit(
            bell_circuit, shots=100, infrastructure="local"
        )
        assert isinstance(result.counts, list), (
            f"Expected list, got {type(result.counts)}"
        )

    def test_returns_one_element(self, bell_circuit):
        import polypus

        result = polypus.run_quantum_circuit(
            bell_circuit, shots=100, infrastructure="local"
        )
        assert len(result.counts) == 1

    def test_element_is_dict(self, bell_circuit):
        import polypus

        result = polypus.run_quantum_circuit(
            bell_circuit, shots=100, infrastructure="local"
        )
        assert isinstance(result.counts[0], dict)

    def test_only_bell_states(self, bell_circuit):
        """A Bell circuit can only produce '00' or '11'."""
        import polypus

        result = polypus.run_quantum_circuit(
            bell_circuit, shots=1000, infrastructure="local"
        )
        assert set(result.counts[0].keys()).issubset({"00", "11"}), (
            f"Unexpected bitstrings in Bell result: {result.counts[0].keys()}"
        )

    def test_total_shots(self, bell_circuit):
        import polypus

        shots = 512
        result = polypus.run_quantum_circuit(
            bell_circuit, shots=shots, infrastructure="local"
        )
        assert sum(result.counts[0].values()) == shots

    def test_both_bell_states_observed(self, bell_circuit):
        """With 1000 shots both '00' and '11' should appear."""
        import polypus

        result = polypus.run_quantum_circuit(
            bell_circuit, shots=1000, infrastructure="local"
        )
        counts = result.counts[0]
        assert "00" in counts and "11" in counts

    def test_manifest_defaults_for_aer(self, bell_circuit):
        """The manifest records the run metadata; the default Aer backend is
        seeded too (contract C-7), so an omitted seed still reports the
        fresh OS-entropy value that was actually used."""
        import polypus

        result = polypus.run_quantum_circuit(
            bell_circuit, shots=100, infrastructure="local"
        )
        assert result.backend == "aer"
        assert result.infrastructure == "local"
        assert isinstance(result.seed, int)


class TestRunQuantumCircuitMultipleQpus:
    """n_qpus>1 goes through DistributeByShotsRun, which merges partial results
    into a single counts dict, exposed as ``RunResult.counts``."""

    def test_distributed_returns_dict(self, bell_circuit):
        import polypus

        result = polypus.run_quantum_circuit(
            bell_circuit, shots=400, infrastructure="local", n_qpus=4
        )
        assert isinstance(result.counts, dict), (
            f"Expected merged dict for n_qpus>1, got {type(result.counts)}"
        )

    def test_distributed_only_bell_states(self, bell_circuit):
        import polypus

        result = polypus.run_quantum_circuit(
            bell_circuit, shots=400, infrastructure="local", n_qpus=4
        )
        assert set(result.counts.keys()).issubset({"00", "11"}), (
            f"Unexpected bitstrings in distributed Bell result: {result.counts.keys()}"
        )

    def test_distributed_total_shots(self, bell_circuit):
        import polypus

        shots = 400
        result = polypus.run_quantum_circuit(
            bell_circuit, shots=shots, infrastructure="local", n_qpus=4
        )
        assert sum(result.counts.values()) == shots

    def test_distributed_total_shots_not_divisible(self, bell_circuit):
        """Contract C-3: shots % n_qpus != 0 must still conserve the total.
        1000 / 3 leaves a remainder of 1; the old `shots /= n_qpus` ran 999."""
        import polypus

        shots = 1000
        result = polypus.run_quantum_circuit(
            bell_circuit, shots=shots, infrastructure="local", n_qpus=3
        )
        assert sum(result.counts.values()) == shots

    def test_distributed_total_shots_fewer_than_qpus(self, bell_circuit):
        """Contract C-3 degenerate case: shots < n_qpus. 5 shots on 8 QPUs must
        run exactly 5 shots (one-per-QPU on the first 5), not 0 (5 // 8 == 0)."""
        import polypus

        shots = 5
        result = polypus.run_quantum_circuit(
            bell_circuit, shots=shots, infrastructure="local", n_qpus=8
        )
        assert sum(result.counts.values()) == shots


class TestCountsKeyOrder:
    """Contract C-3: every counts dict lists its keys in ascending bitstring
    order, so a float reduction over ``.items()`` adds its terms in the same
    order in every call and every process."""

    @pytest.mark.parametrize("n_qpus", [1, 4])
    def test_keys_in_ascending_bitstring_order(self, n_qpus):
        import polypus

        qc = polypus.Circuit(4).h(0).h(1).h(2).h(3).measure_all()
        result = polypus.run_quantum_circuit(
            qc,
            shots=4000,
            infrastructure="local",
            backend="polypus",
            n_qpus=n_qpus,
            seed=5,
        )
        counts = result.counts[0] if n_qpus == 1 else result.counts
        # With all 16 outcomes present, a random order passes with probability 1/16!.
        assert len(counts) == 16
        assert list(counts) == sorted(counts)
