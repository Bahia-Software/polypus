"""
Backend selection tests — the `backend` parameter chooses the local *device*:

- ``backend="aer"`` (default): Qiskit Aer simulator (Python).
- ``backend="polypus"``: the pure-Rust ``polypus-sim`` statevector backend,
  which consumes native circuits directly (no OpenQASM round-trip, no GIL).

These tests verify both paths are reachable through the same entry points
(``run_quantum_circuit`` and ``train``), that the native backend produces
Aer-compatible counts, and that invalid combinations are rejected early.
"""

import math

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# run_quantum_circuit with the native backend
# ─────────────────────────────────────────────────────────────────────────────


class TestRunQuantumCircuitBackends:
    def test_native_bell_only_correlated_outcomes(self):
        import polypus

        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        result = polypus.run_quantum_circuit(
            qc, shots=2000, infrastructure="local", backend="polypus"
        )
        assert isinstance(result.counts, list) and len(result.counts) == 1
        counts = result.counts[0]
        assert sum(counts.values()) == 2000
        assert set(counts.keys()) <= {"00", "11"}
        # A fair Bell state: both outcomes appear, roughly balanced.
        assert "00" in counts and "11" in counts
        assert abs(counts["00"] / 2000 - 0.5) < 0.1
        # The native backend is seeded: the manifest reports the effective seed.
        assert result.backend == "polypus"
        assert isinstance(result.seed, int)

    def test_native_backend_accepts_qasm_string(self):
        import polypus

        qasm = polypus.Circuit(1).x(0).measure_all().to_qasm2()
        result = polypus.run_quantum_circuit(
            qasm, shots=256, infrastructure="local", backend="polypus"
        )
        # X|0> = |1>: every shot reads "1".
        assert result.counts[0] == {"1": 256}

    def test_default_backend_is_aer(self):
        import polypus

        # No `backend` kwarg → Aer. Native Bell circuit still works end-to-end.
        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        result = polypus.run_quantum_circuit(qc, shots=1000, infrastructure="local")
        counts = result.counts[0]
        assert sum(counts.values()) == 1000
        assert set(counts.keys()) <= {"00", "11"}

    def test_qiskit_circuit_rejected_by_native_backend(self):
        import polypus
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        with pytest.raises(ValueError, match="cannot execute a Qiskit"):
            polypus.run_quantum_circuit(
                qc, shots=100, infrastructure="local", backend="polypus"
            )

    def test_unknown_backend_rejected(self):
        import polypus

        qc = polypus.Circuit(1).h(0).measure_all()
        with pytest.raises(ValueError, match="unknown local backend"):
            polypus.run_quantum_circuit(
                qc, shots=100, infrastructure="local", backend="does-not-exist"
            )

    def test_native_backend_rejects_noise_model(self):
        import polypus

        qc = polypus.Circuit(1).h(0).measure_all()
        with pytest.raises(ValueError, match="noise_model"):
            polypus.run_quantum_circuit(
                qc,
                shots=100,
                infrastructure="local",
                backend="polypus",
                noise_model=object(),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Boundary validation of shots / n_qpus (contract C-3)
#
# Validation happens at the Python-facing boundary before any backend work, so
# these do not need Aer or a QPU. n_qpus=0 previously reached
# DistributeByShotsRun and panicked with a division by zero; shots<1 silently
# ran a truncated (or empty) execution. Both must now raise a clear ValueError.
# ─────────────────────────────────────────────────────────────────────────────


class TestShotsQpusValidation:
    def test_zero_qpus_rejected(self):
        import polypus

        qc = polypus.Circuit(1).h(0).measure_all()
        with pytest.raises(ValueError, match="n_qpus must be >= 1"):
            polypus.run_quantum_circuit(qc, shots=100, infrastructure="local", n_qpus=0)

    def test_zero_shots_rejected(self):
        import polypus

        qc = polypus.Circuit(1).h(0).measure_all()
        with pytest.raises(ValueError, match="shots must be >= 1"):
            polypus.run_quantum_circuit(qc, shots=0, infrastructure="local", n_qpus=1)

    def test_zero_shots_rejected_in_train(self, simple_expectation_fn):
        """The same boundary guard protects the train() entry point."""
        import polypus

        qc = polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()
        with pytest.raises(ValueError, match="shots must be >= 1"):
            polypus.train(
                qc,
                polypus.DE(generations=2, population_size=4),
                shots=0,
                n_qpus=1,
                dimensions=1,
                expectation_function=simple_expectation_fn,
                infrastructure="local",
                nodes=1,
                cores_per_qpu=1,
                id="test_train_zero_shots",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Boundary validation of the CUNQA allocation params (issue #74)
#
# `nodes`/`cores_per_qpu` size the SLURM allocation and are consumed only by
# `infrastructure="cunqa"` (they reach CUNQA's `qraise`). A zero for either has
# no sane SLURM meaning, so it must be rejected at the boundary — before any
# `raise_qpus`/`connect_to_infrastructure` seam call, exactly like the zero
# shots/qpus guards above, so these need no real CUNQA/SLURM environment. The
# `local`/`qmio` paths ignore both params, so their validation is gated on
# `cunqa` and non-cunqa calls that omit them keep working unchanged.
# ─────────────────────────────────────────────────────────────────────────────


class TestCunqaAllocationValidation:
    def test_zero_nodes_rejected_for_cunqa(self):
        import polypus

        qc = polypus.Circuit(1).h(0).measure_all()
        with pytest.raises(ValueError, match="nodes must be >= 1"):
            polypus.run_quantum_circuit(
                qc, shots=100, infrastructure="cunqa", nodes=0
            )

    def test_zero_cores_per_qpu_rejected_for_cunqa(self):
        import polypus

        qc = polypus.Circuit(1).h(0).measure_all()
        with pytest.raises(ValueError, match="cores_per_qpu must be >= 1"):
            polypus.run_quantum_circuit(
                qc, shots=100, infrastructure="cunqa", cores_per_qpu=0
            )

    def test_local_ignores_allocation_defaults(self):
        """Non-cunqa calls omit nodes/cores_per_qpu; the inert defaults must not
        trip the cunqa-only validation, proving backward compatibility."""
        import polypus

        qc = polypus.Circuit(1).h(0).measure_all()
        result = polypus.run_quantum_circuit(qc, shots=128, infrastructure="local")
        assert sum(result.counts[0].values()) == 128

    def test_zero_nodes_rejected_in_train(self, simple_expectation_fn):
        """The same guard protects the train() entry point for cunqa."""
        import polypus

        qc = polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()
        with pytest.raises(ValueError, match="nodes must be >= 1"):
            polypus.train(
                qc,
                polypus.DE(generations=2, population_size=4),
                shots=100,
                n_qpus=1,
                dimensions=1,
                expectation_function=simple_expectation_fn,
                infrastructure="cunqa",
                nodes=0,
                cores_per_qpu=2,
                id="test_train_zero_nodes",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Native vs Aer equivalence on the same native circuit
# ─────────────────────────────────────────────────────────────────────────────


class TestNativeVsAerEquivalence:
    def test_distributions_agree_on_qaoa_like_circuit(self):
        import polypus

        def build():
            qc = polypus.Circuit(3).h(0).h(1).h(2)
            qc = qc.rzz(0, 1, 0.7).rzz(1, 2, 0.7).rx(0, 1.1).rx(1, 1.1).rx(2, 1.1)
            return qc.measure_all()

        shots = 8000
        aer = polypus.run_quantum_circuit(
            build(), shots=shots, infrastructure="local", backend="aer"
        ).counts[0]
        native = polypus.run_quantum_circuit(
            build(), shots=shots, infrastructure="local", backend="polypus"
        ).counts[0]

        # Same bitstring format (3-bit keys) and close probabilities.
        assert all(len(k) == 3 for k in native)
        keys = set(aer) | set(native)
        for k in keys:
            pa = aer.get(k, 0) / shots
            pn = native.get(k, 0) / shots
            assert abs(pa - pn) < 0.05, f"{k}: aer={pa:.3f} native={pn:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# train() with the native backend
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_KW = dict(
    shots=512,
    n_qpus=1,
    dimensions=1,
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
)


@pytest.fixture
def native_ry_circuit():
    import polypus

    return polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()


@pytest.mark.integration
@pytest.mark.vqc
class TestTrainNativeBackend:
    def test_train_de_with_native_backend(
        self, native_ry_circuit, simple_expectation_fn
    ):
        import polypus

        result = polypus.train(
            native_ry_circuit,
            polypus.DE(generations=10, population_size=10, tolerance=1e-9),
            expectation_function=simple_expectation_fn,
            id="test_native_backend_de",
            backend="polypus",
            **_TRAIN_KW,
        )
        theta = result.best_params[0]
        p1 = math.sin(theta / 2) ** 2
        assert p1 > 0.9, f"native backend did not converge: θ={theta:.3f}, p1={p1:.3f}"

    def test_qiskit_template_rejected_with_native_backend(
        self, parametrized_circuit, simple_expectation_fn
    ):
        import polypus

        with pytest.raises(ValueError, match="requires a native"):
            polypus.train(
                parametrized_circuit,
                polypus.DE(generations=2, population_size=4),
                expectation_function=simple_expectation_fn,
                id="test_native_backend_reject_qiskit",
                backend="polypus",
                **_TRAIN_KW,
            )
