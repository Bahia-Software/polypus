"""Central validation of backend measurement results.

A backend that returns an empty counts map, a shot total that does not match the
request (contract C-3 conservation), or the wrong number of result maps must
surface as a typed ``polypus.BackendError`` — *not* a silently-zero fitness (for
``train``) or a malformed payload (for ``run_quantum_circuit``).

The pure-Rust native backend cannot produce such results, so these mock the Aer
seam (``polypus_python.run_qcs``) to hand back malformed counts, exercising the
``run_quantum_circuit`` (single-run) and ``train`` (oracle) validation sites.
"""

import pytest


def _mock_run_qcs(monkeypatch, maker):
    import polypus_python

    monkeypatch.setattr(
        polypus_python,
        "run_qcs",
        lambda infrastructure, **kwargs: maker(kwargs),
    )


@pytest.mark.integration
class TestRunQuantumCircuitResultValidation:
    def test_empty_counts_raises(self, monkeypatch):
        import polypus

        _mock_run_qcs(monkeypatch, lambda kw: [{} for _ in kw["qcs"]])
        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        with pytest.raises(polypus.BackendError):
            polypus.run_quantum_circuit(
                qc, shots=100, infrastructure="local", backend="aer"
            )

    def test_shot_non_conservation_raises(self, monkeypatch):
        import polypus

        # 5 shots reported for a 100-shot request violates C-3.
        _mock_run_qcs(monkeypatch, lambda kw: [{"11": 5} for _ in kw["qcs"]])
        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        with pytest.raises(polypus.BackendError):
            polypus.run_quantum_circuit(
                qc, shots=100, infrastructure="local", backend="aer"
            )


@pytest.mark.integration
@pytest.mark.vqc
class TestTrainResultValidation:
    def test_empty_counts_is_not_a_silent_zero(
        self, monkeypatch, parametrized_circuit, simple_expectation_fn
    ):
        import polypus

        _mock_run_qcs(monkeypatch, lambda kw: [{} for _ in kw["qcs"]])
        with pytest.raises(polypus.BackendError):
            polypus.train(
                parametrized_circuit,
                polypus.DE(generations=2, population_size=4, tolerance=0.5),
                shots=256,
                n_qpus=1,
                dimensions=1,
                expectation_function=simple_expectation_fn,
                infrastructure="local",
                backend="aer",
                nodes=1,
                cores_per_qpu=1,
                id="test_train_empty_counts",
            )
