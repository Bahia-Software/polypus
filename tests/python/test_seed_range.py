"""The seed range of each backend (contract C-7, #393).

Qiskit Aer's ``seed_simulator`` is a signed 64-bit integer, so a backend that
runs on Aer (``backend="aer"``, and CUNQA's simulated QPUs) takes a seed from 0
to ``2**63 - 1``. A seed Polypus draws from OS entropy is always in that range,
so the seed a run reports is the one Aer receives and replays the run outside
Polypus too. An explicit seed of ``2**63`` or more on an Aer path raises
``polypus.ConfigurationError`` (a ``ValueError``) before anything runs, instead
of reaching Aer masked to 63 bits, where two seeds differing only in bit 63
gave the same counts. The native backend takes any 64-bit seed.

Every entry point that resolves a run seed is covered: ``run_quantum_circuit``,
``train`` (the kwarg and an optimizer's ``seed`` field), ``qml.train``,
``qml.predict``, and the native QML API's ``Model.fit`` and
``TrainedModel.run``. CUNQA's own cases are in ``test_cunqa_run_flow.py``.
"""

import pytest
from qml_native_helpers import X, Y, make_model

LIMIT = 2**63
TOO_LARGE = [LIMIT, 2**64 - 1]


def _uniform3():
    import polypus

    return polypus.Circuit(3).h(0).h(1).h(2).measure_all()


def _rejects(seed):
    """`pytest.raises` for the out-of-range error, which is both a
    `polypus.ConfigurationError` and a `ValueError` and names the seed."""
    import polypus

    assert issubclass(polypus.ConfigurationError, ValueError)
    return pytest.raises(
        polypus.ConfigurationError, match=rf"seed {seed} is out of range for"
    )


def _forbid_the_seam(monkeypatch):
    """Fail the test if any circuit reaches `polypus_python`: the seed must be
    rejected before a backend runs."""
    import polypus_python

    def run_qcs(*args, **kwargs):
        raise AssertionError("an out-of-range seed must be rejected before any run")

    monkeypatch.setattr(polypus_python, "run_qcs", run_qcs)


@pytest.mark.integration
class TestRunQuantumCircuit:
    @pytest.mark.parametrize("seed", TOO_LARGE)
    def test_aer_rejects_a_seed_of_two_to_the_63_or_more(self, seed, monkeypatch):
        import polypus

        _forbid_the_seam(monkeypatch)
        with _rejects(seed) as raised:
            polypus.run_quantum_circuit(
                _uniform3(), shots=64, infrastructure="local", backend="aer", seed=seed
            )
        message = str(raised.value)
        assert 'Qiskit Aer (backend="aer" or target=polypus.Aer())' in message
        assert str(LIMIT - 1) in message
        # Where a 64-bit seed is taken, without suggesting a switch.
        assert "only the native backend and registered backends" in message

    def test_a_target_names_the_same_backend(self, monkeypatch):
        # A caller who chose Aer with a target sees the target named too.
        import polypus

        _forbid_the_seam(monkeypatch)
        with _rejects(LIMIT) as raised:
            polypus.run_quantum_circuit(
                _uniform3(), shots=64, target=polypus.Aer(), seed=LIMIT
            )
        assert "target=polypus.Aer()" in str(raised.value)

    def test_aer_takes_the_largest_63_bit_seed(self):
        import polypus

        kwargs = dict(shots=256, infrastructure="local", backend="aer", seed=LIMIT - 1)
        first = polypus.run_quantum_circuit(_uniform3(), **kwargs)
        again = polypus.run_quantum_circuit(_uniform3(), **kwargs)
        assert first.seed == LIMIT - 1
        assert first.counts == again.counts

    @pytest.mark.parametrize("seed", TOO_LARGE)
    def test_the_native_backend_takes_any_64_bit_seed(self, seed):
        import polypus

        kwargs = dict(shots=256, infrastructure="local", backend="polypus", seed=seed)
        first = polypus.run_quantum_circuit(_uniform3(), **kwargs)
        assert first.seed == seed
        assert first.counts == polypus.run_quantum_circuit(_uniform3(), **kwargs).counts

    def test_an_unseeded_aer_run_reports_the_seed_aer_received(self):
        """The drawn seed is below 2**63 (half of the old draws were not), and
        Aer takes it as it is, outside Polypus too."""
        import polypus
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        seeds = [
            polypus.run_quantum_circuit(
                _uniform3(), shots=8, infrastructure="local", backend="aer"
            ).seed
            for _ in range(24)
        ]
        assert all(0 <= seed < LIMIT for seed in seeds), seeds
        assert len(set(seeds)) == len(seeds)
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        # The low-level TypeError the issue describes is gone for every seed.
        for seed in seeds[:4]:
            AerSimulator(seed_simulator=seed).run(qc, shots=8).result()


@pytest.mark.integration
@pytest.mark.vqc
class TestTrain:
    @staticmethod
    def _train(backend, seed=None, method_seed=None):
        import polypus

        qc = polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()
        return polypus.train(
            qc,
            polypus.DE(
                generations=2, population_size=4, tolerance=1e-12, seed=method_seed
            ),
            shots=32,
            n_qpus=1,
            dimensions=1,
            expectation_function=lambda bits: float(bits == "1"),
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id="seed_range",
            backend=backend,
            seed=seed,
        )

    @pytest.mark.parametrize("seed", TOO_LARGE)
    def test_aer_rejects_the_kwarg_seed(self, seed, monkeypatch):
        _forbid_the_seam(monkeypatch)
        with _rejects(seed):
            self._train("aer", seed=seed)

    def test_aer_rejects_the_optimizer_field_seed(self, monkeypatch):
        """The precedence picks the field; the resolved seed is what is checked."""
        _forbid_the_seam(monkeypatch)
        with _rejects(LIMIT):
            self._train("aer", method_seed=LIMIT)

    def test_the_native_backend_trains_with_any_64_bit_seed(self):
        result = self._train("polypus", seed=2**64 - 1)
        assert result.seed == 2**64 - 1


@pytest.mark.integration
@pytest.mark.vqc
class TestQiskitQml:
    @staticmethod
    def _circuits():
        from qiskit.circuit.library import real_amplitudes, zz_feature_map

        return zz_feature_map(feature_dimension=2, reps=1), real_amplitudes(
            num_qubits=2, reps=1
        )

    def test_qml_train_rejects_a_seed_of_two_to_the_63(self, monkeypatch):
        import numpy as np
        import polypus

        _forbid_the_seam(monkeypatch)
        feature_map, ansatz = self._circuits()
        with _rejects(LIMIT):
            polypus.qml.train(
                feature_map,
                ansatz,
                np.zeros((2, 2)),
                polypus.DE(generations=2, population_size=4, tolerance=1e-12),
                shots=16,
                n_qpus=1,
                dimensions=len(ansatz.parameters),
                expectation_function=lambda b: sum(int(c) for c in b) / len(b),
                infrastructure="local",
                nodes=1,
                cores_per_qpu=1,
                id="seed_range_qml",
                seed=LIMIT,
            )

    def test_qml_predict_rejects_a_seed_of_two_to_the_63(self, monkeypatch):
        import numpy as np
        import polypus

        _forbid_the_seam(monkeypatch)
        feature_map, ansatz = self._circuits()
        with _rejects(LIMIT):
            polypus.qml.predict(
                feature_map,
                ansatz,
                np.zeros((2, 2)),
                [0.0] * len(ansatz.parameters),
                16,
                infrastructure="local",
                seed=LIMIT,
            )


@pytest.mark.integration
class TestNativeQml:
    def test_fit_on_aer_rejects_a_seed_of_two_to_the_63(self, monkeypatch):
        import polypus

        _forbid_the_seam(monkeypatch)
        with _rejects(LIMIT):
            make_model().fit(
                X,
                Y,
                method=polypus.Adam(max_iters=2, patience=10_000),
                loss="hinge",
                backend="aer",
                shots=16,
                seed=LIMIT,
            )

    def test_run_on_aer_rejects_a_seed_of_two_to_the_63(self, monkeypatch):
        _forbid_the_seam(monkeypatch)
        model = make_model()
        trained = model.bind([0.0] * model.num_params)
        with _rejects(LIMIT):
            trained.run(X, backend="aer", shots=16, seed=LIMIT)

    def test_fit_on_the_native_backend_takes_any_64_bit_seed(self):
        import polypus

        result = make_model().fit(
            X,
            Y,
            method=polypus.Adam(max_iters=2, patience=10_000),
            loss="hinge",
            shots=16,
            seed=2**64 - 1,
        )
        assert result.seed == 2**64 - 1
