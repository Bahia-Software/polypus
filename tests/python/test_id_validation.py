"""
`id` charset validation at the Python boundary (contract C-9).

These cover issue #89: the `id` kwarg of ``train``/``qml.train`` travelled
verbatim into ``ExecutionConfig::id`` and from there into CUNQA's SLURM
``family_name``/``family_id`` (forwarded to ``qraise``) plus the run's temp file
and log stream names, with no character validation anywhere in the crate. A
whitespace character, a path separator (``../``) or a shell metacharacter could
therefore reach whatever CUNQA/SLURM does with the name downstream.

The prefix is now restricted to ``[A-Za-z0-9._-]``, non-empty and at most 64
characters, rejected with a ``ValueError`` naming the offending character. The
check runs upfront — before any seam call, any backend creation and before the
UUID v4 suffix is appended — so the rejection tests never reach a real backend
and need no optimizer/backend mocking. The acceptance tests do run, on
``infrastructure="local"`` with a deliberately tiny optimizer budget.
"""

import pytest

# Accepted: plain alphanumeric, the three allowed punctuation characters mixed
# in, a single character, and exactly the 64-character maximum.
VALID_IDS = [
    "run1",
    "my.run_2-final",
    "a",
    "i" * 64,
]

# Rejected, paired with the fragment of the ValueError message that proves the
# rejection was specific rather than incidental.
INVALID_IDS = [
    ("", "id must not be empty"),
    ("my run", r"invalid character ' '"),
    ("../etc/passwd", r"invalid character '/'"),
    ("runs/1", r"invalid character '/'"),
    ("run;rm -rf /", "invalid character ';'"),
    ("run|tee", r"invalid character '\|'"),
    ("run`whoami`", "invalid character '`'"),
    ("run$USER", r"invalid character '\$'"),
    ("run&background", "invalid character '&'"),
    ("run\nid", r"invalid character '\\n'"),
    ("i" * 65, "id must be at most 64 characters, got 65"),
]


def _train(id_):
    """`polypus.train` with every kwarg pinned except `id`."""
    import polypus

    qc = polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()
    return polypus.train(
        qc,
        polypus.DE(generations=2, population_size=4),
        shots=64,
        n_qpus=1,
        dimensions=1,
        expectation_function=lambda b: float(all(c == "1" for c in b)),
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id=id_,
        seed=7,
    )


def _qml_train(id_):
    """`polypus.qml.train` with every kwarg pinned except `id`."""
    import polypus
    from qiskit.circuit.library import real_amplitudes, zz_feature_map

    feature_map = zz_feature_map(feature_dimension=2, reps=1)
    ansatz = real_amplitudes(num_qubits=2, reps=1)
    return polypus.qml.train(
        feature_map,
        ansatz,
        [[0.1, 0.2]],
        polypus.DE(generations=2, population_size=4),
        shots=64,
        n_qpus=1,
        dimensions=len(ansatz.parameters),
        expectation_function=lambda b: sum(int(c) for c in b) / len(b),
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id=id_,
        seed=7,
    )


@pytest.mark.integration
@pytest.mark.vqc
class TestTrainIdValidation:
    @pytest.mark.parametrize("id_", VALID_IDS)
    def test_valid_id_accepted(self, id_):
        result = _train(id_)
        # The effective id keeps the accepted prefix and appends the UUID v4.
        assert result.id.startswith(f"{id_}_")
        assert result.id != id_

    @pytest.mark.parametrize("id_,message", INVALID_IDS)
    def test_invalid_id_rejected(self, id_, message):
        with pytest.raises(ValueError, match=message):
            _train(id_)


@pytest.mark.integration
@pytest.mark.vqc
class TestQmlTrainIdValidation:
    @pytest.mark.parametrize("id_", VALID_IDS)
    def test_valid_id_accepted(self, id_):
        result = _qml_train(id_)
        assert result.id.startswith(f"{id_}_")
        assert result.id != id_

    @pytest.mark.parametrize("id_,message", INVALID_IDS)
    def test_invalid_id_rejected(self, id_, message):
        with pytest.raises(ValueError, match=message):
            _qml_train(id_)
