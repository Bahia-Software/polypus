"""The CUNQA run flow end to end, against a fake CUNQA SDK (issue #357).

``polypus.run_quantum_circuit`` and ``polypus.train`` with
``infrastructure="cunqa"`` go through the Rust ``CunqaBackend`` and the
``polypus_python.cunqa`` seal into ``fake_cunqa.FakeCunqa``, which binds every
call against CUNQA's real signatures at one release tag and simulates each job
with Aer. Pinned here: the ``qraise`` reservation, one ``run`` per QPU with the
run options, ``gather``, the release with ``qdrop`` (also after a failure), the
C-3 counts shape and the C-7 per-QPU seeds.

The fake reports Aer's counts, so the key format of CUNQA itself (#253, #304) is
only checked by ``test_cunqa_real.py`` on a real family.
"""

import pytest

pytest.importorskip("qiskit_aer")

import polypus  # noqa: E402
from fake_cunqa import SUPPORTED_TAGS, FakeCunqa  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402

_MASK = 0x7FFFFFFFFFFFFFFF


def _uniform3():
    return polypus.Circuit(3).h(0).h(1).h(2).measure_all()


def _run(fake, monkeypatch, qc=None, **kwargs):
    fake.install(monkeypatch)
    kwargs.setdefault("shots", 64)
    kwargs.setdefault("n_qpus", 2)
    return polypus.run_quantum_circuit(
        _uniform3() if qc is None else qc,
        infrastructure="cunqa",
        backend="aer",
        **kwargs,
    )


def _args(fake, name):
    return [arguments for called, arguments in fake.calls if called == name]


@pytest.mark.parametrize("tag", SUPPORTED_TAGS)
def test_run_flow_end_to_end(monkeypatch, tag):
    fake = FakeCunqa(tag)
    result = _run(
        fake,
        monkeypatch,
        seed=11,
        nodes=1,
        cores_per_qpu=4,
        walltime="02:00:00",
        sim_method="statevector",
    )

    assert fake.names() == ["qraise", "get_QPUs", "run", "run", "gather", "qdrop"]
    (raised,) = _args(fake, "qraise")
    assert (raised["n"], raised["t"]) == (2, "02:00:00")
    assert raised["family"] == result.id
    assert raised["n_nodes"] == 1
    assert raised["cores_per_qpu"] == 4
    assert raised["co_located"] is True and raised["quantum_comm"] is False
    (lookup,) = _args(fake, "get_QPUs")
    assert lookup == {"co_located": True, "family": result.id}

    # One job per QPU, in order, with the split shots and the run options.
    assert [run["qpu"].index for run in fake.runs] == [0, 1]
    assert all(isinstance(run["circuit"], QuantumCircuit) for run in fake.runs)
    assert [run["options"] for run in fake.runs] == [
        {"shots": 32, "transpile": False, "method": "statevector", "seed": 11},
        {"shots": 32, "transpile": False, "method": "statevector", "seed": 12},
    ]
    assert fake.dropped == [result.id]

    # C-3: one dict per QPU, n_qpus long, keys num_clbits wide, shots conserved.
    assert len(result.counts) == 2
    for counts in result.counts:
        assert sum(counts.values()) == 32
        assert all(len(key) == 3 and set(key) <= {"0", "1"} for key in counts)
        assert all(isinstance(n, int) for n in counts.values())
    assert sum(result.merged_counts.values()) == 64
    assert result.seed == 11
    assert result.infrastructure == "cunqa"


def test_counts_match_aer_with_the_same_per_qpu_seed(monkeypatch):
    # The counts are the vQPUs' own, unmodified: QPU i equals Aer run with
    # seed_simulator = seed + i.
    from qiskit_aer import AerSimulator

    result = _run(FakeCunqa(), monkeypatch, seed=5)
    qc = QuantumCircuit.from_qasm_str(_uniform3().to_qasm2())
    for i, counts in enumerate(result.counts):
        expected = AerSimulator().run(qc, shots=32, seed_simulator=5 + i)
        assert counts == expected.result().get_counts()


def test_same_seed_reproduces_counts(monkeypatch):
    # C-7: same seed and batch, same counts, on every QPU.
    first = _run(FakeCunqa(), monkeypatch, seed=2024)
    second = _run(FakeCunqa(), monkeypatch, seed=2024)
    assert first.counts == second.counts


def test_qpus_get_distinct_seeds(monkeypatch):
    # C-7: each QPU is seeded `seed + i`, so replicas of one circuit are not
    # copies of each other.
    fake = FakeCunqa()
    result = _run(fake, monkeypatch, seed=7, n_qpus=4, shots=400)
    assert [run["options"]["seed"] for run in fake.runs] == [7, 8, 9, 10]
    assert len({tuple(sorted(c.items())) for c in result.counts}) == 4


def test_seed_is_masked_to_aer_range(monkeypatch):
    # Seeds are masked to 63 bits (CUNQA's vQPUs are Aer-based); the per-QPU
    # offset wraps inside that range.
    fake = FakeCunqa()
    _run(fake, monkeypatch, seed=2**64 - 1)
    assert [run["options"]["seed"] for run in fake.runs] == [_MASK, 0]


def test_unseeded_run_reports_a_seed_that_replays_it(monkeypatch):
    # C-7: `seed=None` draws one; the reported seed reproduces the counts.
    first = _run(FakeCunqa(), monkeypatch)
    assert isinstance(first.seed, int)
    replay = _run(FakeCunqa(), monkeypatch, seed=first.seed)
    assert replay.counts == first.counts


def test_qiskit_circuit_runs_as_given(monkeypatch):
    qc = QuantumCircuit(2, 2)
    qc.x(1)
    qc.measure([0, 1], [0, 1])
    fake = FakeCunqa()
    result = _run(fake, monkeypatch, qc=qc, seed=1, shots=20)
    assert result.counts == [{"10": 10}, {"10": 10}]
    assert result.merged_counts == {"10": 20}


def test_qpus_are_dropped_when_a_job_fails(monkeypatch):
    # The reservation is released even when the run raises (RAII, C-1).
    fake = FakeCunqa(run_error=RuntimeError("vQPU crashed"))
    with pytest.raises(RuntimeError, match="vQPU crashed"):
        _run(fake, monkeypatch, seed=1)
    assert fake.names()[-1] == "qdrop"
    assert fake.dropped == list(fake.families)


def test_train_runs_each_wave_on_the_family(monkeypatch):
    # A DE population is evaluated in waves of at most n_qpus circuits, each
    # wave one `run` per QPU on the one family raised for the whole training.
    fake = FakeCunqa().install(monkeypatch)
    result = polypus.train(
        polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all(),
        polypus.DE(generations=1, population_size=4),
        shots=16,
        n_qpus=2,
        dimensions=1,
        expectation_function=lambda bits: float(bits == "1"),
        infrastructure="cunqa",
        nodes=1,
        cores_per_qpu=1,
        id="cunqa_flow",
        backend="aer",
        seed=3,
    )
    names = fake.names()
    assert names.count("qraise") == 1 and names[-1] == "qdrop"
    assert fake.dropped == [result.id]
    assert len(fake.runs) >= 4
    for run in fake.runs:
        assert run["options"]["shots"] == 16
        assert run["qpu"].family == result.id
    # Within a wave, QPU i gets seed + i.
    for run in fake.runs:
        assert run["options"]["seed"] == 3 + run["qpu"].index


@pytest.mark.xfail(
    strict=True,
    reason="#357 finding: an uneven shot split runs one run_qcs call per batch "
    "size, so QPU indices and seed offsets restart at 0 for each call",
)
def test_uneven_shot_split_gives_every_replica_its_own_qpu_and_seed(monkeypatch):
    fake = FakeCunqa()
    result = _run(fake, monkeypatch, seed=11, shots=10, n_qpus=3)
    assert [sum(c.values()) for c in result.counts] == [4, 3, 3]
    assert sorted(run["qpu"].index for run in fake.runs) == [0, 1, 2]
    assert sorted(run["options"]["seed"] for run in fake.runs) == [11, 12, 13]


@pytest.mark.xfail(
    strict=True,
    reason="#304: CUNQA separates classical registers with spaces in its "
    "counts keys and the seal forwards them; C-3 keys have none",
)
def test_several_registers_give_one_key_without_spaces(monkeypatch):
    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg a[2];\ncreg b[3];\n'
        "x q[1];\nmeasure q[1] -> a[1];\n"
    )
    result = _run(FakeCunqa(), monkeypatch, qc=qasm, seed=1, shots=8)
    assert result.merged_counts == {"00010": 8}
