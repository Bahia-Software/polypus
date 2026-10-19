"""Opt-in integration test against a real CUNQA family (issues #357, #253, #304).

Skipped unless ``POLYPUS_CUNQA_REAL=1`` and ``cunqa`` imports, so CI never runs
it. It reserves real vQPUs through SLURM: every ``run_quantum_circuit`` call
below raises its own family (``qraise``) and drops it at the end, so expect a
few minutes of queueing per case.

How to run, on a CESGA login node with CUNQA ``>=2.4,<4`` set up (``qraise`` on
``PATH``, the CUNQA libraries on ``LD_LIBRARY_PATH``) and Polypus installed::

    POLYPUS_CUNQA_REAL=1 python -m pytest tests/python/test_cunqa_real.py -v -rA

Optional settings: ``POLYPUS_CUNQA_WALLTIME`` (default ``00:20:00``),
``POLYPUS_CUNQA_CORES`` (cores per QPU, default ``1``), and the seal's own
``POLYPUS_CUNQA_PATH`` / ``POLYPUS_CUNQA_BACKEND``.

What it checks, with the native backend as the reference for the keys:

* C-3 counts shape (one dict per QPU, shots conserved) and C-7 seeding (same
  seed reproduces, QPUs differ) for a Bell circuit;
* #253: a circuit without measurements is read out on the full quantum
  register (``num_qubits`` wide keys);
* #304: an imported circuit whose classical registers are wider than what it
  measures keeps the declared width, as text and as an imported
  ``polypus.Circuit`` (OpenQASM 2 and 3).

Record the CUNQA version and the output in #253 / #304 when you run it.
"""

import os

import pytest

if os.environ.get("POLYPUS_CUNQA_REAL") != "1":
    pytest.skip(
        "real CUNQA test: set POLYPUS_CUNQA_REAL=1 on a CUNQA host",
        allow_module_level=True,
    )
pytest.importorskip("cunqa")

import polypus  # noqa: E402

pytestmark = pytest.mark.cunqa

_SHOTS = 200
_SETTINGS = dict(
    infrastructure="cunqa",
    backend="aer",
    nodes=1,
    cores_per_qpu=int(os.environ.get("POLYPUS_CUNQA_CORES", "1")),
    walltime=os.environ.get("POLYPUS_CUNQA_WALLTIME", "00:20:00"),
)

_QASM2_HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
_QASM3_HEADER = 'OPENQASM 3.0;\ninclude "stdgates.inc";\n'

# #304 cases: (name, OpenQASM 2 body, OpenQASM 3 body, expected key).
_WIDE_REGISTERS = [
    (
        "wider_creg",
        "qreg q[2];\ncreg c[3];\nx q[0];\nmeasure q[0] -> c[1];\n",
        "qubit[2] q;\nbit[3] c;\nx q[0];\nc[1] = measure q[0];\n",
        "010",
    ),
    (
        "several_registers",
        "qreg q[2];\ncreg a[2];\ncreg b[3];\nx q[1];\nmeasure q[1] -> a[1];\n",
        "qubit[2] q;\nbit[2] a;\nbit[3] b;\nx q[1];\na[1] = measure q[1];\n",
        "00010",
    ),
    (
        "full_measurement_wider_creg",
        "qreg q[2];\ncreg c[4];\nx q[1];\nmeasure q[0] -> c[0];\n"
        "measure q[1] -> c[1];\n",
        "qubit[2] q;\nbit[4] c;\nx q[1];\nc[0] = measure q[0];\nc[1] = measure q[1];\n",
        "0010",
    ),
    (
        "no_measurement_with_creg",
        "qreg q[2];\ncreg c[3];\nx q[0];\n",
        "qubit[2] q;\nbit[3] c;\nx q[0];\n",
        "01",
    ),
]


def _cunqa(qc, n_qpus, **kwargs):
    return polypus.run_quantum_circuit(
        qc, shots=_SHOTS, n_qpus=n_qpus, **_SETTINGS, **kwargs
    )


def _native(qc):
    return polypus.run_quantum_circuit(
        qc, shots=_SHOTS, infrastructure="local", backend="polypus", seed=1
    )


def _assert_c3(result, n_qpus):
    assert len(result.counts) == n_qpus
    assert sum(sum(c.values()) for c in result.counts) == _SHOTS
    assert sum(result.merged_counts.values()) == _SHOTS


def test_bell_counts_shape_and_seeding():
    bell = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
    first = _cunqa(bell, 2, seed=11)
    _assert_c3(first, 2)
    assert set(first.merged_counts) <= {"00", "11"}
    assert first.counts[0] != first.counts[1], "QPUs must not replay one stream"
    again = _cunqa(bell, 2, seed=11)
    assert again.counts == first.counts, "same seed must reproduce (C-7)"


@pytest.mark.parametrize("n_qpus", [1, 2])
def test_circuit_without_measurements_reads_the_full_register(n_qpus):
    # #253
    qc = polypus.Circuit(2).x(0)
    assert _native(qc).merged_counts == {"01": _SHOTS}
    result = _cunqa(qc, n_qpus, seed=3)
    _assert_c3(result, n_qpus)
    assert result.merged_counts == {"01": _SHOTS}


@pytest.mark.parametrize("n_qpus", [1, 2])
@pytest.mark.parametrize("form", ["qasm2_text", "from_qasm2", "from_qasm3"])
@pytest.mark.parametrize(
    "name,qasm2,qasm3,key", _WIDE_REGISTERS, ids=[c[0] for c in _WIDE_REGISTERS]
)
def test_declared_classical_width_is_kept(name, qasm2, qasm3, key, form, n_qpus):
    # #304
    if form == "qasm2_text":
        qc = _QASM2_HEADER + qasm2
    elif form == "from_qasm2":
        qc = polypus.Circuit.from_qasm2(_QASM2_HEADER + qasm2)
    else:
        qc = polypus.Circuit.from_qasm3(_QASM3_HEADER + qasm3)
    assert _native(qc).merged_counts == {key: _SHOTS}
    result = _cunqa(qc, n_qpus, seed=5)
    _assert_c3(result, n_qpus)
    assert result.merged_counts == {key: _SHOTS}
