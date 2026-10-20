"""A stateful fake of the CUNQA SDK, for driving the CUNQA seal without SLURM.

No CUNQA is installed in CI. ``FakeCunqa`` stands in for the four functions
``polypus_python.cunqa`` imports (``cunqa.qpu.qraise``, ``get_QPUs``, ``run``,
``qdrop`` and ``cunqa.qjob.gather``) and keeps the state a real family has:

* every call is bound against the signature CUNQA has at one release tag
  (``data/cunqa/signatures.json``, extracted from the sources), so a name or
  arity CUNQA would reject raises ``TypeError`` here too;
* ``qraise`` creates a family of ``n`` QPUs, ``get_QPUs`` returns them in order,
  ``qdrop`` releases it; using a family that was never raised or was already
  dropped fails, like a real allocation that is gone;
* ``run`` simulates the circuit with Qiskit Aer, honouring ``shots``,
  ``method`` and ``seed`` the way CUNQA's Aer-based vQPUs consume them
  (``seed`` -> ``seed_simulator``), and ``gather`` returns objects with a
  ``counts`` dict.

The output mimics what CUNQA 3.0.0 returned on a real family (#253, #304): a
circuit without measurements fails the job, and the key of a circuit with
several classical registers is cut with spaces by ``cunqa_keys``. The bits
themselves are Aer's; ``test_cunqa_real.py`` checks them on a real family.
"""

import inspect
import json
import sys
import types
from pathlib import Path

SIGNATURES = json.loads(
    (Path(__file__).parent / "data" / "cunqa" / "signatures.json").read_text()
)
SUPPORTED_TAGS = ["2.4.0", "3.0.0", "3.1.0"]

# The run options the seal may send; anything else is a seal change to review.
RUN_OPTIONS = {"shots", "transpile", "method", "seed"}


NO_COUNTS_ERROR = (
    "Error during simulation, please check availability of QPUs, run arguments "
    "syntax and circuit syntax: [json.exception.out_of_range.403] key 'counts' "
    "not found"
)


def cunqa_keys(counts, circuit):
    """Aer's ``counts`` keyed as CUNQA 3.0.0's ``Result.counts`` keys them: the
    flat bitstring, cut with spaces from the left by the register sizes in
    declaration order when the circuit has several registers."""
    sizes = [creg.size for creg in circuit.cregs]
    flat = {key.replace(" ", ""): n for key, n in counts.items()}
    if len(sizes) <= 1:
        return flat
    cuts = [0]
    for size in sizes:
        cuts.append(cuts[-1] + size)
    return {
        " ".join(key[i:j] for i, j in zip(cuts, cuts[1:])): n for key, n in flat.items()
    }


_QASM2_HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
_QASM3_HEADER = 'OPENQASM 3.0;\ninclude "stdgates.inc";\n'

# #304 cases, shared with test_cunqa_real.py: (name, OpenQASM 2 body,
# OpenQASM 3 body, expected key).
WIDE_REGISTERS = [
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

CIRCUIT_FORMS = ["qasm2_text", "from_qasm2", "from_qasm3"]


def circuit_in_form(form, qasm2, qasm3):
    """A ``WIDE_REGISTERS`` case as OpenQASM 2 text or an imported
    ``polypus.Circuit``."""
    import polypus

    if form == "qasm2_text":
        return _QASM2_HEADER + qasm2
    if form == "from_qasm2":
        return polypus.Circuit.from_qasm2(_QASM2_HEADER + qasm2)
    return polypus.Circuit.from_qasm3(_QASM3_HEADER + qasm3)


def signature(parameters: str) -> inspect.Signature:
    """An ``inspect.Signature`` from a parameter list as written in the source."""
    namespace = {}
    exec(f"def f({parameters}): pass", namespace)
    return inspect.signature(namespace["f"])


class FakeQPU:
    def __init__(self, family, index):
        self.family = family
        self.index = index

    def __repr__(self):
        return f"FakeQPU({self.family!r}, {self.index})"


class FakeCunqa:
    """The fake SDK for CUNQA release ``tag``. ``calls`` records every call as
    ``(name, bound arguments)`` in order; ``runs`` records each ``run`` as a dict
    with the circuit, the QPU and the run options."""

    def __init__(self, tag="3.1.0", run_error=None):
        self.tag = tag
        self.signatures = {n: signature(p) for n, p in SIGNATURES[tag].items()}
        self.families = {}
        self.dropped = []
        self.calls = []
        self.runs = []
        self.run_error = run_error

    def _bind(self, name, args, kwargs):
        bound = self.signatures[name].bind(*args, **kwargs)
        bound.apply_defaults()
        self.calls.append((name, dict(bound.arguments)))
        return bound.arguments

    def _live(self, family):
        if family not in self.families:
            raise RuntimeError(f"no CUNQA family named {family!r} was raised")
        if family in self.dropped:
            raise RuntimeError(f"CUNQA family {family!r} was already dropped")
        return self.families[family]

    # cunqa.qpu.qraise
    def qraise(self, *args, **kwargs):
        a = self._bind("qraise", args, kwargs)
        family = a["family"] or f"family-{len(self.families)}"
        if family in self.families and family not in self.dropped:
            raise RuntimeError(f"CUNQA family {family!r} is already raised")
        self.families[family] = [FakeQPU(family, i) for i in range(a["n"])]
        return family

    # cunqa.qpu.get_QPUs
    def get_QPUs(self, *args, **kwargs):
        a = self._bind("get_QPUs", args, kwargs)
        return list(self._live(a["family"]))

    # cunqa.qpu.run
    def run(self, *args, **kwargs):
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        a = self._bind("run", args, kwargs)
        circuit, qpu, options = a["circuits"], a["qpus"], a["run_args"]
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(f"expected one QuantumCircuit, got {type(circuit)!r}")
        if not isinstance(qpu, FakeQPU) or qpu not in self._live(qpu.family):
            raise TypeError(f"expected one QPU of a live family, got {qpu!r}")
        unknown = set(options) - RUN_OPTIONS
        if unknown:
            raise TypeError(f"unexpected run options {sorted(unknown)}")
        seed = options.get("seed")
        if seed is not None and not 0 <= seed < 2**63:
            raise ValueError(f"seed {seed} is outside Aer's 63-bit range")
        self.runs.append({"circuit": circuit, "qpu": qpu, "options": dict(options)})
        if self.run_error is not None:
            raise self.run_error
        if not any(i.operation.name == "measure" for i in circuit.data):
            # CUNQA 3.0.0 fails the job; `gather` raises it.
            return types.SimpleNamespace(error=RuntimeError(NO_COUNTS_ERROR))
        simulator = AerSimulator(method=options.get("method", "automatic"))
        run_options = {"shots": options["shots"]}
        if seed is not None:
            run_options["seed_simulator"] = seed
        counts = simulator.run(circuit, **run_options).result().get_counts()
        return types.SimpleNamespace(counts=cunqa_keys(counts, circuit), error=None)

    # cunqa.qjob.gather
    def gather(self, *args, **kwargs):
        a = self._bind("gather", args, kwargs)
        for job in a["qjobs"]:
            if job.error is not None:
                raise job.error
        return [types.SimpleNamespace(counts=job.counts) for job in a["qjobs"]]

    # cunqa.qpu.qdrop
    def qdrop(self, *args, **kwargs):
        a = self._bind("qdrop", args, kwargs)
        families = a["families"]
        # 2.4 takes `families` as one name or a list; 3.x as `*families`.
        if isinstance(families, str):
            families = [families]
        for family in families:
            self._live(family)
            self.dropped.append(family)

    def names(self):
        """The SDK functions called, in order."""
        return [name for name, _ in self.calls]

    def install(self, monkeypatch):
        """Register this fake as the ``cunqa`` package for the test."""
        qpu = types.ModuleType("cunqa.qpu")
        for name in ("qraise", "get_QPUs", "run", "qdrop"):
            setattr(qpu, name, getattr(self, name))
        qjob = types.ModuleType("cunqa.qjob")
        qjob.gather = self.gather
        monkeypatch.setitem(sys.modules, "cunqa", types.ModuleType("cunqa"))
        monkeypatch.setitem(sys.modules, "cunqa.qpu", qpu)
        monkeypatch.setitem(sys.modules, "cunqa.qjob", qjob)
        # Force a fresh import so the seal binds these functions.
        monkeypatch.delitem(sys.modules, "polypus_python.cunqa", raising=False)
        monkeypatch.delenv("POLYPUS_CUNQA_BACKEND", raising=False)
        return self
