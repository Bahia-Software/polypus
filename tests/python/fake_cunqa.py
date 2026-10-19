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

What it cannot pin is CUNQA's own output: the counts are Aer's
``get_counts()``, so the key format for circuits without measurements (#253)
or with a wider ``creg`` (#304) is Aer's, not necessarily CUNQA's. Those cases
are for ``test_cunqa_real.py`` on a real family.
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
        simulator = AerSimulator(method=options.get("method", "automatic"))
        run_options = {"shots": options["shots"]}
        if seed is not None:
            run_options["seed_simulator"] = seed
        counts = simulator.run(circuit, **run_options).result().get_counts()
        return types.SimpleNamespace(counts=dict(counts))

    # cunqa.qjob.gather
    def gather(self, *args, **kwargs):
        a = self._bind("gather", args, kwargs)
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
