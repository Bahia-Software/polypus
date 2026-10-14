import os
import sys
from functools import lru_cache

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import get_standard_gate_name_mapping

# CUNQA is an optional, site-installed dependency (the root pyproject.toml
# declares no `cunqa` extra). Prefer a normal import -- already installed or on
# PYTHONPATH. As an explicit escape hatch for a non-packaged site
# install, honour POLYPUS_CUNQA_PATH (a documented, deployment-specific location)
# instead of the previous unconditional `sys.path.append(os.getenv("HOME"))`,
# which was non-portable and appended `None` to sys.path when HOME was unset.
_cunqa_path = os.getenv("POLYPUS_CUNQA_PATH")
if _cunqa_path and _cunqa_path not in sys.path:
    sys.path.append(_cunqa_path)

from cunqa.qjob import gather  # noqa: E402  (must follow the path setup above)
from cunqa.qpu import get_QPUs, qdrop, qraise, run  # noqa: E402

try:
    # The instruction names CUNQA's QuantumCircuit converter accepts (CUNQA 3.x).
    from cunqa.circuit.ir import SUPPORTED_QISKIT_OPERATIONS  # noqa: E402
except ImportError:
    SUPPORTED_QISKIT_OPERATIONS = None

from .infrastructure import Infraestructure  # noqa: E402
from .local import _outside_basis  # noqa: E402


@lru_cache(maxsize=1)
def _cunqa_basis():
    """The instruction names CUNQA converts, and the gates among them that
    ``qiskit.transpile`` can target (its ``basis_gates`` takes standard gates
    only; ``unitary``, ``ccy``, ``if_else`` and CUNQA's own instructions are
    accepted as they are but are no lowering target)."""
    supported = frozenset(SUPPORTED_QISKIT_OPERATIONS)
    return supported, sorted(supported & set(get_standard_gate_name_mapping()))


def _lower_to_cunqa_basis(qc):
    """``qc`` when CUNQA converts every instruction in it, else an equivalent
    copy rewritten into the gates it converts.

    CUNQA's converter rejects what is not in ``SUPPORTED_QISKIT_OPERATIONS``
    (``ch``, ``u0``, ``rccx``, ``rcccx``, ``c3sx``, ``mcx`` -- Qiskit's name for
    ``c3x`` and ``c4x`` -- and a ``gate`` declared in the OpenQASM source) and,
    like Aer, does not unroll it. The circuit is lowered here, a lowering
    boundary of contract C-2 as in ``local.py``: rewrite only
    (``optimization_level=0``: no reordering or merging), only when needed (a circuit CUNQA
    converts is handed over as it is), never touching the exported QASM. With a
    CUNQA that does not publish the list, the circuit is forwarded unchanged."""
    if SUPPORTED_QISKIT_OPERATIONS is None:
        return qc
    supported, basis_gates = _cunqa_basis()
    if not _outside_basis(qc, supported):
        return qc
    return transpile(qc, basis_gates=basis_gates, optimization_level=0)


class Cunqa(Infraestructure):
    def get_qpus(self, **kwargs) -> object:
        n = kwargs["n"]
        t = kwargs["t"]
        n_nodes = kwargs["n_nodes"]
        family_name = kwargs["family_name"]
        # Cores per QPU is forwarded by the Rust CunqaBackend (`raise_qpus` sets
        # kwargs["cores_per_qpu"]); it used to be dropped here, so the requested
        # allocation was silently ignored. Pass it through to qraise. `cores`
        # follows CUNQA's qraise API (part of the unverified integration; see the
        # note in run_qcs).
        cores_per_qpu = kwargs.get("cores_per_qpu")

        qraise_kwargs = {
            "quantum_comm": False,
            "co_located": True,
            "n_nodes": n_nodes,
            "family": family_name,
        }
        if cores_per_qpu is not None:
            qraise_kwargs["cores"] = cores_per_qpu
        # The vQPU backend spec is deployment-specific: read it from
        # POLYPUS_CUNQA_BACKEND (a path to a CUNQA backend JSON) rather than a
        # hardcoded absolute path. Unset -> let CUNQA choose its own default.
        backend = os.getenv("POLYPUS_CUNQA_BACKEND")
        if backend:
            qraise_kwargs["backend"] = backend

        family = qraise(n, t, **qraise_kwargs)
        return family

    def drop_qpus(self, **kwargs) -> object:
        # Contract C-1: the Rust side (`drop_qpus` in
        # crates/polypus/src/infrastructure/cunqa.rs) passes the `family`
        # handle returned by `get_qpus`/`qraise`, not a SLURM job id. CUNQA's
        # `qdrop(*families)` accepts family names only (it explicitly does NOT
        # take a job id), so `family` is what must be forwarded here.
        family = kwargs["family"]
        qdrop(family)
        return None

    def run_qcs(self, **args) -> object:

        family_id = args["family_id"]
        # UNVERIFIED (issue #218): contract C-3 reads a circuit without
        # measurements out on the full quantum register, and the local Aer path
        # adds that read-out itself (`local._with_full_readout`) because Aer
        # returns no counts for such a circuit. Whether CUNQA's QPUs return
        # counts for it, and in which key format, cannot be checked without a
        # CUNQA install, so this path forwards the circuit unchanged rather than
        # guess; `local._with_full_readout` is the helper to reuse once it is
        # known.
        #
        # Native polypus circuits arrive as OpenQASM 2.0 strings; CUNQA QPUs
        # currently consume QuantumCircuit objects, so parse here, then lower
        # what CUNQA's converter does not take.
        qcs = [
            _lower_to_cunqa_basis(
                QuantumCircuit.from_qasm_str(qc) if isinstance(qc, str) else qc
            )
            for qc in args["qcs"]
        ]
        shots = args["shots"]

        seed = args.get("seed", None)

        # Simulation method for the vQPUs' Aer backend. The Rust side always
        # sends it (`CunqaBackend::run_circuits` sets kwargs["sim_method"]),
        # and CUNQA consumes it as the per-job `run_config["method"]`, whose
        # own default is "automatic" (cunqa/qjob.py). Until this was forwarded
        # the caller's choice was silently DROPPED: every CUNQA run simulated
        # with "automatic" no matter what `polypus.train(sim_method=...)`
        # asked for, while the result metadata recorded the requested value.
        # Absent/empty keeps CUNQA's own default rather than guessing one.
        sim_method = args.get("sim_method") or None

        try:
            qpus = get_QPUs(co_located=True, family=family_id)
        except Exception as e:
            raise e

        # Asynchronously run the quantum circuits on the QPUs
        try:
            # Shared per-job run configuration. `method` lands in CUNQA's
            # `run_config` verbatim (QJob merges **run_parameters over its
            # defaults), so the string must be one Aer understands, e.g.
            # "statevector" or "matrix_product_state".
            run_kwargs = {"shots": shots, "transpile": False}
            if sim_method is not None:
                run_kwargs["method"] = sim_method

            qjobs = []
            for i in range(len(qcs)):
                if seed is not None:
                    # `seed` (not `seed_simulator`) is the kwarg CUNQA's own
                    # docs name (reference/api/run_configuration.html).
                    # Offset by QPU index (mirrors the native backend's
                    # `base_seed + i` in native.rs) so each QPU gets a
                    # distinct, deterministic seed instead of every QPU
                    # reproducing the exact same counts; masked to the 63-bit
                    # range Aer accepts, since CUNQA's simulated QPUs are
                    # Aer-based (same reason as local.py's mask).
                    #
                    # UNVERIFIED beyond the kwarg name: the `cunqa` package
                    # isn't installed anywhere this can be exercised, and
                    # CESGA-Quantum-Spain/cunqa@2.3.0 (the version README.md
                    # pins) has no `QPU.run()` method at all (only
                    # `.execute()`) and no `cunqa.qutils` module (the legacy
                    # `running_functions.run_qc_in_qpu` path still imports it,
                    # see #212) — this whole integration may not run at all
                    # against that version.
                    qjob = run(
                        qcs[i],
                        qpus[i],
                        seed=(seed + i) & 0x7FFFFFFFFFFFFFFF,
                        **run_kwargs,
                    )
                else:
                    qjob = run(qcs[i], qpus[i], **run_kwargs)
                qjobs.append(qjob)

            results = gather(qjobs)
            counts = [result.counts for result in results]
            return counts
        except Exception as e:
            raise e
