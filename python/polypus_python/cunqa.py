import os
import sys
from functools import lru_cache

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import get_standard_gate_name_mapping

# CUNQA is an optional, site-installed dependency (not on PyPI, so the root
# pyproject.toml declares no `cunqa` extra). Supported range: cunqa>=2.4,<4. The
# seal sends `qraise(cores_per_qpu=...)`, the name CUNQA introduced in 2.4.0 (2.3.0
# called it `cores` and has no `**kwargs`, so it rejects the new name); the
# signatures at 2.3.0, 2.4.0, 3.0.0 and 3.1.0 are pinned by
# tests/python/test_cunqa_contract.py. Prefer a normal import -- already installed or on
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
from .local import _c3_counts, _outside_basis, _with_full_readout  # noqa: E402


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
        # kwargs["cores_per_qpu"]) and goes to qraise under the same name, as
        # CUNQA >= 2.4 spells it (2.3.0's `cores` is unsupported). qraise has no
        # **kwargs, so a name it does not know raises TypeError.
        cores_per_qpu = kwargs.get("cores_per_qpu")

        qraise_kwargs = {
            "quantum_comm": False,
            "co_located": True,
            "n_nodes": n_nodes,
            "family": family_name,
        }
        if cores_per_qpu is not None:
            qraise_kwargs["cores_per_qpu"] = cores_per_qpu
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
        # Native polypus circuits arrive as OpenQASM 2.0 strings; CUNQA QPUs
        # currently consume QuantumCircuit objects, so parse here, then lower
        # what CUNQA's converter does not take.
        #
        # Contract C-3 reads a circuit without measurements out on the full
        # quantum register. CUNQA's vQPUs return no counts for one (CUNQA 3.0.0
        # fails the job with "key 'counts' not found", #253), so the read-out
        # is added here as on the local Aer path, after lowering.
        prepared = [
            _with_full_readout(
                _lower_to_cunqa_basis(
                    QuantumCircuit.from_qasm_str(qc) if isinstance(qc, str) else qc
                )
            )
            for qc in args["qcs"]
        ]
        qcs = [qc for qc, _ in prepared]
        shots = args["shots"]

        seed = args.get("seed", None)

        # Where these circuits start among the family's QPUs: circuit `i` runs on
        # QPU `qpu_offset + i` with seed `seed + qpu_offset + i`. The Rust side
        # sends one call per batch size when the shots split unevenly across the
        # QPUs, and this keeps every replica on its own QPU and seed (#370).
        qpu_offset = args.get("qpu_offset", 0)

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
            for i, qc in enumerate(qcs):
                k = qpu_offset + i
                if seed is not None:
                    # `seed` (not `seed_simulator`) is the kwarg CUNQA's own
                    # docs name (reference/api/run_configuration.html).
                    # Offset by QPU index so each QPU gets a distinct,
                    # deterministic seed instead of every QPU reproducing the
                    # exact same counts; masked to the 63-bit range Aer
                    # accepts, since CUNQA's simulated QPUs are Aer-based (same
                    # reason as local.py's mask) (contract C-7).
                    #
                    # Pinned against a fake CUNQA (tests/python/
                    # test_cunqa_run_flow.py) and checked on CUNQA 3.0.0
                    # (tests/python/test_cunqa_real.py, opt-in).
                    qjob = run(
                        qc,
                        qpus[k],
                        seed=(seed + k) & 0x7FFFFFFFFFFFFFFF,
                        **run_kwargs,
                    )
                else:
                    qjob = run(qc, qpus[k], **run_kwargs)
                qjobs.append(qjob)

            results = gather(qjobs)
            # CUNQA splits the key per classical register with spaces (and,
            # in 3.0.0, at the wrong positions: cut from the left in
            # declaration order, so `creg a[2]; creg b[3];` gives "00 010"
            # for "00010"); `_c3_counts` flattens it (#304).
            return [
                _c3_counts(result.counts, readout)
                for result, (_, readout) in zip(results, prepared)
            ]
        except Exception as e:
            raise e
