import os
import sys

from qiskit import QuantumCircuit

# CUNQA is an optional, site-installed dependency (see the `cunqa` extra in
# polypus_python's pyproject). Prefer a normal import -- installed via that extra
# or already on PYTHONPATH. As an explicit escape hatch for a non-packaged site
# install, honour POLYPUS_CUNQA_PATH (a documented, deployment-specific location)
# instead of the previous unconditional `sys.path.append(os.getenv("HOME"))`,
# which was non-portable and appended `None` to sys.path when HOME was unset.
_cunqa_path = os.getenv("POLYPUS_CUNQA_PATH")
if _cunqa_path and _cunqa_path not in sys.path:
    sys.path.append(_cunqa_path)

from cunqa.qjob import gather  # noqa: E402  (must follow the path setup above)
from cunqa.qpu import get_QPUs, qdrop, qraise, run  # noqa: E402

from .infrastructure import Infraestructure  # noqa: E402


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
        # Native polypus circuits arrive as OpenQASM 2.0 strings; CUNQA QPUs
        # currently consume QuantumCircuit objects, so parse here.
        qcs = [
            QuantumCircuit.from_qasm_str(qc) if isinstance(qc, str) else qc
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
                    # `.execute()`) and no `cunqa.qutils` module, which this
                    # file already imports from above — this whole
                    # integration may not run at all against that version.
                    # See the follow-up task for reconciling this file with
                    # the real, deployed CUNQA API.
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
