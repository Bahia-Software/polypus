
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import os, sys
sys.path.append(os.getenv("HOME"))

from .infrastructure import Infraestructure
from cunqa.qutils import get_QPUs, qraise, qdrop
from cunqa.qjob import gather

class Cunqa(Infraestructure):

    def get_qpus(self, **kwargs) -> object:

        n = kwargs["n"]
        t = kwargs["t"]
        n_nodes = kwargs["n_nodes"]
        family_name = kwargs["family_name"]
        family = qraise(n, t, quantum_comm=False, cloud=True, n_nodes=n_nodes, family=family_name)
        return family

    def drop_qpus(self, **kwargs) -> object:
        slurm_job_id = kwargs["slurm_job_id"]
        qdrop(slurm_job_id)
        return None

    def run_qcs(self, **args) -> object:

        family_id = args["family_id"]
        backend = args["backend"] 
        # Native polypus circuits arrive as OpenQASM 2.0 strings; CUNQA QPUs
        # currently consume QuantumCircuit objects, so parse here.
        qcs = [
            QuantumCircuit.from_qasm_str(qc) if isinstance(qc, str) else qc
            for qc in args["qcs"]
        ]
        shots = args["shots"]
        sim_method = args.get("sim_method", "automatic")

        seed = args.get("seed", None)

        sys.path.append(os.getenv("HOME"))
        try:
            qpus = get_QPUs(local=False, family=family_id)
        except Exception as e:
            raise e

        # Asynchronously run the quantum circuits on the QPUs
        try:
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
                    qjob = qpus[i].run(
                        qcs[i],
                        shots=shots,
                        transpile=False,
                        seed=(seed + i) & 0x7FFFFFFFFFFFFFFF,
                    )
                else:
                    qjob = qpus[i].run(qcs[i], shots=shots, transpile=False)
                qjobs.append(qjob)
            
            results = gather(qjobs)
            counts = [result.counts for result in results]
            return counts
        except Exception as e:
            raise e