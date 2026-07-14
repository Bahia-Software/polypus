import os
import sys

from qiskit import QuantumCircuit

sys.path.append(os.getenv("HOME"))

from cunqa.qjob import gather
from cunqa.qutils import get_QPUs, qdrop, qraise

from .infrastructure import Infraestructure


class Cunqa(Infraestructure):
    def get_qpus(self, **kwargs) -> object:

        n = kwargs["n"]
        t = kwargs["t"]
        n_nodes = kwargs["n_nodes"]
        family_name = kwargs["family_name"]
        family = qraise(
            n, t, quantum_comm=False, cloud=True, n_nodes=n_nodes, family=family_name
        )
        return family

    def drop_qpus(self, **kwargs) -> object:
        slurm_job_id = kwargs["slurm_job_id"]
        qdrop(slurm_job_id)
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

        sys.path.append(os.getenv("HOME"))
        try:
            qpus = get_QPUs(local=False, family=family_id)
        except Exception as e:
            raise e

        # Asynchronously run the quantum circuits on the QPUs
        try:
            qjobs = []
            for i in range(len(qcs)):
                qjob = qpus[i].run(qcs[i], shots=shots, transpile=False)
                qjobs.append(qjob)

            results = gather(qjobs)
            counts = [result.counts for result in results]
            return counts
        except Exception as e:
            raise e
