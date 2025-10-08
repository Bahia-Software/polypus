
from qiskit_aer import AerSimulator
import os, sys

from .infrastructure import Infraestructure
from cunqa import getQPUs, gather

class Cunqa(Infraestructure):

    def run_qc(self, **args) -> object:

        family_id = args["family_id"]
        backend = args["backend"] 
        qc = args["qc"]
        shots = args["shots"]

        sys.path.append(os.getenv("HOME"))
        try:
            qpus = getQPUs(local=False, family=family_id)
        except Exception as e:
            raise e

        # Asynchronously run the quantum circuits on the QPUs
        try:
            qjobs = []
            for i in range(len(qpus)):
                qjob = qpus[i].run(qc, shots = shots, transpile=False)
                qjobs.append(qjob)
            
            results = gather(qjobs)
            counts = [result.counts for result in results]
            return counts
        except Exception as e:
            raise e