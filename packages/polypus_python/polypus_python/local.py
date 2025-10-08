
from qiskit_aer import AerSimulator

from .infrastructure import Infraestructure

class Local(Infraestructure):

    def run_qc(self, **args) -> object:
        id = args["id"]
        backend = args["backend"] 
        qc = args["qc"]
        shots = args["shots"]

        try:
            backend = AerSimulator()
            qjob = backend.run(qc, shots = shots, transpile=False)
            return qjob.result().get_counts(qc)
        except Exception as e:
            raise e

