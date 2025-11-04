
from qiskit_aer import AerSimulator

from .infrastructure import Infraestructure

class Local(Infraestructure):

    def get_qpus(self, **kwargs) -> object:
        pass

    def run_qcs(self, **args) -> object:
        id = args["id"]
        backend = args["backend"] 
        qcs = args["qcs"]
        shots = args["shots"]

        results = []
        try:
            backend = AerSimulator()
            for qc in qcs:
                qjob = backend.run(qc, shots = shots, transpile=False)
                results.append(qjob.result().get_counts(qc))
            return results
        except Exception as e:
            raise e

