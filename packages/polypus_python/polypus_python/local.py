
from qiskit_aer import AerSimulator

from .infrastructure import Infraestructure

class Local(Infraestructure):
    """Local AerSimulator backend.

    Note on parallelism: AerSimulator holds the Python GIL throughout the
    entire simulation, so thread-level parallelism (ThreadPoolExecutor) does
    not provide speedup.  True QPU parallelism requires separate OS processes,
    as provided by the CUNQA distributed infrastructure.  For local runs,
    n_qpus distributes shots across multiple Aer calls sequentially; the
    overhead of extra calls may slightly increase wall-clock time compared
    to a single call with the full shot count.
    """

    def get_qpus(self, **kwargs) -> object:
        pass

    def run_qcs(self, **args) -> object:
        id = args["id"]
        backend = args["backend"] 
        qcs = args["qcs"]
        shots = args["shots"]
        sim_method = args.get("sim_method", "automatic")
        noise_model = args.get("noise_model", None)

        results = []
        try:
            sim = AerSimulator(method=sim_method)
            for qc in qcs:
                qjob = sim.run(qc, shots=shots, transpile=False, noise_model=noise_model)
                results.append(qjob.result().get_counts(0))
            return results
        except Exception as e:
            raise e

