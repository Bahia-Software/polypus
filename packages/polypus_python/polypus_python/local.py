
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from .infrastructure import Infraestructure


def _ensure_quantum_circuits(qcs):
    """Accept both Qiskit ``QuantumCircuit`` objects and OpenQASM 2.0 strings.

    Circuits built with the native Rust layer (``polypus.Circuit``) cross the
    FFI boundary as QASM text; they are parsed here, at the last possible
    moment, because AerSimulator needs ``QuantumCircuit`` instances. Future
    backends that speak QASM natively can skip this step entirely.
    """
    return [
        QuantumCircuit.from_qasm_str(qc) if isinstance(qc, str) else qc
        for qc in qcs
    ]


class Local(Infraestructure):
    """Local AerSimulator backend.

    Note on parallelism: a single Python-level Aer call holds the GIL for its
    whole duration, so dispatching circuits one-by-one (or via Python threads)
    runs them sequentially.  Instead, all circuits are submitted in a *single*
    ``AerSimulator.run`` call with ``max_parallel_experiments=0`` (use all
    available threads): Aer's C++ engine then executes the experiments in
    parallel across CPU cores while releasing the GIL.  This is the only real
    parallelism available for local simulation; true QPU parallelism across
    processes is provided by the CUNQA distributed infrastructure.
    """

    def get_qpus(self, **kwargs) -> object:
        pass

    def run_qcs(self, **args) -> object:
        qcs = _ensure_quantum_circuits(args["qcs"])
        shots = args["shots"]
        sim_method = args.get("sim_method", "automatic")
        noise_model = args.get("noise_model", None)

        # Submit every circuit in one Aer call so the C++ engine can run the
        # experiments in parallel across cores (GIL released) instead of looping
        # one circuit at a time under the GIL.
        sim = AerSimulator(
            method=sim_method,
            noise_model=noise_model,
            max_parallel_experiments=0,
        )
        result = sim.run(qcs, shots=shots, transpile=False).result()
        return [result.get_counts(i) for i in range(len(qcs))]

