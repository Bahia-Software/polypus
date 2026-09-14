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
        QuantumCircuit.from_qasm_str(qc) if isinstance(qc, str) else qc for qc in qcs
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
        seed = args.get("seed", None)
        # How many experiments Aer may run in parallel. Aer's default of 0
        # ("auto") spawns one process per experiment and OOMs at high qubit
        # counts, so the caller (the Rust local backend) supplies a value sized to
        # a statevector memory budget; 0 is kept only as the fallback for a direct
        # caller that does not, preserving that path's historical behaviour. Aer
        # seeds each experiment deterministically, so this bound never changes the
        # counts — only peak memory and speed.
        max_parallel_experiments = args.get("max_parallel_experiments", 0)

        # Submit every circuit in one Aer call so the C++ engine can run the
        # experiments in parallel across cores (GIL released) instead of looping
        # one circuit at a time under the GIL.
        sim = AerSimulator(
            method=sim_method,
            noise_model=noise_model,
            max_parallel_experiments=max_parallel_experiments,
        )
        if seed is not None:
            # Rust resolves `seed` from the full u64 range, but Aer's
            # `seed_simulator` takes a signed 64-bit int (rejects values
            # >= 2**63 with a confusing low-level TypeError); mask down to
            # that range rather than truncating precision unevenly.
            result = sim.run(
                qcs, shots=shots, seed_simulator=seed & 0x7FFFFFFFFFFFFFFF
            ).result()
        else:
            result = sim.run(qcs, shots=shots).result()

        return [result.get_counts(i) for i in range(len(qcs))]
