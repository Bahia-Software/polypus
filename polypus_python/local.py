from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.result import marginal_distribution
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


def _has_measurement(qc):
    """Whether ``qc`` contains a ``measure`` anywhere, control-flow bodies
    included. Scanned from the end: measurements are terminal in the common
    case, so a measured circuit is usually recognised after a few steps."""
    for instruction in reversed(qc.data):
        operation = instruction.operation
        if operation.name == "measure":
            return True
        if any(_has_measurement(block) for block in getattr(operation, "blocks", ())):
            return True
    return False


def _with_full_readout(qc):
    """Return the circuit Aer should run for ``qc`` and the classical bits to
    read its counts from (``None``: every bit, as Qiskit reports them).

    Contract C-3: a circuit with **no measurement instruction** is read out on
    the full quantum register, like the native backend does, whatever classical
    registers it declares. Aer returns no counts at all for such a circuit, so
    a copy measures every qubit into a register of its own, and only that
    register's bits make up the key: ``num_qubits`` wide, qubit 0 rightmost.
    ``measure_all`` is not used because it adds its register next to the
    declared ones, widening the key to ``num_qubits + num_clbits``. The
    caller's circuit is never modified.
    """
    if _has_measurement(qc):
        return qc, None
    taken = {creg.name for creg in qc.cregs}
    name = "meas"
    suffix = 0
    while name in taken:
        suffix += 1
        name = f"meas{suffix}"
    readout = ClassicalRegister(qc.num_qubits, name)
    measured = qc.copy()
    measured.add_register(readout)
    measured.measure(measured.qubits, readout)
    return measured, [measured.find_bit(bit).index for bit in readout]


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
        prepared = [
            _with_full_readout(qc) for qc in _ensure_quantum_circuits(args["qcs"])
        ]
        qcs = [qc for qc, _ in prepared]
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

        # Qiskit space-separates the keys per ClassicalRegister ("0 1 0") when a
        # circuit declares several; contract C-3 wants one flat bitstring. The
        # groups are already in clbit order, so dropping the spaces is enough.
        # A full read-out keeps only the bits of the register it added.
        counts = []
        for i, (_, readout) in enumerate(prepared):
            raw = result.get_counts(i)
            if readout is None:
                counts.append({key.replace(" ", ""): n for key, n in raw.items()})
            else:
                counts.append(dict(marginal_distribution(raw, readout)))
        return counts
