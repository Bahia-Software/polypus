from functools import lru_cache

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import CONTROL_FLOW_OP_NAMES
from qiskit.result import marginal_distribution
from qiskit_aer import AerSimulator

from .infrastructure import Infraestructure


def _raise_if_out_of_memory(result, max_memory_mb):
    """Raise ``polypus.InsufficientMemoryError`` if Aer refused an experiment for
    lack of memory (issue #215).

    Aer validates each experiment against ``max_memory_mb`` (the Polypus memory
    budget, when the Rust side knows one) for the method it actually picked, and
    reports a refusal as an unsuccessful experiment whose ``status`` reads
    "Insufficient memory to run circuit ...". Surfacing it as the typed Polypus
    class keeps it catchable as ``polypus.PolypusError`` (contract C-1), instead
    of the generic ``QiskitError`` that ``get_counts`` would raise. Any other
    failure is left to ``get_counts``, unchanged.
    """
    for index, experiment in enumerate(result.results):
        status = str(getattr(experiment, "status", ""))
        if not experiment.success and "insufficient memory" in status.lower():
            # Imported here, not at module level: `polypus` (the extension)
            # imports this package at runtime, so a top-level import would be
            # circular.
            from polypus import InsufficientMemoryError

            if max_memory_mb is not None:
                where = (
                    f"the limit is the Polypus memory budget ({max_memory_mb} MiB, "
                    "POLYPUS_MEM_BUDGET when set, else the detected RAM / cgroup "
                    "memory limit minus a safety reserve), passed to Aer as "
                    "max_memory_mb"
                )
                hint = (
                    "Use fewer qubits or a less memory-hungry sim_method, or, if "
                    "more memory really is available, set POLYPUS_MEM_BUDGET "
                    "(e.g. POLYPUS_MEM_BUDGET=64G) to override the budget."
                )
            else:
                where = "the limit is Aer's default (the host's memory)"
                hint = "Use fewer qubits or a less memory-hungry sim_method."
            raise InsufficientMemoryError(
                f"not enough memory for circuit {index} on Aer: {status.strip()} "
                f"({where}). Refused before starting rather than be killed by the "
                f"out-of-memory killer. {hint}"
            )


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


@lru_cache(maxsize=1)
def _default_aer():
    """The default ``AerSimulator()`` and the instruction names its target
    supports. Built on first use, not at import: the extension imports this
    package at runtime. The *default* simulator, whatever ``sim_method`` or
    ``noise_model`` a run asks for: a method-specific basis is smaller (a
    ``noise_model`` cuts it from 98 to 40 gates), and lowering against it would
    rewrite circuits Aer runs as they are or turn a method's own refusal into a
    transpiler error."""
    simulator = AerSimulator()
    return simulator, frozenset(simulator.target.operation_names)


def _outside_aer_basis(qc, supported):
    """Whether ``qc`` holds an instruction Aer's target does not support, at any
    depth (control-flow bodies included). ``barrier`` is a directive Aer always
    accepts. Judged on the names Qiskit parsed the circuit into (``rc3x`` is
    ``rcccx``, ``c3x`` and ``c4x`` are ``mcx``, which Aer does run), not on
    Polypus's.

    The names come from ``count_ops``, which Qiskit computes without building a
    ``CircuitInstruction`` per instruction; ``qc.data`` is walked only for the
    control-flow operations Qiskit declares (``CONTROL_FLOW_OP_NAMES``), to look
    into their bodies. One that is not in Aer's target (today ``box``) is outside
    the basis already by its name, before any body is looked at."""
    names = qc.count_ops()
    if any(name != "barrier" and name not in supported for name in names):
        return True
    if CONTROL_FLOW_OP_NAMES.isdisjoint(names):
        return False
    return any(
        _outside_aer_basis(block, supported)
        for instruction in qc.data
        if instruction.operation.name in CONTROL_FLOW_OP_NAMES
        for block in instruction.operation.blocks
    )


def _lower_to_aer_basis(qc):
    """``qc`` when every instruction is in Aer's basis, else an equivalent copy
    rewritten into it.

    Aer does not unroll what is outside its basis (``ch``, ``rccx``, ``rcccx``,
    ``c3sx``, ``u0``, a ``gate`` declared in the OpenQASM source): it raises
    ``AerError: unknown instruction``. The circuit is lowered here, third
    lowering boundary of contract C-2, and never earlier, so the exported QASM
    stays the program the user wrote. A circuit already in the basis is handed
    over as it is, neither copied nor transpiled, so its counts do not move.
    ``optimization_level=0`` only rewrites; it neither reorders nor merges
    gates. ``transpile`` returns a new circuit: the caller's is never modified.
    """
    simulator, supported = _default_aer()
    if not _outside_aer_basis(qc, supported):
        return qc
    return transpile(qc, simulator, optimization_level=0)


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
        # Lowering comes first: the read-out register is added to the lowered
        # circuit, whose qubits and classical bits keep their order.
        prepared = [
            _with_full_readout(_lower_to_aer_basis(qc))
            for qc in _ensure_quantum_circuits(args["qcs"])
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
        # Per-experiment memory limit (MiB) for Aer's own validation (issue #215),
        # sent by the Rust local backend only when it knows a real memory budget.
        # Passed through only when present: a direct caller without it keeps Aer's
        # default (the host's memory), and 0 would mean "no limit" to Aer.
        max_memory_mb = args.get("max_memory_mb", None)
        sim_options = {}
        if max_memory_mb is not None:
            sim_options["max_memory_mb"] = max_memory_mb

        # Submit every circuit in one Aer call so the C++ engine can run the
        # experiments in parallel across cores (GIL released) instead of looping
        # one circuit at a time under the GIL.
        sim = AerSimulator(
            method=sim_method,
            noise_model=noise_model,
            max_parallel_experiments=max_parallel_experiments,
            **sim_options,
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
        _raise_if_out_of_memory(result, max_memory_mb)

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
