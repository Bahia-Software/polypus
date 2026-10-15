import threading
from collections import OrderedDict
from functools import lru_cache

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import CONTROL_FLOW_OP_NAMES
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.result import marginal_distribution
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasisTranslator, HighLevelSynthesis
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
def _aer_basis():
    """The instruction names the default ``AerSimulator()``'s target supports.
    Built on first use, not at import: the extension imports this package at
    runtime. The *default* simulator, whatever ``sim_method`` or
    ``noise_model`` a run asks for: a method-specific basis is smaller (a
    ``noise_model`` cuts it from 98 to 40 gates), and lowering into it would
    rewrite circuits Aer runs as they are or turn a method's own refusal into a
    transpiler error."""
    return frozenset(AerSimulator().target.operation_names)


def _outside_basis(qc, supported):
    """Whether ``qc`` holds an instruction whose name is not in ``supported``, at
    any depth (control-flow bodies included). ``barrier`` is a directive every
    backend accepts. Judged on the names Qiskit parsed the circuit into
    (``rc3x`` is ``rcccx``, ``c3x`` and ``c4x`` are ``mcx``), not on Polypus's.
    Shared by the Aer path and the CUNQA one (``cunqa.py``).

    The names come from ``count_ops``, which Qiskit computes without building a
    ``CircuitInstruction`` per instruction; ``qc.data`` is walked only for the
    control-flow operations Qiskit declares (``CONTROL_FLOW_OP_NAMES``), to look
    into their bodies. One that is not in ``supported`` (for Aer, today ``box``)
    is outside the basis already by its name, before any body is looked at."""
    names = qc.count_ops()
    if any(name != "barrier" and name not in supported for name in names):
        return True
    if CONTROL_FLOW_OP_NAMES.isdisjoint(names):
        return False
    return any(
        _outside_basis(block, supported)
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
    over as it is, neither copied nor rewritten, so its counts do not move.
    """
    supported = _aer_basis()
    if not _outside_basis(qc, supported):
        return qc
    return _rewrite_into_aer_basis(qc, supported)


def _rewrite_into_aer_basis(qc, supported):
    """Synthesis and basis translation into ``supported``, nothing else: the
    translation stage of ``transpile(qc, AerSimulator(), optimization_level=0)``,
    which gives the same circuit, without its layout stage. That stage was a
    fixed cost on every lowered call, larger than the rewriting itself, and it
    caps the width at the target's qubit count (derived from the host memory),
    so it rejected wide circuits Aer runs, Clifford ones on the stabilizer
    method among them. The names are Aer's own, not only Qiskit's standard
    gates: ``mcx`` (Qiskit's ``c3x``/``c4x``) stays ``mcx``. A fresh pass
    manager per call shares no state between threads. Returns a new circuit:
    the caller's is never modified."""
    names = sorted(supported)
    return PassManager(
        [
            HighLevelSynthesis(
                equivalence_library=SessionEquivalenceLibrary, basis_gates=names
            ),
            BasisTranslator(SessionEquivalenceLibrary, names),
        ]
    ).run(qc)


class _LoweredQasmCache:
    """The lowered circuits of the last OpenQASM programs that needed lowering,
    keyed by their text. Least recently used first out, bounded both in entries
    and in total instructions; a circuit larger than the instruction bound is
    not kept. Thread-safe: a lock guards every access, never the lowering
    itself, so two threads missing the same text at once both lower it (to the
    same circuit) and the second store overwrites the first."""

    def __init__(self, max_entries=64, max_instructions=200_000):
        self._max_entries = max_entries
        self._max_instructions = max_instructions
        self._entries = OrderedDict()
        self._instructions = 0
        self._lock = threading.Lock()

    def __len__(self):
        with self._lock:
            return len(self._entries)

    def get(self, text):
        with self._lock:
            lowered = self._entries.get(text)
            if lowered is not None:
                self._entries.move_to_end(text)
            return lowered

    def put(self, text, lowered):
        size = len(lowered.data)
        if size > self._max_instructions:
            return
        with self._lock:
            previous = self._entries.pop(text, None)
            if previous is not None:
                self._instructions -= len(previous.data)
            self._entries[text] = lowered
            self._instructions += size
            while (
                len(self._entries) > self._max_entries
                or self._instructions > self._max_instructions
            ):
                _, evicted = self._entries.popitem(last=False)
                self._instructions -= len(evicted.data)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._instructions = 0


_LOWERED_QASM = _LoweredQasmCache()


def _for_aer(qc):
    """The circuit Aer runs for ``qc``: a ``QuantumCircuit`` or an OpenQASM 2.0
    string, lowered into Aer's basis when it needs it (``_lower_to_aer_basis``).

    Circuits built with the native Rust layer (``polypus.Circuit``) cross the
    FFI boundary as QASM text and are parsed here, at the last possible moment,
    because AerSimulator needs ``QuantumCircuit`` instances. Parsing builds the
    instructions anew, the definitions of declared gates included, and
    unrolling those definitions is most of the lowering's cost (it roughly
    doubles per qubit in MQT Bench's Grover files). So a text that needed
    lowering keeps its lowered circuit in ``_LOWERED_QASM``: running the same
    program again, or on several QPUs, neither parses nor lowers it again. The
    parse is deterministic, so the circuit kept is the one lowering the text
    afresh gives. It is shared between calls and never modified: only Aer reads
    it, and ``_with_full_readout`` copies before adding its register. A text
    already in the basis is not kept, so a loop of changing angles does not
    evict the programs that need lowering. A ``QuantumCircuit`` is mutable and
    is lowered on each call; Qiskit keeps the definitions it built on the
    caller's instructions, so lowering it again costs much less."""
    if not isinstance(qc, str):
        return _lower_to_aer_basis(qc)
    lowered = _LOWERED_QASM.get(qc)
    if lowered is not None:
        return lowered
    parsed = QuantumCircuit.from_qasm_str(qc)
    lowered = _lower_to_aer_basis(parsed)
    if lowered is not parsed:
        _LOWERED_QASM.put(qc, lowered)
    return lowered


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
        prepared = [_with_full_readout(_for_aer(qc)) for qc in args["qcs"]]
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
