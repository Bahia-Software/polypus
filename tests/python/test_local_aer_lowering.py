"""
Lowering of instructions outside Aer's basis on the local backend (issue #252).

``ch``, ``rccx``, ``rc3x``, ``c3sqrtx``, ``u0`` and the calls of declared
``gate`` blocks are in the Polypus vocabulary (C-2) and run on the native
backend, but Aer does not unroll them (``AerError: unknown instruction``). The
local backend therefore lowers a circuit with ``qiskit.transpile`` just before
handing it to Aer, and only when it holds such an instruction: a circuit
already in Aer's basis reaches Aer as it is.
"""

import math

import numpy as np
import pytest

pytestmark = pytest.mark.integration

SEED = 4242
SHOTS = 4000
N_QUBITS = 5
# Tail probability allowed on each side when comparing a sampled count with the
# binomial it should follow: P(X <= k) and P(X >= k) must each be at least this
# for the count k to be accepted. 3e-7 is the one-sided tail of 5 standard
# deviations, so the two-sided false-failure rate per outcome is below 1e-6.
TAIL = 3e-7

H = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

# name -> (declarations, statement). Operands are non-ascending and the input
# state is generic, so a swapped control/target changes the distribution.
CASES = {
    "ch": ("", "ch q[0],q[2];"),
    "rccx": ("", "rccx q[4],q[0],q[2];"),
    "rc3x": ("", "rc3x q[3],q[1],q[4],q[0];"),
    "c3sqrtx": ("", "c3sqrtx q[4],q[3],q[1],q[2];"),
    # Qiskit reads u0(n) as "idle for n identity slots": an integer argument.
    "u0": ("", "u0(2) q[4];"),
    "declared_gate": (
        "gate mixer(t) a,b,c { ch a,b; rz(t) c; ccx c,b,a; }\n",
        "mixer(0.7) q[3],q[0],q[2];",
    ),
}
INPUTS = ["polypus_circuit", "qasm_string", "quantum_circuit"]


def _program(case, measure=True):
    declarations, statement = CASES[case]
    lines = [H + declarations + f"qreg q[{N_QUBITS}];"]
    if measure:
        lines.append(f"creg c[{N_QUBITS}];")
    for q in range(N_QUBITS):
        lines.append(f"ry({0.3 + 0.4 * q!r}) q[{q}];")
        lines.append(f"rz({0.2 + 0.5 * q!r}) q[{q}];")
    lines.append(statement)
    if measure:
        lines.append("measure q -> c;")
    return "\n".join(lines) + "\n"


def _circuit(case, kind):
    import polypus
    from qiskit import QuantumCircuit

    src = _program(case)
    return {
        "polypus_circuit": lambda: polypus.Circuit.from_qasm2(src),
        "qasm_string": lambda: src,
        "quantum_circuit": lambda: QuantumCircuit.from_qasm_str(src),
    }[kind]()


def _run(circuit, backend="aer", shots=SHOTS, **kwargs):
    import polypus

    return polypus.run_quantum_circuit(
        circuit,
        shots=shots,
        infrastructure="local",
        backend=backend,
        seed=SEED,
        **kwargs,
    ).counts[0]


def _transpiled_reference(src):
    """Qiskit parse -> transpile (optimisation off) -> Aer, same seed: what
    ``_aer_counts`` in ``test_qasm_gate_equivalence.py`` computes."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    qc = transpile(QuantumCircuit.from_qasm_str(src), sim, optimization_level=0)
    return sim.run(qc, shots=SHOTS, seed_simulator=SEED).result().get_counts()


def _acceptable_counts(p):
    """Smallest and largest count ``k`` of ``Binomial(SHOTS, p)`` whose lower
    tail ``P(X <= k)`` and upper tail ``P(X >= k)`` are both at least ``TAIL``.
    Exact, so it holds for a small ``p`` too, where the normal approximation
    does not. A deterministic outcome (p = 1) accepts only ``SHOTS``."""
    if p >= 1 - 1e-12:
        return SHOTS, SHOTS
    log_p, log_q = math.log(p), math.log1p(-p)
    log_fact = math.lgamma(SHOTS + 1)
    pmf = [
        math.exp(
            log_fact
            - math.lgamma(k + 1)
            - math.lgamma(SHOTS - k + 1)
            + k * log_p
            + (SHOTS - k) * log_q
        )
        for k in range(SHOTS + 1)
    ]
    lower, cumulative = 0, 0.0
    for k, mass in enumerate(pmf):
        cumulative += mass
        if cumulative >= TAIL:
            lower = k
            break
    upper, cumulative = SHOTS, 0.0
    for k in range(SHOTS, -1, -1):
        cumulative += pmf[k]
        if cumulative >= TAIL:
            upper = k
            break
    return lower, upper


def _assert_binomial(counts, probabilities, label):
    """Every outcome's count is one ``Binomial(SHOTS, p)`` produces without a
    tail below ``TAIL`` (see ``_acceptable_counts``). An outcome that cannot
    happen (p = 0) must be absent, and a deterministic one (p = 1) exact."""
    assert sum(counts.values()) == SHOTS
    for key in counts:
        assert key in probabilities, f"{label}: impossible outcome {key}"
    for key, p in probabilities.items():
        lower, upper = _acceptable_counts(p)
        got = counts.get(key, 0)
        assert lower <= got <= upper, (
            f"{label}: outcome {key} has {got} of {SHOTS}, expected "
            f"{SHOTS * p:.1f} (accepted {lower}..{upper})"
        )


def _exact_probabilities(case):
    import polypus

    state = polypus.statevector(polypus.Circuit.from_qasm2(_program(case, False)))
    return {
        format(i, f"0{N_QUBITS}b"): p
        for i, p in enumerate(np.abs(state) ** 2)
        if p > 1e-12
    }


@pytest.mark.parametrize("kind", INPUTS)
@pytest.mark.parametrize("case", CASES)
def test_gate_outside_aer_basis_runs_on_aer(case, kind):
    """Exact counts against the transpiled reference, whatever the input."""
    assert _run(_circuit(case, kind)) == _transpiled_reference(_program(case))


@pytest.mark.parametrize("case", CASES)
def test_aer_and_native_agree_with_the_exact_distribution(case):
    """Aer (lowered) and native sample the distribution the statevector gives.
    Deterministic outcomes must be exact; the others are held to the binomial
    spread, since the two backends draw their shots differently."""
    circuit = _circuit(case, "polypus_circuit")
    probabilities = _exact_probabilities(case)
    _assert_binomial(_run(circuit, "aer"), probabilities, f"aer/{case}")
    _assert_binomial(_run(circuit, "polypus"), probabilities, f"polypus/{case}")


def _bell():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.barrier()
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _forbid_transpile(monkeypatch):
    import polypus_python.local as local

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a circuit already in Aer's basis must not be transpiled")

    monkeypatch.setattr(local, "transpile", forbidden)


class TestCircuitInBasisIsUntouched:
    def test_aer_receives_the_same_objects(self, monkeypatch):
        from polypus_python.local import Local
        from qiskit_aer import AerSimulator

        _forbid_transpile(monkeypatch)
        received = []
        original_run = AerSimulator.run

        def spy(self, circuits, *args, **kwargs):
            received.append(circuits)
            return original_run(self, circuits, *args, **kwargs)

        monkeypatch.setattr(AerSimulator, "run", spy)
        first, second = _bell(), _bell()
        Local().run_qcs(qcs=[first, second], shots=10, seed=1)
        assert len(received) == 1 and len(received[0]) == 2
        assert received[0][0] is first and received[0][1] is second

    def test_mixed_batch_keeps_order_and_lowers_only_what_needs_it(self, monkeypatch):
        import polypus_python.local as local
        from polypus_python.local import Local
        from qiskit import QuantumCircuit

        lowered = []
        original = local.transpile

        def spy(qc, *args, **kwargs):
            lowered.append(qc)
            return original(qc, *args, **kwargs)

        monkeypatch.setattr(local, "transpile", spy)
        outside = QuantumCircuit(2, 2)
        outside.x(0)
        outside.ch(0, 1)
        outside.x(1)
        outside.measure([0, 1], [0, 1])
        in_basis = QuantumCircuit(2, 2)
        in_basis.x(0)
        in_basis.measure([0, 1], [0, 1])
        counts = Local().run_qcs(qcs=[in_basis, outside, in_basis], shots=50, seed=1)
        assert lowered == [outside]
        # x(0) -> "01"; x(0), ch(0,1) leaves q1 random, then x(1) flips it.
        assert counts[0] == counts[2] == {"01": 50}
        assert set(counts[1]) <= {"01", "11"} and sum(counts[1].values()) == 50

    def test_counts_with_seed_match_a_direct_aer_call(self, monkeypatch):
        from qiskit_aer import AerSimulator

        _forbid_transpile(monkeypatch)
        qc = _bell()
        direct = (
            AerSimulator()
            .run(qc, shots=SHOTS, seed_simulator=SEED)
            .result()
            .get_counts()
        )
        assert _run(qc) == {k.replace(" ", ""): v for k, v in direct.items()}

    def test_a_qasm_string_in_basis_is_not_transpiled_either(self, monkeypatch):
        _forbid_transpile(monkeypatch)
        counts = _run(H + "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q -> c;\n")
        assert counts == {"1": SHOTS}

    def test_noise_model_run_of_a_circuit_in_basis_does_not_change(self, monkeypatch):
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        _forbid_transpile(monkeypatch)
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(0.2, 2), ["cx"])
        qc = _bell()
        direct = (
            AerSimulator(noise_model=noise)
            .run(qc, shots=SHOTS, seed_simulator=SEED)
            .result()
            .get_counts()
        )
        got = _run(qc, noise_model=noise)
        assert got == {k.replace(" ", ""): v for k, v in direct.items()}
        # The noise really acts: the noiseless Bell state has no "01"/"10".
        assert got.get("01", 0) + got.get("10", 0) > 0


class TestLoweringDetails:
    def test_the_users_quantum_circuit_is_not_modified(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.ch(0, 1)
        qc.measure([0, 1], [0, 1])
        before = qc.copy()
        data_before = list(qc.data)
        _run(qc)
        assert qc == before
        assert list(qc.data) == data_before
        assert [i.operation.name for i in qc.data] == ["x", "ch", "measure", "measure"]

    def test_gate_inside_control_flow_is_lowered(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], 1)):
            qc.ch(0, 1)
        qc.measure(1, 1)
        counts = _run(qc)
        # q0 is 1 exactly when c0 is, and only then does `ch` act on q1.
        assert set(counts) == {"00", "01", "11"}
        assert sum(counts.values()) == SHOTS

    def test_barrier_inside_a_control_flow_body_is_not_outside_the_basis(
        self, monkeypatch
    ):
        from qiskit import QuantumCircuit

        _forbid_transpile(monkeypatch)
        qc = QuantumCircuit(1, 1)
        with qc.for_loop(range(2)):
            qc.barrier()
            qc.x(0)
        qc.measure(0, 0)
        # x twice: back to |0>.
        assert _run(qc) == {"0": SHOTS}

    def test_gate_nested_in_two_levels_of_control_flow_is_lowered(self, monkeypatch):
        import polypus_python.local as local
        from qiskit import QuantumCircuit

        lowered = []
        original = local.transpile

        def spy(qc, *args, **kwargs):
            lowered.append(qc)
            return original(qc, *args, **kwargs)

        monkeypatch.setattr(local, "transpile", spy)
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        with qc.for_loop(range(2)):
            with qc.if_test((qc.clbits[1], 0)):
                qc.ch(0, 1)
        qc.measure([0, 1], [0, 1])
        # c1 is still 0, so `ch` runs twice: H twice leaves q1 at |0>.
        assert _run(qc) == {"01": SHOTS}
        assert lowered == [qc]

    def test_barrier_alone_is_not_outside_the_basis(self, monkeypatch):
        from qiskit import QuantumCircuit

        _forbid_transpile(monkeypatch)
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.barrier()
        qc.measure(0, 0)
        assert _run(qc) == {"1": SHOTS}

    @pytest.mark.parametrize("kind", ["qasm_string", "quantum_circuit"])
    def test_unmeasured_circuit_keeps_the_full_readout(self, kind):
        """C-3: ``num_qubits`` bits, qubit 0 rightmost, whatever classical
        registers are declared; the lowering must not widen or reorder it."""
        from qiskit import QuantumCircuit

        src = (
            H
            + "qreg q[3];\ncreg c[5];\nx q[0];\nx q[1];\nrccx q[0],q[1],q[2];\n"
            + "x q[2];\n"
        )
        circuit = src if kind == "qasm_string" else QuantumCircuit.from_qasm_str(src)
        # q0 = 1, q1 = 1, q2 = rccx flips it to 1 and x flips it back to 0.
        assert _run(circuit) == {"011": SHOTS}
        assert _run(src, "polypus") == {"011": SHOTS}

    def test_measured_circuit_keeps_its_clbit_order(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3, 3)
        qc.x(0)
        qc.x(1)
        qc.rccx(0, 1, 2)
        qc.x(2)
        # c0 <- q2, c1 <- q0, c2 <- q1: 0, 1, 1 -> "c2 c1 c0" = "110".
        qc.measure(2, 0)
        qc.measure(0, 1)
        qc.measure(1, 2)
        assert _run(qc) == {"110": SHOTS}

    def test_method_specific_failure_is_still_the_methods_own(self, monkeypatch):
        """``t`` is in the default basis, so nothing is lowered and the
        stabilizer method refuses it exactly as before: a ``QiskitError`` from
        Aer, not a transpiler error."""
        import polypus
        from qiskit import QuantumCircuit
        from qiskit.exceptions import QiskitError

        _forbid_transpile(monkeypatch)
        qc = QuantumCircuit(1, 1)
        qc.t(0)
        qc.measure(0, 0)
        with pytest.raises(polypus.BackendError) as info:
            _run(qc, sim_method="stabilizer")
        assert type(info.value.__cause__) is QiskitError

    def test_a_transpile_failure_reaches_the_caller_as_backend_error(self):
        """C-1: whatever Qiskit raises while lowering is a ``BackendError``
        chained to the original."""
        import polypus
        from qiskit import QuantumCircuit
        from qiskit.circuit import Instruction
        from qiskit.transpiler.exceptions import TranspilerError

        qc = QuantumCircuit(1, 1)
        qc.append(Instruction("opaque_thing", 1, 0, []), [0])
        qc.measure(0, 0)
        with pytest.raises(polypus.BackendError) as info:
            _run(qc)
        assert isinstance(info.value.__cause__, TranspilerError)
        assert "TranspilerError" in str(info.value)

    def test_noise_model_does_not_change_the_lowering(self, monkeypatch):
        """Pinning test, green before and after issue #307 by design: it fails
        if the lowering ever targets the noisy simulator. A ``noise_model``
        narrows ``AerSimulator(noise_model=...).target`` (here, without ``h``)
        but not what Aer runs, so the circuit is lowered against the default
        simulator: ``h`` reaches Aer as ``h``, not rewritten to ``u2``, and the
        noise falls on the gates the user wrote."""
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(0.2, 2), ["cx"])
        assert "h" not in AerSimulator(noise_model=noise).target.operation_names
        received = []
        original_run = AerSimulator.run

        def spy(self, circuits, *args, **kwargs):
            received.extend(circuits)
            return original_run(self, circuits, *args, **kwargs)

        monkeypatch.setattr(AerSimulator, "run", spy)
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.ch(0, 1)
        qc.measure([0, 1], [0, 1])
        got = _run(qc, noise_model=noise)
        assert len(received) == 1
        ops = received[0].count_ops()
        assert ops.get("h", 0) >= 1 and "u2" not in ops and "ch" not in ops
        lowered = transpile(qc, AerSimulator(), optimization_level=0)
        monkeypatch.setattr(AerSimulator, "run", original_run)
        direct = (
            AerSimulator(noise_model=noise)
            .run(lowered, shots=SHOTS, seed_simulator=SEED)
            .result()
            .get_counts()
        )
        assert got == {k.replace(" ", ""): v for k, v in direct.items()}


# Wider than any host's default ``AerSimulator()`` target, which depends on the
# host's memory (issue #307).
WIDE = 200
MPS = {"sim_method": "matrix_product_state"}


def _default_width():
    from qiskit_aer import AerSimulator

    return AerSimulator().target.num_qubits


def _wide_program(case):
    """``_program(case)`` on the first ``N_QUBITS`` of a ``WIDE``-qubit register:
    the other qubits stay at |0>, so the low ``N_QUBITS`` bits of every key follow
    the 5-qubit statevector and the high ones are all 0."""
    declarations, statement = CASES[case]
    lines = [H + declarations + f"qreg q[{WIDE}];", f"creg c[{WIDE}];"]
    for q in range(N_QUBITS):
        lines.append(f"ry({0.3 + 0.4 * q!r}) q[{q}];")
        lines.append(f"rz({0.2 + 0.5 * q!r}) q[{q}];")
    lines.append(statement)
    lines.append("measure q -> c;")
    return "\n".join(lines) + "\n"


def _spy_transpile(monkeypatch):
    """Record every ``transpile`` call ``local.py`` makes (args, kwargs)."""
    import polypus_python.local as local

    calls = []
    original = local.transpile

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(local, "transpile", spy)
    return calls


def _wide_ch():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(WIDE)
    qc.x(0)
    qc.ch(0, 1)
    qc.measure_all()
    return qc


class TestWideCircuits:
    """A circuit wider than the default ``AerSimulator()``'s target is lowered
    against a target with the same instructions and the circuit's width, so a
    ``sim_method`` that holds it runs it (issue #307)."""

    def test_wide_circuit_outside_the_basis_runs_with_mps(self):
        """The reproducer of issue #307: it used to fail with
        ``CircuitTooWideForTarget`` while lowering."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(WIDE)
        qc.x(0)
        qc.x(1)
        qc.rccx(0, 1, 2)
        qc.measure_all()
        assert _run(qc, **MPS) == {"0" * (WIDE - 3) + "111": SHOTS}
        zeros = "0" * (WIDE - 2)
        _assert_binomial(
            _run(_wide_ch(), **MPS), {zeros + "01": 0.5, zeros + "11": 0.5}, "ch"
        )

    @pytest.mark.parametrize("kind", INPUTS)
    @pytest.mark.parametrize("case", CASES)
    def test_every_gate_outside_the_basis_runs_wide(self, case, kind):
        """Every out-of-basis gate and a declared gate, from every input kind:
        the low bits sample the exact 5-qubit distribution, the rest stay 0."""
        import polypus
        from qiskit import QuantumCircuit

        src = _wide_program(case)
        circuit = {
            "polypus_circuit": lambda: polypus.Circuit.from_qasm2(src),
            "qasm_string": lambda: src,
            "quantum_circuit": lambda: QuantumCircuit.from_qasm_str(src),
        }[kind]()
        zeros = "0" * (WIDE - N_QUBITS)
        probabilities = {
            zeros + key: p for key, p in _exact_probabilities(case).items()
        }
        _assert_binomial(_run(circuit, **MPS), probabilities, f"wide/{case}/{kind}")

    def test_wide_gate_inside_control_flow_is_lowered(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(WIDE, 3)
        qc.x(0)
        qc.x(1)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], 1)):
            qc.rccx(0, 1, 2)
            qc.ch(1, WIDE - 1)
            qc.ch(1, WIDE - 1)
        qc.measure(1, 1)
        qc.measure(2, 2)
        # c0 = 1, so the body runs: rccx flips q2, and H twice leaves q199 at |0>.
        assert _run(qc, **MPS) == {"111": SHOTS}

    def test_wide_clifford_lowering_runs_on_stabilizer(self):
        """``dcx``, ``iswap`` and a declared gate of ``h`` and ``cx`` are outside
        Aer's basis and lower to Clifford gates, which ``stabilizer`` takes."""
        from qiskit import QuantumCircuit

        src = (
            H
            + "gate bell a,b { h a; cx a,b; }\n"
            + f"qreg q[{WIDE}];\ncreg c[{WIDE}];\nbell q[4],q[5];\n"
        )
        qc = QuantumCircuit.from_qasm_str(src)
        qc.x(0)
        qc.dcx(0, 1)
        qc.x(2)
        qc.iswap(2, 3)
        qc.measure(qc.qubits, qc.clbits)
        # dcx: |q1 q0> = |01> -> |11> -> |10>; iswap: |q3 q2> = |01> -> i|10>;
        # bell: q4 = q5, each value with probability 1/2.
        zeros = "0" * (WIDE - 6)
        _assert_binomial(
            _run(qc, sim_method="stabilizer"),
            {zeros + "001010": 0.5, zeros + "111010": 0.5},
            "stabilizer",
        )

    def test_lowering_against_the_wide_target_equals_the_narrow_one(self):
        """The wide target changes the width cap, not the policy: on circuits the
        default simulator holds, both lowerings give the same circuit. Besides
        the Polypus gates, every Qiskit standard gate outside Aer's basis (with
        bound angles) and the four control-flow blocks (``if_else``,
        ``for_loop``, ``while_loop``, ``switch_case``), so a Qiskit release
        where the two diverge is caught here."""
        from polypus_python.local import _default_aer, _wide_aer_target
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import Gate
        from qiskit.circuit.library import get_standard_gate_name_mapping

        simulator, supported = _default_aer()
        circuits = [QuantumCircuit.from_qasm_str(_program(case)) for case in CASES]
        polypus_gates = len(circuits)
        for name, gate in sorted(get_standard_gate_name_mapping().items()):
            if not isinstance(gate, Gate) or name in supported:
                continue
            qc = QuantumCircuit(gate.num_qubits, name=name)
            qc.append(gate, range(gate.num_qubits))
            angles = {p: 0.3 + 0.1 * i for i, p in enumerate(qc.parameters)}
            circuits.append(qc.assign_parameters(angles))
        flow = QuantumCircuit(3, 3)
        flow.h(0)
        flow.measure(0, 0)
        with flow.if_test((flow.clbits[0], 1)):
            flow.ch(0, 1)
        with flow.for_loop(range(2)):
            flow.rccx(0, 1, 2)
        flow.measure([1, 2], [1, 2])
        circuits.append(flow)
        loops = QuantumCircuit(3, 3)
        loops.h(0)
        loops.measure(0, 0)
        with loops.while_loop((loops.clbits[0], 1)):
            loops.ch(0, 1)
            loops.measure(0, 0)
        with loops.switch(loops.clbits[1]) as case:
            with case(0):
                loops.ecr(0, 1)
                loops.rccx(0, 1, 2)
            with case(1):
                loops.iswap(2, 0)
        loops.measure([0, 1, 2], [0, 1, 2])
        circuits.append(loops)
        # The sweep is not empty: ``ch``, at least, is a standard gate outside.
        assert len(circuits) > polypus_gates + 2
        for qc in circuits:
            narrow = transpile(qc, simulator, optimization_level=0)
            wide = transpile(
                qc, target=_wide_aer_target(qc.num_qubits), optimization_level=0
            )
            assert wide == narrow, qc.name

    def test_wide_target_has_the_default_instructions_and_the_given_width(self):
        from polypus_python.local import _default_aer, _wide_aer_target

        simulator, _ = _default_aer()
        target = _wide_aer_target(WIDE)
        assert target.num_qubits == WIDE
        assert set(target.operation_names) == set(simulator.target.operation_names)

    def test_a_circuit_the_default_target_holds_takes_the_narrow_path(
        self, monkeypatch
    ):
        """``W`` qubits, ``W`` the default target's width (host-dependent):
        ``transpile(qc, simulator, optimization_level=0)`` with the default
        simulator, exactly the call made before issue #307."""
        from polypus_python.local import _default_aer
        from qiskit import QuantumCircuit

        calls = _spy_transpile(monkeypatch)
        width = _default_width()
        qc = QuantumCircuit(width)
        qc.x(0)
        qc.ch(0, 1)
        qc.measure_all()
        counts = _run(qc, shots=32, **MPS)
        assert set(counts) <= {"0" * (width - 2) + "01", "0" * (width - 2) + "11"}
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[0] is qc and args[1] is _default_aer()[0] and len(args) == 2
        assert kwargs == {"optimization_level": 0}

    def test_one_qubit_wider_takes_the_wide_path(self, monkeypatch):
        from qiskit import QuantumCircuit

        calls = _spy_transpile(monkeypatch)
        width = _default_width() + 1
        qc = QuantumCircuit(width)
        qc.x(0)
        qc.ch(0, 1)
        qc.measure_all()
        counts = _run(qc, shots=32, **MPS)
        assert set(counts) <= {"0" * (width - 2) + "01", "0" * (width - 2) + "11"}
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert len(args) == 1 and args[0] is qc
        assert set(kwargs) == {"target", "optimization_level"}
        assert kwargs["optimization_level"] == 0
        assert kwargs["target"].num_qubits == width

    def test_a_wide_circuit_in_the_basis_is_not_transpiled(self, monkeypatch):
        from qiskit import QuantumCircuit

        _forbid_transpile(monkeypatch)
        qc = QuantumCircuit(WIDE)
        qc.x(0)
        qc.cx(0, WIDE - 1)
        qc.measure_all()
        assert _run(qc, **MPS) == {"1" + "0" * (WIDE - 2) + "1": SHOTS}

    def test_default_sim_method_refuses_for_memory_not_in_the_transpiler(
        self, tmp_path
    ):
        """With ``sim_method`` left at ``automatic``, a wide non-Clifford circuit
        outside the basis is lowered and then refused by Aer's memory check, as
        the same circuit in the basis is: ``polypus.InsufficientMemoryError``,
        no transpiler error. ``POLYPUS_MEM_BUDGET`` is set in a fresh child (see
        ``test_memory_budget.py``) so the refusal does not depend on the host."""
        import os
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent(
            f"""
            import polypus
            from qiskit import QuantumCircuit

            for gate in ("ch", "cx"):
                qc = QuantumCircuit({WIDE})
                qc.h(0)
                getattr(qc, gate)(0, 1)
                qc.t(1)
                qc.measure_all()
                try:
                    polypus.run_quantum_circuit(
                        qc, shots=8, infrastructure="local", backend="aer", seed=1
                    )
                except polypus.PolypusError as exc:
                    assert type(exc) is polypus.InsufficientMemoryError, (gate, exc)
                    assert "Transpiler" not in str(exc), (gate, exc)
                    assert "TooWide" not in str(exc), (gate, exc)
                    assert "insufficient memory" in str(exc).lower(), (gate, exc)
                else:
                    raise AssertionError(gate + ": expected InsufficientMemoryError")
                print("refused:", gate)
            """
        )
        env = os.environ.copy()
        env["POLYPUS_MEM_BUDGET"] = "1G"
        env["XDG_CACHE_HOME"] = str(tmp_path)
        env["POLYPUS_NO_AUTOCALIBRATE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        assert result.returncode == 0, (
            f"child failed (rc={result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
        assert result.stdout.split() == ["refused:", "ch", "refused:", "cx"]
