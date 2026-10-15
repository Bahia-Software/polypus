"""
Lowering of instructions outside Aer's basis on the local backend (issue #252).

``ch``, ``rccx``, ``rc3x``, ``c3sqrtx``, ``u0`` and the calls of declared
``gate`` blocks are in the Polypus vocabulary (C-2) and run on the native
backend, but Aer does not unroll them (``AerError: unknown instruction``). The
local backend therefore lowers a circuit into Aer's basis just before handing
it to Aer, and only when it holds such an instruction: a circuit already in
Aer's basis reaches Aer as it is. The lowered circuit is the one
``transpile(qc, AerSimulator(), optimization_level=0)`` gives, without its
layout stage.
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


def _forbid_lowering(monkeypatch):
    import polypus_python.local as local

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a circuit already in Aer's basis must not be lowered")

    monkeypatch.setattr(local, "_rewrite_into_aer_basis", forbidden)


class TestCircuitInBasisIsUntouched:
    def test_aer_receives_the_same_objects(self, monkeypatch):
        from polypus_python.local import Local
        from qiskit_aer import AerSimulator

        _forbid_lowering(monkeypatch)
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
        original = local._rewrite_into_aer_basis

        def spy(qc, *args, **kwargs):
            lowered.append(qc)
            return original(qc, *args, **kwargs)

        monkeypatch.setattr(local, "_rewrite_into_aer_basis", spy)
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

        _forbid_lowering(monkeypatch)
        qc = _bell()
        direct = (
            AerSimulator()
            .run(qc, shots=SHOTS, seed_simulator=SEED)
            .result()
            .get_counts()
        )
        assert _run(qc) == {k.replace(" ", ""): v for k, v in direct.items()}

    def test_a_qasm_string_in_basis_is_not_lowered_either(self, monkeypatch):
        _forbid_lowering(monkeypatch)
        counts = _run(H + "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q -> c;\n")
        assert counts == {"1": SHOTS}

    def test_noise_model_run_of_a_circuit_in_basis_does_not_change(self, monkeypatch):
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        _forbid_lowering(monkeypatch)
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

        _forbid_lowering(monkeypatch)
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
        original = local._rewrite_into_aer_basis

        def spy(qc, *args, **kwargs):
            lowered.append(qc)
            return original(qc, *args, **kwargs)

        monkeypatch.setattr(local, "_rewrite_into_aer_basis", spy)
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

        _forbid_lowering(monkeypatch)
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

        _forbid_lowering(monkeypatch)
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

    @pytest.mark.parametrize("nesting", ["top", "for_loop", "if_test"])
    def test_box_is_refused_with_one_message(self, nesting):
        """``box`` cannot be rewritten into Aer's basis. It is refused before
        the rewrite, at any depth, with the ``TranspilerError`` ``transpile``
        raises for a top-level one: the basis translator's own error listed
        the circuit's operations and the whole basis in an order that changed
        from call to call (issue #307)."""
        import polypus
        from qiskit import QuantumCircuit
        from qiskit.transpiler.exceptions import TranspilerError

        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)
        if nesting == "top":
            with qc.box():
                qc.ch(0, 1)
        elif nesting == "for_loop":
            with qc.for_loop(range(2)):
                with qc.box():
                    qc.ch(0, 1)
        else:
            with qc.if_test((qc.clbits[0], 0)):
                with qc.box():
                    qc.ch(0, 1)
        qc.measure(1, 1)
        messages = set()
        for _ in range(3):
            with pytest.raises(polypus.BackendError) as info:
                _run(qc)
            assert type(info.value.__cause__) is TranspilerError
            messages.add(str(info.value))
        assert messages == {
            'qiskit.transpiler.exceptions.TranspilerError: "The control-flow '
            "construct 'box' is not supported by the backend.\""
        }


def _received_by_aer(monkeypatch, circuit, **kwargs):
    """The circuits ``AerSimulator.run`` receives when ``circuit`` runs."""
    from qiskit_aer import AerSimulator

    received = []
    original_run = AerSimulator.run

    def spy(self, circuits, *args, **kwargs):
        received.extend(circuits)
        return original_run(self, circuits, *args, **kwargs)

    monkeypatch.setattr(AerSimulator, "run", spy)
    _run(circuit, shots=10, **kwargs)
    return received


class TestLoweredProgram:
    @pytest.mark.parametrize("case", CASES)
    def test_aer_receives_what_transpile_against_aer_gives(self, case, monkeypatch):
        """Same circuit as the full ``transpile`` against Aer's target, gate
        for gate, not only the same counts."""
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        qc = QuantumCircuit.from_qasm_str(_program(case))
        expected = transpile(qc, AerSimulator(), optimization_level=0)
        assert _received_by_aer(monkeypatch, qc) == [expected]

    def test_a_multi_controlled_x_aer_runs_is_kept(self, monkeypatch):
        """``mcx`` is in Aer's basis but not a Qiskit standard gate: lowering
        the ``ch`` next to it must not decompose it."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(5, 5)
        qc.x([0, 1, 2])
        qc.ch(0, 4)
        qc.mcx([0, 1, 2], 3)
        qc.measure(range(5), range(5))
        (received,) = _received_by_aer(monkeypatch, qc)
        ops = received.count_ops()
        assert ops["mcx"] == 1 and "ch" not in ops

    @pytest.mark.parametrize("kind", ["qasm_string", "polypus_circuit"])
    def test_a_circuit_wider_than_the_target_is_lowered(self, kind):
        """Aer's target is only as wide as the host memory allows a
        statevector, but a Clifford circuit runs on the stabilizer method at
        any width: lowering must not reject it."""
        import polypus

        n = 64
        lines = [H + "gate pair a,b { cx a,b; }", f"qreg q[{n}];", f"creg c[{n}];"]
        lines.append("h q[0];")
        lines += [f"pair q[{i}],q[{i + 1}];" for i in range(n - 1)]
        lines.append("measure q -> c;")
        src = "\n".join(lines) + "\n"
        circuit = src if kind == "qasm_string" else polypus.Circuit.from_qasm2(src)
        counts = _run(circuit, shots=200)
        assert set(counts) <= {"0" * n, "1" * n} and sum(counts.values()) == 200

    def test_a_wide_non_clifford_circuit_is_lowered_for_mps(self):
        """The reproducer of issue #307: ``ch`` is not Clifford, so this runs
        on ``matrix_product_state``, not on the stabilizer method. It failed
        with ``CircuitTooWideForTarget`` while the lowering had a layout
        stage."""
        from qiskit import QuantumCircuit

        n = 200
        qc = QuantumCircuit(n)
        qc.x(0)
        qc.ch(0, 1)
        qc.measure_all()
        counts = _run(qc, shots=200, sim_method="matrix_product_state")
        zeros = "0" * (n - 2)
        assert set(counts) <= {zeros + "01", zeros + "11"}
        assert sum(counts.values()) == 200

    def test_a_noise_model_does_not_change_the_lowering(self, monkeypatch):
        """A ``noise_model`` narrows ``AerSimulator(noise_model=...).target``
        to the gates it names plus ``cx``, ``id``, ``rz`` and ``sx`` (here
        without ``h``), but Aer still runs every gate of the default basis
        under it (issue #307). So the circuit is lowered into the default
        basis, as without noise: ``h`` reaches Aer as ``h``, not as ``rz`` and
        ``sx``, and the noise falls on the gates the circuit holds."""
        from polypus_python.local import _aer_basis, _rewrite_into_aer_basis
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(0.2, 2), ["cx"])
        assert "h" not in AerSimulator(noise_model=noise).target.operation_names
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.ch(0, 1)
        qc.measure([0, 1], [0, 1])
        (received,) = _received_by_aer(monkeypatch, qc, noise_model=noise)
        assert received == _rewrite_into_aer_basis(qc, _aer_basis())
        assert "h" in received.count_ops()
