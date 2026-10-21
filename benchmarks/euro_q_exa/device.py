"""
Euro-Q-Exa (IQM, backend ``EQE1``) device model for the emulator benchmark (#391).

Turns the two files LRZ sends — a calibration snapshot (JSON) and the backend's
coupling map — into what the benchmark needs:

  * a :class:`Device`: the qubits the backend exposes, in its own (compact)
    labels, with each one's calibration and the calibrated CZ edges;
  * a Qiskit :class:`~qiskit.transpiler.Target` with the native gate set
    (``r``, ``cz``, ``measure``) and the calibrated errors, so routing and layout
    selection see the device's weak spots;
  * an Aer :class:`~qiskit_aer.noise.NoiseModel` that reproduces the calibrated
    average gate fidelities;
  * optional idle noise: relaxation on the qubits that wait while others run
    gates, added to a scheduled circuit by :func:`add_idle_noise`.

Labels. The calibration is keyed by the chip's *original* labels (54 qubits),
the backend by *compact* ones (52): qubits that are out of service are dropped
and the rest renumbered in order. A qubit is out of service when its calibration
has no finite T1, T2 or single-qubit fidelity. :func:`build_device` derives the
compact → original mapping from that rule and refuses to build a device unless
the result explains the coupling map exactly: as many qubits as the coupling map
names, and every coupling edge a calibrated edge between active qubits (and vice
versa). A mismatch is a ``ValueError``, never a silently shifted noise model.

Noise model. Follows the recipe of IQM's own fake backends (``qiskit-iqm``'s
``IQMErrorProfile``): per gate, thermal relaxation over the gate's duration
followed by a depolarizing channel whose strength is solved so that the
composed channel's *average gate fidelity* equals the calibrated one. For
``E = D_λ ∘ T`` on dimension ``d`` the process fidelity is
``(1 - λ) F_pro(T) + λ / d²``, so ``λ`` has a closed form. When relaxation alone
is already worse than the calibration, ``λ = 0`` and the gate is reported as
*relaxation-limited* (the model is then pessimistic for it). Readout errors are
per qubit and symmetric: the calibration gives one readout fidelity, read as
``1 - (P(1|0) + P(0|1)) / 2``.

Gates use the echo T2, and every relaxation channel caps T2 at T1 (Aer
0.17.2 cannot run the Kraus channels T2 > T1 needs; see ``_cap_t2``). The Ramsey T2 (1-3 us on many EQE1 qubits) is not usable
inside gates: with it, relaxation alone over a 40-60 ns gate exceeds the
calibrated infidelity of most gates (48 of 52 ``r``, 36 of 82 CZ on the
2026-10-19 snapshot), i.e. it contradicts the device's own benchmarks. The slow
frequency noise it measures matters while a qubit *waits*, which is where idle
noise can use it.

Idle noise. A circuit transpiled against :func:`build_target`'s target with
``scheduling_method="alap"`` carries ``delay`` instructions where qubits wait;
:func:`add_idle_noise` attaches thermal relaxation over each delay (T1 and the
idle T2 of :class:`NoiseAssumptions`: echo by default, Ramsey as a pessimistic
bound for a device without dynamical decoupling). Qubits the circuit never uses
get no delay at all, so Aer can still drop them from the simulation. ALAP keeps
qubits in ``|0>`` (where relaxation does nothing) until their first gate.

Assumptions (see :class:`NoiseAssumptions`): gate durations are not published
for EQE1, so the defaults are IQM Radiance's typical values (PRX 40 ns, CZ
60 ns); ``r`` uses the calibration's per-gate ``1q_fidelity`` (not the
per-Clifford one); the measurement duration only places the terminal
measurements in the schedule and does not change the noise. Crosstalk and
leakage are not modelled.

Usage (summary + validation against a real snapshot):
    python benchmarks/euro_q_exa/device.py \
        --calibration calibration_data_Euro-Q-Exa_20261019_active_qubits.json \
        --coupling-map coupling_map.txt
"""

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Delay, Measure, Parameter
from qiskit.circuit.library import CZGate, RGate
from qiskit.quantum_info import average_gate_fidelity
from qiskit.transpiler import InstructionProperties, Target
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)

# Native instruction names of EQE1 (as its Qiskit target reports them), apart
# from `id`, `reset` and `if_else`, which the benchmark does not use.
ONE_QUBIT_GATE = "r"
TWO_QUBIT_GATE = "cz"
# Schedule resolution. Durations are whole nanoseconds, so delays come out
# exact (a target without `dt` schedules in float seconds and leaves ~1e-23 s
# rounding delays behind).
DT_S = 1e-9


@dataclass(frozen=True)
class Calibration:
    """A calibration snapshot, keyed by the chip's original qubit labels.

    Times are in microseconds, fidelities in [0, 1]; a missing value is NaN.
    ``cz_fidelity`` keys are sorted pairs ``(a, b)`` with ``a < b``.
    """

    backend_name: str
    timestamp: str
    t1_us: tuple[float, ...]
    t2_echo_us: tuple[float, ...]
    t2_ramsey_us: tuple[float, ...]
    one_qubit_fidelity: tuple[float, ...]
    readout_fidelity: tuple[float, ...]
    cz_fidelity: dict[tuple[int, int], float]

    @property
    def num_qubits(self) -> int:
        return len(self.t1_us)


def load_calibration(path) -> Calibration:
    """Read LRZ's calibration JSON (``calibration_data_Euro-Q-Exa_*.json``).

    Its ``T2`` field is the Ramsey T2, ``T2_echo`` the echo one. The file holds
    bare ``NaN`` tokens for out-of-service qubits, which Python's ``json`` reads.
    """
    raw = json.loads(Path(path).read_text())
    fields = ("T1", "T2", "T2_echo", "1q_fidelity", "readout_fidelity")
    lengths = {name: len(raw[name]) for name in fields}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"calibration per-qubit fields differ in length: {lengths}")
    cz = {}
    for key, fidelity in raw["two_q_fidelity"].items():
        a, b = (int(x) for x in key.split(","))
        cz[(min(a, b), max(a, b))] = float(fidelity)
    return Calibration(
        backend_name=raw.get("backend_name", ""),
        timestamp=raw.get("timestamp", ""),
        t1_us=tuple(float(x) for x in raw["T1"]),
        t2_echo_us=tuple(float(x) for x in raw["T2_echo"]),
        t2_ramsey_us=tuple(float(x) for x in raw["T2"]),
        one_qubit_fidelity=tuple(float(x) for x in raw["1q_fidelity"]),
        readout_fidelity=tuple(float(x) for x in raw["readout_fidelity"]),
        cz_fidelity=cz,
    )


def load_coupling_map(path) -> list[tuple[int, int]]:
    """Read a coupling map: a JSON/Python list of pairs, optionally after a
    ``"<label>:"`` prefix (LRZ's ``Backend EQE1 coupling map: [[0, 1], ...]``)."""
    text = Path(path).read_text().strip()
    if not text.startswith("["):
        text = text.split(":", 1)[1].strip()
    pairs = ast.literal_eval(text)
    return [(int(a), int(b)) for a, b in pairs]


@dataclass(frozen=True)
class Device:
    """The backend's qubits in its compact labels, with their calibration.

    ``to_original[c]`` is the original (calibration) label of compact qubit
    ``c``. ``edges`` are the coupling map's pairs as given (compact labels);
    ``cz_fidelity`` is keyed by sorted compact pairs.
    """

    calibration: Calibration
    to_original: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    cz_fidelity: dict[tuple[int, int], float]

    @property
    def num_qubits(self) -> int:
        return len(self.to_original)

    def _per_qubit(self, values) -> np.ndarray:
        return np.array([values[o] for o in self.to_original])

    @property
    def t1_us(self) -> np.ndarray:
        return self._per_qubit(self.calibration.t1_us)

    @property
    def t2_echo_us(self) -> np.ndarray:
        return self._per_qubit(self.calibration.t2_echo_us)

    @property
    def t2_ramsey_us(self) -> np.ndarray:
        return self._per_qubit(self.calibration.t2_ramsey_us)

    @property
    def one_qubit_fidelity(self) -> np.ndarray:
        return self._per_qubit(self.calibration.one_qubit_fidelity)

    @property
    def readout_fidelity(self) -> np.ndarray:
        return self._per_qubit(self.calibration.readout_fidelity)


def _active(cal: Calibration, q: int) -> bool:
    return all(
        math.isfinite(v[q]) for v in (cal.t1_us, cal.t2_echo_us, cal.one_qubit_fidelity)
    )


def build_device(cal: Calibration, coupling_map) -> Device:
    """Match the calibration to the backend's coupling map (module docstring).

    Raises ``ValueError`` when the active qubits do not explain the coupling map
    exactly, or when a qubit the device uses lacks a value the noise model needs.
    """
    to_original = tuple(q for q in range(cal.num_qubits) if _active(cal, q))
    named = {q for edge in coupling_map for q in edge}
    if not named:
        raise ValueError("the coupling map is empty")
    if named != set(range(len(to_original))):
        raise ValueError(
            f"{len(to_original)} active calibrated qubits, but the coupling map "
            f"names {len(named)} qubits labelled {min(named)}..{max(named)}"
        )

    cz = {}
    for a, b in coupling_map:
        pair = (
            min(to_original[a], to_original[b]),
            max(to_original[a], to_original[b]),
        )
        if pair not in cal.cz_fidelity:
            raise ValueError(
                f"coupling edge ({a}, {b}) = original {pair} has no calibrated CZ"
            )
        cz[(min(a, b), max(a, b))] = cal.cz_fidelity[pair]
    active = set(to_original)
    calibrated_active = {
        p for p in cal.cz_fidelity if p[0] in active and p[1] in active
    }
    original_edges = {
        (min(to_original[a], to_original[b]), max(to_original[a], to_original[b]))
        for a, b in coupling_map
    }
    if extra := sorted(calibrated_active - original_edges):
        raise ValueError(f"calibrated CZ edges missing from the coupling map: {extra}")

    device = Device(
        calibration=cal,
        to_original=to_original,
        edges=tuple((int(a), int(b)) for a, b in coupling_map),
        cz_fidelity=cz,
    )
    ro = device.readout_fidelity
    if bad := [c for c in range(device.num_qubits) if not math.isfinite(ro[c])]:
        raise ValueError(f"readout fidelity missing for compact qubits {bad}")
    if bad := [e for e, f in cz.items() if not math.isfinite(f)]:
        raise ValueError(f"CZ fidelity missing for edges {bad}")
    return device


@dataclass(frozen=True)
class NoiseAssumptions:
    """Values the calibration does not give (module docstring). Durations are
    whole nanoseconds; ``idle_t2`` is ``"echo"`` or ``"ramsey"``."""

    r_duration_ns: int = 40
    cz_duration_ns: int = 60
    measure_duration_ns: int = 1000
    idle_t2: str = "echo"

    def __post_init__(self):
        if self.idle_t2 not in ("echo", "ramsey"):
            raise ValueError(
                f"idle_t2 must be 'echo' or 'ramsey', not {self.idle_t2!r}"
            )
        for name in ("r_duration_ns", "cz_duration_ns", "measure_duration_ns"):
            value = getattr(self, name)
            if value <= 0 or value != int(value):
                raise ValueError(f"{name} must be a positive whole number, not {value}")


def build_target(
    device: Device, assumptions: NoiseAssumptions = NoiseAssumptions()
) -> Target:
    """Qiskit target: ``r`` on every qubit, ``cz`` on every coupling edge (both
    orientations: CZ is symmetric), ``measure`` on every qubit, each carrying
    ``error = 1 - calibrated fidelity`` and its assumed duration; plus ``delay``,
    so that ALAP scheduling pads waiting qubits (:func:`add_idle_noise`)."""
    target = Target(num_qubits=device.num_qubits, dt=DT_S)
    r_s = assumptions.r_duration_ns * 1e-9
    cz_s = assumptions.cz_duration_ns * 1e-9
    measure_s = assumptions.measure_duration_ns * 1e-9
    f1 = device.one_qubit_fidelity
    target.add_instruction(
        RGate(Parameter("θ"), Parameter("φ")),
        {
            (c,): InstructionProperties(duration=r_s, error=1 - f1[c])
            for c in range(device.num_qubits)
        },
    )
    cz = {}
    for (a, b), fidelity in device.cz_fidelity.items():
        cz[(a, b)] = InstructionProperties(duration=cz_s, error=1 - fidelity)
        cz[(b, a)] = InstructionProperties(duration=cz_s, error=1 - fidelity)
    target.add_instruction(CZGate(), cz)
    ro = device.readout_fidelity
    target.add_instruction(
        Measure(),
        {
            (c,): InstructionProperties(duration=measure_s, error=1 - ro[c])
            for c in range(device.num_qubits)
        },
    )
    target.add_instruction(
        Delay(Parameter("t")), {(c,): None for c in range(device.num_qubits)}
    )
    return target


def _cap_t2(t1_us, t2_us):
    """The T2 every relaxation channel uses: the calibrated one, capped at T1.

    Physically T2 may reach 2 T1, but with T2 > T1 thermal relaxation is a
    Kraus channel, and Aer 0.17.2 fails on noise models holding such channels
    next to the others: "Kraus is empty", a failed eigendecomposition, or a
    segmentation fault (seen with QAOA circuits on the 2026-10-19 EQE1
    snapshot; every part of the model alone ran). Capped, relaxation is a mix of
    Paulis and reset, which Aer handles. For gates the calibrated fidelity is
    still matched exactly (the depolarizing part makes up the rest); for idle
    noise the cap makes dephasing slightly stronger on the qubits with T2 > T1.
    """
    return np.minimum(t2_us, np.asarray(t1_us))


def _relaxation(t1_us: float, t2_us: float, duration_ns: float) -> QuantumError:
    """Thermal relaxation over one gate (T2 capped, :func:`_cap_t2`)."""
    return thermal_relaxation_error(
        t1_us * 1e3, _cap_t2(t1_us, t2_us) * 1e3, duration_ns
    )


def _process_fidelity(average: float, dim: int) -> float:
    return ((dim + 1) * average - 1) / dim


def _matched_error(relaxation: QuantumError, target_fidelity: float, num_qubits: int):
    """Compose ``relaxation`` with the depolarizing channel that brings the
    average gate fidelity to ``target_fidelity``. Returns the error and whether
    relaxation alone already exceeded the target infidelity (then ``λ = 0``)."""
    dim = 2**num_qubits
    f_relax = _process_fidelity(average_gate_fidelity(relaxation), dim)
    f_goal = _process_fidelity(target_fidelity, dim)
    if f_relax <= f_goal:
        return relaxation, True
    lam = (f_relax - f_goal) / (f_relax - 1 / dim**2)
    return relaxation.compose(depolarizing_error(lam, num_qubits)), False


@dataclass(frozen=True)
class NoiseReport:
    """What :func:`build_noise_model` did: the gates whose relaxation alone was
    worse than the calibration (``λ = 0``; the model is pessimistic there)."""

    relaxation_limited_r: tuple[int, ...]
    relaxation_limited_cz: tuple[tuple[int, int], ...]


def gate_errors(device: Device, assumptions: NoiseAssumptions = NoiseAssumptions()):
    """Per-gate errors of the noise model, in compact labels.

    Returns ``(r_errors, cz_errors, report)``: ``r_errors[c]`` for qubit ``c``,
    ``cz_errors[(a, b)]`` for the ordered pair ``(a, b)`` (both orientations of
    every edge: the relaxation part is per qubit, so it follows the order).
    """
    t1 = device.t1_us
    t2 = device.t2_echo_us
    f1 = device.one_qubit_fidelity

    r_errors, limited_r = {}, []
    for c in range(device.num_qubits):
        relax = _relaxation(t1[c], t2[c], assumptions.r_duration_ns)
        r_errors[c], limited = _matched_error(relax, f1[c], 1)
        if limited:
            limited_r.append(c)

    cz_errors, limited_cz = {}, []
    for (a, b), fidelity in sorted(device.cz_fidelity.items()):
        for q0, q1 in ((a, b), (b, a)):
            # Qiskit is little-endian: in `x.tensor(y)`, `y` acts on the first
            # qubit of the pair the error is attached to.
            relax = _relaxation(t1[q1], t2[q1], assumptions.cz_duration_ns).tensor(
                _relaxation(t1[q0], t2[q0], assumptions.cz_duration_ns)
            )
            cz_errors[(q0, q1)], limited = _matched_error(relax, fidelity, 2)
        if limited:
            limited_cz.append((a, b))

    report = NoiseReport(tuple(limited_r), tuple(limited_cz))
    return r_errors, cz_errors, report


def readout_errors(device: Device) -> dict[int, ReadoutError]:
    """Symmetric per-qubit readout error from the calibrated readout fidelity."""
    errors = {}
    for c, fidelity in enumerate(device.readout_fidelity):
        e = 1 - fidelity
        errors[c] = ReadoutError([[1 - e, e], [e, 1 - e]])
    return errors


def build_noise_model(
    device: Device,
    assumptions: NoiseAssumptions = NoiseAssumptions(),
    qubits=None,
):
    """Aer noise model on the native gate set. Returns ``(noise_model, report)``.

    ``qubits`` (from :func:`compact`) builds it for a compacted circuit instead:
    position ``i`` of the model is device qubit ``qubits[i]``. Without it the
    model covers every device qubit, for circuits on the full device width.
    """
    position = {
        q: i
        for i, q in enumerate(
            qubits if qubits is not None else range(device.num_qubits)
        )
    }
    r_errors, cz_errors, report = gate_errors(device, assumptions)
    model = NoiseModel(basis_gates=[ONE_QUBIT_GATE, TWO_QUBIT_GATE])
    for c, error in r_errors.items():
        if c in position:
            model.add_quantum_error(error, ONE_QUBIT_GATE, [position[c]])
    for (q0, q1), error in cz_errors.items():
        if q0 in position and q1 in position:
            model.add_quantum_error(error, TWO_QUBIT_GATE, [position[q0], position[q1]])
    for c, error in readout_errors(device).items():
        if c in position:
            model.add_readout_error(error, [position[c]])
    return model, report


def compact(circuit: QuantumCircuit):
    """The same circuit on only the device qubits it uses.

    Returns ``(compacted, qubits)``: ``compacted`` acts on ``len(qubits)``
    qubits, position ``i`` being device qubit ``qubits[i]`` (ascending), with
    the same classical bits, so measured counts read the same. Delays on unused
    qubits are dropped. Run it with ``build_noise_model(..., qubits=qubits)``.

    Why: a routed circuit spans all 52 device qubits. Aer can drop the unused
    ones itself, but not reliably with a noise model: its automatic method picks
    a density matrix (2^(2n) entries for the n qubits left), and even an explicit
    statevector run of a scheduled 52-qubit circuit under noise crashed Aer
    0.17.2 (segmentation fault). A compacted circuit avoids both.
    """
    used = {
        q
        for inst in circuit.data
        if inst.operation.name not in ("delay", "barrier")
        for q in inst.qubits
    }
    qubits = tuple(sorted(circuit.find_bit(q).index for q in used))
    position = {q: i for i, q in enumerate(qubits)}
    register = QuantumRegister(len(qubits), "q")
    compacted = QuantumCircuit(
        register, *circuit.cregs, global_phase=circuit.global_phase
    )
    loose = [b for b in circuit.clbits if not circuit.find_bit(b).registers]
    if loose:
        compacted.add_bits(loose)
    for inst in circuit.data:
        indices = [circuit.find_bit(q).index for q in inst.qubits]
        if not all(i in position for i in indices):
            # Only a delay or barrier can touch an unused qubit.
            continue
        compacted.append(
            inst.operation, [register[position[i]] for i in indices], inst.clbits
        )
    return compacted, qubits


def fidelity_deviation(
    device: Device, assumptions: NoiseAssumptions = NoiseAssumptions()
):
    """Largest ``|model fidelity - calibrated fidelity|`` over the gates that are
    not relaxation-limited, as ``(max_r, max_cz, report)``: the model's
    validation (exact channels, no sampling)."""
    r_errors, cz_errors, report = gate_errors(device, assumptions)
    f1 = device.one_qubit_fidelity
    max_r = max(
        (
            abs(average_gate_fidelity(e) - f1[c])
            for c, e in r_errors.items()
            if c not in report.relaxation_limited_r
        ),
        default=0.0,
    )
    max_cz = max(
        (
            abs(average_gate_fidelity(e) - device.cz_fidelity[(min(q), max(q))])
            for q, e in cz_errors.items()
            if (min(q), max(q)) not in report.relaxation_limited_cz
        ),
        default=0.0,
    )
    return max_r, max_cz, report


def idle_t2_us(device: Device, assumptions: NoiseAssumptions = NoiseAssumptions()):
    """T2 used for idle noise, per compact qubit, capped at T1 (:func:`_cap_t2`).

    Returns ``(t2_us, echo_fallback)``: under ``idle_t2="ramsey"`` the qubits
    whose snapshot has no Ramsey T2 keep the echo one, and ``echo_fallback``
    lists them (empty under ``"echo"``)."""
    t2 = device.t2_echo_us
    fallback = ()
    if assumptions.idle_t2 == "ramsey":
        ramsey = device.t2_ramsey_us
        fallback = tuple(
            c for c in range(device.num_qubits) if not math.isfinite(ramsey[c])
        )
        t2 = np.where(np.isfinite(ramsey), ramsey, t2)
    return _cap_t2(device.t1_us, t2), fallback


def add_idle_noise(
    scheduled: QuantumCircuit,
    device: Device,
    assumptions: NoiseAssumptions = NoiseAssumptions(),
) -> QuantumCircuit:
    """Attach thermal relaxation to every wait of a scheduled circuit.

    ``scheduled`` is a circuit on the device's qubits transpiled against
    ``build_target(device, assumptions)`` with ``scheduling_method="alap"``.
    Delays on qubits the circuit never uses are dropped first: relaxation there
    acts on ``|0>`` and does nothing, but its instructions would make Aer
    simulate every device qubit. The result keeps the circuit's parameters and
    runs on Aer next to :func:`build_noise_model`'s gate and readout noise.
    """
    if scheduled.num_qubits != device.num_qubits:
        raise ValueError(
            f"circuit has {scheduled.num_qubits} qubits, the device {device.num_qubits}: "
            "transpile it against build_target(device) first"
        )
    used = {
        q
        for inst in scheduled.data
        if inst.operation.name not in ("delay", "barrier")
        for q in inst.qubits
    }
    t1_ns = device.t1_us * 1e3
    t2_ns = idle_t2_us(device, assumptions)[0] * 1e3
    # Built on an empty copy rather than with Aer's RelaxationNoisePass, which
    # drops the transpiled layout.
    noisy = scheduled.copy_empty_like()
    for inst in scheduled.data:
        if inst.operation.name != "delay":
            noisy.append(inst)
            continue
        (qubit,) = inst.qubits
        if qubit not in used:
            continue
        noisy.append(inst)
        c = scheduled.find_bit(qubit).index
        wait_ns = _delay_ns(inst.operation)
        if wait_ns > 0:
            error = thermal_relaxation_error(t1_ns[c], t2_ns[c], wait_ns)
            noisy.append(error.to_instruction(), [qubit])
    return noisy


_UNIT_NS = {"dt": DT_S * 1e9, "s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}


def _delay_ns(delay: Delay) -> float:
    if delay.unit not in _UNIT_NS:
        raise ValueError(f"delay in unsupported unit {delay.unit!r}")
    return float(delay.duration) * _UNIT_NS[delay.unit]


def _summary(device: Device) -> str:
    cal = device.calibration
    dead = sorted(set(range(cal.num_qubits)) - set(device.to_original))
    cz = np.array(list(device.cz_fidelity.values()))
    lines = [
        f"backend {cal.backend_name}, calibration {cal.timestamp}",
        f"{cal.num_qubits} calibrated qubits, out of service (original labels): {dead}",
        f"device: {device.num_qubits} qubits, {len(device.cz_fidelity)} CZ edges",
        f"median 1q F {np.median(device.one_qubit_fidelity):.4f}, "
        f"CZ F {np.median(cz):.4f}, readout F {np.median(device.readout_fidelity):.4f}, "
        f"T1 {np.median(device.t1_us):.1f} us, T2 echo {np.median(device.t2_echo_us):.1f} us",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--coupling-map", required=True, type=Path)
    args = parser.parse_args()

    device = build_device(
        load_calibration(args.calibration), load_coupling_map(args.coupling_map)
    )
    print(_summary(device))
    variants = {
        "default gates (r 40 ns, cz 60 ns)": NoiseAssumptions(),
        "slow gates (r 60 ns, cz 100 ns)": NoiseAssumptions(60, 100),
    }
    for name, assumptions in variants.items():
        max_r, max_cz, report = fidelity_deviation(device, assumptions)
        print(
            f"{name}: max |dF| r {max_r:.1e}, cz {max_cz:.1e}; relaxation-limited "
            f"r {list(report.relaxation_limited_r)}, cz {list(report.relaxation_limited_cz)}"
        )
    for idle in ("echo", "ramsey"):
        t2, fallback = idle_t2_us(device, NoiseAssumptions(idle_t2=idle))
        print(
            f"idle T2 {idle}: median {np.median(t2):.1f} us, min {np.min(t2):.1f} us"
            + (f"; echo kept on {list(fallback)}" if fallback else "")
        )


if __name__ == "__main__":
    main()
