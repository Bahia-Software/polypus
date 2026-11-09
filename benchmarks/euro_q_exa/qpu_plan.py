"""
QPU execution plan for Euro-Q-Exa (#391, phase F4).

Counts the circuits, shots and jobs of the plan for the hours awarded on the
physical QPU, and estimates its QPU time under explicit assumptions, since
EQE1's time per shot, per-circuit overhead and accounting rule are not
published. Nothing here runs on hardware.

Six hours were awarded. The plan is sized for at most five, keeping one for
recalibration and failed jobs, and runs in two sessions on different days.
Every circuit gets 4000 shots unless a block says otherwise.

Session 1
  1. QAOA: MaxCut on 3-regular graphs, n = 16, p = 1, three instances. Per
     instance, one job with the parameters trained on the emulator (three DE
     seeds), the published fixed angles and a p = 0 uniform-random baseline;
     plus one job with the readout-calibration circuits (all zeros, all ones)
     of every instance.
  2. Validation of polypus against Qiskit: the inverse QFT applied to the
     Fourier state of a basis state x, whose ideal output is x. Arm A submits
     each front-end's own circuit and lets its toolchain compile it: exact
     transform at n = 2, 4, ..., 16 and 20, approximate transform (AQFT) at
     n = 16, 20, 24. Arm B submits the same native circuit through both
     front-ends with ``no_modify``, at n = 4, 8, 12. Three values of x and five
     repetitions per size and front-end; one job per (size, repetition,
     front-end) holds the three values of x.
  3. Parallel circuits: three 16-qubit AQFT circuits on disjoint qubit
     regions, each one alone and the three at once, for the three values of x
     and five repetitions; one job per (x, repetition).

Session 2
  4. Hybrid loop: VQE of H2 (4 qubits, 5 measurement groups), 50 SPSA
     iterations of two energy evaluations each, 1000 shots, one job per
     iteration.
  5. Drift: arm B in full and the exact QFT at n = 4, 8, 12, 16, 20 (one x,
     five repetitions, both front-ends); one job per (size, repetition,
     front-end), which also carries arm B's circuits of that size.
  6. More shots where the signal is low: 16000 extra shots per (case, x,
     front-end) for the exact QFT at n = 16, 20 and the AQFT at n = 24; one job
     per (case, front-end).

QPU time, for s shots, c circuits and j jobs:

    T_exec        = s * t_shot
    T_reservation = s * t_shot + c * t_overhead + j * t_latency

``T_reservation`` counts the whole time the device is held for us,
``T_exec`` only the shots. The latency is conservative: VTT reports 3-4 s per
job on IQM hardware.

Usage:
    python benchmarks/euro_q_exa/qpu_plan.py --out benchmarks/euro_q_exa/results/analysis
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SHOTS = 4000
FRONT_ENDS = ("polypus", "qiskit")
X_VALUES = 3
REPETITIONS = 5

# Block 1
QAOA_INSTANCES = 3
DE_SEEDS = 3
READOUT_CIRCUITS = 2  # all zeros, all ones
# Block 2
EXACT_SIZES = (2, 4, 6, 8, 10, 12, 14, 16, 20)
AQFT_SIZES = (16, 20, 24)
ARM_B_SIZES = (4, 8, 12)
# Block 3
PARALLEL_REGIONS = 3
# Block 4
VQE_GROUPS = 5
SPSA_ITERATIONS = 50
SPSA_EVALUATIONS = 2
VQE_SHOTS = 1000
# Block 5
DRIFT_SIZES = (4, 8, 12, 16, 20)
# Block 6
LOW_SIGNAL_CASES = (("exact", 16), ("exact", 20), ("aqft", 24))
EXTRA_SHOTS = 16_000

T_SHOT_MS = (0.3, 1.0, 3.0)
T_OVERHEAD_S = 1.0
T_LATENCY_S = 10.0
BUDGET_H = 5.0


@dataclass(frozen=True)
class Block:
    number: int
    name: str
    session: int
    circuits: int
    shots: int
    jobs: int


def qaoa() -> Block:
    per_instance = DE_SEEDS + 1 + 1  # trained seeds, fixed angles, p = 0
    circuits = QAOA_INSTANCES * (per_instance + READOUT_CIRCUITS)
    return Block(1, "QAOA", 1, circuits, circuits * SHOTS, QAOA_INSTANCES + 1)


def validation() -> Block:
    sizes = len(EXACT_SIZES) + len(AQFT_SIZES) + len(ARM_B_SIZES)
    jobs = sizes * REPETITIONS * len(FRONT_ENDS)
    circuits = jobs * X_VALUES
    return Block(
        2, "Validation (polypus vs. Qiskit)", 1, circuits, circuits * SHOTS, jobs
    )


def parallel() -> Block:
    jobs = X_VALUES * REPETITIONS
    circuits = jobs * (PARALLEL_REGIONS + 1)  # each region alone, then all at once
    return Block(3, "Parallel circuits", 1, circuits, circuits * SHOTS, jobs)


def hybrid() -> Block:
    circuits = SPSA_ITERATIONS * SPSA_EVALUATIONS * VQE_GROUPS
    return Block(
        4, "Hybrid loop (VQE)", 2, circuits, circuits * VQE_SHOTS, SPSA_ITERATIONS
    )


def drift() -> Block:
    # Arm B's sizes are a subset of the drift sizes, so its circuits ride in
    # the drift jobs of the same size.
    if not set(ARM_B_SIZES) <= set(DRIFT_SIZES):
        raise ValueError("every arm B size must also be a drift size")
    jobs = len(DRIFT_SIZES) * REPETITIONS * len(FRONT_ENDS)
    arm_b = len(ARM_B_SIZES) * REPETITIONS * len(FRONT_ENDS) * X_VALUES
    circuits = jobs + arm_b
    return Block(5, "Drift", 2, circuits, circuits * SHOTS, jobs)


def more_shots() -> Block:
    jobs = len(LOW_SIGNAL_CASES) * len(FRONT_ENDS)
    circuits = jobs * X_VALUES
    return Block(6, "More shots", 2, circuits, circuits * EXTRA_SHOTS, jobs)


def plan() -> list[Block]:
    return [qaoa(), validation(), parallel(), hybrid(), drift(), more_shots()]


def seconds(
    blocks, t_shot_ms, t_overhead_s=T_OVERHEAD_S, t_latency_s=T_LATENCY_S
) -> tuple[float, float]:
    """``(reservation_s, exec_s)`` for ``blocks`` (formulas: module docstring)."""
    exec_s = sum(b.shots for b in blocks) * t_shot_ms / 1e3
    reservation_s = (
        exec_s
        + sum(b.circuits for b in blocks) * t_overhead_s
        + sum(b.jobs for b in blocks) * t_latency_s
    )
    return reservation_s, exec_s


def session_blocks(blocks, session) -> list[Block]:
    return [b for b in blocks if b.session == session]


def blocks_table(blocks) -> str:
    lines = [
        "| Block | Session | Circuits | Shots | Jobs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for b in blocks:
        lines.append(
            f"| {b.number}. {b.name} | {b.session} | {b.circuits:,} | {b.shots:,} | {b.jobs} |"
        )
    lines.append(
        f"| Total | | {sum(b.circuits for b in blocks):,} "
        f"| {sum(b.shots for b in blocks):,} | {sum(b.jobs for b in blocks)} |"
    )
    return "\n".join(lines)


def _time_row(name, blocks, divisor, fmt):
    per = [seconds(blocks, t) for t in T_SHOT_MS]
    cells = [fmt.format(r / divisor) for r, _ in per] + [
        fmt.format(e / divisor) for _, e in per
    ]
    return f"| {name} | " + " | ".join(cells) + " |"


def time_table(blocks) -> str:
    """Minutes per block and hours per session, for each time per shot."""
    head = " | ".join(f"{t:g} ms" for t in T_SHOT_MS)
    lines = [
        f"| Block | Reservation {head} | Execution only {head} |",
        "|" + " --- |" * (1 + 2 * len(T_SHOT_MS)),
    ]
    for b in blocks:
        lines.append(_time_row(f"{b.number}. {b.name} (min)", [b], 60, "{:.1f}"))
    for s in sorted({b.session for b in blocks}):
        lines.append(
            _time_row(f"Session {s}", session_blocks(blocks, s), 3600, "{:.2f} h")
        )
    lines.append(_time_row("Total", blocks, 3600, "{:.2f} h"))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    blocks = plan()
    text = (
        f"Assumptions: {T_OVERHEAD_S:g} s overhead per circuit, {T_LATENCY_S:g} s "
        f"latency per job, budget {BUDGET_H:g} h.\n\n"
        f"{blocks_table(blocks)}\n\n{time_table(blocks)}\n"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "qpu_plan.md").write_text(text)

    def hours(group):
        reservation_s, exec_s = zip(*(seconds(group, t) for t in T_SHOT_MS))
        return {
            "reservation": [s / 3600 for s in reservation_s],
            "execution": [s / 3600 for s in exec_s],
        }

    report = {
        "assumptions": {
            "t_shot_ms": T_SHOT_MS,
            "t_overhead_s": T_OVERHEAD_S,
            "t_latency_s": T_LATENCY_S,
            "budget_h": BUDGET_H,
        },
        "blocks": [asdict(b) for b in blocks],
        "hours": {
            "total": hours(blocks),
            **{
                f"session_{s}": hours(session_blocks(blocks, s))
                for s in sorted({b.session for b in blocks})
            },
        },
    }
    (args.out / "qpu_plan.json").write_text(json.dumps(report, indent=1))
    print(text)


if __name__ == "__main__":
    main()
