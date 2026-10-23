"""
QPU execution plan for a future Euro-Q-Exa application (#391, phase F4).

Counts the circuits and shots a physical run would need and estimates QPU hours
under explicit assumptions, since EQE1's time per shot, per-circuit overhead
and accounting rules are not published. Nothing here runs on hardware.

Two modes:

  * base (transfer): parameters trained on the emulator; the QPU only runs
    closed batches. Per (family, n, p, instance): the circuits of the trained
    DE seeds, the fixed angles (3-regular only), one p = 0 circuit (uniform
    random baseline), two readout-calibration circuits on the same layout
    (all-zeros and all-ones) and, on a second day, the evaluation circuits again
    to measure drift;
  * refinement (optional): a short closed-loop DE on the QPU itself, one batch
    per generation, for one 3-regular instance per size at p = 1.

    T_exec        = shots * t_shot + circuits * t_overhead
    T_reservation = T_exec + jobs * t_latency

``T_reservation`` is what a SLURM reservation of the QPU would cost if it is held
while the classical side works between batches (``--gres=qpu:iqm_eqe1`` at LRZ).

Usage:
    python benchmarks/euro_q_exa/qpu_plan.py --out benchmarks/euro_q_exa/results/analysis
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

SIZES = (16, 20)
DEPTHS = (1, 2)
INSTANCES_PER_SIZE = 3
DE_SEEDS = 3
EVAL_SHOTS = 4000
CALIBRATION_SHOTS = 4000
MAX_CIRCUITS_PER_JOB = 200
T_SHOT_MS = (0.3, 1.0, 3.0)
T_OVERHEAD_S = 1.0
T_LATENCY_S = 10.0


@dataclass(frozen=True)
class Block:
    name: str
    circuits: int
    shots: int
    jobs: int


def _jobs(circuits: int, max_per_job: int) -> int:
    return math.ceil(circuits / max_per_job) if circuits else 0


def base_plan(
    sizes=SIZES,
    depths=DEPTHS,
    instances=INSTANCES_PER_SIZE,
    de_seeds=DE_SEEDS,
    include_control=True,
    eval_shots=EVAL_SHOTS,
    calibration_shots=CALIBRATION_SHOTS,
    repeat=True,
    max_per_job=MAX_CIRCUITS_PER_JOB,
) -> list[Block]:
    """Blocks of the transfer mode (module docstring)."""
    families = ("3reg", "lattice") if include_control else ("3reg",)
    units = {f: len(sizes) * len(depths) * instances for f in families}
    trained = sum(units.values()) * de_seeds
    fixed = units["3reg"]
    baseline = sum(units.values())
    calibration = 2 * sum(units.values())
    evaluation = trained + fixed + baseline
    blocks = [
        Block(
            "Trained parameters",
            trained,
            trained * eval_shots,
            _jobs(trained, max_per_job),
        ),
        Block(
            "Fixed angles (3-regular)",
            fixed,
            fixed * eval_shots,
            _jobs(fixed, max_per_job),
        ),
        Block(
            "p = 0 baseline",
            baseline,
            baseline * eval_shots,
            _jobs(baseline, max_per_job),
        ),
        Block(
            "Readout calibration",
            calibration,
            calibration * calibration_shots,
            _jobs(calibration, max_per_job),
        ),
    ]
    if repeat:
        blocks.append(
            Block(
                "Repeat on a second day (drift)",
                evaluation,
                evaluation * eval_shots,
                _jobs(evaluation, max_per_job),
            )
        )
    return blocks


def refinement_plan(
    sizes=SIZES, population=10, generations=20, shots=1000
) -> list[Block]:
    """Closed-loop DE on the QPU: one 3-regular instance per size, p = 1, one
    job per batch (the initial population and each generation)."""
    batches = (generations + 1) * len(sizes)
    circuits = population * batches
    return [Block("On-QPU DE refinement (p = 1)", circuits, circuits * shots, batches)]


def hours(blocks, t_shot_ms, t_overhead_s=T_OVERHEAD_S, t_latency_s=T_LATENCY_S):
    """``(exec_h, reservation_h)`` for ``blocks`` (formulas: module docstring)."""
    shots = sum(b.shots for b in blocks)
    circuits = sum(b.circuits for b in blocks)
    jobs = sum(b.jobs for b in blocks)
    exec_s = shots * t_shot_ms / 1e3 + circuits * t_overhead_s
    return exec_s / 3600, (exec_s + jobs * t_latency_s) / 3600


def _totals(blocks):
    return Block(
        "Total",
        sum(b.circuits for b in blocks),
        sum(b.shots for b in blocks),
        sum(b.jobs for b in blocks),
    )


def blocks_table(blocks) -> str:
    lines = ["| Block | Circuits | Shots | Jobs |", "| --- | --- | --- | --- |"]
    for b in [*blocks, _totals(blocks)]:
        lines.append(f"| {b.name} | {b.circuits} | {b.shots:,} | {b.jobs} |")
    return "\n".join(lines)


def hours_table(plans) -> str:
    """Hours per plan and t_shot, execution-only and with the reservation."""
    head = " | ".join(f"t_shot {t} ms" for t in T_SHOT_MS)
    lines = [f"| Plan | Accounting | {head} |", "|" + " --- |" * (len(T_SHOT_MS) + 2)]
    for name, blocks in plans.items():
        per = [hours(blocks, t) for t in T_SHOT_MS]
        lines.append(
            f"| {name} | execution | " + " | ".join(f"{e:.2f} h" for e, _ in per) + " |"
        )
        lines.append(
            f"| {name} | reservation | "
            + " | ".join(f"{r:.2f} h" for _, r in per)
            + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    base = base_plan()
    refinement = refinement_plan()
    plans = {"Base": base, "Base + refinement": base + refinement}
    text = (
        "Assumptions: per-circuit overhead "
        f"{T_OVERHEAD_S:g} s, {T_LATENCY_S:g} s between jobs, at most "
        f"{MAX_CIRCUITS_PER_JOB} circuits per job.\n\n"
        f"{blocks_table(base + refinement)}\n\n{hours_table(plans)}\n"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "qpu_plan.md").write_text(text)
    report = {
        "assumptions": {
            "t_shot_ms": T_SHOT_MS,
            "t_overhead_s": T_OVERHEAD_S,
            "t_latency_s": T_LATENCY_S,
            "max_circuits_per_job": MAX_CIRCUITS_PER_JOB,
        },
        "blocks": [asdict(b) for b in base + refinement],
        "hours": {
            name: {str(t): hours(blocks, t) for t in T_SHOT_MS}
            for name, blocks in plans.items()
        },
    }
    (args.out / "qpu_plan.json").write_text(json.dumps(report, indent=1))
    print(text)


if __name__ == "__main__":
    main()
