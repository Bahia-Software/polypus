"""
Analysis of the Euro-Q-Exa emulation sweep (#391, phase F4).

Reads the per-(instance, p) files ``run_emulator.py`` writes and aggregates the
approximation ratio of every condition per (family, n, p): mean and standard
deviation over instances and DE seeds (fixed angles and P0: over instances),
the probability of sampling an optimal cut, and the share of the ideal gain over
random cuts that survives the noise, ``(C - P0) / (A - P0)``.

The approximation ratio (mean cut / optimum) suits positive weights. With the
signed weights of the lattice control its scale depends on the instance: one
of them, mostly negative edges, has optimum 4 and random cuts at -4, so its
ratio goes negative and dominates the spread. Every row therefore also gets the
normalized ratio ``(cut - random) / (optimum - random)`` = ``(ar - P0) / (1 - P0)``
with its own instance's P0: 0 for random cuts, 1 for the optimum.

Writes into ``--out``:
  * ``ar_summary.csv``: one row per (family, n, p, condition);
  * ``ar_by_instance.csv``: one row per (instance, p, run, condition);
  * ``ar_summary.md``: the main table for the report;
  * ``ar_normalized.md``: the same conditions as normalized ratios.

Usage:
    python benchmarks/euro_q_exa/analyze.py \
        --sweep benchmarks/euro_q_exa/results/sweep --out benchmarks/euro_q_exa/results/analysis
"""

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_rows(sweep_dir) -> list[dict]:
    """Every row of every unit file in ``sweep_dir``."""
    rows = []
    for path in sorted(Path(sweep_dir).glob("*.json.gz")):
        with gzip.open(path, "rt") as f:
            rows.extend(json.load(f)["rows"])
    return rows


def _key(row):
    return (row["family"], row["n"], row["p"])


def summarize(rows) -> list[dict]:
    """Mean and standard deviation of the ratio per (family, n, p, condition).

    Conditions of trained runs are averaged over instances and DE seeds; ``FA_*``
    (fixed angles) and ``P0`` over instances. ``kept`` is filled on the ``C``
    and ``FA_C`` rows: ``(C - P0) / (A - P0)`` with the group's means.
    """
    groups = defaultdict(lambda: defaultdict(list))
    normalized = defaultdict(lambda: defaultdict(list))
    p_opt = defaultdict(lambda: defaultdict(list))
    random_ratio = {
        (r["instance"], r["p"]): r["ar"] for r in rows if r["condition"] == "P0"
    }
    for row in rows:
        if row["run"].startswith("de"):
            label = row["condition"]
        elif row["run"] == "fa":
            label = f"FA_{row['condition']}"
        else:
            label = row["condition"]
        groups[_key(row)][label].append(row["ar"])
        p0 = random_ratio[(row["instance"], row["p"])]
        normalized[_key(row)][label].append((row["ar"] - p0) / (1 - p0))
        if row.get("p_opt") is not None:
            p_opt[_key(row)][label].append(row["p_opt"])

    out = []
    for key in sorted(groups):
        family, n, p = key
        means = {label: statistics.mean(v) for label, v in groups[key].items()}
        for label, values in groups[key].items():
            record = {
                "family": family,
                "n": n,
                "p": p,
                "condition": label,
                "ar_mean": means[label],
                "ar_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
                "p_opt_mean": statistics.mean(p_opt[key][label])
                if p_opt[key][label]
                else None,
                "kept": None,
                "norm_mean": statistics.mean(normalized[key][label]),
                "norm_sd": statistics.stdev(normalized[key][label])
                if len(values) > 1
                else 0.0,
            }
            ideal = {"C": "A", "FA_C": "FA_A"}.get(label)
            if ideal in means and "P0" in means:
                record["kept"] = (means[label] - means["P0"]) / (
                    means[ideal] - means["P0"]
                )
            out.append(record)
    return out


def summary_table(summary) -> str:
    """Markdown table: one row per (family, n, p), ratio means (± sd for the
    noisy main condition) and the kept share."""
    by = defaultdict(dict)
    for r in summary:
        by[(r["family"], r["n"], r["p"])][r["condition"]] = r
    columns = ["P0", "A", "C", "D", "I", "E", "I_R", "FA_A", "FA_C"]
    lines = [
        "| Family | n | p | " + " | ".join(columns) + " | Kept (C) |",
        "|" + " --- |" * (len(columns) + 4),
    ]
    for (family, n, p), cells in sorted(by.items()):
        values = []
        for c in columns:
            r = cells.get(c)
            if r is None:
                values.append("-")
            elif c == "C":
                values.append(f"{r['ar_mean']:.3f} ± {r['ar_sd']:.3f}")
            else:
                values.append(f"{r['ar_mean']:.3f}")
        kept = cells.get("C", {}).get("kept")
        kept_text = f"{kept:.0%}" if kept is not None else "-"
        lines.append(
            f"| {family} | {n} | {p} | " + " | ".join(values) + f" | {kept_text} |"
        )
    return "\n".join(lines)


def normalized_table(summary) -> str:
    """Markdown table of normalized ratios (mean ± sd over instances and seeds
    for the main conditions)."""
    by = defaultdict(dict)
    for r in summary:
        by[(r["family"], r["n"], r["p"])][r["condition"]] = r
    columns = ["A", "C", "D", "I", "FA_A", "FA_C"]
    lines = [
        "| Family | n | p | " + " | ".join(columns) + " |",
        "|" + " --- |" * (len(columns) + 3),
    ]
    for (family, n, p), cells in sorted(by.items()):
        values = [
            f"{cells[c]['norm_mean']:.3f} ± {cells[c]['norm_sd']:.3f}"
            if c in cells
            else "-"
            for c in columns
        ]
        lines.append(f"| {family} | {n} | {p} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = load_rows(args.sweep)
    if not rows:
        raise SystemExit(f"no unit files in {args.sweep}")
    summary = summarize(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.out / "ar_summary.csv",
        summary,
        [
            "family",
            "n",
            "p",
            "condition",
            "ar_mean",
            "ar_sd",
            "count",
            "p_opt_mean",
            "kept",
        ],
    )
    _write_csv(
        args.out / "ar_by_instance.csv",
        rows,
        [
            "instance",
            "family",
            "n",
            "p",
            "run",
            "condition",
            "ar",
            "p_opt",
            "best_cut_ratio",
        ],
    )
    table = summary_table(summary)
    (args.out / "ar_summary.md").write_text(table + "\n")
    normalized = normalized_table(summary)
    (args.out / "ar_normalized.md").write_text(normalized + "\n")
    print(table + "\n\n" + normalized)


if __name__ == "__main__":
    main()
