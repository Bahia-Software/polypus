# MaxCut–QAOA experiment

The flagship Polypus example: optimise **MaxCut** with **QAOA** on a local
simulator and compare Polypus' three optimizers — **DE**, **PSO**, **QNG** —
against a pure-scipy `differential_evolution` baseline ("no Polypus"), across a
range of qubit counts, with repetitions so the numbers carry a statistical
spread rather than a single lucky run.

> This README is intentional (the task asked for a documented, reportable
> experiment). It documents *how to run and interpret* the experiment; the
> generated report documents *what a given run found*.

## Design: run / sweep / report are three separate stages

Executing and reporting are deliberately decoupled, so a report can be
regenerated at any time without re-running any QAOA.

| Stage | Script | Does |
|---|---|---|
| **run** | `run_maxcut.py` | Runs **one** `(qubits, method, seed)` optimization and appends **one** row to the results CSV. No plotting, no aggregation. |
| **sweep** | `sweep_maxcut.py` | Orchestrates `qubits × methods × repeats`, each run as an isolated subprocess; continues past failures and reports them; writes a run manifest. |
| **report** | `report_maxcut.py` | Reads the accumulated CSV, computes aggregated metrics, renders figures, and writes `maxcut_report.md` + `.html`. |

## Quickstart

Everything below assumes the extension module is built and importable
(`maturin develop --release --features extension-module`, with
`LD_LIBRARY_PATH` pointing at the conda env — see `docs/CONTRIBUTING.md`).

```bash
# 1. Preview the plan (run count + rough time estimate), run nothing:
python examples/max_cut/sweep_maxcut.py --dry-run

# 2. Run the default sweep (qubits 4–7, 4 methods, 10 repeats = 160 runs):
python examples/max_cut/sweep_maxcut.py --seed 42

# 3. Generate the report from whatever the CSV has accumulated:
python examples/max_cut/report_maxcut.py
```

Outputs land under `examples/max_cut/output/` (git-ignored):
`maxcut_results.csv`, `maxcut_manifest_*.json`, `logs/`, and `report/`
(`maxcut_report.md`, `maxcut_report.html`, `figures/`).

A small **committed demo** lives in [`demo/`](demo/) so the deliverable is
visible in the repo/PR without running anything — see
[`demo/maxcut_report.md`](demo/maxcut_report.md). It is a deliberately reduced,
reproducible sample; regenerate it with:

```bash
python examples/max_cut/sweep_maxcut.py --qubits 4 5 6 --repeats 3 --shots 4000 \
    --seed 42 --fresh --csv examples/max_cut/demo/maxcut_results.csv
python examples/max_cut/report_maxcut.py \
    --csv examples/max_cut/demo/maxcut_results.csv --out-dir examples/max_cut/demo
rm -rf examples/max_cut/demo/logs examples/max_cut/demo/maxcut_manifest_*.json
```

### Regenerating only the report

The report stage never re-runs QAOA. Re-render figures/tables from existing data
(e.g. after tweaking a chart) with just:

```bash
python examples/max_cut/report_maxcut.py --csv examples/max_cut/output/maxcut_results.csv
```

It accepts several CSVs (`--csv a.csv b.csv`) and combines them, so data from
multiple sweeps aggregates cleanly.

## Reproducibility — the whole sweep from one number

Repetition *r* of every `(qubits, method)` uses `seed = base_seed + r`. Every
source of shot randomness is seeded from that seed — the training oracle
(through `polypus.train(..., seed=…)`, contract C-7), the QNG variance callback,
the final evaluation sampling, and the scipy baseline's Aer sampling — so **two
runs with the same seed produce the identical approximation ratio and best
bitstring**. Pass a single `--seed` to the sweep and the entire barrido is
reproducible. The problem graph depends only on the qubit count (fixed
`--graph-seed`), so a given size always poses the same MaxCut instance and the
variance you see comes purely from the optimizer seed.

## Estimated time

`--dry-run` prints a rough estimate. As a guide, the default 160-run sweep
(qubits 4–7, 10 repeats, 10 000 shots) is on the order of ~20 minutes on a
laptop; it grows steeply with qubit count (statevector cost) but only weakly
with shots (sampling is cheap next to simulating the circuit). Shrink it while
iterating:

```bash
python examples/max_cut/sweep_maxcut.py --qubits 4 5 --repeats 2 --shots 2000
```

The scipy baseline can be sped up with `--scipy-workers N` (multiprocessing).

## Reading the metrics table

One row per `(method, qubit count)`:

- **`ratio mean`** — the primary quality metric: expected cut ÷ optimal cut
  (brute-forced), so `1.0` is perfect. The **95% CI** is a t-interval over the
  `n` repetitions. *Higher is better.*
- **`time mean [s]`** — wall-clock seconds per run. *Lower is better.*
- `std / median / min / max` describe the spread across repetitions.

The report's "Polypus vs scipy" figure makes the head-to-head explicit: Polypus
optimizers typically match or beat scipy on ratio while running several times
faster.

## Methods and scope

Default swept methods: `polypus_local` (DE), `polypus_local_pso` (PSO),
`polypus_local_qng` (QNG), and `scipy`. The distributed **CUNQA** and real-QPU
**QMIO** paths are intentionally *not* part of the default sweep (there is no
SLURM cluster in this environment), but they remain invokable manually, e.g.:

```bash
python examples/max_cut/run_maxcut.py --qubits 5 --method polypus_cunqa --seed 1 \
    --nodes 1 --cores-per-qpu 9
```

## Files

| File | Role |
|---|---|
| `maxcut_lib.py` | Importable core: graph/objective, circuits, seeded run engine, CSV schema, validation. |
| `run_maxcut.py` | Run stage CLI (one run → one CSV row). |
| `sweep_maxcut.py` | Sweep orchestrator CLI. |
| `report_maxcut.py` | Report stage CLI (CSV → metrics + figures + `.md`/`.html`). |
| `demo/` | Committed sample report. |
