# Euro-Q-Exa emulator benchmark (#391)

QAOA-MaxCut on an emulated Euro-Q-Exa (IQM, backend `EQE1`): circuits are
transpiled to the native gate set (`r`, `cz`) and coupling map, and run on Aer
with a noise model built from an LRZ calibration snapshot. The physical QPU is
not used.

Status: device model (F1), instances and routing study (F2). Emulation runs,
analysis and the QPU plan come next.

## Inputs

LRZ calibration data is not committed. Pass the two files LRZ sends:

- `calibration_data_Euro-Q-Exa_<date>_active_qubits.json`: per-qubit T1, T2
  (Ramsey), T2 echo, single-qubit and readout fidelities, CZ fidelity per edge,
  keyed by the chip's original labels (54 qubits).
- `coupling_map.txt`: the backend's coupling map, in its compact labels
  (52 qubits; out-of-service qubits are dropped and the rest renumbered).

## Device model (`device.py`)

```bash
python benchmarks/euro_q_exa/device.py --calibration <calibration.json> --coupling-map <coupling_map.txt>
```

prints a summary and checks the noise model against the calibration for each
noise variant. See the module docstring for the label mapping, the noise-model
recipe and the assumptions (gate durations, T2 choice, symmetric readout).

Pipeline for one circuit:

1. `transpile(qc, target=build_target(device, assumptions), scheduling_method="alap", ...)`;
2. optionally `add_idle_noise(scheduled, device, assumptions)` (relaxation while
   qubits wait; idle T2 echo, or Ramsey as a pessimistic bound);
3. `compact(circuit)` → `(small, qubits)`, then
   `build_noise_model(device, assumptions, qubits=qubits)`;
4. run on Aer with `method="statevector"`.

Never run the full 52-qubit circuit with a noise model: Aer's automatic method
picks a density matrix, and Aer 0.17.2 crashed on such circuits (see
`compact`). Every relaxation channel uses T2 capped at T1 for the same reason
(see `_cap_t2`).

## Instances (`instances.py`)

QAOA-MaxCut on random 3-regular graphs (main set) and on signed subgraphs of the
device's own coupling graph (control, no routing cost), n = 16 and 20, three of
each, from fixed seeds, with exact optima. Circuits come from polypus's
`build_qaoa_circuit` (parameters: all betas, then all gammas); the cost layer is
`rzz(-γ w)` per edge, so QAOA maximizes the cut. Counts are read with qubit 0
rightmost; `approximation_ratio` uses polypus's `expectation_value`.

## Routing study (`routing.py`)

```bash
python benchmarks/euro_q_exa/routing.py --calibration <calibration.json> \
    --coupling-map <coupling_map.txt> --out benchmarks/euro_q_exa/results
```

transpiles every instance for p = 1..3 with 10 transpiler seeds, keeps the
highest-ESP run, and writes `instances.json`, `routing.csv` (every seed) and
`routing_summary.md` (the two-qubit gate count and depth table for LRZ).
`results/` is git-ignored.

## Tests

```bash
pytest benchmarks/euro_q_exa/tests
```

They use a small synthetic calibration and are **not run by CI** on purpose:
this benchmark may never become part of the library. Before merging into
`main`, decide whether they move into `tests/python/` or into `ci.yml`.
