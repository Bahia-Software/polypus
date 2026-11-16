# Euro-Q-Exa device emulation (#391)

Emulation of QAOA-MaxCut on a model of Euro-Q-Exa (IQM, backend `EQE1`):
circuits are transpiled to the native gate set (`r`, `cz`) and coupling map, and
run on Aer with a noise model built from a calibration snapshot of the device.
Nothing here runs on the physical QPU, and polypus has no backend for it.

## Setup

The scripts and tests need the polypus wheel plus Qiskit, Qiskit Aer, pytest
(`requirements-dev.txt`) and networkx (`requirements-examples.txt`):

```bash
pip install -r requirements-dev.txt -r requirements-examples.txt
maturin build --release --out dist
pip install --no-deps --force-reinstall dist/polypus_quantum-*.whl
```

## Inputs

No calibration data is committed: the scripts read the two files the provider
supplies, passed on the command line.

- A calibration snapshot (JSON): per-qubit T1, T2 (Ramsey), T2 echo,
  single-qubit and readout fidelities and CZ fidelity per edge, keyed by the
  chip's original labels.
- `coupling_map.txt`: the backend's coupling map, in its compact labels
  (out-of-service qubits are dropped and the rest renumbered).

Without them only the tests run, on a small synthetic calibration
(`tests/synthetic.py`).

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

Never run the full-device circuit with a noise model: Aer's automatic method
picks a density matrix, and Aer 0.17.2 crashed on such circuits (see
`compact`). Every relaxation channel uses T2 capped at T1 for the same reason
(see `_cap_t2`).

## Instances (`instances.py`)

QAOA-MaxCut on random 3-regular graphs (main set) and on signed subgraphs of the
device's own coupling graph (control, no routing cost), from fixed seeds, with
exact optima. Circuits come from polypus's `build_qaoa_circuit` (parameters: all
betas, then all gammas); the cost layer is `rzz(-γ w)` per edge, so QAOA
maximizes the cut. `native_qaoa_circuit` builds the same circuit as a
`polypus.Circuit`, and `ising_cost` gives the cut as a native `polypus.Ising`
for `polypus.train`. Counts are read with qubit 0 rightmost;
`approximation_ratio` uses polypus's `expectation_value`.

## Routing study (`routing.py`)

```bash
python benchmarks/euro_q_exa/routing.py --calibration <calibration.json> \
    --coupling-map <coupling_map.txt> --out benchmarks/euro_q_exa/results
```

transpiles every instance for p = 1..3 with 10 transpiler seeds, keeps the
highest-ESP run, and writes `instances.json`, `routing.csv` (every seed) and
`routing_summary.md` (two-qubit gate count and depth). `results/` is
git-ignored.

## Tests

```bash
python -m pytest benchmarks/euro_q_exa/tests
```

They use the synthetic calibration and are deliberately **not run by CI**: this
is a study, not part of the library.
