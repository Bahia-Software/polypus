# MaxCut–QAOA experiment report

## What this experiment demonstrates

MaxCut is optimised with QAOA on a local statevector/Aer simulator, comparing three Polypus optimizers (**DE**, **PSO**, **QNG**) against a pure-scipy `differential_evolution` baseline ("no Polypus"). Each cell below is the aggregate of several independent repetitions, so the numbers carry a spread, not a single lucky/unlucky run.

## Reproducibility

The whole sweep is reproducible from a single base seed. Repetition *r* of every (qubits, method) uses `seed = base_seed + r`, and every source of shot randomness (training oracle, QNG variance callback, final evaluation, scipy's Aer sampling) is seeded from it — so two runs with the same seed produce the identical approximation ratio and best bitstring. This report's data used base seed(s) **42**. Regenerate the data with:

```bash
python examples/max_cut/sweep_maxcut.py --qubits 4 5 6 --repeats 3 --seed 42
python examples/max_cut/report_maxcut.py
```

## How to read the metrics table

One row per (method, qubit count). `ratio mean` is the primary quality metric (expected cut ÷ optimal cut, so 1.0 is perfect); the 95% CI is a t-interval over the `n` repetitions. `time` columns are wall-clock seconds per run. Higher ratio is better; lower time is better.

> The CI shown is a 95% t-interval **clipped to each metric's physical range** (ratio to `[0, 1]`, time to `[0, ∞)`). With few repetitions the raw t-interval can extend past those bounds; the mean and standard deviation are the unmodified sample values — only the displayed interval (and its error bars) is clamped.

## Headline

At 6 qubits, the best Polypus optimizer (**Polypus PSO**) reached a mean approximation ratio of **0.9160**, 0.0672 above the scipy baseline (0.8488), with a mean run time of 5.24s vs scipy's 18.37s (≈3.50× time ratio).

## Provenance

- runs: **36** | qubits: [4, 5, 6] | methods: ['Polypus DE', 'Polypus PSO', 'Polypus QNG', 'scipy DE (baseline)']
- repeats: 3 | base seed(s): [42] | shots: [4000]
- git commit(s): ['0ffd244'] | polypus: ['0.6.0']
- timestamps: 2026-07-15T07:51:28Z … 2026-07-15T07:54:03Z

## Configuration

The training configuration behind the metrics above. The scaling factors and shot count are expected to be constant across the whole sweep; the problem sizing is derived from them and `n_qubits`. A **⚠** marks a value that is not constant where one was expected — it is shown as a list of the distinct values rather than silently collapsed to one (same fail-high stance as the CSV reader).

**Scaling factors & shots**

| parameter | value |
| --- | --- |
| layers_factor | 2 |
| population_factor | 4 |
| generations_factor | 5 |
| n_shots | 4000 |

**Hyper-parameters per method** — only the knobs each optimizer actually uses; `n/a` means the method does not consult that knob.

*Polypus DE*

| hyper-parameter | value |
| --- | --- |
| tolerance | 1e-05 |

*Polypus PSO*

| hyper-parameter | value |
| --- | --- |
| tolerance | 1e-05 |

*Polypus QNG*

| hyper-parameter | value |
| --- | --- |
| learning_rate | 0.1 |
| finite-diff step | 0.1 |
| tikhonov reg | 0.05 |

*scipy DE (baseline)*

| hyper-parameter | value |
| --- | --- |
| tolerance | 1e-05 |
| popsize (→ differential_evolution) | 4 |
| scipy_workers | 1 |

**Problem sizing by qubit count** — varies with `n_qubits`; sets the scale of each run and gives context for the ratio/time rows above.

| qubits | layers | dimensions | population_size | max_generations | optimal cut (bruteforce) |
| --- | --- | --- | --- | --- | --- |
| 4 | 4 | 8 | 32 | 20 | 4 |
| 5 | 4 | 8 | 32 | 25 | 6 |
| 6 | 6 | 12 | 48 | 30 | 9 |

> `population_size` is the population handed to the **Polypus** DE/PSO optimizers (`dimensions × population_factor`). The **scipy** baseline instead receives `popsize = population_factor` and multiplies it by `dimensions` internally, so its per-method row reports `popsize` while the sizing table reports `population_size` — two conventions for (effectively) the same total population, not the same number.

## Metrics

| method | qubits | n | ratio mean | ratio std | ratio median | ratio min | ratio max | ratio 95% CI | time mean [s] | time std [s] | time 95% CI [s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Polypus DE | 4 | 3 | 0.9195 | 0.0204 | 0.9130 | 0.9032 | 0.9424 | [0.8688, 0.9703] | 1.033 | 0.019 | [0.985, 1.080] |
| Polypus DE | 5 | 3 | 0.9675 | 0.0152 | 0.9588 | 0.9588 | 0.9850 | [0.9299, 1.0000] | 1.747 | 0.003 | [1.739, 1.755] |
| Polypus DE | 6 | 3 | 0.8408 | 0.0137 | 0.8424 | 0.8264 | 0.8536 | [0.8067, 0.8748] | 5.250 | 0.015 | [5.212, 5.288] |
| Polypus PSO | 4 | 3 | 0.9753 | 0.0196 | 0.9708 | 0.9583 | 0.9968 | [0.9265, 1.0000] | 0.953 | 0.011 | [0.925, 0.980] |
| Polypus PSO | 5 | 3 | 0.9937 | 0.0042 | 0.9941 | 0.9893 | 0.9977 | [0.9831, 1.0000] | 1.729 | 0.016 | [1.689, 1.769] |
| Polypus PSO | 6 | 3 | 0.9160 | 0.0309 | 0.9248 | 0.8816 | 0.9415 | [0.8392, 0.9927] | 5.245 | 0.029 | [5.172, 5.318] |
| Polypus QNG | 4 | 3 | 0.8819 | 0.0883 | 0.9324 | 0.7799 | 0.9333 | [0.6625, 1.0000] | 1.130 | 0.060 | [0.982, 1.278] |
| Polypus QNG | 5 | 3 | 0.8869 | 0.0523 | 0.8869 | 0.8347 | 0.9393 | [0.7570, 1.0000] | 1.773 | 0.108 | [1.504, 2.042] |
| Polypus QNG | 6 | 3 | 0.7989 | 0.0021 | 0.7977 | 0.7976 | 0.8013 | [0.7937, 0.8041] | 5.714 | 0.738 | [3.881, 7.547] |
| scipy DE (baseline) | 4 | 3 | 0.9418 | 0.0292 | 0.9523 | 0.9087 | 0.9643 | [0.8692, 1.0000] | 1.841 | 0.013 | [1.808, 1.874] |
| scipy DE (baseline) | 5 | 3 | 0.9474 | 0.0106 | 0.9428 | 0.9400 | 0.9596 | [0.9211, 0.9738] | 3.163 | 0.215 | [2.630, 3.696] |
| scipy DE (baseline) | 6 | 3 | 0.8488 | 0.0096 | 0.8436 | 0.8429 | 0.8598 | [0.8250, 0.8725] | 18.372 | 1.277 | [15.200, 21.544] |

## Figures

### Approximation ratio vs. qubits

![Approximation ratio vs. qubits](figures/ratio_vs_qubits.png)

### Run time vs. qubits

![Run time vs. qubits](figures/time_vs_qubits.png)

### Approximation-ratio distribution per method

![Approximation-ratio distribution per method](figures/ratio_distribution.png)

### Polypus vs. scipy baseline

![Polypus vs. scipy baseline](figures/polypus_vs_scipy.png)

