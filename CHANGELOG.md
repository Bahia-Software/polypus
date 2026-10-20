# Changelog

All notable changes to Polypus are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). While the version is `0.x`, a minor
release may contain breaking changes; they are marked **BREAKING** below
(Conventional Commits `!`).

Entries were curated from the Conventional Commits history. Releases before
0.6.0 predate the convention and are summarised by hand.

## [Unreleased]

Changes on `main` since 0.7.2.

### Added
- `polypus.CMAES(generations=100, population_size=None, sigma0=math.pi / 2,
  tolerance=0.01, patience=20, restarts=0, seed=None)`, accepted by `train`
  and `qml.train`: Hansen's (μ/μ_w, λ)-CMA-ES for periodic parameters, with
  optional IPOP restarts that double the population. In Rust it is
  `AlgorithmCmaEs`, with `AlgorithmCmaEsArgs` and `CmaEsBatchSchedule`, in
  `polypus-optimizers`, with its own symmetric eigensolver (no new
  dependency). A seeded run reproduces on the same platform only. With
  restarts, the CUNQA walltime warning assumes a restart after every
  generation. See the training guide for when to prefer it over DE.
- CUNQA run-flow tests against a fake CUNQA SDK (`qraise`, `run`, `gather`,
  `qdrop`, C-3 counts, C-7 per-QPU seeds), and an opt-in test for a real CUNQA
  family, `POLYPUS_CUNQA_REAL=1 pytest tests/python/test_cunqa_real.py` (#357).
- OpenQASM 3 profile: `Circuit.from_qasm3`, `to_qasm3`, `param_names`; the
  importer records the declared classical width.
- `walltime` kwarg for the SLURM walltime of CUNQA virtual QPUs.
- `polypus.__version__`, PEP 561 typing stubs and `py.typed`.
- `qaoa_params_to_dict` helper, with the real QAOA parameter order documented.
- Angle expressions from `polypus.Param` in Python (`2 * Param(0)`,
  `polypus.sin`, `pi`, ...), stored as circuit expressions.
- `Circuit.clbits_width`: the classical width an imported program declares.
- A warning when training on CUNQA is projected to outlive the walltime.
- Native circuit drawings: `Circuit.draw()` (Unicode or ASCII text, or SVG),
  `print(circuit)` and SVG display in Jupyter; `draw_text`/`draw_svg` and
  `Display` in Rust. No new dependencies.
- Adaptive Differential Evolution: `polypus.DE(strategy="shade")` (SHADE) and
  `strategy="lshade"` (L-SHADE, whose population shrinks linearly towards 4).
  For `"lshade"`, `generations` sets the evaluation budget, `population_size ×
  (generations + 1)` as for the classic DE, and the run spends it in more,
  smaller generations: more sequential rounds on several QPUs, where
  `"shade"` suits better. The default, `strategy="rand1bin"`, is the classic
  DE, unchanged; an unknown name raises `ValueError`. In Rust, `DeStrategy`
  (with `max_generations`) and
  `AlgorithmDifferentialEvolution::optimize_with_strategy`.

### Changed
- **BREAKING (Rust):** `polypus_backend_conformance::circuits::as_qasm2`
  returns a `Result` instead of panicking on an unexportable circuit (#350).
- **BREAKING:** the native backend derives each circuit's seed from the base
  seed and the circuit's index with SplitMix64 instead of `seed + index`, so
  neighbouring seeds no longer share streams shifted by one circuit (circuit
  `k` of seed `s` equalled circuit `k-1` of seed `s+1`). The same seed and batch
  still reproduce bit-for-bit (C-7), but counts and `backend="polypus"`
  training trajectories for a fixed seed differ from earlier releases (#348).
- `import polypus` no longer calibrates the gate-parallel threshold or writes
  the cache: the first native statevector run of a process does it
  (`polypus.statevector`, or `run_quantum_circuit`/`train` with
  `backend="polypus"`). The cache, `POLYPUS_NO_AUTOCALIBRATE` and the
  SLURM/OpenMPI rank check are unchanged (#351).
- **BREAKING:** Python 3.9 (end of life) is no longer supported;
  `requires-python` is `>=3.10` and wheels are built for CPython 3.10–3.13.
  Every wheel is installed and smoke-tested on its platform before it is
  published (#360).
- **BREAKING (Rust):** `ParameterizedCircuit` and `ConcreteCircuit` fields are
  private; read them with `num_qubits()`, `num_params()`, `gates()` and
  `ConcreteCircuit::into_gates()`. `ConcreteCircuit::new` validates its gates
  like the builder and panics on invalid ones; `try_new` and `try_with_gates`
  (for transpilers) return the error instead (#349).
- **BREAKING (Rust):** `GateInstruction`, `CircuitError`, `BackendError` and
  `SimError` are `#[non_exhaustive]`; a `match` on them needs a wildcard arm.
  The simulator reports an instruction it does not know as
  `SimError::UnsupportedInstruction` (#349).
- **BREAKING:** the declared classical width is kept through circuit binding
  and export (`fix(circuit)!`).
- **BREAKING:** migrated to the `rand` 0.10 API (`fix(deps)!`); seeded streams
  may differ from earlier releases.
- Native and Aer backends lower gates outside Aer's basis before running; the
  Aer lowering of a repeated OpenQASM program is cached.
- **BREAKING:** `templates::qft` and `qft_with_options` return a `Result`:
  above `MAX_QFT_QUBITS` (2827) qubits, whose gates would exceed what the
  OpenQASM importers read back, they fail with
  `CircuitError::TemplateTooLarge` (`ValueError` in Python) instead of
  exhausting memory (#356).
- **BREAKING:** QMIO has no default endpoint; set `ZMQ_SERVER` or pass
  `options={"endpoint": ...}`. The transport threat model is documented.
- **BREAKING:** QMIO reports a Qiskit circuit as `UnsupportedCircuit`
  instead of a generic `QmioError`.
- **BREAKING:** `run_qc_in_qpu`, `run_qcs_in_qpu` and `run_worker_qpu.py` are
  removed; they could not run against any supported CUNQA version.
- The QIR export declares the circuit's classical width in
  `required_num_results`, as the OpenQASM exports do.
- **BREAKING:** `expectation_value` and `expectation_values` raise `ValueError`
  for counts that sum to zero instead of returning `0.0`.
- Rust no longer sends the `backend` and `id` kwargs that `run_qcs` never read
  (contract C-1).
- The QML candidate window also follows the backend's concurrency, within the
  memory bound.
- **BREAKING (Rust):** `polypus_orchestration::DeConfig` has a `strategy`
  field; `DeStrategy::Rand1Bin` keeps the previous behaviour.

### Deprecated
- Rust: `ConcreteCircuit::to_qasm2` and `to_qir`, which panic on circuits
  they cannot export; use `try_to_qasm2` and the new `try_to_qir`. No
  internal caller uses the panicking exports any more (#350).

### Fixed
- With `infrastructure="cunqa"`, a shot split that is uneven across the QPUs
  (e.g. 10 shots on 3 QPUs) gives every replica its own QPU and seed: the
  calls per batch size restarted at QPU 0 and the base seed, so two replicas
  shared both and one QPU idled. CUNQA's `run_qcs` takes a `qpu_offset` kwarg
  (C-1) (#370).
- An expression id (`ExprId`) is a 64-bit checksum of the expression, resolved
  through the circuit and trusted only after the stored nodes compare equal;
  with the 32-bit checksum it had, a circuit could take a foreign id for a
  different expression it held at the same position. A circuit stores each
  distinct expression once, and accepts an id of an identical expression
  wherever it stored it (#354).
- The OpenQASM 2.0 export renames a gate declaration named `q` or `c`
  (`c_1`), which clashed with the registers it declares and made Qiskit's
  `qasm2.loads` reject the program; its callers are printed with the new
  name (#352).
- **BREAKING (Rust):** the native simulator rejects a gate that names one
  qubit twice (`Cx(0, 0)` in a hand-built `ConcreteCircuit`) with the new
  `SimError::RepeatedQubit` instead of writing out of bounds; the kernels'
  operand checks are now `assert!`s (#342).
- **BREAKING (Rust):** `Statevector::expectation_z` returns
  `Result<f64, SimError>` and rejects a qubit outside the register instead of
  returning a wrong value or overflowing; a repeated qubit cancels (`Z·Z = I`).
  An out-of-range `parallel_threshold` in the calibration cache (e.g. `0`) is
  ignored as a cache miss (#353).
- OpenQASM 2.0 export keeps angles at full round-trip precision.
- With `infrastructure="cunqa"`, a non-Aer `backend` or a `noise_model` raises
  `ValueError` instead of being silently ignored.
- A Qiskit circuit with free parameters is rejected before any backend, and
  `backend_compatibility` reports it.
- Backends report circuits OpenQASM 2.0 cannot express instead of panicking.
- QMIO: reply bitstrings are validated and malformed `QMIO_*` tuning variables
  are rejected.
- Subprocess backend bounds the worker frame length.
- The OpenQASM 2.0 importer enforces the OpenQASM 3 importer's size budgets
  (source, tokens, declarations, instructions after broadcasting, barrier
  operands, expression nodes) and no longer tokenizes the whole source up
  front: a few hundred bytes could take gigabytes (#343).
- The OpenQASM 3 importer bounds the qubits all barriers list: a full-register
  `barrier q;` counted as one instruction however large `q` was (#344).
- The OpenQASM importers normalise the instructions in place instead of
  copying them; QIR bitcode export stops `llvm-as` after 120 s and always
  reaps it (#356).
- CUNQA receives `cores_per_qpu` under the name CUNQA >= 2.4 expects.
- Python errors and OS-entropy failures surface as typed errors, not panics.
- Memory budget honours `K`/`M`/`G` suffixes and detects the cgroup/RAM limit.

## [0.7.2] - 2026-09-30

### Added
- Third-party backends can be registered without touching this repository;
  conformance battery and reference templates.
- OpenQASM 2.0 coverage: gate declarations, the rest of `qelib1.inc`, angle
  expressions over free parameters, and named free parameters.
- Supervised QML: `y_train` in `polypus.qml.train` and batched inference with
  `polypus.qml.predict`.
- `RunResult.merged_counts`; `RunResult.counts` is now a `list[dict]` (C-3).
- `QuantumBackend` extracted into the pyo3-free `polypus-backend` crate.

### Changed
- **BREAKING:** `CompoundResult::mass_fractions` iterates in symbol order
  (`fix(physics)!`).
- Counts dicts are returned in ascending bitstring order.

### Fixed
- Last-write-wins for repeated measurements into one classical bit.
- Calibration cache is keyed by hardware fingerprint and its writes are locked.
- Aer counts keys no longer contain per-register spaces.

## [0.7.1] - 2026-09-23

### Added
- Gate-parallel threshold calibrated per machine at import and cached by
  thread count; a warning is emitted when it is uncalibrated.
- `polypus-physics` reads ENDF-6 data from local `reference-data`, never
  downloading it.

### Fixed
- Planner respects backend concurrency caps and bases wave splitting on the
  memory limit.
- `QmioBackend` no longer holds its socket lock across await points.
- `CancelToken` handle exposed from `Scheduler::run`.
- PSO rejects non-finite bounds.

## [0.7.0] - 2026-09-14

### Added
- Native GIL-free cost observables (QUBO/Ising) with a callback fallback.
- Optional gate fusion in the simulator (`fusion`, on by default).
- Statevector simulation can be interrupted mid-run.
- Polyenergetic beams, coherent scattering and ENDF-6 data in `polypus-physics`.
- QFT circuit template and native `swap` gate.
- Seeds exposed for `run_quantum_circuit` and `train` (C-7).

### Changed
- **BREAKING:** `statevector()` returns a NumPy array (`feat(bindings)!`).
- Core refactor: the Planner, `polypus-infrastructure`, `polypus-orchestration`
  and `polypus-evaluation` crates; the Python edge is a thin layer over
  `Scheduler`/`Flow`.
- Native backend parallelises the DE population across circuits.

### Fixed
- Training and `run_quantum_circuit` release the GIL and honour Ctrl+C.
- Unique auto-generated run ids; validated `qml.train` inputs.
- Backend counts, expectation values and gate parameters are validated
  (C-3, C-5); out-of-range qubit indices are rejected.
- Native and Aer concurrency bounded by a memory budget.
- QMIO distinguishes a send failure from an unknown result.

## [0.6.0] - 2026-07-03

### Added
- QMIO backend, native statevector simulator for local tests, and QIR support.
- Pure-Rust circuit crate (`polypus-circuit`) with OpenQASM 2.0 import/export.
- `polypus-optimizers` extracted as a pure-Rust crate; Cargo workspace.
- Structured JSON logger with a `logs/` directory.
- `polypus-physics` crate (photon interaction, geometry and voxels).

## [0.5.0] - 2026-06-10

### Added
- Oracle abstraction and the `polypus.qml` module.
- `install.sh` installer and compilation tests.

(The tag is spelled `v.0.5.0`.)

## [0.4.0] - 2026-05-22

### Changed
- Python bindings split into modules; documentation rebuilt for 0.4.

## [0.3.0] - 2026-05-05

### Added
- Quantum Natural Gradient (QNG) and Particle Swarm (PSO) optimizers.
- CUNQA 2.x support.

## [0.2.0] - 2025-11-04

### Added
- Modular infrastructures (local, CUNQA, QMIO) and differential evolution with
  batch processing and shot distribution.
- `run_quantum_circuit` for single runs and multiple expectation values.

## [0.1.0] - 2025-05-26

Initial release.

[Unreleased]: https://github.com/Bahia-Software/polypus/compare/v0.7.2...HEAD
[0.7.2]: https://github.com/Bahia-Software/polypus/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/Bahia-Software/polypus/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/Bahia-Software/polypus/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Bahia-Software/polypus/compare/v.0.5.0...v0.6.0
[0.5.0]: https://github.com/Bahia-Software/polypus/compare/v0.4.0...v.0.5.0
[0.4.0]: https://github.com/Bahia-Software/polypus/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Bahia-Software/polypus/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Bahia-Software/polypus/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Bahia-Software/polypus/releases/tag/v0.1.0
