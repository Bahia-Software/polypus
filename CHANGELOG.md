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
- OpenQASM 3 profile: `Circuit.from_qasm3`, `to_qasm3`, `param_names`; the
  importer records the declared classical width.
- `walltime` kwarg for the SLURM walltime of CUNQA virtual QPUs.
- `polypus.__version__`, PEP 561 typing stubs and `py.typed`.
- `qaoa_params_to_dict` helper, with the real QAOA parameter order documented.
- Angle expressions from `polypus.Param` in Python (`2 * Param(0)`,
  `polypus.sin`, `pi`, ...), stored as circuit expressions.
- `Circuit.clbits_width`: the classical width an imported program declares.
- A warning when training on CUNQA is projected to outlive the walltime.

### Changed
- **BREAKING:** the declared classical width is kept through circuit binding
  and export (`fix(circuit)!`).
- **BREAKING:** migrated to the `rand` 0.10 API (`fix(deps)!`); seeded streams
  may differ from earlier releases.
- Native and Aer backends lower gates outside Aer's basis before running; the
  Aer lowering of a repeated OpenQASM program is cached.
- **BREAKING:** QMIO has no default endpoint; set `ZMQ_SERVER` or pass
  `options={"endpoint": ...}`. The transport threat model is documented.
- **BREAKING:** QMIO reports a Qiskit circuit as `UnsupportedCircuit`
  instead of a generic `QmioError`.
- **BREAKING:** `run_qc_in_qpu`, `run_qcs_in_qpu` and `run_worker_qpu.py` are
  removed; they could not run against any supported CUNQA version.
- The QIR export declares the circuit's classical width in
  `required_num_results`, as the OpenQASM exports do.
- The QML candidate window also follows the backend's concurrency, within the
  memory bound.

### Fixed
- OpenQASM 2.0 export keeps angles at full round-trip precision.
- A Qiskit circuit with free parameters is rejected before any backend, and
  `backend_compatibility` reports it.
- Backends report circuits OpenQASM 2.0 cannot express instead of panicking.
- QMIO: reply bitstrings are validated and malformed `QMIO_*` tuning variables
  are rejected.
- Subprocess backend bounds the worker frame length.
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
