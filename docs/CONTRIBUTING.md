# Contributing to Polypus

Thanks for your interest in Polypus! This guide covers environment setup, the
test suite, and how changes get merged. It applies to human and AI-assisted
contributions alike. Two companion documents complete the picture:
[`docs/ENGINEERING.md`](ENGINEERING.md) (code-level invariants and Rust style) and
[`docs/CONTRACTS.md`](CONTRACTS.md) (the inter-layer contracts).

## Setting up

The interactive installer handles dependencies, the wheel build and the test
suite:

```bash
bash install.sh            # guided
bash install.sh --yes      # non-interactive (CI)
```

Manual route: `pip install -r requirements-dev.txt`, then
`python -m build packages/polypus_python/ && pip install packages/polypus_python/`,
then `maturin develop --release --features extension-module`.

**Toolchain:** the Minimum Supported Rust Version is declared as
`rust-version` in the workspace `Cargo.toml` — that field is the single place
it lives (Cargo enforces it and Clippy picks it up automatically).

## Running the checks

Everything CI runs, you can run locally:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                     # all crates, no hand-maintained list
cargo test -p polypus --features qmio      # QMIO backend (skips gracefully without QPU access)
cargo deny check                           # licenses + advisories (needs cargo-deny, deny.toml)
pytest tests/python                        # needs the extension installed
```

A green run is a **precondition** for opening a PR, not something to fix
afterwards.

> Note: we intentionally use `--workspace` instead of listing crates one by
> one — a hand-maintained crate list silently goes stale every time a crate is
> added or split (it already happened once). The one deliberate exception is
> CI's all-features pass over the pure-Rust crates (see
> `.github/workflows/ci.yml`): `--workspace --all-features` would enable
> `extension-module`, which breaks building the test targets.

Performance claims have their own bar: before claiming a speed-up, back it
with the scripts in `benchmarks/` (`run_benchmarks.py`,
`bench_native_vs_qiskit.py`, `bench_batching.py`) and include the numbers in
the PR description.

## Lint policy

`clippy.toml` only tunes lint thresholds/behaviour. Lint *levels* are set in CI
(`-D warnings`) and, optionally, in the `[workspace.lints]` table of the root
`Cargo.toml`. Do not sprinkle `#[allow(...)]` to silence CI; if a lint is wrong
for the whole workspace, change the policy in one place and say why in the PR.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org):
`fix(cunqa): align disconnect kwargs with contract C-1`,
`feat(circuit): add cp to the QASM importer`, `test(seam): mock CUNQA contract`.
The type+scope prefix feeds the changelog; the subject line explains the
*intent*, not the action ("Update" tells a reviewer nothing).

## Pull requests

All changes land through a PR against `main` — the branch is protected: at
least one approving review from someone other than the author, and CI green.
The PR template asks for three declarations that we take seriously:

1. **Scope** — which module(s) the change targets.
2. **Collateral files** — every touched file *outside* that scope, each with a
   one-line justification. Unexplained collateral changes are the single most
   common source of regressions in this project.
3. **Contracts** — whether the change touches anything listed in
   [`docs/CONTRACTS.md`](CONTRACTS.md). If it does, the same PR must
   update the contract document **and** its enforcing test. *No seam change
   without a seam test.*

Keep PRs small and single-purpose. If your work naturally produced unrelated
improvements, split them out — small PRs get reviewed in hours, mixed ones sit
for days.

## Adding things that have a defined shape

- **A gate**: four places (QASM export, QASM import, simulator, QIR) plus a row
  in the vocabulary equivalence test — see contract C-2.
- **A backend/provider**: one `BackendConfig` variant + one `QuantumBackend`
  impl + a contract entry + a seam test. Algorithms must not change.
- **An optimizer**: implement `Optimizer` against `EvaluationOracle`; respect
  contract C-5 (validated preconditions, `best_fitness == f(best_params)`).
- **A crate**: inherit the workspace metadata (`version.workspace = true`,
  `edition.workspace = true`, `license.workspace = true`, …), add it to
  `[workspace] members` and — if other crates consume it — to
  `[workspace.dependencies]`. If it is pure Rust, also add it to CI's
  all-features test list (see the note above). Pure crates must not depend on
  `pyo3` — see [`docs/ENGINEERING.md`](ENGINEERING.md) §2.

## Reporting bugs and security issues

Use the issue tracker for bugs and feature requests. For anything with
security impact, do **not** open a public issue — follow
[`SECURITY.md`](../SECURITY.md) instead.

## License

Polypus is licensed under the **European Union Public Licence (EUPL),
version 1.2** — see the [`LICENSE`](../LICENSE) file and the `license` field in
the workspace `Cargo.toml`. By submitting a contribution you agree that it is
your own work (or you have the right to submit it) and that it is provided
under the project license.

## Versioning and releases

Single source of truth: the workspace `Cargo.toml` version (SemVer). Release
tags must match it exactly (`vX.Y.Z`), and every release updates
[`CHANGELOG.md`](../CHANGELOG.md). The short checklist lives in
[`docs/RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).

## AI-assisted development

AI-generated contributions are welcome and held to the same bar: they arrive
with their tests, they respect the scope/collateral discipline above, and the
author remains responsible for the diff.

Practical guidance for assistant sessions (Claude Code, Cursor, etc.):

- Include **this file, [`docs/CONTRACTS.md`](CONTRACTS.md) and
  [`docs/ENGINEERING.md`](ENGINEERING.md)** in the session context — all three are
  written for exactly that purpose.
- Team members: the internal `CLAUDE.md` (private repo, never committed here)
  mirrors the verification commands above; keep the two in sync when this
  section changes.
- Before proposing a diff, run the same checks CI runs (see "Running the
  checks"). An assistant-produced PR with red CI is treated like any other
  red PR: it waits.
