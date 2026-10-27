# Contributing to Polypus

Thanks for your interest in Polypus! This guide covers environment setup, the
test suite, and how changes get merged. It applies to human and AI-assisted
contributions alike. Two companion documents complete the picture:
[`docs/ENGINEERING.md`](ENGINEERING.md) (code-level invariants and Rust style) and
[`docs/CONTRACTS.md`](CONTRACTS.md) (the inter-layer contracts).

## Setting up

> Same project, three names — `pip install polypus-quantum` · `import polypus` · `cargo add polypus`. The PyPI name `polypus` is taken, so the distribution is `polypus-quantum`; the import name and the Rust crate both stay `polypus`.

The interactive installer handles dependencies, the wheel build and the test
suite:

```bash
bash install.sh            # guided
bash install.sh --yes      # non-interactive (CI)
```

Development route — an editable install of both packages (`polypus` and the
`polypus_python` helpers, which live side by side under `python/`):

```bash
pip install -r requirements-dev.txt
```

```bash
maturin develop --release
```

Edits to any `.py` file under `python/` take effect on the next import, with no
rebuild. After a Rust change, run `maturin develop --release` again; it rebuilds
the extension incrementally and drops it in place next to
`python/polypus/__init__.py` (git-ignored). Keep `--release`, matching the
wheel CI tests: several Python tests time real simulations.

Wheel route — what `install.sh`, CI and users get (one wheel carries both the
`polypus` extension and the `polypus_python` helpers):

```bash
pip install -r requirements-dev.txt
```

```bash
maturin build --release --out dist
```

```bash
pip install --force-reinstall dist/polypus_quantum-*.whl
```

`--force-reinstall` makes pip replace an already-installed wheel that has the
same version after a rebuild; it also replaces an editable install.

**Toolchain:** the Minimum Supported Rust Version is declared as
`rust-version` in the workspace `Cargo.toml` — that field is the single place
it lives (Cargo enforces it and Clippy picks it up automatically). The
workspace uses the MSRV-aware resolver (`resolver = "3"`), so `cargo update`
avoids dependency releases that need a newer Rust than `rust-version`; the
`msrv` CI job still builds with `--locked` and catches anything that slips
through. Dependabot does not honour `rust-version`
([dependabot-core#5423](https://github.com/dependabot/dependabot-core/issues/5423)),
so `.github/dependabot.yml` ignores `uuid >=1.27`, which needs Rust 1.89. Remove
that `ignore` when `rust-version` is raised to 1.89 or later (#276, #278). If
another dependency starts to need a newer Rust, add an `ignore` for it the same
way.

## Running the checks

These are the checks CI runs (`.github/workflows/ci.yml` and `hygiene.yml`);
each can be run locally. They use the Rust that `rust-toolchain.toml` pins,
which rustup selects in this checkout (`rustup toolchain install` installs
it). Lints and builds:

```bash
cargo fmt --all --check
ruff check && ruff format --check          # ruff pinned in requirements-dev.txt
cargo deny check                           # licenses + advisories (needs cargo-deny, deny.toml)
cargo metadata --locked --format-version 1 > /dev/null   # Cargo.lock matches the manifests
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo clippy --locked -p polypus-infrastructure --features qmio --all-targets -- -D warnings
cargo clippy --locked -p polypus --features qmio --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps
python .github/scripts/hygiene.py --base origin/main   # repository hygiene (commit first)
cargo machete                              # unused dependencies (needs cargo-machete)
```

Declared dependency floors (the `rust-minimal-versions` and `python-floors`
jobs). The first rewrites `Cargo.lock`; restore it afterwards with
`git checkout Cargo.lock`. The second needs a fresh environment on Python 3.10
and installs the build requirements at their floors first (see the job):

```bash
cargo +nightly-2026-10-22 update -Z direct-minimal-versions
cargo check --workspace --all-targets
uv pip install --resolution lowest-direct -r pyproject.toml -r requirements-dev.txt
```

Rust tests:

```bash
cargo test --locked --workspace            # every crate, default features
                                           # (polypus-subprocess-backend's tests spawn a Python
                                           #  worker; they skip gracefully if python3 is absent)
cargo test --locked --all-features -p polypus-circuit -p polypus-sim -p polypus-physics \
  -p polypus-optimizers -p polypus-observable -p polypus-backend \
  -p polypus-backend-conformance -p polypus-subprocess-backend
                                           # pure-Rust crates, every feature
CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=false \
  cargo test --locked --release -p polypus-sim --features parallel --test memory_safety
CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=false \
  cargo test --locked --release -p polypus-sim --features parallel --lib kernels::
cargo test --locked -p polypus-logger --features debug-logs   # the wheel's log level
cargo test --locked -p polypus-infrastructure --features qmio
cargo test --locked -p polypus --features qmio   # QMIO backend (skips gracefully without QPU access)
cargo test --locked --manifest-path examples/rust-backend-template/Cargo.toml
cargo test --locked --manifest-path examples/python-backend-template/Cargo.toml
```

Python tests, with the wheel installed (see Setting up):

```bash
python -m compileall -q examples benchmarks
pytest tests/python -rs --strict-markers   # -rs lists every skip with its reason
python -m mypy.stubtest polypus \
  --allowlist tests/python/stubtest_allowlist.txt \
  --mypy-config-file tests/python/stubtest_mypy.ini
                                           # typing stubs vs the installed extension
                                           # (mypy from requirements-dev.txt)
OMP_NUM_THREADS=1 cargo test --locked -p polypus-infrastructure --test conformance \
  -- --ignored --exact local_aer_backend_conformance
                                           # the conformance battery against Aer
```

A few tests skip when what they need is missing. CI provides it and sets a
variable that turns those skips into failures: `POLYPUS_REQUIRE_LLVM_AS=1`
for the QIR tests that run `llvm-as` (Rust and Python), `POLYPUS_REQUIRE_AER=1`
for the Aer conformance battery, and `POLYPUS_REQUIRE_FIXTURES_QISKIT=1` in the
leg that installs the Qiskit `tests/python/qasm3_fixtures.py` names, for the
pinned OpenQASM 3 fixtures (contract C-10). The other legs use the latest
Qiskit, where those fixture tests skip. The latest-Python leg installs pinned
Optuna, Nevergrad and pycma and sets `POLYPUS_REQUIRE_EXTERNAL_LIBRARIES=1`,
so the tests of `polypus.external` that train through them run there
(contract C-5); elsewhere they skip unless you install them.

CI also builds the wheel on Linux, macOS and Windows and executes every
notebook under `tutorials/` (`jupyter nbconvert --to notebook --execute`).
Three jobs set another toolchain with `RUSTUP_TOOLCHAIN`: `msrv` checks the
workspace with the MSRV (`rust-version` in `Cargo.toml`), and `fuzz` and
`miri` fuzz the OpenQASM importers and run Miri over `polypus-sim`'s `unsafe`
kernels on a dated nightly. Their jobs in `ci.yml` show the exact commands.

A green run is a **precondition** for opening a PR, not something to fix
afterwards.

> Note: CI tests with `--workspace` instead of listing crates one by one — a
> hand-maintained crate list silently goes stale every time a crate is added
> or split (it already happened once). The one deliberate exception is the
> all-features pass over the pure-Rust crates: `--workspace --all-features`
> would enable `extension-module`, which breaks building the test targets.

Performance claims have their own bar: before claiming a speed-up, back it
with the scripts in `benchmarks/` (`run_benchmarks.py`,
`bench_native_vs_qiskit.py`, `bench_batching.py`) and include the numbers in
the PR description.

## Repository hygiene

`.github/workflows/hygiene.yml` runs `.github/scripts/hygiene.py` (standard
library only, Python ≥ 3.9) on every pull request and push to `main`. It
rejects:

- internal notes and local tool configuration that `.gitignore` already
  excludes, and any file added although a versioned `.gitignore` ignores it;
- infrastructure data and credentials on added lines: cluster file-system
  paths, login-node names, private IPv4 addresses, private-key headers and
  GitHub tokens;
- notebooks under `tutorials/` with outputs or execution counts;
- blobs larger than 1 MiB;
- a `CITATION.cff` version different from `Cargo.toml` (contract C-6);
- submodules.

Run it before pushing:

```bash
python .github/scripts/hygiene.py --base origin/main
```

It analyses **commits** — the net diff between the merge base and `HEAD` —
not uncommitted changes, so commit first. Renamed files are analysed as new
files. Exit code 0 means clean, 1 violations, 2 a configuration or execution
error. The output names the file, line and rule, never the matched text.

A notebook with outputs is fixed with `nbstripout <notebook>`, then commit the
result; `nbstripout --install` once per clone prevents it.

Legitimate exceptions go in `.github/hygiene-allow.txt`, one per line as
`RULE<TAB>PATTERN<TAB># reason`: one rule, the narrowest path or glob, and a
reason. Forbidden names, notebooks, versions and submodules admit none.
Changes to the allow list, the script, the workflow or the policy-relevant
`.gitignore` files need explicit human review.

Maintainers also run private patterns, stored as a repository secret, on each
push to `main`; pull requests never receive that secret, so a green PR means
the public rules passed. A post-merge finding cannot undo the publication.
The check covers the net diff only, not every intermediate commit nor the
existing history, and does not replace checking your own changes before
pushing.

## Code quality dashboard

`.github/workflows/sonar.yml` sends the code to SonarQube Cloud on every pull
request and push to `main` (configuration in `sonar-project.properties`):
Rust coverage from `cargo llvm-cov` over the `test (Rust)` passes, the clippy
report, and Python coverage of `python/` from the pytest suite. Rust reached
only through the Python extension is not counted. The subprocess worker
template (`crates/polypus-subprocess-backend/python/worker_template.py`) is
left out of the coverage figures: it runs as a separate process spawned by the
Rust integration tests, which no tool instruments; Sonar still analyses it for
issues. The dashboard is informational: no check is required and the quality
gate does not fail the run. Without the `SONAR_TOKEN` secret (pull requests
from forks) the reports are still built but not uploaded. Python coverage
locally, with the wheel installed:

```bash
python -m coverage run -m pytest tests/python && python -m coverage combine && python -m coverage report
```

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

- **A gate**: six places (OpenQASM 2.0 export and import, OpenQASM 3 export,
  simulator, QIR export, Python bindings and their `.pyi` stub), the
  OpenQASM 3 importer if `stdgates.inc` names the gate, plus a row in the
  vocabulary equivalence test — see contract C-2.
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
version 1.2** — see the [`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE) files and the `license` field in
the workspace `Cargo.toml`. By submitting a contribution you agree that it is
your own work (or you have the right to submit it) and that it is provided
under the project license.

## Versioning and releases

Single source of truth: the workspace `Cargo.toml` version (SemVer). Release
tags must match it exactly (`vX.Y.Z`), and every release updates
[`CHANGELOG.md`](../CHANGELOG.md). The short checklist lives in
[`docs/RELEASING.md`](RELEASING.md).

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
