# The checks CI runs, one recipe each (#421). `.github/workflows/ci.yml` calls
# these recipes instead of repeating their commands, so what runs locally and
# what runs in CI cannot drift apart; `tests/docs/test_ci_recipes.py` fails if
# a CI step runs a command of its own or names a recipe this file lacks.
#
#     just --list      every recipe, with its description
#     just ci          everything CI checks that this machine can reproduce
#     just ci-all      that, plus the jobs on other toolchains (MSRV, nightly)
#
# What CI does and one machine cannot: build the wheel on macOS and Windows,
# and run the Python suite on each supported Python (the matrix legs, with
# their POLYPUS_REQUIRE_* and POLYPUS_ALLOWED_SKIPS variables). `just pytest`
# uses the Python of the active environment. Installing a tool (cargo-deny,
# cargo-machete, ruff, maturin, llvm-as) is left to you; docs/CONTRIBUTING.md
# lists them.
#
# Recipes run under bash. On Windows they run under `sh`, Git for Windows'
# (the one GitHub runners provide): `bash` there can resolve to the WSL
# launcher in System32 instead. CI runs only `build-wheel` on Windows.

set shell := ["bash", "-euo", "pipefail", "-c"]
set windows-shell := ["sh", "-eu", "-c"]

# The minimum supported Rust (`rust-version` in Cargo.toml, which a test
# checks against this value) and the qmio feature's own (zeromq 0.7).
msrv := "1.85.0"
qmio_msrv := "1.89.0"

# A dated nightly for the dependency floors, fuzzing and Miri. Bump it
# deliberately: a newer one can need a newer cargo-fuzz.
nightly := "nightly-2026-10-22"

# The pure-Rust crates tested with every feature: the one hand-maintained crate
# list (`--workspace --all-features` would enable `polypus`'s extension-module).
# The crates that link libpython stay out: polypus-infrastructure,
# polypus-evaluation, polypus-orchestration (pyo3-free, but it links libpython
# through polypus-infrastructure) and polypus.
pure_crates := "-p polypus-circuit -p polypus-sim -p polypus-physics -p polypus-optimizers -p polypus-observable -p polypus-qml -p polypus-backend -p polypus-backend-conformance -p polypus-subprocess-backend"

# Where `wheel` builds; emptied first, so only this build's wheel is installed.
wheel_dir := "target/wheels-just"

# List the recipes.
default:
    @just --list

# Everything CI checks that this machine can reproduce (see the header).
ci: fmt ruff lockfile deny machete clippy doc docs-site test-rust hygiene wheel compile-examples pytest stubtest conformance-aer tutorials

# `ci`, plus the jobs on other toolchains and the packaging checks.
ci-all: ci msrv minimal-versions package semver fuzz miri

# ---------------------------------------------------------------- lints

# Rust formatting.
fmt:
    cargo fmt --all --check

# Python lint and formatting (ruff pinned in requirements-dev.txt).
ruff:
    ruff check
    ruff format --check

# Every feature, so the dependencies only an optional feature pulls in
# (plotters, qmio) are checked too.
# Licenses, advisories, bans and sources (deny.toml).
deny:
    cargo deny --all-features check

# Cargo.lock matches the manifests.
lockfile:
    cargo metadata --locked --format-version 1 > /dev/null

# Dependencies declared but never used.
machete:
    cargo machete

# Clippy, default features and the qmio backend.
clippy: clippy-workspace clippy-qmio

# Never --all-features: it would enable `polypus`'s extension-module, which
# breaks linking the test targets.
# Clippy over the workspace, default features.
clippy-workspace:
    cargo clippy --locked --workspace --all-targets -- -D warnings

# Clippy over the optional QMIO backend, in its crate and through `polypus`.
clippy-qmio:
    cargo clippy --locked -p polypus-infrastructure --features qmio --all-targets -- -D warnings
    cargo clippy --locked -p polypus --features qmio --all-targets -- -D warnings

# The rustdoc of every workspace crate; a broken intra-doc link fails it.
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps

# The documentation site and its hooks (needs requirements-docs.txt).
docs-site:
    pytest tests/docs
    mkdocs build --strict -d target/site

# Repository hygiene of the commits since `base` (commit first).
hygiene base="origin/main":
    python .github/scripts/hygiene.py --base {{ base }}

# ---------------------------------------------------------------- Rust tests

# Every Rust test pass CI runs.
test-rust: test-workspace test-features test-memory-safety test-logger test-qmio test-templates

# Every workspace crate, default features.
test-workspace:
    cargo test --locked --workspace

# The pure-Rust crates with every feature (`parallel` in the simulator).
test-features:
    cargo test --locked {{ pure_crates }} --all-features

# The build downstream crates get: release, without overflow checks, with
# the parallel kernels.
# The simulator's memory-safety guards.
test-memory-safety:
    CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=false cargo test --locked --release -p polypus-sim --features parallel --test memory_safety
    CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=false cargo test --locked --release -p polypus-sim --features parallel --lib kernels::

# polypus-logger at the wheel's log level (its features exclude each other).
test-logger:
    cargo test --locked -p polypus-logger --features debug-logs

# The optional QMIO backend, in its crate and through `polypus`.
test-qmio:
    cargo test --locked -p polypus-infrastructure --features qmio
    cargo test --locked -p polypus --features qmio

# The two backend templates, standalone crates outside the workspace.
test-templates:
    cargo test --locked --manifest-path examples/rust-backend-template/Cargo.toml
    cargo test --locked --manifest-path examples/python-backend-template/Cargo.toml

# ---------------------------------------------------------------- Python

# The first install pulls the runtime dependencies and keeps any already
# installed that satisfy them (a pinned Qiskit stays); the second replaces an
# installed polypus of the same version.
# Build the polypus-quantum wheel and install it in the active environment.
wheel:
    rm -rf {{ wheel_dir }}
    maturin build --release --out {{ wheel_dir }}
    pip install {{ wheel_dir }}/polypus_quantum-*.whl
    pip install --force-reinstall --no-deps {{ wheel_dir }}/polypus_quantum-*.whl

# Extra arguments go to maturin (`--out dist`).
# Build the wheel without installing it (the platform smoke test).
[positional-arguments]
build-wheel *args:
    maturin build --release "$@"

# Examples and benchmarks are not run, but they must stay valid Python.
compile-examples:
    python -m compileall -q examples benchmarks

# Extra arguments go to pytest as given (`just pytest -k "qml and not slow"`).
# The Python suite against the installed wheel.
[positional-arguments]
pytest *args:
    pytest tests/python -rs --strict-markers "$@"

# The typing stubs against the installed extension.
stubtest:
    POLYPUS_NO_AUTOCALIBRATE=1 python -m mypy.stubtest polypus --allowlist tests/python/stubtest_allowlist.txt --mypy-config-file tests/python/stubtest_mypy.ini

# One OpenMP thread: Aer is fragile when it fans experiments out from a
# cargo-test thread.
# The conformance battery against Aer (needs the installed wheel).
conformance-aer:
    OMP_NUM_THREADS=1 cargo test --locked -p polypus-infrastructure --test conformance -- --ignored --exact local_aer_backend_conformance --nocapture

# Needs the installed wheel, nbconvert, ipykernel and matplotlib. The executed
# copy is discarded, so the notebooks in the checkout stay without outputs.
# Execute every tutorial notebook.
tutorials:
    #!/usr/bin/env bash
    set -euo pipefail
    export MPLBACKEND=Agg
    find tutorials -name '*.ipynb' -print0 | sort -z | while IFS= read -r -d '' nb; do
        echo "Executing $nb"
        jupyter nbconvert --to notebook --execute --stdout "$nb" > /dev/null
    done

# ---------------------------------------------------------------- other toolchains

# The workspace on the MSRV, and the qmio backend on its own minimum.
msrv: msrv-workspace msrv-qmio

# The workspace on the MSRV.
msrv-workspace:
    rustup toolchain install {{ msrv }} --profile minimal --no-self-update
    cargo +{{ msrv }} check --locked --workspace --all-targets

# The qmio backend on its minimum Rust.
msrv-qmio:
    rustup toolchain install {{ qmio_msrv }} --profile minimal --no-self-update
    cargo +{{ qmio_msrv }} check --locked -p polypus-infrastructure --features qmio --all-targets
    cargo +{{ qmio_msrv }} check --locked -p polypus --features qmio --all-targets

# Cargo.lock is restored afterwards, even when the check fails.
# Build with every direct dependency at its declared floor (nightly resolver).
minimal-versions:
    #!/usr/bin/env bash
    set -euo pipefail
    backup=$(mktemp)
    cp Cargo.lock "$backup"
    trap 'cp "$backup" Cargo.lock; rm -f "$backup"' EXIT
    {{ just_executable() }} minimal-versions-resolve
    {{ just_executable() }} minimal-versions-check

# Resolve every direct dependency to its floor (rewrites Cargo.lock).
minimal-versions-resolve:
    rustup toolchain install {{ nightly }} --profile minimal --no-self-update
    cargo +{{ nightly }} update -Z direct-minimal-versions

# Check the build against the resolved floors.
minimal-versions-check:
    cargo check --workspace --all-targets
    cargo check -p polypus-physics --all-features --all-targets
    cargo check -p polypus-sim --features parallel --all-targets
    cargo check -p polypus --features qmio --all-targets

# Package every crate and build it from its .crate, as a release does.
package:
    CARGO_INCREMENTAL=0 cargo package --workspace --locked

# Crates not published yet have no baseline and are skipped.
# The public API of the crates on crates.io against their latest release.
semver:
    #!/usr/bin/env bash
    set -euo pipefail
    new=$(python .github/scripts/publish_crates.py --unpublished)
    exclude=()
    for name in $new; do
        echo "::notice::$name is not on crates.io yet: skipped"
        exclude+=(--exclude "$name")
    done
    # `${a[@]+...}`: an empty array under `set -u` is an error in bash 3.2 (macOS).
    CARGO_INCREMENTAL=0 cargo semver-checks --workspace --default-features ${exclude[@]+"${exclude[@]}"}

# Fuzz both OpenQASM importers for `seconds` each (needs cargo-fuzz).
fuzz seconds="60":
    rustup toolchain install {{ nightly }} --profile minimal --no-self-update
    cargo +{{ nightly }} fuzz run from_qasm2 -- -max_total_time={{ seconds }} -rss_limit_mb=2048
    cargo +{{ nightly }} fuzz run from_qasm3 -- -max_total_time={{ seconds }} -rss_limit_mb=2048

# MIRIFLAGS: Tree Borrows (crossbeam-epoch trips Stacked Borrows), isolation
# off (the simulator reads the calibration cache and CPU info) and leaks
# ignored (rayon's pool outlives the test thread).
# Miri over polypus-sim's unsafe kernels.
miri:
    rustup toolchain install {{ nightly }} --profile minimal --component miri,rust-src --no-self-update
    cargo +{{ nightly }} miri setup
    MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-disable-isolation -Zmiri-ignore-leaks" cargo +{{ nightly }} miri test --locked -p polypus-sim --features parallel --lib "kernels::"
    MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-disable-isolation -Zmiri-ignore-leaks" cargo +{{ nightly }} miri test --locked -p polypus-sim --features parallel --test memory_safety
