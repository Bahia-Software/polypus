# Releasing Polypus (with a citable DOI)

Short, order-sensitive checklist for cutting a release that is published to PyPI
and crates.io and archived on Zenodo with a DOI.

> **Status:** the GitHub → Zenodo integration is **already enabled** for
> `Bahia-Software/polypus`, and the citation metadata is prepared in
> [`CITATION.cff`](../CITATION.cff) and [`.zenodo.json`](../.zenodo.json).
> The first DOI was minted from the `v0.7.1` release; every release since
> reuses the same concept DOI automatically — see [Release steps](#release-steps).

## Why Zenodo had to be enabled first (already done)

Zenodo only archives releases created *after* the repository is switched on in
Zenodo; it does **not** archive past releases retroactively. That switch is now
**On**, so every release from here on is archived automatically — no action
needed on that front. (Recorded here so nobody turns it off or wonders about the
ordering later.)

## The metadata files

- [`CITATION.cff`](../CITATION.cff) — powers GitHub's "Cite this repository"
  button and gives a machine-readable citation. Author list, ORCIDs and license
  are final.
- [`.zenodo.json`](../.zenodo.json) — what Zenodo reads to fill the archived
  record (title, creators + ORCIDs, keywords, license, the HTML `description`
  including the **funding** acknowledgements, and related identifiers). This is
  the field grant justifications draw from, and it is already complete.

Keep the two in sync: the author list and ORCIDs are identical in both today,
and any future change must be applied to both.

## Versions

`[workspace.package].version` in the root `Cargo.toml` is the single source of
truth (contract C-6 in [`CONTRACTS.md`](CONTRACTS.md)).

| | release | `main` between releases |
|---|---|---|
| `Cargo.toml` and internal pins | `0.8.0` | `0.8.1-dev` |
| wheel / sdist (maturin, PEP 440) | `0.8.0` | `0.8.1.dev0` |
| `polypus.__version__` | `0.8.0` | `0.8.1.dev0` |
| `CITATION.cff` `version` | `0.8.0` | `0.8.0` (the last release) |

A pull request that breaks the API of a crate on crates.io raises `main` to the
next minor (`0.9.0-dev`); the `semver` CI job enforces it
([`CONTRIBUTING.md`](CONTRIBUTING.md#versioning-and-releases)).

The release gate (`gate` job of [`release.yml`](../.github/workflows/release.yml),
rules in [`release_check.py`](../.github/scripts/release_check.py)) refuses a
tag, or a manual run with `target=pypi`, unless:

- the tag is `vX.Y.Z` and equals `v` + the workspace version, which has no
  `-dev` suffix;
- every internal pin in `[workspace.dependencies]` equals that version;
- `CHANGELOG.md` has a `## [X.Y.Z] - YYYY-MM-DD` section;
- `CITATION.cff` has the same `version` and `date-released`;
- `ci.yml` has a successful push run for the tagged commit.

Run the same check locally with
`python .github/scripts/release_check.py --tag vX.Y.Z --publish` (Python 3.11+).

## Release steps

1. **Freeze.** The release ships what is merged on `main`. Every workflow is
   green on the commit to release.
2. **Release pull request** (branch `release/X.Y.Z`):
   1. Set the version of every crate, the internal pins and `Cargo.lock` in
      one step with [cargo-edit](https://github.com/killercup/cargo-edit):
      ```bash
      cargo set-version --workspace X.Y.Z
      for t in examples/*-backend-template; do cargo metadata --format-version 1 --manifest-path "$t/Cargo.toml" > /dev/null; done
      ```
      The second line refreshes the lockfiles of the backend templates, which
      depend on the workspace crates by path. Also set the `version` of the
      BibTeX entry in `README.md`.
   2. In `CHANGELOG.md`, rename `## [Unreleased]` to
      `## [X.Y.Z] - YYYY-MM-DD`, add an empty `## [Unreleased]` above it, and
      update the comparison links at the end. A release with **BREAKING**
      entries opens with a short *Upgrading from X.(Y-1)* subsection, grouped
      by audience, that points to those entries.
   3. In `CITATION.cff`, set `version:` to `X.Y.Z` and `date-released:` to the
      CHANGELOG date. `.zenodo.json` carries no version. Validate:
      ```bash
      pipx run cffconvert --validate -i CITATION.cff   # "valid according to schema 1.2.0"
      ```
   4. Check: `python .github/scripts/release_check.py --tag vX.Y.Z --publish`.

   The date is the day of the tag. If the pull request merges on an earlier
   day, update both dates on `main` before tagging.
3. **TestPyPI dry run.** Actions → Release → Run workflow on `main` with
   `target=testpypi`. It builds and smoke-tests every wheel and the sdist and
   uploads them to test.pypi.org; then, in a clean environment:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ polypus-quantum==X.Y.Z
   ```
   TestPyPI never accepts the same file twice, so a second dry run of the same
   version skips the files already there.
4. **crates.io credentials.** If the release adds a crate (`python
   .github/scripts/publish_crates.py --plan` prints `(new crate)`), a
   `CARGO_REGISTRY_TOKEN` secret with the `publish-new` and `publish-update`
   scopes must exist before the tag; see [crates.io](#cratesio).
5. **Publish the GitHub release.** Create a **GitHub Release** (not just a bare
   tag) named `vX.Y.Z` on the release commit. This:
   - triggers [`release.yml`](../.github/workflows/release.yml) → the gate,
     then builds the wheels + sdist, smoke-tests every wheel on its platform and
     the sdist (built from source in a clean container), publishes to PyPI
     (Trusted Publishing) and the Rust crates to crates.io (see
     [crates.io](#cratesio); new crates hit a 429 rate limit, resumed with
     "Re-run failed jobs"), then runs the clean-install verification;
   - fires the Zenodo webhook → Zenodo archives the release, reads `.zenodo.json`,
     and mints a version DOI under the existing concept DOI (below) —
     no manual step needed.
6. **Back to a development version.** Right after the tag, a pull request
   moves `main` to the next patch with `-dev`:
   ```bash
   cargo set-version --workspace X.Y.(Z+1)-dev
   for t in examples/*-backend-template; do cargo metadata --format-version 1 --manifest-path "$t/Cargo.toml" > /dev/null; done
   ```
   `CHANGELOG.md` keeps its empty `## [Unreleased]`; `CITATION.cff` and the
   README BibTeX keep the release.

## crates.io

The `publish-crates` job of `release.yml` publishes every workspace crate to
crates.io on a `vX.Y.Z` tag push (not on manual runs), after the same gates as
PyPI: the release gate, every wheel and sdist build and their smoke tests.

- **Which crates.** All fourteen workspace crates. `polypus-backend`,
  `polypus-subprocess-backend` and `polypus-backend-conformance` are needed by
  the others (`polypus-infrastructure` depends on the first two and
  dev-depends on the third, and crates.io requires dev-dependencies to exist),
  and the backend and conformance crates are public contracts for third-party
  backends. A crate that must stay internal gets `publish = false` and must not
  be a dependency of a published crate; the script refuses that combination.
- **Order.** Dependencies first, computed from `cargo metadata` by
  [`.github/scripts/publish_crates.py`](../.github/scripts/publish_crates.py)
  (dev-dependencies included, ties broken alphabetically). Today:
  `polypus-circuit`, `polypus-logger`, `polypus-observable`, `polypus-backend`,
  `polypus-backend-conformance`, `polypus-optimizers`, `polypus-physics`,
  `polypus-sim`, `polypus-qml`, `polypus-subprocess-backend`, `polypus-infrastructure`,
  `polypus-orchestration`, `polypus-evaluation`, `polypus`. Print the plan
  locally with `python .github/scripts/publish_crates.py --plan`.
- **Authentication.** crates.io Trusted Publishing (OIDC, no stored token)
  when no `CARGO_REGISTRY_TOKEN` secret exists; while one exists (repository
  secret, or a secret of the `crates-io` environment), the job uses it instead;
  it needs the `publish-new` and `publish-update` scopes and must belong to an
  owner of the `polypus*` crates.
  Trusted Publishing cannot create a crate: a release that adds a crate needs
  the token for that release, and without one the job stops at the plan step,
  before publishing anything (the plan marks such crates `(new crate)`).
- **Moving to Trusted Publishing** (one-time, repository and crates.io
  admin). The token stays until every crate exists on crates.io, so do this
  after the first release that publishes `polypus-backend`,
  `polypus-backend-conformance` and `polypus-subprocess-backend`:
  1. Check that every crate is on crates.io:
     `python .github/scripts/publish_crates.py --plan --trusted-publishing`
     must not stop with "not on crates.io yet".
  2. For each of the thirteen crates, on
     `https://crates.io/crates/<crate>/settings`, add a GitHub trusted
     publisher: owner `Bahia-Software`, repository `polypus`, workflow
     filename `release.yml`, environment `crates-io`. A crate left out makes
     the run stop at it with a 403, after its dependencies are uploaded:
     configure it and use "Re-run failed jobs".
  3. Optional, per crate on the same page: enable "Require trusted
     publishing", so crates.io refuses API tokens for that crate.
  4. Delete the `CARGO_REGISTRY_TOKEN` secret from the repository
     (Settings → Secrets and variables → Actions) and from the `crates-io`
     environment (Settings → Environments → `crates-io`), whichever holds it.
  5. Revoke the token on `https://crates.io/settings/tokens`.
  6. The next tag publishes through Trusted Publishing; its plan step logs
     "using crates.io Trusted Publishing".

  A later release that adds a crate needs a token again for that release:
  publish it with a short-lived token (scope `publish-new`, that crate
  only) stored as `CARGO_REGISTRY_TOKEN`, then repeat steps 2–5 for it.
- **Re-running.** The script skips every crate whose version is already in the
  crates.io index, so a failed or partial run is resumed with "Re-run failed
  jobs" on the tag's workflow run. crates.io rate-limits brand-new crates (a
  burst of a few, then one every ten minutes); `cargo publish` reports this as
  a 429, and a re-run later continues where it stopped.
- **Checking a change.** `cargo package --workspace --locked` builds and
  verifies every `.crate` against the others without uploading. It needs
  cargo 1.90 or newer to resolve not-yet-published workspace versions. CI's
  `cargo package` job runs it on every pull request and push to `main`, and
  `publish-crates` runs it again before uploading anything.

## Resolved (kept for context)

- **DOI** — the Zenodo concept DOI is `10.5281/zenodo.22913065` (constant
  across versions), wired into `CITATION.cff` (`identifiers`) and the README
  badge + BibTeX. Every release reuses this same concept DOI; there is nothing
  to substitute per release.
- **First version archived** — `0.7.0` was already on PyPI without a DOI, so
  the first Zenodo archive/DOI was cut from the `v0.7.1` GitHub Release.

_Author list, order, ORCIDs, affiliations and the license are **final** (6
authors, all affiliated to Bahía Software S.L.U., + CESGA as an entity),
identical in `CITATION.cff` and `.zenodo.json`._
