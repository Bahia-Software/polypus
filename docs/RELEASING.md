# Releasing Polypus (with a citable DOI)

This is the short, order-sensitive checklist for cutting a release that is both
published to PyPI and archived on Zenodo with a DOI. Read the ⚠️ ordering rule
first — getting it wrong means the release does not get a DOI.

> **Status:** DOI/Zenodo is prepared but **not yet activated**. Several fields
> are still placeholders — see [Pending decisions](#pending-decisions) before
> the first real release.

## ⚠️ The one ordering rule that matters

**Enable the GitHub → Zenodo integration BEFORE you publish the GitHub release.**
Zenodo only archives releases created *after* the repository is switched on in
Zenodo; it does **not** archive past releases retroactively. If you publish
first and enable Zenodo afterwards, that release gets **no DOI** and you must cut
a new one.

## Release procedure

1. **Finalize the citation metadata.** Resolve every placeholder in
   [`CITATION.cff`](../CITATION.cff) (author list/order, ORCID, affiliation) and
   the BibTeX block + DOI badge in [`README.md`](../README.md). See
   [Pending decisions](#pending-decisions). Validate:
   ```bash
   pipx run cffconvert --validate -i CITATION.cff   # "valid according to schema 1.2.0"
   ```
2. **Bump the version.** Update `[workspace.package] version` **and** all
   `[workspace.dependencies]` version pins in `Cargo.toml` (+ `Cargo.lock`), and
   the `version:` field in `CITATION.cff`. Commit and merge to `main` via PR.
3. **Enable Zenodo — do this now, before step 5.** Log in to
   <https://zenodo.org> with the org GitHub account, open *Account → GitHub*, and
   flip the **`Bahia-Software/polypus`** switch **On**. (One-time; stays on for
   future releases.)
4. **Draft the Zenodo metadata** you will confirm after the release — see
   [Zenodo metadata draft](#zenodo-metadata-draft). Much of it is pre-filled from
   `CITATION.cff`, but the funding field is not and will be requested when
   justifying grants.
5. **Publish the GitHub release.** Create a **GitHub Release** (not just a bare
   tag) named `vX.Y.Z`. This:
   - triggers [`release.yml`](../.github/workflows/release.yml) → builds wheels +
     sdist and publishes to PyPI (Trusted Publishing), then the post-publish
     clean-install verification;
   - fires the Zenodo webhook → Zenodo archives the release and **mints the DOI**.
6. **Collect the DOI.** In Zenodo, open the new record. You get two DOIs:
   - a **concept DOI** (constant, always resolves to the latest version) — use
     this in the README badge and `CITATION.cff`;
   - a **version DOI** (specific to `vX.Y.Z`).
   Confirm/complete the record's metadata against the draft below and publish it.
7. **Backfill the DOI into the repo.** Replace the `PENDING` markers with the
   concept DOI in `CITATION.cff` (`doi:`), the README DOI badge, and the BibTeX
   `doi` field. Commit to `main`. (This commit is *after* the archived snapshot,
   which is fine — the DOI resolves to the archived release, and the badge/CFF
   simply start displaying it.)

## Zenodo metadata draft

Copy-paste starting point for the Zenodo record. Creators and funding are
**placeholders** — do not invent them.

- **Resource type:** Software.
- **Title:** Polypus: A Distributed Quantum Computing Library.
- **Version:** `X.Y.Z` (the released version; `0.7.0` today).
- **License:** European Union Public Licence 1.2 (`EUPL-1.2`).
- **Creators:** `<PENDING — must match the agreed CITATION.cff author list,
  order, ORCID and affiliation. Do not invent.>`
- **Keywords:** quantum, quantum-computing, vqc, qml, qaoa, qiskit, simulator,
  optimization.
- **Description:**
  > Polypus is an open-source distributed quantum computing library. It runs
  > quantum circuits and trains variational quantum algorithms (VQE, QAOA, QML)
  > across one or many QPUs — simulated or real — without changing the circuit
  > code. The core is written in Rust for performance and correctness; Python
  > bindings (via PyO3) make it a drop-in accelerator for existing Qiskit
  > workflows. It targets HPC execution: local Aer, a native Rust statevector
  > simulator, CESGA's CUNQA distributed QPU platform, and CESGA's QMIO real QPU.
- **Related identifiers:** repository <https://github.com/Bahia-Software/polypus>;
  PyPI distribution `polypus-quantum`.
- **Funding / Grants:** `<PENDING>` — this is the field grant justifications will
  ask for later. For each award, provide:
  - Funder: `<FUNDER — PENDING, e.g. selected from Zenodo's funder list>`
  - Grant/Award number: `<GRANT NUMBER — PENDING>`
  - Project name: `<PROJECT NAME — PENDING>`
  Do not invent funders or grant numbers; leave the markers until confirmed.

## Pending decisions

Everything below must be decided by the team before the first release; each is a
marker in the repo today (nothing here is invented):

- **Author list & order** for the citation — who is listed, in what order, and
  whether **CESGA** appears as an entity author. Starting reference: `Cargo.toml`
  `authors` and the README "Credits"/BibTeX, but the citation authorship is an
  explicit, separate agreement.
- **ORCID** for each author — none are recorded in the repo.
- **Affiliation** for each author — none are recorded in the repo.
- **DOI** — assigned by Zenodo on the first archived release (both the concept
  and version DOI); until then it is `PENDING` in `CITATION.cff`, the README
  badge and the BibTeX block.
- **`date-released`** in `CITATION.cff` — the actual release date.
- **Funding metadata** for Zenodo — funder, grant/award number, project name.
- **Version to archive** — `0.7.0` is already on PyPI; decide whether the first
  DOI is cut for `0.7.0` (needs a GitHub Release created *after* Zenodo is
  enabled) or a subsequent version.
