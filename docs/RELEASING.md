# Releasing Polypus (with a citable DOI)

Short, order-sensitive checklist for cutting a release that is published to PyPI
and archived on Zenodo with a DOI.

> **Status:** the GitHub → Zenodo integration is **already enabled** for
> `Bahia-Software/polypus`, and the citation metadata is prepared in
> [`CITATION.cff`](../CITATION.cff) and [`.zenodo.json`](../.zenodo.json).
> What remains for the first DOI is small — see [What's left](#whats-left).

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

## What's left

1. **Release date — set.** `CITATION.cff` has `date-released: 2026-09-23`. If the
   release actually happens on another day, update it. (If releasing a version
   other than `0.7.1`, also bump `version:` here and in `Cargo.toml` /
   `Cargo.lock`.) Validate:
   ```bash
   pipx run cffconvert --validate -i CITATION.cff   # "valid according to schema 1.2.0"
   ```
2. **Publish the GitHub release.** Create a **GitHub Release** (not just a bare
   tag) named `vX.Y.Z`. This:
   - triggers [`release.yml`](../.github/workflows/release.yml) → builds the
     wheels + sdist, publishes to PyPI (Trusted Publishing), then runs the
     clean-install verification;
   - fires the Zenodo webhook → Zenodo archives the release, reads `.zenodo.json`,
     and **mints the DOI**.
3. **Substitute the concept DOI.** In Zenodo you get two DOIs: a **concept DOI**
   (constant, always resolves to the latest version) and a **version DOI** (this
   `vX.Y.Z`). Use the **concept DOI**. Then:
   - in `CITATION.cff`, uncomment the `identifiers:` block and set the concept
     DOI (`10.5281/zenodo.XXXXXXX`);
   - in the **README** (Credits + BibTeX `doi` + the DOI badge), fill the concept
     DOI. Per team decision this README update is a **separate PR**, done only
     once the DOI exists.
   These commits land *after* the archived snapshot, which is fine — the DOI
   resolves to the archived release; the repo just starts displaying it.

## Pending decisions

- **DOI** — **set**: the Zenodo concept DOI is `10.5281/zenodo.22913065`
  (constant across versions), wired into `CITATION.cff` (`identifiers`) and the
  README badge + BibTeX. Future releases reuse the same concept DOI.
- **Version to archive** — **decided: `0.7.1`.** `0.7.0` is already on PyPI, so
  the first DOI is cut from a fresh `0.7.1` GitHub Release (the workspace version,
  `Cargo.lock`, `CITATION.cff` and the README are already bumped to `0.7.1`).

_Author list, order, ORCIDs, affiliations and the license are **final** (6
authors, all affiliated to Bahía Software S.L.U., + CESGA as an entity),
identical in `CITATION.cff` and `.zenodo.json`._
