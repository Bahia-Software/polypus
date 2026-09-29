# Releasing Polypus (with a citable DOI)

Short, order-sensitive checklist for cutting a release that is published to PyPI
and archived on Zenodo with a DOI.

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

## Release steps

1. **Bump the version and release date.** Set `version:` and `date-released:`
   in `CITATION.cff` to match the workspace `Cargo.toml` version and the day
   the release actually happens (also update `Cargo.toml` / `Cargo.lock` if the
   version itself is changing). Validate:
   ```bash
   pipx run cffconvert --validate -i CITATION.cff   # "valid according to schema 1.2.0"
   ```
2. **Publish the GitHub release.** Create a **GitHub Release** (not just a bare
   tag) named `vX.Y.Z`. This:
   - triggers [`release.yml`](../.github/workflows/release.yml) → builds the
     wheels + sdist, publishes to PyPI (Trusted Publishing), then runs the
     clean-install verification;
   - fires the Zenodo webhook → Zenodo archives the release, reads `.zenodo.json`,
     and mints a version DOI under the existing concept DOI (below) —
     no manual step needed.

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
