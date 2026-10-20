# ENDF-6 (EPDL) reference backup

Manual, read-only backup of the 100 raw ENDF-6/EPDL evaluations,
originally fetched from:

https://www-nds.iaea.org/epics/ENDF2023/EPDL.ELEMENTS/ZA{Z:03d}000

**Not used by any build or runtime code.** `polypus-physics` fetches
this data live and caches it on disk (see
`mass_attenuation_coefficients.rs`). This folder exists purely as a
fallback reference in case the IAEA service URL or format ever
changes — kept outside `crates/polypus-physics/` so it's never part
of the published crate package.
