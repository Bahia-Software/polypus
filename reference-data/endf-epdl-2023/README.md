# ENDF-6 (EPDL2023) photon data

The 100 raw ENDF-6/EPDL2023 evaluations (`ZA{Z:03}000.txt`, Z = 1–100) that
`polypus-physics` reads at runtime, unmodified from:

https://www-nds.iaea.org/epics/ENDF2023/EPDL.ELEMENTS/ZA{Z:03d}000

`polypus-physics` never downloads data and does not embed or package these
files. It reads them from `POLYPUS_ENDF_DATA_DIR` if set, otherwise from this
folder (located relative to the crate's source in this repository). Outside
a checkout, set `POLYPUS_ENDF_DATA_DIR` to a copy of this folder. See
`crates/polypus-physics/src/interactions/photon/mass_attenuation_coefficients.rs`.

Source, references and licensing: [PROVENANCE.md](PROVENANCE.md).
