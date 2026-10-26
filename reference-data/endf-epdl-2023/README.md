# ENDF-6 (EPDL2023) photon data

The 100 raw ENDF-6/EPDL2023 evaluations (`ZA{Z:03}000.txt`, Z = 1–100) that
`polypus-physics` reads at runtime, unmodified from:

https://www-nds.iaea.org/epics/ENDF2023/EPDL.ELEMENTS/ZA{Z:03}000

`polypus-physics` never downloads data and does not embed or package these
files. It reads them from `POLYPUS_ENDF_DATA_DIR` if set, otherwise from this
folder (located relative to the crate's source in this repository). Outside
a checkout, e.g. when using the crate from crates.io, fetch this folder alone
and point `POLYPUS_ENDF_DATA_DIR` at it:

```sh
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/Bahia-Software/polypus.git polypus-data
git -C polypus-data sparse-checkout set reference-data/endf-epdl-2023
export POLYPUS_ENDF_DATA_DIR=$PWD/polypus-data/reference-data/endf-epdl-2023
```

Saving only the elements you need from the IAEA URL above with a browser (the
site may refuse non-browser clients), as `ZA{Z:03}000.txt`, works too. See the "ENDF-6 photon data" section of the
`polypus-physics` crate docs.

Source, references and licensing: [PROVENANCE.md](PROVENANCE.md).
