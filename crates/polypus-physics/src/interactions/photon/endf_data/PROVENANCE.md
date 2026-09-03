# Provenance and attribution — embedded photon data (`ZA*.txt`)

The `ZA*.txt` files in this directory are **tabulated photon interaction
cross-sections** in the **ENDF-6** format, taken from the **Evaluated Photon
Data Library (EPDL)**, as distributed in **EPICS2023 / EPDL2023**.

They are third-party scientific data files. They are **not** original source
code of this project and are **not** covered by this repository's software
license (see "Licensing" below).

## Source

- **Library:** Evaluated Photon Data Library (EPDL), EPICS2023 release
  (previously EPDL97, in the ENDF-6 format).
- **Distributed by:** Nuclear Data Section (NDS), International Atomic Energy
  Agency (IAEA), Vienna, Austria — release **NDS-IAEA-225**.
- **Evaluation / authors:** D.E. Cullen et al., Lawrence Livermore National
  Laboratory (LLNL) and IAEA-NDS.
- **Official download:** IAEA Nuclear Data Services — <https://nds.iaea.org/>
- Each file also carries its own ENDF-6 header identifying the material,
  MF/MT reaction channels, evaluation date, and references.

## File naming

Each file is named `ZAzzz000.txt`, where `zzz` is the atomic number Z of the
element (e.g. `ZA001000.txt` = hydrogen, Z=1; `ZA008000.txt` = oxygen, Z=8).
The directory contains evaluations for Z = 1 to 100.

## References

1. D.E. Cullen, M.H. Chen, J.H. Hubbell, S.T. Perkins, E.F. Plechaty,
   J.A. Rathkopf and J.H. Scofield, *Tables and graphs of photon interaction
   cross sections from 10 eV to 100 GeV derived from the LLNL Evaluated Photon
   Data Library (EPDL)*, UCRL-50400, Vol. 6, Rev. 4, Part A (Z = 1–50) and
   Part B (Z = 51–100), Lawrence Livermore National Laboratory (1989).
2. D.E. Cullen, J.H. Hubbell and L.D. Kissel, *EPDL97: the Evaluated Photon
   Data Library, '97 Version*, UCRL-50400, Vol. 6, Rev. 5, LLNL (1997).
3. D.E. Cullen, *EPICS2023: August 2023 Status Report*, IAEA-NDS-242,
   August 2023, Nuclear Data Section, IAEA, Vienna, Austria.

## Licensing / usage terms

The IAEA Nuclear Data Section distributes the ENDF/EPDL/EPICS evaluated data
libraries for **free and open use**, including copying and redistribution,
provided the source is acknowledged. These files consist of factual scientific
measurement data and are redistributed here unmodified.

This attribution file, together with the per-file ENDF-6 headers, is intended
to preserve that acknowledgement. The data files retain their original IAEA/NDS
provenance and are **not** relicensed under this repository's license.
