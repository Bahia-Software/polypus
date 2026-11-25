"""Generate the external FCIDUMP/STO-3G reference data used by polypus-physics tests.

Runs PySCF (RHF, FCIDUMP writer, FCI) for H2, LiH and BeH2 and writes, next
to this script:

- one ``<molecule>_<geometry>.fcidump`` per molecule and geometry, as
  ``pyscf.tools.fcidump.from_scf`` writes it (4-fold symmetric listing);
- ``fci_energies.csv``: the RHF energy and the FCI energy of the
  ``(n_up, n_down)`` sector (hartree), the FCI computed from the integrals
  read back from the written file.

Conventions (see README.md): STO-3G, all electrons, RHF orbitals sorted by
energy, coordinates converted to bohr with polypus-physics's Bohr radius
(CODATA 2018) so that both sides place the nuclei identically, and each
orbital's sign fixed so that its first largest-magnitude AO coefficient is
positive.

Usage (pinned versions in ``requirements.txt``):

    python -m venv env
    env/bin/pip install -r requirements.txt
    env/bin/python generate.py
"""

import csv
import pathlib

import numpy as np
from pyscf import ao2mo, fci, gto, scf
from pyscf.tools import fcidump

# `BOHR_RADIUS_ANGSTROM` in crates/polypus-physics/src/constants.rs. PySCF's
# own constant differs in the last digits, enough to shift the nuclear
# repulsion by ~1e-11; passing bohr avoids it.
BOHR_RADIUS_ANGSTROM = 0.529177210903
# Integrals at or below this magnitude are not written (PySCF's default).
TOL = 1e-15
CONV_TOL = 1e-12

HERE = pathlib.Path(__file__).resolve().parent


def h2(r):
    return [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, r))]


def lih(r):
    return [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, r))]


def beh2(r):
    """Linear H-Be-H with both Be-H bonds of length r."""
    return [("Be", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, -r)), ("H", (0.0, 0.0, r))]


# (file stem, builder, geometry in angstrom, as written in the file name).
SYSTEMS = (
    [("h2", h2, r) for r in ["0.3", "0.5", "0.7414", "1.0", "1.5", "2.0", "2.5"]]
    + [("lih", lih, r) for r in ["1.595", "3.0"]]
    + [("beh2", beh2, r) for r in ["1.326", "2.6"]]
)


def fix_signs(mo_coeff):
    """Make each orbital's first largest-|c| AO coefficient positive."""
    fixed = mo_coeff.copy()
    for j in range(fixed.shape[1]):
        column = fixed[:, j]
        largest = np.max(np.abs(column))
        first = np.flatnonzero(np.abs(column) >= largest * (1 - 1e-8))[0]
        if column[first] < 0:
            fixed[:, j] = -column
    return fixed


def sector_fci(path, n_up, n_down):
    """The lowest FCI energy of the (n_up, n_down) sector, from the file.

    Davidson (``direct_spin1``) can converge to an excited root, so the
    whole sector is also diagonalized densely and the two must agree.
    """
    data = fcidump.read(str(path))
    norb = data["NORB"]
    h1e, ecore = data["H1"], data["ECORE"]
    h2e = ao2mo.restore(1, data["H2"], norb)
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = CONV_TOL
    davidson, _ = solver.kernel(h1e, h2e, norb, (n_up, n_down), ecore=ecore)
    dimension = fci.cistring.num_strings(norb, n_up) * fci.cistring.num_strings(
        norb, n_down
    )
    _, block = fci.direct_spin1.pspace(h1e, h2e, norb, (n_up, n_down), np=dimension)
    dense = np.linalg.eigvalsh(block)[0] + ecore
    if abs(davidson - dense) > 1e-10:
        raise RuntimeError(f"{path.name}: Davidson {davidson} vs dense {dense}")
    return dense


def main():
    rows = []
    for stem, build, r in SYSTEMS:
        atoms = [
            (symbol, tuple(x / BOHR_RADIUS_ANGSTROM for x in xyz))
            for symbol, xyz in build(float(r))
        ]
        mol = gto.M(atom=atoms, unit="Bohr", basis="sto-3g", symmetry=True, verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = CONV_TOL
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"{stem} {r}: RHF did not converge")
        mf.mo_coeff = fix_signs(mf.mo_coeff)

        path = HERE / f"{stem}_{r}.fcidump"
        fcidump.from_scf(mf, str(path), tol=TOL)
        n_up, n_down = mol.nelec
        energy = sector_fci(path, n_up, n_down)
        rows.append((stem, r, n_up, n_down, repr(float(mf.e_tot)), repr(float(energy))))

    with open(HERE / "fci_energies.csv", "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "molecule",
                "geometry_angstrom",
                "n_up",
                "n_down",
                "hf_energy",
                "fci_energy",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":
    main()
