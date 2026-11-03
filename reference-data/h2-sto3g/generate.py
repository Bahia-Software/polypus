"""Generate the external H2/STO-3G reference data used by polypus-physics tests.

Runs PySCF (integrals, Hartree-Fock, FCI) and OpenFermion (Jordan-Wigner) for
H2 at a fixed list of bond lengths and writes, next to this script:

- ``jw_coefficients.csv``: Jordan-Wigner Pauli coefficients (hartree) per bond
  length, with spin-orbitals in block order (all spin-up, then all spin-down)
  and Pauli strings in the polypus text form (``X0 X1 Y2 Y3``, ``I``).
- ``fci_energies.csv``: Hartree-Fock and FCI energies (hartree).

Usage (pinned versions in ``requirements.txt``):

    python -m venv env
    env/bin/pip install -r requirements.txt
    env/bin/python generate.py
"""

import csv
import pathlib
import tempfile

from openfermion import (
    MolecularData,
    get_fermion_operator,
    jordan_wigner,
    reorder,
    up_then_down,
)
from openfermionpyscf import run_pyscf

BOND_LENGTHS_ANGSTROM = ["0.3", "0.5", "0.7414", "1.0", "1.5", "2.0", "2.5"]
# Terms with |c| <= this are rounding noise; physical H2 terms are above 1e-4.
# Same value and rule as H2_ROUNDING_TOLERANCE in chemistry/h2.rs, so both
# sides keep exactly the same Pauli strings.
DROP_AT_OR_BELOW = 1e-14
# A Hermitian operator has real Pauli coefficients up to rounding.
MAX_IMAGINARY = 1e-12

HERE = pathlib.Path(__file__).resolve().parent


def pauli_text(term):
    """OpenFermion term ((qubit, 'X'), ...) -> polypus text form."""
    return " ".join(f"{axis}{qubit}" for qubit, axis in sorted(term)) or "I"


def main():
    jw_rows = []
    energy_rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for r in BOND_LENGTHS_ANGSTROM:
            molecule = MolecularData(
                [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, float(r)))],
                basis="sto-3g",
                multiplicity=1,
                charge=0,
                data_directory=scratch,
            )
            molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
            energy_rows.append(
                (r, repr(float(molecule.hf_energy)), repr(float(molecule.fci_energy)))
            )

            fermions = get_fermion_operator(molecule.get_molecular_hamiltonian())
            qubits = jordan_wigner(reorder(fermions, up_then_down))
            for term, c in sorted(qubits.terms.items(), key=lambda t: pauli_text(t[0])):
                if abs(c.imag) > MAX_IMAGINARY:
                    raise ValueError(
                        f"R={r}: {pauli_text(term)} has imaginary part {c.imag}"
                    )
                if abs(c.real) > DROP_AT_OR_BELOW:
                    jw_rows.append((r, pauli_text(term), repr(float(c.real))))

    with open(HERE / "jw_coefficients.csv", "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["R_angstrom", "pauli_string", "coefficient"])
        writer.writerows(jw_rows)
    with open(HERE / "fci_energies.csv", "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["R_angstrom", "hf_energy", "fci_energy"])
        writer.writerows(energy_rows)


if __name__ == "__main__":
    main()
