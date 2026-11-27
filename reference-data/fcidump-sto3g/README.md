# FCIDUMP / STO-3G reference data

External reference for the FCIDUMP reader and the sector diagonalization of
`polypus-physics` (`chemistry::Fcidump`,
`FermionHamiltonian::sector_ground_energy`), generated outside Polypus with
PySCF (RHF, FCIDUMP writer, FCI). The tests in
`crates/polypus-physics/tests/fcidump_reference.rs` compare against these
files; nothing in the crate reads them at runtime.

## Files

| File | Content |
|---|---|
| `<molecule>_<R>.fcidump` | The integrals of `<molecule>` at bond length `<R>` Å, as `pyscf.tools.fcidump.from_scf` writes them. |
| `fci_energies.csv` | `molecule, geometry_angstrom, n_up, n_down, hf_energy, fci_energy`. |
| `generate.py` | The script that writes all of the above. |
| `requirements.txt` | The exact package versions used (Python 3.13). |

Molecules and bond lengths:

- H₂: 0.3, 0.5, 0.7414, 1.0, 1.5, 2.0 and 2.5 Å, as in `../h2-sto3g/`;
- LiH: 1.595 Å (near equilibrium) and 3.0 Å (stretched);
- linear BeH₂ (H–Be–H, both Be–H bonds of length R): 1.326 Å (near
  equilibrium) and 2.6 Å (stretched).

## Conventions

- Basis `sto-3g`; neutral singlets; **all electrons** (no frozen core), so
  the sectors are LiH `(2, 2)` in 6 orbitals (dimension 225) and BeH₂
  `(3, 3)` in 7 orbitals (dimension 1225).
- Energies in hartree; bond lengths in ångström. The coordinates are
  converted to bohr with polypus-physics's Bohr radius
  (`BOHR_RADIUS_ANGSTROM`, CODATA 2018) and passed to PySCF in bohr: PySCF's
  own constant differs in the last digits, enough to shift the nuclear
  repulsion by about 1e-11.
- Orbitals: RHF, sorted by energy, each with the sign that makes its first
  largest-magnitude AO coefficient positive. For H₂ this gives
  `σ = φ_A + φ_B` and `σ* = φ_A − φ_B`, the orbitals of `chemistry::h2`.
- FCIDUMP: PySCF's default 4-fold listing, so both `(ij|kl)` and `(kl|ij)`
  appear and may differ in the last digit; integrals with `|v| ≤ 1e-15` are
  omitted; the core energy `0 0 0 0` is the nuclear repulsion. `ORBSYM` is
  written as all ones: fixing the orbital signs copies `mo_coeff`, which
  drops PySCF's symmetry labels (the reader ignores `ORBSYM` anyway).
- `fci_energy` is the lowest eigenvalue of the `(n_up, n_down)` sector,
  from the integrals read back from the written file: a dense
  diagonalization of the whole sector (`pyscf.fci.direct_spin1.pspace`),
  checked against `pyscf.fci.direct_spin1`'s Davidson solver, which has no
  point-group restriction but can converge to an excited root;
  `hf_energy` is PySCF's RHF energy.

## Regenerating

```sh
python -m venv env
env/bin/pip install -r requirements.txt
env/bin/python generate.py
```

The energies are reproducible to within 1e-10. The LiH and BeH₂ files
themselves are not guaranteed to be byte-identical across machines: their
virtual π orbitals are degenerate, so BLAS-backed steps can return a
different orthonormal pair, which changes some integrals but not the
energies. The H₂ files are reproducible to within rounding.
