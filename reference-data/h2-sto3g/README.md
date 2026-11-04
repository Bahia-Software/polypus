# H2 / STO-3G reference data

External reference for the H2 Hamiltonian of `polypus-physics`
(`chemistry::H2Molecule`), generated outside Polypus with PySCF (integrals,
Hartree–Fock, FCI) and OpenFermion (Jordan–Wigner). The tests in
`crates/polypus-physics/src/chemistry/h2.rs` compare against these files;
nothing in the crate reads them at runtime.

## Files

| File | Content |
|---|---|
| `jw_coefficients.csv` | `R_angstrom, pauli_string, coefficient`: the Jordan–Wigner Pauli sum at each bond length (15 terms each). |
| `fci_energies.csv` | `R_angstrom, hf_energy, fci_energy` at each bond length. |
| `generate.py` | The script that writes both CSV files. |
| `requirements.txt` | The exact package versions used (Python 3.13). |

Bond lengths: 0.3, 0.5, 0.7414, 1.0, 1.5, 2.0 and 2.5 Å.

## Conventions

- Geometry `H (0, 0, 0)`, `H (0, 0, R)`; basis `sto-3g`; neutral singlet.
- Energies and coefficients in hartree; bond lengths in ångström.
- Spin-orbitals in **block order** (all spin-up, then all spin-down; OpenFermion
  `reorder(…, up_then_down)`), the order used by `polypus-physics`. Qubit `p`
  is spin-orbital `p`.
- Pauli strings in the `polypus-physics` text form: `X0 X1 Y2 Y3`, `I` for the
  identity. The identity coefficient includes the nuclear repulsion.
- Terms with `|c| ≤ 1e-14` are dropped as rounding noise, the same rule as
  `H2_ROUNDING_TOLERANCE` in `chemistry/h2.rs`; every physical H2 term is
  above `1e-4`.

## Regenerating

```sh
python -m venv env
env/bin/pip install -r requirements.txt
env/bin/python generate.py
```

The output is deterministic up to rounding: re-running with the pinned
versions gives the same Pauli strings and every value to within 1e-12.
Only the direct dependencies are pinned, and BLAS-backed steps can differ
in the last digits across CPUs and library builds.
