//! Reading molecular integrals from an FCIDUMP file (issue #535).
//!
//! FCIDUMP is the plain-text integral format written by PySCF, Molpro, Psi4
//! and most quantum-chemistry codes. It brings molecules the built-in STO-3G
//! engine of [`super::sto3g`] cannot build (anything beyond H and He, other
//! basis sets, frozen cores) to [`FermionHamiltonian`].
//!
//! # Format
//!
//! A Fortran namelist header, then one integral per line:
//!
//! ```text
//!  &FCI NORB=   2,NELEC= 2,MS2=0,
//!   ORBSYM=1,1,
//!   ISYM=1,
//!  &END
//!  0.6746 1 1 1 1        (pq|rs), all four indices > 0, chemists' notation
//!  -1.2528 1 1 0 0       h_pq: the last two indices are 0
//!  0.7137 0 0 0 0        the core energy
//! ```
//!
//! - **Header.** `&FCI … &END` (or `/` instead of `&END`), keys in any case,
//!   values separated by commas or blanks, over any number of lines; the
//!   end marker must be separated from the last value by a comma or a blank
//!   (`ISYM=1/` and `MS2=0&END` are rejected, which no known writer
//!   produces). `NORB` (spatial orbitals, at most [`MAX_FCIDUMP_ORBITALS`])
//!   and `NELEC` are required; `MS2` (`N↑ − N↓`) defaults to 0. `ORBSYM`,
//!   `ISYM` and any other key (`OCC`, `CLOSED`, `TREL=.FALSE.`,
//!   `UHF=.FALSE.`, …) are accepted and ignored, except those that change
//!   the meaning of the integrals:
//!   `IUHF` other than 0, `UHF=.TRUE.` (unrestricted orbitals) and
//!   `TREL=.TRUE.` (complex integrals) are rejected. `ISYM` is not applied:
//!   [`FermionHamiltonian::sector_ground_energy`] gives the lowest state of
//!   the sector in **any** spatial symmetry, which is lower than or equal to
//!   the energy of the state of symmetry `ISYM` the file was written for, for
//!   any `ISYM`, 1 included: the lowest state of a sector need not be totally
//!   symmetric (O₂'s ³Σg⁻ ground state is `B1g` in `D2h`).
//! - **Integrals.** `value i j k l`, indices 1-based. All four `> 0`: the
//!   two-electron integral `(ij|kl)`; `k = l = 0`: the one-electron `h_ij`;
//!   all `0`: the core energy (nuclear repulsion, plus any frozen core),
//!   which becomes [`FermionHamiltonian::nuclear_repulsion`]; `j = k = l = 0`:
//!   an orbital energy, which is not needed and is ignored. A value may use a
//!   Fortran exponent (`1.0D-02`); a three-digit exponent written without its
//!   letter (`1.0-100`, as Fortran's `E` and `ES` editing write values below
//!   `1e-99`) is rejected as not a number, since writers omit integrals that
//!   small. Blank lines are skipped.
//! - **Symmetry.** A file lists one integral of each group of equal ones (8
//!   permutations of `(ij|kl)`, 2 of `h_ij`) or more of them; PySCF's default
//!   lists both `(ij|kl)` and `(kl|ij)`, which may differ in the last digit.
//!   Every listed value is kept, a missing member of a group is filled from a
//!   listed one, and [`FermionHamiltonian::new`] checks the group within
//!   [`INTEGRAL_SYMMETRY_TOLERANCE`]
//!   and averages it. An integral not listed in any member is 0. The same
//!   index list given twice with different values is an error.
//!
//! The electron counts give the sector [`Fcidump::sector`],
//! `(N↑, N↓) = ((NELEC + MS2)/2, (NELEC − MS2)/2)`, for
//! [`FermionHamiltonian::sector_ground_energy`].

use std::path::Path;

use crate::error::PhysicsError;
use crate::hamiltonians::fermion::chemist_permutations;
use crate::hamiltonians::{FermionHamiltonian, INTEGRAL_SYMMETRY_TOLERANCE};

/// The most spatial orbitals an FCIDUMP may declare. While reading, the
/// `K⁴` two-electron integrals are held as listed and as filled in (about
/// 24 MB at 32 orbitals, before [`FermionHamiltonian::new`] makes its own
/// copies), and the sector solver needs the `2K` spin-orbitals in a 64-bit
/// determinant, so a larger `NORB` is rejected before anything is
/// allocated.
pub const MAX_FCIDUMP_ORBITALS: usize = 32;

/// The integrals and electron counts of an FCIDUMP file (see the
/// [module docs](self)).
#[derive(Debug, Clone, PartialEq)]
pub struct Fcidump {
    hamiltonian: FermionHamiltonian,
    num_electrons: usize,
    ms2: i64,
}

impl Fcidump {
    /// Reads an FCIDUMP from its text.
    ///
    /// # Errors
    /// [`PhysicsError::InvalidFcidump`] for a malformed or unsupported file,
    /// with the line when the problem is on one, including two listed
    /// members of one symmetry group that differ by more than
    /// [`INTEGRAL_SYMMETRY_TOLERANCE`].
    pub fn parse(text: &str) -> Result<Self, PhysicsError> {
        // A UTF-8 byte-order mark is not part of the header.
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let (header, body_start) = read_header(text)?;
        let k = header.norb;
        // Each listed value with the line it is on.
        let mut one: Vec<Option<(f64, usize)>> = vec![None; k * k];
        let mut two: Vec<Option<(f64, usize)>> = vec![None; k * k * k * k];
        let mut core: Option<(f64, usize)> = None;
        let mut listed = 0usize;
        for (number, line) in text.lines().enumerate().skip(body_start) {
            let number = number + 1;
            if line.trim().is_empty() {
                continue;
            }
            listed += 1;
            let (value, [i, j, kk, l]) = parse_integral(line, number, k)?;
            let slot = match (i, j, kk, l) {
                (0, 0, 0, 0) => &mut core,
                (_, 0, 0, 0) => continue, // an orbital energy
                (i, j, 0, 0) if i > 0 && j > 0 => &mut one[(i - 1) * k + (j - 1)],
                (i, j, kk, l) if i > 0 && j > 0 && kk > 0 && l > 0 => {
                    &mut two[(((i - 1) * k + (j - 1)) * k + (kk - 1)) * k + (l - 1)]
                }
                _ => {
                    return Err(invalid(
                        Some(number),
                        format!(
                        "indices {i} {j} {kk} {l} are not an integral, a core or an orbital energy"
                    ),
                    ))
                }
            };
            match *slot {
                Some((previous, _)) if previous != value => {
                    return Err(invalid(
                        Some(number),
                        format!(
                            "integral {i} {j} {kk} {l} is given twice, as {previous} and {value}"
                        ),
                    ))
                }
                _ => *slot = Some((value, number)),
            }
        }
        if listed == 0 {
            return Err(invalid(
                None,
                "the file has no integral after its header".to_string(),
            ));
        }
        check_listed_symmetry(&one, &two, k)?;

        let one_body: Vec<f64> = (0..k * k)
            .map(|index| {
                let (p, q) = (index / k, index % k);
                one[index].or(one[q * k + p]).map_or(0.0, |(v, _)| v)
            })
            .collect();
        let idx = |p: usize, q: usize, r: usize, s: usize| ((p * k + q) * k + r) * k + s;
        let mut two_body = vec![0.0; k * k * k * k];
        for p in 0..k {
            for q in 0..k {
                for r in 0..k {
                    for s in 0..k {
                        two_body[idx(p, q, r, s)] = two[idx(p, q, r, s)]
                            .map(|(v, _)| v)
                            .or_else(|| {
                                chemist_permutations(p, q, r, s)
                                    .iter()
                                    .find_map(|&(a, b, c, d)| two[idx(a, b, c, d)].map(|(v, _)| v))
                            })
                            .unwrap_or(0.0);
                    }
                }
            }
        }
        let core = core.map_or(0.0, |(v, _)| v);
        let hamiltonian = FermionHamiltonian::new(k, core, one_body, two_body)?;
        Ok(Self {
            hamiltonian,
            num_electrons: header.nelec,
            ms2: header.ms2,
        })
    }

    /// Reads an FCIDUMP file.
    ///
    /// # Errors
    /// [`PhysicsError::IoError`] if the file cannot be read; otherwise as
    /// [`Fcidump::parse`].
    pub fn read(path: impl AsRef<Path>) -> Result<Self, PhysicsError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| PhysicsError::IoError {
            message: format!("cannot read FCIDUMP {}: {e}", path.display()),
        })?;
        Self::parse(&text)
    }

    /// The electronic Hamiltonian of the file's integrals.
    pub fn hamiltonian(&self) -> &FermionHamiltonian {
        &self.hamiltonian
    }

    /// The number of spatial orbitals, `NORB`.
    pub fn num_orbitals(&self) -> usize {
        self.hamiltonian.num_spatial_orbitals()
    }

    /// The number of electrons, `NELEC`.
    pub fn num_electrons(&self) -> usize {
        self.num_electrons
    }

    /// `MS2 = N↑ − N↓`, twice the spin projection.
    pub fn ms2(&self) -> i64 {
        self.ms2
    }

    /// The electron sector `(N↑, N↓)` of the file's `NELEC` and `MS2`. `MS2`
    /// is twice the spin **projection** `M_S`, not the multiplicity: the
    /// sector's lowest state has any total spin `S ≥ |M_S|` and any spatial
    /// symmetry (`ISYM` is not applied; see the [module docs](self)).
    pub fn sector(&self) -> (usize, usize) {
        let n = self.num_electrons as i64;
        // `parse` checked that both are whole and between 0 and NORB.
        (((n + self.ms2) / 2) as usize, ((n - self.ms2) / 2) as usize)
    }
}

/// Two listed members of one symmetry group (`h_ij`/`h_ji`, or the 8
/// permutations of `(ij|kl)`) must agree within
/// [`INTEGRAL_SYMMETRY_TOLERANCE`]; the error names them as the file does
/// (1-based) and on the second's line.
fn check_listed_symmetry(
    one: &[Option<(f64, usize)>],
    two: &[Option<(f64, usize)>],
    k: usize,
) -> Result<(), PhysicsError> {
    let disagree = |first: (f64, usize), second: (f64, usize), a: String, b: String| {
        let ((v, line_v), (w, line_w)) = (first, second);
        invalid(
            Some(line_v.max(line_w)),
            format!(
                "{a} = {v} (line {line_v}) and {b} = {w} (line {line_w}) must be equal, but \
                 differ by more than {INTEGRAL_SYMMETRY_TOLERANCE}"
            ),
        )
    };
    for p in 0..k {
        for q in 0..p {
            if let (Some(a), Some(b)) = (one[p * k + q], one[q * k + p]) {
                if (a.0 - b.0).abs() > INTEGRAL_SYMMETRY_TOLERANCE {
                    return Err(disagree(
                        a,
                        b,
                        format!("h {} {}", p + 1, q + 1),
                        format!("h {} {}", q + 1, p + 1),
                    ));
                }
            }
        }
    }
    let idx = |p: usize, q: usize, r: usize, s: usize| ((p * k + q) * k + r) * k + s;
    for p in 0..k {
        for q in 0..k {
            for r in 0..k {
                for s in 0..k {
                    let Some(own) = two[idx(p, q, r, s)] else {
                        continue;
                    };
                    for (a, b, c, d) in chemist_permutations(p, q, r, s) {
                        if let Some(other) = two[idx(a, b, c, d)] {
                            if (own.0 - other.0).abs() > INTEGRAL_SYMMETRY_TOLERANCE {
                                return Err(disagree(
                                    own,
                                    other,
                                    format!("({}{}|{}{})", p + 1, q + 1, r + 1, s + 1),
                                    format!("({}{}|{}{})", a + 1, b + 1, c + 1, d + 1),
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn invalid(line: Option<usize>, message: String) -> PhysicsError {
    PhysicsError::InvalidFcidump { line, message }
}

/// What the header declares.
struct Header {
    norb: usize,
    nelec: usize,
    ms2: i64,
}

/// A header token and the 1-based line it is on.
struct Token {
    text: String,
    line: usize,
}

/// Reads the `&FCI … &END` namelist; returns it and the index of the first
/// body line.
fn read_header(text: &str) -> Result<(Header, usize), PhysicsError> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut started = false;
    let mut body_start = None;
    for (index, line) in text.lines().enumerate() {
        let number = index + 1;
        if !started && line.trim().is_empty() {
            continue;
        }
        // `=` and `,` separate tokens; `=` is kept as a token of its own.
        let spaced = line.replace('=', " = ").replace(',', " ");
        let mut words = spaced.split_whitespace();
        while let Some(word) = words.next() {
            let upper = word.to_ascii_uppercase();
            if !started {
                if upper != "&FCI" && upper != "$FCI" {
                    return Err(invalid(
                        Some(number),
                        format!("expected the header `&FCI`, found `{word}`"),
                    ));
                }
                started = true;
                continue;
            }
            if upper == "&END" || upper == "$END" || upper == "/" {
                // Integrals start on the next line; anything after the end
                // marker would be lost (or a whole file whose lines are not
                // split, as with old Mac `\r` endings).
                if let Some(extra) = words.next() {
                    return Err(invalid(
                        Some(number),
                        format!("`{extra}` follows the end of the header on its line"),
                    ));
                }
                body_start = Some(index + 1);
                break;
            }
            tokens.push(Token {
                text: upper,
                line: number,
            });
        }
        if body_start.is_some() {
            break;
        }
    }
    let Some(body_start) = body_start else {
        return Err(invalid(
            None,
            if started {
                "the header `&FCI` has no `&END` (or `/`)".to_string()
            } else {
                "the file is empty".to_string()
            },
        ));
    };

    // `KEY = value value …`: a key is a token followed by `=`.
    let mut norb = None;
    let mut nelec = None;
    let mut ms2 = 0i64;
    let mut seen: Vec<&str> = Vec::new();
    let mut position = 0;
    while position < tokens.len() {
        let key = &tokens[position];
        if tokens.get(position + 1).map(|t| t.text.as_str()) != Some("=") {
            return Err(invalid(
                Some(key.line),
                format!("expected `KEY=value` in the header, found `{}`", key.text),
            ));
        }
        let mut end = position + 2;
        while end < tokens.len() && tokens.get(end + 1).map(|t| t.text.as_str()) != Some("=") {
            end += 1;
        }
        let values = &tokens[position + 2..end];
        let single = || -> Result<&Token, PhysicsError> {
            match values {
                [value] => Ok(value),
                _ => Err(invalid(
                    Some(key.line),
                    format!("`{}` takes one value, found {}", key.text, values.len()),
                )),
            }
        };
        let integer = |token: &Token| -> Result<i64, PhysicsError> {
            token.text.parse::<i64>().map_err(|_| {
                invalid(
                    Some(token.line),
                    format!("`{}` is not an integer for `{}`", token.text, key.text),
                )
            })
        };
        if matches!(
            key.text.as_str(),
            "NORB" | "NELEC" | "MS2" | "IUHF" | "UHF" | "TREL"
        ) {
            if seen.contains(&key.text.as_str()) {
                return Err(invalid(
                    Some(key.line),
                    format!("`{}` is given twice", key.text),
                ));
            }
            seen.push(key.text.as_str());
        }
        match key.text.as_str() {
            "NORB" => norb = Some(integer(single()?)?),
            "NELEC" => nelec = Some(integer(single()?)?),
            "MS2" => ms2 = integer(single()?)?,
            "IUHF" => {
                let value = single()?;
                if integer(value)? != 0 {
                    return Err(invalid(
                        Some(value.line),
                        "unrestricted integrals (`IUHF`) are not supported".to_string(),
                    ));
                }
            }
            "UHF" | "TREL" => {
                let value = single()?;
                // A Fortran logical is true when it reads `T`, after an
                // optional `.` (`T`, `.T.`, `.TRUE.`, `TRUE`, any case).
                if value.text.trim_start_matches('.').starts_with('T') {
                    let what = if key.text == "UHF" {
                        "unrestricted integrals (`UHF=.TRUE.`)"
                    } else {
                        "complex integrals (`TREL=.TRUE.`)"
                    };
                    return Err(invalid(
                        Some(value.line),
                        format!("{what} are not supported"),
                    ));
                }
            }
            _ => {} // ORBSYM, ISYM and any other key: not needed
        }
        position = end;
    }

    let norb = norb.ok_or_else(|| invalid(None, "the header has no `NORB`".to_string()))?;
    let nelec = nelec.ok_or_else(|| invalid(None, "the header has no `NELEC`".to_string()))?;
    if !(1..=MAX_FCIDUMP_ORBITALS as i64).contains(&norb) {
        return Err(invalid(
            None,
            format!("`NORB` = {norb} is outside 1..={MAX_FCIDUMP_ORBITALS}"),
        ));
    }
    let k = norb;
    // Ranges first, so that `nelec ± ms2` below cannot overflow.
    if !(0..=2 * k).contains(&nelec) || ms2.unsigned_abs() > nelec as u64 || (nelec + ms2) % 2 != 0
    {
        return Err(invalid(
            None,
            format!("`NELEC` = {nelec} and `MS2` = {ms2} are not electron counts"),
        ));
    }
    let (n_up, n_down) = ((nelec + ms2) / 2, (nelec - ms2) / 2);
    if n_up > k || n_down > k {
        return Err(invalid(
            None,
            format!("{n_up} spin-up and {n_down} spin-down electrons do not fit in {k} orbitals"),
        ));
    }
    Ok((
        Header {
            norb: k as usize,
            nelec: nelec as usize,
            ms2,
        },
        body_start,
    ))
}

/// `value i j k l` on line `number`, indices checked against `norb`.
fn parse_integral(
    line: &str,
    number: usize,
    norb: usize,
) -> Result<(f64, [usize; 4]), PhysicsError> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    let [value, i, j, k, l] = fields.as_slice() else {
        return Err(invalid(
            Some(number),
            format!("expected `value i j k l`, found {} fields", fields.len()),
        ));
    };
    let value: f64 = value
        .replace(['D', 'd'], "E")
        .parse()
        .map_err(|_| invalid(Some(number), format!("`{value}` is not a number")))?;
    if !value.is_finite() {
        return Err(invalid(
            Some(number),
            format!("value {value} is not finite"),
        ));
    }
    let mut indices = [0usize; 4];
    for (slot, text) in indices.iter_mut().zip([i, j, k, l]) {
        *slot = text
            .parse()
            .map_err(|_| invalid(Some(number), format!("`{text}` is not an orbital index")))?;
        if *slot > norb {
            return Err(invalid(
                Some(number),
                format!("orbital index {slot} is above `NORB` = {norb}"),
            ));
        }
    }
    Ok((value, indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-orbital file in PySCF's layout, with a 4-fold pair that
    /// differs in the last digit and an orbital energy.
    const SMALL: &str = " &FCI NORB=   2,NELEC= 2,MS2=0,
  ORBSYM=1,1,
  ISYM=1,
 &END
 0.6746 1 1 1 1
 0.6634680964235673 1 1 2 2
 0.6634680964235676 2 2 1 1
 0.1813 2 1 2 1
 0.6975 2 2 2 2
 -1.2528 1 1 0 0
 -0.4759 2 2 0 0
 -0.5 1 0 0 0
 0.7137 0 0 0 0
";

    fn parse(text: &str) -> Fcidump {
        Fcidump::parse(text).expect("a valid FCIDUMP")
    }

    fn error_of(text: &str) -> PhysicsError {
        Fcidump::parse(text).expect_err("an invalid FCIDUMP")
    }

    #[test]
    fn a_small_file_gives_its_integrals_and_sector() {
        let file = parse(SMALL);
        let h = file.hamiltonian();
        assert_eq!(file.num_orbitals(), 2);
        assert_eq!(
            (file.num_electrons(), file.ms2(), file.sector()),
            (2, 0, (1, 1))
        );
        // Exact: values read as written (a group of one is its own mean).
        assert_eq!(h.nuclear_repulsion(), 0.7137);
        assert_eq!(h.one_body(0, 0), Some(-1.2528));
        assert_eq!(h.one_body(1, 1), Some(-0.4759));
        assert_eq!(h.one_body(0, 1), Some(0.0));
        assert_eq!(h.two_body(0, 0, 0, 0), Some(0.6746));
        // (21|21) listed once fills its 8 permutations.
        for (p, q, r, s) in chemist_permutations(1, 0, 1, 0) {
            assert_eq!(h.two_body(p, q, r, s), Some(0.1813), "({p}{q}|{r}{s})");
        }
        // (11|22) and (22|11), 3e-16 apart, become their mean in both.
        let pair = h.two_body(0, 0, 1, 1).unwrap();
        assert_eq!(pair, h.two_body(1, 1, 0, 0).unwrap());
        assert!((pair - 0.6634680964235674).abs() < 1e-15, "{pair}");
    }

    #[test]
    fn the_header_may_be_written_in_any_style() {
        let base = parse(SMALL);
        let body = SMALL.split_once("&END\n").unwrap().1;
        for header in [
            "&fci norb=2, nelec=2, ms2=0, orbsym=1,1, isym=1 /\n",
            "&FCI\n NORB=2,\n NELEC=2,\n UHF=.FALSE., TREL=.FALSE., OCC=1,0,\n&END\n",
            "  $FCI NORB = 2 NELEC = 2 IUHF = 0 $END\n",
        ] {
            let file = parse(&format!("{header}{body}"));
            // Exact: the same integrals, read from the same text.
            assert_eq!(file.hamiltonian(), base.hamiltonian(), "{header}");
            assert_eq!(file.sector(), (1, 1));
        }
        // Fortran exponents and blank lines.
        let fortran = SMALL.replace(" 0.6746 1 1 1 1", "\n 6.746D-01 1 1 1 1\n");
        // Exact: `6.746D-01` and `0.6746` parse to the same f64.
        assert_eq!(parse(&fortran).hamiltonian(), base.hamiltonian());
    }

    #[test]
    fn an_open_shell_header_gives_its_sector() {
        let doublet = SMALL.replace("NELEC= 2,MS2=0", "NELEC= 1,MS2=1");
        assert_eq!(parse(&doublet).sector(), (1, 0));
        let triplet = SMALL.replace("NELEC= 2,MS2=0", "NELEC= 2,MS2=-2");
        assert_eq!(parse(&triplet).sector(), (0, 2));
    }

    /// Each rejection names its problem and, when on one line, the line.
    #[test]
    fn malformed_files_are_rejected_with_their_line() {
        let cases: [(&str, Option<usize>, &str); 29] = [
            // Extreme counts must be rejected, never overflow (#535 review).
            (
                " &FCI NORB=1 NELEC=0 MS2=-9223372036854775808 &END\n",
                None,
                "not electron counts",
            ),
            (
                " &FCI NORB=1 NELEC=9223372036854775807 MS2=1 &END\n",
                None,
                "not electron counts",
            ),
            (
                " &FCI NORB=1 NELEC=9223372036854775807 MS2=-1 &END\n",
                None,
                "not electron counts",
            ),
            (
                " &FCI NORB=2 NELEC=2 MS2=-4 &END\n",
                None,
                "not electron counts",
            ),
            (" &FCI NORB=0 NELEC=0 &END\n", None, "outside"),
            (" &FCI NORB=33 NELEC=2 &END\n", None, "outside"),
            (" &FCI NORB=2 NELEC=2 UHF=T &END\n", Some(1), "unrestricted"),
            (
                " &FCI NORB=2 NELEC=2 UHF=.true &END\n",
                Some(1),
                "unrestricted",
            ),
            (
                " &FCI NORB=2 NELEC=2 UHF=TRUE &END\n",
                Some(1),
                "unrestricted",
            ),
            // Nothing may follow the end marker on its line.
            (
                " &FCI NORB=2 NELEC=2 &END 0.5 1 1 1 1\n",
                Some(1),
                "follows the end",
            ),
            // Old Mac `\r` endings: one line, so the integrals would be lost.
            (
                " &FCI NORB=2 NELEC=2 &END\r 0.5 1 1 1 1\r",
                Some(1),
                "follows the end",
            ),
            (" &FCI NORB=2 NELEC=2 &END\n\n", None, "no integral"),
            (" &FCI NORB=1 NORB=2 NELEC=2 &END\n", Some(1), "given twice"),
            ("", None, "empty"),
            ("NORB=2\n", Some(1), "expected the header"),
            (" &FCI NORB=2, NELEC=2,\n 0.5 1 1 1 1\n", None, "no `&END`"),
            (" &FCI NELEC=2 &END\n", None, "no `NORB`"),
            (" &FCI NORB=2 &END\n", None, "no `NELEC`"),
            (" &FCI NORB=40 NELEC=2 &END\n", None, "outside"),
            (
                " &FCI NORB=2 NELEC=3 MS2=0 &END\n",
                None,
                "not electron counts",
            ),
            (
                " &FCI NORB=2 NELEC=2 MS2=4 &END\n",
                None,
                "not electron counts",
            ),
            (" &FCI NORB=1 NELEC=2 MS2=2 &END\n", None, "do not fit"),
            (
                " &FCI NORB=2 NELEC=2 IUHF=1 &END\n",
                Some(1),
                "unrestricted",
            ),
            (
                " &FCI NORB=2 NELEC=2\n UHF=.TRUE. &END\n",
                Some(2),
                "unrestricted",
            ),
            (
                " &FCI NORB=2 NELEC=2 TREL=.TRUE. &END\n",
                Some(1),
                "complex",
            ),
            (
                " &FCI NORB=2 NELEC=2 &END\n 0.5 3 1 1 1\n",
                Some(2),
                "above `NORB`",
            ),
            (
                " &FCI NORB=2 NELEC=2 &END\n inf 1 1 1 1\n",
                Some(2),
                "not finite",
            ),
            (
                " &FCI NORB=2 NELEC=2 &END\n 0.5 1 0 1 0\n",
                Some(2),
                "not an integral",
            ),
            (
                " &FCI NORB=2 NELEC=2 &END\n 0.5 1 1 1 1\n 0.6 1 1 1 1\n",
                Some(3),
                "given twice",
            ),
        ];
        for (text, line, message) in cases {
            let error = error_of(text);
            let PhysicsError::InvalidFcidump {
                line: got,
                message: got_message,
            } = &error
            else {
                panic!("{text:?}: {error:?}");
            };
            assert_eq!(*got, line, "{text:?}: {error}");
            assert!(got_message.contains(message), "{text:?}: {error}");
        }
        for (text, message) in [
            (" &FCI NORB=2 NELEC=2 &END\n 0.5 1 1 1\n", "4 fields"),
            (
                " &FCI NORB=2 NELEC=2 &END\n 0.12-100 1 1 1 1\n",
                "not a number",
            ),
            (" &FCI NORB=2 NELEC=2 &END\n 0.5 1 x 1 1\n", "orbital index"),
            (" &FCI NORB=two NELEC=2 &END\n", "not an integer"),
        ] {
            let error = error_of(text);
            assert!(error.to_string().contains(message), "{text:?}: {error}");
        }
    }

    /// Two listed members of a group that differ beyond the tolerance are
    /// rejected, named as in the file (1-based) with their lines, not
    /// silently averaged.
    #[test]
    fn a_broken_symmetry_is_rejected_with_its_lines() {
        let two = " &FCI NORB=2 NELEC=2 &END\n 0.5 1 1 2 2\n 0.6 2 2 1 1\n";
        let error = error_of(two);
        assert!(
            matches!(&error, PhysicsError::InvalidFcidump { line: Some(3), message }
                if message.contains("(11|22) = 0.5 (line 2)") && message.contains("(22|11) = 0.6 (line 3)")),
            "{error}"
        );
        let one = " &FCI NORB=2 NELEC=2 &END\n 0.5 1 2 0 0\n 0.6 2 1 0 0\n";
        let error = error_of(one);
        assert!(
            matches!(&error, PhysicsError::InvalidFcidump { line: Some(3), message }
                if message.contains("h 2 1") && message.contains("h 1 2")),
            "{error}"
        );
    }

    /// An 8-fold listing (PySCF's `from_integrals` default) gives each
    /// group once, in any member: `(22|11)` alone fills `(11|22)` and the
    /// other six.
    #[test]
    fn one_member_fills_its_whole_group() {
        let file = parse(" &FCI NORB=2 NELEC=2 &END\n 0.25 2 2 1 1\n 0.5 2 1 0 0\n");
        let h = file.hamiltonian();
        for (p, q, r, s) in chemist_permutations(1, 1, 0, 0) {
            assert_eq!(h.two_body(p, q, r, s), Some(0.25), "({p}{q}|{r}{s})");
        }
        // Exact: a value listed once is stored as read.
        assert_eq!(h.one_body(0, 1), Some(0.5));
        assert_eq!(h.one_body(1, 0), Some(0.5));
    }

    /// The largest `NORB` is accepted; a byte-order mark is ignored.
    #[test]
    fn the_orbital_cap_and_a_byte_order_mark_are_accepted() {
        let file = parse("\u{feff} &FCI NORB=32 NELEC=2 &END\n 1.5 0 0 0 0\n");
        assert_eq!(file.num_orbitals(), MAX_FCIDUMP_ORBITALS);
        // Exact: the core energy as read.
        assert_eq!(file.hamiltonian().nuclear_repulsion(), 1.5);
    }

    #[test]
    fn a_missing_file_is_an_io_error() {
        let error = Fcidump::read("/nonexistent/polypus.fcidump").unwrap_err();
        assert!(matches!(error, PhysicsError::IoError { .. }), "{error:?}");
    }
}
