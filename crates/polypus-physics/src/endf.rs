//! ENDF-6 (EPDL) photon cross-section data reading and interpolation.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use include_dir::{include_dir, Dir};

use crate::constants::{AVOGADRO, BARN_TO_CM2};
use crate::error::PhysicsError;

/// The 100 ENDF-6 files, embedded in the binary.
static ENDF_FILES: Dir = include_dir!("$CARGO_MANIFEST_DIR/src/endf_data");

/// Extracts (Z, symbol) from a file's first line, e.g. "  26-Fe ...".
fn read_element_header(first_line: &str) -> Option<(u32, String)> {
    let (before, after) = first_line.trim_start().split_once('-')?;
    let z: u32 = before.trim_end().parse().ok()?;
    let symbol: String = after
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    (1..=2).contains(&symbol.len()).then_some((z, symbol))
}

/// The last `from_end - to_end` characters of a line, like Python's
/// `line[-from_end:-to_end]`. Assumes ASCII (ENDF-6's actual format).
fn last_chars(line: &str, from_end: usize, to_end: usize) -> &str {
    let n = line.len();
    &line[n - from_end..n - to_end]
}

/// IUPAC atomic masses
const ATOMIC_MASSES: [f64; 100] = [
    1.0080,
    4.002602,
    6.94,
    9.0121831,
    10.81,
    12.011,
    14.007,
    15.999,
    18.998403162,
    20.1797,
    22.98976928,
    24.305,
    26.9815384,
    28.085,
    30.973761998,
    32.06,
    35.45,
    39.95,
    39.0983,
    40.078,
    44.955907,
    47.867,
    50.9415,
    51.9961,
    54.938043,
    55.845,
    58.933194,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.630,
    74.921595,
    78.971,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.905838,
    91.222,
    92.90637,
    95.95,
    97.0,
    101.07,
    102.90549,
    106.42,
    107.8682,
    112.414,
    114.818,
    118.710,
    121.760,
    127.60,
    126.90447,
    131.293,
    132.90545196,
    137.327,
    138.90547,
    140.116,
    140.90766,
    144.242,
    145.0,
    150.36,
    151.964,
    157.249,
    158.925354,
    162.500,
    164.930329,
    167.259,
    168.934219,
    173.045,
    174.96669,
    178.486,
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.966570,
    200.592,
    204.38,
    207.2,
    208.98040,
    209.0,
    210.0,
    222.0,
    223.0,
    226.0,
    227.0,
    232.0377,
    231.03588,
    238.02891,
    237.0,
    244.0,
    243.0,
    247.0,
    247.0,
    251.0,
    252.0,
    257.0,
];

/// Returns the atomic mass for a given Z, or `None` if it is out of range.
fn atomic_mass(z: u32) -> Option<f64> {
    if z == 0 {
        return None;
    }
    ATOMIC_MASSES.get((z - 1) as usize).copied()
}

/// The parsed identity and raw contents of one embedded ENDF-6 file.
struct EndfEntry {
    z: u32,
    contents: &'static str,
}

/// Builds a symbol -> file lookup by reading the first line of every
/// embedded ENDF-6 file. Built once, on first use, and cached from then on.
fn endf_index() -> &'static HashMap<String, EndfEntry> {
    static INDEX: OnceLock<HashMap<String, EndfEntry>> = OnceLock::new();

    INDEX.get_or_init(|| {
        let mut index = HashMap::new();
        for file in ENDF_FILES.files() {
            let contents = file
                .contents_utf8()
                .expect("embedded ENDF-6 files must be valid UTF-8");
            let first_line = contents.lines().next().unwrap_or("");
            if let Some((z, symbol)) = read_element_header(first_line) {
                index.insert(symbol, EndfEntry { z, contents });
            }
        }
        index
    })
}

/// Looks up an element's atomic number from its chemical symbol (e.g. "Fe" -> 26).
pub fn z_for_symbol(symbol: &str) -> Option<u32> {
    endf_index().get(symbol).map(|entry| entry.z)
}

/// A single (energy, cross-section) data point straight from an ENDF-6 file.
struct SectionPoint {
    energy_ev: f64,
    sigma_barn: f64,
}

/// A single (energy, mass attenuation coefficient) data point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MuPoint {
    /// Photon energy, in electronvolts.
    pub energy_ev: f64,
    /// Mass attenuation coefficient, in cm²/g.
    pub mu_m: f64,
}

/// Extracts the MF/MT section of interest from an ENDF-6 file's raw
/// contents, and returns its (energy, cross-section) data points.
fn read_section(contents: &str, mf: u32, mt: u32) -> Result<Vec<SectionPoint>, PhysicsError> {
    let mf_str = mf.to_string();
    let mt_str = mt.to_string();

    let mut section: Vec<&str> = Vec::new();
    for line in contents.lines() {
        let line = line.trim_end();
        if line.len() < 14 {
            continue;
        }
        let line_mf = last_chars(line, 10, 8).trim();
        let line_mt = last_chars(line, 8, 5).trim();
        if line_mf == mf_str && line_mt == mt_str {
            section.push(&line[..line.len() - 14]);
        }
    }

    if section.len() < 2 {
        return Err(PhysicsError::MalformedEndfData {
            message: format!("section MF={mf} MT={mt} not found or incomplete"),
        });
    }

    let tab1_tokens: Vec<&str> = section[1].split_whitespace().collect();
    let nr: u32 = tab1_tokens
        .get(4)
        .ok_or_else(|| PhysicsError::MalformedEndfData {
            message: "TAB1 record is missing the NR field".to_string(),
        })?
        .parse()
        .map_err(|_| PhysicsError::MalformedEndfData {
            message: "NR is not a valid integer".to_string(),
        })?;
    let np: usize = tab1_tokens
        .get(5)
        .ok_or_else(|| PhysicsError::MalformedEndfData {
            message: "TAB1 record is missing the NP field".to_string(),
        })?
        .parse()
        .map_err(|_| PhysicsError::MalformedEndfData {
            message: "NP is not a valid integer".to_string(),
        })?;

    let interpolation_lines = nr.div_ceil(3) as usize;
    let data_start = 2 + interpolation_lines;

    let data_lines = section
        .get(data_start..)
        .ok_or_else(|| PhysicsError::MalformedEndfData {
            message: "no data lines after the interpolation table".to_string(),
        })?;

    let values: Vec<f64> = data_lines
        .iter()
        .flat_map(|line| line.split_whitespace())
        .map(str::parse::<f64>)
        .collect::<Result<_, _>>()
        .map_err(|_| PhysicsError::MalformedEndfData {
            message: "could not parse a numeric value in the section".to_string(),
        })?;

    let points = values
        .chunks_exact(2)
        .take(np)
        .map(|pair| SectionPoint {
            energy_ev: pair[0],
            sigma_barn: pair[1],
        })
        .collect();

    Ok(points)
}

/// Computes the mass attenuation coefficient mu_m = sigma_t * N_A / A for a
/// single element, from its embedded ENDF-6 evaluation.
///
/// `symbol` is the chemical symbol (e.g. `"Fe"`), and `mt` is the ENDF-6
/// reaction type to read (501 = total, 522 = photoelectric, 504 =
/// incoherent/Compton, 502 = coherent/Rayleigh, 515/516/517 = pair
/// production channels).
///
/// Returns the atomic number, atomic mass (g/mol), and the resulting
/// (energy, mu_m) points, sorted by increasing energy as given by the
/// evaluation.
pub fn mu_m_for_element(symbol: &str, mt: u32) -> Result<(u32, f64, Vec<MuPoint>), PhysicsError> {
    let entry = endf_index()
        .get(symbol)
        .ok_or_else(|| PhysicsError::UnknownElement {
            symbol: symbol.to_string(),
        })?;

    let a = atomic_mass(entry.z).ok_or_else(|| PhysicsError::MalformedEndfData {
        message: format!("no atomic mass known for Z={}", entry.z),
    })?;

    let section = read_section(entry.contents, 23, mt)?;

    let points = section
        .iter()
        .map(|point| MuPoint {
            energy_ev: point.energy_ev,
            mu_m: (point.sigma_barn * BARN_TO_CM2 * AVOGADRO) / a,
        })
        .collect();

    Ok((entry.z, a, points))
}

/// Writes a set of (energy, mu_m) points to a CSV file with two
/// metadata rows, a blank row, a header row, and one data row per point.
#[cfg(feature = "csv-export")]
pub fn write_mu_m_csv(
    symbol: &str,
    z: u32,
    a: f64,
    points: &[MuPoint],
    path: &Path,
) -> Result<(), PhysicsError> {
    #[derive(serde::Serialize)]
    struct Row {
        #[serde(rename = "Energy_eV")]
        energy_ev: f64,
        #[serde(rename = "Energy_MeV")]
        energy_mev: f64,
        #[serde(rename = "mu_m_cm2_g")]
        mu_m: f64,
    }

    let to_io_error = |e: std::io::Error| PhysicsError::IoError {
        message: format!("could not write CSV: {e}"),
    };
    let to_csv_error = |e: csv::Error| PhysicsError::IoError {
        message: format!("could not write CSV: {e}"),
    };

    let mut writer = csv::WriterBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(to_csv_error)?;

    writer
        .write_record(["Element", symbol])
        .map_err(to_csv_error)?;
    writer
        .write_record(["Z", &z.to_string()])
        .map_err(to_csv_error)?;
    writer
        .write_record(["A", &format!("{a:?}")])
        .map_err(to_csv_error)?;
    writer.write_record([""]).map_err(to_csv_error)?;

    for point in points {
        writer
            .serialize(Row {
                energy_ev: point.energy_ev,
                energy_mev: point.energy_ev * 1e-6,
                mu_m: point.mu_m,
            })
            .map_err(to_csv_error)?;
    }

    writer.flush().map_err(to_io_error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_all_100_elements() {
        let index = endf_index();
        assert_eq!(index.len(), 100);
    }

    #[test]
    fn oxygen_symbol_resolves_to_z_8() {
        assert_eq!(z_for_symbol("O"), Some(8));
    }

    #[test]
    fn unknown_symbol_returns_none() {
        assert_eq!(z_for_symbol("Xx"), None);
    }

    #[test]
    fn is_cached_across_calls() {
        let first = endf_index() as *const _;
        let second = endf_index() as *const _;
        assert_eq!(
            first, second,
            "endf_index() should return the same cached map"
        );
    }

    #[test]
    fn atomic_mass_of_hydrogen() {
        assert_eq!(atomic_mass(1), Some(1.0080));
    }

    #[test]
    fn atomic_mass_of_iron() {
        assert_eq!(atomic_mass(26), Some(55.845));
    }

    #[test]
    fn atomic_mass_rejects_out_of_range_z() {
        assert_eq!(atomic_mass(0), None);
        assert_eq!(atomic_mass(999), None);
    }

    #[test]
    fn hydrogen_total_cross_section_has_2021_points() {
        let entry = endf_index().get("H").expect("H should be indexed");
        let points =
            read_section(entry.contents, 23, 501).expect("MF=23 MT=501 should exist for H");
        assert_eq!(points.len(), 2021);
    }

    #[test]
    fn hydrogen_total_cross_section_first_point() {
        let entry = endf_index().get("H").expect("H should be indexed");
        let points =
            read_section(entry.contents, 23, 501).expect("MF=23 MT=501 should exist for H");
        let first = &points[0];
        assert_eq!(first.energy_ev, 1.0);
        assert!((first.sigma_barn - 4.62084e-6).abs() < 1e-12);
    }

    #[test]
    fn hydrogen_total_cross_section_last_point() {
        let entry = endf_index().get("H").expect("H should be indexed");
        let points =
            read_section(entry.contents, 23, 501).expect("MF=23 MT=501 should exist for H");
        let last = points.last().unwrap();
        assert_eq!(last.energy_ev, 100_000_000_000.0);
        assert!((last.sigma_barn - 0.020718042).abs() < 1e-9);
    }

    #[test]
    fn unknown_mt_channel_errors() {
        let entry = endf_index().get("H").expect("H should be indexed");
        let result = read_section(entry.contents, 23, 999);
        assert!(matches!(
            result,
            Err(PhysicsError::MalformedEndfData { .. })
        ));
    }

    #[test]
    fn mu_m_for_hydrogen_total_matches_known_values() {
        let (z, a, points) = mu_m_for_element("H", 501).expect("H total mu_m should succeed");
        assert_eq!(z, 1);
        assert_eq!(a, 1.008);
        assert_eq!(points.len(), 2021);
        assert_eq!(points[0].energy_ev, 1.0);
        assert!((points[0].mu_m - 2.7606496933966664e-6).abs() < 1e-15);
        let last = points.last().unwrap();
        assert!((last.mu_m - 0.012377675118610307).abs() < 1e-12);
    }

    #[test]
    fn mu_m_for_unknown_element_errors() {
        let result = mu_m_for_element("Xx", 501);
        assert!(matches!(result, Err(PhysicsError::UnknownElement { .. })));
    }

    #[cfg(feature = "csv-export")]
    #[test]
    fn write_mu_m_csv_produces_a_file() {
        let (z, a, points) = mu_m_for_element("H", 501).unwrap();
        let path = std::env::temp_dir().join("H_mu_m_test.csv");
        write_mu_m_csv("H", z, a, &points, &path).unwrap();
        assert!(path.exists());
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.starts_with("Element,H"));
        println!("CSV written to: {}", path.display());
    }
}
