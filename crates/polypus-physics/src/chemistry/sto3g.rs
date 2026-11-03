//! Integrals over contracted s-type Gaussian orbitals (STO-3G).
//!
//! Closed forms of Szabo & Ostlund, *Modern Quantum Chemistry*, Appendix A,
//! for primitive Gaussians `exp(−α |r − A|²)`: overlap, kinetic energy,
//! nuclear attraction and electron repulsion (chemists' notation `(ab|cd)`).
//! The last two need the Boys function `F₀(t) = ½ √(π/t) erf(√t)`.
//!
//! # Limitations
//!
//! - **s orbitals only.** p and d orbitals need higher Boys functions and
//!   angular factors that are not implemented. Molecules beyond H₂-like
//!   systems (e.g. LiH, H₂O) are meant to enter through FCIDUMP files
//!   (issue #535), not through this engine.
//! - Units are atomic: positions in bohr, energies in hartree.

use crate::error::PhysicsError;
use std::f64::consts::PI;

/// STO-3G exponents of the hydrogen 1s orbital (Slater exponent ζ = 1.24).
const STO3G_H_EXPONENTS: [f64; 3] = [3.425_250_91, 0.623_913_73, 0.168_855_40];

/// STO-3G contraction coefficients of the 1s orbital.
const STO3G_1S_COEFFICIENTS: [f64; 3] = [0.154_328_97, 0.535_328_14, 0.444_634_54];

/// Below this argument `F₀` uses its Taylor series `1 − t/3 + t²/10`
/// (truncation error `< t³/42`), avoiding the `0/0` of the erf form.
const BOYS_SERIES_LIMIT: f64 = 1e-8;

/// A contracted s-type Gaussian orbital centred at `center` (bohr):
/// `φ(r) = Σ_k c_k exp(−α_k |r − center|²)`, with each `c_k` already
/// including the primitive's normalization `(2α_k/π)^{3/4}`.
#[derive(Debug, Clone, PartialEq)]
pub struct ContractedGaussian {
    center: [f64; 3],
    /// `(α_k, c_k)` pairs.
    primitives: Vec<(f64, f64)>,
}

impl ContractedGaussian {
    /// Builds an orbital from `(exponent, contraction coefficient)` pairs; each
    /// coefficient is multiplied by its primitive's normalization.
    ///
    /// # Errors
    /// [`PhysicsError::InvalidIntegrals`] if there are no primitives, an
    /// exponent is not finite and positive, or a coefficient or coordinate is
    /// not finite.
    pub fn new(center: [f64; 3], primitives: &[(f64, f64)]) -> Result<Self, PhysicsError> {
        let invalid = |message: String| PhysicsError::InvalidIntegrals { message };
        if primitives.is_empty() {
            return Err(invalid(
                "a contracted Gaussian needs at least one primitive".into(),
            ));
        }
        if center.iter().any(|x| !x.is_finite()) {
            return Err(invalid(format!("orbital center is not finite: {center:?}")));
        }
        let mut normalized = Vec::with_capacity(primitives.len());
        for &(alpha, d) in primitives {
            if !(alpha.is_finite() && alpha > 0.0) {
                return Err(invalid(format!(
                    "Gaussian exponent must be finite and positive, got {alpha}"
                )));
            }
            if !d.is_finite() {
                return Err(invalid(format!(
                    "contraction coefficient is not finite: {d}"
                )));
            }
            normalized.push((alpha, d * (2.0 * alpha / PI).powf(0.75)));
        }
        Ok(Self {
            center,
            primitives: normalized,
        })
    }

    /// The STO-3G hydrogen 1s orbital centred at `center` (bohr).
    ///
    /// # Errors
    /// [`PhysicsError::InvalidIntegrals`] if a coordinate is not finite.
    pub fn sto3g_hydrogen_1s(center: [f64; 3]) -> Result<Self, PhysicsError> {
        let primitives: Vec<(f64, f64)> = STO3G_H_EXPONENTS
            .into_iter()
            .zip(STO3G_1S_COEFFICIENTS)
            .collect();
        Self::new(center, &primitives)
    }

    /// Center of the orbital (bohr).
    pub fn center(&self) -> [f64; 3] {
        self.center
    }
}

/// A point nucleus: position (bohr) and charge (units of `e`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nucleus {
    /// Position (bohr).
    pub position: [f64; 3],
    /// Charge `Z`.
    pub charge: f64,
}

fn distance_sq(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum()
}

/// Gaussian product center `(α A + β B) / (α + β)`.
fn product_center(alpha: f64, a: [f64; 3], beta: f64, b: [f64; 3]) -> [f64; 3] {
    let p = alpha + beta;
    [0, 1, 2].map(|i| (alpha * a[i] + beta * b[i]) / p)
}

/// Boys function of order 0, `F₀(t) = ∫₀¹ exp(−t u²) du`.
fn boys_f0(t: f64) -> f64 {
    if t < BOYS_SERIES_LIMIT {
        1.0 - t / 3.0 + t * t / 10.0
    } else {
        0.5 * (PI / t).sqrt() * libm::erf(t.sqrt())
    }
}

/// Overlap `S_ab = ∫ φ_a φ_b`.
pub fn overlap(a: &ContractedGaussian, b: &ContractedGaussian) -> f64 {
    let r2 = distance_sq(a.center, b.center);
    let mut s = 0.0;
    for &(alpha, ca) in &a.primitives {
        for &(beta, cb) in &b.primitives {
            let p = alpha + beta;
            s += ca * cb * (PI / p).powf(1.5) * (-alpha * beta / p * r2).exp();
        }
    }
    s
}

/// Kinetic energy `T_ab = ∫ φ_a (−½∇²) φ_b`.
pub fn kinetic(a: &ContractedGaussian, b: &ContractedGaussian) -> f64 {
    let r2 = distance_sq(a.center, b.center);
    let mut t = 0.0;
    for &(alpha, ca) in &a.primitives {
        for &(beta, cb) in &b.primitives {
            let p = alpha + beta;
            let mu = alpha * beta / p;
            t += ca * cb * mu * (3.0 - 2.0 * mu * r2) * (PI / p).powf(1.5) * (-mu * r2).exp();
        }
    }
    t
}

/// Nuclear attraction `V_ab = Σ_C ∫ φ_a (−Z_C / |r − C|) φ_b`.
pub fn nuclear_attraction(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    nuclei: &[Nucleus],
) -> f64 {
    let r2 = distance_sq(a.center, b.center);
    let mut v = 0.0;
    for &(alpha, ca) in &a.primitives {
        for &(beta, cb) in &b.primitives {
            let p = alpha + beta;
            let prefactor = ca * cb * 2.0 * PI / p * (-alpha * beta / p * r2).exp();
            let center = product_center(alpha, a.center, beta, b.center);
            for nucleus in nuclei {
                v -=
                    nucleus.charge * prefactor * boys_f0(p * distance_sq(center, nucleus.position));
            }
        }
    }
    v
}

/// Electron repulsion `(ab|cd) = ∫∫ φ_a(1) φ_b(1) (1/r₁₂) φ_c(2) φ_d(2)`
/// (chemists' notation).
pub fn electron_repulsion(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    c: &ContractedGaussian,
    d: &ContractedGaussian,
) -> f64 {
    let r2_ab = distance_sq(a.center, b.center);
    let r2_cd = distance_sq(c.center, d.center);
    let mut g = 0.0;
    for &(alpha, ca) in &a.primitives {
        for &(beta, cb) in &b.primitives {
            let p = alpha + beta;
            let pc = product_center(alpha, a.center, beta, b.center);
            let ab = ca * cb * (-alpha * beta / p * r2_ab).exp();
            for &(gamma, cc) in &c.primitives {
                for &(delta, cd) in &d.primitives {
                    let q = gamma + delta;
                    let qc = product_center(gamma, c.center, delta, d.center);
                    let cdf = cc * cd * (-gamma * delta / q * r2_cd).exp();
                    g += ab * cdf * 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt())
                        * boys_f0(p * q / (p + q) * distance_sq(pc, qc));
                }
            }
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    /// H₂ at the Szabo & Ostlund geometry, R = 1.4 bohr along z.
    fn szabo_h2() -> (ContractedGaussian, ContractedGaussian, [Nucleus; 2]) {
        let (ra, rb) = ([0.0, 0.0, 0.0], [0.0, 0.0, 1.4]);
        (
            ContractedGaussian::sto3g_hydrogen_1s(ra).unwrap(),
            ContractedGaussian::sto3g_hydrogen_1s(rb).unwrap(),
            [
                Nucleus {
                    position: ra,
                    charge: 1.0,
                },
                Nucleus {
                    position: rb,
                    charge: 1.0,
                },
            ],
        )
    }

    /// Szabo & Ostlund publish 4 decimals: agree to half a unit of the last.
    fn assert_published(value: f64, published: f64, name: &str) {
        assert!(
            (value - published).abs() <= 5e-5,
            "{name}: {value} vs published {published}"
        );
    }

    #[test]
    fn integrals_match_szabo_ostlund_at_1_4_bohr() {
        let (a, b, nuclei) = szabo_h2();
        assert_published(overlap(&a, &b), 0.6593, "S12");
        assert_published(kinetic(&a, &a), 0.7600, "T11");
        assert_published(kinetic(&a, &b), 0.2365, "T12");
        let one = |p: &ContractedGaussian, q: &ContractedGaussian| {
            kinetic(p, q) + nuclear_attraction(p, q, &nuclei)
        };
        assert_published(one(&a, &a), -1.1204, "H11");
        assert_published(one(&a, &b), -0.9584, "H12");
        assert_published(nuclear_attraction(&a, &a, &nuclei[..1]), -1.2266, "V11(1)");
        assert_published(nuclear_attraction(&a, &a, &nuclei[1..]), -0.6538, "V11(2)");
        assert_published(nuclear_attraction(&a, &b, &nuclei[..1]), -0.5974, "V12(1)");
        assert_published(electron_repulsion(&a, &a, &a, &a), 0.7746, "(11|11)");
        assert_published(electron_repulsion(&a, &a, &b, &b), 0.5697, "(11|22)");
        assert_published(electron_repulsion(&b, &a, &a, &a), 0.4441, "(21|11)");
        assert_published(electron_repulsion(&b, &a, &b, &a), 0.2970, "(21|21)");
    }

    #[test]
    fn boys_function_matches_reference_values() {
        // Reference: ½ √(π/t) erf(√t) with CPython's math.erf.
        let cases = [
            (0.0, 1.0),
            (1e-10, 0.9999999999666668),
            (1e-8, 0.9999999966666667),
            (1e-6, 0.9999996666667668),
            (0.5, 0.8556243918921488),
            (1.0, 0.746824132812427),
            (10.0, 0.28024739050664277),
            (50.0, 0.12533141373155002),
        ];
        for (t, expected) in cases {
            let got = boys_f0(t);
            assert!(
                (got - expected).abs() <= 1e-15,
                "F0({t}) = {got}, expected {expected}"
            );
        }
        // No jump where the Taylor series hands over to the erf form.
        let below = boys_f0(BOYS_SERIES_LIMIT * (1.0 - 1e-12));
        let above = boys_f0(BOYS_SERIES_LIMIT);
        assert!((below - above).abs() < 1e-15);
    }

    #[test]
    fn sto3g_orbital_is_normalized_to_contraction_accuracy() {
        // STO-3G coefficients are given to 8 decimals; S_aa = 1 − 9e-9.
        let (a, _, _) = szabo_h2();
        assert!((overlap(&a, &a) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn integrals_have_their_permutation_symmetries() {
        let (a, b, nuclei) = szabo_h2();
        assert_eq!(overlap(&a, &b).to_bits(), overlap(&b, &a).to_bits());
        assert!((kinetic(&a, &b) - kinetic(&b, &a)).abs() < 1e-15);
        assert!(
            (nuclear_attraction(&a, &b, &nuclei) - nuclear_attraction(&b, &a, &nuclei)).abs()
                < 1e-15
        );
        let reference = electron_repulsion(&a, &b, &a, &a);
        for g in [
            electron_repulsion(&b, &a, &a, &a),
            electron_repulsion(&a, &a, &a, &b),
            electron_repulsion(&a, &a, &b, &a),
        ] {
            assert!((g - reference).abs() < 1e-15);
        }
    }

    #[test]
    fn far_apart_orbitals_behave_as_point_charges() {
        // At R = 50 bohr: no overlap, (aa|bb) → 1/R and V → −Z/R, each
        // times the orbital's norm (1 − 9e-9).
        let r = 50.0;
        let a = ContractedGaussian::sto3g_hydrogen_1s([0.0; 3]).unwrap();
        let b = ContractedGaussian::sto3g_hydrogen_1s([0.0, 0.0, r]).unwrap();
        let far = Nucleus {
            position: [0.0, 0.0, r],
            charge: 2.0,
        };
        assert!(overlap(&a, &b) < 1e-90);
        assert!((electron_repulsion(&a, &a, &b, &b) * r - 1.0).abs() < 1e-7);
        assert!((nuclear_attraction(&a, &a, &[far]) * r + 2.0).abs() < 1e-7);
    }

    #[test]
    fn invalid_orbitals_are_rejected() {
        let bad = [
            ContractedGaussian::new([0.0; 3], &[]),
            ContractedGaussian::new([0.0; 3], &[(0.0, 1.0)]),
            ContractedGaussian::new([0.0; 3], &[(-1.0, 1.0)]),
            ContractedGaussian::new([0.0; 3], &[(f64::INFINITY, 1.0)]),
            ContractedGaussian::new([0.0; 3], &[(1.0, f64::NAN)]),
            ContractedGaussian::new([f64::NAN, 0.0, 0.0], &[(1.0, 1.0)]),
        ];
        for r in bad {
            assert!(matches!(r, Err(PhysicsError::InvalidIntegrals { .. })));
        }
    }
}
