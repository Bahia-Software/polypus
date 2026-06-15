//! Energy-spectrum sampling for primary particles.
//!
//! A particle source is rarely monoenergetic: an X-ray tube at 100 kVp emits a
//! continuous bremsstrahlung spectrum, not a single 100 keV photon. This module
//! lets the [`MonteCarloEngine`](super::MonteCarloEngine) draw each primary's
//! initial energy from a distribution.
//!
//! # Scalable design
//!
//! All spectra implement the object-safe [`EnergySpectrum`] trait, so adding a
//! new source model requires **only** a new implementor — no changes to the
//! engine or any other module (the same contract used by `Particle`, `Medium`
//! and `InteractionModel`).
//!
//! Two families are provided:
//!
//! | Kind | Type | Data source |
//! |---|---|---|
//! | Trivial | [`Monoenergetic`], [`UniformSpectrum`] | none |
//! | Analytical | [`KramersSpectrum`] | closed-form (public-domain Kramers' law) |
//! | Tabulated | [`TabulatedSpectrum`] | arbitrary `(energy, weight)` bins |
//!
//! [`TabulatedSpectrum`] is the **bridge to future real data**: any externally
//! generated spectrum (e.g. an open-source SpekPy export, TASMIP coefficients,
//! or a measured CSV) becomes a `TabulatedSpectrum` via inverse-CDF sampling,
//! with no new sampling logic. A `serde`-gated loader can be added later
//! without touching this trait.

use crate::error::PhysicsError;
use rand::Rng;
use rand::RngCore;

/// A probability distribution over primary-particle energies (MeV).
///
/// Object-safe: `Box<dyn EnergySpectrum>` and `&dyn EnergySpectrum` are usable.
/// The RNG is passed as `&mut dyn RngCore` (rather than a generic `impl Rng`)
/// precisely to keep the trait object-safe.
///
/// # Adding a new spectrum
///
/// Implement this trait for a new type. The
/// [`MonteCarloEngine`](super::MonteCarloEngine) consumes it through
/// [`run_with_spectrum`](super::MonteCarloEngine::run_with_spectrum) with zero
/// changes elsewhere.
pub trait EnergySpectrum: Send + Sync + std::fmt::Debug {
    /// Draw one energy sample (MeV) from the distribution.
    ///
    /// The returned value is guaranteed to lie in
    /// `[min_energy_mev, max_energy_mev]`.
    fn sample_energy_mev(&self, rng: &mut dyn RngCore) -> f64;

    /// Lowest energy the distribution can return (MeV).
    fn min_energy_mev(&self) -> f64;

    /// Highest energy the distribution can return (MeV).
    fn max_energy_mev(&self) -> f64;
}

/// A delta-function source: every primary has exactly the same energy.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Monoenergetic {
    energy_mev: f64,
}

impl Monoenergetic {
    /// Create a monoenergetic source at `energy_mev` (MeV).
    ///
    /// # Errors
    ///
    /// [`PhysicsError::NonPositiveEnergy`] if `energy_mev <= 0`.
    pub fn new(energy_mev: f64) -> Result<Self, PhysicsError> {
        if energy_mev <= 0.0 {
            return Err(PhysicsError::NonPositiveEnergy { energy_mev });
        }
        Ok(Monoenergetic { energy_mev })
    }

    /// The fixed energy of this source (MeV).
    pub fn energy_mev(&self) -> f64 {
        self.energy_mev
    }
}

impl EnergySpectrum for Monoenergetic {
    fn sample_energy_mev(&self, _rng: &mut dyn RngCore) -> f64 {
        self.energy_mev
    }

    fn min_energy_mev(&self) -> f64 {
        self.energy_mev
    }

    fn max_energy_mev(&self) -> f64 {
        self.energy_mev
    }
}

/// A flat (uniform) energy distribution on `[min_energy_mev, max_energy_mev]`.
///
/// Mostly useful for testing and as a featureless baseline.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UniformSpectrum {
    min_energy_mev: f64,
    max_energy_mev: f64,
}

impl UniformSpectrum {
    /// Create a uniform spectrum on `[min_energy_mev, max_energy_mev]` (MeV).
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidSpectrum`] if `min_energy_mev <= 0` or
    /// `min_energy_mev >= max_energy_mev`.
    pub fn new(min_energy_mev: f64, max_energy_mev: f64) -> Result<Self, PhysicsError> {
        if min_energy_mev <= 0.0 || min_energy_mev >= max_energy_mev {
            return Err(PhysicsError::InvalidSpectrum {
                message: format!(
                    "require 0 < min < max, got min = {min_energy_mev}, max = {max_energy_mev}"
                ),
            });
        }
        Ok(UniformSpectrum {
            min_energy_mev,
            max_energy_mev,
        })
    }
}

impl EnergySpectrum for UniformSpectrum {
    fn sample_energy_mev(&self, rng: &mut dyn RngCore) -> f64 {
        rng.gen_range(self.min_energy_mev..self.max_energy_mev)
    }

    fn min_energy_mev(&self) -> f64 {
        self.min_energy_mev
    }

    fn max_energy_mev(&self) -> f64 {
        self.max_energy_mev
    }
}

/// Thick-target bremsstrahlung spectrum from **Kramers' law** (1923).
///
/// The emitted photon-number distribution per unit energy is
///
/// ```text
/// N(E) ∝ (E_max − E) / E,     E_min ≤ E ≤ E_max,
/// ```
///
/// where `E_max` is the tube endpoint energy (numerically equal to the tube
/// potential: a 100 kVp tube gives `E_max = 0.1 MeV`) and `E_min` is the
/// low-energy cutoff imposed by inherent filtration (`N` would otherwise
/// diverge as `1/E`). Characteristic lines are **not** modelled — Kramers' law
/// is the continuous bremsstrahlung background only.
///
/// Sampling uses rejection against the bounding value `N(E_min)` (the
/// distribution is monotonically decreasing in `E`).
///
/// # Example
///
/// ```
/// use polypus_physics::monte_carlo::spectrum::{EnergySpectrum, KramersSpectrum};
/// use rand::{rngs::StdRng, SeedableRng};
///
/// // 100 kVp tube, 10 keV inherent-filtration cutoff.
/// let spec = KramersSpectrum::from_kvp(100.0, 10.0).unwrap();
/// let mut rng = StdRng::seed_from_u64(0);
/// let e = spec.sample_energy_mev(&mut rng);
/// assert!((0.01..=0.1).contains(&e));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KramersSpectrum {
    min_energy_mev: f64,
    max_energy_mev: f64,
}

impl KramersSpectrum {
    /// Create a Kramers spectrum on `[min_energy_mev, max_energy_mev]` (MeV).
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidSpectrum`] if `min_energy_mev <= 0` or
    /// `min_energy_mev >= max_energy_mev`.
    pub fn new(min_energy_mev: f64, max_energy_mev: f64) -> Result<Self, PhysicsError> {
        if min_energy_mev <= 0.0 || min_energy_mev >= max_energy_mev {
            return Err(PhysicsError::InvalidSpectrum {
                message: format!(
                    "require 0 < min < max, got min = {min_energy_mev}, max = {max_energy_mev}"
                ),
            });
        }
        Ok(KramersSpectrum {
            min_energy_mev,
            max_energy_mev,
        })
    }

    /// Convenience constructor from tube parameters in **kilovolts** / keV.
    ///
    /// - `kvp`: tube peak potential (kV). The endpoint energy is
    ///   `E_max = kvp / 1000` MeV (e.g. `100 kVp → 0.1 MeV`).
    /// - `low_cutoff_kev`: inherent-filtration low-energy cutoff (keV),
    ///   `E_min = low_cutoff_kev / 1000` MeV.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidSpectrum`] if the resulting bounds violate
    /// `0 < E_min < E_max`.
    pub fn from_kvp(kvp: f64, low_cutoff_kev: f64) -> Result<Self, PhysicsError> {
        Self::new(low_cutoff_kev * 1e-3, kvp * 1e-3)
    }

    /// Unnormalized photon-number density `N(E) = (E_max − E) / E`.
    fn pdf(&self, energy_mev: f64) -> f64 {
        (self.max_energy_mev - energy_mev) / energy_mev
    }
}

impl EnergySpectrum for KramersSpectrum {
    fn sample_energy_mev(&self, rng: &mut dyn RngCore) -> f64 {
        // The pdf is monotonically decreasing, so its supremum on the support
        // is at E_min.
        let bound = self.pdf(self.min_energy_mev);
        // Rejection sampling. The acceptance probability is well above zero for
        // any physical (E_min, E_max); the cap is a defensive guarantee of
        // termination only and is essentially never reached.
        const MAX_TRIES: u32 = 10_000;
        for _ in 0..MAX_TRIES {
            let candidate = rng.gen_range(self.min_energy_mev..self.max_energy_mev);
            let accept: f64 = rng.gen_range(0.0..1.0);
            if accept * bound <= self.pdf(candidate) {
                return candidate;
            }
        }
        self.min_energy_mev
    }

    fn min_energy_mev(&self) -> f64 {
        self.min_energy_mev
    }

    fn max_energy_mev(&self) -> f64 {
        self.max_energy_mev
    }
}

/// A histogrammed (tabulated) spectrum sampled by **inverse-CDF lookup**.
///
/// This is the general-purpose data-driven path: each bin has a representative
/// energy and a (relative) weight, and sampling returns a bin energy with
/// probability proportional to its weight. Discretised spectra from any source
/// — an open-source generator, a published parametrisation, or a measurement —
/// map directly onto this type.
///
/// Sampling is piecewise-constant (it returns the stored bin energy). Linear
/// interpolation within a bin is a possible future refinement.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TabulatedSpectrum {
    /// Representative energy of each bin (MeV).
    energies_mev: Vec<f64>,
    /// Normalized cumulative distribution; strictly increasing, ending at 1.0.
    cdf: Vec<f64>,
    /// Cached minimum bin energy (MeV).
    min_energy_mev: f64,
    /// Cached maximum bin energy (MeV).
    max_energy_mev: f64,
}

impl TabulatedSpectrum {
    /// Build a tabulated spectrum from per-bin energies and weights.
    ///
    /// - `energies_mev`: representative energy of each bin (MeV), each `> 0`.
    /// - `weights`: relative (un-normalized) probability of each bin, each
    ///   `>= 0`, not all zero. Units are arbitrary; only the ratios matter.
    ///
    /// The two slices must have equal, non-zero length.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidSpectrum`] if the lengths differ, the input is
    /// empty, any energy is non-positive, any weight is negative, or the
    /// weights sum to zero.
    pub fn new(energies_mev: Vec<f64>, weights: Vec<f64>) -> Result<Self, PhysicsError> {
        if energies_mev.len() != weights.len() {
            return Err(PhysicsError::InvalidSpectrum {
                message: format!(
                    "energies and weights differ in length ({} vs {})",
                    energies_mev.len(),
                    weights.len()
                ),
            });
        }
        if energies_mev.is_empty() {
            return Err(PhysicsError::InvalidSpectrum {
                message: "spectrum has no bins".to_string(),
            });
        }
        if energies_mev.iter().any(|&e| e <= 0.0) {
            return Err(PhysicsError::InvalidSpectrum {
                message: "every bin energy must be strictly positive".to_string(),
            });
        }
        if weights.iter().any(|&w| w < 0.0) {
            return Err(PhysicsError::InvalidSpectrum {
                message: "weights must be non-negative".to_string(),
            });
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(PhysicsError::InvalidSpectrum {
                message: "weights sum to zero".to_string(),
            });
        }

        // Build the normalized cumulative distribution.
        let mut cdf = Vec::with_capacity(weights.len());
        let mut running = 0.0;
        for &w in &weights {
            running += w;
            cdf.push(running / total);
        }
        // Pin the last entry to exactly 1.0 against rounding drift.
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }

        let min_energy_mev = energies_mev.iter().copied().fold(f64::INFINITY, f64::min);
        let max_energy_mev = energies_mev
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        Ok(TabulatedSpectrum {
            energies_mev,
            cdf,
            min_energy_mev,
            max_energy_mev,
        })
    }

    /// Number of energy bins.
    pub fn len(&self) -> usize {
        self.energies_mev.len()
    }

    /// Whether the spectrum has no bins. Always `false` for a value built by
    /// [`new`](Self::new) (which rejects empty input); provided for API
    /// completeness alongside [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        self.energies_mev.is_empty()
    }
}

impl EnergySpectrum for TabulatedSpectrum {
    fn sample_energy_mev(&self, rng: &mut dyn RngCore) -> f64 {
        let u: f64 = rng.gen_range(0.0..1.0);
        // First bin whose cumulative probability reaches `u`.
        let idx = self.cdf.partition_point(|&c| c < u).min(self.cdf.len() - 1);
        self.energies_mev[idx]
    }

    fn min_energy_mev(&self) -> f64 {
        self.min_energy_mev
    }

    fn max_energy_mev(&self) -> f64 {
        self.max_energy_mev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn monoenergetic_returns_fixed_energy() {
        let spec = Monoenergetic::new(0.1).unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..100 {
            assert_eq!(spec.sample_energy_mev(&mut rng), 0.1);
        }
        assert_eq!(spec.min_energy_mev(), 0.1);
        assert_eq!(spec.max_energy_mev(), 0.1);
    }

    #[test]
    fn monoenergetic_rejects_non_positive() {
        assert!(matches!(
            Monoenergetic::new(0.0),
            Err(PhysicsError::NonPositiveEnergy { .. })
        ));
    }

    #[test]
    fn uniform_samples_within_bounds() {
        let spec = UniformSpectrum::new(0.01, 0.1).unwrap();
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..10_000 {
            let e = spec.sample_energy_mev(&mut rng);
            assert!((0.01..0.1).contains(&e));
        }
    }

    #[test]
    fn uniform_rejects_bad_bounds() {
        assert!(matches!(
            UniformSpectrum::new(0.1, 0.1),
            Err(PhysicsError::InvalidSpectrum { .. })
        ));
        assert!(matches!(
            UniformSpectrum::new(0.0, 0.1),
            Err(PhysicsError::InvalidSpectrum { .. })
        ));
    }

    #[test]
    fn kramers_samples_within_bounds() {
        let spec = KramersSpectrum::new(0.01, 0.1).unwrap();
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..10_000 {
            let e = spec.sample_energy_mev(&mut rng);
            assert!((0.01..=0.1).contains(&e));
        }
    }

    #[test]
    fn kramers_favours_low_energies() {
        // N(E) ∝ (E_max − E)/E decreases with E, so more samples fall below the
        // midpoint than above it.
        let spec = KramersSpectrum::new(0.01, 0.1).unwrap();
        let mut rng = StdRng::seed_from_u64(4);
        let mid = 0.5 * (0.01 + 0.1);
        let mut below = 0u32;
        let n = 20_000;
        for _ in 0..n {
            if spec.sample_energy_mev(&mut rng) < mid {
                below += 1;
            }
        }
        assert!(
            below > n / 2,
            "expected majority below midpoint, got {below}/{n}"
        );
    }

    #[test]
    fn kramers_from_kvp_maps_bounds() {
        let spec = KramersSpectrum::from_kvp(100.0, 10.0).unwrap();
        assert!((spec.max_energy_mev() - 0.1).abs() < 1e-12);
        assert!((spec.min_energy_mev() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn kramers_rejects_bad_bounds() {
        assert!(matches!(
            KramersSpectrum::new(0.1, 0.05),
            Err(PhysicsError::InvalidSpectrum { .. })
        ));
    }

    #[test]
    fn tabulated_single_active_bin() {
        // Only the second bin has weight, so it is always selected.
        let spec = TabulatedSpectrum::new(vec![0.05, 0.09], vec![0.0, 1.0]).unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..1_000 {
            assert_eq!(spec.sample_energy_mev(&mut rng), 0.09);
        }
    }

    #[test]
    fn tabulated_respects_weights() {
        // Equal weights → roughly half the samples in each bin.
        let spec = TabulatedSpectrum::new(vec![0.05, 0.09], vec![1.0, 1.0]).unwrap();
        let mut rng = StdRng::seed_from_u64(6);
        let n = 10_000;
        let mut first = 0u32;
        for _ in 0..n {
            if spec.sample_energy_mev(&mut rng) == 0.05 {
                first += 1;
            }
        }
        let frac = first as f64 / n as f64;
        assert!((0.45..0.55).contains(&frac), "fraction = {frac}");
    }

    #[test]
    fn tabulated_min_max() {
        let spec = TabulatedSpectrum::new(vec![0.09, 0.02, 0.05], vec![1.0, 1.0, 1.0]).unwrap();
        assert_eq!(spec.min_energy_mev(), 0.02);
        assert_eq!(spec.max_energy_mev(), 0.09);
        assert_eq!(spec.len(), 3);
        assert!(!spec.is_empty());
    }

    #[test]
    fn tabulated_rejects_invalid_input() {
        assert!(TabulatedSpectrum::new(vec![0.1], vec![1.0, 2.0]).is_err());
        assert!(TabulatedSpectrum::new(vec![], vec![]).is_err());
        assert!(TabulatedSpectrum::new(vec![-0.1], vec![1.0]).is_err());
        assert!(TabulatedSpectrum::new(vec![0.1], vec![-1.0]).is_err());
        assert!(TabulatedSpectrum::new(vec![0.1, 0.2], vec![0.0, 0.0]).is_err());
    }

    #[test]
    fn spectrum_trait_is_object_safe() {
        let specs: Vec<Box<dyn EnergySpectrum>> = vec![
            Box::new(Monoenergetic::new(0.1).unwrap()),
            Box::new(UniformSpectrum::new(0.01, 0.1).unwrap()),
            Box::new(KramersSpectrum::new(0.01, 0.1).unwrap()),
            Box::new(TabulatedSpectrum::new(vec![0.05], vec![1.0]).unwrap()),
        ];
        let mut rng = StdRng::seed_from_u64(7);
        for s in &specs {
            let e = s.sample_energy_mev(&mut rng);
            assert!(e >= s.min_energy_mev() - 1e-12 && e <= s.max_energy_mev() + 1e-12);
        }
    }
}
