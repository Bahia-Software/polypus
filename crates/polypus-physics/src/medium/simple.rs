//! A spatially uniform, single-element or single-compound medium.

use super::Medium;

/// A spatially uniform, single-element (or single-compound) medium.
///
/// All quantities are bulk averages. For compounds `Z` is the usual
/// photon-attenuation effective atomic number, while `A` is **not** the naive
/// mean atomic mass: it is chosen so that `n·Z = ρ·N_A·(Z/A)` reproduces the
/// true electron density of the material (the quantity that sets the Compton
/// attenuation). See [`HomogeneousMedium::water`].
#[derive(Debug, Clone)]
pub struct HomogeneousMedium {
    /// Human-readable material name.
    pub name: String,
    /// Mass density `ρ` (kg/m³).
    pub density_kg_m3: f64,
    /// Effective atomic number `Z`.
    pub effective_z: f64,
    /// Effective atomic mass `A` (g/mol).
    pub effective_a: f64,
}

impl HomogeneousMedium {
    /// Liquid water at STP: `ρ = 1000 kg/m³`, `Z_eff = 7.42`.
    ///
    /// `A_eff = 13.37 g/mol` is chosen for **electron-density consistency**, not
    /// as a mean atomic mass: it makes `n·Z_eff` equal the true electron density
    /// of water, whose `Z/A = 10 / 18.015 = 0.5551`
    /// (`A_eff = Z_eff / (Z/A) = 7.42 / 0.5551`). The naive value `11.9` would
    /// give `Z/A = 0.6235` and overestimate the electron density — and hence the
    /// Compton attenuation, which dominates from ~30 keV to ~20 MeV — by ~12 %
    /// (validated against NIST/XCOM `μ/ρ`).
    pub fn water() -> Self {
        HomogeneousMedium {
            name: "water".to_string(),
            density_kg_m3: 1000.0,
            effective_z: 7.42,
            effective_a: 13.37,
        }
    }

    /// Dry air at STP: `ρ = 1.293 kg/m³`, `Z_eff = 7.22`,
    /// `A_eff = 14.5 g/mol`.
    pub fn air() -> Self {
        HomogeneousMedium {
            name: "air".to_string(),
            density_kg_m3: 1.293,
            effective_z: 7.22,
            effective_a: 14.5,
        }
    }

    /// Lead: `ρ = 11 350 kg/m³`, `Z = 82`, `A = 207.2 g/mol`.
    pub fn lead() -> Self {
        HomogeneousMedium {
            name: "lead".to_string(),
            density_kg_m3: 11_350.0,
            effective_z: 82.0,
            effective_a: 207.2,
        }
    }

    /// ICRU soft tissue: `ρ = 1060 kg/m³`, `Z_eff = 7.64`.
    ///
    /// `A_eff = 13.86 g/mol` is set for electron-density consistency
    /// (`A_eff = Z_eff / (Z/A)` with the ICRU-44 soft-tissue `Z/A ≈ 0.5512`);
    /// see [`HomogeneousMedium::water`]. The previous value `12.3` overstated
    /// the electron density (and Compton attenuation) by ~13 %.
    pub fn soft_tissue() -> Self {
        HomogeneousMedium {
            name: "soft_tissue".to_string(),
            density_kg_m3: 1060.0,
            effective_z: 7.64,
            effective_a: 13.86,
        }
    }
}

impl Medium for HomogeneousMedium {
    fn density_kg_m3(&self) -> f64 {
        self.density_kg_m3
    }

    fn effective_z(&self) -> f64 {
        self.effective_z
    }

    fn effective_a(&self) -> f64 {
        self.effective_a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_number_density_is_physical() {
        let n = HomogeneousMedium::water().number_density();
        // Water molecular number density ≈ 10^28 m⁻³ scale (using effective A).
        assert!(n > 1e28 && n < 1e29);
    }

    #[test]
    fn lead_is_denser_than_water() {
        assert!(
            HomogeneousMedium::lead().density_kg_m3() > HomogeneousMedium::water().density_kg_m3()
        );
    }

    #[test]
    fn medium_is_object_safe() {
        let m: Box<dyn Medium> = Box::new(HomogeneousMedium::air());
        assert!(m.effective_z() > 0.0);
    }
}
