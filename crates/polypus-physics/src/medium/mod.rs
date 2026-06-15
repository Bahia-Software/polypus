//! Material definitions through which particles are transported.

pub mod simple;

/// A material through which particles are transported.
///
/// Implementors describe a homogeneous volume; geometry is handled by the
/// Monte Carlo engine externally. Adding a new material requires only a new
/// implementor of this trait — no engine changes.
///
/// The trait is object-safe, so `Box<dyn Medium>` and `&dyn Medium` are usable.
pub trait Medium: Send + Sync + std::fmt::Debug {
    /// Mass density `ρ` (kg/m³).
    fn density_kg_m3(&self) -> f64;

    /// Effective atomic number `Z` (may be non-integer for compounds).
    fn effective_z(&self) -> f64;

    /// Effective atomic mass `A` (g/mol).
    fn effective_a(&self) -> f64;

    /// Atom number density `n = ρ·N_A / A` (atoms/m³).
    ///
    /// Default implementation: converts the mass density from kg/m³ to g/m³,
    /// multiplies by Avogadro's number, and divides by the molar mass `A`.
    fn number_density(&self) -> f64 {
        (self.density_kg_m3() * 1e3) // kg/m³ → g/m³
            * crate::constants::AVOGADRO
            / (self.effective_a()) // g/mol
    }
}

pub use simple::HomogeneousMedium;
