//! Regular 3-D voxel grid for spatial energy-deposition tallies.
//!
//! A [`VoxelGrid`] partitions a rectangular region into cubic voxels and
//! accumulates the energy deposited in each one. It converts that tally into an
//! absorbed dose in **gray** (J/kg) and collapses it into a depth-dose profile
//! (the percentage depth dose, PDD).
//!
//! # Local-deposition model (valid at ~100 keV)
//!
//! The engine deposits the full energy of each photon interaction **locally**,
//! at the interaction point, instead of transporting the liberated electron.
//! At 100 keV the continuous-slowing-down range of a water electron is below
//! ~0.2 mm — far smaller than the 1 cm voxels used here — so essentially no
//! energy leaks between voxels and the collision kerma equals the absorbed
//! dose. Local deposition is therefore correct at this energy. (Electron
//! transport, which matters for the surface build-up region at higher
//! energies, is a deliberate future enhancement.)
//!
//! # Absolute vs. relative dose
//!
//! Fixing the voxel size fixes the voxel mass, so the **dose per voxel in gray
//! is unambiguous** (unlike a point beam with no cross-sectional area). The
//! absolute value returned by [`VoxelGrid::dose_gy`] is "per the `N` simulated
//! histories"; to compare against a reference Monte Carlo, normalize by the
//! number of primaries or use the dimensionless relative PDD
//! ([`VoxelGrid::relative_pdd`]).

use crate::constants::MEV_TO_JOULE;
use crate::error::PhysicsError;
use crate::medium::Medium;
use crate::particle::Position;

/// A regular grid of cubic voxels accumulating deposited energy (MeV).
///
/// The linear index of voxel `(ix, iy, iz)` is `ix + nx·(iy + ny·iz)`, i.e.
/// `x` varies fastest and whole `z`-slices are contiguous in memory (which
/// makes the depth profile a sequence of contiguous-slice sums).
#[derive(Debug, Clone)]
pub struct VoxelGrid {
    /// Minimum corner of the grid `[x, y, z]` (m); voxel `(0,0,0)` starts here.
    origin: [f64; 3],
    /// Edge length of every (cubic) voxel (m).
    voxel_size_m: f64,
    /// Number of voxels along each axis `[nx, ny, nz]`.
    dims: [usize; 3],
    /// Deposited energy per voxel (MeV), length `nx·ny·nz`, `x`-fastest.
    energy_mev: Vec<f64>,
    /// Energy deposited at positions outside the grid (MeV).
    overflow_mev: f64,
}

impl VoxelGrid {
    /// Build a grid with the given `origin` (minimum corner, m), cubic
    /// `voxel_size_m`, and `dims = [nx, ny, nz]`.
    ///
    /// For the depth-dose experiment: `origin = [-0.05, -0.05, 0.0]`,
    /// `voxel_size_m = 0.01`, `dims = [10, 10, 10]` (a 10 cm cube of 1 cm
    /// voxels).
    ///
    /// # Errors
    ///
    /// [`PhysicsError::SimulationError`] if `voxel_size_m` is not a finite
    /// positive number, or if any entry of `dims` is zero.
    pub fn new(
        origin: [f64; 3],
        voxel_size_m: f64,
        dims: [usize; 3],
    ) -> Result<Self, PhysicsError> {
        if voxel_size_m <= 0.0 || !voxel_size_m.is_finite() {
            return Err(PhysicsError::SimulationError {
                message: format!("voxel_size_m must be finite and positive, got {voxel_size_m}"),
            });
        }
        if dims.iter().any(|&d| d == 0) {
            return Err(PhysicsError::SimulationError {
                message: format!("voxel grid dims must all be > 0, got {dims:?}"),
            });
        }
        let n = dims[0] * dims[1] * dims[2];
        Ok(VoxelGrid {
            origin,
            voxel_size_m,
            dims,
            energy_mev: vec![0.0; n],
            overflow_mev: 0.0,
        })
    }

    /// Number of voxels along each axis `[nx, ny, nz]`.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Minimum corner of the grid `[x, y, z]` (m): the lower bound of voxel
    /// `(0, 0, 0)`. Combine with [`voxel_size_m`](Self::voxel_size_m) to recover
    /// any voxel's spatial extent, e.g. the centre of voxel `i` along an axis is
    /// `origin[axis] + (i + 0.5) · voxel_size_m`.
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Edge length of each cubic voxel (m).
    pub fn voxel_size_m(&self) -> f64 {
        self.voxel_size_m
    }

    /// Per-voxel deposited energy (MeV), in `x`-fastest linear order.
    pub fn energy_mev(&self) -> &[f64] {
        &self.energy_mev
    }

    /// Total energy deposited inside the grid (MeV).
    pub fn total_energy_mev(&self) -> f64 {
        self.energy_mev.iter().sum()
    }

    /// Energy deposited outside the grid (MeV).
    ///
    /// Together with [`total_energy_mev`](Self::total_energy_mev) this conserves
    /// the simulated deposit: `total_energy_mev + overflow_mev` equals the sum
    /// of every history's `total_deposit_mev` (escaping particles deposit
    /// nothing and so contribute to neither side).
    pub fn overflow_mev(&self) -> f64 {
        self.overflow_mev
    }

    /// Linear index of the voxel containing `p`, or `None` if `p` is outside
    /// the grid.
    ///
    /// Each coordinate maps to `floor((coord − origin) / voxel_size)`; the
    /// point is outside if any axis index is negative or `≥` the axis size.
    pub fn voxel_index(&self, p: Position) -> Option<usize> {
        let [nx, ny, _nz] = self.dims;
        let ix = self.axis_index(p.0[0], 0)?;
        let iy = self.axis_index(p.0[1], 1)?;
        let iz = self.axis_index(p.0[2], 2)?;
        Some(ix + nx * (iy + ny * iz))
    }

    /// Index along one axis, or `None` if the coordinate falls outside.
    fn axis_index(&self, coord: f64, axis: usize) -> Option<usize> {
        let local = (coord - self.origin[axis]) / self.voxel_size_m;
        if local < 0.0 {
            return None;
        }
        // Float→int casts saturate in Rust, so a far-outside coordinate yields a
        // huge index that the bound check below rejects (no overflow / UB).
        let idx = local.floor() as usize;
        if idx >= self.dims[axis] {
            return None;
        }
        Some(idx)
    }

    /// Accumulate `energy_mev` into the voxel containing `p`; if `p` is outside
    /// the grid the energy is added to the overflow accumulator instead.
    pub fn score(&mut self, p: Position, energy_mev: f64) {
        match self.voxel_index(p) {
            Some(i) => self.energy_mev[i] += energy_mev,
            None => self.overflow_mev += energy_mev,
        }
    }

    /// Depth-dose profile: deposited energy summed over `x` and `y` for each
    /// `z`-slice (MeV), length `nz`.
    ///
    /// Collapsing the transverse plane makes the profile robust to a pencil
    /// beam that happens to travel along a voxel edge in `x`/`y`. Because whole
    /// `z`-slices are contiguous in the linear layout, each slice is a simple
    /// contiguous-range sum.
    pub fn depth_profile_mev(&self) -> Vec<f64> {
        let [nx, ny, nz] = self.dims;
        let slice_len = nx * ny;
        (0..nz)
            .map(|iz| {
                let start = iz * slice_len;
                self.energy_mev[start..start + slice_len].iter().sum()
            })
            .collect()
    }

    /// Mass of a single voxel in `medium` (kg): `ρ · voxel_size³`.
    ///
    /// A 1 cm³ voxel of water (`ρ = 1000 kg/m³`) has mass
    /// `1000 · (0.01)³ = 1e-3 kg` (1 g), as expected.
    pub fn voxel_mass_kg(&self, medium: &dyn Medium) -> f64 {
        medium.density_kg_m3() * self.voxel_size_m.powi(3)
    }

    /// Absorbed dose per voxel in `medium` (gray = J/kg).
    ///
    /// For voxel `i`: `energy_mev[i] · MEV_TO_JOULE / voxel_mass_kg`. The value
    /// is the dose accumulated over all simulated histories; see the
    /// module-level note on absolute vs. relative dose.
    pub fn dose_gy(&self, medium: &dyn Medium) -> Vec<f64> {
        let mass = self.voxel_mass_kg(medium);
        self.energy_mev
            .iter()
            .map(|&e| e * MEV_TO_JOULE / mass)
            .collect()
    }

    /// Depth-dose curve in gray: absorbed dose per `z`-slice, length `nz`.
    ///
    /// A slice contains `nx · ny` voxels, so its mass is
    /// `ρ · (nx · ny) · voxel_size³`; the slice dose is the slice's deposited
    /// energy (in joules) divided by that mass.
    pub fn depth_dose_gy(&self, medium: &dyn Medium) -> Vec<f64> {
        let [nx, ny, _nz] = self.dims;
        let slice_mass = medium.density_kg_m3() * (nx * ny) as f64 * self.voxel_size_m.powi(3);
        self.depth_profile_mev()
            .iter()
            .map(|&e| e * MEV_TO_JOULE / slice_mass)
            .collect()
    }

    /// Relative percentage depth dose (dimensionless): the depth profile
    /// normalized to its maximum, length `nz`.
    ///
    /// This is the curve compared directly against a reference Monte Carlo,
    /// since it cancels the absolute (per-history) normalization. Returns all
    /// zeros if no energy was deposited.
    pub fn relative_pdd(&self) -> Vec<f64> {
        let profile = self.depth_profile_mev();
        let max = profile.iter().copied().fold(0.0_f64, f64::max);
        if max <= 0.0 {
            return vec![0.0; profile.len()];
        }
        profile.iter().map(|&e| e / max).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;

    fn experiment_grid() -> VoxelGrid {
        VoxelGrid::new([-0.05, -0.05, 0.0], 0.01, [10, 10, 10]).unwrap()
    }

    #[test]
    fn new_rejects_nonpositive_voxel_size() {
        assert!(VoxelGrid::new([0.0; 3], 0.0, [1, 1, 1]).is_err());
        assert!(VoxelGrid::new([0.0; 3], -0.01, [1, 1, 1]).is_err());
    }

    #[test]
    fn new_rejects_zero_dims() {
        assert!(VoxelGrid::new([0.0; 3], 0.01, [0, 1, 1]).is_err());
        assert!(VoxelGrid::new([0.0; 3], 0.01, [1, 1, 0]).is_err());
    }

    #[test]
    fn voxel_index_maps_known_positions() {
        let g = experiment_grid();
        // Just inside the origin corner → voxel (0,0,0) = index 0.
        assert_eq!(g.voxel_index(Position([-0.05, -0.05, 0.0])), Some(0));
        // Centre of voxel (2, 3, 4): origin + (i+0.5)·size.
        let p = Position([-0.05 + 2.5 * 0.01, -0.05 + 3.5 * 0.01, 4.5 * 0.01]);
        assert_eq!(g.voxel_index(p), Some(2 + 10 * (3 + 10 * 4)));
    }

    #[test]
    fn voxel_index_is_none_outside_grid() {
        let g = experiment_grid();
        assert_eq!(g.voxel_index(Position([0.0, 0.0, -0.001])), None);
        assert_eq!(g.voxel_index(Position([0.0, 0.0, 0.10001])), None);
        assert_eq!(g.voxel_index(Position([0.06, 0.0, 0.05])), None);
    }

    #[test]
    fn origin_dims_and_voxel_size_round_trip() {
        let g = experiment_grid();
        assert_eq!(g.origin(), [-0.05, -0.05, 0.0]);
        assert_eq!(g.dims(), [10, 10, 10]);
        assert!((g.voxel_size_m() - 0.01).abs() < 1e-15);
    }

    #[test]
    fn score_inside_grid_accumulates_in_correct_voxel() {
        let mut g = experiment_grid();
        let p = Position([0.0, 0.0, 0.025]); // voxel (5, 5, 2)
        g.score(p, 0.04);
        g.score(p, 0.06);
        let idx = g.voxel_index(p).unwrap();
        assert!((g.energy_mev()[idx] - 0.10).abs() < 1e-12);
        assert_eq!(g.overflow_mev(), 0.0);
    }

    #[test]
    fn score_outside_grid_goes_to_overflow() {
        let mut g = experiment_grid();
        g.score(Position([0.0, 0.0, 0.5]), 0.07); // well past the exit face
        assert_eq!(g.total_energy_mev(), 0.0);
        assert!((g.overflow_mev() - 0.07).abs() < 1e-12);
    }

    #[test]
    fn depth_profile_collapses_transverse_plane() {
        let mut g = experiment_grid();
        // Two deposits in slice z = 2, at different (x, y); one in slice z = 5.
        g.score(Position([-0.04, -0.04, 0.025]), 1.0);
        g.score(Position([0.04, 0.04, 0.025]), 2.0);
        g.score(Position([0.0, 0.0, 0.055]), 4.0);
        let profile = g.depth_profile_mev();
        assert_eq!(profile.len(), 10);
        assert!((profile[2] - 3.0).abs() < 1e-12);
        assert!((profile[5] - 4.0).abs() < 1e-12);
        assert_eq!(profile[0], 0.0);
    }

    #[test]
    fn voxel_mass_of_one_cubic_cm_water_is_one_gram() {
        let g = experiment_grid();
        let water = HomogeneousMedium::water();
        assert!((g.voxel_mass_kg(&water) - 1e-3).abs() < 1e-12);
    }

    #[test]
    fn dose_gy_is_energy_in_joules_over_voxel_mass() {
        let mut g = experiment_grid();
        let water = HomogeneousMedium::water();
        let p = Position([0.0, 0.0, 0.025]);
        let e_mev = 0.5;
        g.score(p, e_mev);
        let idx = g.voxel_index(p).unwrap();
        let expected = e_mev * MEV_TO_JOULE / 1e-3; // E_J / 1 g
        assert!((g.dose_gy(&water)[idx] - expected).abs() < 1e-18);
    }

    #[test]
    fn relative_pdd_is_normalized_to_unity() {
        let mut g = experiment_grid();
        g.score(Position([0.0, 0.0, 0.005]), 5.0); // slice 0
        g.score(Position([0.0, 0.0, 0.025]), 2.0); // slice 2
        let pdd = g.relative_pdd();
        assert!((pdd[0] - 1.0).abs() < 1e-12);
        assert!((pdd[2] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn relative_pdd_is_all_zero_for_empty_grid() {
        let g = experiment_grid();
        assert!(g.relative_pdd().iter().all(|&v| v == 0.0));
    }
}
