//! Experiment: a 100 kVp X-ray photon beam transported through water.
//!
//! This mirrors the classic question "can we simulate the cross-section of a
//! 100 kVp photon?" — but instead of a single 100 keV photon it uses the
//! realistic *polyenergetic* bremsstrahlung spectrum of a 100 kVp tube
//! (Kramers' law), sampled per primary by the Monte Carlo engine.
//!
//! It demonstrates the full classical pipeline of `polypus-physics`:
//!   1. Per-process photon cross-sections vs energy (process competition).
//!   2. Sampling the 100 kVp source spectrum (the bremsstrahlung shape).
//!   3. Monte Carlo transport of the whole beam through a water phantom.
//!   4. A monoenergetic beam for comparison.
//!   5. 3-D voxel dosimetry: a monoenergetic point-source beam and, separately,
//!      a polyenergetic (Kramers) point-source beam, both through the same
//!      finite 40 × 40 × 20 cm water phantom (5 mm voxels), each giving a
//!      depth-dose profile (PDD).
//!   6. Three PNG plots for water — the ENDF-6 mass attenuation coefficient
//!      and the two central-axis PDDs (monoenergetic, polyenergetic) — using
//!      the same library plotting utilities exercised by the crate's own
//!      `monte_carlo` tests (`mass_attenuation_coefficients_plots::plot_compound`,
//!      `voxel_plots::plot_relative_pdd`). Requires `--features plotters`.
//!   7. The numerical data behind all three plots, exported to CSV, for
//!      quantitative comparison against a validated reference Monte Carlo
//!      (Geant4 / EGSnrc / PENELOPE). Requires `--features csv-export`.
//!
//! Run it with:
//! ```text
//! cargo run -p polypus-physics --example photon_100kvp_water --release \
//!     --features plotters,csv-export
//! ```

use polypus_physics::error::PhysicsError;
use polypus_physics::interactions::photon::mass_attenuation_coefficients::{
    mu_m_for_compound, CompoundResult,
};
use polypus_physics::interactions::photon::PhotonInteractionModel;
use polypus_physics::interactions::InteractionModel;
use polypus_physics::medium::{CompoundMedium, Medium, PhotonChannel};
use polypus_physics::monte_carlo::{
    beam::DivergentBeam,
    spectrum::{EnergySpectrum, KramersSpectrum, Monoenergetic},
    Geometry, MonteCarloEngine, RunConfig, SimulationResult, VoxelGrid,
};
use polypus_physics::particle::photon::Photon;
use polypus_physics::particle::Position;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::{Path, PathBuf};

/// ENDF-6 MT=501: total photon cross-section (the standard "mass attenuation
/// coefficient" curve, summing every reaction channel).
const MT_TOTAL: u32 = 501;

/// X-ray tube peak potential (kV) for the polyenergetic sections (the Kramers'
/// law bremsstrahlung spectrum) → endpoint energy TUBE_KVP keV.
const TUBE_KVP: f64 = 100.0;
/// Inherent-filtration low-energy cutoff (keV) for the TUBE_KVP spectrum.
const FILTER_CUTOFF_KEV: f64 = 10.0;
/// Photon energy (keV) for the monoenergetic-beam sections — a single exact
/// energy, unlike TUBE_KVP (a tube's peak potential, which only bounds the
/// *endpoint* of a spread spectrum).
const ENERGY_KEV: f64 = 100.0;
/// Number of primary histories to transport.
const N_HISTORIES: usize = 20_000;
/// Master RNG seed (reproducible runs).
const SEED: u64 = 2026;
/// Energy-grid resolution for the ENDF-6-backed [`CompoundMedium`]s built by
/// this example (mirrors the crate's own tests, e.g.
/// `CompoundMedium::new("H2O", 1000.0, 5000)`).
const N_POINTS: usize = 5000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==================================================================");
    println!(" Polypus-physics — {TUBE_KVP:.0} kVp photon beam in water");
    println!("==================================================================\n");

    let water = CompoundMedium::new("H2O", 1000.0, N_POINTS)?;
    let model = PhotonInteractionModel;

    cross_section_table(&model, &water)?;
    spectrum_histogram()?;
    let spectrum_result = transport_spectrum(&water)?;
    let mono_result = transport_monoenergetic(&water)?;
    let dose_grid = transport_voxel_dose(&water)?;
    let dose_grid_kramers = transport_voxel_dose_kramers(&water)?;

    summarize(&spectrum_result, &mono_result);

    // Linear attenuation coefficient μ (m⁻¹) of ENERGY_KEV photons in water,
    // used in the results CSV header.
    let mu_mono = model.total_cross_section_per_m(
        &Photon,
        &Photon::state_along_z(ENERGY_KEV * 1e-3),
        &water,
    )?;

    // ENDF-6 total mass attenuation coefficient of water, shared by the plot
    // and the CSV export below.
    let mu_result = mu_m_for_compound("H2O", MT_TOTAL, N_POINTS)?;

    generate_plots(&mu_result, &dose_grid, &dose_grid_kramers)?;
    save_results(&water, &mu_result, &dose_grid, &dose_grid_kramers, mu_mono)?;
    Ok(())
}

/// Section 1 — per-process mass attenuation coefficients (from ENDF-6 data)
/// and the total linear attenuation coefficient μ (m⁻¹) and mean free path
/// (mm) in water.
fn cross_section_table(
    model: &PhotonInteractionModel,
    water: &CompoundMedium,
) -> Result<(), PhysicsError> {
    println!(
        "[1] Photon mass attenuation coefficients in water (H2O, rho = {:.1} kg/m³)",
        water.density_kg_m3()
    );
    println!(
        "    {:>8} | {:>14} {:>14} {:>14} | {:>10} {:>10}",
        "E (keV)", "photoel cm²/g", "compton cm²/g", "pair cm²/g", "μ (1/m)", "mfp (mm)"
    );
    println!("    {}", "-".repeat(86));

    for &e_kev in &[10.0_f64, 30.0, 50.0, 100.0, 500.0, 1000.0] {
        let e_mev = e_kev * 1e-3;
        let tau = water.mu_m_cm2_g(PhotonChannel::Photoelectric, e_mev);
        let sigma = water.mu_m_cm2_g(PhotonChannel::Incoherent, e_mev);
        let kappa = water.mu_m_cm2_g(PhotonChannel::PairProductionTotal, e_mev);

        let state = Photon::state_along_z(e_mev);
        let mu = model.total_cross_section_per_m(&Photon, &state, water)?;
        let mfp_mm = if mu > 0.0 { 1e3 / mu } else { f64::INFINITY };

        println!(
            "    {e_kev:>8.0} | {tau:>14.3e} {sigma:>14.3e} {kappa:>14.3e} | {mu:>10.3} {mfp_mm:>10.3}"
        );
    }
    println!("    → at 100 keV Compton dominates; below ~30 keV photoelectric grows fast.\n");
    Ok(())
}

/// Section 2 — sample the 100 kVp Kramers spectrum and draw an ASCII histogram
/// of the sampled photon energies (the bremsstrahlung shape, peaking low).
fn spectrum_histogram() -> Result<(), PhysicsError> {
    let spectrum = KramersSpectrum::from_kvp(TUBE_KVP, FILTER_CUTOFF_KEV)?;
    let mut rng = StdRng::seed_from_u64(SEED);

    const N_SAMPLES: usize = 200_000;
    const N_BINS: usize = 18;
    let lo = spectrum.min_energy_mev();
    let hi = spectrum.max_energy_mev();
    let width = (hi - lo) / N_BINS as f64;

    let mut bins = [0usize; N_BINS];
    let mut energy_sum = 0.0;
    for _ in 0..N_SAMPLES {
        let e = spectrum.sample_energy_mev(&mut rng);
        energy_sum += e;
        let idx = (((e - lo) / width) as usize).min(N_BINS - 1);
        bins[idx] += 1;
    }
    let mean_kev = (energy_sum / N_SAMPLES as f64) * 1e3;

    println!("[2] Sampled {N_SAMPLES} photons from the {TUBE_KVP:.0} kVp spectrum");
    let peak = bins.iter().copied().max().unwrap_or(1).max(1);
    for (i, &count) in bins.iter().enumerate() {
        let e_lo_kev = (lo + i as f64 * width) * 1e3;
        let bar = count * 50 / peak;
        println!("    {e_lo_kev:>5.0} keV | {} {count}", "█".repeat(bar));
    }
    println!("    → mean photon energy ≈ {mean_kev:.1} keV (well below the 100 keV endpoint)\n");
    Ok(())
}

/// Section 3 — Monte Carlo transport of the full 100 kVp beam through water.
fn transport_spectrum(water: &CompoundMedium) -> Result<SimulationResult, PhysicsError> {
    let spectrum = KramersSpectrum::from_kvp(TUBE_KVP, FILTER_CUTOFF_KEV)?;
    let engine = MonteCarloEngine::new(
        Photon,
        water.clone(),
        PhotonInteractionModel,
        RunConfig {
            n_histories: N_HISTORIES,
            seed: SEED,
            ..Default::default()
        },
    );

    let mut rng = StdRng::seed_from_u64(SEED);
    let result = engine.run_with_spectrum(
        &spectrum,
        Position([0.0, 0.0, 0.0]),
        [0.0, 0.0, 1.0],
        &mut rng,
    )?;

    println!("[3] Transported {N_HISTORIES} primary histories ({TUBE_KVP:.0} kVp beam in water)");
    report(&result);
    println!();
    Ok(result)
}

/// Section 4 — monoenergetic ENERGY_KEV beam, for comparison with the spectrum.
fn transport_monoenergetic(water: &CompoundMedium) -> Result<SimulationResult, PhysicsError> {
    let mono = Monoenergetic::new(ENERGY_KEV * 1e-3)?;
    let engine = MonteCarloEngine::new(
        Photon,
        water.clone(),
        PhotonInteractionModel,
        RunConfig {
            n_histories: N_HISTORIES,
            seed: SEED,
            ..Default::default()
        },
    );

    let mut rng = StdRng::seed_from_u64(SEED);
    let result =
        engine.run_with_spectrum(&mono, Position([0.0, 0.0, 0.0]), [0.0, 0.0, 1.0], &mut rng)?;

    println!(
        "[4] Transported {N_HISTORIES} primary histories (monoenergetic {ENERGY_KEV:.0} keV beam)"
    );
    report(&result);
    println!();
    Ok(result)
}

/// Number of primary histories for the voxel-dosimetry beams, and their RNG
/// seeds. The phantom/beam geometry (not the history count, which is much
/// higher here for a cleaner plot) is shared with
/// `source_and_voxels_conserve_energy_for_divergent_and_parallel_beams` in
/// `monte_carlo::tests`, so the two stay comparable.
const PDD_N_HISTORIES: usize = 400_000;
const PDD_SEED: u64 = 456;
const PDD_KRAMERS_N_HISTORIES: usize = 400_000;
const PDD_KRAMERS_SEED: u64 = 42;
/// Inherent-filtration cutoff (keV) for the Kramers PDD beam specifically —
/// distinct from `FILTER_CUTOFF_KEV` used by the spectrum-shape sections
/// above.
const PDD_KRAMERS_FILTER_CUTOFF_KEV: f64 = 15.0;
/// Shared phantom/grid geometry for both voxel-dosimetry beams: a 40 × 40 cm
/// cross-section, 20 cm deep water phantom, entrance face at z = 0, tallied
/// on an 80 × 80 × 40 grid of 5 mm cubic voxels.
const PDD_PHANTOM_MIN: [f64; 3] = [-0.20, -0.20, 0.0];
const PDD_PHANTOM_MAX: [f64; 3] = [0.20, 0.20, 0.20];
const PDD_VOXEL_SIZE_M: f64 = 0.005;
const PDD_GRID_DIMS: [usize; 3] = [80, 80, 40];
/// Central-axis window (voxels on each side) for the monoenergetic PDD.
const PDD_WINDOW: usize = 1;
/// Central-axis window (voxels on each side) for the Kramers PDD — wider,
/// since the softer/broader Kramers spectrum scatters more off-axis.
const PDD_KRAMERS_WINDOW: usize = 2;

/// Section 5 — 3-D voxel dosimetry. Transport a monoenergetic ENERGY_KEV
/// point-source beam (10 × 10 cm field at the surface, 10 cm source-to-surface
/// distance) through the phantom described by the `PDD_*` constants above.
///
/// The per-voxel mass is fixed (a 5 mm cubic voxel of water = 0.125 g), so the
/// absorbed dose in gray is unambiguous (see [`VoxelGrid::dose_gy`]). Prints
/// the central-axis depth-dose profile (PDD) and returns the grid for
/// plotting.
///
/// Local deposition is valid here: at 100 keV the secondary-electron range in
/// water is < 0.2 mm, far below the 5 mm voxel, so no energy leaks between
/// voxels and collision kerma ≈ absorbed dose.
fn transport_voxel_dose(water: &CompoundMedium) -> Result<VoxelGrid, PhysicsError> {
    let beam = DivergentBeam {
        source_to_surface_distance_m: 0.10,
        field_side_m: 0.10,
        energy_source: Box::new(Monoenergetic::new(ENERGY_KEV * 1e-3)?),
    };
    let engine = MonteCarloEngine::new(
        Photon,
        water.clone(),
        PhotonInteractionModel,
        RunConfig {
            n_histories: PDD_N_HISTORIES,
            seed: PDD_SEED,
            ..Default::default()
        },
    )
    .with_geometry(Geometry::Box {
        min: PDD_PHANTOM_MIN,
        max: PDD_PHANTOM_MAX,
    });

    let grid = VoxelGrid::new(PDD_PHANTOM_MIN, PDD_VOXEL_SIZE_M, PDD_GRID_DIMS)?;
    let mut rng = StdRng::seed_from_u64(PDD_SEED);
    let (_result, grid) = engine.run_with_source_and_voxels(&beam, grid, &mut rng)?;

    println!(
        "[5] Voxel dosimetry: {PDD_N_HISTORIES} × {ENERGY_KEV:.0} keV photons in a 40×40×20 cm water phantom"
    );
    print_pdd_table(&grid, PDD_WINDOW);
    println!("    → PDD falls more slowly than exp(-μz): forward Compton scatter (≈85 % of each");
    println!("      {ENERGY_KEV:.0} keV interaction's energy) is carried downstream and deposited deeper.\n");
    Ok(grid)
}

/// Section 5b — same phantom/grid as [`transport_voxel_dose`], but with a
/// polyenergetic TUBE_KVP Kramers spectrum beam instead of a single energy.
fn transport_voxel_dose_kramers(water: &CompoundMedium) -> Result<VoxelGrid, PhysicsError> {
    let spectrum = KramersSpectrum::from_kvp(TUBE_KVP, PDD_KRAMERS_FILTER_CUTOFF_KEV)?;
    let beam = DivergentBeam {
        source_to_surface_distance_m: 0.10,
        field_side_m: 0.10,
        energy_source: Box::new(spectrum),
    };
    let engine = MonteCarloEngine::new(
        Photon,
        water.clone(),
        PhotonInteractionModel,
        RunConfig {
            n_histories: PDD_KRAMERS_N_HISTORIES,
            seed: PDD_KRAMERS_SEED,
            ..Default::default()
        },
    )
    .with_geometry(Geometry::Box {
        min: PDD_PHANTOM_MIN,
        max: PDD_PHANTOM_MAX,
    });

    let grid = VoxelGrid::new(PDD_PHANTOM_MIN, PDD_VOXEL_SIZE_M, PDD_GRID_DIMS)?;
    let mut rng = StdRng::seed_from_u64(PDD_KRAMERS_SEED);
    let (_result, grid) = engine.run_with_source_and_voxels(&beam, grid, &mut rng)?;

    println!(
        "[5b] Voxel dosimetry: {PDD_KRAMERS_N_HISTORIES} × {TUBE_KVP:.0} kVp (Kramers) photons in a 40×40×20 cm water phantom"
    );
    print_pdd_table(&grid, PDD_KRAMERS_WINDOW);
    println!("    → the polyenergetic PDD falls faster near the surface than the monoenergetic");
    println!("      one: the softer low-energy tail of the spectrum is absorbed shallower.\n");
    Ok(grid)
}

/// Print the `z (cm) | deposit MeV | rel. PDD` table shared by both
/// voxel-dosimetry sections, for the given central-axis `window`.
fn print_pdd_table(grid: &VoxelGrid, window: usize) {
    let profile = grid.depth_profile_mev(window);
    let pdd = grid.relative_pdd(window);
    println!(
        "    {:>7} | {:>12} | {:>10}",
        "z (cm)", "deposit MeV", "rel. PDD"
    );
    println!("    {}", "-".repeat(38));
    let voxel_cm = grid.voxel_size_m() * 100.0;
    for iz in 0..grid.dims()[2] {
        let z_lo = iz as f64 * voxel_cm;
        let z_hi = z_lo + voxel_cm;
        let label = format!("{z_lo:.1}-{z_hi:.1}");
        println!("    {label:>7} | {:>12.2} | {:>10.4}", profile[iz], pdd[iz]);
    }
}

/// Print per-run statistics: mean deposit, standard error, mean interactions.
fn report(result: &SimulationResult) {
    let n = result.histories.len().max(1) as f64;
    let std_err = result.variance_deposit_mev2.sqrt() / n.sqrt();
    let mean_interactions = result
        .histories
        .iter()
        .map(|h| h.track.len().saturating_sub(1))
        .sum::<usize>() as f64
        / n;

    println!(
        "    mean deposit       = {:.4} ± {:.4} MeV/history",
        result.mean_deposit_mev, std_err
    );
    println!(
        "    deposit std-dev    = {:.4} MeV",
        result.variance_deposit_mev2.sqrt()
    );
    println!("    mean interactions  = {mean_interactions:.2} per history");
}

/// Section 5 — side-by-side comparison of the two beams.
fn summarize(spectrum: &SimulationResult, mono: &SimulationResult) {
    println!("==================================================================");
    println!(" Summary");
    println!("==================================================================");
    println!(
        "    {:<22} {:>14} {:>14}",
        "", "100 kVp beam", "100 keV mono"
    );
    println!(
        "    {:<22} {:>14.4} {:>14.4}",
        "mean deposit (MeV)", spectrum.mean_deposit_mev, mono.mean_deposit_mev
    );
    println!(
        "    {:<22} {:>14.4} {:>14.4}",
        "deposit std-dev (MeV)",
        spectrum.variance_deposit_mev2.sqrt(),
        mono.variance_deposit_mev2.sqrt()
    );
    println!("\n    The polyenergetic beam deposits less per history than a pure");
    println!("    100 keV beam: most tube photons are far softer than the endpoint,");
    println!("    so the average energy available per primary is lower.");
}

// ─────────────────────────────────────────────────────────────────────────
// Numerical-result export (CSV, for comparison against a reference MC)
// ─────────────────────────────────────────────────────────────────────────

/// Section 6 — persist the numerical results to CSV under
/// `<crate>/examples/results/`, so they can later be loaded and compared
/// quantitatively against a validated reference Monte Carlo (Geant4 / EGSnrc /
/// PENELOPE). These are the data behind the plots from [`generate_plots`]:
///   * `depth_dose_100kev_water.csv` — the monoenergetic central-axis PDD.
///   * `depth_dose_100kvp_kramers_water.csv` — the polyenergetic (Kramers)
///     central-axis PDD.
///   * `mass_attenuation_water.csv` — the ENDF-6 total mass attenuation
///     coefficient of water vs energy, via `write_compound_csv`. Requires
///     `--features csv-export`; skipped (with a note) when that feature is
///     disabled.
///
/// Each PDD file begins with a `#`-prefixed metadata block (beam, geometry,
/// conservation totals) that pandas/NumPy skip with `comment='#'`; `rel_pdd`
/// is the dimensionless shape, robust to absolute normalization.
fn save_results(
    water: &CompoundMedium,
    mu_result: &CompoundResult,
    grid: &VoxelGrid,
    grid_kramers: &VoxelGrid,
    mu_mono: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("results");
    std::fs::create_dir_all(&dir)?;

    let header = results_header(
        water,
        grid,
        &format!(
            "monoenergetic {ENERGY_KEV:.0} keV point source, SSD=10cm, 10x10cm field at the surface"
        ),
        Some(ENERGY_KEV * 1e-3),
        PDD_N_HISTORIES,
        PDD_SEED,
        Some(mu_mono),
    );
    let p1 = write_depth_dose_csv(
        &dir,
        grid,
        &header,
        PDD_WINDOW,
        "depth_dose_100kev_water.csv",
    )?;

    let header_kramers = results_header(
        water,
        grid_kramers,
        &format!(
            "polyenergetic {TUBE_KVP:.0} kVp Kramers spectrum point source, SSD=10cm, \
             10x10cm field, {PDD_KRAMERS_FILTER_CUTOFF_KEV:.0} keV filter cutoff"
        ),
        None,
        PDD_KRAMERS_N_HISTORIES,
        PDD_KRAMERS_SEED,
        None,
    );
    let p1_kramers = write_depth_dose_csv(
        &dir,
        grid_kramers,
        &header_kramers,
        PDD_KRAMERS_WINDOW,
        "depth_dose_100kvp_kramers_water.csv",
    )?;

    println!("\n[results] CSV files written (load with pandas read_csv(..., comment='#')):");
    println!("    {}", p1.display());
    println!("    {}", p1_kramers.display());

    #[cfg(feature = "csv-export")]
    {
        use polypus_physics::interactions::photon::mass_attenuation_coefficients::write_compound_csv;
        let p2 = dir.join("mass_attenuation_water.csv");
        write_compound_csv("H2O", mu_result, &p2)?;
        println!("    {}", p2.display());
    }
    #[cfg(not(feature = "csv-export"))]
    {
        let _ = mu_result;
        println!("    (mass_attenuation_water.csv skipped — rebuild with --features csv-export)");
    }

    Ok(())
}

/// Build the shared `#`-prefixed metadata header carried by every results CSV.
///
/// Records everything needed to reproduce the run and to normalize the absolute
/// dose: beam, medium, geometry, voxel size, seed, μ, and the conservation
/// totals (`total_deposit_in_grid_mev + overflow_mev` equals the summed
/// per-history deposit). `beam_energy_mev` and `mu_per_m` are `None` for a
/// polyenergetic beam, which has no single energy or attenuation coefficient.
fn results_header(
    water: &CompoundMedium,
    grid: &VoxelGrid,
    beam_description: &str,
    beam_energy_mev: Option<f64>,
    n_histories: usize,
    seed: u64,
    mu_per_m: Option<f64>,
) -> String {
    let o = grid.origin();
    let [nx, ny, nz] = grid.dims();
    let s = grid.voxel_size_m();
    let cm = 100.0;
    let mut h = String::new();
    h.push_str("# polypus-physics depth-dose experiment — numerical results\n");
    h.push_str(&format!("# beam: {beam_description}\n"));
    h.push_str(&format!(
        "# medium: {} (rho = {:.1} kg/m^3)\n",
        water.formula,
        water.density_kg_m3()
    ));
    if let Some(e) = beam_energy_mev {
        h.push_str(&format!("# beam_energy_mev: {e:.6}\n"));
    }
    h.push_str(&format!("# n_primaries: {n_histories}\n"));
    h.push_str(&format!("# seed: {seed}\n"));
    h.push_str(&format!(
        "# geometry_box_min_cm: [{:.3}, {:.3}, {:.3}]\n",
        o[0] * cm,
        o[1] * cm,
        o[2] * cm
    ));
    h.push_str(&format!(
        "# geometry_box_max_cm: [{:.3}, {:.3}, {:.3}]\n",
        (o[0] + nx as f64 * s) * cm,
        (o[1] + ny as f64 * s) * cm,
        (o[2] + nz as f64 * s) * cm
    ));
    h.push_str(&format!("# voxel_size_cm: {:.3}\n", s * cm));
    h.push_str(&format!("# dims_nx_ny_nz: [{nx}, {ny}, {nz}]\n"));
    if let Some(mu) = mu_per_m {
        h.push_str(&format!("# mu_per_m: {mu:.4}\n"));
    }
    h.push_str(&format!(
        "# total_deposit_in_grid_mev: {:.6}\n",
        grid.total_energy_mev()
    ));
    h.push_str(&format!("# overflow_mev: {:.6}\n", grid.overflow_mev()));
    h.push_str("# note: dose_gy is absolute for n_primaries; dose_gy_per_primary = dose_gy / n_primaries\n");
    h.push_str("# note: local deposition is valid at 100 keV (e- range < 0.2 mm << 1 cm voxel)\n");
    h
}

/// Write the 1-D central-axis depth-dose profile (PDD), one row per `z`-slice,
/// using the given central-axis `window` (see [`VoxelGrid::relative_pdd`]).
fn write_depth_dose_csv(
    dir: &Path,
    grid: &VoxelGrid,
    header: &str,
    window: usize,
    filename: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let profile = grid.depth_profile_mev(window);
    let pdd = grid.relative_pdd(window);
    let s_cm = grid.voxel_size_m() * 100.0;
    let z0_cm = grid.origin()[2] * 100.0;

    let mut csv = String::from(header);
    csv.push_str("z_min_cm,z_max_cm,z_center_cm,deposit_mev,rel_pdd\n");
    for iz in 0..grid.dims()[2] {
        let z_lo = z0_cm + iz as f64 * s_cm;
        let z_hi = z_lo + s_cm;
        let z_c = z_lo + 0.5 * s_cm;
        csv.push_str(&format!(
            "{z_lo:.3},{z_hi:.3},{z_c:.3},{:.6e},{:.6}\n",
            profile[iz], pdd[iz],
        ));
    }

    let path = dir.join(filename);
    std::fs::write(&path, csv)?;
    Ok(path)
}

// ─────────────────────────────────────────────────────────────────────────
// Plot generation (PNG, via the crate's own `plotters`-backed utilities)
// ─────────────────────────────────────────────────────────────────────────

/// Section 7 — three PNG plots for water under `<crate>/examples/plots/`,
/// reusing the same library plotting utilities the crate's own `monte_carlo`
/// tests already exercise:
///   * `mass_attenuation_water.png` — the ENDF-6 total mass attenuation
///     coefficient vs energy
///     ([`mass_attenuation_coefficients_plots::plot_compound`]).
///   * `pdd_water.png` — the monoenergetic central-axis depth-dose profile
///     (PDD) ([`voxel_plots::plot_relative_pdd`]).
///   * `pdd_water_kramers.png` — the polyenergetic (Kramers) central-axis PDD,
///     same function.
///
/// All three are gated behind the `plotters` feature; skipped (with a note)
/// when it is disabled.
///
/// [`mass_attenuation_coefficients_plots::plot_compound`]: polypus_physics::interactions::photon::mass_attenuation_coefficients_plots::plot_compound
/// [`voxel_plots::plot_relative_pdd`]: polypus_physics::monte_carlo::voxel_plots::plot_relative_pdd
fn generate_plots(
    mu_result: &CompoundResult,
    dose_grid: &VoxelGrid,
    dose_grid_kramers: &VoxelGrid,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("plots");
    std::fs::create_dir_all(&dir)?;

    #[cfg(feature = "plotters")]
    {
        use polypus_physics::interactions::photon::mass_attenuation_coefficients_plots::plot_compound;
        use polypus_physics::monte_carlo::voxel_plots::plot_relative_pdd;

        let p1 = dir.join("mass_attenuation_water.png");
        plot_compound("H2O", mu_result, &p1)?;

        let pdd = dose_grid.relative_pdd(PDD_WINDOW);
        let p2 = dir.join("pdd_water.png");
        plot_relative_pdd(
            &pdd,
            dose_grid.voxel_size_m(),
            &format!("PDD - H2O, {ENERGY_KEV:.0} keV monoenergetic point source"),
            &p2,
        )?;

        let pdd_kramers = dose_grid_kramers.relative_pdd(PDD_KRAMERS_WINDOW);
        let p3 = dir.join("pdd_water_kramers.png");
        plot_relative_pdd(
            &pdd_kramers,
            dose_grid_kramers.voxel_size_m(),
            &format!("PDD - H2O, {TUBE_KVP:.0} kVp (Kramers) point source"),
            &p3,
        )?;

        println!("\n[plots] PNG files written:");
        println!("    {}", p1.display());
        println!("    {}", p2.display());
        println!("    {}", p3.display());
    }
    #[cfg(not(feature = "plotters"))]
    {
        let _ = (mu_result, dose_grid, dose_grid_kramers);
        println!("\n[plots] skipped (rebuild with --features plotters to generate PNGs)");
    }

    Ok(())
}
