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
//!   4. A monoenergetic 100 keV beam for comparison.
//!   5. 3-D voxel dosimetry: the 100 keV beam in a finite 10 × 10 × 10 cm water
//!      cube (1 cm voxels), giving a depth-dose profile (PDD) and dose in gray.
//!   6. Saved SVG plots — cross-sections, spectrum, attenuation, depth-dose,
//!      a 2-D dose heat-map of the beam-axis plane, and lateral dose profiles —
//!      for visual verification against references such as NIST XCOM.
//!   7. Numerical results exported to CSV (depth-dose PDD + full 3-D voxel
//!      tally) for quantitative comparison against a validated reference Monte
//!      Carlo (Geant4 / EGSnrc / PENELOPE).
//!
//! Run it with:
//! ```text
//! cargo run -p polypus-physics --example photon_100kvp_water --release
//! ```

use polypus_physics::constants::PAIR_PRODUCTION_THRESHOLD_MEV;
use polypus_physics::error::PhysicsError;
use polypus_physics::interactions::photon::{
    compton, pair_production, photoelectric, PhotonInteractionModel,
};
use polypus_physics::interactions::InteractionModel;
use polypus_physics::medium::{HomogeneousMedium, Medium};
use polypus_physics::monte_carlo::{
    spectrum::{EnergySpectrum, KramersSpectrum, Monoenergetic},
    Geometry, MonteCarloEngine, RunConfig, SimulationResult, VoxelGrid,
};
use polypus_physics::particle::photon::Photon;
use polypus_physics::particle::Position;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::{Path, PathBuf};

/// X-ray tube peak potential (kV) → endpoint energy 0.1 MeV.
const TUBE_KVP: f64 = 100.0;
/// Inherent-filtration low-energy cutoff (keV).
const FILTER_CUTOFF_KEV: f64 = 10.0;
/// Number of primary histories to transport.
const N_HISTORIES: usize = 20_000;
/// Master RNG seed (reproducible runs).
const SEED: u64 = 2026;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==================================================================");
    println!(" Polypus-physics — {TUBE_KVP:.0} kVp photon beam in water");
    println!("==================================================================\n");

    let water = HomogeneousMedium::water();
    let model = PhotonInteractionModel;

    cross_section_table(&model, &water)?;
    spectrum_histogram()?;
    let spectrum_result = transport_spectrum(&water)?;
    let mono_result = transport_monoenergetic(&water)?;
    let dose_grid = transport_voxel_dose(&water)?;

    summarize(&spectrum_result, &mono_result);

    // Linear attenuation coefficient μ (m⁻¹) of 100 keV photons in water, used
    // as the analytic reference curve in the attenuation and depth-dose plots.
    let mu_100kev =
        model.total_cross_section_per_m(&Photon, &Photon::state_along_z(0.1), &water)?;
    generate_plots(&water, &mono_result, &dose_grid, mu_100kev)?;
    save_results(&water, &dose_grid, mu_100kev)?;
    Ok(())
}

/// Section 1 — per-process atomic cross-sections and the total linear
/// attenuation coefficient μ (m⁻¹) and mean free path (mm) in water.
fn cross_section_table(
    model: &PhotonInteractionModel,
    water: &HomogeneousMedium,
) -> Result<(), PhysicsError> {
    println!(
        "[1] Photon cross-sections in water (Z_eff = {:.2})",
        water.effective_z()
    );
    println!(
        "    {:>8} | {:>12} {:>12} {:>12} | {:>10} {:>10}",
        "E (keV)", "photoel m²", "compton m²", "pair m²", "μ (1/m)", "mfp (mm)"
    );
    println!("    {}", "-".repeat(74));

    let z = water.effective_z();
    for &e_kev in &[10.0_f64, 30.0, 50.0, 100.0, 500.0, 1000.0] {
        let e_mev = e_kev * 1e-3;
        let tau = photoelectric::cross_section(e_mev, z);
        let sigma = compton::cross_section(e_mev, z);
        let kappa = pair_production::cross_section(e_mev, z);

        let state = Photon::state_along_z(e_mev);
        let mu = model.total_cross_section_per_m(&Photon, &state, water)?;
        let mfp_mm = if mu > 0.0 { 1e3 / mu } else { f64::INFINITY };

        println!(
            "    {e_kev:>8.0} | {tau:>12.3e} {sigma:>12.3e} {kappa:>12.3e} | {mu:>10.3} {mfp_mm:>10.3}"
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
        let bar = (count * 50 / peak).max(0);
        println!("    {e_lo_kev:>5.0} keV | {} {count}", "█".repeat(bar));
    }
    println!("    → mean photon energy ≈ {mean_kev:.1} keV (well below the 100 keV endpoint)\n");
    Ok(())
}

/// Section 3 — Monte Carlo transport of the full 100 kVp beam through water.
fn transport_spectrum(water: &HomogeneousMedium) -> Result<SimulationResult, PhysicsError> {
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

/// Section 4 — monoenergetic 100 keV beam, for comparison with the spectrum.
fn transport_monoenergetic(water: &HomogeneousMedium) -> Result<SimulationResult, PhysicsError> {
    let mono = Monoenergetic::new(TUBE_KVP * 1e-3)?; // 100 keV
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

    println!("[4] Transported {N_HISTORIES} primary histories (monoenergetic 100 keV beam)");
    report(&result);
    println!();
    Ok(result)
}

/// Section 5 — 3-D voxel dosimetry. Transport the monoenergetic 100 keV beam
/// through a *finite* 10 × 10 × 10 cm water cube (entrance face at z = 0) and
/// tally the energy deposited in a 10 × 10 × 10 grid of 1 cm cubic voxels. The
/// per-voxel mass is fixed (1 cm³ of water = 1 g), so the absorbed dose in gray
/// is unambiguous. Prints the depth-dose profile (PDD) and returns the grid for
/// plotting.
///
/// Local deposition is valid here: at 100 keV the secondary-electron range in
/// water is < 0.2 mm, far below the 1 cm voxel, so no energy leaks between
/// voxels and collision kerma ≈ absorbed dose.
fn transport_voxel_dose(water: &HomogeneousMedium) -> Result<VoxelGrid, PhysicsError> {
    // Finite water cube: 10 cm on a side, entrance face at z = 0, centred on
    // x = y = 0. A photon that steps outside it escapes.
    let cube = Geometry::Box {
        min: [-0.05, -0.05, 0.0],
        max: [0.05, 0.05, 0.10],
    };
    let engine = MonteCarloEngine::new(
        Photon,
        water.clone(),
        PhotonInteractionModel,
        RunConfig {
            n_histories: N_HISTORIES,
            seed: SEED,
            ..Default::default()
        },
    )
    .with_geometry(cube);

    // 10 × 10 × 10 grid of 1 cm cubic voxels filling the cube.
    let grid = VoxelGrid::new([-0.05, -0.05, 0.0], 0.01, [10, 10, 10])?;
    let mut rng = StdRng::seed_from_u64(SEED);
    let (_result, grid) = engine.run_with_voxels(Photon::state_along_z(0.1), grid, &mut rng)?;

    println!("[5] Voxel dosimetry: {N_HISTORIES} × 100 keV photons in a 10 cm water cube");
    let profile = grid.depth_profile_mev();
    let pdd = grid.relative_pdd();
    let dose = grid.depth_dose_gy(water);
    println!(
        "    {:>7} | {:>12} | {:>10} | {:>15}",
        "z (cm)", "deposit MeV", "rel. PDD", "slice dose Gy"
    );
    println!("    {}", "-".repeat(54));
    let voxel_cm = grid.voxel_size_m() * 100.0;
    for iz in 0..grid.dims()[2] {
        let z_lo = iz as f64 * voxel_cm;
        let z_hi = z_lo + voxel_cm;
        let label = format!("{z_lo:.0}-{z_hi:.0}");
        println!(
            "    {label:>7} | {:>12.2} | {:>10.4} | {:>15.3e}",
            profile[iz], pdd[iz], dose[iz]
        );
    }
    // The dose in gray is unambiguous once the voxel mass is fixed; the absolute
    // value is "per the N simulated primaries" (normalize by N to compare runs).
    let central = dose[dose.len() / 2 - 1];
    println!(
        "    → central slice (z = 4–5 cm) dose ≈ {central:.3e} Gy for {N_HISTORIES} primaries"
    );
    println!("    → PDD falls more slowly than exp(-μz): forward Compton scatter (≈85 % of each");
    println!("      100 keV interaction's energy) is carried downstream and deposited deeper.\n");
    Ok(grid)
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
/// PENELOPE).
///
/// Two self-describing files are written:
///   * `depth_dose_100kev_water.csv` — the 1-D depth-dose profile (PDD): the
///     headline curve to overlay on a reference MC.
///   * `voxel_dose_100kev_water.csv` — the full 3-D per-voxel tally (one row per
///     voxel) for a complete 3-D dose comparison or gamma analysis.
///
/// Each file begins with a `#`-prefixed metadata block (beam, geometry,
/// conservation totals) that pandas/NumPy skip with `comment='#'`. The
/// `dose_gy_per_primary` column is the run-independent quantity to compare in
/// absolute terms (a reference MC typically reports dose per source particle);
/// `rel_pdd` is the dimensionless shape, robust to absolute normalization.
fn save_results(
    water: &HomogeneousMedium,
    grid: &VoxelGrid,
    mu_per_m: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("results");
    std::fs::create_dir_all(&dir)?;

    let header = results_header(water, grid, mu_per_m);
    let p1 = write_depth_dose_csv(&dir, water, grid, &header)?;
    let p2 = write_voxel_dose_csv(&dir, water, grid, &header)?;

    println!("\n[results] CSV files written (load with pandas read_csv(..., comment='#')):");
    for p in [&p1, &p2] {
        println!("    {}", p.display());
    }
    Ok(())
}

/// Build the shared `#`-prefixed metadata header carried by every results CSV.
///
/// Records everything needed to reproduce the run and to normalize the absolute
/// dose: beam, medium, geometry, voxel size, seed, μ, and the conservation
/// totals (`total_deposit_in_grid_mev + overflow_mev` equals the summed
/// per-history deposit).
fn results_header(water: &HomogeneousMedium, grid: &VoxelGrid, mu_per_m: f64) -> String {
    let o = grid.origin();
    let [nx, ny, nz] = grid.dims();
    let s = grid.voxel_size_m();
    let cm = 100.0;
    let mut h = String::new();
    h.push_str("# polypus-physics depth-dose experiment — numerical results\n");
    h.push_str("# beam: monoenergetic 100 keV photon pencil beam along +z from (0,0,0)\n");
    h.push_str(&format!(
        "# medium: {} (rho = {:.1} kg/m^3)\n",
        water.name,
        water.density_kg_m3()
    ));
    h.push_str(&format!("# beam_energy_mev: {:.6}\n", TUBE_KVP * 1e-3));
    h.push_str(&format!("# n_primaries: {N_HISTORIES}\n"));
    h.push_str(&format!("# seed: {SEED}\n"));
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
    h.push_str(&format!("# mu_100kev_water_per_m: {mu_per_m:.4}\n"));
    h.push_str(&format!(
        "# total_deposit_in_grid_mev: {:.6}\n",
        grid.total_energy_mev()
    ));
    h.push_str(&format!("# overflow_mev: {:.6}\n", grid.overflow_mev()));
    h.push_str("# note: dose_gy is absolute for n_primaries; dose_gy_per_primary = dose_gy / n_primaries\n");
    h.push_str("# note: local deposition is valid at 100 keV (e- range < 0.2 mm << 1 cm voxel)\n");
    h
}

/// Write the 1-D depth-dose profile (PDD), one row per `z`-slice.
fn write_depth_dose_csv(
    dir: &Path,
    water: &HomogeneousMedium,
    grid: &VoxelGrid,
    header: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let profile = grid.depth_profile_mev();
    let pdd = grid.relative_pdd();
    let dose = grid.depth_dose_gy(water);
    let s_cm = grid.voxel_size_m() * 100.0;
    let z0_cm = grid.origin()[2] * 100.0;
    let n = N_HISTORIES as f64;

    let mut csv = String::from(header);
    csv.push_str("z_min_cm,z_max_cm,z_center_cm,deposit_mev,rel_pdd,dose_gy,dose_gy_per_primary\n");
    for iz in 0..grid.dims()[2] {
        let z_lo = z0_cm + iz as f64 * s_cm;
        let z_hi = z_lo + s_cm;
        let z_c = z_lo + 0.5 * s_cm;
        csv.push_str(&format!(
            "{z_lo:.3},{z_hi:.3},{z_c:.3},{:.6e},{:.6},{:.6e},{:.6e}\n",
            profile[iz],
            pdd[iz],
            dose[iz],
            dose[iz] / n
        ));
    }

    let path = dir.join("depth_dose_100kev_water.csv");
    std::fs::write(&path, csv)?;
    Ok(path)
}

/// Write the full 3-D per-voxel tally, one row per voxel (`x`-fastest order).
fn write_voxel_dose_csv(
    dir: &Path,
    water: &HomogeneousMedium,
    grid: &VoxelGrid,
    header: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let [nx, ny, nz] = grid.dims();
    let energy = grid.energy_mev();
    let dose = grid.dose_gy(water);
    let o = grid.origin();
    let s = grid.voxel_size_m();
    let cm = 100.0;
    let n = N_HISTORIES as f64;

    let mut csv = String::from(header);
    csv.push_str(
        "ix,iy,iz,x_center_cm,y_center_cm,z_center_cm,energy_mev,dose_gy,dose_gy_per_primary\n",
    );
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let i = ix + nx * (iy + ny * iz);
                let x_c = (o[0] + (ix as f64 + 0.5) * s) * cm;
                let y_c = (o[1] + (iy as f64 + 0.5) * s) * cm;
                let z_c = (o[2] + (iz as f64 + 0.5) * s) * cm;
                csv.push_str(&format!(
                    "{ix},{iy},{iz},{x_c:.3},{y_c:.3},{z_c:.3},{:.6e},{:.6e},{:.6e}\n",
                    energy[i],
                    dose[i],
                    dose[i] / n
                ));
            }
        }
    }

    let path = dir.join("voxel_dose_100kev_water.csv");
    std::fs::write(&path, csv)?;
    Ok(path)
}

// ─────────────────────────────────────────────────────────────────────────
// Plot generation (pure-Rust SVG, no external crates)
// ─────────────────────────────────────────────────────────────────────────

/// Generate and save the SVG figures under `<crate>/examples/plots/`.
fn generate_plots(
    water: &HomogeneousMedium,
    mono: &SimulationResult,
    dose_grid: &VoxelGrid,
    mu_100kev: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("plots");
    std::fs::create_dir_all(&dir)?;

    let p1 = plot_cross_sections(&dir, water, "water", "cross_sections_water.svg")?;
    let p_lead = plot_cross_sections(
        &dir,
        &HomogeneousMedium::lead(),
        "lead",
        "cross_sections_lead.svg",
    )?;
    let p2 = plot_spectrum(&dir)?;
    let p3 = plot_attenuation(&dir, mono, mu_100kev)?;
    let p4 = plot_depth_dose(&dir, dose_grid, mu_100kev)?;
    let p5 = plot_dose_map(&dir, dose_grid)?;
    let p6 = plot_lateral_profile(&dir, dose_grid)?;

    println!("\n[plots] SVG files written (open in any browser):");
    for p in [&p1, &p_lead, &p2, &p3, &p4, &p5, &p6] {
        println!("    {}", p.display());
    }
    Ok(())
}

/// Plot A — per-atom photon cross-sections vs energy (log-log) for `medium`.
/// Compare the shapes and crossover energies against a NIST XCOM figure. For a
/// high-`Z` medium (lead) the photoelectric K absorption edge appears as a
/// near-vertical jump near 88 keV.
fn plot_cross_sections(
    dir: &Path,
    medium: &HomogeneousMedium,
    material: &str,
    filename: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let z = medium.effective_z();
    let (lx0, lx1) = (0.01_f64.log10(), 10.0_f64.log10()); // energy MeV (log)

    // Auto-scale the σ axis to the data with a fixed 6-decade dynamic range
    // anchored at the peak total cross-section. Six decades is wide enough to
    // show the photoelectric curvature (slope −3.5 → −1) without squashing the
    // Compton fall-off into a flat line, and adapts to each material (low-Z
    // water, high-Z lead) automatically.
    let total = |e: f64| {
        photoelectric::cross_section(e, z)
            + compton::cross_section(e, z)
            + pair_production::cross_section(e, z)
    };
    let mut tmax = 0.0_f64;
    for i in 0..=200 {
        let e = 10f64.powf(lx0 + (i as f64 / 200.0) * (lx1 - lx0));
        tmax = tmax.max(total(e));
    }
    let ly1 = tmax.log10().ceil(); // σ m²/atom (log), top
    let ly0 = ly1 - 6.0; // 6-decade window

    let (w, h) = (780.0, 480.0);
    let (ml, mr, mt, mb) = (84.0, 168.0, 48.0, 58.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;
    let mut c = svg::Canvas::new(w, h);

    let xmap = |e: f64| ml + (e.log10() - lx0) / (lx1 - lx0) * pw;
    let ymap = |s: f64| mt + (1.0 - (s.log10() - ly0) / (ly1 - ly0)) * ph;

    // x decade gridlines + labels
    for k in (lx0 as i32)..=(lx1 as i32) {
        let e = 10f64.powi(k);
        let x = xmap(e);
        c.line(x, mt, x, mt + ph, "#e8e8e8", 1.0);
        let lbl = if e < 1.0 {
            format!("{e}")
        } else {
            format!("{e:.0}")
        };
        c.text(x, mt + ph + 18.0, &lbl, 12.0, "middle", "#333");
    }
    // y decade gridlines + labels
    for d in (ly0 as i32)..=(ly1 as i32) {
        let y = ymap(10f64.powi(d));
        c.line(ml, y, ml + pw, y, "#e8e8e8", 1.0);
        c.text(ml - 8.0, y + 4.0, &format!("1e{d}"), 11.0, "end", "#333");
    }
    frame(&mut c, ml, mt, pw, ph);

    // Log-spaced energy grid, plus two points straddling the K edge so the
    // photoelectric discontinuity renders as a near-vertical jump.
    let n = 320;
    let e_lo = 10f64.powf(lx0);
    let e_hi = 10f64.powf(lx1);
    let mut energies: Vec<f64> = (0..=n)
        .map(|i| 10f64.powf(lx0 + (i as f64 / n as f64) * (lx1 - lx0)))
        .collect();
    let e_k = photoelectric::k_edge_energy_mev(z);
    if e_k > e_lo && e_k < e_hi {
        energies.push(e_k * 0.9994);
        energies.push(e_k * 1.0006);
        energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    let curve = |f: &dyn Fn(f64) -> f64| -> Vec<(f64, f64)> {
        let y_floor = 10f64.powf(ly0);
        let y_ceil = 10f64.powf(ly1);
        energies
            .iter()
            .map(|&e| (e, f(e)))
            .filter(|&(_, v)| v >= y_floor) // drop points below the frame
            .map(|(e, v)| (xmap(e), ymap(v.min(y_ceil))))
            .collect()
    };
    c.polyline(
        &curve(&|e| photoelectric::cross_section(e, z)),
        "#d62728",
        1.8,
    );
    c.polyline(&curve(&|e| compton::cross_section(e, z)), "#1f77b4", 1.8);
    c.polyline(
        &curve(&|e| pair_production::cross_section(e, z)),
        "#2ca02c",
        1.8,
    );
    c.polyline(
        &curve(&|e| {
            photoelectric::cross_section(e, z)
                + compton::cross_section(e, z)
                + pair_production::cross_section(e, z)
        }),
        "#000000",
        2.4,
    );

    // Mark the pair-production threshold (1.022 MeV). The cross-section turns on
    // exactly here but is tiny just above it (it grows as (k−2)³), so the green
    // curve only becomes visible higher up; this line makes the true onset
    // explicit.
    let x_thr = xmap(PAIR_PRODUCTION_THRESHOLD_MEV);
    c.line(x_thr, mt, x_thr, mt + ph, "#2ca02c", 0.8);
    c.text(
        x_thr + 3.0,
        mt + ph - 6.0,
        "pair threshold",
        10.0,
        "start",
        "#2ca02c",
    );

    // legend
    let lx = ml + pw + 16.0;
    let mut ly = mt + 12.0;
    for (name, color) in [
        ("photoelectric", "#d62728"),
        ("Compton", "#1f77b4"),
        ("pair production", "#2ca02c"),
        ("total", "#000000"),
    ] {
        c.line(lx, ly, lx + 22.0, ly, color, 3.0);
        c.text(lx + 28.0, ly + 4.0, name, 12.0, "start", "#222");
        ly += 22.0;
    }
    // annotate the K edge when it falls inside the plot
    if e_k > e_lo && e_k < e_hi {
        ly += 12.0;
        c.text(lx, ly, "K edge", 12.0, "start", "#d62728");
        ly += 16.0;
        c.text(
            lx,
            ly,
            &format!("{:.0} keV", e_k * 1e3),
            12.0,
            "start",
            "#555",
        );
    }

    c.text(
        w / 2.0,
        26.0,
        &format!("Photon cross-sections in {material} (per atom)"),
        16.0,
        "middle",
        "#111",
    );
    c.text(
        ml + pw / 2.0,
        h - 14.0,
        "Energy (MeV)",
        13.0,
        "middle",
        "#111",
    );
    c.text_rotated(
        20.0,
        mt + ph / 2.0,
        "Cross-section (m² / atom)",
        13.0,
        "#111",
    );

    let path = dir.join(filename);
    std::fs::write(&path, c.render())?;
    Ok(path)
}

/// Plot B — the sampled 100 kVp bremsstrahlung spectrum (histogram).
fn plot_spectrum(dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let spectrum = KramersSpectrum::from_kvp(TUBE_KVP, FILTER_CUTOFF_KEV)?;
    let mut rng = StdRng::seed_from_u64(SEED);
    const N: usize = 300_000;
    const BINS: usize = 18;
    let lo = spectrum.min_energy_mev();
    let hi = spectrum.max_energy_mev();
    let bw = (hi - lo) / BINS as f64;

    let mut bins = [0u64; BINS];
    let mut sum = 0.0;
    for _ in 0..N {
        let e = spectrum.sample_energy_mev(&mut rng);
        sum += e;
        let idx = (((e - lo) / bw) as usize).min(BINS - 1);
        bins[idx] += 1;
    }
    let mean_kev = sum / N as f64 * 1e3;
    let ymax = *bins.iter().max().unwrap_or(&1) as f64;

    let (w, h) = (780.0, 480.0);
    let (ml, mr, mt, mb) = (84.0, 40.0, 48.0, 58.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;
    let mut c = svg::Canvas::new(w, h);

    let xmap = |kev: f64| ml + (kev - lo * 1e3) / ((hi - lo) * 1e3) * pw;
    let ymap = |v: f64| mt + (1.0 - v / ymax) * ph;

    for t in 0..=5 {
        let v = ymax * t as f64 / 5.0;
        let y = ymap(v);
        c.line(ml, y, ml + pw, y, "#e8e8e8", 1.0);
        c.text(ml - 8.0, y + 4.0, &format!("{v:.0}"), 11.0, "end", "#333");
    }
    for (i, &cnt) in bins.iter().enumerate() {
        let x = xmap((lo + i as f64 * bw) * 1e3);
        let xr = xmap((lo + (i as f64 + 1.0) * bw) * 1e3);
        let y = ymap(cnt as f64);
        c.rect(x + 1.0, y, (xr - x - 2.0).max(1.0), mt + ph - y, "#1f77b4");
    }
    let mut kev = (lo * 1e3).round();
    while kev <= hi * 1e3 + 0.1 {
        c.text(
            xmap(kev),
            mt + ph + 18.0,
            &format!("{kev:.0}"),
            12.0,
            "middle",
            "#333",
        );
        kev += 10.0;
    }
    let xm = xmap(mean_kev);
    c.line(xm, mt, xm, mt + ph, "#d62728", 1.6);
    c.text(
        xm + 5.0,
        mt + 14.0,
        &format!("mean {mean_kev:.0} keV"),
        12.0,
        "start",
        "#d62728",
    );
    frame(&mut c, ml, mt, pw, ph);

    c.text(
        w / 2.0,
        26.0,
        &format!("{TUBE_KVP:.0} kVp bremsstrahlung spectrum (Kramers' law)"),
        16.0,
        "middle",
        "#111",
    );
    c.text(
        ml + pw / 2.0,
        h - 14.0,
        "Photon energy (keV)",
        13.0,
        "middle",
        "#111",
    );
    c.text_rotated(20.0, mt + ph / 2.0, "Sampled photons", 13.0, "#111");

    let path = dir.join("spectrum_100kvp.svg");
    std::fs::write(&path, c.render())?;
    Ok(path)
}

/// Plot C — Beer–Lambert attenuation. The simulated surviving fraction (from
/// the depth of each photon's first interaction) should overlap the analytic
/// `exp(-μz)` curve, confirming the sampled mean free path matches μ.
fn plot_attenuation(
    dir: &Path,
    mono: &SimulationResult,
    mu_per_m: f64,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let depths: Vec<f64> = mono
        .histories
        .iter()
        .filter(|hh| hh.track.len() >= 2)
        .map(|hh| hh.track[1].position.0[2] * 1e3) // metres → mm
        .collect();
    let n = depths.len().max(1) as f64;
    let mfp_mm = 1e3 / mu_per_m;
    let zmax = 5.0 * mfp_mm;

    let (w, h) = (780.0, 480.0);
    let (ml, mr, mt, mb) = (84.0, 168.0, 48.0, 58.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;
    let mut c = svg::Canvas::new(w, h);

    let xmap = |z: f64| ml + z / zmax * pw;
    let ymap = |s: f64| mt + (1.0 - s) * ph;

    for t in 0..=5 {
        let s = t as f64 / 5.0;
        let y = ymap(s);
        c.line(ml, y, ml + pw, y, "#e8e8e8", 1.0);
        c.text(ml - 8.0, y + 4.0, &format!("{s:.1}"), 11.0, "end", "#333");
        let z = zmax * t as f64 / 5.0;
        let x = xmap(z);
        c.line(x, mt, x, mt + ph, "#e8e8e8", 1.0);
        c.text(
            x,
            mt + ph + 18.0,
            &format!("{z:.0}"),
            12.0,
            "middle",
            "#333",
        );
    }

    let m = 120;
    let mut emp = Vec::with_capacity(m + 1);
    let mut ana = Vec::with_capacity(m + 1);
    for i in 0..=m {
        let z = zmax * i as f64 / m as f64;
        let surviving = depths.iter().filter(|&&d| d > z).count() as f64 / n;
        emp.push((xmap(z), ymap(surviving)));
        ana.push((xmap(z), ymap((-mu_per_m * z * 1e-3).exp())));
    }
    c.polyline(&ana, "#d62728", 2.2);
    c.polyline(&emp, "#1f77b4", 2.0);
    frame(&mut c, ml, mt, pw, ph);

    let lx = ml + pw + 16.0;
    let mut ly = mt + 12.0;
    c.line(lx, ly, lx + 22.0, ly, "#1f77b4", 3.0);
    c.text(lx + 28.0, ly + 4.0, "simulated", 12.0, "start", "#222");
    ly += 22.0;
    c.line(lx, ly, lx + 22.0, ly, "#d62728", 3.0);
    c.text(lx + 28.0, ly + 4.0, "exp(-μz)", 12.0, "start", "#222");
    ly += 30.0;
    c.text(
        lx,
        ly,
        &format!("μ = {mu_per_m:.1} /m"),
        12.0,
        "start",
        "#555",
    );
    ly += 18.0;
    c.text(
        lx,
        ly,
        &format!("mfp = {mfp_mm:.1} mm"),
        12.0,
        "start",
        "#555",
    );

    c.text(
        w / 2.0,
        26.0,
        "Beer–Lambert attenuation: 100 keV photons in water",
        15.0,
        "middle",
        "#111",
    );
    c.text(
        ml + pw / 2.0,
        h - 14.0,
        "Depth of first interaction (mm)",
        13.0,
        "middle",
        "#111",
    );
    c.text_rotated(20.0, mt + ph / 2.0, "Surviving fraction", 13.0, "#111");

    let path = dir.join("attenuation_100kev_water.svg");
    std::fs::write(&path, c.render())?;
    Ok(path)
}

/// Plot D — depth-dose (PDD). The simulated relative depth dose (blue, from the
/// voxel tally) is overlaid on the analytic primary-only curve `exp(-μz)` (red),
/// both normalized to the first slice. They coincide near the surface, then the
/// simulated PDD rides progressively ABOVE `exp(-μz)`: accumulated forward
/// Compton scatter deposits extra energy downstream, flattening the falloff
/// (the depth-dose drops more slowly than the primary fluence). Near the exit
/// face the simulated curve steepens again as scattered photons leak out of the
/// finite cube.
fn plot_depth_dose(
    dir: &Path,
    grid: &VoxelGrid,
    mu_per_m: f64,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let pdd = grid.relative_pdd();
    let nz = pdd.len();
    let voxel_m = grid.voxel_size_m();
    let voxel_cm = voxel_m * 100.0;
    let zmax = nz as f64 * voxel_cm;

    let (w, h) = (780.0, 480.0);
    let (ml, mr, mt, mb) = (84.0, 168.0, 48.0, 58.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;
    let mut c = svg::Canvas::new(w, h);

    let xmap = |z_cm: f64| ml + z_cm / zmax * pw;
    let ymap = |s: f64| mt + (1.0 - s) * ph;

    for t in 0..=5 {
        let s = t as f64 / 5.0;
        let y = ymap(s);
        c.line(ml, y, ml + pw, y, "#e8e8e8", 1.0);
        c.text(ml - 8.0, y + 4.0, &format!("{s:.1}"), 11.0, "end", "#333");
    }
    for iz in 0..=nz {
        let z = iz as f64 * voxel_cm;
        let x = xmap(z);
        c.line(x, mt, x, mt + ph, "#e8e8e8", 1.0);
        c.text(
            x,
            mt + ph + 18.0,
            &format!("{z:.0}"),
            12.0,
            "middle",
            "#333",
        );
    }

    // Analytic exp(-μz), normalized to the first slice centre so both curves
    // equal 1.0 there and diverge with depth.
    let z0_m = 0.5 * voxel_m;
    let ana: Vec<(f64, f64)> = (0..nz)
        .map(|iz| {
            let zc_m = (iz as f64 + 0.5) * voxel_m;
            let zc_cm = (iz as f64 + 0.5) * voxel_cm;
            (xmap(zc_cm), ymap((-mu_per_m * (zc_m - z0_m)).exp()))
        })
        .collect();
    let sim: Vec<(f64, f64)> = (0..nz)
        .map(|iz| {
            let zc_cm = (iz as f64 + 0.5) * voxel_cm;
            (xmap(zc_cm), ymap(pdd[iz]))
        })
        .collect();
    c.polyline(&ana, "#d62728", 2.2);
    c.polyline(&sim, "#1f77b4", 2.0);
    // Mark each simulated voxel-slice sample.
    for &(x, y) in &sim {
        c.rect(x - 2.0, y - 2.0, 4.0, 4.0, "#1f77b4");
    }
    frame(&mut c, ml, mt, pw, ph);

    let lx = ml + pw + 16.0;
    let mut ly = mt + 12.0;
    c.line(lx, ly, lx + 22.0, ly, "#1f77b4", 3.0);
    c.text(lx + 28.0, ly + 4.0, "simulated PDD", 12.0, "start", "#222");
    ly += 22.0;
    c.line(lx, ly, lx + 22.0, ly, "#d62728", 3.0);
    c.text(lx + 28.0, ly + 4.0, "exp(-μz)", 12.0, "start", "#222");
    ly += 30.0;
    c.text(
        lx,
        ly,
        &format!("μ = {mu_per_m:.1} /m"),
        12.0,
        "start",
        "#555",
    );
    ly += 18.0;
    c.text(lx, ly, "scatter lifts", 12.0, "start", "#555");
    ly += 16.0;
    c.text(lx, ly, "PDD at depth", 12.0, "start", "#555");

    c.text(
        w / 2.0,
        26.0,
        "Depth-dose (PDD): 100 keV photons in a water cube",
        15.0,
        "middle",
        "#111",
    );
    c.text(
        ml + pw / 2.0,
        h - 14.0,
        "Depth z (cm)",
        13.0,
        "middle",
        "#111",
    );
    c.text_rotated(20.0, mt + ph / 2.0, "Relative dose", 13.0, "#111");

    let path = dir.join("depth_dose_100kev_water.svg");
    std::fs::write(&path, c.render())?;
    Ok(path)
}

/// Plot E — 2-D dose distribution in the plane that contains the beam axis.
///
/// The per-voxel deposited energy is projected through the `y` axis (summed
/// over every `y` row) onto the depth–lateral (`z`–`x`) plane and drawn as a
/// heat map on a three-decade logarithmic colour scale. This is the "side
/// view" of the experiment: a bright band along the central beam axis that
/// attenuates with depth and bleeds laterally as Compton-scattered photons
/// migrate off-axis. Because the medium is uniform, relative deposited energy
/// equals relative dose. A logarithmic scale is used so the faint scatter halo
/// (orders of magnitude below the axis) is visible alongside the bright core.
fn plot_dose_map(dir: &Path, grid: &VoxelGrid) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let [nx, ny, nz] = grid.dims();
    let e = grid.energy_mev();
    let voxel_cm = grid.voxel_size_m() * 100.0;

    // Project the deposited energy through y onto the z–x plane.
    let proj =
        |ix: usize, iz: usize| -> f64 { (0..ny).map(|iy| e[ix + nx * (iy + ny * iz)]).sum() };
    let mut vmax = 0.0_f64;
    for iz in 0..nz {
        for ix in 0..nx {
            vmax = vmax.max(proj(ix, iz));
        }
    }
    const DECADES: i32 = 3;
    let vfloor = vmax * 10f64.powi(-DECADES);
    let norm = |v: f64| -> f64 {
        if v <= vfloor || vmax <= 0.0 {
            0.0
        } else {
            ((v.ln() - vfloor.ln()) / (vmax.ln() - vfloor.ln())).clamp(0.0, 1.0)
        }
    };

    let (w, h) = (820.0, 470.0);
    let (ml, mr, mt, mb) = (70.0, 150.0, 56.0, 58.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;
    let mut c = svg::Canvas::new(w, h);

    // z increases to the right; +x is drawn upward (ix = nx-1 at the top). A
    // half-pixel overlap on each cell avoids hairline seams between voxels.
    let cw = pw / nz as f64;
    let chh = ph / nx as f64;
    for iz in 0..nz {
        for ix in 0..nx {
            let x = ml + iz as f64 * cw;
            let y = mt + (nx - 1 - ix) as f64 * chh;
            c.rect(x, y, cw + 0.5, chh + 0.5, &hot_color(norm(proj(ix, iz))));
        }
    }
    frame(&mut c, ml, mt, pw, ph);

    // Beam axis (x = 0) runs along the lateral centre of the grid.
    let y_axis = mt + ph / 2.0;
    c.line(ml, y_axis, ml + pw, y_axis, "#1f77b4", 1.0);
    c.text(
        ml + 4.0,
        y_axis - 4.0,
        "beam axis",
        11.0,
        "start",
        "#1f77b4",
    );

    // Depth (z) ticks every 2 cm.
    let zspan = nz as f64 * voxel_cm;
    let mut zc = 0.0;
    while zc <= zspan + 0.01 {
        let x = ml + zc / zspan * pw;
        c.text(
            x,
            mt + ph + 18.0,
            &format!("{zc:.0}"),
            12.0,
            "middle",
            "#333",
        );
        zc += 2.0;
    }
    // Lateral (x) ticks at -5, 0, +5 cm (grid centred on 0; +x upward).
    let half = 0.5 * nx as f64 * voxel_cm;
    for xc in [-half, 0.0, half] {
        let y = y_axis - xc / (nx as f64 * voxel_cm) * ph;
        c.text(ml - 8.0, y + 4.0, &format!("{xc:.0}"), 11.0, "end", "#333");
    }

    // Colour bar (log scale) on the right edge.
    let cbx = ml + pw + 28.0;
    let cbw = 18.0;
    let steps = 64usize;
    for i in 0..steps {
        let t = i as f64 / (steps - 1) as f64;
        let y = mt + (1.0 - t) * ph;
        c.rect(cbx, y, cbw, ph / steps as f64 + 0.8, &hot_color(t));
    }
    c.line(cbx, mt, cbx, mt + ph, "#333", 1.0);
    c.line(cbx + cbw, mt, cbx + cbw, mt + ph, "#333", 1.0);
    for d in 0..=DECADES {
        let t = d as f64 / DECADES as f64;
        let y = mt + (1.0 - t) * ph;
        let exp = -DECADES + d;
        c.line(cbx + cbw, y, cbx + cbw + 4.0, y, "#333", 1.0);
        c.text(
            cbx + cbw + 7.0,
            y + 4.0,
            &format!("1e{exp}"),
            11.0,
            "start",
            "#333",
        );
    }
    c.text(cbx + cbw / 2.0, mt - 8.0, "rel.", 11.0, "middle", "#333");

    c.text(
        w / 2.0,
        26.0,
        "Dose distribution in the water cube (beam-axis plane)",
        15.0,
        "middle",
        "#111",
    );
    c.text(
        ml + pw / 2.0,
        h - 14.0,
        "Depth z (cm)",
        13.0,
        "middle",
        "#111",
    );
    c.text_rotated(20.0, mt + ph / 2.0, "Lateral x (cm)", 13.0, "#111");

    let path = dir.join("dose_map_100kev_water.svg");
    std::fs::write(&path, c.render())?;
    Ok(path)
}

/// Plot F — lateral dose profiles at a shallow, a central and a deep slice.
///
/// Deposited energy is projected through `y` and plotted against the lateral
/// coordinate `x`, each profile normalized to its own on-axis value so the
/// *shape* can be compared across depths. The profiles broaden with depth:
/// every primary deposits its first-interaction energy on the central axis,
/// while later-generation Compton-scattered photons populate the off-axis
/// wings, which grow with depth. This is the lateral counterpart of the
/// depth-dose curve.
fn plot_lateral_profile(
    dir: &Path,
    grid: &VoxelGrid,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let [nx, ny, nz] = grid.dims();
    let e = grid.energy_mev();
    let voxel_cm = grid.voxel_size_m() * 100.0;
    let half = 0.5 * nx as f64 * voxel_cm;

    // Lateral profile of slice iz: deposited energy projected through y.
    let lateral = |iz: usize| -> Vec<f64> {
        (0..nx)
            .map(|ix| (0..ny).map(|iy| e[ix + nx * (iy + ny * iz)]).sum::<f64>())
            .collect()
    };
    let x_cm = |ix: usize| -half + (ix as f64 + 0.5) * voxel_cm;

    let (w, h) = (780.0, 470.0);
    let (ml, mr, mt, mb) = (84.0, 168.0, 52.0, 58.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;
    let mut c = svg::Canvas::new(w, h);

    let xmap = |x: f64| ml + (x + half) / (2.0 * half) * pw;
    let ymap = |s: f64| mt + (1.0 - s) * ph;

    for t in 0..=5 {
        let s = t as f64 / 5.0;
        let y = ymap(s);
        c.line(ml, y, ml + pw, y, "#e8e8e8", 1.0);
        c.text(ml - 8.0, y + 4.0, &format!("{s:.1}"), 11.0, "end", "#333");
    }
    for xc in [-half, -half / 2.0, 0.0, half / 2.0, half] {
        let x = xmap(xc);
        c.line(x, mt, x, mt + ph, "#e8e8e8", 1.0);
        c.text(
            x,
            mt + ph + 18.0,
            &format!("{xc:.1}"),
            12.0,
            "middle",
            "#333",
        );
    }

    // Shallow / central / deep slices.
    let slices = [
        (0usize, "#1f77b4"),
        (nz / 2, "#2ca02c"),
        (nz.saturating_sub(2), "#d62728"),
    ];
    let lx = ml + pw + 16.0;
    let mut ly = mt + 12.0;
    for (iz, color) in slices {
        let prof = lateral(iz);
        let pmax = prof
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let pts: Vec<(f64, f64)> = prof
            .iter()
            .enumerate()
            .map(|(ix, &v)| (xmap(x_cm(ix)), ymap(v / pmax)))
            .collect();
        c.polyline(&pts, color, 2.0);
        for &(x, y) in &pts {
            c.rect(x - 2.0, y - 2.0, 4.0, 4.0, color);
        }
        let z_lo = iz as f64 * voxel_cm;
        c.line(lx, ly, lx + 22.0, ly, color, 3.0);
        c.text(
            lx + 28.0,
            ly + 4.0,
            &format!("z = {z_lo:.0}-{:.0} cm", z_lo + voxel_cm),
            12.0,
            "start",
            "#222",
        );
        ly += 22.0;
    }
    ly += 12.0;
    for line in ["each profile", "normalized to", "its own axis"] {
        c.text(lx, ly, line, 12.0, "start", "#555");
        ly += 16.0;
    }
    frame(&mut c, ml, mt, pw, ph);

    c.text(
        w / 2.0,
        26.0,
        "Lateral dose profiles vs depth (scatter broadening)",
        15.0,
        "middle",
        "#111",
    );
    c.text(
        ml + pw / 2.0,
        h - 14.0,
        "Lateral position x (cm)",
        13.0,
        "middle",
        "#111",
    );
    c.text_rotated(20.0, mt + ph / 2.0, "Relative dose", 13.0, "#111");

    let path = dir.join("lateral_profile_100kev_water.svg");
    std::fs::write(&path, c.render())?;
    Ok(path)
}

/// Map a normalized value `t ∈ [0, 1]` to an `rgb(...)` colour string on a
/// sequential light→dark "YlOrRd" heat scale (pale yellow → orange → dark red),
/// interpolating linearly between five control stops. Used by the dose heat
/// map; it prints legibly and is roughly perceptually ordered.
fn hot_color(t: f64) -> String {
    const STOPS: [(f64, (f64, f64, f64)); 5] = [
        (0.00, (255.0, 255.0, 204.0)),
        (0.25, (254.0, 217.0, 118.0)),
        (0.50, (253.0, 141.0, 60.0)),
        (0.75, (240.0, 59.0, 32.0)),
        (1.00, (189.0, 0.0, 38.0)),
    ];
    let t = t.clamp(0.0, 1.0);
    let mut i = 0;
    while i + 1 < STOPS.len() && t > STOPS[i + 1].0 {
        i += 1;
    }
    let (t0, c0) = STOPS[i];
    let (t1, c1) = STOPS[(i + 1).min(STOPS.len() - 1)];
    let f = if (t1 - t0).abs() < 1e-12 {
        0.0
    } else {
        (t - t0) / (t1 - t0)
    };
    let lerp = |a: f64, b: f64| (a + (b - a) * f).round() as u8;
    format!(
        "rgb({},{},{})",
        lerp(c0.0, c1.0),
        lerp(c0.1, c1.1),
        lerp(c0.2, c1.2)
    )
}

/// Draw a rectangular plot frame from four lines.
fn frame(c: &mut svg::Canvas, ml: f64, mt: f64, pw: f64, ph: f64) {
    c.line(ml, mt, ml + pw, mt, "#333", 1.0);
    c.line(ml, mt + ph, ml + pw, mt + ph, "#333", 1.0);
    c.line(ml, mt, ml, mt + ph, "#333", 1.0);
    c.line(ml + pw, mt, ml + pw, mt + ph, "#333", 1.0);
}

/// Minimal self-contained SVG canvas builder (pure Rust, zero dependencies).
mod svg {
    use std::fmt::Write;

    /// An SVG drawing surface that accumulates elements into a string.
    pub struct Canvas {
        width: f64,
        height: f64,
        body: String,
    }

    impl Canvas {
        /// Create a white canvas of the given pixel size.
        pub fn new(width: f64, height: f64) -> Self {
            let mut body = String::new();
            let _ = write!(
                body,
                r#"<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>"#
            );
            Canvas {
                width,
                height,
                body,
            }
        }

        /// Draw a straight line segment.
        pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, color: &str, width: f64) {
            let _ = write!(
                self.body,
                r#"<line x1="{x1:.2}" y1="{y1:.2}" x2="{x2:.2}" y2="{y2:.2}" stroke="{color}" stroke-width="{width}"/>"#
            );
        }

        /// Draw a filled rectangle.
        pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str) {
            let _ = write!(
                self.body,
                r#"<rect x="{x:.2}" y="{y:.2}" width="{w:.2}" height="{h:.2}" fill="{fill}"/>"#
            );
        }

        /// Draw a connected polyline through `pts` with no fill.
        pub fn polyline(&mut self, pts: &[(f64, f64)], color: &str, width: f64) {
            if pts.is_empty() {
                return;
            }
            let mut d = String::with_capacity(pts.len() * 12);
            for (x, y) in pts {
                let _ = write!(d, "{x:.2},{y:.2} ");
            }
            let _ = write!(
                self.body,
                r#"<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}"/>"#
            );
        }

        /// Draw a text label. `anchor` is one of `start`, `middle`, `end`.
        pub fn text(&mut self, x: f64, y: f64, s: &str, size: f64, anchor: &str, fill: &str) {
            let _ = write!(
                self.body,
                r#"<text x="{x:.2}" y="{y:.2}" font-family="sans-serif" font-size="{size}" text-anchor="{anchor}" fill="{fill}">{s}</text>"#
            );
        }

        /// Draw a vertically stacked (rotated −90°) text label, centred on `(x, y)`.
        pub fn text_rotated(&mut self, x: f64, y: f64, s: &str, size: f64, fill: &str) {
            let _ = write!(
                self.body,
                r#"<text x="{x:.2}" y="{y:.2}" transform="rotate(-90 {x:.2} {y:.2})" font-family="sans-serif" font-size="{size}" text-anchor="middle" fill="{fill}">{s}</text>"#
            );
        }

        /// Serialize the accumulated elements into a complete SVG document.
        pub fn render(&self) -> String {
            format!(
                r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{body}</svg>"#,
                w = self.width,
                h = self.height,
                body = self.body
            )
        }
    }
}
