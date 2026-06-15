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
//!   5. Saved SVG plots (cross-sections, spectrum, attenuation) for visual
//!      verification against references such as NIST XCOM.
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
    MonteCarloEngine, RunConfig, SimulationResult,
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

    summarize(&spectrum_result, &mono_result);

    // Linear attenuation coefficient μ (m⁻¹) of 100 keV photons in water, used
    // as the analytic reference curve in the attenuation plot.
    let mu_100kev =
        model.total_cross_section_per_m(&Photon, &Photon::state_along_z(0.1), &water)?;
    generate_plots(&water, &mono_result, mu_100kev)?;
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
// Plot generation (pure-Rust SVG, no external crates)
// ─────────────────────────────────────────────────────────────────────────

/// Generate and save the three SVG figures under `<crate>/examples/plots/`.
fn generate_plots(
    water: &HomogeneousMedium,
    mono: &SimulationResult,
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

    println!("\n[plots] SVG files written (open in any browser):");
    for p in [&p1, &p_lead, &p2, &p3] {
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
