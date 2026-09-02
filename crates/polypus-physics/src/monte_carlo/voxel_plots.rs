//! Plotting utilities for VoxelGrid depth-dose (PDD) profiles.

#![cfg(feature = "plotters")]

use std::path::Path;

use crate::error::PhysicsError;

/// Renders a relative percentage-depth-dose (PDD) curve and saves it as a
/// PNG image.
///
/// `pdd` are the values from [`super::voxel::VoxelGrid::relative_pdd`]
/// (0.0–1.0), one per depth layer, each `voxel_size_m` metres thick.
#[cfg(feature = "plotters")]
pub fn plot_relative_pdd(
    pdd: &[f64],
    voxel_size_m: f64,
    title: &str,
    path: &Path,
) -> Result<(), PhysicsError> {
    use plotters::prelude::*;

    if pdd.is_empty() {
        return Err(PhysicsError::SimulationError {
            message: "no PDD points to plot".to_string(),
        });
    }

    let to_plot_error = |e: String| PhysicsError::IoError {
        message: format!("could not render plot: {e}"),
    };

    // Center of each layer, in cm (más intuitivo para radioterapia que metros).
    let depths_cm: Vec<f64> = (0..pdd.len())
        .map(|i| (i as f64 + 0.5) * voxel_size_m * 100.0)
        .collect();
    let pdd_percent: Vec<f64> = pdd.iter().map(|v| v * 100.0).collect();
    let x_max = *depths_cm.last().unwrap() + voxel_size_m * 100.0 * 0.5;

    let root = BitMapBackend::new(path, (1200, 825)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| to_plot_error(e.to_string()))?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..x_max, 0.0..110.0)
        .map_err(|e| to_plot_error(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc("Depth (cm)")
        .y_desc("PDD (%)")
        .light_line_style(RGBColor(220, 220, 220))
        .draw()
        .map_err(|e| to_plot_error(e.to_string()))?;

    let points: Vec<(f64, f64)> = depths_cm
        .iter()
        .zip(pdd_percent.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    chart
        .draw_series(LineSeries::new(
            points.iter().copied(),
            RGBColor(37, 99, 235),
        ))
        .map_err(|e| to_plot_error(e.to_string()))?;
    chart
        .draw_series(
            points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 3, RGBColor(37, 99, 235).filled())),
        )
        .map_err(|e| to_plot_error(e.to_string()))?;

    root.present().map_err(|e| to_plot_error(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "plotters")]
    #[test]
    fn plot_relative_pdd_produces_a_png() {
        let pdd = vec![0.98, 1.0, 0.97, 0.93, 0.85, 0.78, 0.69, 0.61, 0.52, 0.41];
        let path = std::env::temp_dir().join("pdd_test.png");
        plot_relative_pdd(&pdd, 0.01, "Test PDD", &path).unwrap();
        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
    }
}
