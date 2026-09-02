//! Plotting utilities for ENDF-6 photon mu_m curves, rendered as log-log
//! line charts and saved as PNG images.

#![cfg(feature = "plotters")]

use std::path::Path;

use crate::error::PhysicsError;

use super::mass_attenuation_coefficients::{CompoundResult, MuPoint};

/// Renders a set of (energy, mu_m) points as a log-log line chart, and
/// saves it as a PNG image.
#[cfg(feature = "plotters")]
fn plot_mu_m_points(points: &[MuPoint], title: &str, path: &Path) -> Result<(), PhysicsError> {
    use plotters::prelude::*;

    if points.is_empty() {
        return Err(PhysicsError::MalformedEndfData {
            message: "no points to plot".to_string(),
        });
    }

    let to_plot_error = |e: String| PhysicsError::IoError {
        message: format!("could not render plot: {e}"),
    };

    let (x_min, x_max, y_min, y_max) = points.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(x_min, x_max, y_min, y_max), p| {
            let x = p.energy_ev * 1e-6;
            (
                x_min.min(x),
                x_max.max(x),
                y_min.min(p.mu_m),
                y_max.max(p.mu_m),
            )
        },
    );

    let root = BitMapBackend::new(path, (1200, 825)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| to_plot_error(e.to_string()))?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d((x_min..x_max).log_scale(), (y_min..y_max).log_scale())
        .map_err(|e| to_plot_error(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc("Photon energy (MeV)")
        .y_desc("mu_m (cm2/g)")
        .light_line_style(RGBColor(220, 220, 220))
        .x_label_formatter(&|x| format!("{x:.1e}"))
        .y_label_formatter(&|y| format!("{y:.1e}"))
        .draw()
        .map_err(|e| to_plot_error(e.to_string()))?;

    chart
        .draw_series(LineSeries::new(
            points.iter().map(|p| (p.energy_ev * 1e-6, p.mu_m)),
            RGBColor(37, 99, 235),
        ))
        .map_err(|e| to_plot_error(e.to_string()))?;

    root.present().map_err(|e| to_plot_error(e.to_string()))?;
    Ok(())
}

/// Renders a single element's mu_m curve and saves it as a PNG image.
#[cfg(feature = "plotters")]
pub fn plot_element(symbol: &str, points: &[MuPoint], path: &Path) -> Result<(), PhysicsError> {
    let title = format!("{symbol}: mass attenuation coefficient");
    plot_mu_m_points(points, &title, path)
}

/// Renders a compound's mu_m curve and saves it as a PNG image.
#[cfg(feature = "plotters")]
pub fn plot_compound(
    formula: &str,
    result: &CompoundResult,
    path: &Path,
) -> Result<(), PhysicsError> {
    let title = format!("{formula}: mass attenuation coefficient");
    plot_mu_m_points(&result.points, &title, path)
}

#[cfg(test)]
mod tests {
    use super::super::mass_attenuation_coefficients::{mu_m_for_compound, mu_m_for_element};
    use super::*;

    #[cfg(feature = "plotters")]
    #[test]
    fn plot_element_produces_a_png() {
        let (_z, _a, points) = mu_m_for_element("H", 501).unwrap();
        let path = std::env::temp_dir().join("H_mu_m_test.png");
        plot_element("H", &points, &path).unwrap();
        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
    }

    #[cfg(feature = "plotters")]
    #[test]
    fn plot_compound_produces_a_png() {
        let result = mu_m_for_compound("H2O", 501, 5000).unwrap();
        let path = std::env::temp_dir().join("H2O_mu_m_test.png");
        plot_compound("H2O", &result, &path).unwrap();
        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
    }
}
