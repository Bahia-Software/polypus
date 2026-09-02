//! Plotting utilities for raw ENDF-6 photon cross-sections (barns),
//! rendered as log-log line charts and saved as PNG images.

use std::path::Path;

use crate::error::PhysicsError;
use crate::interactions::photon::mass_attenuation_coefficients::cross_section_for_element;

#[cfg(feature = "plotters")]
const KNOWN_CHANNELS: &[(u32, &str, (u8, u8, u8))] = &[
    (501, "Total", (37, 99, 235)),
    (502, "Coherent (Rayleigh)", (16, 185, 129)),
    (504, "Incoherent (Compton)", (234, 88, 12)),
    (522, "Photoelectric", (220, 38, 38)),
    (534, "Photoelectric (K-shell)", (147, 51, 234)),
    (515, "Pair production (electron field)", (13, 148, 136)),
    (516, "Pair production (total)", (202, 138, 4)),
    (517, "Pair production (nuclear field)", (219, 39, 119)),
];

/// Renders every ENDF-6 photon reaction channel available for one element
/// (total, coherent, incoherent, photoelectric, pair production...) as
/// overlaid log-log cross-section curves, and saves the result as a PNG.
///
/// A channel missing from the element's evaluation is silently skipped.
/// Points where a channel's cross-section is exactly zero (below its
/// physical threshold) are omitted, since zero has no logarithm and
/// cannot be shown on a log axis.
#[cfg(feature = "plotters")]
pub fn plot_element_cross_sections(symbol: &str, path: &Path) -> Result<(), PhysicsError> {
    use plotters::prelude::*;

    /// One channel's plotted series: its label, RGB color, and (energy_MeV,
    /// value) points.
    type ChannelSeries<'a> = (&'a str, (u8, u8, u8), Vec<(f64, f64)>);
    let mut series: Vec<ChannelSeries> = Vec::new();

    for &(mt, label, color) in KNOWN_CHANNELS {
        let Ok((_z, points)) = cross_section_for_element(symbol, mt) else {
            continue;
        };
        let xy: Vec<(f64, f64)> = points
            .iter()
            .filter(|p| p.sigma_barn > 0.0)
            .map(|p| (p.energy_ev * 1e-6, p.sigma_barn))
            .collect();
        if !xy.is_empty() {
            series.push((label, color, xy));
        }
    }

    if series.is_empty() {
        return Err(PhysicsError::UnknownElement {
            symbol: symbol.to_string(),
        });
    }

    let to_plot_error = |e: String| PhysicsError::IoError {
        message: format!("could not render plot: {e}"),
    };

    let (x_min, x_max, y_min, y_max) = series.iter().flat_map(|(_, _, xy)| xy.iter()).fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(x_min, x_max, y_min, y_max), &(x, y)| {
            (x_min.min(x), x_max.max(x), y_min.min(y), y_max.max(y))
        },
    );

    let root = BitMapBackend::new(path, (1200, 825)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| to_plot_error(e.to_string()))?;

    let title = format!("{symbol}: photon cross-sections");
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
        .y_desc("Cross-section (barn)")
        .light_line_style(RGBColor(220, 220, 220))
        .x_label_formatter(&|x| format!("{x:.1e}"))
        .y_label_formatter(&|y| format!("{y:.1e}"))
        .draw()
        .map_err(|e| to_plot_error(e.to_string()))?;

    for (label, (r, g, b), xy) in &series {
        let color = RGBColor(*r, *g, *b);
        chart
            .draw_series(LineSeries::new(xy.iter().copied(), color))
            .map_err(|e| to_plot_error(e.to_string()))?
            .label(*label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|e| to_plot_error(e.to_string()))?;

    root.present().map_err(|e| to_plot_error(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "plotters")]
    #[test]
    fn plot_element_cross_sections_produces_a_png() {
        let path = std::env::temp_dir().join("Fe_cross_sections_test.png");
        plot_element_cross_sections("Fe", &path).unwrap();
        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
    }
}
