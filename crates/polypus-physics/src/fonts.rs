//! Embeds and registers the fonts `plotters` needs to render text, since
//! the `ab_glyph` backend (used instead of `font-kit`/fontconfig, which
//! isn't available on all CI runners) does not discover system fonts on
//! its own.

use std::sync::OnceLock;

static DEJAVU_SANS: &[u8] = include_bytes!("../assets/fonts/DejaVuSans.ttf");
static DEJAVU_SANS_BOLD: &[u8] = include_bytes!("../assets/fonts/DejaVuSans-Bold.ttf");

/// Registers the embedded fonts with `plotters`, the first time this is
/// called. Safe to call from every plotting function — subsequent calls
/// are no-ops.
pub(crate) fn register_fonts() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(|| {
        use plotters::style::{register_font, FontStyle};
        register_font("sans-serif", FontStyle::Normal, DEJAVU_SANS)
            .map_err(|_| "invalid embedded DejaVuSans.ttf")
            .unwrap();
        register_font("sans-serif", FontStyle::Bold, DEJAVU_SANS_BOLD)
            .map_err(|_| "invalid embedded DejaVuSans-Bold.ttf")
            .unwrap();
    });
}
