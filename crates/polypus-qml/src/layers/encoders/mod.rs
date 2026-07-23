//! Encoders: layers that turn data features `x` into fixed rotation angles.
//! They consume zero trainable parameters (design doc §5.1).

mod amplitude;
mod angle;

pub use amplitude::AmplitudeEncoder;
pub use angle::AngleEncoder;
