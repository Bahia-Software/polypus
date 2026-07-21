//! Encoders: layers that turn data features `x` into fixed rotation angles.
//! They consume zero trainable parameters (design doc §5.1).

mod angle;

pub use angle::AngleEncoder;
