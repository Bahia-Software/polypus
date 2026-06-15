//! Particle-track history records.

use crate::particle::Position;

/// One step along a particle track.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackPoint {
    /// Position at this point (m).
    pub position: Position,
    /// Particle energy at this point (MeV).
    pub energy_mev: f64,
    /// Step length from the previous track point (m).
    pub step_length_m: f64,
}

/// Complete history of one primary particle and all secondaries it produces.
#[derive(Debug, Clone)]
pub struct ParticleHistory {
    /// PDG ID of the tracked particle.
    pub pdg_id: i32,
    /// Ordered sequence of track points.
    pub track: Vec<TrackPoint>,
    /// Sum of energy deposited along this track (MeV).
    pub total_deposit_mev: f64,
    /// Recursively expanded secondary histories.
    pub secondaries: Vec<ParticleHistory>,
}

impl ParticleHistory {
    /// Create an empty history for a particle with the given PDG ID.
    pub fn new(pdg_id: i32) -> Self {
        ParticleHistory {
            pdg_id,
            track: Vec::new(),
            total_deposit_mev: 0.0,
            secondaries: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_history_is_empty() {
        let h = ParticleHistory::new(22);
        assert_eq!(h.pdg_id, 22);
        assert!(h.track.is_empty());
        assert_eq!(h.total_deposit_mev, 0.0);
        assert!(h.secondaries.is_empty());
    }
}
