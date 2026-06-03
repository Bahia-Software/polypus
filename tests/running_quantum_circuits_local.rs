use polypus::algorithms::{
    AlgorithmSingleRun, AlgorithmDifferentialEvolution,
    AlgorithmPSO, AlgorithmQNG, DistributeByShotsRun,
};
use polypus::infrastructure::Infrastructure;
use polypus::AlgorithmTrait;

// ─────────────────────────────────────────────────────────────────────────────
// AlgorithmTrait metadata — name() and description() are pure Rust, no Python
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn single_run_name() {
    assert_eq!(AlgorithmSingleRun.name(), "Single Run Algorithm");
}

#[test]
fn single_run_description_is_non_empty() {
    assert!(!AlgorithmSingleRun.description().is_empty());
}

#[test]
fn distribute_by_shots_name() {
    assert_eq!(DistributeByShotsRun.name(), "Distribute By Shots Run Algorithm");
}

#[test]
fn distribute_by_shots_description_is_non_empty() {
    assert!(!DistributeByShotsRun.description().is_empty());
}

#[test]
fn differential_evolution_name() {
    assert_eq!(AlgorithmDifferentialEvolution.name(), "Differential Evolution");
}

#[test]
fn differential_evolution_description_is_non_empty() {
    assert!(!AlgorithmDifferentialEvolution.description().is_empty());
}

#[test]
fn pso_name() {
    assert_eq!(AlgorithmPSO.name(), "Particle Swarm Optimization");
}

#[test]
fn pso_description_is_non_empty() {
    assert!(!AlgorithmPSO.description().is_empty());
}

#[test]
fn qng_name() {
    assert_eq!(AlgorithmQNG.name(), "Quantum Natural Gradient");
}

#[test]
fn qng_description_is_non_empty() {
    assert!(!AlgorithmQNG.description().is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Infrastructure::from_str — pure Rust parsing, no Python
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn infrastructure_from_str_local() {
    assert!(matches!(Infrastructure::from_str("local"), Infrastructure::Local));
}

#[test]
fn infrastructure_from_str_cunqa() {
    assert!(matches!(Infrastructure::from_str("cunqa"), Infrastructure::Cunqa));
}

#[test]
#[should_panic(expected = "Unknown infrastructure")]
fn infrastructure_from_str_unknown_panics() {
    Infrastructure::from_str("unknown_backend");
}
