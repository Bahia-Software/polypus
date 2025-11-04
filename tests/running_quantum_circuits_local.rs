use polypus::algorithms::{AlgorithmSingleRun,AlgorithmDifferentialEvolution};
use polypus::AlgorithmTrait;

#[test]
fn run_single_run() {    
    AlgorithmSingleRun.description();
}

// #[test]
// fn run_distribute_by_shots() {    
//     AlgorithmDistributeByShots.description();
// }

#[test]
fn run_differential_evolution() {    
    AlgorithmDifferentialEvolution.description();
}