//! Rotosolve on exact expectations of native circuits (issue #398).
//!
//! The oracle simulates each candidate with the native statevector and returns
//! the exact expectation of a diagonal cost, `Σ_b cost(b)·|⟨b|ψ(θ)⟩|²`: the
//! fitness `polypus.train` estimates from the shots, without the shot noise.
//! On a circuit that passes `check_single_frequency`, each step lands on the
//! exact maximum along its parameter; no Python interpreter is involved.

use polypus::circuit::{Param, ParamExpr, ParameterizedCircuit};
use polypus::optimizers::{
    AlgorithmRotosolve, AlgorithmRotosolveArgs, EvaluationOracle, Optimizer,
};
use polypus_sim::{Simulator, StatevectorSimulator};
use std::f64::consts::{FRAC_PI_2, TAU};
use std::sync::{Arc, Mutex};

/// A 3-qubit circuit with every single-frequency gate, entanglers between
/// them and one angle `−θ₇ + π/2`.
fn ansatz() -> ParameterizedCircuit {
    let mut qc = ParameterizedCircuit::new(3);
    let shifted = qc
        .add_expr(-ParamExpr::param(7) + FRAC_PI_2)
        .expect("an affine angle is a valid expression");
    qc.h(0)
        .rx(0, Param(0))
        .ry(1, Param(1))
        .rz(2, Param(2))
        .cx(0, 1)
        .rxx(1, 2, Param(3))
        .h(2)
        .rzz(0, 2, Param(4))
        .p(1, Param(5))
        .cx(1, 2)
        .u1(0, Param(6))
        .ry(2, shifted)
        .h(1)
        .cx(2, 0)
}

/// A cost per bitstring with no symmetry, so the landscape is not flat.
fn cost(state: usize) -> f64 {
    [0.3, -1.0, 2.0, 0.7, -0.4, 1.5, -2.2, 0.9][state]
}

/// The exact expectation of [`cost`] for one candidate.
fn expectation(qc: &ParameterizedCircuit, params: &[f64]) -> f64 {
    let concrete = qc
        .assign_parameters(params)
        .expect("a candidate has one value per parameter");
    let state = StatevectorSimulator::new()
        .run(&concrete)
        .expect("a 3-qubit circuit simulates");
    state
        .probabilities()
        .iter()
        .enumerate()
        .map(|(b, p)| cost(b) * p)
        .sum()
}

/// Scores candidates with [`expectation`] and records every batch.
#[derive(Clone)]
struct Exact {
    circuit: Arc<ParameterizedCircuit>,
    batches: Arc<Mutex<Vec<Vec<Vec<f64>>>>>,
}

impl EvaluationOracle for Exact {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        self.batches.lock().unwrap().push(candidates.to_vec());
        candidates
            .iter()
            .map(|c| expectation(&self.circuit, c))
            .collect()
    }
}

/// The point step `k` (coordinate `j`) moves from, read off its batch.
fn point_of(batch: &[Vec<f64>], j: usize) -> Vec<f64> {
    match batch {
        [x, _, _] => x.clone(),
        [plus, minus] => {
            let mut x = plus.clone();
            x[j] = 0.5 * (plus[j] + minus[j]);
            x
        }
        _ => panic!("a step's batch has 2 or 3 candidates"),
    }
}

#[test]
fn every_step_lands_on_the_exact_maximum_along_its_parameter() {
    let circuit = ansatz();
    assert_eq!(circuit.check_single_frequency(), Ok(()));
    let dims = circuit.num_params();
    assert_eq!(dims, 8);
    let oracle = Exact {
        circuit: Arc::new(circuit.clone()),
        batches: Arc::default(),
    };
    let outcome = AlgorithmRotosolve
        .optimize(AlgorithmRotosolveArgs {
            oracle: Box::new(oracle.clone()),
            sweeps: 4,
            reevaluate_every: 5,
            tolerance: 0.0,
            patience: 1,
            dimensions: dims as u32,
            seed: Some(2024),
            initial_point: None,
        })
        .expect("valid Rotosolve args optimize successfully");
    let batches = oracle.batches.lock().unwrap().clone();
    assert_eq!(batches.len(), 4 * dims);
    let f = |x: &[f64]| expectation(&circuit, x);
    const GRID: usize = 4000;
    for k in 0..batches.len() - 1 {
        let (j, next_j) = (k % dims, (k + 1) % dims);
        let x = point_of(&batches[k], j);
        let next = point_of(&batches[k + 1], next_j);
        let mut moved = x.clone();
        moved[j] = next[j];
        let grid_best = (0..GRID)
            .map(|g| {
                let mut y = x.clone();
                y[j] = TAU * g as f64 / GRID as f64;
                f(&y)
            })
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            f(&moved) >= grid_best - 1e-12,
            "step {k}: {} below the grid's {grid_best}",
            f(&moved)
        );
        // Coordinate ascent on exact values never goes down.
        assert!(f(&moved) >= f(&x) - 1e-12, "step {k}");
    }
    assert_eq!(outcome.best_fitness, f(&outcome.best_params));
    assert_eq!(outcome.iterations_run, 4 * dims);
    // The landscape is not flat: the run climbs well above θ₀.
    let theta0 = f(&batches[0][0]);
    assert!(
        outcome.best_fitness > theta0 + 0.5,
        "{theta0} to {}",
        outcome.best_fitness
    );
}

/// A controlled rotation is not single-frequency in its angle: with the
/// control in superposition and read out in the X basis, the fitness has
/// period 4π. The check rejects the circuit, and the closed-form step indeed
/// misses the maximum.
#[test]
fn a_controlled_rotation_fails_the_check_and_the_step() {
    let circuit = ParameterizedCircuit::new(2).h(0).crx(0, 1, Param(0)).h(0);
    assert!(circuit.check_single_frequency().is_err());
    let f = |t: f64| expectation(&circuit, &[t]);
    let oracle = Exact {
        circuit: Arc::new(circuit.clone()),
        batches: Arc::default(),
    };
    AlgorithmRotosolve
        .optimize(AlgorithmRotosolveArgs {
            oracle: Box::new(oracle.clone()),
            sweeps: 2,
            reevaluate_every: 1,
            tolerance: 0.0,
            patience: 1,
            dimensions: 1,
            seed: None,
            initial_point: Some(vec![0.4]),
        })
        .expect("valid Rotosolve args optimize successfully");
    let moved = oracle.batches.lock().unwrap()[1][0][0];
    let best = (0..4000)
        .map(|g| f(2.0 * TAU * g as f64 / 4000.0))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(f(moved) < best - 1e-3, "{} vs {best}", f(moved));
}
