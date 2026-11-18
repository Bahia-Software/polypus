//! Acceptance tests of issue #530: qubit-wise-commuting grouping of the
//! 4-qubit Jordan–Wigner H2 Hamiltonian, and the parity reducer against the
//! simulator's exact `⟨Z…Z⟩`, against seeded samples, and against
//! `IsingObservable`.

use std::collections::HashMap;

use polypus_circuit::{GateInstruction, GateParam};
use polypus_observable::{
    group_qubit_wise, CostObservable, IsingObservable, ParityObservable, Pauli, PauliMask,
};
use polypus_physics::chemistry::{H2Encoding, H2Molecule};
use polypus_physics::hamiltonians::{PauliAxis, PauliSum, PhysicsHamiltonian};
use polypus_sim::{SplitMix64, Statevector};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const NUM_QUBITS: usize = 4;

/// The 4-qubit Jordan–Wigner H2 at equilibrium.
fn h2() -> PauliSum {
    H2Molecule::new(0.7414, H2Encoding::JordanWigner)
        .unwrap()
        .to_pauli_sum()
}

/// A `PauliSum` term by term as `(coefficient, mask)`: the conversion a caller
/// of this crate writes (it does not depend on `polypus-physics`).
fn to_masks(sum: &PauliSum) -> Vec<(f64, PauliMask)> {
    sum.terms()
        .iter()
        .map(|term| {
            let ops = term.string.ops().iter().map(|&(qubit, axis)| {
                let pauli = match axis {
                    PauliAxis::X => Pauli::X,
                    PauliAxis::Y => Pauli::Y,
                    PauliAxis::Z => Pauli::Z,
                };
                (qubit, pauli)
            });
            (term.coefficient, PauliMask::new(ops).unwrap())
        })
        .collect()
}

/// An entangled 4-qubit state with real amplitudes (`Ry` and `CX` only), so
/// `⟨ψ|H|ψ⟩ = ψᵀHψ` with the real dense matrix of `H`.
fn entangled_state() -> Statevector {
    let ry = |qubit, angle| GateInstruction::Ry {
        qubit,
        theta: GateParam::Fixed(angle),
    };
    let gates = [
        ry(0, 0.7),
        ry(1, 1.9),
        ry(2, -0.4),
        ry(3, 2.6),
        GateInstruction::Cx(0, 1),
        GateInstruction::Cx(1, 2),
        GateInstruction::Cx(2, 3),
        ry(0, -1.3),
        ry(1, 0.5),
        ry(2, 1.1),
        ry(3, -0.8),
        GateInstruction::Cx(3, 0),
    ];
    let mut state = Statevector::new(NUM_QUBITS).unwrap();
    for gate in &gates {
        state.apply(gate).unwrap();
    }
    state
}

#[test]
fn h2_groups_cover_every_term_once_and_commute_qubit_wise() {
    let terms = to_masks(&h2());
    let grouping = group_qubit_wise(NUM_QUBITS, &terms).unwrap();

    // Deviation from the issue's wording, by design (decision 3): "the groups
    // cover every term exactly once" holds for every non-identity term; the
    // identity term is in `constant`, since its value needs no measurement.
    let mut seen = vec![0usize; terms.len()];
    for group in grouping.groups() {
        for &index in group.term_indices() {
            seen[index] += 1;
        }
        for &a in group.term_indices() {
            for &b in group.term_indices() {
                assert!(terms[a].1.qubit_wise_commutes(&terms[b].1));
            }
        }
    }
    let mut identity = 0.0;
    for (index, &(coefficient, mask)) in terms.iter().enumerate() {
        if mask.is_identity() {
            assert_eq!(seen[index], 0, "identity term {index} is in a group");
            identity += coefficient;
        } else {
            assert_eq!(seen[index], 1, "term {index} is in {} groups", seen[index]);
        }
    }
    assert_eq!(grouping.constant(), identity);
    assert_ne!(identity, 0.0);

    // 15 terms: the identity, 10 Z-only terms (one group), and the four
    // XXYY-type exchange terms, which differ on every qubit (one group each).
    assert_eq!(terms.len(), 15);
    assert_eq!(grouping.groups().len(), 5);
}

#[test]
fn h2_energy_from_rotated_groups_equals_the_dense_expectation() {
    // The whole chain: rotate a copy of the state per group, reduce its exact
    // probabilities, add the constant; compare with ψᵀHψ.
    let sum = h2();
    let grouping = group_qubit_wise(NUM_QUBITS, &to_masks(&sum)).unwrap();
    let state = entangled_state();

    let mut energy = grouping.constant();
    for group in grouping.groups() {
        let mut rotated = state.clone();
        for (qubit, pauli) in group.rotations() {
            if pauli == Pauli::Y {
                rotated.apply(&GateInstruction::Sdg(qubit)).unwrap();
            }
            rotated.apply(&GateInstruction::H(qubit)).unwrap();
        }
        energy += group
            .observable()
            .expectation_from_probabilities(&rotated.probabilities())
            .unwrap();
    }

    let matrix = sum.to_dense_real().unwrap();
    let psi: Vec<f64> = state.amplitudes().iter().map(|a| a.re).collect();
    assert!(state.amplitudes().iter().all(|a| a.im == 0.0));
    let dim = psi.len();
    let mut dense = 0.0;
    for r in 0..dim {
        for c in 0..dim {
            dense += psi[r] * matrix.get(r, c).unwrap() * psi[c];
        }
    }
    assert!(
        (energy - dense).abs() < 1e-12,
        "grouped {energy} vs dense {dense}"
    );
}

/// A diagonal observable with `Z` strings of every weight 0..=4 (as
/// `(coefficient, qubits)`), and its mask form.
fn diagonal_terms() -> Vec<(f64, Vec<usize>)> {
    vec![
        (0.3, vec![]),
        (-1.2, vec![0]),
        (0.8, vec![2]),
        (0.45, vec![1, 3]),
        (-0.6, vec![0, 1, 2]),
        (1.7, vec![0, 1, 2, 3]),
    ]
}

fn parity_observable(terms: &[(f64, Vec<usize>)]) -> ParityObservable {
    let masks = terms
        .iter()
        .map(|(c, qubits)| (*c, qubits.iter().fold(0u64, |m, &q| m | (1 << q))))
        .collect();
    ParityObservable::new(NUM_QUBITS, masks, 1.0).unwrap()
}

#[test]
fn exact_probabilities_match_the_simulator_expectation() {
    let state = entangled_state();
    let terms = diagonal_terms();
    let reduced = parity_observable(&terms)
        .expectation_from_probabilities(&state.probabilities())
        .unwrap();
    let exact: f64 = terms
        .iter()
        .map(|(c, qubits)| c * state.expectation_z(qubits).unwrap())
        .sum();
    assert!(
        (reduced - exact).abs() < 1e-12,
        "reduced {reduced} vs exact {exact}"
    );
}

#[test]
fn seeded_samples_are_within_three_sigma() {
    let state = entangled_state();
    let observable = parity_observable(&diagonal_terms());
    let probabilities = state.probabilities();

    let mean = observable
        .expectation_from_probabilities(&probabilities)
        .unwrap();
    let second: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(b, p)| p * observable.value(b as u64).powi(2))
        .sum();
    let shots = 20_000;
    let sigma = ((second - mean * mean) / shots as f64).sqrt();
    assert!(sigma > 0.0);

    let mut rng = SplitMix64::new(0x5EED_0530);
    let counts: HashMap<String, u64> = state
        .sample(shots, &mut rng)
        .into_iter()
        .map(|(b, n)| (format!("{b:0width$b}", width = NUM_QUBITS), n))
        .collect();
    let estimate = observable.expectation_batch(&[counts]).unwrap()[0];
    assert!(
        (estimate - mean).abs() < 3.0 * sigma,
        "estimate {estimate}, exact {mean}, sigma {sigma}"
    );
}

/// Weight ≤ 2 terms as both reducers take them: `ParityObservable` masks and
/// `IsingObservable` fields, couplings and constant.
struct WeightTwo {
    constant: f64,
    fields: Vec<(usize, f64)>,
    couplings: Vec<(usize, usize, f64)>,
}

impl WeightTwo {
    fn random(rng: &mut StdRng, coefficient: &mut impl FnMut(&mut StdRng) -> f64) -> Self {
        let constant = coefficient(rng);
        let fields = (0..NUM_QUBITS).map(|i| (i, coefficient(rng))).collect();
        let mut couplings = Vec::new();
        for i in 0..NUM_QUBITS {
            for j in i + 1..NUM_QUBITS {
                couplings.push((i, j, coefficient(rng)));
            }
        }
        WeightTwo {
            constant,
            fields,
            couplings,
        }
    }

    fn parity(&self) -> ParityObservable {
        let mut terms = vec![(self.constant, 0u64)];
        terms.extend(self.fields.iter().map(|&(i, h)| (h, 1u64 << i)));
        terms.extend(
            self.couplings
                .iter()
                .map(|&(i, j, c)| (c, (1u64 << i) | (1u64 << j))),
        );
        ParityObservable::new(NUM_QUBITS, terms, 1.0).unwrap()
    }

    fn ising(&self) -> IsingObservable {
        IsingObservable::new(
            NUM_QUBITS,
            self.fields.clone(),
            self.couplings.clone(),
            self.constant,
            1.0,
        )
        .unwrap()
    }

    fn abs_sum(&self) -> f64 {
        self.constant.abs()
            + self.fields.iter().map(|f| f.1.abs()).sum::<f64>()
            + self.couplings.iter().map(|c| c.2.abs()).sum::<f64>()
    }
}

/// Random counts over all `2^4` bitstrings, some absent.
fn random_counts(rng: &mut StdRng) -> HashMap<String, u64> {
    let mut counts = HashMap::new();
    for b in 0..1u64 << NUM_QUBITS {
        if rng.random_bool(0.8) {
            let key = format!("{b:0width$b}", width = NUM_QUBITS);
            counts.insert(key, rng.random_range(1..1000));
        }
    }
    counts
}

#[test]
fn weight_two_agrees_with_ising_bit_for_bit_on_dyadic_coefficients() {
    // The issue asks for bit-for-bit agreement with `IsingObservable` on
    // weight ≤ 2 inputs. In general that is impossible (see the next test):
    // `IsingObservable` evaluates the QUBO expansion `z = 1 − 2x`, the parity
    // reducer sums `±c` directly, and the two round in different places. With
    // dyadic coefficients (multiples of 1/8 here) every operation of both is
    // exact, so they must agree to the last bit; any sign, bit-order or parity
    // error would show.
    let mut rng = StdRng::seed_from_u64(0x530_D1AD);
    let mut dyadic = |rng: &mut StdRng| rng.random_range(-64i32..=64) as f64 / 8.0;
    for _ in 0..50 {
        let terms = WeightTwo::random(&mut rng, &mut dyadic);
        let counts: Vec<_> = (0..4).map(|_| random_counts(&mut rng)).collect();
        let parity = terms.parity().expectation_batch(&counts).unwrap();
        let ising = terms.ising().expectation_batch(&counts).unwrap();
        for (p, i) in parity.iter().zip(&ising) {
            assert_eq!(p.to_bits(), i.to_bits(), "parity {p} vs Ising {i}");
        }
    }
}

#[test]
fn weight_two_agrees_with_ising_to_rounding_on_any_coefficients() {
    // With arbitrary coefficients the two reducers round differently, so
    // bit-for-bit agreement is replaced by a tolerance relative to the
    // coefficients' scale. Example: c = 0.3, h0 = 0.1, h1 = 0.7 on bits "11"
    // is exactly 0.3 − 0.1 − 0.7 = −0.5; the parity sum gives −0.5 and the
    // QUBO expansion (0.3 + 0.1 + 0.7) − 0.2 − 1.4 gives −0.4999999999999998.
    let mut rng = StdRng::seed_from_u64(0x530_F10A);
    let mut real = |rng: &mut StdRng| rng.random_range(-2.0..2.0);
    for _ in 0..50 {
        let terms = WeightTwo::random(&mut rng, &mut real);
        let counts: Vec<_> = (0..4).map(|_| random_counts(&mut rng)).collect();
        let parity = terms.parity().expectation_batch(&counts).unwrap();
        let ising = terms.ising().expectation_batch(&counts).unwrap();
        let tolerance = 1e-12 * terms.abs_sum();
        for (p, i) in parity.iter().zip(&ising) {
            assert!((p - i).abs() <= tolerance, "parity {p} vs Ising {i}");
        }
    }

    let example = ParityObservable::new(2, vec![(0.3, 0), (0.1, 0b01), (0.7, 0b10)], 1.0).unwrap();
    let counts = HashMap::from([("11".to_string(), 1)]);
    assert_eq!(example.expectation_batch(&[counts]).unwrap(), vec![-0.5]);
}
