//! Equivalence: the optimized `QuboObservable` agrees, over random counts, with
//! a naive reference implementation of the same cost written through the
//! `CostObservable` trait. This is entirely GIL-free (no Python interpreter),
//! proving the native aggregation matches the reference weighted-average formula.

use std::collections::HashMap;

use polypus_observable::{CostObservable, ObservableError, QuboObservable};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A reference observable: a boxed `Fn(&str) -> f64` cost, aggregated the same
/// way the historical Python `expectation_values` did (count-weighted mean).
struct ClosureObservable<F: Fn(&str) -> f64 + Send + Sync> {
    f: F,
}

impl<F: Fn(&str) -> f64 + Send + Sync> CostObservable for ClosureObservable<F> {
    fn expectation_batch(
        &self,
        counts: &[HashMap<String, u64>],
    ) -> Result<Vec<f64>, ObservableError> {
        Ok(counts
            .iter()
            .map(|c| {
                let mut num = 0.0f64;
                let mut den = 0u64;
                for (k, &n) in c {
                    num += (self.f)(k) * n as f64;
                    den += n;
                }
                if den == 0 {
                    0.0
                } else {
                    num / den as f64
                }
            })
            .collect())
    }
}

/// Naive `f(x)` for a QUBO spec, decoding bits with the MSB-left convention.
fn naive_qubo(
    key: &str,
    num_vars: usize,
    linear: &[f64],
    quadratic: &[(usize, usize, f64)],
    constant: f64,
    scale: f64,
) -> f64 {
    let bytes = key.as_bytes();
    let w = bytes.len();
    let bit = |i: usize| -> f64 {
        if bytes[w - 1 - i] == b'1' {
            1.0
        } else {
            0.0
        }
    };
    let mut acc = constant;
    for (i, &c) in linear.iter().enumerate().take(num_vars) {
        acc += c * bit(i);
    }
    for &(i, j, wq) in quadratic {
        acc += wq * bit(i) * bit(j);
    }
    acc * scale
}

#[test]
fn native_qubo_matches_naive_reference() {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let num_vars = 6;

    // Random dense linear + a few random quadratic terms.
    let linear: Vec<f64> = (0..num_vars).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let mut quadratic = Vec::new();
    for i in 0..num_vars {
        for j in (i + 1)..num_vars {
            if rng.gen_bool(0.5) {
                quadratic.push((i, j, rng.gen_range(-1.5..1.5)));
            }
        }
    }
    let constant = rng.gen_range(-1.0..1.0);
    let scale = -1.0;

    let sparse_linear: Vec<(usize, f64)> = linear.iter().copied().enumerate().collect();
    let native =
        QuboObservable::new(num_vars, sparse_linear, quadratic.clone(), constant, scale).unwrap();

    let lin = linear.clone();
    let quad = quadratic.clone();
    let reference = ClosureObservable {
        f: move |key: &str| naive_qubo(key, num_vars, &lin, &quad, constant, scale),
    };

    // Random batch of candidates, each a handful of random bitstrings + counts.
    let mut batch: Vec<HashMap<String, u64>> = Vec::new();
    for _ in 0..32 {
        let mut c = HashMap::new();
        for _ in 0..rng.gen_range(1..12) {
            let key: String = (0..num_vars)
                .map(|_| if rng.gen_bool(0.5) { '1' } else { '0' })
                .collect();
            *c.entry(key).or_insert(0) += rng.gen_range(1..50);
        }
        batch.push(c);
    }

    let a = native.expectation_batch(&batch).unwrap();
    let b = reference.expectation_batch(&batch).unwrap();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        // Tolerance, not equality: HashMap iteration order makes the FP
        // summation order differ between the two paths.
        assert!((x - y).abs() < 1e-9, "native {x} != reference {y}");
    }
}
