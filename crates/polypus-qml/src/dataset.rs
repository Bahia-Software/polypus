//! [`Dataset`]: validated features + labels, deterministic splits, scaling.
//!
//! Features are stored flat and row-major (not `Vec<Vec<f64>>`): a single
//! allocation, cache-friendly access as a slice per sample, and a cheap
//! conversion from NumPy in a future bindings layer. Labels are kept as plain
//! `f64`; the required domain ({0,1}, {−1,+1}, continuous) is not fixed by the
//! dataset but by the loss/decision chosen downstream, so one dataset type
//! serves both classification and regression.

use crate::error::ValidationError;
use crate::rng::{shuffle, SplitMix64};

/// A validated supervised dataset: a rectangular feature matrix plus one label
/// per sample.
///
/// Construct with [`Dataset::from_rows`], which rejects every non-finite value
/// up front (mirroring C-2's `NonFiniteParam` policy). Once built, a `Dataset`
/// is guaranteed non-empty, rectangular, and entirely finite.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    /// Row-major: sample `i`, feature `j` is `features[i * num_features + j]`.
    features: Vec<f64>,
    labels: Vec<f64>,
    num_samples: usize,
    num_features: usize,
}

impl Dataset {
    /// Build a dataset from feature `rows` and their `labels`.
    ///
    /// Validation runs in order and fails on the first violation found:
    ///
    /// 1. `rows` empty → [`ValidationError::EmptyDataset`].
    /// 2. a row whose width differs from the first row's width →
    ///    [`ValidationError::RaggedRows`] for the first such row.
    /// 3. `labels.len() != rows.len()` →
    ///    [`ValidationError::LabelCountMismatch`].
    /// 4. a non-finite feature → [`ValidationError::NonFiniteFeature`] for the
    ///    first one in row-major order.
    /// 5. a non-finite label → [`ValidationError::NonFiniteLabel`] for the
    ///    first one.
    ///
    /// Storage is a single flat, row-major allocation.
    pub fn from_rows(rows: &[Vec<f64>], labels: &[f64]) -> Result<Dataset, ValidationError> {
        if rows.is_empty() {
            return Err(ValidationError::EmptyDataset);
        }

        let num_features = rows[0].len();
        for (sample, row) in rows.iter().enumerate() {
            if row.len() != num_features {
                return Err(ValidationError::RaggedRows {
                    sample,
                    expected: num_features,
                    got: row.len(),
                });
            }
        }

        if labels.len() != rows.len() {
            return Err(ValidationError::LabelCountMismatch {
                rows: rows.len(),
                labels: labels.len(),
            });
        }

        for (sample, row) in rows.iter().enumerate() {
            for (index, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(ValidationError::NonFiniteFeature { sample, index });
                }
            }
        }

        for (sample, &label) in labels.iter().enumerate() {
            if !label.is_finite() {
                return Err(ValidationError::NonFiniteLabel { sample });
            }
        }

        let num_samples = rows.len();
        let mut features = Vec::with_capacity(num_samples * num_features);
        for row in rows {
            features.extend_from_slice(row);
        }

        Ok(Dataset {
            features,
            labels: labels.to_vec(),
            num_samples,
            num_features,
        })
    }

    /// Number of samples (rows). Always `>= 1`.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Number of features per sample.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// The feature slice of sample `i` (length [`num_features`](Self::num_features)).
    ///
    /// Panics only on an out-of-range `i`, exactly as slice indexing does —
    /// callers pass `i < num_samples()`.
    pub fn sample(&self, i: usize) -> &[f64] {
        let start = i * self.num_features;
        &self.features[start..start + self.num_features]
    }

    /// All labels, in sample order.
    pub fn labels(&self) -> &[f64] {
        &self.labels
    }

    /// Split into `(train, test)` by shuffling sample indices with a
    /// [`SplitMix64`](crate::rng::SplitMix64) seeded with `seed`, then cutting.
    ///
    /// `test_fraction` must lie in the **open** interval `(0.0, 1.0)`; the
    /// endpoints are rejected with [`ValidationError::InvalidTestFraction`]
    /// because either would leave a partition empty. The test set takes the
    /// first `floor(num_samples * test_fraction)` shuffled samples and the
    /// train set takes the rest, so the rounding rule is: **the test partition
    /// rounds down**.
    ///
    /// The seed is explicit and mandatory in this pure crate; resolving
    /// `None → OS entropy` is the bindings layer's job. The same seed always
    /// produces the same split, byte for byte.
    pub fn train_test_split(
        &self,
        test_fraction: f64,
        seed: u64,
    ) -> Result<(Dataset, Dataset), ValidationError> {
        // `!(0 < f < 1)` also rejects NaN, whose comparisons are all false.
        if !(test_fraction > 0.0 && test_fraction < 1.0) {
            return Err(ValidationError::InvalidTestFraction {
                fraction: test_fraction,
            });
        }

        let mut indices: Vec<usize> = (0..self.num_samples).collect();
        let mut rng = SplitMix64::new(seed);
        shuffle(&mut indices, &mut rng);

        let n_test = (self.num_samples as f64 * test_fraction).floor() as usize;
        let (test_idx, train_idx) = indices.split_at(n_test);

        log::debug!(
            "train_test_split: seed={seed}, test_fraction={test_fraction}, {} train / {} test",
            train_idx.len(),
            test_idx.len()
        );

        Ok((self.select(train_idx), self.select(test_idx)))
    }

    /// Build a new dataset from the samples at `indices`, preserving order.
    ///
    /// `pub(crate)` so [`QmlProblem::from_subset`](crate::QmlProblem) (another
    /// module of this crate) can carve a minibatch out of a validated dataset;
    /// `train_test_split` above is the other in-crate caller.
    pub(crate) fn select(&self, indices: &[usize]) -> Dataset {
        let num_features = self.num_features;
        let mut features = Vec::with_capacity(indices.len() * num_features);
        let mut labels = Vec::with_capacity(indices.len());
        for &i in indices {
            features.extend_from_slice(self.sample(i));
            labels.push(self.labels[i]);
        }
        Dataset {
            features,
            labels,
            num_samples: indices.len(),
            num_features,
        }
    }

    /// Min–max scale every feature of **this** dataset into `[lo, hi]`, using
    /// ranges computed over this dataset.
    ///
    /// A constant feature (its min equals its max) has no range to normalize
    /// against; the whole column is mapped to `lo`. The convention recommended
    /// for angle encoding is `[0, π]`.
    pub fn scale_features_to(&mut self, lo: f64, hi: f64) {
        let ranges = self.feature_ranges();
        self.apply_scaling(&ranges, lo, hi);
    }

    /// The current `(min, max)` of each feature over this dataset.
    ///
    /// Useful to freeze the train set's ranges and replay them on the test set
    /// with [`scale_features_with`](Self::scale_features_with). The returned
    /// vector has one entry per feature, in feature order.
    pub fn feature_ranges(&self) -> Vec<(f64, f64)> {
        let mut ranges = vec![(f64::INFINITY, f64::NEG_INFINITY); self.num_features];
        for i in 0..self.num_samples {
            for (j, &value) in self.sample(i).iter().enumerate() {
                if value < ranges[j].0 {
                    ranges[j].0 = value;
                }
                if value > ranges[j].1 {
                    ranges[j].1 = value;
                }
            }
        }
        ranges
    }

    /// Apply the min–max scaling described by `ranges` to **this** dataset,
    /// mapping into `[lo, hi]`.
    ///
    /// `ranges` is typically the [`feature_ranges`](Self::feature_ranges) of a
    /// train set, replayed here on a test set so both are scaled identically.
    /// Its length must equal [`num_features`](Self::num_features), otherwise
    /// [`ValidationError::FeatureCountMismatch`]. A degenerate range (min
    /// equals max) maps its column to `lo`, as in
    /// [`scale_features_to`](Self::scale_features_to). Test values outside the
    /// supplied range are mapped linearly and may fall outside `[lo, hi]` —
    /// that is the intended behaviour of a frozen scaler.
    pub fn scale_features_with(
        &mut self,
        ranges: &[(f64, f64)],
        lo: f64,
        hi: f64,
    ) -> Result<(), ValidationError> {
        if ranges.len() != self.num_features {
            return Err(ValidationError::FeatureCountMismatch {
                expected: self.num_features,
                got: ranges.len(),
            });
        }
        self.apply_scaling(ranges, lo, hi);
        Ok(())
    }

    /// Rewrite `features` in place with the min–max map of `ranges` onto
    /// `[lo, hi]`. `ranges.len()` is assumed to equal `num_features` (callers
    /// validate first).
    fn apply_scaling(&mut self, ranges: &[(f64, f64)], lo: f64, hi: f64) {
        let num_features = self.num_features;
        for i in 0..self.num_samples {
            let base = i * num_features;
            for (j, &(min, max)) in ranges.iter().enumerate() {
                let span = max - min;
                let slot = &mut self.features[base + j];
                *slot = if span == 0.0 {
                    lo
                } else {
                    lo + (*slot - min) / span * (hi - lo)
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(data: &[&[f64]]) -> Vec<Vec<f64>> {
        data.iter().map(|r| r.to_vec()).collect()
    }

    #[test]
    fn from_rows_stores_row_major() {
        let ds = Dataset::from_rows(
            &rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]),
            &[0.0, 1.0, 0.0],
        )
        .unwrap();
        assert_eq!(ds.num_samples(), 3);
        assert_eq!(ds.num_features(), 2);
        assert_eq!(ds.sample(0), &[1.0, 2.0]);
        assert_eq!(ds.sample(1), &[3.0, 4.0]);
        assert_eq!(ds.sample(2), &[5.0, 6.0]);
        assert_eq!(ds.labels(), &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn empty_dataset_rejected() {
        assert_eq!(
            Dataset::from_rows(&[], &[]),
            Err(ValidationError::EmptyDataset)
        );
    }

    #[test]
    fn ragged_rows_reports_first_offender() {
        let err = Dataset::from_rows(&rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0]]), &[0.0, 0.0, 0.0])
            .unwrap_err();
        assert_eq!(
            err,
            ValidationError::RaggedRows {
                sample: 2,
                expected: 2,
                got: 1,
            }
        );
    }

    #[test]
    fn label_count_mismatch_rejected() {
        let err = Dataset::from_rows(&rows(&[&[1.0], &[2.0]]), &[0.0, 1.0, 0.0]).unwrap_err();
        assert_eq!(
            err,
            ValidationError::LabelCountMismatch { rows: 2, labels: 3 }
        );
    }

    #[test]
    fn non_finite_feature_reports_position() {
        let err =
            Dataset::from_rows(&rows(&[&[1.0, 2.0], &[3.0, f64::NAN]]), &[0.0, 0.0]).unwrap_err();
        assert_eq!(
            err,
            ValidationError::NonFiniteFeature {
                sample: 1,
                index: 1
            }
        );

        let err = Dataset::from_rows(&rows(&[&[f64::INFINITY]]), &[0.0]).unwrap_err();
        assert_eq!(
            err,
            ValidationError::NonFiniteFeature {
                sample: 0,
                index: 0
            }
        );
    }

    #[test]
    fn non_finite_label_reports_sample() {
        let err = Dataset::from_rows(&rows(&[&[1.0], &[2.0]]), &[0.0, f64::NAN]).unwrap_err();
        assert_eq!(err, ValidationError::NonFiniteLabel { sample: 1 });
    }

    #[test]
    fn split_is_deterministic_for_a_seed() {
        let ds = Dataset::from_rows(
            &rows(&[&[0.0], &[1.0], &[2.0], &[3.0], &[4.0], &[5.0]]),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let (tr1, te1) = ds.train_test_split(0.5, 99).unwrap();
        let (tr2, te2) = ds.train_test_split(0.5, 99).unwrap();
        assert_eq!(tr1, tr2);
        assert_eq!(te1, te2);
    }

    #[test]
    fn split_partitions_are_disjoint_and_complete() {
        let ds = Dataset::from_rows(
            &rows(&[&[0.0], &[1.0], &[2.0], &[3.0], &[4.0], &[5.0], &[6.0]]),
            &[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        .unwrap();
        let (train, test) = ds.train_test_split(0.3, 7).unwrap();
        // floor(7 * 0.3) = 2 in test, 5 in train.
        assert_eq!(test.num_samples(), 2);
        assert_eq!(train.num_samples(), 5);

        // Union of labels reconstructs the original multiset without gaps or
        // duplicates (labels here uniquely tag each original row).
        let mut recovered: Vec<f64> = train.labels().to_vec();
        recovered.extend_from_slice(test.labels());
        recovered.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(recovered, vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn invalid_test_fraction_at_boundaries() {
        let ds = Dataset::from_rows(&rows(&[&[0.0], &[1.0]]), &[0.0, 1.0]).unwrap();
        for f in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
            assert!(matches!(
                ds.train_test_split(f, 0),
                Err(ValidationError::InvalidTestFraction { .. })
            ));
        }
    }

    #[test]
    fn scale_features_to_maps_min_max() {
        let mut ds = Dataset::from_rows(
            &rows(&[&[0.0, 10.0], &[5.0, 20.0], &[10.0, 30.0]]),
            &[0.0, 0.0, 0.0],
        )
        .unwrap();
        ds.scale_features_to(0.0, 1.0);
        // Feature 0: 0..10 → 0, 0.5, 1. Feature 1: 10..30 → 0, 0.5, 1.
        assert_eq!(ds.sample(0), &[0.0, 0.0]);
        assert_eq!(ds.sample(1), &[0.5, 0.5]);
        assert_eq!(ds.sample(2), &[1.0, 1.0]);
    }

    #[test]
    fn scale_features_to_handles_constant_feature() {
        let mut ds = Dataset::from_rows(&rows(&[&[7.0, 1.0], &[7.0, 3.0]]), &[0.0, 0.0]).unwrap();
        ds.scale_features_to(0.0, 1.0);
        // Constant feature 0 collapses to lo; feature 1 scales normally.
        assert_eq!(ds.sample(0), &[0.0, 0.0]);
        assert_eq!(ds.sample(1), &[0.0, 1.0]);
    }

    #[test]
    fn feature_ranges_reports_current_min_max() {
        let ds = Dataset::from_rows(
            &rows(&[&[1.0, -2.0], &[3.0, 4.0], &[2.0, 0.0]]),
            &[0.0, 0.0, 0.0],
        )
        .unwrap();
        assert_eq!(ds.feature_ranges(), vec![(1.0, 3.0), (-2.0, 4.0)]);
    }

    #[test]
    fn scale_features_with_length_mismatch() {
        let mut ds = Dataset::from_rows(&rows(&[&[1.0, 2.0]]), &[0.0]).unwrap();
        let err = ds.scale_features_with(&[(0.0, 1.0)], 0.0, 1.0).unwrap_err();
        assert_eq!(
            err,
            ValidationError::FeatureCountMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn freeze_train_ranges_and_apply_to_test() {
        let mut train = Dataset::from_rows(&rows(&[&[0.0], &[10.0]]), &[0.0, 1.0]).unwrap();
        let ranges = train.feature_ranges();
        assert_eq!(ranges, vec![(0.0, 10.0)]);
        train.scale_features_to(0.0, 1.0);
        assert_eq!(train.sample(0), &[0.0]);
        assert_eq!(train.sample(1), &[1.0]);

        // Test set scaled with the *train* ranges: value 5 → 0.5, and a value
        // outside the train range (20) extrapolates past 1.0 by design.
        let mut test = Dataset::from_rows(&rows(&[&[5.0], &[20.0]]), &[0.0, 1.0]).unwrap();
        test.scale_features_with(&ranges, 0.0, 1.0).unwrap();
        assert_eq!(test.sample(0), &[0.5]);
        assert_eq!(test.sample(1), &[2.0]);
    }
}
