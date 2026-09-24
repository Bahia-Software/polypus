//! Supervised objectives for `qml.train` with `y_train`: each training sample's
//! counts are scored against that sample's label.
//!
//! [`QmlOracle`](crate::QmlOracle) pairs the counts with the labels and hands both
//! to a [`SupervisedObjective`]. Two Python-backed objectives exist:
//!
//! - [`PyLabelledCost`]: a per-shot `f(bitstring, label)`, called once per unique
//!   `(label, bitstring)` pair of a batch in one GIL section, like
//!   [`PyCallbackObservable`](crate::PyCallbackObservable).
//! - [`PySampleCost`]: a per-sample `g(counts, label)` over the whole distribution,
//!   so losses that are not linear in the outcome probabilities (a log-likelihood)
//!   can be expressed.
//!
//! Both sum, and build the dicts they pass to Python, in sorted-bitstring order, so
//! results do not depend on `HashMap` iteration order.

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use polypus_infrastructure::Counts;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
use rayon::prelude::*;

use crate::EvaluationError;

/// A training sample's label. The edge gives every label of a run the same kind:
/// `Class` (passed to Python as `int`) when all labels are integers, `Real`
/// (passed as `float`) otherwise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Label {
    /// An integer label, such as a class index.
    Class(i64),
    /// A real-valued target.
    Real(f64),
}

impl Label {
    /// Deduplication key. `Real` compares by bit pattern, so `0.0` and `-0.0`
    /// stay distinct.
    fn key(self) -> LabelKey {
        match self {
            Label::Class(value) => LabelKey::Class(value),
            Label::Real(value) => LabelKey::Real(value.to_bits()),
        }
    }

    /// The value passed to Python: an `int` or a `float`.
    fn to_object<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Label::Class(value) => value.into_bound_py_any(py),
            Label::Real(value) => value.into_bound_py_any(py),
        }
    }
}

/// Hashable, ordered form of a [`Label`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum LabelKey {
    Class(i64),
    Real(u64),
}

impl LabelKey {
    /// The label this key was made from.
    fn label(self) -> Label {
        match self {
            LabelKey::Class(value) => Label::Class(value),
            LabelKey::Real(bits) => Label::Real(f64::from_bits(bits)),
        }
    }
}

/// Scores counts against labels: `counts[i]` came from a sample labelled
/// `labels[i]`, and the result holds one score per pair, in order. Scores are
/// maximised; the caller checks their count and that they are finite.
pub trait SupervisedObjective: Send + Sync {
    /// Score every `(counts[i], labels[i])` pair.
    fn score_batch(&self, counts: &[Counts], labels: &[Label])
        -> Result<Vec<f64>, EvaluationError>;
}

/// Per-shot score `f(bitstring, label) -> float`: a sample scores the
/// count-weighted mean of `f` over its bitstrings.
pub struct PyLabelledCost {
    cost_fn: Py<PyAny>,
    /// Cross-generation memo keyed by `(label, bitstring)`, for
    /// `polypus.CachedCost`. Sound only for a pure `f`.
    cache: Option<RwLock<HashMap<(LabelKey, String), f64>>>,
}

impl PyLabelledCost {
    /// Wrap `cost_fn`; `cache` enables the cross-generation memo.
    pub fn new(cost_fn: Py<PyAny>, cache: bool) -> Self {
        Self {
            cost_fn,
            cache: cache.then(|| RwLock::new(HashMap::new())),
        }
    }
}

impl SupervisedObjective for PyLabelledCost {
    fn score_batch(
        &self,
        counts: &[Counts],
        labels: &[Label],
    ) -> Result<Vec<f64>, EvaluationError> {
        // Unique (label, bitstring) pairs of the whole batch.
        let mut unique: HashSet<(LabelKey, &str)> = HashSet::new();
        for (sample_counts, &label) in counts.iter().zip(labels) {
            let key = label.key();
            for bitstring in sample_counts.keys() {
                unique.insert((key, bitstring.as_str()));
            }
        }

        // Reuse memoised values and collect the pairs still to evaluate.
        let mut lookup: HashMap<(LabelKey, &str), f64> = HashMap::with_capacity(unique.len());
        let mut missing: Vec<(LabelKey, &str)> = Vec::new();
        match &self.cache {
            Some(cache) => {
                let memo = cache.read().unwrap_or_else(|p| p.into_inner());
                for &(key, bitstring) in &unique {
                    match memo.get(&(key, bitstring.to_string())) {
                        Some(&value) => {
                            lookup.insert((key, bitstring), value);
                        }
                        None => missing.push((key, bitstring)),
                    }
                }
            }
            None => missing.extend(unique.iter().copied()),
        }
        // A fixed call order, so a callback with side effects sees the same
        // sequence on every run.
        missing.sort_unstable();

        // One GIL section for all of them. A Python exception is kept as is, so
        // the entry point re-raises its original type.
        if !missing.is_empty() {
            let computed: Vec<f64> = Python::with_gil(|py| {
                let f = self.cost_fn.bind(py);
                missing
                    .iter()
                    .map(|&(key, bitstring)| {
                        f.call1((bitstring, key.label().to_object(py)?))?
                            .extract::<f64>()
                    })
                    .collect::<PyResult<Vec<f64>>>()
            })
            .map_err(EvaluationError::Python)?;

            if let Some(cache) = &self.cache {
                let mut memo = cache.write().unwrap_or_else(|p| p.into_inner());
                for (&(key, bitstring), &value) in missing.iter().zip(&computed) {
                    memo.insert((key, bitstring.to_string()), value);
                }
            }
            lookup.extend(missing.into_iter().zip(computed));
        }

        // Count-weighted mean per sample, GIL released, summed in sorted order.
        let scores = counts
            .par_iter()
            .zip(labels)
            .map(|(sample_counts, &label)| {
                let key = label.key();
                let mut outcomes: Vec<(&str, u64)> = sample_counts
                    .iter()
                    .map(|(bitstring, &n)| (bitstring.as_str(), n))
                    .collect();
                outcomes.sort_unstable_by(|a, b| a.0.cmp(b.0));
                let mut weighted = 0.0f64;
                let mut shots = 0u64;
                for (bitstring, n) in outcomes {
                    weighted += lookup[&(key, bitstring)] * n as f64;
                    shots += n;
                }
                // An empty map scores 0.0, as in `CostObservable`.
                if shots == 0 {
                    0.0
                } else {
                    weighted / shots as f64
                }
            })
            .collect();
        Ok(scores)
    }
}

/// Per-sample score `g(counts, label) -> float` (`polypus.SampleCost`), where
/// `counts` is the sample's whole distribution as a `dict` with sorted keys. One
/// call per `(candidate, sample)`, one GIL section per batch.
pub struct PySampleCost {
    cost_fn: Py<PyAny>,
}

impl PySampleCost {
    /// Wrap `cost_fn`, called as `cost_fn(counts, label) -> float`.
    pub fn new(cost_fn: Py<PyAny>) -> Self {
        Self { cost_fn }
    }
}

impl SupervisedObjective for PySampleCost {
    fn score_batch(
        &self,
        counts: &[Counts],
        labels: &[Label],
    ) -> Result<Vec<f64>, EvaluationError> {
        Python::with_gil(|py| {
            let g = self.cost_fn.bind(py);
            counts
                .iter()
                .zip(labels)
                .map(|(sample_counts, &label)| {
                    let mut outcomes: Vec<(&String, &u64)> = sample_counts.iter().collect();
                    outcomes.sort_unstable_by(|a, b| a.0.cmp(b.0));
                    let dict = PyDict::new(py);
                    for (bitstring, n) in outcomes {
                        dict.set_item(bitstring, n)?;
                    }
                    g.call1((dict, label.to_object(py)?))?.extract::<f64>()
                })
                .collect::<PyResult<Vec<f64>>>()
        })
        .map_err(EvaluationError::Python)
    }
}

/// These need an interpreter but no Qiskit: the callbacks are small Python
/// functions that record how they were called.
#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyModule;
    use std::ffi::CString;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Compiled once per test under a unique module name: `from_code` registers it
    /// in `sys.modules`, and parallel tests must not share one `calls` list.
    const RECORDERS: &std::ffi::CStr = cr#"
calls = []


def per_shot(bitstring, label):
    calls.append((bitstring, label, type(label).__name__))
    # Distinct per (label, bitstring) and not dyadic, so a wrong pairing shows.
    return int(bitstring, 2) + 10 * label + 0.1


def correct(bitstring, label):
    calls.append((bitstring, label, type(label).__name__))
    return 1.0 if int(bitstring, 2) == label else 0.0


def whole_sample(counts, label):
    calls.append((list(counts.items()), label, type(label).__name__))
    return sum(n for b, n in counts.items() if int(b, 2) == label) / sum(counts.values())


def raises(*_args):
    raise ValueError("user objective blew up")
"#;

    fn recorders(py: Python<'_>) -> Bound<'_, PyModule> {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let name = format!(
            "supervised_recorders_{}",
            SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let file = CString::new(format!("{name}.py")).expect("no interior NUL");
        let module = CString::new(name).expect("no interior NUL");
        PyModule::from_code(py, RECORDERS, file.as_c_str(), module.as_c_str())
            .expect("the recorder module compiles")
    }

    fn callback(module: &Bound<'_, PyModule>, name: &str) -> Py<PyAny> {
        module.getattr(name).expect("the recorder exists").unbind()
    }

    /// One recorded call: `(bitstring or sorted counts items, label, label type name)`.
    type Call<T> = (T, f64, String);

    /// Every recorded call, in order.
    fn calls<'py, T: FromPyObject<'py>>(module: &Bound<'py, PyModule>) -> Vec<Call<T>> {
        module
            .getattr("calls")
            .expect("the recorder list exists")
            .extract()
            .expect("the calls are (arg, label, type) tuples")
    }

    fn counts(entries: &[(&str, u64)]) -> Counts {
        entries.iter().map(|&(b, n)| (b.to_string(), n)).collect()
    }

    /// Count-weighted mean summed in sorted-bitstring order, as `PyLabelledCost`
    /// computes it.
    fn sorted_weighted_mean(entries: &[(&str, u64)], f: impl Fn(&str) -> f64) -> f64 {
        let mut sorted = entries.to_vec();
        sorted.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let shots: u64 = sorted.iter().map(|&(_, n)| n).sum();
        sorted.iter().map(|&(b, n)| f(b) * n as f64).sum::<f64>() / shots as f64
    }

    #[test]
    fn labelled_cost_calls_python_once_per_unique_label_and_bitstring() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = recorders(py);
            let cost = PyLabelledCost::new(callback(&module, "per_shot"), false);
            let batch = [
                counts(&[("00", 3), ("01", 5)]),
                counts(&[("01", 2), ("11", 7)]),
                counts(&[("00", 1), ("11", 4)]),
                counts(&[("00", 6), ("01", 6)]),
            ];
            let labels = [
                Label::Class(0),
                Label::Class(1),
                Label::Class(0),
                Label::Class(1),
            ];

            let scores = py
                .allow_threads(|| cost.score_batch(&batch, &labels))
                .expect("a healthy batch scores");

            // Two labels times three bitstrings: six calls, not one per occurrence.
            let recorded: Vec<Call<String>> = calls(&module);
            assert_eq!(
                recorded.len(),
                6,
                "one call per unique (label, bitstring): {recorded:?}"
            );
            // Ordered by label, then bitstring.
            let order: Vec<(String, f64)> =
                recorded.iter().map(|(b, l, _)| (b.clone(), *l)).collect();
            assert_eq!(
                order,
                [
                    ("00", 0.0),
                    ("01", 0.0),
                    ("11", 0.0),
                    ("00", 1.0),
                    ("01", 1.0),
                    ("11", 1.0)
                ]
                .map(|(b, l)| (b.to_string(), l))
            );

            let per_shot = |label: i64| {
                move |b: &str| i64::from_str_radix(b, 2).unwrap() as f64 + 10.0 * label as f64 + 0.1
            };
            let expected = [
                sorted_weighted_mean(&[("00", 3), ("01", 5)], per_shot(0)),
                sorted_weighted_mean(&[("01", 2), ("11", 7)], per_shot(1)),
                sorted_weighted_mean(&[("00", 1), ("11", 4)], per_shot(0)),
                sorted_weighted_mean(&[("00", 6), ("01", 6)], per_shot(1)),
            ];
            assert_eq!(
                scores, expected,
                "each sample is averaged against its own label"
            );
        });
    }

    #[test]
    fn labelled_cost_hands_python_int_classes_and_float_targets() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = recorders(py);
            let cost = PyLabelledCost::new(callback(&module, "per_shot"), false);
            let batch = [counts(&[("1", 4)]), counts(&[("1", 4)])];

            cost.score_batch(&batch, &[Label::Class(-1), Label::Real(2.5)])
                .expect("both label kinds score");

            let mut types: Vec<(f64, String)> = calls::<String>(&module)
                .into_iter()
                .map(|(_, label, ty)| (label, ty))
                .collect();
            types.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert_eq!(
                types,
                [(-1.0, "int".to_string()), (2.5, "float".to_string())],
                "a Class label is a Python int and a Real label a Python float"
            );
        });
    }

    #[test]
    fn cached_labelled_cost_reuses_pairs_seen_in_earlier_batches() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = recorders(py);
            let cost = PyLabelledCost::new(callback(&module, "correct"), true);
            let labels = [Label::Class(0), Label::Class(1)];
            let first = [counts(&[("0", 5), ("1", 3)]), counts(&[("0", 2), ("1", 6)])];

            let a = cost
                .score_batch(&first, &labels)
                .expect("first batch scores");
            assert_eq!(
                calls::<String>(&module).len(),
                4,
                "(0|1) x (0|1): four new pairs"
            );

            // A later generation over the same pairs makes no call…
            let b = cost
                .score_batch(&first, &labels)
                .expect("second batch scores");
            assert_eq!(
                calls::<String>(&module).len(),
                4,
                "every pair came from the memo"
            );
            assert_eq!(a, b, "a memoised value is the value first computed");
            assert_eq!(a, vec![5.0 / 8.0, 6.0 / 8.0]);

            // …and only a new pair reaches Python.
            cost.score_batch(&[counts(&[("0", 1)])], &[Label::Class(7)])
                .expect("a new label scores");
            assert_eq!(calls::<String>(&module).len(), 5, "only (7, '0') is new");
        });
    }

    #[test]
    fn labelled_cost_scores_an_empty_sample_as_zero() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = recorders(py);
            let cost = PyLabelledCost::new(callback(&module, "per_shot"), false);
            let scores = cost
                .score_batch(&[Counts::new()], &[Label::Class(1)])
                .expect("an empty map is not an error");
            assert_eq!(scores, vec![0.0], "same contract as CostObservable");
            assert!(calls::<String>(&module).is_empty(), "nothing to evaluate");
        });
    }

    #[test]
    fn sample_cost_sees_its_whole_distribution_sorted_and_its_label() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = recorders(py);
            let cost = PySampleCost::new(callback(&module, "whole_sample"));
            // Inserted out of order; the dict passed to Python is still sorted.
            let batch = [
                counts(&[("11", 1), ("00", 2), ("10", 5)]),
                counts(&[("10", 3), ("01", 1)]),
            ];
            let labels = [Label::Class(2), Label::Class(1)];

            let scores = py
                .allow_threads(|| cost.score_batch(&batch, &labels))
                .expect("a healthy batch scores");

            let recorded: Vec<Call<Vec<(String, u64)>>> = calls(&module);
            let expected_calls = vec![
                (
                    vec![
                        ("00".to_string(), 2),
                        ("10".to_string(), 5),
                        ("11".to_string(), 1),
                    ],
                    2.0,
                    "int".to_string(),
                ),
                (
                    vec![("01".to_string(), 1), ("10".to_string(), 3)],
                    1.0,
                    "int".to_string(),
                ),
            ];
            assert_eq!(
                recorded, expected_calls,
                "one call per sample, sorted counts, own label"
            );
            assert_eq!(scores, vec![5.0 / 8.0, 1.0 / 4.0]);
        });
    }

    #[test]
    fn a_raising_objective_is_carried_as_the_original_python_exception() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = recorders(py);
            let batch = [counts(&[("0", 1)])];
            let labels = [Label::Class(0)];
            let objectives: [Box<dyn SupervisedObjective>; 2] = [
                Box::new(PyLabelledCost::new(callback(&module, "raises"), false)),
                Box::new(PySampleCost::new(callback(&module, "raises"))),
            ];
            for objective in objectives {
                match objective.score_batch(&batch, &labels) {
                    Err(EvaluationError::Python(err)) => {
                        assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                        assert!(err.to_string().contains("user objective blew up"));
                    }
                    other => panic!("expected the verbatim ValueError, got {other:?}"),
                }
            }
        });
    }
}
