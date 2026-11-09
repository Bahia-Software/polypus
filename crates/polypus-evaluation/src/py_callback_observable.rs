//! Fallback [`CostObservable`] wrapping an arbitrary user Python cost function.
//!
//! This preserves the historical `expectation_function=<callable>` API for
//! costs that the declarative [`QuboObservable`](polypus_observable::QuboObservable)
//! / [`IsingObservable`](polypus_observable::IsingObservable) cannot express, but
//! it removes the old bottleneck: instead of calling Python once per bitstring
//! per candidate inside a Python aggregation loop, it
//!
//! 1. deduplicates the union of unique bitstrings across the **whole batch**,
//! 2. calls the cost function once per unique bitstring in a **single GIL
//!    section** (never from a rayon worker — that would only re-serialise on the
//!    GIL), then
//! 3. performs the count-weighted aggregation in Rust, in parallel over
//!    candidates with the GIL released.
//!
//! Both the calls and each candidate's sum go in ascending bitstring order, never
//! `HashMap` iteration order, which differs between maps and between processes
//! and would move the last bit of a sum.
//!
//! An optional cross-generation memo (opt-in) skips re-evaluating bitstrings
//! seen in earlier generations; it assumes the cost function is **pure**.

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use polypus_infrastructure::{attach_or, DisplaySafePyErr, InterpreterUnavailable};
use polypus_observable::{CostObservable, ObservableError};
use pyo3::prelude::*;
use rayon::prelude::*;

/// A cost observable backed by a Python callable `bitstring -> float`.
pub struct PyCallbackObservable {
    /// The user cost function; `Send + Sync` so it can be shared across oracle
    /// worker threads.
    cost_fn: Py<PyAny>,
    /// Optional cross-generation memo `bitstring -> value`. `RwLock` (not a bare
    /// map) because `QmlOracle` evaluates candidates from concurrent
    /// `spawn_blocking` tasks, so the observable must be `Sync`. Present only
    /// when caching is enabled; the cost function must be pure for it to be sound.
    cache: Option<RwLock<HashMap<String, f64>>>,
}

impl PyCallbackObservable {
    /// Wrap `cost_fn`. With `cache = true`, values are memoised across
    /// generations (valid only if `cost_fn` is pure); with `cache = false`, only
    /// per-batch deduplication is applied.
    pub fn new(cost_fn: Py<PyAny>, cache: bool) -> Self {
        Self {
            cost_fn,
            cache: if cache {
                Some(RwLock::new(HashMap::new()))
            } else {
                None
            },
        }
    }
}

impl CostObservable for PyCallbackObservable {
    fn locality(&self) -> polypus_observable::ReducerLocality {
        // Holds a Python callback and must run in this process under the GIL, so it
        // can never be pushed to a remote reduce-at-source worker.
        polypus_observable::ReducerLocality::Local
    }

    fn expectation_batch(
        &self,
        counts: &[HashMap<String, u64>],
    ) -> Result<Vec<f64>, ObservableError> {
        // 1) Union of unique bitstrings across the whole batch.
        let mut unique: HashSet<&str> = HashSet::new();
        for c in counts {
            for k in c.keys() {
                unique.insert(k.as_str());
            }
        }

        // 2) Prefill the batch-local lookup from the cross-generation cache (if
        //    enabled); collect the still-unknown keys.
        let mut lookup: HashMap<&str, f64> = HashMap::with_capacity(unique.len());
        let mut missing: Vec<&str> = Vec::new();
        if let Some(cache) = &self.cache {
            let guard = cache.read().unwrap_or_else(|p| p.into_inner());
            for &k in &unique {
                match guard.get(k) {
                    Some(&v) => {
                        lookup.insert(k, v);
                    }
                    None => missing.push(k),
                }
            }
        } else {
            missing.extend(unique.iter().copied());
        }
        // Call the cost function in a fixed order, so a callback with side effects
        // (logging, a seeded noise model) sees the same sequence on every run.
        missing.sort_unstable();

        // 3) One GIL section: evaluate each missing key exactly once. A Python
        //    exception is boxed verbatim so the entry point re-raises its
        //    original type across the FFI, as a `DisplaySafePyErr` so that
        //    formatting the error while detached cannot panic. This runs while
        //    the run is detached, so the attach is fallible: an unreachable
        //    interpreter is `InterpreterUnavailable` (docs/ENGINEERING.md §9).
        if !missing.is_empty() {
            let computed: Vec<(&str, f64)> = attach_or(
                || Err(ObservableError::External(Box::new(InterpreterUnavailable))),
                |py| {
                    let f = self.cost_fn.bind(py);
                    missing
                        .iter()
                        .map(|&k| {
                            let v: f64 = f.call1((k,))?.extract()?;
                            Ok((k, v))
                        })
                        .collect::<PyResult<Vec<_>>>()
                        .map_err(|e| ObservableError::External(Box::new(DisplaySafePyErr::from(e))))
                },
            )?;

            if let Some(cache) = &self.cache {
                let mut guard = cache.write().unwrap_or_else(|p| p.into_inner());
                for &(k, v) in &computed {
                    guard.insert(k.to_string(), v);
                }
            }
            for (k, v) in computed {
                lookup.insert(k, v);
            }
        }

        // 4) Count-weighted aggregation in Rust, parallel over candidates, GIL
        //    released. Every key is present in `lookup` by construction. Summed in
        //    ascending bitstring order so the result does not depend on `HashMap`
        //    iteration order.
        let out = counts
            .par_iter()
            .map(|c| {
                let mut outcomes: Vec<(&str, u64)> =
                    c.iter().map(|(k, &n)| (k.as_str(), n)).collect();
                outcomes.sort_unstable_by(|a, b| a.0.cmp(b.0));
                let mut num = 0.0f64;
                let mut den = 0u64;
                for (k, n) in outcomes {
                    num += lookup[k] * n as f64;
                    den += n;
                }
                if den == 0 {
                    0.0
                } else {
                    num / den as f64
                }
            })
            .collect();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polypus_observable::ReducerLocality;
    use pyo3::types::{IntoPyDict, PyDict};

    fn counts(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|&(k, n)| (k.to_string(), n)).collect()
    }

    #[test]
    fn reducer_is_local_because_it_holds_a_gil_bound_callback() {
        // A Python-callback reducer can never be pushed to a remote reduce-at-source
        // worker (plan §7.1): it must run in this process under the GIL. Pin that
        // classification so a future Planner never treats it as `Portable`.
        pyo3::Python::initialize();
        let obs = Python::attach(|py| PyCallbackObservable::new(py.None(), false));
        assert_eq!(obs.locality(), ReducerLocality::Local);
    }

    #[test]
    fn a_raising_callback_is_boxed_as_a_display_safe_py_err() {
        // The FFI edge re-raises only a boxed `DisplaySafePyErr` verbatim (contract
        // C-1), and only that carrier is safe to format while detached, so the
        // callback's exception must be boxed as one, its class kept.
        pyo3::Python::initialize();
        let obs = Python::attach(|py| {
            let cost_fn = PyModule::from_code(
                py,
                c"def cost(bitstring):\n    raise ZeroDivisionError('cost divided by zero')\n",
                c"stand_in.py",
                c"stand_in",
            )
            .and_then(|module| module.getattr("cost"))
            .expect("the stand-in callback compiles")
            .unbind();
            PyCallbackObservable::new(cost_fn, false)
        });
        let counts = [HashMap::from([("01".to_string(), 3)])];
        match obs.expectation_batch(&counts) {
            Err(ObservableError::External(boxed)) => {
                let err = boxed
                    .downcast_ref::<DisplaySafePyErr>()
                    .expect("the callback's exception is boxed as a DisplaySafePyErr");
                Python::attach(|py| {
                    assert!(err
                        .as_py_err()
                        .is_instance_of::<pyo3::exceptions::PyZeroDivisionError>(py));
                });
                assert_eq!(err.to_string(), "ZeroDivisionError: cost divided by zero");
            }
            other => panic!("expected ObservableError::External, got {other:?}"),
        }
    }

    /// The observable is evaluated while the run is detached. When the
    /// interpreter cannot be attached to, it fails with `InterpreterUnavailable`
    /// boxed in `ObservableError::External` and never calls the cost function,
    /// which records every call.
    #[test]
    fn fails_typed_without_calling_python_when_attaching_is_forbidden() {
        use polypus_infrastructure::InterpreterUnavailable;
        pyo3::Python::initialize();
        let (module, observables) = Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                c"calls = []\ndef cost(bitstring):\n    calls.append(bitstring)\n    return 1.0\n",
                c"observable_recorder.py",
                c"observable_recorder",
            )
            .expect("the recorder compiles");
            let cost = module
                .getattr("cost")
                .expect("the callback exists")
                .unbind();
            let observables = [
                PyCallbackObservable::new(cost.clone_ref(py), false),
                PyCallbackObservable::new(cost, true),
            ];
            (module.unbind(), observables)
        });
        for observable in observables {
            let result = crate::test_support::run_while_attaching_is_forbidden(move || {
                observable.expectation_batch(&[HashMap::from([("01".to_string(), 3)])])
            });
            match result {
                Err(ObservableError::External(boxed)) => {
                    assert!(boxed.is::<InterpreterUnavailable>(), "{boxed}")
                }
                other => panic!("expected External(InterpreterUnavailable), got {other:?}"),
            }
        }
        Python::attach(|py| {
            let calls = module.bind(py).getattr("calls").unwrap();
            assert_eq!(
                calls.len().unwrap(),
                0,
                "the cost function must not be called"
            );
        });
    }

    #[test]
    fn mean_does_not_depend_on_hashmap_iteration_order() {
        // Not dyadic, so the sum depends on the order its terms are added in. Four
        // shots make the division by the shot count exact, so that difference
        // reaches the mean.
        let ascending = ((0.1 * 2.0 + 0.2) + 0.7) / 4.0;
        assert_ne!(ascending, ((0.2 + 0.7) + 0.1 * 2.0) / 4.0);
        pyo3::Python::initialize();
        let obs = Python::attach(|py| {
            let values = [("001", 0.1), ("010", 0.2), ("100", 0.7)]
                .into_py_dict(py)
                .expect("a str -> float dict");
            let cost_fn = values.getattr("__getitem__").expect("a dict lookup");
            PyCallbackObservable::new(cost_fn.unbind(), false)
        });
        // Each map gets its own `RandomState`, hence its own iteration order.
        let batch: Vec<_> = (0..128)
            .map(|_| counts(&[("001", 2), ("010", 1), ("100", 1)]))
            .collect();
        for ev in obs.expectation_batch(&batch).expect("all keys valued") {
            assert_eq!(ev, ascending, "summed in ascending bitstring order");
        }
    }

    #[test]
    fn cost_function_is_called_once_per_bitstring_in_ascending_order() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let namespace = PyDict::new(py);
            py.run(
                c"calls = []\ndef cost(bitstring):\n    calls.append(bitstring)\n    return 0.5\n",
                Some(&namespace),
                None,
            )
            .expect("the recorder compiles");
            let item = |name: &str| {
                namespace
                    .get_item(name)
                    .expect("a str key")
                    .expect("defined by the recorder")
            };
            let obs = PyCallbackObservable::new(item("cost").unbind(), false);
            // Eight keys, so `HashSet` order is ascending by chance only once in
            // 8! = 40320 runs.
            let batch = [
                counts(&[("110", 1), ("011", 2), ("101", 1), ("000", 1)]),
                counts(&[("100", 3), ("011", 4), ("001", 5), ("111", 1), ("010", 2)]),
            ];
            obs.expectation_batch(&batch).expect("a healthy batch");
            let calls: Vec<String> = item("calls").extract().expect("a list of str");
            assert_eq!(
                calls,
                ["000", "001", "010", "011", "100", "101", "110", "111"]
            );
        });
    }
}
