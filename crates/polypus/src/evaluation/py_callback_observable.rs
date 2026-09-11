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
//! An optional cross-generation memo (opt-in) skips re-evaluating bitstrings
//! seen in earlier generations; it assumes the cost function is **pure**.

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

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

        // 3) One GIL section: evaluate each missing key exactly once. A Python
        //    exception is boxed verbatim so the entry point re-raises its
        //    original type across the FFI.
        if !missing.is_empty() {
            let computed: Vec<(&str, f64)> = Python::with_gil(|py| {
                let f = self.cost_fn.bind(py);
                missing
                    .iter()
                    .map(|&k| {
                        let v: f64 = f.call1((k,))?.extract()?;
                        Ok((k, v))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .map_err(|e| ObservableError::External(Box::new(e)))?;

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
        //    released. Every key is present in `lookup` by construction.
        let out = counts
            .par_iter()
            .map(|c| {
                let mut num = 0.0f64;
                let mut den = 0u64;
                for (k, &n) in c {
                    num += lookup[k.as_str()] * n as f64;
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
