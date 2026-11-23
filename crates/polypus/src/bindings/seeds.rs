//! The seed and the run id of an entry point (contract C-7): the seed
//! precedence, the OS-entropy fallback and the UUID-suffixed run id, each with
//! its entropy source injectable so its failure path can be tested.

use pyo3::prelude::*;

use super::adam::Adam;
use super::cmaes::CMAES;
use super::de::DE;
use super::pso::PSO;
use super::qng::QNG;
use super::rotosolve::Rotosolve;
use super::spsa::Spsa;
use crate::infrastructure::execution_config::random_id_bytes;
use crate::infrastructure::{BackendConfig, EntropyError};

/// Resolve the optimizer seed for `train` / `qml_train` (contract C-7).
///
/// Precedence: the explicit `seed` kwarg wins when provided; otherwise the
/// `seed` field pinned on the `DE`/`PSO`/`QNG`/`CMAES`/`Adam`/`SPSA`/`Rotosolve` instance;
/// otherwise a fresh OS-entropy value, drawn with `draw` (`random_seed` in production). The chosen
/// value both drives the optimizer and is reported back in the [`TrainResult`],
/// so the run can be replayed.
pub(super) fn resolve_optimizer_seed(
    kwarg_seed: Option<u64>,
    method_seed: Option<u64>,
    draw: impl FnOnce() -> Result<u64, EntropyError>,
) -> PyResult<u64> {
    seed_or_entropy(kwarg_seed.or(method_seed), draw)
}

/// `seed` if given, else a fresh OS-entropy value from `draw` (`random_seed` in
/// production) — the default-seed rule every entry point shares (contract C-7).
/// `draw` is only called when `seed` is `None`, so an explicit seed never touches
/// the OS entropy source. A failed draw raises `polypus.BackendError`, never a
/// panic (C-1).
pub(super) fn seed_or_entropy(
    seed: Option<u64>,
    draw: impl FnOnce() -> Result<u64, EntropyError>,
) -> PyResult<u64> {
    match seed {
        Some(seed) => Ok(seed),
        None => draw().map_err(|e| {
            crate::exceptions::backend_error_to_pyerr(crate::infrastructure::BackendError::from(e))
        }),
    }
}

/// Reject a resolved run seed the selected backend cannot take (contract C-7,
/// #393): `2^63` or more on Aer or CUNQA, whose simulators take a signed 64-bit
/// seed, raises `polypus.ConfigurationError` (a `ValueError`) instead of
/// reaching Aer masked to 63 bits. A seed drawn from OS entropy is always
/// below `2^63`, so only an explicit one can fail; `None` (no seed, `qmio`)
/// passes. Called by [`backend_config`](super::prepare::backend_config) for the
/// entry points that share it, and by native QML's own preparation in
/// `qml.rs`, before any backend is built.
pub(super) fn check_seed_for_backend(
    seed: Option<u64>,
    backend_config: &BackendConfig,
) -> PyResult<()> {
    match seed {
        Some(seed) => backend_config
            .check_seed(seed)
            .map_err(|e| crate::exceptions::configuration_error(e.to_string())),
        None => Ok(()),
    }
}

/// Suffix `base` with a UUID v4 so concurrent calls never share an
/// `ExecutionConfig::id` — which names SLURM families/allocations, temp files
/// and log streams. Used by `run_quantum_circuit`'s auto-generated id,
/// `train`'s and `qml_train`'s caller-supplied id (#45 / PR #70, #75): in
/// every case `base` is kept as a human-readable prefix on the effective id.
///
/// A failed OS-entropy read raises `polypus.BackendError`, never a panic (C-1,
/// C-7) — which is why the UUID is built from [`random_id_bytes`] rather than
/// drawn by the `uuid` crate itself, which panics on that failure (#264).
pub(super) fn unique_id(base: &str) -> PyResult<String> {
    unique_id_from(base, random_id_bytes)
}

/// `base` suffixed with a UUID v4 built from the bytes `draw` returns
/// (`random_id_bytes` in production): the body of [`unique_id`], with the
/// entropy source injectable so its failure path can be tested.
pub(super) fn unique_id_from(
    base: &str,
    draw: impl FnOnce() -> Result<[u8; 16], EntropyError>,
) -> PyResult<String> {
    let bytes = draw().map_err(|e| {
        crate::exceptions::backend_error_to_pyerr(crate::infrastructure::BackendError::from(e))
    })?;
    Ok(format!(
        "{}_{}",
        base,
        uuid::Builder::from_random_bytes(bytes).into_uuid()
    ))
}

/// Read the `seed` field pinned on the optimizer object passed as `method`,
/// whichever of `DE`/`PSO`/`QNG`/`CMAES`/`Adam`/`SPSA`/`Rotosolve` it is (`None` if it is none
/// of them — the type error is surfaced later by the dispatch that actually runs the
/// optimizer). `External` has no seed field: its library owns its seed (contract
/// C-7), so it is `None` too, and the run's seed then only fixes the shot
/// sampling.
pub(super) fn method_seed(method: &Bound<'_, PyAny>) -> Option<u64> {
    if let Ok(de) = method.extract::<PyRef<DE>>() {
        return de.seed;
    }
    if let Ok(pso) = method.extract::<PyRef<PSO>>() {
        return pso.seed;
    }
    if let Ok(qng) = method.extract::<PyRef<QNG>>() {
        return qng.seed;
    }
    if let Ok(cmaes) = method.extract::<PyRef<CMAES>>() {
        return cmaes.seed;
    }
    if let Ok(adam) = method.extract::<PyRef<Adam>>() {
        return adam.seed;
    }
    if let Ok(spsa) = method.extract::<PyRef<Spsa>>() {
        return spsa.seed;
    }
    if let Ok(rotosolve) = method.extract::<PyRef<Rotosolve>>() {
        return rotosolve.seed;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::execution_config::random_seed;

    /// Regression for #75, at the shared-helper boundary. `train`/`qml_train`
    /// take a caller-supplied `id` and used to pass it verbatim into
    /// `ExecutionConfig::id` — which names SLURM families/allocations, temp
    /// files and log streams — so two concurrent training runs sharing an `id`
    /// (the doc examples all use `id="run1"`/`id="qml_run"`) could clobber one
    /// another. All three id-bearing entry points now funnel through
    /// [`unique_id`], which keeps the caller's string as a human-readable prefix
    /// and appends a UUID v4 for uniqueness.
    ///
    /// This asserts that guarantee directly on `unique_id` rather than by
    /// calling `train`/`qml_train`: those need a live Python oracle
    /// (`polypus_python` → Qiskit/Aer) and `qml_train` needs Qiskit for its
    /// feature-map/ansatz composition, but the Rust suite is Python-runtime-free
    /// by design (see `.github/workflows/ci.yml` and
    /// `resolved_seed_drives_the_optimizer_deterministically`). The end-to-end
    /// `train`/`qml.train` id-uniqueness checks live in the Python suite
    /// (`tests/python/test_seed_reproducibility.py`), mirroring
    /// `run_quantum_circuit`'s `test_id_is_unique_across_identical_calls`.
    #[test]
    fn unique_id_keeps_prefix_and_is_unique_per_call() {
        // Same base (as with `train(..., id="run1")` twice) ⇒ unique ids that
        // both keep the supplied string as a prefix.
        for base in ["run1", "qml_run", "run_1_local"] {
            let id1 = unique_id(base).unwrap();
            let id2 = unique_id(base).unwrap();
            let prefix = format!("{base}_");
            assert!(
                id1.starts_with(&prefix) && id2.starts_with(&prefix),
                "the caller-supplied id must survive as a prefix, got {id1} / {id2}"
            );
            assert_ne!(
                id1, id2,
                "ids built from an identical base must differ (unique per call)"
            );
        }
    }

    #[test]
    fn resolve_optimizer_seed_follows_precedence() {
        // Explicit kwarg wins over the optimizer field...
        assert_eq!(
            resolve_optimizer_seed(Some(7), Some(9), random_seed).unwrap(),
            7
        );
        // ...the optimizer field is the fallback...
        assert_eq!(
            resolve_optimizer_seed(None, Some(9), random_seed).unwrap(),
            9
        );
        // ...and with neither set, a fresh entropy seed is drawn each time.
        assert_ne!(
            resolve_optimizer_seed(None, None, random_seed).unwrap(),
            resolve_optimizer_seed(None, None, random_seed).unwrap()
        );
    }

    /// An entropy source that cannot be read, as `random_seed` reports it.
    fn failing_draw() -> Result<u64, EntropyError> {
        Err(EntropyError::new(std::io::Error::other(
            "getrandom: device not available",
        )))
    }

    /// A draw that must never happen: an explicit seed must not touch OS entropy.
    fn forbidden_draw() -> Result<u64, EntropyError> {
        panic!("an explicit seed must not draw from the OS entropy source")
    }

    /// Assert `err` is the `polypus.BackendError` a failed entropy draw raises
    /// (contract C-1/C-7): catchable as `Exception`, never a `PanicException`.
    fn assert_entropy_failure(err: &PyErr) {
        pyo3::Python::initialize();
        Python::attach(|py| {
            assert!(
                err.is_instance_of::<crate::exceptions::BackendError>(py),
                "wrong exception class: {err}"
            );
            assert!(
                !err.is_instance_of::<pyo3::panic::PanicException>(py),
                "an entropy failure must not cross as a PanicException: {err}"
            );
            assert!(
                err.to_string().contains("OS entropy source unavailable"),
                "message lost: {err}"
            );
        });
    }

    #[test]
    fn seed_or_entropy_uses_an_explicit_seed_without_drawing() {
        assert_eq!(seed_or_entropy(Some(11), forbidden_draw).unwrap(), 11);
    }

    #[test]
    fn seed_or_entropy_returns_the_drawn_value() {
        assert_eq!(seed_or_entropy(None, || Ok(31)).unwrap(), 31);
    }

    #[test]
    fn seed_or_entropy_raises_backend_error_when_the_draw_fails() {
        let err = seed_or_entropy(None, failing_draw).expect_err("a failed draw must be an Err");
        assert_entropy_failure(&err);
    }

    #[test]
    fn resolve_optimizer_seed_propagates_a_failed_draw() {
        // Either explicit seed short-circuits the draw...
        assert_eq!(
            resolve_optimizer_seed(Some(7), None, forbidden_draw).unwrap(),
            7
        );
        assert_eq!(
            resolve_optimizer_seed(None, Some(9), forbidden_draw).unwrap(),
            9
        );
        // ...and with neither, the entropy failure surfaces as an error, not a panic.
        let err = resolve_optimizer_seed(None, None, failing_draw)
            .expect_err("a failed draw must be an Err");
        assert_entropy_failure(&err);
    }

    /// The run id's UUID is built from exactly the bytes drawn, with the v4
    /// version and RFC 4122 variant bits set — the id format is unchanged.
    #[test]
    fn unique_id_from_builds_a_v4_uuid_from_the_drawn_bytes() {
        let id = unique_id_from("run1", || Ok(std::array::from_fn(|i| i as u8))).unwrap();
        assert_eq!(id, "run1_00010203-0405-4607-8809-0a0b0c0d0e0f");
        let suffix = uuid::Uuid::parse_str(id.strip_prefix("run1_").unwrap()).unwrap();
        assert_eq!(suffix.get_version(), Some(uuid::Version::Random));
        assert_eq!(suffix.get_variant(), uuid::Variant::RFC4122);
    }

    /// Every entry point draws a run id, even with an explicit seed or on
    /// `qmio`: a failed draw must raise `polypus.BackendError` (catchable as
    /// `Exception`), never a `PanicException` (#264, contract C-1/C-7).
    #[test]
    fn unique_id_from_raises_backend_error_when_the_draw_fails() {
        let err = unique_id_from("run1", || {
            Err(EntropyError::for_run_id(std::io::Error::other(
                "getrandom: device not available",
            )))
        })
        .expect_err("a failed draw must be an Err");
        assert_entropy_failure(&err);
        Python::attach(|py| {
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyException>(py),
                "must be catchable as Exception: {err}"
            );
            assert!(
                err.to_string().contains("run id"),
                "message must say the run id draw failed: {err}"
            );
        });
    }

    /// The seed the binding resolves must actually make the optimizer
    /// reproducible (same seed ⇒ identical outcome) and its entropy fallback must
    /// vary (no seed ⇒ different outcomes) — the `train`/`qml.train` acceptance
    /// criterion, exercised at the binding's seed-resolution boundary. `train()`
    /// itself needs a Python oracle + `polypus_python`, which the
    /// Python-runtime-free Rust suite forbids, so the optimizer is driven
    /// directly with a pure-Rust oracle (mirroring
    /// `optimizers.rs::de_is_deterministic_for_a_fixed_seed`, one layer up).
    #[test]
    fn resolved_seed_drives_the_optimizer_deterministically() {
        use polypus_optimizers::{
            AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, EvaluationOracle,
            Optimizer,
        };

        struct Quadratic;
        impl EvaluationOracle for Quadratic {
            fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
                candidates
                    .iter()
                    .map(|c| -c.iter().map(|x| (x - 0.7).powi(2)).sum::<f64>())
                    .collect()
            }
        }

        let run = |kwarg: Option<u64>, field: Option<u64>| {
            let seed = resolve_optimizer_seed(kwarg, field, random_seed).unwrap();
            AlgorithmDifferentialEvolution
                .optimize(AlgorithmDifferentialEvolutionArgs {
                    oracle: Box::new(Quadratic),
                    population_size: 20,
                    generations: 40,
                    dimensions: 3,
                    tolerance: 1e-9,
                    patience: 20,
                    seed: Some(seed),
                    initial_point: None,
                    mutation_factor: None,
                    crossover_rate: None,
                })
                .expect("valid DE args optimize successfully")
        };

        // Same explicit seed ⇒ identical outcome (params, fitness, iters, converged).
        assert_eq!(run(Some(123), None), run(Some(123), None));
        // A method-field seed reproduces just as well.
        assert_eq!(run(None, Some(55)), run(None, Some(55)));
        // Omitted seed ⇒ (almost surely) different outcomes.
        assert_ne!(run(None, None), run(None, None));
    }

    /// A seed of 2^63 or more is a `polypus.ConfigurationError` (a
    /// `ValueError`) on Aer and CUNQA, never masked; below it, with no seed, or
    /// on the native backend, it passes (C-7, #393).
    #[test]
    fn check_seed_for_backend_rejects_two_to_the_63_on_aer_393() {
        use crate::infrastructure::execution_config::AER_SEED_LIMIT;
        pyo3::Python::initialize();
        Python::attach(|py| {
            let aer = BackendConfig::Local {
                sim_method: "automatic".to_string(),
                noise_model: None,
            };
            let err = check_seed_for_backend(Some(AER_SEED_LIMIT), &aer)
                .expect_err("2^63 is rejected on Aer");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            let configuration_error = crate::exceptions::configuration_error_type(py).unwrap();
            assert!(err.is_instance(py, configuration_error.as_any()));
            let message = err.value(py).to_string();
            assert!(message.contains("9223372036854775808"), "{message}");
            assert!(message.contains("backend=\"aer\""), "{message}");
            check_seed_for_backend(Some(AER_SEED_LIMIT - 1), &aer).unwrap();
            check_seed_for_backend(None, &aer).unwrap();
            let native = BackendConfig::LocalNative { fusion: true };
            check_seed_for_backend(Some(u64::MAX), &native).unwrap();
        });
    }
}
