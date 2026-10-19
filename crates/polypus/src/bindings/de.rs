use polypus_optimizers::{DeStrategy, OptimizerError};
use pyo3::prelude::*;

/// Differential Evolution optimizer configuration.
#[pyclass(module = "polypus")]
pub struct DE {
    #[pyo3(get, set)]
    pub generations: u32,
    #[pyo3(get, set)]
    pub population_size: u32,
    /// Minimum cumulative improvement of the best fitness over the last
    /// `patience` generations below which the run stops early. DE maximises, so
    /// this is a threshold on quality progress in the cost/fitness units, not a
    /// population-spread threshold (contract C-5).
    #[pyo3(get, set)]
    pub tolerance: f64,
    /// Number of generations over which the best-fitness improvement is measured
    /// for the stagnation early stop; no stop can fire before generation
    /// `patience`. Defaults to 20.
    #[pyo3(get, set)]
    pub patience: u32,
    /// Optional RNG seed pinned on the optimizer object. Consumed by
    /// `train`/`qml.train` per the precedence rule (contract C-7): the explicit
    /// `seed` kwarg passed to the call wins; this field is the fallback; a fresh
    /// OS-entropy value is used when neither is set. `None` by default.
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    /// How trial vectors are built. Python reads and writes it by name
    /// (`"rand1bin"`, `"shade"`, `"lshade"`); an unknown name raises
    /// `ValueError` as soon as it is passed or set, so a run never starts with
    /// one.
    pub strategy: DeStrategy,
}

/// Parse a `strategy` name; an unknown one is a `ValueError` naming the valid
/// names.
fn parse_strategy(name: &str) -> PyResult<DeStrategy> {
    name.parse()
        .map_err(|e: OptimizerError| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pymethods]
impl DE {
    #[new]
    #[pyo3(signature = (generations = 100, population_size = 50, tolerance = 0.01, patience = 20, seed = None, strategy = "rand1bin"))]
    pub fn new(
        generations: u32,
        population_size: u32,
        tolerance: f64,
        patience: u32,
        seed: Option<u64>,
        strategy: &str,
    ) -> PyResult<Self> {
        Ok(DE {
            generations,
            population_size,
            tolerance,
            patience,
            seed,
            strategy: parse_strategy(strategy)?,
        })
    }

    /// How trial vectors are built: `"rand1bin"` (the default; DE/rand/1/bin
    /// with fixed `F = 0.8`, `CR = 0.7`), `"shade"` (SHADE: current-to-pbest/1
    /// with `F` and `CR` adapted from a success history) or `"lshade"`
    /// (L-SHADE: SHADE with a population that shrinks linearly towards 4).
    #[getter]
    fn strategy(&self) -> &'static str {
        self.strategy.as_str()
    }

    #[setter]
    fn set_strategy(&mut self, strategy: &str) -> PyResult<()> {
        self.strategy = parse_strategy(strategy)?;
        Ok(())
    }
}
