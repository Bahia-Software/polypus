use polypus_optimizers::{AlgorithmDifferentialEvolution, DeStrategy, OptimizerError};
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
    /// DE/rand/1/bin's mutation factor `F`, in `(0, 2]`; `None` is 0.8.
    /// SHADE and L-SHADE adapt `F` themselves, so it must stay `None` for
    /// them. Checked when it or `strategy` is passed or set.
    pub mutation_factor: Option<f64>,
    /// DE/rand/1/bin's crossover rate `CR`, in `[0, 1]`; `None` is 0.7. One
    /// coordinate of each trial always comes from the mutant. `None` for SHADE
    /// and L-SHADE, as `mutation_factor`.
    pub crossover_rate: Option<f64>,
}

/// Parse a `strategy` name; an unknown one is a `ValueError` naming the valid
/// names.
fn parse_strategy(name: &str) -> PyResult<DeStrategy> {
    name.parse()
        .map_err(|e: OptimizerError| crate::exceptions::configuration_error(e.to_string()))
}

/// Check `F` and `CR` against `strategy`; a violation is a `ValueError`.
fn check_rates(
    strategy: DeStrategy,
    mutation_factor: Option<f64>,
    crossover_rate: Option<f64>,
) -> PyResult<()> {
    AlgorithmDifferentialEvolution::check_rates(strategy, mutation_factor, crossover_rate)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pymethods]
impl DE {
    #[new]
    #[pyo3(signature = (generations = 100, population_size = 50, tolerance = 0.01, patience = 20, seed = None, strategy = "rand1bin", mutation_factor = None, crossover_rate = None))]
    pub fn new(
        generations: u32,
        population_size: u32,
        tolerance: f64,
        patience: u32,
        seed: Option<u64>,
        strategy: &str,
        mutation_factor: Option<f64>,
        crossover_rate: Option<f64>,
    ) -> PyResult<Self> {
        let strategy = parse_strategy(strategy)?;
        check_rates(strategy, mutation_factor, crossover_rate)?;
        Ok(DE {
            generations,
            population_size,
            tolerance,
            patience,
            seed,
            strategy,
            mutation_factor,
            crossover_rate,
        })
    }

    /// How trial vectors are built: `"rand1bin"` (the default; DE/rand/1/bin
    /// with a fixed `F` and `CR`), `"shade"` (SHADE: current-to-pbest/1
    /// with `F` and `CR` adapted from a success history) or `"lshade"`
    /// (L-SHADE: SHADE with a population that shrinks linearly towards 4; it
    /// takes `generations` as an evaluation budget, `population_size ×
    /// (generations + 1)`, and spends it in more, smaller generations).
    #[getter]
    fn strategy(&self) -> &'static str {
        self.strategy.as_str()
    }

    #[setter]
    fn set_strategy(&mut self, strategy: &str) -> PyResult<()> {
        let strategy = parse_strategy(strategy)?;
        check_rates(strategy, self.mutation_factor, self.crossover_rate)?;
        self.strategy = strategy;
        Ok(())
    }

    /// DE/rand/1/bin's mutation factor `F`, which scales the difference
    /// vector: in `(0, 2]`, or `None` for 0.8. SHADE and L-SHADE adapt `F`
    /// themselves and raise `ValueError` for a value.
    #[getter]
    fn mutation_factor(&self) -> Option<f64> {
        self.mutation_factor
    }

    #[setter]
    fn set_mutation_factor(&mut self, mutation_factor: Option<f64>) -> PyResult<()> {
        check_rates(self.strategy, mutation_factor, self.crossover_rate)?;
        self.mutation_factor = mutation_factor;
        Ok(())
    }

    /// DE/rand/1/bin's crossover rate `CR`, the probability that a coordinate
    /// of a trial comes from the mutant (one always does): in `[0, 1]`, or
    /// `None` for 0.7. SHADE and L-SHADE adapt `CR` themselves and raise
    /// `ValueError` for a value.
    #[getter]
    fn crossover_rate(&self) -> Option<f64> {
        self.crossover_rate
    }

    #[setter]
    fn set_crossover_rate(&mut self, crossover_rate: Option<f64>) -> PyResult<()> {
        check_rates(self.strategy, self.mutation_factor, crossover_rate)?;
        self.crossover_rate = crossover_rate;
        Ok(())
    }
}
