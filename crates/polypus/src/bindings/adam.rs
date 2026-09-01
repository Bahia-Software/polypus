use pyo3::prelude::*;

/// Adam optimizer configuration.
#[pyclass]
pub struct Adam {
    #[pyo3(get, set)]
    pub max_iters: u32,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub beta1: f64,
    #[pyo3(get, set)]
    pub beta2: f64,
    #[pyo3(get, set)]
    pub epsilon: f64,
    #[pyo3(get, set)]
    pub bounds: (f64, f64),
    /// Early-stopping tolerance on the gradient norm. Defaults to `0.01`, the
    /// same default DE/PSO use for their convergence test.
    #[pyo3(get, set)]
    pub tolerance: f64,
    /// Consecutive sub-`tolerance` iterations required before the run reports
    /// `converged`. Defaults to `3`: one iteration is not trusted, because the
    /// gradient the optimizer sees may be a minibatch gradient that cancelled to
    /// zero (see the minibatch note beside C-5/C-7 in `docs/CONTRACTS.md`).
    /// `patience=1` restores the single-iteration rule.
    #[pyo3(get, set)]
    pub patience: usize,
    /// Optional RNG seed pinned on the optimizer object. Consumed by
    /// `train`/`qml.train` per the precedence rule (contract C-7): the explicit
    /// `seed` kwarg passed to the call wins; this field is the fallback; a fresh
    /// OS-entropy value is used when neither is set. `None` by default.
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl Adam {
    #[new]
    // `learning_rate = 0.05` (not the classic deep-learning 0.001) because these
    // parameters are circuit rotation angles in radians, not neural-network
    // weights, so a larger step is well-scaled here. `beta1`/`beta2`/`epsilon`
    // are the standard values from the literature.
    #[pyo3(signature = (max_iters = 100, learning_rate = 0.05, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, bounds = (-std::f64::consts::PI, std::f64::consts::PI), tolerance = 0.01, patience = 3, seed = None))]
    pub fn new(
        max_iters: u32,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        bounds: (f64, f64),
        tolerance: f64,
        patience: usize,
        seed: Option<u64>,
    ) -> Self {
        Adam {
            max_iters,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            bounds,
            tolerance,
            patience,
            seed,
        }
    }
}
