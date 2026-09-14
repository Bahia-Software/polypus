//! Training [`Flow`]s: they build an oracle from the run's [`Resources`] and drive
//! `dispatch_optimizer`. These live here (not in `polypus-scheduler`) because they
//! *construct oracles*, which is this crate's job; `RunCircuitFlow`, which builds
//! no oracle, stays in the scheduler.

use crate::{CircuitSource, CostObservable, OracleErrorSlot, QmlOracle, VqcOracle};
use polypus_infrastructure::CancelToken;
use polypus_optimizers::OptimizationOutcome;
use polypus_scheduler::{dispatch_optimizer, Flow, Method, OracleError, Resources};
use pyo3::prelude::*;
use std::sync::Arc;

/// Standard VQC training as a [`Flow`]: build a [`VqcOracle`] from the run's
/// [`Resources`] plus this flow's template/observable/method, then optimise.
pub struct TrainVqcFlow {
    /// Parameterised circuit template (ansatz parameters unbound).
    pub circuit: CircuitSource,
    /// Reduces measurement counts to the fitness scalar.
    pub observable: Arc<dyn CostObservable>,
    /// Chosen optimizer + its config (built at the edge from the `polypus.DE/PSO/QNG`
    /// object; `Qng` carries its variance oracle).
    pub method: Method,
    /// Free-parameter count validated at the edge.
    pub dimensions: u32,
    /// Optimizer RNG seed (contract C-7).
    pub seed: u64,
    /// Shared failure slot: the oracle records into it, `dispatch_optimizer` takes
    /// it, and the edge downcasts + re-raises.
    pub errors: OracleErrorSlot,
}

impl Flow for TrainVqcFlow {
    type Output = OptimizationOutcome;
    type Error = OracleError;

    fn run(
        self,
        resources: &Resources,
        cancel: &CancelToken,
    ) -> Result<OptimizationOutcome, OracleError> {
        let oracle = VqcOracle {
            circuit: self.circuit,
            config: Arc::clone(&resources.config),
            backend: Arc::clone(&resources.backend),
            planner: Arc::clone(&resources.planner),
            observable: self.observable,
            cancel: cancel.clone(),
            errors: self.errors.clone(),
        };
        dispatch_optimizer(
            self.method,
            Box::new(oracle),
            self.dimensions,
            &self.errors,
            self.seed,
        )
    }
}

/// QML training as a [`Flow`]: build a [`QmlOracle`] from the run's [`Resources`]
/// plus this flow's training circuits/observable/method, then optimise.
pub struct TrainQmlFlow {
    /// Pre-bound training circuits (feature-map parameters fixed).
    pub training_circuits: Vec<Py<PyAny>>,
    /// Reduces each candidate's measurement counts to the fitness scalar.
    pub observable: Arc<dyn CostObservable>,
    /// Chosen optimizer + its config (see [`TrainVqcFlow::method`]).
    pub method: Method,
    /// Free-parameter count validated at the edge.
    pub dimensions: u32,
    /// Optimizer RNG seed (contract C-7).
    pub seed: u64,
    /// Shared failure slot (see [`TrainVqcFlow::errors`]).
    pub errors: OracleErrorSlot,
}

impl Flow for TrainQmlFlow {
    type Output = OptimizationOutcome;
    type Error = OracleError;

    fn run(
        self,
        resources: &Resources,
        cancel: &CancelToken,
    ) -> Result<OptimizationOutcome, OracleError> {
        let oracle = QmlOracle {
            training_circuits: self.training_circuits,
            config: Arc::clone(&resources.config),
            backend: Arc::clone(&resources.backend),
            planner: Arc::clone(&resources.planner),
            observable: self.observable,
            cancel: cancel.clone(),
            errors: self.errors.clone(),
        };
        dispatch_optimizer(
            self.method,
            Box::new(oracle),
            self.dimensions,
            &self.errors,
            self.seed,
        )
    }
}
