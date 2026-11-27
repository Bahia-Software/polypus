//! The `method` argument of `train`, `qml.train` and `Model.fit`: its parsing
//! into a pyo3-free [`Method`], the checks made before anything runs (the
//! optimizer's preconditions, Rotosolve's circuit, `final_shots`) and the
//! training observer of a `callback`.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyModule};

use super::adam::Adam;
use super::callback::PyTrainObserver;
use super::cmaes::CMAES;
use super::de::DE;
use super::external::External;
use super::pso::PSO;
use super::qng::QNG;
use super::rotosolve::Rotosolve;
use super::spsa::Spsa;
use crate::evaluation::{CircuitSource, OracleErrorSlot, PyVarianceOracle};
use crate::orchestration::{
    AdamConfig, CmaesConfig, DeConfig, ExternalConfig, Method, PsoConfig, QngConfig,
    RotosolveConfig, SpsaConfig, TrainObserver,
};
use polypus_optimizers::VarianceOracle;

/// `ValueError` when `method` is a `polypus.External` and an `initial_point`
/// is given: the external library chooses its own start. Checked after
/// External's configuration and instead of the point's length and finiteness,
/// so the message says why any point is refused.
fn check_external_initial_point(
    method: &Bound<'_, PyAny>,
    initial_point: Option<&[f64]>,
) -> PyResult<()> {
    if initial_point.is_none() || method.extract::<PyRef<External>>().is_err() {
        return Ok(());
    }
    Err(crate::exceptions::configuration_error(
        "polypus.External takes no initial_point: the external optimizer chooses its own \
         start. Give the starting point to the library, or to the initial_point of the \
         polypus.ask_tell adapter (from_optuna, from_nevergrad); for pycma it is the x0 of \
         cma.CMAEvolutionStrategy",
    ))
}

/// Whether `method` is a `polypus.Rotosolve`, which only some circuits and
/// objectives qualify for.
pub(super) fn is_rotosolve(method: &Bound<'_, PyAny>) -> bool {
    method.extract::<PyRef<Rotosolve>>().is_ok()
}

/// The `ValueError` for a circuit Rotosolve cannot train, with the first
/// reason the check found.
fn rotosolve_circuit_error(reason: &str) -> PyErr {
    crate::exceptions::configuration_error(format!(
        "polypus.Rotosolve cannot train this circuit: {reason}. Rotosolve needs every \
         trainable parameter to be exactly one angle of an rx, ry, rz, rxx, rzz, p, u1, u, \
         u2 or u3 gate (also ryy, rzx, or the θ of an r, in a Qiskit circuit), as ±θ + c, \
         so that the fitness is a sinusoid of each parameter; train this circuit with \
         another optimizer"
    ))
}

/// Check, before anything runs, that `polypus.Rotosolve` can train `circuit`
/// (issue #398): every trainable parameter the angle of exactly one
/// single-Pauli rotation, as `±θ + c`. A `polypus.Circuit` is checked in
/// Rust ([`ParameterizedCircuit::check_single_frequency`]); a Qiskit circuit
/// by `polypus_python.rotosolve`, with `trainable` its trainable parameters
/// (`None`: all of them; `qml.train` passes the ansatz's).
///
/// [`ParameterizedCircuit::check_single_frequency`]: crate::circuit::ParameterizedCircuit::check_single_frequency
pub(super) fn check_rotosolve_circuit(
    circuit: &Bound<'_, PyAny>,
    source: Option<&CircuitSource>,
    trainable: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if let Some(CircuitSource::Native(native)) = source {
        return native
            .check_single_frequency()
            .map_err(|problem| rotosolve_circuit_error(&problem.to_string()));
    }
    let py = circuit.py();
    let problem = PyModule::import(py, "polypus_python.rotosolve")?
        .getattr("single_frequency_problem")?
        .call1((circuit, trainable))?;
    if problem.is_none() {
        return Ok(());
    }
    Err(rotosolve_circuit_error(&problem.extract::<String>()?))
}

/// Check the optimizer's preconditions, `initial_point` included, before
/// anything runs in `train`, `qml.train` and `Model.fit` (contract C-5): every
/// check the optimizer makes before its first RNG draw, in its order, so its
/// configuration errors come before the point's (`dimensions` finite values,
/// within `bounds` for PSO), through [`Method::check_preconditions`]. A failure
/// raises `ValueError` with the optimizer's own message before any QPU is
/// requested; a `method` that is no optimizer raises the `TypeError` of
/// [`method_from_pyclass`]. `External` refuses any point after its
/// configuration, with [`check_external_initial_point`]'s message.
pub(super) fn validate_method(
    method: &Bound<'_, PyAny>,
    dimensions: u32,
    initial_point: Option<&[f64]>,
) -> PyResult<()> {
    // A throwaway `Method`: the checks read only its configuration, never the
    // QNG variance oracle or the External optimizer, so its error slot and run
    // id are placeholders.
    let parsed = method_from_pyclass(method, dimensions, &OracleErrorSlot::new(), "")?;
    // `External`'s point is refused below, with the message that names the
    // adapters, rather than with dispatch's.
    let point = match parsed {
        Method::External(..) => None,
        _ => initial_point,
    };
    parsed
        .check_preconditions(dimensions, point)
        .map_err(|e| crate::exceptions::configuration_error(e.to_string()))?;
    check_external_initial_point(method, initial_point)
}

/// Check the `final_shots` of `train`, `qml.train` and `Model.fit` before
/// anything runs (contract C-5), as `shots` is checked: `None`, or an integer
/// from 1 to 4294967295, the shots a circuit takes, before any backend or CUNQA
/// allocation exists. What is and is not an integer, and which error each
/// raises, is [`positive_index`]'s.
pub(super) fn validate_final_shots(
    final_shots: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<u32>> {
    let Some(value) = final_shots.filter(|value| !value.is_none()) else {
        return Ok(None);
    };
    let index = positive_index(value, "final_shots")?;
    match index.extract::<u32>() {
        Ok(shots) => Ok(Some(shots)),
        Err(_) => Err(too_large(value, "final_shots", u32::MAX)),
    }
}

/// The run's observer for a validated `callback` of `train`, `qml.train` or
/// `Model.fit`: `None` without one, so the run is exactly as before.
pub(crate) fn training_observer(
    callback: Option<Py<PyAny>>,
    errors: &OracleErrorSlot,
    run_id: &str,
) -> Option<Box<dyn TrainObserver>> {
    callback.map(|callback| -> Box<dyn TrainObserver> {
        Box::new(PyTrainObserver::new(callback, errors, run_id))
    })
}

/// `value` as a Python `int` `>= 1`, for the setting `name`: a Python `int` or
/// anything with `__index__` (a NumPy integer) is an integer, and one below 1
/// raises `ValueError` (a `polypus.ConfigurationError`). A `bool`, a `float`
/// (even `4.0`), a `str`, `None` or anything else is not an integer and raises
/// `TypeError`, as an integer argument such as `shots` does. Only the
/// `TypeError` of `operator.index` means "not an integer"; any other
/// exception, raised inside a custom `__index__` or a `KeyboardInterrupt`,
/// propagates unchanged. The caller converts the result to its own type and
/// reports a value too large for it with [`too_large`].
pub(super) fn positive_index<'py>(
    value: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let message = || format!("{name} must be an integer >= 1, got {}", shown_repr(value));
    let not_an_integer = || pyo3::exceptions::PyTypeError::new_err(message());
    if value.is_instance_of::<PyBool>() {
        return Err(not_an_integer());
    }
    let py = value.py();
    let index = match PyModule::import(py, "operator")?.call_method1("index", (value,)) {
        Ok(index) => index,
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) => {
            return Err(not_an_integer());
        }
        Err(err) => return Err(err),
    };
    if index.lt(1)? {
        return Err(crate::exceptions::configuration_error(message()));
    }
    Ok(index)
}

/// The `ValueError` for a setting `name` whose integer `value` is above `max`.
pub(super) fn too_large(
    value: &Bound<'_, PyAny>,
    name: &str,
    max: impl std::fmt::Display,
) -> PyErr {
    crate::exceptions::configuration_error(format!(
        "{name} must be at most {max}, got {}",
        shown_repr(value)
    ))
}

/// `value`'s `repr` for an error message, or a fixed text when it fails.
fn shown_repr(value: &Bound<'_, PyAny>) -> String {
    value
        .repr()
        .map_or_else(|_| "an unprintable value".to_string(), |r| r.to_string())
}

/// Parse a `polypus.DE` / `PSO` / `QNG` / `CMAES` / `Adam` / `SPSA` / `Rotosolve` /
/// `External` object into a pyo3-free [`Method`].
///
/// The QNG `variance_function` is adapted into a [`PyVarianceOracle`] here, and
/// the `External` optimizer into a [`PyAskTell`](external::PyAskTell), which
/// reads each row up to one value past `dimensions` — the only Python touches —
/// so [`dispatch_optimizer`] itself stays Python-free. A non-method object is a
/// `TypeError`, exactly as the previous inline dispatch (shared now by both
/// `train` and `qml_train`).
pub(super) fn method_from_pyclass(
    method: &Bound<'_, PyAny>,
    dimensions: u32,
    errors: &OracleErrorSlot,
    run_id: &str,
) -> PyResult<Method> {
    if let Ok(de) = method.extract::<PyRef<DE>>() {
        return Ok(Method::De(DeConfig {
            generations: de.generations,
            population_size: de.population_size,
            tolerance: de.tolerance,
            patience: de.patience,
            strategy: de.strategy,
            mutation_factor: de.mutation_factor,
            crossover_rate: de.crossover_rate,
        }));
    }
    if let Ok(pso) = method.extract::<PyRef<PSO>>() {
        return Ok(Method::Pso(PsoConfig {
            generations: pso.generations,
            population_size: pso.population_size,
            bounds: pso.bounds,
            inertia_weight: pso.inertia_weight,
            cognitive_weight: pso.cognitive_weight,
            social_weight: pso.social_weight,
            tolerance: pso.tolerance,
        }));
    }
    if let Ok(qng) = method.extract::<PyRef<QNG>>() {
        let variance_oracle: Box<dyn VarianceOracle> = Box::new(PyVarianceOracle {
            variance_function: qng.variance_function.clone_ref(method.py()),
            errors: errors.clone(),
            run_id: run_id.to_string(),
        });
        return Ok(Method::Qng(
            QngConfig {
                max_iters: qng.max_iters,
                learning_rate: qng.learning_rate,
                finite_difference_step: qng.finite_difference_step,
                bounds: qng.bounds,
                tikhonov_reg: qng.tikhonov_reg,
            },
            variance_oracle,
        ));
    }
    if let Ok(cmaes) = method.extract::<PyRef<CMAES>>() {
        return Ok(Method::Cmaes(CmaesConfig {
            generations: cmaes.generations,
            population_size: cmaes.population_size,
            sigma0: cmaes.sigma0,
            tolerance: cmaes.tolerance,
            patience: cmaes.patience,
            restarts: cmaes.restarts,
        }));
    }
    if let Ok(adam) = method.extract::<PyRef<Adam>>() {
        return Ok(Method::Adam(AdamConfig {
            max_iters: adam.max_iters,
            learning_rate: adam.learning_rate,
            beta1: adam.beta1,
            beta2: adam.beta2,
            epsilon: adam.epsilon,
            bounds: adam.bounds,
            tolerance: adam.tolerance,
            patience: adam.patience,
            finite_difference_step: adam.finite_difference_step,
            rule: adam.rule,
        }));
    }
    if let Ok(spsa) = method.extract::<PyRef<Spsa>>() {
        return Ok(Method::Spsa(SpsaConfig {
            max_iters: spsa.max_iters,
            resamplings: spsa.resamplings,
            perturbation: spsa.perturbation,
            target_step: spsa.target_step,
            alpha: spsa.alpha,
            gamma: spsa.gamma,
            stability_constant: spsa.stability_constant,
            eval_interval: spsa.eval_interval,
        }));
    }
    if let Ok(rotosolve) = method.extract::<PyRef<Rotosolve>>() {
        return Ok(Method::Rotosolve(RotosolveConfig {
            sweeps: rotosolve.sweeps,
            reevaluate_every: rotosolve.reevaluate_every,
            tolerance: rotosolve.tolerance,
            patience: rotosolve.patience,
        }));
    }
    if let Ok(external) = method.extract::<PyRef<External>>() {
        let (max_evaluations, optimizer) =
            external.for_run(method.py(), dimensions, errors, run_id);
        return Ok(Method::External(
            ExternalConfig { max_evaluations },
            Box::new(optimizer),
        ));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "method must be an instance of polypus.DE, polypus.PSO, polypus.QNG, polypus.CMAES, \
         polypus.Adam, polypus.SPSA, polypus.Rotosolve, or polypus.External",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::seeds::method_seed;

    /// `polypus.DE(strategy=...)` rejects an unknown name with `ValueError`
    /// when it is passed or set, before any run, and `method_from_pyclass`
    /// hands the parsed strategy to dispatch.
    #[test]
    fn de_strategy_is_validated_and_reaches_the_method() {
        use polypus_optimizers::DeStrategy;
        Python::initialize();
        Python::attach(|py| {
            let err = DE::new(100, 50, 0.01, 20, None, "LSHADE", None, None)
                .err()
                .expect("an unknown strategy is rejected");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(err.to_string().contains("\"lshade\""), "{err}");

            let de = Py::new(
                py,
                DE::new(100, 50, 0.01, 20, None, "shade", None, None).unwrap(),
            )
            .unwrap();
            let bound = de.bind(py);
            assert!(bound.setattr("strategy", "jade").is_err());
            assert_eq!(
                bound
                    .getattr("strategy")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "shade",
                "a rejected name leaves the strategy unchanged"
            );
            bound.setattr("strategy", "lshade").unwrap();
            match method_from_pyclass(bound.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::De(cfg)) => assert_eq!(cfg.strategy, DeStrategy::LShade),
                _ => panic!("a polypus.DE parses into Method::De"),
            }
        });
    }

    /// `polypus.Adam(rule=...)` rejects an unknown name with `ValueError`
    /// when it is passed or set, before any run, naming the valid ones.
    #[test]
    fn adam_rule_is_validated_when_passed_or_set() {
        Python::initialize();
        Python::attach(|py| {
            let new = |rule| {
                Adam::new(
                    100,
                    0.05,
                    0.9,
                    0.999,
                    1e-8,
                    (-1.0, 1.0),
                    0.01,
                    20,
                    0.1,
                    rule,
                    None,
                )
            };
            let err = new("Momentum").err().expect("an unknown rule is rejected");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.to_string().contains("\"adam\" or \"momentum\""),
                "{err}"
            );

            let adam = Py::new(py, new("adam").unwrap()).unwrap();
            let bound = adam.bind(py);
            let err = bound
                .setattr("rule", "nesterov")
                .expect_err("an unknown rule is rejected when set");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.to_string().contains("\"adam\" or \"momentum\""),
                "{err}"
            );
            assert_eq!(
                bound.getattr("rule").unwrap().extract::<String>().unwrap(),
                "adam",
                "a rejected name leaves the rule unchanged"
            );
            bound.setattr("rule", "momentum").unwrap();
            assert_eq!(
                bound.getattr("rule").unwrap().extract::<String>().unwrap(),
                "momentum"
            );
        });
    }

    /// `polypus.CMAES` parses into `Method::Cmaes` with every field, its pinned
    /// seed is the C-7 fallback, and a non-method object's `TypeError` names it.
    #[test]
    fn cmaes_reaches_the_method_and_its_seed_is_read() {
        Python::initialize();
        Python::attach(|py| {
            let cmaes = Py::new(py, CMAES::new(40, Some(12), 0.75, 0.5, 7, 3, Some(99)))
                .expect("the pyclass instantiates");
            let bound = cmaes.bind(py);
            assert_eq!(method_seed(bound.as_any()), Some(99));
            match method_from_pyclass(bound.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::Cmaes(cfg)) => {
                    assert_eq!(cfg.generations, 40);
                    assert_eq!(cfg.population_size, Some(12));
                    assert_eq!(cfg.sigma0, 0.75);
                    assert_eq!(cfg.tolerance, 0.5);
                    assert_eq!(cfg.patience, 7);
                    assert_eq!(cfg.restarts, 3);
                }
                _ => panic!("a polypus.CMAES parses into Method::Cmaes"),
            }
            bound.setattr("population_size", py.None()).unwrap();
            match method_from_pyclass(bound.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::Cmaes(cfg)) => assert_eq!(cfg.population_size, None),
                _ => panic!("a polypus.CMAES parses into Method::Cmaes"),
            }

            let not_a_method = 3_i32.into_pyobject(py).unwrap();
            assert_eq!(method_seed(not_a_method.as_any()), None);
            let err = method_from_pyclass(not_a_method.as_any(), 1, &OracleErrorSlot::new(), "run")
                .err()
                .expect("an int is not a method");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("polypus.CMAES"), "{err}");
        });
    }

    /// `polypus.Adam` parses into `Method::Adam` with every field, its pinned
    /// seed is the C-7 fallback, and a non-method object's `TypeError` names it.
    #[test]
    fn adam_reaches_the_method_and_its_seed_is_read() {
        use polypus_optimizers::AdamRule;
        Python::initialize();
        Python::attach(|py| {
            let adam = Py::new(
                py,
                Adam::new(
                    40,
                    0.2,
                    0.8,
                    0.99,
                    1e-6,
                    (-1.0, 2.0),
                    0.5,
                    7,
                    0.05,
                    "momentum",
                    Some(99),
                )
                .expect("a valid rule"),
            )
            .expect("the pyclass instantiates");
            let bound = adam.bind(py);
            assert_eq!(method_seed(bound.as_any()), Some(99));
            match method_from_pyclass(bound.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::Adam(cfg)) => {
                    assert_eq!(cfg.max_iters, 40);
                    assert_eq!(cfg.learning_rate, 0.2);
                    assert_eq!(cfg.beta1, 0.8);
                    assert_eq!(cfg.beta2, 0.99);
                    assert_eq!(cfg.epsilon, 1e-6);
                    assert_eq!(cfg.bounds, (-1.0, 2.0));
                    assert_eq!(cfg.tolerance, 0.5);
                    assert_eq!(cfg.patience, 7);
                    assert_eq!(cfg.finite_difference_step, 0.05);
                    assert_eq!(cfg.rule, AdamRule::Momentum);
                }
                _ => panic!("a polypus.Adam parses into Method::Adam"),
            }

            let not_a_method = 3_i32.into_pyobject(py).unwrap();
            let err = method_from_pyclass(not_a_method.as_any(), 1, &OracleErrorSlot::new(), "run")
                .err()
                .expect("an int is not a method");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("polypus.Adam"), "{err}");
        });
    }

    /// `polypus.SPSA` parses into `Method::Spsa` with every field, its pinned
    /// seed is the C-7 fallback, and a non-method object's `TypeError` names it.
    #[test]
    fn spsa_reaches_the_method_and_its_seed_is_read() {
        Python::initialize();
        Python::attach(|py| {
            let spsa = Py::new(py, Spsa::new(40, 8, 0.2, 1.2, 1.0, 0.25, 5, Some(99), 4.0))
                .expect("the pyclass instantiates");
            let bound = spsa.bind(py);
            assert_eq!(method_seed(bound.as_any()), Some(99));
            match method_from_pyclass(bound.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::Spsa(cfg)) => {
                    assert_eq!(cfg.max_iters, 40);
                    assert_eq!(cfg.resamplings, 8);
                    assert_eq!(cfg.perturbation, 0.2);
                    assert_eq!(cfg.target_step, 1.2);
                    assert_eq!(cfg.alpha, 1.0);
                    assert_eq!(cfg.gamma, 0.25);
                    assert_eq!(cfg.stability_constant, 4.0);
                    assert_eq!(cfg.eval_interval, 5);
                }
                _ => panic!("a polypus.SPSA parses into Method::Spsa"),
            }

            let not_a_method = 3_i32.into_pyobject(py).unwrap();
            let err = method_from_pyclass(not_a_method.as_any(), 1, &OracleErrorSlot::new(), "run")
                .err()
                .expect("an int is not a method");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("polypus.SPSA"), "{err}");
        });
    }

    /// `polypus.Rotosolve` parses into `Method::Rotosolve` with every field,
    /// its pinned seed is the C-7 fallback, and a non-method object's
    /// `TypeError` names it.
    #[test]
    fn rotosolve_reaches_the_method_and_its_seed_is_read() {
        Python::initialize();
        Python::attach(|py| {
            let rotosolve = Py::new(py, Rotosolve::new(7, 3, 0.25, 4, Some(99)))
                .expect("the pyclass instantiates");
            let bound = rotosolve.bind(py);
            assert!(is_rotosolve(bound.as_any()));
            assert_eq!(method_seed(bound.as_any()), Some(99));
            match method_from_pyclass(bound.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::Rotosolve(cfg)) => {
                    assert_eq!(cfg.sweeps, 7);
                    assert_eq!(cfg.reevaluate_every, 3);
                    assert_eq!(cfg.tolerance, 0.25);
                    assert_eq!(cfg.patience, 4);
                }
                _ => panic!("a polypus.Rotosolve parses into Method::Rotosolve"),
            }

            let not_a_method = 3_i32.into_pyobject(py).unwrap();
            assert!(!is_rotosolve(not_a_method.as_any()));
            let err = method_from_pyclass(not_a_method.as_any(), 1, &OracleErrorSlot::new(), "run")
                .err()
                .expect("an int is not a method");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("polypus.Rotosolve"), "{err}");
        });
    }

    /// `polypus.External` parses into `Method::External` with its budget, has
    /// no seed of its own (C-7), refuses any `initial_point` before the point
    /// is checked, and is named by a non-method object's `TypeError`.
    #[test]
    fn external_reaches_the_method_without_a_seed() {
        Python::initialize();
        Python::attach(|py| {
            let optimizer = PyModule::from_code(
                py,
                c"class Done:\n    def ask(self):\n        return None\n    def tell(self, c, v):\n        pass\n",
                c"external_done.py",
                c"external_done",
            )
            .and_then(|module| module.getattr("Done")?.call0())
            .expect("the optimizer instantiates");
            let external = Bound::new(
                py,
                External::new(optimizer, 12_i32.into_pyobject(py).unwrap().into_any())
                    .expect("a valid External"),
            )
            .expect("the pyclass instantiates");
            assert_eq!(method_seed(external.as_any()), None);
            match method_from_pyclass(external.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::External(cfg, _)) => assert_eq!(cfg.max_evaluations, 12),
                _ => panic!("a polypus.External parses into Method::External"),
            }

            assert!(check_external_initial_point(external.as_any(), None).is_ok());
            // Even a point of the wrong length gets External's reason.
            let err = check_external_initial_point(external.as_any(), Some(&[0.5]))
                .expect_err("External refuses a point");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.value(py)
                    .to_string()
                    .starts_with("polypus.External takes no initial_point"),
                "{err}"
            );
            let rotosolve = Py::new(py, Rotosolve::new(7, 3, 0.25, 4, None)).unwrap();
            assert!(
                check_external_initial_point(rotosolve.bind(py).as_any(), Some(&[0.5])).is_ok()
            );

            let not_a_method = 3_i32.into_pyobject(py).unwrap();
            let err = method_from_pyclass(not_a_method.as_any(), 1, &OracleErrorSlot::new(), "run")
                .err()
                .expect("an int is not a method");
            assert!(err.to_string().contains("or polypus.External"), "{err}");
        });
    }

    /// A native circuit is checked in Rust: one that qualifies passes, and the
    /// first problem of one that does not is the `ValueError`'s reason.
    #[test]
    fn a_native_circuit_is_checked_for_rotosolve_in_rust() {
        use crate::circuit::{Param, ParameterizedCircuit};
        Python::initialize();
        Python::attach(|py| {
            let unused = 0_i32.into_pyobject(py).unwrap();
            let good = CircuitSource::Native(
                ParameterizedCircuit::new(2)
                    .ry(0, Param(0))
                    .cx(0, 1)
                    .rz(1, Param(1)),
            );
            assert!(check_rotosolve_circuit(unused.as_any(), Some(&good), None).is_ok());
            let bad = CircuitSource::Native(ParameterizedCircuit::new(2).ry(0, Param(0)).crx(
                0,
                1,
                Param(1),
            ));
            let err = check_rotosolve_circuit(unused.as_any(), Some(&bad), None)
                .expect_err("a controlled rotation does not qualify");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            let message = err.value(py).to_string();
            assert!(
                message.starts_with(
                    "polypus.Rotosolve cannot train this circuit: parameter 1 ('theta_1') is \
                     an angle of 'crx' (instruction 1), which is not a rotation generated by a \
                     single Pauli operator. Rotosolve needs"
                ),
                "{message}"
            );
        });
    }
}
