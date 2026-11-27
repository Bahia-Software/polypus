use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use pyo3::PyResult;

pub mod adam;
mod backend_config;
pub mod calibration;
mod call_trace;
pub mod callback;
pub mod circuit;
pub mod cmaes;
mod compatibility;
pub mod de;
#[cfg(test)]
mod entry_tests;
pub mod expr;
pub mod external;
mod extract;
mod interrupt;
pub mod logging;
pub mod observable;
mod optimizer;
mod options;
mod prepare;
pub mod pso;
pub mod qml;
mod qml_entry;
pub mod qng;
mod results;
pub mod rotosolve;
mod run;
mod seeds;
pub mod spsa;
mod targets;
#[cfg(test)]
mod test_support;
mod training;

use adam::Adam;
use calibration::calibrate_parallel_threshold;
use callback::TrainProgress;
use circuit::{qft, shor_order_finding, statevector, Circuit, Param};
use cmaes::CMAES;
use compatibility::backend_compatibility;
use de::DE;
use external::External;
use logging::init_logger;
use observable::{CachedCost, Ising, Qubo, SampleCost};
use pso::PSO;
use qml_entry::{qml_predict, qml_train};
use qng::QNG;
use results::{RunResult, TrainResult};
use rotosolve::Rotosolve;
use run::run_quantum_circuit;
use spsa::Spsa;
use training::train;

/// Number of backend resource-cleanup (`close`/`Drop`) failures recorded this
/// process. A `Drop` must never panic, so a failed teardown (e.g. releasing a
/// CUNQA SLURM allocation) is logged and counted rather than raised; this
/// exposes the consultable count to Python for diagnostics/monitoring.
#[pyfunction]
fn backend_cleanup_failures() -> u64 {
    crate::infrastructure::cleanup_failure_count()
}

/// The PEP 440 form of a workspace version, the one maturin gives the
/// distribution: `X.Y.Z` stays, the development version `X.Y.Z-dev` of `main`
/// becomes `X.Y.Z.dev0` (C-6).
fn python_version(cargo_version: &str) -> String {
    match cargo_version.strip_suffix("-dev") {
        Some(release) => format!("{release}.dev0"),
        None => cargo_version.to_owned(),
    }
}

/// Polypus — distributed quantum computing (Rust core, Python bindings).
///
/// Same project, three names: install `polypus-quantum` from PyPI
/// (`pip install polypus-quantum`), but import it as `polypus`; the Rust crate
/// is `polypus` (`cargo add polypus`). The PyPI name `polypus` is taken by an
/// unrelated project, so the distribution is published as `polypus-quantum`
/// while the import name and the Rust crate both stay `polypus`.
#[pymodule]
pub fn polypus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Baked in at compile time, so it names the binary actually loaded (C-6).
    m.add("__version__", python_version(env!("CARGO_PKG_VERSION")))?;
    crate::exceptions::register(m)?;
    m.add_class::<DE>()?;
    m.add_class::<PSO>()?;
    m.add_class::<QNG>()?;
    m.add_class::<CMAES>()?;
    m.add_class::<Adam>()?;
    m.add_class::<Spsa>()?;
    m.add_class::<Rotosolve>()?;
    m.add_class::<External>()?;
    m.add_class::<Circuit>()?;
    m.add_class::<Param>()?;
    expr::register(m)?;
    m.add_class::<RunResult>()?;
    m.add_class::<TrainResult>()?;
    m.add_class::<TrainProgress>()?;
    m.add_class::<Qubo>()?;
    m.add_class::<Ising>()?;
    m.add_class::<CachedCost>()?;
    m.add_class::<SampleCost>()?;
    targets::register(m)?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(run_quantum_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(statevector, m)?)?;
    m.add_function(wrap_pyfunction!(init_logger, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_parallel_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(backend_cleanup_failures, m)?)?;
    m.add_function(wrap_pyfunction!(backend_compatibility, m)?)?;

    // qml submodule — exposes polypus.qml.train()
    let py = m.py();
    let qml = PyModule::new(py, "polypus.qml")?;
    qml.add_function(wrap_pyfunction!(qml_train, &qml)?)?;
    qml.add_function(wrap_pyfunction!(qml_predict, &qml)?)?;
    qml::register(&qml)?;
    // Attach under the short key: `add_submodule` would use the dotted `__name__`
    // verbatim as the attribute name, breaking `polypus.qml.train` access.
    m.add("qml", &qml)?;
    // Register in sys.modules so `import polypus.qml` also works
    let sys = PyModule::import(py, "sys")?;
    sys.getattr("modules")?.set_item("polypus.qml", &qml)?;

    // circuits.templates subpackage — exposes polypus.circuits.templates.qft()
    // and shor_order_finding()
    let circuits = PyModule::new(py, "polypus.circuits")?;
    let templates = PyModule::new(py, "polypus.circuits.templates")?;
    templates.add_function(wrap_pyfunction!(qft, &templates)?)?;
    templates.add_function(wrap_pyfunction!(shor_order_finding, &templates)?)?;
    // Attach under short keys (see the qml note above) and mirror both levels
    // into sys.modules so `import polypus.circuits.templates` also resolves.
    circuits.add("templates", &templates)?;
    m.add("circuits", &circuits)?;
    let modules = sys.getattr("modules")?;
    modules.set_item("polypus.circuits", &circuits)?;
    modules.set_item("polypus.circuits.templates", &templates)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_version_keeps_a_release_version() {
        assert_eq!(python_version("0.8.0"), "0.8.0");
    }

    #[test]
    fn python_version_maps_a_dev_version_as_maturin_does() {
        assert_eq!(python_version("0.8.1-dev"), "0.8.1.dev0");
    }
}
