use crate::algorithms::{AlgorithmTrait, AlgorithmArgs};
use crate::infrastructure::{Infrastructure, QuantumRunner, LocalRunner, CunqaRunner};

/// AlgorithmSingleRun struct to run a single quantum circuit using the specified infrastructure.
pub struct AlgorithmSingleRun;

impl AlgorithmTrait for AlgorithmSingleRun {
	type Args = AlgorithmArgs;
	type AlgorithmReturnType = pyo3::PyObject;

    fn run(&self, args: AlgorithmArgs) -> Self::AlgorithmReturnType {
        let infra = Infrastructure::from_str(&args.infrastructure);
        let runner: Box<dyn QuantumRunner> = match infra {
            Infrastructure::Local => Box::new(LocalRunner),
            Infrastructure::Cunqa => Box::new(CunqaRunner::new(args.n_qpus, args.nodes, &args.id)),
        };
        let running_result = runner.run(&args);

        // Close
		runner.close();

        running_result
    }

	fn name(&self) -> String {
		String::from("Single Run Algorithm")
	}

	fn description(&self) -> String {
		String::from("Base algorithm to run a quantum circuit")
	}
}
