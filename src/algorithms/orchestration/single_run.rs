use crate::algorithms::{AlgorithmTrait, AlgorithmArgs};
use crate::infrastructure::{Infrastructure, QuantumRunner, LocalRunner, CunqaRunner};

pub struct AlgorithmSingleRun;

impl AlgorithmTrait for AlgorithmSingleRun {
	type Args<'py> = AlgorithmArgs<'py>;
	type AlgorithmReturnType = pyo3::PyObject;

    fn run<'py>(&self, args: AlgorithmArgs<'py>) -> Self::AlgorithmReturnType {
        let infra = Infrastructure::from_str(&args.infrastructure);
        let runner: Box<dyn QuantumRunner> = match infra {
            Infrastructure::Local => Box::new(LocalRunner),
            Infrastructure::Cunqa => Box::new(CunqaRunner),
        };
        runner.run(&args)
    }

	fn name(&self) -> String {
		String::from("Single Run Algorithm")
	}

	fn description(&self) -> String {
		String::from("Base algorithm to run a quantum circuit")
	}
}
